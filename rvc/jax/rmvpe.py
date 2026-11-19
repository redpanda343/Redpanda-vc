"""Native JAX/TPU batched RMVPE F0 extractor matching shared.predictors.RMVPE."""

from __future__ import annotations

import math
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from rvc.jax import PrecisionConfig
from rvc.jax.mel import hann_window_np, mel_filterbank_np
from rvc.jax.ops import (
    Array,
    avg_pool2d,
    conv2d,
    conv2d_weight_torch_to_jax,
    conv_transpose2d,
    linear,
    linear_weight_torch_to_jax,
    stft_complex,
)

N_MELS = 128
N_CLASS = 360
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
MEL_FMIN = 30.0
MEL_FMAX = 8000.0
RMVPE_CONST = 1997.3794084376191


def _fold_bn2d_conv(
    w_pt: np.ndarray,
    b_pt: np.ndarray | None,
    gamma: np.ndarray,
    beta: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    eps: float = 1e-5,
    *,
    is_transpose: bool = False,
) -> dict[str, np.ndarray]:
    """Fold eval BatchNorm2d strictly in float32 into Conv2d / ConvTranspose2d weights & bias."""
    scale = (np.asarray(gamma, dtype=np.float32) / np.sqrt(np.asarray(var, dtype=np.float32) + eps)).astype(np.float32)
    b_zero = np.zeros_like(scale) if b_pt is None else np.asarray(b_pt, dtype=np.float32)
    shift = (np.asarray(beta, dtype=np.float32) + (b_zero - np.asarray(mean, dtype=np.float32)) * scale).astype(np.float32)

    w_jax = conv2d_weight_torch_to_jax(w_pt)  # (Kh, Kw, C_1, C_0)
    if is_transpose:
        # PyTorch ConvTranspose2d weight is (C_in, C_out, Kh, Kw) -> w_jax is (Kh, Kw, C_out, C_in)
        w_jax = w_jax * scale[None, None, :, None]
    else:
        w_jax = w_jax * scale[None, None, None, :]
    return {"weight": np.ascontiguousarray(w_jax, dtype=np.float32), "bias": shift}


def _extract_conv_block_res(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "conv1": _fold_bn2d_conv(
            state[f"{prefix}.conv.0.weight"],
            None,
            state[f"{prefix}.conv.1.weight"],
            state[f"{prefix}.conv.1.bias"],
            state[f"{prefix}.conv.1.running_mean"],
            state[f"{prefix}.conv.1.running_var"],
        ),
        "conv2": _fold_bn2d_conv(
            state[f"{prefix}.conv.3.weight"],
            None,
            state[f"{prefix}.conv.4.weight"],
            state[f"{prefix}.conv.4.bias"],
            state[f"{prefix}.conv.4.running_mean"],
            state[f"{prefix}.conv.4.running_var"],
        ),
    }
    if f"{prefix}.shortcut.weight" in state:
        out["shortcut"] = {
            "weight": conv2d_weight_torch_to_jax(state[f"{prefix}.shortcut.weight"]),
            "bias": np.asarray(state[f"{prefix}.shortcut.bias"], dtype=np.float32).copy(),
        }
    return out


def rmvpe_params_from_torch_file(model_path: str) -> dict[str, Any]:
    """Load RMVPE (rmvpe_rvc.pt or rmvpe.pt) PyTorch weights into folded-BN JAX PyTree."""
    import torch

    ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v) for k, v in raw.items()}

    # Input BatchNorm2d(1) in unet.encoder.bn
    bn_var = np.asarray(state["unet.encoder.bn.running_var"], dtype=np.float32)
    bn_scale = np.asarray(state["unet.encoder.bn.weight"], dtype=np.float32) / np.sqrt(bn_var + 1e-5)
    bn_bias = np.asarray(state["unet.encoder.bn.bias"], dtype=np.float32) - np.asarray(
        state["unet.encoder.bn.running_mean"], dtype=np.float32
    ) * bn_scale

    enc_layers = [
        [_extract_conv_block_res(state, f"unet.encoder.layers.{i}.conv.{j}") for j in range(4)]
        for i in range(5)
    ]
    inter_layers = [
        [_extract_conv_block_res(state, f"unet.intermediate.layers.{i}.conv.{j}") for j in range(4)]
        for i in range(4)
    ]
    dec_layers = []
    for i in range(5):
        p = f"unet.decoder.layers.{i}"
        dec_layers.append(
            {
                "conv1": _fold_bn2d_conv(
                    state[f"{p}.conv1.0.weight"],
                    None,
                    state[f"{p}.conv1.1.weight"],
                    state[f"{p}.conv1.1.bias"],
                    state[f"{p}.conv1.1.running_mean"],
                    state[f"{p}.conv1.1.running_var"],
                    is_transpose=True,
                ),
                "conv2": [_extract_conv_block_res(state, f"{p}.conv2.{j}") for j in range(4)],
            }
        )

    cnn = {
        "weight": conv2d_weight_torch_to_jax(state["cnn.weight"]),
        "bias": np.asarray(state["cnn.bias"], dtype=np.float32).copy(),
    }

    gru = {
        "w_ih_fwd": linear_weight_torch_to_jax(state["fc.0.gru.weight_ih_l0"]),
        "b_ih_fwd": np.asarray(state["fc.0.gru.bias_ih_l0"], dtype=np.float32).copy(),
        "w_hh_fwd": linear_weight_torch_to_jax(state["fc.0.gru.weight_hh_l0"]),
        "b_hh_fwd": np.asarray(state["fc.0.gru.bias_hh_l0"], dtype=np.float32).copy(),
        "w_ih_rev": linear_weight_torch_to_jax(state["fc.0.gru.weight_ih_l0_reverse"]),
        "b_ih_rev": np.asarray(state["fc.0.gru.bias_ih_l0_reverse"], dtype=np.float32).copy(),
        "w_hh_rev": linear_weight_torch_to_jax(state["fc.0.gru.weight_hh_l0_reverse"]),
        "b_hh_rev": np.asarray(state["fc.0.gru.bias_hh_l0_reverse"], dtype=np.float32).copy(),
    }
    fc_linear = {
        "weight": linear_weight_torch_to_jax(state["fc.1.weight"]),
        "bias": np.asarray(state["fc.1.bias"], dtype=np.float32).copy(),
    }
    return {
        "bn_in": {"scale": bn_scale.astype(np.float32), "bias": bn_bias.astype(np.float32)},
        "encoder": enc_layers,
        "intermediate": inter_layers,
        "decoder": dec_layers,
        "cnn": cnn,
        "gru": gru,
        "fc_linear": fc_linear,
    }


def _conv_block_res_forward(
    x: Array,
    params: Mapping[str, Any],
    prec: PrecisionConfig,
    *,
    mask_bt11: Array | None = None,
) -> Array:
    h = conv2d(x, params["conv1"], padding=(1, 1), dtype=prec.compute_dtype, precision=prec.compute_precision)
    h = jax.nn.relu(h)
    if mask_bt11 is not None:
        h = h * mask_bt11
    h = conv2d(h, params["conv2"], padding=(1, 1), dtype=prec.compute_dtype, precision=prec.compute_precision)
    h = jax.nn.relu(h)
    if mask_bt11 is not None:
        h = h * mask_bt11
    if "shortcut" in params:
        res = conv2d(x, params["shortcut"], padding=(0, 0), dtype=prec.compute_dtype, precision=prec.compute_precision)
        if mask_bt11 is not None:
            res = res * mask_bt11
    else:
        res = jnp.asarray(x, jnp.float32)
    return h + res


def _masked_gru_scan(
    x: Array,
    w_ih: Array,
    b_ih: Array,
    w_hh: Array,
    b_hh: Array,
    valid_mask_bt: Array,
    *,
    reverse: bool,
    prec: PrecisionConfig,
) -> Array:
    """Strictly FP32 GRU recurrent scan with vectorized input projection before scan."""
    b, t, _ = x.shape
    h_dim = b_hh.shape[0] // 3
    # One batched MXU matmul across all timesteps (B, T, 3*H) -> compute once cleanly!
    x_proj = (
        jnp.matmul(
            jnp.asarray(x, prec.compute_dtype),
            jnp.asarray(w_ih, prec.compute_dtype),
            precision=prec.compute_precision,
            preferred_element_type=jnp.float32,
        )
        + jnp.asarray(b_ih, jnp.float32)
    )
    w_hh_f32 = jnp.asarray(w_hh, jnp.float32)
    b_hh_f32 = jnp.asarray(b_hh, jnp.float32)

    x_tbc = jnp.transpose(x_proj, (1, 0, 2))  # (T, B, 3*H)
    m_tb = valid_mask_bt.T[..., None]  # (T, B, 1)
    if reverse:
        x_tbc = x_tbc[::-1]
        m_tb = m_tb[::-1]

    def _step(h_prev: Array, inp: tuple[Array, Array]) -> tuple[Array, Array]:
        xp, m = inp
        hh = jnp.matmul(h_prev, w_hh_f32, precision=prec.fp32_precision) + b_hh_f32
        r = jax.nn.sigmoid(xp[:, :h_dim] + hh[:, :h_dim])
        z = jax.nn.sigmoid(xp[:, h_dim : 2 * h_dim] + hh[:, h_dim : 2 * h_dim])
        n = jnp.tanh(xp[:, 2 * h_dim :] + r * hh[:, 2 * h_dim :])
        h_next = (1.0 - z) * n + z * h_prev
        # Zero hidden state on invalid padded prefix during reverse scan so valid end starts at 0.
        h_next = jnp.where(m, h_next, 0.0)
        return h_next, h_next

    scan_unroll = 4 if int(x_tbc.shape[0]) >= 4 else 1
    _, out_tbc = lax.scan(_step, jnp.zeros((b, h_dim), dtype=jnp.float32), (x_tbc, m_tb), unroll=scan_unroll)
    if reverse:
        out_tbc = out_tbc[::-1]
    return jnp.transpose(out_tbc, (1, 0, 2))


def _reflect_pad_batch_right_512(audio_bt: Array, lengths_b: Array) -> Array:
    """Apply exact clip-specific 512-sample left and right reflect padding at `lengths_b` for center=True STFT."""
    b, l_max = audio_bt.shape
    pad = N_FFT // 2  # 512

    def _pad_one(wav: Array, length: Array) -> Array:
        # Left reflect: wav[pad:0:-1] followed by wav[:] and space for 512 right samples.
        left = wav[pad:0:-1]
        ext = jnp.concatenate([left, wav, jnp.zeros((pad,), dtype=jnp.float32)], axis=0)
        # Right reflect slice at `pad + length` (from wav[length-2 : length-514 : -1]).
        start = jnp.maximum(pad + length - 1 - pad, 0)
        seg = lax.dynamic_slice(ext, (start,), (pad,))[::-1]
        return lax.dynamic_update_slice(ext, seg, (pad + length,))

    return jax.vmap(_pad_one, in_axes=(0, 0))(jnp.asarray(audio_bt, jnp.float32), jnp.asarray(lengths_b, jnp.int32))


def rmvpe_mel_from_audio(audio_bt: Array, audio_lengths: Array) -> Array:
    """Strictly FP32 batched RMVPE MelSpectrogram matching shared.predictors.RMVPE.MelSpectrogram."""
    padded = _reflect_pad_batch_right_512(audio_bt, audio_lengths)
    win = hann_window_np(N_FFT)
    real, imag = stft_complex(padded, N_FFT, HOP_LENGTH, N_FFT, win, pad_left=0, pad_right=0)
    mag = jnp.sqrt(jnp.maximum(real * real + imag * imag, 1e-24))  # (B, F, TT)
    basis = jnp.asarray(mel_filterbank_np(SAMPLE_RATE, N_FFT, N_MELS, fmin=MEL_FMIN, fmax=MEL_FMAX, htk=True), jnp.float32)
    mel = jnp.matmul(basis[None, :, :], mag, precision=lax.Precision.HIGHEST)  # (B, 128, TT)
    log_mel = jnp.log(jnp.maximum(mel, 1e-5))
    return jnp.transpose(log_mel, (0, 2, 1))  # (B, TT, 128)


def _reflect_pad_mel_to_len(mel_btf: Array, n_frames_b: Array, pad32_b: Array) -> Array:
    """Reflect tail of each clip from `n_frames` to `pad32` inside padded `(B, T_bucket, 128)` mel."""

    def _fix_one(m: Array, nf: Array, p32: Array) -> Array:
        t_len = m.shape[0]
        i = jnp.arange(t_len, dtype=jnp.int32)
        src = jnp.clip(2 * (nf - 1) - i, 0, jnp.maximum(nf - 1, 0))
        reflected = m[src]
        valid_out = (i < p32)[:, None]
        is_tail = (i >= nf)[:, None]
        return jnp.where(valid_out, jnp.where(is_tail, reflected, m), 0.0)

    return jax.vmap(_fix_one, in_axes=(0, 0, 0))(mel_btf, n_frames_b, pad32_b)


def rmvpe_salience_forward(
    audio_bt: Array,
    audio_lengths: Array,
    params: Mapping[str, Any],
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Compute batched RMVPE salience probabilities (B, T_mel, 360) strictly in FP32 out."""
    mel_raw = rmvpe_mel_from_audio(audio_bt, audio_lengths)  # (B, T_mel_bucket, 128), T_mel_bucket % 32 == 0
    t_mel = int(mel_raw.shape[1])
    n_frames = jnp.asarray(audio_lengths, jnp.int32) // HOP_LENGTH + 1
    pad32 = jnp.minimum(32 * ((n_frames - 1) // 32 + 1), t_mel)
    mel = _reflect_pad_mel_to_len(mel_raw, n_frames, pad32)

    # Exact per-level valid masks (B, T_lvl, 1, 1) so padded bucket tail [pad32:] stays strictly
    # zero across every conv/bn bias and never bleeds into valid frames (exact match to unbatched PyTorch).
    lvl_masks = [
        (jnp.arange(t_mel >> lvl, dtype=jnp.int32)[None, :] < (pad32 >> lvl)[:, None]).astype(jnp.float32)[:, :, None, None]
        for lvl in range(6)
    ]

    # Apply input BatchNorm2d(1) strictly in FP32 and zero padded tail
    x = mel[..., None]  # (B, T, 128, 1) NHWC
    x = (x * jnp.asarray(params["bn_in"]["scale"], jnp.float32) + jnp.asarray(params["bn_in"]["bias"], jnp.float32)) * lvl_masks[0]

    # DeepUnet Encoder (5 levels)
    skips: list[Array] = []
    for lvl_idx, lvl_blocks in enumerate(params["encoder"]):
        m_lvl = lvl_masks[lvl_idx]
        for blk in lvl_blocks:
            x = _conv_block_res_forward(x, blk, prec, mask_bt11=m_lvl)
        skips.append(x)
        x = avg_pool2d(x, (2, 2), (2, 2))

    # Intermediate (4 levels)
    m_inter = lvl_masks[5]
    for lvl_blocks in params["intermediate"]:
        for blk in lvl_blocks:
            x = _conv_block_res_forward(x, blk, prec, mask_bt11=m_inter)

    # Decoder (5 levels)
    for dec_idx, (dec_lvl, skip) in enumerate(zip(params["decoder"], reversed(skips))):
        m_lvl = lvl_masks[4 - dec_idx]
        x = conv_transpose2d(
            x,
            dec_lvl["conv1"],
            stride=(2, 2),
            padding=(1, 1),
            output_padding=(1, 1),
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        x = jax.nn.relu(x) * m_lvl
        x = jnp.concatenate([x, skip], axis=-1)
        for blk in dec_lvl["conv2"]:
            x = _conv_block_res_forward(x, blk, prec, mask_bt11=m_lvl)

    # cnn: Conv2d(16 -> 3), then transpose (B, T, 128, 3) -> (B, T, 3, 128) -> (B, T, 384)
    x = conv2d(x, params["cnn"], padding=(1, 1), dtype=prec.compute_dtype, precision=prec.compute_precision) * lvl_masks[0]
    b_dim, t_dim = x.shape[0], x.shape[1]
    x_flat = jnp.transpose(x, (0, 1, 3, 2)).reshape(b_dim, t_dim, 3 * N_MELS)

    # Masked BiGRU in FP32
    gru_p = params["gru"]
    t_idx = jnp.arange(t_dim, dtype=jnp.int32)[None, :]
    valid_mask = t_idx < pad32[:, None]

    h_fwd = _masked_gru_scan(
        x_flat, gru_p["w_ih_fwd"], gru_p["b_ih_fwd"], gru_p["w_hh_fwd"], gru_p["b_hh_fwd"], valid_mask, reverse=False, prec=prec
    )
    h_rev = _masked_gru_scan(
        x_flat, gru_p["w_ih_rev"], gru_p["b_ih_rev"], gru_p["w_hh_rev"], gru_p["b_hh_rev"], valid_mask, reverse=True, prec=prec
    )
    bi_out = jnp.concatenate([h_fwd, h_rev], axis=-1)

    logits = jnp.asarray(linear(bi_out, params["fc_linear"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)
    return jax.nn.sigmoid(logits)


def decode_rmvpe_f0_jax(hidden_btc: Array, threshold: float = 0.03) -> Array:
    """Exact on-device local-average F0 decoding matching decode_rmvpe_f0_np."""
    sal = jnp.asarray(hidden_btc, jnp.float32)
    centers = jnp.argmax(sal, axis=-1).astype(jnp.int32)
    max_sal = jnp.max(sal, axis=-1)
    padded = jnp.pad(sal, ((0, 0), (0, 0), (4, 4)))
    offsets = jnp.arange(-4, 5, dtype=jnp.int32)
    idx = (centers + 4)[..., None] + offsets[None, None, :]
    local_sal = jnp.take_along_axis(padded, idx, axis=-1)
    cents_map = jnp.asarray(
        np.pad(20.0 * np.arange(N_CLASS, dtype=np.float64) + RMVPE_CONST, (4, 4)),
        jnp.float32,
    )
    local_cents = cents_map[idx]
    prod_sum = jnp.sum(local_sal * local_cents, axis=-1)
    w_sum = jnp.sum(local_sal, axis=-1)
    cents = jnp.where(w_sum > 0.0, prod_sum / jnp.maximum(w_sum, 1e-12), 0.0)
    cents = jnp.where(max_sal > jnp.asarray(threshold, jnp.float32), cents, 0.0)
    f0 = 10.0 * jnp.exp2(cents / 1200.0)
    return jnp.where(f0 == 10.0, 0.0, f0)


def rmvpe_f0_forward(
    audio_bt: Array,
    audio_lengths: Array,
    params: Mapping[str, Any],
    *,
    threshold: float = 0.03,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Compute batched RMVPE F0 contours `(B, T_mel)` directly on TPU to minimize host transfer."""
    salience = rmvpe_salience_forward(audio_bt, audio_lengths, params, prec=prec)
    return decode_rmvpe_f0_jax(salience, threshold=threshold)


def decode_rmvpe_f0_np(hidden: np.ndarray, threshold: float = 0.03) -> np.ndarray:
    """Exact NumPy local-average F0 decoding matching RMVPE0Predictor.decode / to_local_average_f0."""
    salience = np.asarray(hidden, dtype=np.float32)
    centers = np.argmax(salience, axis=1)
    padded = np.pad(salience, ((0, 0), (4, 4)))
    centers_p = centers + 4
    offsets = np.arange(-4, 5)
    idx = centers_p[:, None] + offsets[None, :]
    cents_map = np.pad(20.0 * np.arange(N_CLASS, dtype=np.float64) + RMVPE_CONST, (4, 4))
    local_sal = padded[np.arange(salience.shape[0])[:, None], idx]
    prod_sum = np.sum(local_sal * cents_map[idx], axis=1)
    w_sum = np.sum(local_sal, axis=1)
    cents = np.divide(prod_sum, w_sum, out=np.zeros_like(prod_sum), where=w_sum > 0)
    cents[np.max(padded, axis=1) <= float(threshold)] = 0.0
    f0 = 10.0 * (2.0 ** (cents / 1200.0))
    f0[f0 == 10.0] = 0.0
    return f0
