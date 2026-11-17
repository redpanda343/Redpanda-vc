"""JAX NWC/NHWC MultiPeriodDiscriminator (S, P, R) matching rvc.algorithm.discriminators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rvc.jax import PrecisionConfig
from rvc.jax.ops import (
    Array,
    conv1d,
    conv2d,
    export_conv_to_torch_state,
    extract_conv_from_torch_state,
    get_padding,
    init_conv1d_params,
    init_conv2d_params,
    leaky_relu,
    reflect_pad_1d,
    resolve_kernel,
    stft_complex,
)
from rvc.jax.residuals import LRELU_SLOPE

_DISCRIMINATOR_S_SPECS = (
    (1, 16, 15, 1, 7, 1),
    (16, 64, 41, 4, 20, 4),
    (64, 256, 41, 4, 20, 16),
    (256, 1024, 41, 4, 20, 64),
    (1024, 1024, 41, 4, 20, 256),
    (1024, 1024, 5, 1, 2, 1),
)

_DISCRIMINATOR_P_SPECS = (
    (1, 32, 3),
    (32, 128, 3),
    (128, 512, 3),
    (512, 1024, 3),
    (1024, 1024, 1),
)

_DISCRIMINATOR_R_SPECS = (
    (1, 32, (3, 9), (1, 1), (1, 4)),
    (32, 32, (3, 9), (1, 2), (1, 4)),
    (32, 32, (3, 9), (1, 2), (1, 4)),
    (32, 32, (3, 9), (1, 2), (1, 4)),
    (32, 32, (3, 3), (1, 1), (1, 1)),
)


@dataclass(frozen=True)
class MultiPeriodDiscriminatorConfig:
    use_spectral_norm: bool = False
    checkpointing: bool = False
    version: str = "v2"

    @property
    def periods(self) -> tuple[int, ...]:
        if self.version == "v1":
            return (2, 3, 5, 7, 11, 17)
        if self.version == "v3":
            return (2, 3, 5, 7, 11)
        return (2, 3, 5, 7, 11, 17, 23, 37)

    @property
    def resolutions(self) -> tuple[tuple[int, int, int], ...]:
        if self.version == "v3":
            return ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))
        return ()


def _init_disc_s(rng: np.random.Generator, use_spectral_norm: bool) -> dict[str, Any]:
    wn = not use_spectral_norm
    convs = [
        init_conv1d_params(rng, in_c, out_c, k, groups=g, weight_norm=wn, spectral_norm=use_spectral_norm)
        for in_c, out_c, k, _, _, g in _DISCRIMINATOR_S_SPECS
    ]
    conv_post = init_conv1d_params(rng, 1024, 1, 3, weight_norm=wn, spectral_norm=use_spectral_norm)
    return {"convs": convs, "conv_post": conv_post}


def _init_disc_p(rng: np.random.Generator, use_spectral_norm: bool, kernel_size: int = 5) -> dict[str, Any]:
    wn = not use_spectral_norm
    convs = [
        init_conv2d_params(rng, in_c, out_c, (kernel_size, 1), weight_norm=wn, spectral_norm=use_spectral_norm)
        for in_c, out_c, _ in _DISCRIMINATOR_P_SPECS
    ]
    conv_post = init_conv2d_params(rng, 1024, 1, (3, 1), weight_norm=wn, spectral_norm=use_spectral_norm)
    return {"convs": convs, "conv_post": conv_post}


def _init_disc_r(rng: np.random.Generator, use_spectral_norm: bool) -> dict[str, Any]:
    wn = not use_spectral_norm
    convs = [
        init_conv2d_params(rng, in_c, out_c, kernel, weight_norm=wn, spectral_norm=use_spectral_norm)
        for in_c, out_c, kernel, _, _ in _DISCRIMINATOR_R_SPECS
    ]
    conv_post = init_conv2d_params(rng, 32, 1, (3, 3), weight_norm=wn, spectral_norm=use_spectral_norm)
    return {"convs": convs, "conv_post": conv_post}


def init_mpd_params(rng: np.random.Generator, cfg: MultiPeriodDiscriminatorConfig) -> dict[str, Any]:
    """Initialize MultiPeriodDiscriminator parameters matching PyTorch disc.discriminators list."""
    discs: list[dict[str, Any]] = [_init_disc_s(rng, cfg.use_spectral_norm)]
    discs.extend(_init_disc_p(rng, cfg.use_spectral_norm) for _ in cfg.periods)
    discs.extend(_init_disc_r(rng, cfg.use_spectral_norm) for _ in cfg.resolutions)
    return {"discriminators": discs}


def _disc_s_forward(x: Array, params: Mapping[str, Any], prec: PrecisionConfig) -> tuple[Array, list[Array]]:
    fmap: list[Array] = []
    cur = jnp.asarray(x, prec.compute_dtype)
    for layer_p, (_, _, _, stride, pad, groups) in zip(params["convs"], _DISCRIMINATOR_S_SPECS):
        cur = conv1d(cur, layer_p, stride=stride, padding=pad, groups=groups, dtype=prec.compute_dtype, precision=prec.compute_precision)
        cur = leaky_relu(cur, LRELU_SLOPE)
        fmap.append(cur)
    cur = conv1d(cur, params["conv_post"], stride=1, padding=1, dtype=prec.compute_dtype, precision=prec.compute_precision)
    fmap.append(cur)
    return cur.reshape(cur.shape[0], -1), fmap


def _disc_p_1d_layer_params(layer_p: Mapping[str, Any]) -> dict[str, Any]:
    w = resolve_kernel(layer_p)
    w_1d = w[:, 0, :, :] if w.ndim == 4 else w
    out: dict[str, Any] = {"weight": w_1d}
    if "bias" in layer_p and layer_p["bias"] is not None:
        out["bias"] = layer_p["bias"]
    return out


def _disc_p_forward(
    x: Array,
    params: Mapping[str, Any],
    period: int,
    prec: PrecisionConfig,
    kernel_size: int = 5,
    *,
    transpose_fmap: bool = True,
) -> tuple[Array, list[Array]]:
    """Run period discriminator in packed 1D NWC (B*P, T//P, C) to avoid narrow W-tile padding on TPU."""
    fmap: list[Array] = []
    b, t, c = x.shape
    if t % period != 0:
        n_pad = period - (t % period)
        x = reflect_pad_1d(x, 0, n_pad)
        t = t + n_pad
    t_sub = t // period
    cur = jnp.transpose(x.reshape(b, t_sub, period, c), (0, 2, 1, 3)).reshape(b * period, t_sub, c)
    cur = jnp.asarray(cur, prec.compute_dtype)
    pad_h = get_padding(kernel_size, 1)
    for layer_p, (_, _, stride) in zip(params["convs"], _DISCRIMINATOR_P_SPECS):
        cur = conv1d(
            cur,
            _disc_p_1d_layer_params(layer_p),
            stride=stride,
            padding=pad_h,
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        cur = leaky_relu(cur, LRELU_SLOPE)
        fmap.append(cur.reshape(b, period, cur.shape[1], cur.shape[2]).transpose(0, 2, 1, 3) if transpose_fmap else cur)
    cur = conv1d(
        cur,
        _disc_p_1d_layer_params(params["conv_post"]),
        stride=1,
        padding=1,
        dtype=prec.compute_dtype,
        precision=prec.compute_precision,
    )
    cur_4d = cur.reshape(b, period, cur.shape[1], cur.shape[2]).transpose(0, 2, 1, 3)
    fmap.append(cur_4d if transpose_fmap else cur)
    return cur_4d.reshape(b, -1), fmap


def _disc_r_spectrogram(x: Array, resolution: tuple[int, int, int]) -> Array:
    n_fft, hop_length, win_length = resolution
    pad = (n_fft - hop_length) // 2
    win = np.ones((win_length,), dtype=np.float32)
    real, imag = stft_complex(x, n_fft, hop_length, win_length, win, pad_left=pad, pad_right=pad)
    # Use eps guard on norm^2 so zero signals produce finite gradients.
    mag = jnp.sqrt(jnp.maximum(real * real + imag * imag, 1e-24))
    return mag[..., None]  # (B, F, TT, 1) NHWC


def _disc_r_forward(
    x: Array,
    params: Mapping[str, Any],
    resolution: tuple[int, int, int],
    prec: PrecisionConfig,
) -> tuple[Array, list[Array]]:
    fmap: list[Array] = []
    cur = jnp.asarray(_disc_r_spectrogram(x, resolution), prec.compute_dtype)
    for layer_p, (_, _, _, stride, pad) in zip(params["convs"], _DISCRIMINATOR_R_SPECS):
        cur = conv2d(cur, layer_p, stride=stride, padding=pad, dtype=prec.compute_dtype, precision=prec.compute_precision)
        cur = leaky_relu(cur, 0.1)
        fmap.append(cur)
    cur = conv2d(cur, params["conv_post"], stride=(1, 1), padding=(1, 1), dtype=prec.compute_dtype, precision=prec.compute_precision)
    fmap.append(cur)
    return cur.reshape(cur.shape[0], -1), fmap


def resolve_mpd_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Pre-resolve weight_norm/spectral_norm kernels once for frozen discriminator passes."""
    resolved_discs: list[dict[str, Any]] = []
    for disc_p in params["discriminators"]:
        convs = [
            {
                "weight": resolve_kernel(layer_p),
                **({"bias": layer_p["bias"]} if "bias" in layer_p and layer_p["bias"] is not None else {}),
            }
            for layer_p in disc_p["convs"]
        ]
        cp = disc_p["conv_post"]
        conv_post = {
            "weight": resolve_kernel(cp),
            **({"bias": cp["bias"]} if "bias" in cp and cp["bias"] is not None else {}),
        }
        resolved_discs.append({"convs": convs, "conv_post": conv_post})
    return {"discriminators": resolved_discs}


def mpd_forward_single(
    params: Mapping[str, Any],
    cfg: MultiPeriodDiscriminatorConfig,
    x: Array,
    prec: PrecisionConfig = PrecisionConfig(),
    *,
    transpose_fmap: bool = True,
) -> tuple[list[Array], list[list[Array]]]:
    """Single-batch (B) discriminator forward pass for independent real/fake evaluation."""
    x_f32 = jnp.asarray(x, jnp.float32)
    discs = params["discriminators"]
    logits_list: list[Array] = []
    fmap_list: list[list[Array]] = []

    def _run_sub(fn, p_sub, x_in):
        return jax.checkpoint(fn)(x_in, p_sub) if cfg.checkpointing else fn(x_in, p_sub)

    logits_s, fmap_s = _run_sub(lambda x_, p_: _disc_s_forward(x_, p_, prec), discs[0], x_f32)
    logits_list.append(logits_s)
    fmap_list.append(fmap_s)

    idx = 1
    for period in cfg.periods:
        p_val = int(period)
        logits_p, fmap_p = _run_sub(
            lambda x_, p_, per=p_val: _disc_p_forward(x_, p_, per, prec, transpose_fmap=transpose_fmap),
            discs[idx],
            x_f32,
        )
        logits_list.append(logits_p)
        fmap_list.append(fmap_p)
        idx += 1

    for resolution in cfg.resolutions:
        res_t = tuple(int(v) for v in resolution)
        logits_r, fmap_r = _run_sub(lambda x_, p_, res=res_t: _disc_r_forward(x_, p_, res, prec), discs[idx], x_f32)
        logits_list.append(logits_r)
        fmap_list.append(fmap_r)
        idx += 1

    return logits_list, fmap_list


def mpd_forward(
    params: Mapping[str, Any],
    cfg: MultiPeriodDiscriminatorConfig,
    y: Array,
    y_hat: Array,
    prec: PrecisionConfig = PrecisionConfig(),
    *,
    transpose_fmap: bool = True,
) -> tuple[list[Array], list[Array], list[list[Array]], list[list[Array]]]:
    """Batched 2B discriminator forward for (y, y_hat) maximizing TPU MXU utilization."""
    batch = y.shape[0]
    y_both = jnp.concatenate([jnp.asarray(y, jnp.float32), jnp.asarray(y_hat, jnp.float32)], axis=0)
    discs = params["discriminators"]

    y_d_rs: list[Array] = []
    y_d_gs: list[Array] = []
    fmap_rs: list[list[Array]] = []
    fmap_gs: list[list[Array]] = []

    def _run_sub(fn, p_sub, x_in):
        return jax.checkpoint(fn)(x_in, p_sub) if cfg.checkpointing else fn(x_in, p_sub)

    # Index 0: DiscriminatorS
    logits_s, fmap_s = _run_sub(lambda x_, p_: _disc_s_forward(x_, p_, prec), discs[0], y_both)
    y_d_rs.append(logits_s[:batch])
    y_d_gs.append(logits_s[batch:])
    fmap_rs.append([f[:batch] for f in fmap_s])
    fmap_gs.append([f[batch:] for f in fmap_s])

    idx = 1
    for period in cfg.periods:
        p_val = int(period)
        logits_p, fmap_p = _run_sub(
            lambda x_, p_, per=p_val: _disc_p_forward(x_, p_, per, prec, transpose_fmap=transpose_fmap),
            discs[idx],
            y_both,
        )
        y_d_rs.append(logits_p[:batch])
        y_d_gs.append(logits_p[batch:])
        split = batch if transpose_fmap else batch * p_val
        fmap_rs.append([f[:split] for f in fmap_p])
        fmap_gs.append([f[split:] for f in fmap_p])
        idx += 1

    for resolution in cfg.resolutions:
        res_t = tuple(int(v) for v in resolution)
        logits_r, fmap_r = _run_sub(lambda x_, p_, res=res_t: _disc_r_forward(x_, p_, res, prec), discs[idx], y_both)
        y_d_rs.append(logits_r[:batch])
        y_d_gs.append(logits_r[batch:])
        fmap_rs.append([f[:batch] for f in fmap_r])
        fmap_gs.append([f[batch:] for f in fmap_r])
        idx += 1

    return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def mpd_from_torch_state(state: Mapping[str, Any], cfg: MultiPeriodDiscriminatorConfig) -> dict[str, Any]:
    """Load PyTorch MultiPeriodDiscriminator state_dict into JAX param PyTree."""
    discs: list[dict[str, Any]] = []
    # Index 0: DiscriminatorS
    p0 = "discriminators.0"
    discs.append(
        {
            "convs": [extract_conv_from_torch_state(state, f"{p0}.convs.{j}") for j in range(len(_DISCRIMINATOR_S_SPECS))],
            "conv_post": extract_conv_from_torch_state(state, f"{p0}.conv_post"),
        }
    )
    total = 1 + len(cfg.periods) + len(cfg.resolutions)
    for idx in range(1, total):
        pi = f"discriminators.{idx}"
        n_convs = len(_DISCRIMINATOR_P_SPECS) if idx <= len(cfg.periods) else len(_DISCRIMINATOR_R_SPECS)
        discs.append(
            {
                "convs": [extract_conv_from_torch_state(state, f"{pi}.convs.{j}", is_2d=True) for j in range(n_convs)],
                "conv_post": extract_conv_from_torch_state(state, f"{pi}.conv_post", is_2d=True),
            }
        )
    return {"discriminators": discs}


def mpd_to_torch_state(
    params: Mapping[str, Any],
    cfg: MultiPeriodDiscriminatorConfig,
    *,
    remove_weight_norm: bool = False,
) -> dict[str, np.ndarray]:
    """Export JAX MultiPeriodDiscriminator param PyTree into PyTorch state_dict."""
    state: dict[str, np.ndarray] = {}
    discs = params["discriminators"]
    for idx, d_params in enumerate(discs):
        is_2d = idx > 0
        pi = f"discriminators.{idx}"
        for j, conv_p in enumerate(d_params["convs"]):
            export_conv_to_torch_state(state, f"{pi}.convs.{j}", conv_p, is_2d=is_2d, remove_weight_norm=remove_weight_norm)
        export_conv_to_torch_state(state, f"{pi}.conv_post", d_params["conv_post"], is_2d=is_2d, remove_weight_norm=remove_weight_norm)
    return state
