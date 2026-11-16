"""Native JAX/TPU ContentVec & Spin-v2 HuBERT feature extractor matching transformers.HubertModel."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from rvc.jax import PrecisionConfig
from rvc.jax.ops import (
    Array,
    conv1d,
    conv1d_weight_torch_to_jax,
    exact_gelu,
    layer_norm,
    linear,
    linear_weight_torch_to_jax,
    masked_group_norm_1d,
)


@dataclass(frozen=True)
class HubertConfig:
    conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512)
    conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2)
    conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2)
    conv_bias: bool = False
    feat_extract_norm: str = "group"
    feat_proj_layer_norm: bool = True
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16
    do_stable_layer_norm: bool = False
    layer_norm_eps: float = 1e-5
    classifier_proj_size: int = 256
    audio_requires_normalization: bool = False

    @property
    def total_conv_stride(self) -> int:
        return math.prod(self.conv_stride)

    def output_length(self, audio_len: Array | int) -> Array | int:
        n = audio_len
        for k, s in zip(self.conv_kernel, self.conv_stride):
            n = (n - k) // s + 1
        return n


def load_hubert_config(model_dir: str) -> HubertConfig:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    do_norm = False
    preproc_path = os.path.join(model_dir, "preprocessor_config.json")
    if os.path.isfile(preproc_path):
        with open(preproc_path, "r", encoding="utf-8") as pf:
            do_norm = bool(json.load(pf).get("do_normalize", False))
    return HubertConfig(
        conv_dim=tuple(int(x) for x in d["conv_dim"]),
        conv_kernel=tuple(int(x) for x in d["conv_kernel"]),
        conv_stride=tuple(int(x) for x in d["conv_stride"]),
        conv_bias=bool(d.get("conv_bias", False)),
        feat_extract_norm=str(d.get("feat_extract_norm", "group")),
        feat_proj_layer_norm=bool(d.get("feat_proj_layer_norm", True)),
        hidden_size=int(d["hidden_size"]),
        intermediate_size=int(d["intermediate_size"]),
        num_attention_heads=int(d["num_attention_heads"]),
        num_hidden_layers=int(d["num_hidden_layers"]),
        num_conv_pos_embeddings=int(d["num_conv_pos_embeddings"]),
        num_conv_pos_embedding_groups=int(d["num_conv_pos_embedding_groups"]),
        do_stable_layer_norm=bool(d.get("do_stable_layer_norm", False)),
        layer_norm_eps=float(d.get("layer_norm_eps", 1e-5)),
        classifier_proj_size=int(d.get("classifier_proj_size", 256)),
        audio_requires_normalization=do_norm,
    )


def _ln_from_torch(state: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    return {
        "gamma": np.asarray(state[f"{prefix}.weight"], dtype=np.float32).copy(),
        "beta": np.asarray(state[f"{prefix}.bias"], dtype=np.float32).copy(),
    }


def _linear_from_torch(state: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    out = {"weight": linear_weight_torch_to_jax(state[f"{prefix}.weight"])}
    if f"{prefix}.bias" in state and state[f"{prefix}.bias"] is not None:
        out["bias"] = np.asarray(state[f"{prefix}.bias"], dtype=np.float32).copy()
    return out


def _resolve_pos_conv_weight(state: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    """Fold dim=2 weight_norm in PyTorch pos_conv_embed.conv cleanly on CPU."""
    if f"{prefix}.parametrizations.weight.original0" in state:
        g = np.asarray(state[f"{prefix}.parametrizations.weight.original0"], dtype=np.float32)
        v = np.asarray(state[f"{prefix}.parametrizations.weight.original1"], dtype=np.float32)
        v_norm = np.sqrt(np.maximum(np.sum(v * v, axis=(0, 1), keepdims=True), 1e-24))
        w = v * (g / v_norm)
    elif f"{prefix}.weight_g" in state:
        g = np.asarray(state[f"{prefix}.weight_g"], dtype=np.float32)
        v = np.asarray(state[f"{prefix}.weight_v"], dtype=np.float32)
        v_norm = np.sqrt(np.maximum(np.sum(v * v, axis=(0, 1), keepdims=True), 1e-24))
        w = v * (g / v_norm)
    else:
        w = np.asarray(state[f"{prefix}.weight"], dtype=np.float32)
    return {
        "weight": conv1d_weight_torch_to_jax(w),
        "bias": np.asarray(state[f"{prefix}.bias"], dtype=np.float32).copy(),
    }


def hubert_params_from_torch_state(state: Mapping[str, Any], cfg: HubertConfig) -> dict[str, Any]:
    """Convert HuggingFace HubertModelWithFinalProj PyTorch state_dict into JAX PyTree."""
    conv_layers = []
    for i in range(len(cfg.conv_kernel)):
        p = f"feature_extractor.conv_layers.{i}"
        layer_d: dict[str, Any] = {
            "conv": {
                "weight": conv1d_weight_torch_to_jax(state[f"{p}.conv.weight"]),
                **({"bias": np.asarray(state[f"{p}.conv.bias"], dtype=np.float32).copy()} if f"{p}.conv.bias" in state else {}),
            }
        }
        if f"{p}.layer_norm.weight" in state:
            layer_d["norm"] = {
                "weight": np.asarray(state[f"{p}.layer_norm.weight"], dtype=np.float32).copy(),
                "bias": np.asarray(state[f"{p}.layer_norm.bias"], dtype=np.float32).copy(),
            }
        conv_layers.append(layer_d)

    feat_proj: dict[str, Any] = {"projection": _linear_from_torch(state, "feature_projection.projection")}
    if cfg.feat_proj_layer_norm:
        feat_proj["layer_norm"] = _ln_from_torch(state, "feature_projection.layer_norm")

    enc_layers = []
    for i in range(cfg.num_hidden_layers):
        p = f"encoder.layers.{i}"
        enc_layers.append(
            {
                "q_proj": _linear_from_torch(state, f"{p}.attention.q_proj"),
                "k_proj": _linear_from_torch(state, f"{p}.attention.k_proj"),
                "v_proj": _linear_from_torch(state, f"{p}.attention.v_proj"),
                "out_proj": _linear_from_torch(state, f"{p}.attention.out_proj"),
                "layer_norm": _ln_from_torch(state, f"{p}.layer_norm"),
                "intermediate_dense": _linear_from_torch(state, f"{p}.feed_forward.intermediate_dense"),
                "output_dense": _linear_from_torch(state, f"{p}.feed_forward.output_dense"),
                "final_layer_norm": _ln_from_torch(state, f"{p}.final_layer_norm"),
            }
        )

    params: dict[str, Any] = {
        "feature_extractor": conv_layers,
        "feature_projection": feat_proj,
        "pos_conv_embed": _resolve_pos_conv_weight(state, "encoder.pos_conv_embed.conv"),
        "encoder_layer_norm": _ln_from_torch(state, "encoder.layer_norm"),
        "encoder_layers": enc_layers,
    }
    if "final_proj.weight" in state:
        params["final_proj"] = _linear_from_torch(state, "final_proj")
    return params


def load_hubert_from_dir(model_dir: str) -> tuple[HubertConfig, dict[str, Any]]:
    """Load ContentVec or Spin-v2 from local directory containing config.json + pytorch_model.bin."""
    import torch

    cfg = load_hubert_config(model_dir)
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    ckpt = torch.load(bin_path, map_location="cpu", weights_only=True)
    raw = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {k: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v) for k, v in raw.items()}
    return cfg, hubert_params_from_torch_state(state, cfg)


def _hubert_attention(
    hidden_states: Array,
    attn_mask_bt: Array,
    layer_p: Mapping[str, Any],
    cfg: HubertConfig,
    prec: PrecisionConfig,
) -> Array:
    b, t, d = hidden_states.shape
    h = cfg.num_attention_heads
    hd = d // h
    scale = hd ** -0.5

    q = jnp.asarray(linear(hidden_states, layer_p["q_proj"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)
    k = jnp.asarray(linear(hidden_states, layer_p["k_proj"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)
    v = jnp.asarray(linear(hidden_states, layer_p["v_proj"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)

    q_h = jnp.transpose(q.reshape(b, t, h, hd), (0, 2, 1, 3))
    k_h = jnp.transpose(k.reshape(b, t, h, hd), (0, 2, 1, 3))
    v_h = jnp.transpose(v.reshape(b, t, h, hd), (0, 2, 1, 3))

    scores = jnp.matmul(q_h, jnp.swapaxes(k_h, -2, -1), precision=prec.fp32_precision) * scale
    scores = jnp.where(attn_mask_bt[:, None, None, :], scores, -1e9)
    probs = jax.nn.softmax(scores, axis=-1)
    ctx = jnp.matmul(probs, v_h, precision=prec.fp32_precision)
    ctx_flat = jnp.transpose(ctx, (0, 2, 1, 3)).reshape(b, t, d)
    return jnp.asarray(linear(ctx_flat, layer_p["out_proj"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)


def hubert_forward(
    audio_bt: Array,
    audio_lengths: Array,
    params: Mapping[str, Any],
    cfg: HubertConfig,
    *,
    version: str = "v2",
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Batched HuBERT inference on NWC-friendly waveforms `(B, T_wav)` -> FP32 features `(B, T_feat, D)`."""
    x = jnp.asarray(audio_bt, jnp.float32)
    lens = jnp.asarray(audio_lengths, jnp.int32)

    if cfg.audio_requires_normalization:
        w_mask = (jnp.arange(x.shape[1], dtype=jnp.int32)[None, :] < lens[:, None]).astype(jnp.float32)
        denom = jnp.maximum(jnp.sum(w_mask, axis=1, keepdims=True), 1.0)
        mean = jnp.sum(x * w_mask, axis=1, keepdims=True) / denom
        var = jnp.sum(jnp.square(x - mean) * w_mask, axis=1, keepdims=True) / denom
        x = (x - mean) * lax.rsqrt(var + 1e-7) * w_mask

    h = x[..., None]  # (B, T_wav, 1) NWC
    cur_len = lens
    for idx, (k, s, layer_p) in enumerate(zip(cfg.conv_kernel, cfg.conv_stride, params["feature_extractor"])):
        h = conv1d(h, layer_p["conv"], stride=s, padding=0, dtype=prec.compute_dtype, precision=prec.compute_precision)
        cur_len = (cur_len - k) // s + 1
        if "norm" in layer_p:
            c_mask = (jnp.arange(h.shape[1], dtype=jnp.int32)[None, :] < cur_len[:, None]).astype(jnp.float32)[..., None]
            if cfg.feat_extract_norm == "group":
                h = masked_group_norm_1d(h, layer_p["norm"], num_groups=cfg.conv_dim[idx], mask_bt1=c_mask, eps=cfg.layer_norm_eps)
            else:
                h = layer_norm(h, {"gamma": layer_p["norm"]["weight"], "beta": layer_p["norm"]["bias"]}, eps=cfg.layer_norm_eps)
        h = exact_gelu(h)

    # Feature projection
    if cfg.feat_proj_layer_norm:
        h = layer_norm(h, params["feature_projection"]["layer_norm"], eps=cfg.layer_norm_eps)
    h = jnp.asarray(linear(h, params["feature_projection"]["projection"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)

    # Mask padded frames strictly to 0 before pos_conv_embed (exact parity with batch=1 unpadded)
    t_feat = h.shape[1]
    feat_mask_bt = jnp.arange(t_feat, dtype=jnp.int32)[None, :] < cur_len[:, None]
    feat_mask_bt1 = feat_mask_bt.astype(jnp.float32)[..., None]
    h = h * feat_mask_bt1

    pos_pad = cfg.num_conv_pos_embeddings // 2
    pos = conv1d(
        h,
        params["pos_conv_embed"],
        stride=1,
        padding=pos_pad,
        groups=cfg.num_conv_pos_embedding_groups,
        dtype=prec.compute_dtype,
        precision=prec.compute_precision,
    )
    if cfg.num_conv_pos_embeddings % 2 == 0:
        pos = pos[:, :-1, :]
    pos = exact_gelu(jnp.asarray(pos, jnp.float32))
    h = h + pos

    if not cfg.do_stable_layer_norm:
        h = layer_norm(h, params["encoder_layer_norm"], eps=cfg.layer_norm_eps)

    n_layers = 9 if version == "v1" else cfg.num_hidden_layers
    for layer_p in params["encoder_layers"][:n_layers]:
        if cfg.do_stable_layer_norm:
            h_in = layer_norm(h, layer_p["layer_norm"], eps=cfg.layer_norm_eps)
            attn_out = _hubert_attention(h_in, feat_mask_bt, layer_p, cfg, prec)
            h = h + attn_out
            h_ffn = layer_norm(h, layer_p["final_layer_norm"], eps=cfg.layer_norm_eps)
            ffn_1 = exact_gelu(linear(h_ffn, layer_p["intermediate_dense"], dtype=prec.compute_dtype, precision=prec.compute_precision))
            ffn_2 = jnp.asarray(linear(ffn_1, layer_p["output_dense"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)
            h = h + ffn_2
        else:
            attn_out = _hubert_attention(h, feat_mask_bt, layer_p, cfg, prec)
            h = layer_norm(h + attn_out, layer_p["layer_norm"], eps=cfg.layer_norm_eps)
            ffn_1 = exact_gelu(linear(h, layer_p["intermediate_dense"], dtype=prec.compute_dtype, precision=prec.compute_precision))
            ffn_2 = jnp.asarray(linear(ffn_1, layer_p["output_dense"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)
            h = layer_norm(h + ffn_2, layer_p["final_layer_norm"], eps=cfg.layer_norm_eps)

    if cfg.do_stable_layer_norm and version != "v1":
        h = layer_norm(h, params["encoder_layer_norm"], eps=cfg.layer_norm_eps)

    if version == "v1":
        return jnp.asarray(linear(h, params["final_proj"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)
    return jnp.asarray(h, jnp.float32)
