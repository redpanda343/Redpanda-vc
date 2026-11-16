"""JAX NWC MultiHeadAttention and FFN modules matching rvc.algorithm.attentions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from rvc.jax import PrecisionConfig
from rvc.jax.ops import (
    Array,
    constant_pad_1d,
    conv1d,
    dropout,
    export_conv_to_torch_state,
    extract_conv_from_torch_state,
    gelu_1702,
    init_conv1d_params,
)


@dataclass(frozen=True)
class MultiHeadAttentionConfig:
    channels: int
    out_channels: int
    n_heads: int
    p_dropout: float = 0.0
    window_size: int | None = None
    heads_share: bool = True
    block_length: int | None = None
    proximal_bias: bool = False
    proximal_init: bool = False

    @property
    def k_channels(self) -> int:
        if self.channels % self.n_heads != 0:
            raise ValueError(f"channels ({self.channels}) must be divisible by n_heads ({self.n_heads})")
        return self.channels // self.n_heads


def init_mha_params(rng: np.random.Generator, cfg: MultiHeadAttentionConfig) -> dict[str, Any]:
    """Initialize MultiHeadAttention parameters matching PyTorch xavier_uniform init."""
    params: dict[str, Any] = {
        "conv_q": _init_xavier_conv1d_k1(rng, cfg.channels, cfg.channels),
        "conv_k": _init_xavier_conv1d_k1(rng, cfg.channels, cfg.channels),
        "conv_v": _init_xavier_conv1d_k1(rng, cfg.channels, cfg.channels),
        "conv_o": init_conv1d_params(rng, cfg.channels, cfg.out_channels, kernel_size=1),
    }
    if cfg.proximal_init:
        params["conv_k"] = {
            "weight": params["conv_q"]["weight"].copy(),
            "bias": params["conv_q"]["bias"].copy(),
        }
    if cfg.window_size is not None:
        n_heads_rel = 1 if cfg.heads_share else cfg.n_heads
        rel_stddev = cfg.k_channels ** -0.5
        rel_shape = (n_heads_rel, 2 * cfg.window_size + 1, cfg.k_channels)
        params["emb_rel_k"] = rng.normal(0.0, rel_stddev, size=rel_shape).astype(np.float32)
        params["emb_rel_v"] = rng.normal(0.0, rel_stddev, size=rel_shape).astype(np.float32)
    return params


def _init_xavier_conv1d_k1(rng: np.random.Generator, in_ch: int, out_ch: int) -> dict[str, np.ndarray]:
    bound = math.sqrt(6.0 / float(in_ch + out_ch))
    w = rng.uniform(-bound, bound, size=(1, in_ch, out_ch)).astype(np.float32)
    b_bound = 1.0 / math.sqrt(max(1, in_ch))
    b = rng.uniform(-b_bound, b_bound, size=(out_ch,)).astype(np.float32)
    return {"weight": w, "bias": b}


def _relative_embeddings(embeddings: Array, window_size: int, window: int) -> Array:
    start = window_size - window
    return embeddings[:, start : start + 2 * window + 1][None, ...]


def _band_to_absolute(x: Array, window: int) -> Array:
    batch, heads, length, width = x.shape
    span = length + 2 * window
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (0, span + 1 - width)))
    x_flat = x_pad.reshape(batch, heads, length * (span + 1))[:, :, : length * span]
    return x_flat.reshape(batch, heads, length, span)[:, :, :, window : window + length]


def _absolute_to_band(x: Array, window: int) -> Array:
    batch, heads, length, _ = x.shape
    span = length + 2 * window
    x_pad = jnp.pad(x, ((0, 0), (0, 0), (0, 0), (window, window))).reshape(batch, heads, length * span)
    x_pad2 = jnp.pad(x_pad, ((0, 0), (0, 0), (0, length)))
    return x_pad2.reshape(batch, heads, length, span + 1)[:, :, :, : 2 * window + 1]


def _attention_bias_proximal(length: int) -> Array:
    r = jnp.arange(length, dtype=jnp.float32)
    diff = r[None, :] - r[:, None]
    return -jnp.log1p(jnp.abs(diff))[None, None, :, :]


def mha_forward(
    x: Array,
    c: Array,
    params: Mapping[str, Any],
    cfg: MultiHeadAttentionConfig,
    attn_mask: Array | None = None,
    rng_key: Array | None = None,
    deterministic: bool = True,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Apply MultiHeadAttention on NWC inputs (B, T, C) with strict FP32 attention math."""
    q = conv1d(x, params["conv_q"], dtype=prec.compute_dtype, precision=prec.compute_precision)
    k = conv1d(c, params["conv_k"], dtype=prec.compute_dtype, precision=prec.compute_precision)
    v = conv1d(c, params["conv_v"], dtype=prec.compute_dtype, precision=prec.compute_precision)

    # Attention internals always run in strict FP32 (not bf16-eligible).
    q_f32 = jnp.asarray(q, jnp.float32)
    k_f32 = jnp.asarray(k, jnp.float32)
    v_f32 = jnp.asarray(v, jnp.float32)

    b, t_s, d = k_f32.shape
    t_t = q_f32.shape[1]
    q_h = jnp.transpose(q_f32.reshape(b, t_t, cfg.n_heads, cfg.k_channels), (0, 2, 1, 3))
    k_h = jnp.transpose(k_f32.reshape(b, t_s, cfg.n_heads, cfg.k_channels), (0, 2, 1, 3))
    v_h = jnp.transpose(v_f32.reshape(b, t_s, cfg.n_heads, cfg.k_channels), (0, 2, 1, 3))

    q_scaled = q_h / math.sqrt(cfg.k_channels)
    scores = jnp.matmul(q_scaled, jnp.swapaxes(k_h, -2, -1), precision=prec.fp32_precision)

    window = 0
    if cfg.window_size is not None:
        assert t_s == t_t, "Relative attention only supports self-attention."
        window = min(cfg.window_size, t_s - 1)
        rel_k = _relative_embeddings(jnp.asarray(params["emb_rel_k"], jnp.float32), cfg.window_size, window)
        rel_scores = jnp.matmul(q_scaled, jnp.swapaxes(rel_k, -2, -1), precision=prec.fp32_precision)
        scores = scores + _band_to_absolute(rel_scores, window)

    if cfg.proximal_bias:
        assert t_s == t_t, "Proximal bias only supports self-attention."
        scores = scores + _attention_bias_proximal(t_s)

    if attn_mask is not None:
        scores = jnp.where(attn_mask == 0, -1e4, scores)
        if cfg.block_length is not None:
            ones = jnp.ones_like(scores)
            block_mask = jnp.tril(jnp.triu(ones, -cfg.block_length), cfg.block_length)
            scores = jnp.where(block_mask == 0, -1e4, scores)

    p_attn = jax.nn.softmax(scores, axis=-1)
    p_attn = dropout(p_attn, rng_key, cfg.p_dropout, deterministic=deterministic)

    out_h = jnp.matmul(p_attn, v_h, precision=prec.fp32_precision)
    if cfg.window_size is not None:
        rel_v = _relative_embeddings(jnp.asarray(params["emb_rel_v"], jnp.float32), cfg.window_size, window)
        out_h = out_h + jnp.matmul(_absolute_to_band(p_attn, window), rel_v, precision=prec.fp32_precision)

    out_nwc = jnp.transpose(out_h, (0, 2, 1, 3)).reshape(b, t_t, d)
    return conv1d(out_nwc, params["conv_o"], dtype=prec.compute_dtype, precision=prec.compute_precision)


@dataclass(frozen=True)
class FFNConfig:
    in_channels: int
    out_channels: int
    filter_channels: int
    kernel_size: int
    p_dropout: float = 0.0
    activation: str | None = None
    causal: bool = False


def init_ffn_params(rng: np.random.Generator, cfg: FFNConfig) -> dict[str, Any]:
    """Initialize FFN convolution parameters matching PyTorch Conv1d defaults."""
    return {
        "conv_1": init_conv1d_params(rng, cfg.in_channels, cfg.filter_channels, cfg.kernel_size),
        "conv_2": init_conv1d_params(rng, cfg.filter_channels, cfg.out_channels, cfg.kernel_size),
    }


def _ffn_pad(x: Array, cfg: FFNConfig) -> Array:
    if cfg.causal:
        return constant_pad_1d(x, cfg.kernel_size - 1, 0)
    pad_l = (cfg.kernel_size - 1) // 2
    pad_r = cfg.kernel_size // 2
    return constant_pad_1d(x, pad_l, pad_r)


def ffn_forward(
    x: Array,
    x_mask: Array,
    params: Mapping[str, Any],
    cfg: FFNConfig,
    rng_key: Array | None = None,
    deterministic: bool = True,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Apply FFN on NWC input (B, T, C) and mask (B, T, 1)."""
    mask_c = jnp.asarray(x_mask, prec.compute_dtype)
    h = conv1d(_ffn_pad(jnp.asarray(x, prec.compute_dtype) * mask_c, cfg), params["conv_1"], dtype=prec.compute_dtype, precision=prec.compute_precision)
    if cfg.activation == "gelu":
        h = gelu_1702(h)
    else:
        h = jax.nn.relu(h)
    h = dropout(h, rng_key, cfg.p_dropout, deterministic=deterministic)
    h = conv1d(_ffn_pad(h * mask_c, cfg), params["conv_2"], dtype=prec.compute_dtype, precision=prec.compute_precision)
    return jnp.asarray(h, jnp.float32) * jnp.asarray(x_mask, jnp.float32)


def mha_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: MultiHeadAttentionConfig) -> dict[str, Any]:
    params: dict[str, Any] = {
        "conv_q": extract_conv_from_torch_state(state, f"{prefix}.conv_q"),
        "conv_k": extract_conv_from_torch_state(state, f"{prefix}.conv_k"),
        "conv_v": extract_conv_from_torch_state(state, f"{prefix}.conv_v"),
        "conv_o": extract_conv_from_torch_state(state, f"{prefix}.conv_o"),
    }
    if cfg.window_size is not None:
        params["emb_rel_k"] = np.asarray(state[f"{prefix}.emb_rel_k"], dtype=np.float32).copy()
        params["emb_rel_v"] = np.asarray(state[f"{prefix}.emb_rel_v"], dtype=np.float32).copy()
    return params


def mha_to_torch_state(state: dict[str, np.ndarray], prefix: str, params: Mapping[str, Any]) -> None:
    export_conv_to_torch_state(state, f"{prefix}.conv_q", params["conv_q"])
    export_conv_to_torch_state(state, f"{prefix}.conv_k", params["conv_k"])
    export_conv_to_torch_state(state, f"{prefix}.conv_v", params["conv_v"])
    export_conv_to_torch_state(state, f"{prefix}.conv_o", params["conv_o"])
    if "emb_rel_k" in params:
        state[f"{prefix}.emb_rel_k"] = np.asarray(params["emb_rel_k"], dtype=np.float32).copy()
        state[f"{prefix}.emb_rel_v"] = np.asarray(params["emb_rel_v"], dtype=np.float32).copy()


def ffn_from_torch_state(state: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    return {
        "conv_1": extract_conv_from_torch_state(state, f"{prefix}.conv_1"),
        "conv_2": extract_conv_from_torch_state(state, f"{prefix}.conv_2"),
    }


def ffn_to_torch_state(state: dict[str, np.ndarray], prefix: str, params: Mapping[str, Any]) -> None:
    export_conv_to_torch_state(state, f"{prefix}.conv_1", params["conv_1"])
    export_conv_to_torch_state(state, f"{prefix}.conv_2", params["conv_2"])
