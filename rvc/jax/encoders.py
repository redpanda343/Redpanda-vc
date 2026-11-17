"""JAX NWC Encoder, TextEncoder, and PosteriorEncoder matching rvc.algorithm.encoders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from rvc.jax import PrecisionConfig
from rvc.jax.attentions import (
    FFNConfig,
    MultiHeadAttentionConfig,
    ffn_forward,
    ffn_from_torch_state,
    ffn_to_torch_state,
    init_ffn_params,
    init_mha_params,
    mha_forward,
    mha_from_torch_state,
    mha_to_torch_state,
)
from rvc.jax.modules import (
    WaveNetConfig,
    init_wavenet_params,
    wavenet_forward,
    wavenet_from_torch_state,
    wavenet_to_torch_state,
)
from rvc.jax.ops import (
    Array,
    conv1d,
    dropout,
    embedding,
    export_conv_to_torch_state,
    export_layer_norm_to_torch_state,
    export_linear_to_torch_state,
    extract_conv_from_torch_state,
    extract_layer_norm_from_torch_state,
    extract_linear_from_torch_state,
    init_conv1d_params,
    init_embedding_params,
    init_layer_norm_params,
    init_linear_params,
    layer_norm,
    leaky_relu,
    linear,
    sequence_mask,
)


@dataclass(frozen=True)
class EncoderConfig:
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int = 1
    p_dropout: float = 0.0
    window_size: int = 10


def init_encoder_params(rng: np.random.Generator, cfg: EncoderConfig) -> dict[str, Any]:
    mha_cfg = MultiHeadAttentionConfig(
        channels=cfg.hidden_channels,
        out_channels=cfg.hidden_channels,
        n_heads=cfg.n_heads,
        p_dropout=cfg.p_dropout,
        window_size=cfg.window_size,
    )
    ffn_cfg = FFNConfig(
        in_channels=cfg.hidden_channels,
        out_channels=cfg.hidden_channels,
        filter_channels=cfg.filter_channels,
        kernel_size=cfg.kernel_size,
        p_dropout=cfg.p_dropout,
    )
    return {
        "attn_layers": [init_mha_params(rng, mha_cfg) for _ in range(cfg.n_layers)],
        "norm_layers_1": [init_layer_norm_params(cfg.hidden_channels) for _ in range(cfg.n_layers)],
        "ffn_layers": [init_ffn_params(rng, ffn_cfg) for _ in range(cfg.n_layers)],
        "norm_layers_2": [init_layer_norm_params(cfg.hidden_channels) for _ in range(cfg.n_layers)],
    }


def encoder_forward(
    x: Array,
    x_mask: Array,
    params: Mapping[str, Any],
    cfg: EncoderConfig,
    rng_key: Array | None = None,
    deterministic: bool = True,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Apply Transformer Encoder on NWC (B, T, H) and mask (B, T, 1)."""
    mha_cfg = MultiHeadAttentionConfig(
        channels=cfg.hidden_channels,
        out_channels=cfg.hidden_channels,
        n_heads=cfg.n_heads,
        p_dropout=cfg.p_dropout,
        window_size=cfg.window_size,
    )
    ffn_cfg = FFNConfig(
        in_channels=cfg.hidden_channels,
        out_channels=cfg.hidden_channels,
        filter_channels=cfg.filter_channels,
        kernel_size=cfg.kernel_size,
        p_dropout=cfg.p_dropout,
    )
    # (B, T, 1) -> (B, 1, 1, T) matching PyTorch x_mask.unsqueeze(2) on (B, 1, T)
    attn_mask = x_mask[:, None, None, :, 0]
    x_f32 = jnp.asarray(x, jnp.float32) * jnp.asarray(x_mask, jnp.float32)

    for i in range(cfg.n_layers):
        k_attn = k_d1 = k_ffn = k_d2 = None
        if rng_key is not None and not deterministic and cfg.p_dropout > 0.0:
            rng_key, k_attn, k_d1, k_ffn, k_d2 = jax.random.split(rng_key, 5)
        y = mha_forward(x_f32, x_f32, params["attn_layers"][i], mha_cfg, attn_mask=attn_mask, rng_key=k_attn, deterministic=deterministic, prec=prec)
        y = dropout(y, k_d1, cfg.p_dropout, deterministic=deterministic)
        x_f32 = layer_norm(x_f32 + jnp.asarray(y, jnp.float32), params["norm_layers_1"][i])

        y = ffn_forward(x_f32, x_mask, params["ffn_layers"][i], ffn_cfg, rng_key=k_ffn, deterministic=deterministic, prec=prec)
        y = dropout(y, k_d2, cfg.p_dropout, deterministic=deterministic)
        x_f32 = layer_norm(x_f32 + jnp.asarray(y, jnp.float32), params["norm_layers_2"][i])

    return x_f32 * jnp.asarray(x_mask, jnp.float32)


@dataclass(frozen=True)
class TextEncoderConfig:
    out_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    embedding_dim: int
    f0: bool = True


def init_text_encoder_params(rng: np.random.Generator, cfg: TextEncoderConfig) -> dict[str, Any]:
    enc_cfg = EncoderConfig(
        hidden_channels=cfg.hidden_channels,
        filter_channels=cfg.filter_channels,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        kernel_size=cfg.kernel_size,
        p_dropout=cfg.p_dropout,
    )
    proj = init_conv1d_params(rng, cfg.hidden_channels, cfg.out_channels * 2, kernel_size=1)
    # Keep initial prior log-std near 0 (sigma ~= 1) so step-0 KL avoids exp(-2 * logs_p) spikes.
    proj["weight"][..., cfg.out_channels :] *= 0.05
    if "bias" in proj:
        proj["bias"][cfg.out_channels :] = 0.0
    params: dict[str, Any] = {
        "emb_phone": init_linear_params(rng, cfg.embedding_dim, cfg.hidden_channels),
        "encoder": init_encoder_params(rng, enc_cfg),
        "proj": proj,
    }
    if cfg.f0:
        params["emb_pitch"] = init_embedding_params(rng, 256, cfg.hidden_channels)
    return params


def text_encoder_forward(
    phone: Array,
    pitch: Array | None,
    lengths: Array,
    params: Mapping[str, Any],
    cfg: TextEncoderConfig,
    skip_head: int | None = None,
    rng_key: Array | None = None,
    deterministic: bool = True,
    prec: PrecisionConfig = PrecisionConfig(),
) -> tuple[Array, Array, Array]:
    """Encode phoneme features and quantized pitch -> (m, logs, x_mask) in NWC (B, T, C)."""
    x = jnp.asarray(linear(phone, params["emb_phone"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32)
    if pitch is not None and cfg.f0 and "emb_pitch" in params:
        x = x + embedding(pitch, params["emb_pitch"]["weight"])
    x = x * math.sqrt(cfg.hidden_channels)
    x = leaky_relu(x, 0.1)

    x_mask = sequence_mask(lengths, x.shape[1])[..., None]
    enc_cfg = EncoderConfig(
        hidden_channels=cfg.hidden_channels,
        filter_channels=cfg.filter_channels,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        kernel_size=cfg.kernel_size,
        p_dropout=cfg.p_dropout,
    )
    x = encoder_forward(x, x_mask, params["encoder"], enc_cfg, rng_key=rng_key, deterministic=deterministic, prec=prec)
    if skip_head is not None and skip_head > 0:
        x = x[:, skip_head:, :]
        x_mask = x_mask[:, skip_head:, :]

    stats = jnp.asarray(
        conv1d(x, params["proj"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32
    ) * x_mask
    m = stats[..., : cfg.out_channels]
    logs = stats[..., cfg.out_channels :]
    return m, logs, x_mask


@dataclass(frozen=True)
class PosteriorEncoderConfig:
    in_channels: int
    out_channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    gin_channels: int = 0


def init_posterior_encoder_params(rng: np.random.Generator, cfg: PosteriorEncoderConfig) -> dict[str, Any]:
    wn_cfg = WaveNetConfig(
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        dilation_rate=cfg.dilation_rate,
        n_layers=cfg.n_layers,
        gin_channels=cfg.gin_channels,
    )
    pre = init_conv1d_params(rng, cfg.in_channels, cfg.hidden_channels, kernel_size=1)
    # Attenuate initial linear-STFT input projection (raw linear spec peaks > 100) without altering forward().
    pre["weight"] *= 0.1
    if "bias" in pre:
        pre["bias"].fill(0.0)
    return {
        "pre": pre,
        "enc": init_wavenet_params(rng, wn_cfg),
        "proj": init_conv1d_params(rng, cfg.hidden_channels, cfg.out_channels * 2, kernel_size=1),
    }


def posterior_encoder_forward(
    x: Array,
    x_lengths: Array,
    params: Mapping[str, Any],
    cfg: PosteriorEncoderConfig,
    rng_key: Array,
    g: Array | None = None,
    prec: PrecisionConfig = PrecisionConfig(),
) -> tuple[Array, Array, Array, Array]:
    """Encode linear spectrogram x:(B, T, F) -> (z, m, logs, x_mask) in NWC (B, T, C)."""
    x_mask = sequence_mask(x_lengths, x.shape[1])[..., None]
    h = jnp.asarray(conv1d(x, params["pre"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32) * x_mask
    wn_cfg = WaveNetConfig(
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        dilation_rate=cfg.dilation_rate,
        n_layers=cfg.n_layers,
        gin_channels=cfg.gin_channels,
    )
    h = wavenet_forward(h, x_mask, params["enc"], wn_cfg, g=g, deterministic=True, prec=prec)
    stats = jnp.asarray(conv1d(h, params["proj"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32) * x_mask
    m = stats[..., : cfg.out_channels]
    logs = stats[..., cfg.out_channels :]
    noise = jax.random.normal(rng_key, shape=m.shape, dtype=jnp.float32)
    z = (m + noise * jnp.exp(logs)) * x_mask
    return z, m, logs, x_mask


def text_encoder_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: TextEncoderConfig) -> dict[str, Any]:
    mha_cfg = MultiHeadAttentionConfig(
        channels=cfg.hidden_channels,
        out_channels=cfg.hidden_channels,
        n_heads=cfg.n_heads,
        p_dropout=cfg.p_dropout,
        window_size=10,
    )
    enc_params = {
        "attn_layers": [mha_from_torch_state(state, f"{prefix}.encoder.attn_layers.{i}", mha_cfg) for i in range(cfg.n_layers)],
        "norm_layers_1": [extract_layer_norm_from_torch_state(state, f"{prefix}.encoder.norm_layers_1.{i}") for i in range(cfg.n_layers)],
        "ffn_layers": [ffn_from_torch_state(state, f"{prefix}.encoder.ffn_layers.{i}") for i in range(cfg.n_layers)],
        "norm_layers_2": [extract_layer_norm_from_torch_state(state, f"{prefix}.encoder.norm_layers_2.{i}") for i in range(cfg.n_layers)],
    }
    params: dict[str, Any] = {
        "emb_phone": extract_linear_from_torch_state(state, f"{prefix}.emb_phone"),
        "encoder": enc_params,
        "proj": extract_conv_from_torch_state(state, f"{prefix}.proj"),
    }
    if cfg.f0 and f"{prefix}.emb_pitch.weight" in state:
        params["emb_pitch"] = {"weight": np.asarray(state[f"{prefix}.emb_pitch.weight"], dtype=np.float32).copy()}
    return params


def text_encoder_to_torch_state(state: dict[str, np.ndarray], prefix: str, params: Mapping[str, Any]) -> None:
    export_linear_to_torch_state(state, f"{prefix}.emb_phone", params["emb_phone"])
    if "emb_pitch" in params:
        state[f"{prefix}.emb_pitch.weight"] = np.asarray(params["emb_pitch"]["weight"], dtype=np.float32).copy()
    enc = params["encoder"]
    for i, layer in enumerate(enc["attn_layers"]):
        mha_to_torch_state(state, f"{prefix}.encoder.attn_layers.{i}", layer)
    for i, layer in enumerate(enc["norm_layers_1"]):
        export_layer_norm_to_torch_state(state, f"{prefix}.encoder.norm_layers_1.{i}", layer)
    for i, layer in enumerate(enc["ffn_layers"]):
        ffn_to_torch_state(state, f"{prefix}.encoder.ffn_layers.{i}", layer)
    for i, layer in enumerate(enc["norm_layers_2"]):
        export_layer_norm_to_torch_state(state, f"{prefix}.encoder.norm_layers_2.{i}", layer)
    export_conv_to_torch_state(state, f"{prefix}.proj", params["proj"])


def posterior_encoder_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: PosteriorEncoderConfig) -> dict[str, Any]:
    wn_cfg = WaveNetConfig(
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        dilation_rate=cfg.dilation_rate,
        n_layers=cfg.n_layers,
        gin_channels=cfg.gin_channels,
    )
    return {
        "pre": extract_conv_from_torch_state(state, f"{prefix}.pre"),
        "enc": wavenet_from_torch_state(state, f"{prefix}.enc", wn_cfg),
        "proj": extract_conv_from_torch_state(state, f"{prefix}.proj"),
    }


def posterior_encoder_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    export_conv_to_torch_state(state, f"{prefix}.pre", params["pre"])
    wavenet_to_torch_state(state, f"{prefix}.enc", params["enc"], remove_weight_norm=remove_weight_norm)
    export_conv_to_torch_state(state, f"{prefix}.proj", params["proj"])
