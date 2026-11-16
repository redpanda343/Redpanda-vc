"""JAX NWC WaveNet residual module matching rvc.algorithm.modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from rvc.jax import PrecisionConfig
from rvc.jax.ops import (
    Array,
    conv1d,
    dropout,
    export_conv_to_torch_state,
    extract_conv_from_torch_state,
    fused_add_tanh_sigmoid_multiply,
    get_padding,
    init_conv1d_params,
)


@dataclass(frozen=True)
class WaveNetConfig:
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    gin_channels: int = 0
    p_dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.kernel_size % 2 != 1:
            raise ValueError(f"Kernel size must be odd for proper padding (got {self.kernel_size}).")


def init_wavenet_params(rng: np.random.Generator, cfg: WaveNetConfig) -> dict[str, Any]:
    """Initialize WaveNet parameters matching PyTorch Conv1d + weight_norm."""
    params: dict[str, Any] = {}
    if cfg.gin_channels:
        params["cond_layer"] = init_conv1d_params(
            rng, cfg.gin_channels, 2 * cfg.hidden_channels * cfg.n_layers, kernel_size=1, weight_norm=True
        )
    in_layers: list[dict[str, np.ndarray]] = []
    res_skip_layers: list[dict[str, np.ndarray]] = []
    for i in range(cfg.n_layers):
        in_layers.append(
            init_conv1d_params(rng, cfg.hidden_channels, 2 * cfg.hidden_channels, cfg.kernel_size, weight_norm=True)
        )
        res_skip_ch = cfg.hidden_channels if i == cfg.n_layers - 1 else 2 * cfg.hidden_channels
        res_skip_layers.append(init_conv1d_params(rng, cfg.hidden_channels, res_skip_ch, kernel_size=1, weight_norm=True))
    params["in_layers"] = in_layers
    params["res_skip_layers"] = res_skip_layers
    return params


def wavenet_forward(
    x: Array,
    x_mask: Array,
    params: Mapping[str, Any],
    cfg: WaveNetConfig,
    g: Array | None = None,
    rng_key: Array | None = None,
    deterministic: bool = True,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Apply WaveNet blocks on NWC tensors x:(B, T, H), x_mask:(B, T, 1), g:(B, 1, C_g)."""
    x_f32 = jnp.asarray(x, jnp.float32)
    mask_f32 = jnp.asarray(x_mask, jnp.float32)
    output = jnp.zeros_like(x_f32)

    g_proj = (
        conv1d(g, params["cond_layer"], dtype=prec.compute_dtype, precision=prec.compute_precision)
        if g is not None and cfg.gin_channels
        else None
    )

    h = cfg.hidden_channels
    for i in range(cfg.n_layers):
        dilation = cfg.dilation_rate ** i
        padding = get_padding(cfg.kernel_size, dilation)
        x_in = conv1d(
            x_f32,
            params["in_layers"][i],
            padding=padding,
            dilation=dilation,
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        if g_proj is not None:
            g_l = g_proj[..., i * 2 * h : (i + 1) * 2 * h]
        else:
            g_l = jnp.zeros_like(x_in)

        acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, h)
        if rng_key is not None and not deterministic and cfg.p_dropout > 0.0:
            rng_key, sub_key = jax.random.split(rng_key)
        else:
            sub_key = None
        acts = dropout(acts, sub_key, cfg.p_dropout, deterministic=deterministic)

        res_skip_acts = jnp.asarray(
            conv1d(acts, params["res_skip_layers"][i], dtype=prec.compute_dtype, precision=prec.compute_precision),
            jnp.float32,
        )
        if i < cfg.n_layers - 1:
            res_acts = res_skip_acts[..., :h]
            x_f32 = (x_f32 + res_acts) * mask_f32
            output = output + res_skip_acts[..., h:]
        else:
            output = output + res_skip_acts

    return output * mask_f32


def wavenet_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: WaveNetConfig) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if cfg.gin_channels:
        params["cond_layer"] = extract_conv_from_torch_state(state, f"{prefix}.cond_layer")
    params["in_layers"] = [extract_conv_from_torch_state(state, f"{prefix}.in_layers.{i}") for i in range(cfg.n_layers)]
    params["res_skip_layers"] = [
        extract_conv_from_torch_state(state, f"{prefix}.res_skip_layers.{i}") for i in range(cfg.n_layers)
    ]
    return params


def wavenet_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    if "cond_layer" in params:
        export_conv_to_torch_state(
            state, f"{prefix}.cond_layer", params["cond_layer"], remove_weight_norm=remove_weight_norm
        )
    for i, layer in enumerate(params["in_layers"]):
        export_conv_to_torch_state(
            state, f"{prefix}.in_layers.{i}", layer, remove_weight_norm=remove_weight_norm
        )
    for i, layer in enumerate(params["res_skip_layers"]):
        export_conv_to_torch_state(
            state, f"{prefix}.res_skip_layers.{i}", layer, remove_weight_norm=remove_weight_norm
        )
