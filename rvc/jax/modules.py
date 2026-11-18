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
    checkpointing: bool = False

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
    use_dropout = rng_key is not None and not deterministic and cfg.p_dropout > 0.0
    sub_keys = list(jax.random.split(rng_key, cfg.n_layers)) if use_dropout else [None] * cfg.n_layers

    def _run_layer(
        cur_x: Array,
        cur_out: Array,
        layer_idx: int,
        in_p: Mapping[str, Any],
        rs_p: Mapping[str, Any],
        g_slice: Array | None,
        sub_k: Array | None,
    ) -> tuple[Array, Array]:
        dilation = cfg.dilation_rate ** layer_idx
        padding = get_padding(cfg.kernel_size, dilation)
        x_in = conv1d(
            cur_x,
            in_p,
            padding=padding,
            dilation=dilation,
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        acts = fused_add_tanh_sigmoid_multiply(x_in, g_slice, h)
        acts = dropout(acts, sub_k, cfg.p_dropout, deterministic=deterministic)
        res_skip_acts = jnp.asarray(
            conv1d(acts, rs_p, dtype=prec.compute_dtype, precision=prec.compute_precision),
            jnp.float32,
        )
        if layer_idx < cfg.n_layers - 1:
            return (cur_x + res_skip_acts[..., :h]) * mask_f32, cur_out + res_skip_acts[..., h:]
        return cur_x, cur_out + res_skip_acts

    def _g_slice(layer_idx: int) -> Array | None:
        return g_proj[..., layer_idx * 2 * h : (layer_idx + 1) * 2 * h] if g_proj is not None else None

    if cfg.checkpointing and cfg.n_layers > 2:
        chunk_size = 4
        for start in range(0, cfg.n_layers, chunk_size):
            end = min(cfg.n_layers, start + chunk_size)
            in_chunk = params["in_layers"][start:end]
            rs_chunk = params["res_skip_layers"][start:end]
            g_chunk = [_g_slice(idx) for idx in range(start, end)]
            k_chunk = sub_keys[start:end]

            def _run_chunk(
                cur_x: Array,
                cur_out: Array,
                in_c,
                rs_c,
                g_c,
                k_c,
                s: int = start,
            ) -> tuple[Array, Array]:
                for offset in range(len(in_c)):
                    cur_x, cur_out = _run_layer(
                        cur_x, cur_out, s + offset, in_c[offset], rs_c[offset], g_c[offset], k_c[offset]
                    )
                return cur_x, cur_out

            x_f32, output = jax.checkpoint(_run_chunk)(x_f32, output, in_chunk, rs_chunk, g_chunk, k_chunk)
    else:
        for i in range(cfg.n_layers):
            x_f32, output = _run_layer(
                x_f32,
                output,
                i,
                params["in_layers"][i],
                params["res_skip_layers"][i],
                _g_slice(i),
                sub_keys[i],
            )

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
