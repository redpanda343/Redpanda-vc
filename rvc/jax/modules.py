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


def _wavenet_layer(
    x_f32: Array,
    in_params: Mapping[str, Any],
    rs_params: Mapping[str, Any],
    g_l: Array | None,
    padding: int,
    dilation: int,
    hidden_channels: int,
    p_dropout: float,
    rng_key: Array | None,
    deterministic: bool,
    prec: PrecisionConfig,
) -> Array:
    """Execute one WaveNet residual/skip block."""
    x_in = conv1d(
        x_f32,
        in_params,
        padding=padding,
        dilation=dilation,
        dtype=prec.compute_dtype,
        precision=prec.compute_precision,
    )
    g_val = jnp.zeros_like(x_in) if g_l is None else g_l
    acts = fused_add_tanh_sigmoid_multiply(x_in, g_val, hidden_channels)
    acts = dropout(acts, rng_key, p_dropout, deterministic=deterministic)
    return jnp.asarray(
        conv1d(acts, rs_params, dtype=prec.compute_dtype, precision=prec.compute_precision),
        jnp.float32,
    )


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
    layer_keys = jax.random.split(rng_key, cfg.n_layers) if use_dropout else None

    if cfg.dilation_rate == 1 and cfg.n_layers > 2:
        n_body = cfg.n_layers - 1
        pad = get_padding(cfg.kernel_size, 1)
        g_body = (
            g_proj[..., : n_body * 2 * h].reshape(g_proj.shape[0], 1, n_body, 2 * h).transpose(2, 0, 1, 3)
            if g_proj is not None
            else None
        )
        stacked_in = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params["in_layers"][:n_body])
        stacked_rs = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params["res_skip_layers"][:n_body])
        keys_body = layer_keys[:n_body] if layer_keys is not None else None

        def _scan_body(carry: tuple[Array, Array], item: tuple[Any, Any, Array | None, Array | None]):
            cur_x, cur_out = carry
            in_p, rs_p, g_l, sub_k = item
            rs_acts = _wavenet_layer(
                cur_x, in_p, rs_p, g_l, pad, 1, h, cfg.p_dropout, sub_k, deterministic, prec
            )
            next_x = (cur_x + rs_acts[..., :h]) * mask_f32
            next_out = cur_out + rs_acts[..., h:]
            return (next_x, next_out), None

        (x_f32, output), _ = jax.lax.scan(
            _scan_body, (x_f32, output), (stacked_in, stacked_rs, g_body, keys_body)
        )
        last = cfg.n_layers - 1
        g_last = g_proj[..., last * 2 * h : (last + 1) * 2 * h] if g_proj is not None else None
        k_last = layer_keys[last] if layer_keys is not None else None
        rs_last = _wavenet_layer(
            x_f32,
            params["in_layers"][last],
            params["res_skip_layers"][last],
            g_last,
            pad,
            1,
            h,
            cfg.p_dropout,
            k_last,
            deterministic,
            prec,
        )
        return (output + rs_last) * mask_f32

    for i in range(cfg.n_layers):
        dilation = cfg.dilation_rate ** i
        padding = get_padding(cfg.kernel_size, dilation)
        g_l = g_proj[..., i * 2 * h : (i + 1) * 2 * h] if g_proj is not None else None
        sub_k = layer_keys[i] if layer_keys is not None else None
        res_skip_acts = _wavenet_layer(
            x_f32,
            params["in_layers"][i],
            params["res_skip_layers"][i],
            g_l,
            padding,
            dilation,
            h,
            cfg.p_dropout,
            sub_k,
            deterministic,
            prec,
        )
        if i < cfg.n_layers - 1:
            x_f32 = (x_f32 + res_skip_acts[..., :h]) * mask_f32
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
