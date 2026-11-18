"""JAX NWC ResBlock and normalizing-flow ResidualCouplingBlock matching rvc.algorithm.residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rvc.jax import PrecisionConfig
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
    export_conv_to_torch_state,
    extract_conv_from_torch_state,
    get_padding,
    init_conv1d_params,
    leaky_relu,
)

LRELU_SLOPE = 0.1


def init_resblock_params(
    rng: np.random.Generator,
    channels: int,
    kernel_size: int = 3,
    dilations: Sequence[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Initialize ResBlock weight_norm conv1d parameters (init_std=0.01 matching init_weights)."""
    convs1 = [
        init_conv1d_params(rng, channels, channels, kernel_size, weight_norm=True, init_std=0.01)
        for _ in dilations
    ]
    convs2 = [
        init_conv1d_params(rng, channels, channels, kernel_size, weight_norm=True, init_std=0.01)
        for _ in dilations
    ]
    return {"convs1": convs1, "convs2": convs2}


def resblock_forward(
    x: Array,
    params: Mapping[str, Any],
    kernel_size: int = 3,
    dilations: Sequence[int] = (1, 3, 5),
    x_mask: Array | None = None,
    leaky_relu_slope: float = LRELU_SLOPE,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Apply ResBlock on NWC input (B, T, C)."""
    cur = jnp.asarray(x, prec.compute_dtype)
    mask = jnp.asarray(x_mask, prec.compute_dtype) if x_mask is not None else None
    for i, d in enumerate(dilations):
        x_res = cur
        h = leaky_relu(cur, leaky_relu_slope)
        if mask is not None:
            h = h * mask
        h = conv1d(
            h,
            params["convs1"][i],
            padding=get_padding(kernel_size, int(d)),
            dilation=int(d),
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        h = leaky_relu(h, leaky_relu_slope)
        if mask is not None:
            h = h * mask
        h = conv1d(
            h,
            params["convs2"][i],
            padding=get_padding(kernel_size, 1),
            dilation=1,
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        cur = h + x_res
    if mask is not None:
        cur = cur * mask
    return cur


def resblock_from_torch_state(state: Mapping[str, Any], prefix: str, n_dilations: int) -> dict[str, Any]:
    return {
        "convs1": [extract_conv_from_torch_state(state, f"{prefix}.convs1.{i}") for i in range(n_dilations)],
        "convs2": [extract_conv_from_torch_state(state, f"{prefix}.convs2.{i}") for i in range(n_dilations)],
    }


def resblock_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    for i, layer in enumerate(params["convs1"]):
        export_conv_to_torch_state(state, f"{prefix}.convs1.{i}", layer, remove_weight_norm=remove_weight_norm)
    for i, layer in enumerate(params["convs2"]):
        export_conv_to_torch_state(state, f"{prefix}.convs2.{i}", layer, remove_weight_norm=remove_weight_norm)


@dataclass(frozen=True)
class ResidualCouplingBlockConfig:
    channels: int
    hidden_channels: int
    kernel_size: int
    dilation_rate: int
    n_layers: int
    n_flows: int = 4
    gin_channels: int = 0
    mean_only: bool = True
    p_dropout: float = 0.0
    checkpointing: bool = False


def init_flow_params(rng: np.random.Generator, cfg: ResidualCouplingBlockConfig) -> dict[str, Any]:
    """Initialize ResidualCouplingBlock using keys matching PyTorch flow.flows.{0,2,4,6}."""
    half = cfg.channels // 2
    wn_cfg = WaveNetConfig(
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        dilation_rate=cfg.dilation_rate,
        n_layers=cfg.n_layers,
        gin_channels=cfg.gin_channels,
        p_dropout=cfg.p_dropout,
    )
    flows: dict[str, Any] = {}
    for i in range(cfg.n_flows):
        flows[str(i * 2)] = {
            "pre": init_conv1d_params(rng, half, cfg.hidden_channels, kernel_size=1),
            "enc": init_wavenet_params(rng, wn_cfg),
            "post": init_conv1d_params(rng, cfg.hidden_channels, half * (2 - int(cfg.mean_only)), kernel_size=1, zero_init=True),
        }
    return {"flows": flows}


def _coupling_layer_forward(
    x: Array,
    x_mask: Array,
    params: Mapping[str, Any],
    cfg: ResidualCouplingBlockConfig,
    g: Array | None = None,
    reverse: bool = False,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    half = cfg.channels // 2
    x_f32 = jnp.asarray(x, jnp.float32)
    mask_f32 = jnp.asarray(x_mask, jnp.float32)
    x0 = x_f32[..., :half]
    x1 = x_f32[..., half:]

    h = jnp.asarray(conv1d(x0, params["pre"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32) * mask_f32
    wn_cfg = WaveNetConfig(
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        dilation_rate=cfg.dilation_rate,
        n_layers=cfg.n_layers,
        gin_channels=cfg.gin_channels,
        p_dropout=cfg.p_dropout,
    )
    h = wavenet_forward(h, mask_f32, params["enc"], wn_cfg, g=g, deterministic=True, prec=prec)
    stats = jnp.asarray(conv1d(h, params["post"], dtype=prec.compute_dtype, precision=prec.compute_precision), jnp.float32) * mask_f32

    if cfg.mean_only:
        if not reverse:
            return jnp.concatenate([x0, stats + x1 * mask_f32], axis=-1)
        return jnp.concatenate([x0, (x1 - stats) * mask_f32], axis=-1)

    m = stats[..., :half]
    logs = stats[..., half:]
    if not reverse:
        x1_new = m + x1 * jnp.exp(logs) * mask_f32
        return jnp.concatenate([x0, x1_new], axis=-1)
    x1_new = (x1 - m) * jnp.exp(-logs) * mask_f32
    return jnp.concatenate([x0, x1_new], axis=-1)


def flow_forward(
    x: Array,
    x_mask: Array,
    params: Mapping[str, Any],
    cfg: ResidualCouplingBlockConfig,
    g: Array | None = None,
    reverse: bool = False,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """Apply ResidualCouplingBlock forward or reverse (Flip happens on channels axis -1 in NWC)."""
    cur = jnp.asarray(x, jnp.float32)
    flows = params["flows"]

    def _run_flow(c: Array, flow_p: Mapping[str, Any]) -> Array:
        if reverse:
            return _coupling_layer_forward(c[..., ::-1], x_mask, flow_p, cfg, g=g, reverse=True, prec=prec)
        return _coupling_layer_forward(c, x_mask, flow_p, cfg, g=g, reverse=False, prec=prec)[..., ::-1]

    indices = range(cfg.n_flows - 1, -1, -1) if reverse else range(cfg.n_flows)
    for i in indices:
        flow_p = flows[str(i * 2)]
        cur = jax.checkpoint(_run_flow)(cur, flow_p) if cfg.checkpointing else _run_flow(cur, flow_p)
    return cur


def flow_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: ResidualCouplingBlockConfig) -> dict[str, Any]:
    wn_cfg = WaveNetConfig(
        hidden_channels=cfg.hidden_channels,
        kernel_size=cfg.kernel_size,
        dilation_rate=cfg.dilation_rate,
        n_layers=cfg.n_layers,
        gin_channels=cfg.gin_channels,
        p_dropout=cfg.p_dropout,
    )
    flows: dict[str, Any] = {}
    for i in range(cfg.n_flows):
        idx = str(i * 2)
        p_i = f"{prefix}.flows.{idx}"
        flows[idx] = {
            "pre": extract_conv_from_torch_state(state, f"{p_i}.pre"),
            "enc": wavenet_from_torch_state(state, f"{p_i}.enc", wn_cfg),
            "post": extract_conv_from_torch_state(state, f"{p_i}.post"),
        }
    return {"flows": flows}


def flow_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    for idx, layer in params["flows"].items():
        p_i = f"{prefix}.flows.{idx}"
        export_conv_to_torch_state(state, f"{p_i}.pre", layer["pre"])
        wavenet_to_torch_state(state, f"{p_i}.enc", layer["enc"], remove_weight_norm=remove_weight_norm)
        export_conv_to_torch_state(state, f"{p_i}.post", layer["post"])
