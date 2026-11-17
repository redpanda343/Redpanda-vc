"""JAX NWC vocoder generators: HiFiGAN, HiFiGANNSF, HiFiGANMRF, and RefineGAN."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rvc.jax import PrecisionConfig
from rvc.jax.ops import (
    Array,
    conv1d,
    conv_transpose1d,
    export_conv_to_torch_state,
    export_linear_to_torch_state,
    extract_conv_from_torch_state,
    extract_linear_from_torch_state,
    get_padding,
    init_conv1d_params,
    init_conv_transpose1d_params,
    init_linear_params,
    kaiser_sinc_resample_1d,
    leaky_relu,
    linear,
    upsample_linear_1d,
    upsample_nearest_1d,
)
from rvc.jax.residuals import (
    LRELU_SLOPE,
    init_resblock_params,
    resblock_forward,
    resblock_from_torch_state,
    resblock_to_torch_state,
)


def _noise_conv_shape(stride: int) -> tuple[int, int]:
    kernel = 1 if stride == 1 else stride * 2 - stride % 2
    padding = 0 if stride == 1 else (kernel - stride) // 2
    return kernel, padding


# ---------------------------------------------------------------------------
# Harmonic / Noise Excitation Modules (strictly FP32 utilities)
# ---------------------------------------------------------------------------


def _sine_generator_hifigan_forward(
    f0: Array,
    sampling_rate: int,
    upsampling_factor: int,
    rng_key: Array | None,
    num_harmonics: int = 0,
    sine_amplitude: float = 0.1,
    noise_stddev: float = 0.003,
    voiced_threshold: float = 0.0,
    deterministic: bool = False,
) -> Array:
    """Strictly FP32 excitation matching rvc.algorithm.generators.hifigan.SineGenerator."""
    f0_f32 = jnp.asarray(f0, jnp.float32)
    batch_size = f0_f32.shape[0]
    waveform_dim = num_harmonics + 1

    up_grid = jnp.arange(1, upsampling_factor + 1, dtype=jnp.float32)[None, None, :]
    phase_inc = (f0_f32[..., None] / float(sampling_rate)) * up_grid
    phase_rem = jnp.fmod(phase_inc[:, :-1, -1:] + 0.5, 1.0) - 0.5
    cum_phase = jnp.fmod(jnp.cumsum(phase_rem, axis=1), 1.0)
    phase_inc = phase_inc + jnp.pad(cum_phase, ((0, 0), (1, 0), (0, 0)))
    phase_inc = phase_inc.reshape(batch_size, -1, 1)

    harm_scale = jnp.arange(1, waveform_dim + 1, dtype=jnp.float32).reshape(1, 1, -1)
    phase_inc = phase_inc * harm_scale

    if rng_key is not None and not deterministic:
        k_phase, k_noise = jax.random.split(rng_key)
        rand_phase = jax.random.uniform(k_phase, shape=(1, 1, waveform_dim), dtype=jnp.float32)
        rand_phase = rand_phase.at[..., 0].set(0.0)
        phase_inc = phase_inc + rand_phase
        noise_draw = jax.random.normal(k_noise, shape=phase_inc.shape, dtype=jnp.float32)
    else:
        noise_draw = jnp.zeros_like(phase_inc)

    sine_waves = jnp.sin(2.0 * math.pi * phase_inc) * float(sine_amplitude)
    uv = (f0_f32[..., None] > float(voiced_threshold)).astype(jnp.float32)
    voiced_mask = upsample_nearest_1d(uv, upsampling_factor)

    noise_amp = voiced_mask * float(noise_stddev) + (1.0 - voiced_mask) * (float(sine_amplitude) / 3.0)
    sine_waveforms = sine_waves * voiced_mask + noise_amp * noise_draw
    return jax.lax.stop_gradient(sine_waveforms)


def _sine_generator_mrf_forward(
    f0_up: Array,
    sampling_rate: int,
    rng_key: Array | None,
    harmonic_num: int = 8,
    sine_amp: float = 0.1,
    noise_std: float = 0.003,
    voiced_threshold: float = 0.0,
    deterministic: bool = False,
) -> Array:
    """Strictly FP32 excitation matching hifigan_mrf.SineGenerator & refinegan.SineGenerator on upsampled f0:(B,T,1)."""
    f0_f32 = jnp.asarray(f0_up, jnp.float32)
    batch_size, length, _ = f0_f32.shape
    dim = harmonic_num + 1

    harmonics = jnp.arange(1, dim + 1, dtype=jnp.float32)[None, None, :]
    f0_values = f0_f32 * harmonics

    rad_values = jnp.mod(f0_values / float(sampling_rate), 1.0)
    if rng_key is not None and not deterministic:
        k_phase, k_noise = jax.random.split(rng_key)
        rand_ini = jax.random.uniform(k_phase, shape=(batch_size, dim), dtype=jnp.float32)
        rand_ini = rand_ini.at[:, 0].set(0.0)
        rad_values = rad_values.at[:, 0, :].add(rand_ini)
        noise_draw = jax.random.normal(k_noise, shape=f0_values.shape, dtype=jnp.float32)
    else:
        noise_draw = jnp.zeros_like(f0_values)

    tmp_over_one = jnp.mod(jnp.cumsum(rad_values, axis=1), 1.0)
    tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0.0
    cumsum_shift = jnp.zeros_like(rad_values)
    cumsum_shift = cumsum_shift.at[:, 1:, :].set(tmp_over_one_idx.astype(jnp.float32) * -1.0)
    sines = jnp.sin(jnp.cumsum(rad_values + cumsum_shift, axis=1) * (2.0 * math.pi)) * float(sine_amp)

    uv = (f0_f32 > float(voiced_threshold)).astype(jnp.float32)
    noise_amp = uv * float(noise_std) + (1.0 - uv) * (float(sine_amp) / 3.0)
    sine_waves = sines * uv + noise_amp * noise_draw
    return jax.lax.stop_gradient(sine_waves)


# ---------------------------------------------------------------------------
# HiFiGANGenerator (no F0)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiFiGANConfig:
    initial_channel: int
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_rates: tuple[int, ...]
    upsample_initial_channel: int
    upsample_kernel_sizes: tuple[int, ...]
    gin_channels: int = 0
    checkpointing: bool = False


def init_hifigan_params(rng: np.random.Generator, cfg: HiFiGANConfig) -> dict[str, Any]:
    ups: list[dict[str, np.ndarray]] = []
    resblocks: list[dict[str, Any]] = []
    ch = cfg.upsample_initial_channel
    for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
        in_ch = cfg.upsample_initial_channel // (2 ** i)
        ch = cfg.upsample_initial_channel // (2 ** (i + 1))
        ups.append(init_conv_transpose1d_params(rng, in_ch, ch, k, weight_norm=True, init_std=0.01))
        for rk, rd in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
            resblocks.append(init_resblock_params(rng, ch, rk, rd))

    params: dict[str, Any] = {
        "conv_pre": init_conv1d_params(rng, cfg.initial_channel, cfg.upsample_initial_channel, 7),
        "ups": ups,
        "resblocks": resblocks,
        "conv_post": init_conv1d_params(rng, ch, 1, 7, bias=False),
    }
    if cfg.gin_channels != 0:
        params["cond"] = init_conv1d_params(rng, cfg.gin_channels, cfg.upsample_initial_channel, 1)
    return params


def hifigan_forward(
    x: Array,
    params: Mapping[str, Any],
    cfg: HiFiGANConfig,
    g: Array | None = None,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """HiFiGAN forward on NWC input (B, T, C) -> FP32 waveform (B, T * upp, 1)."""
    num_kernels = len(cfg.resblock_kernel_sizes)
    h = conv1d(x, params["conv_pre"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    if g is not None and "cond" in params:
        h = h + conv1d(g, params["cond"], dtype=prec.compute_dtype, precision=prec.compute_precision)

    for i, (u, uk) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
        p_i = params["ups"][i]
        rbs_i = params["resblocks"][i * num_kernels : (i + 1) * num_kernels]

        def _stage(cur_h: Array, up_p: Mapping[str, Any], rb_ps: list[Mapping[str, Any]]) -> Array:
            cur_h = leaky_relu(cur_h, LRELU_SLOPE)
            cur_h = conv_transpose1d(
                cur_h,
                up_p,
                stride=u,
                padding=(uk - u) // 2,
                dtype=prec.compute_dtype,
                precision=prec.compute_precision,
            )
            acc = None
            for j, (rk, rd) in enumerate(zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes)):
                out_j = resblock_forward(cur_h, rb_ps[j], kernel_size=rk, dilations=rd, prec=prec)
                acc = out_j if acc is None else (acc + out_j)
            return acc / float(num_kernels)

        h = jax.checkpoint(_stage)(h, p_i, rbs_i) if cfg.checkpointing else _stage(h, p_i, rbs_i)

    h = leaky_relu(h, 0.01)
    h = conv1d(h, params["conv_post"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    return jnp.tanh(jnp.asarray(h, jnp.float32))


# ---------------------------------------------------------------------------
# HiFiGANNSFGenerator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiFiGANNSFConfig:
    initial_channel: int
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_rates: tuple[int, ...]
    upsample_initial_channel: int
    upsample_kernel_sizes: tuple[int, ...]
    gin_channels: int
    sr: int
    checkpointing: bool = False


def init_hifigan_nsf_params(rng: np.random.Generator, cfg: HiFiGANNSFConfig) -> dict[str, Any]:
    ups: list[dict[str, np.ndarray]] = []
    noise_convs: list[dict[str, np.ndarray]] = []
    resblocks: list[dict[str, Any]] = []
    stride_f0s = [
        math.prod(cfg.upsample_rates[i + 1 :]) if i + 1 < len(cfg.upsample_rates) else 1
        for i in range(len(cfg.upsample_rates))
    ]
    ch = cfg.upsample_initial_channel
    for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
        in_ch = cfg.upsample_initial_channel // (2 ** i)
        ch = cfg.upsample_initial_channel // (2 ** (i + 1))
        ups.append(init_conv_transpose1d_params(rng, in_ch, ch, k, weight_norm=True, init_std=0.01))
        nk, _ = _noise_conv_shape(stride_f0s[i])
        noise_convs.append(init_conv1d_params(rng, 1, ch, nk, init_std=0.01))
        for rk, rd in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes):
            resblocks.append(init_resblock_params(rng, ch, rk, rd))

    params: dict[str, Any] = {
        "m_source": {"l_linear": init_linear_params(rng, 1, 1, bias=True)},
        "conv_pre": init_conv1d_params(rng, cfg.initial_channel, cfg.upsample_initial_channel, 7),
        "ups": ups,
        "noise_convs": noise_convs,
        "resblocks": resblocks,
        "conv_post": init_conv1d_params(rng, ch, 1, 7, bias=False),
    }
    if cfg.gin_channels != 0:
        params["cond"] = init_conv1d_params(rng, cfg.gin_channels, cfg.upsample_initial_channel, 1)
    return params


def hifigan_nsf_forward(
    x: Array,
    f0: Array,
    params: Mapping[str, Any],
    cfg: HiFiGANNSFConfig,
    rng_key: Array | None = None,
    g: Array | None = None,
    deterministic: bool = False,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """HiFiGAN-NSF forward on NWC input (B, T, C) + f0:(B, T) -> FP32 waveform (B, T*upp, 1)."""
    upp = math.prod(cfg.upsample_rates)
    sine_waves = _sine_generator_hifigan_forward(f0, cfg.sr, upp, rng_key, deterministic=deterministic)
    har_source = jnp.tanh(linear(sine_waves, params["m_source"]["l_linear"], dtype=jnp.float32, precision=prec.fp32_precision))

    num_kernels = len(cfg.resblock_kernel_sizes)
    stride_f0s = [
        math.prod(cfg.upsample_rates[i + 1 :]) if i + 1 < len(cfg.upsample_rates) else 1
        for i in range(len(cfg.upsample_rates))
    ]

    h = conv1d(x, params["conv_pre"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    if g is not None and "cond" in params:
        h = h + conv1d(g, params["cond"], dtype=prec.compute_dtype, precision=prec.compute_precision)

    for i, (u, uk) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
        up_pad = (uk - u) // 2 if u % 2 == 0 else (u // 2 + u % 2)
        up_out_pad = u % 2
        n_stride = stride_f0s[i]
        n_kernel, n_pad = _noise_conv_shape(n_stride)

        def _stage(
            cur_h: Array,
            up_p: Mapping[str, Any],
            nc_p: Mapping[str, Any],
            rb_ps: list[Mapping[str, Any]],
        ) -> Array:
            cur_h = leaky_relu(cur_h, LRELU_SLOPE)
            cur_h = conv_transpose1d(
                cur_h,
                up_p,
                stride=u,
                padding=up_pad,
                output_padding=up_out_pad,
                dtype=prec.compute_dtype,
                precision=prec.compute_precision,
            )
            cur_h = cur_h + conv1d(
                har_source,
                nc_p,
                stride=n_stride,
                padding=n_pad,
                dtype=prec.compute_dtype,
                precision=prec.compute_precision,
            )
            acc = None
            for j, (rk, rd) in enumerate(zip(cfg.resblock_kernel_sizes, cfg.resblock_dilation_sizes)):
                out_j = resblock_forward(cur_h, rb_ps[j], kernel_size=rk, dilations=rd, prec=prec)
                acc = out_j if acc is None else (acc + out_j)
            return acc / float(num_kernels)

        p_up = params["ups"][i]
        p_nc = params["noise_convs"][i]
        p_rbs = params["resblocks"][i * num_kernels : (i + 1) * num_kernels]
        h = jax.checkpoint(_stage)(h, p_up, p_nc, p_rbs) if cfg.checkpointing else _stage(h, p_up, p_nc, p_rbs)

    h = leaky_relu(h, 0.01)
    # Remove leaky_relu DC offset in FP32 so conv_post (bias=False) does not coherently amplify temporal DC gradients.
    h_f32 = jnp.asarray(h, jnp.float32)
    h = h_f32 - jnp.mean(h_f32, axis=1, keepdims=True)
    h = conv1d(h, params["conv_post"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    return jnp.tanh(jnp.asarray(h, jnp.float32))


# ---------------------------------------------------------------------------
# HiFiGANMRFGenerator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HiFiGANMRFConfig:
    in_channel: int
    upsample_initial_channel: int
    upsample_rates: tuple[int, ...]
    upsample_kernel_sizes: tuple[int, ...]
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilations: tuple[tuple[int, ...], ...]
    gin_channels: int
    sample_rate: int
    harmonic_num: int = 8
    checkpointing: bool = False


def init_hifigan_mrf_params(rng: np.random.Generator, cfg: HiFiGANMRFConfig) -> dict[str, Any]:
    upsamples: list[dict[str, np.ndarray]] = []
    noise_convs: list[dict[str, np.ndarray]] = []
    mrfs: list[list[dict[str, Any]]] = []

    stride_f0s = [
        math.prod(cfg.upsample_rates[i + 1 :]) if i + 1 < len(cfg.upsample_rates) else 1
        for i in range(len(cfg.upsample_rates))
    ]
    ch = cfg.upsample_initial_channel
    for i, (u, k) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
        in_ch = cfg.upsample_initial_channel // (2 ** i)
        ch = cfg.upsample_initial_channel // (2 ** (i + 1))
        upsamples.append(init_conv_transpose1d_params(rng, in_ch, ch, k, weight_norm=True))
        nk, _ = _noise_conv_shape(stride_f0s[i])
        noise_convs.append(init_conv1d_params(rng, 1, ch, nk))

        stage_mrfs: list[dict[str, Any]] = []
        for rk, rds in zip(cfg.resblock_kernel_sizes, cfg.resblock_dilations):
            layers = [
                {
                    "conv1": init_conv1d_params(rng, ch, ch, rk, weight_norm=True),
                    "conv2": init_conv1d_params(rng, ch, ch, rk, weight_norm=True),
                }
                for _ in rds
            ]
            stage_mrfs.append({"layers": layers})
        mrfs.append(stage_mrfs)

    params: dict[str, Any] = {
        "m_source": {"l_linear": init_linear_params(rng, cfg.harmonic_num + 1, 1, bias=True)},
        "conv_pre": init_conv1d_params(rng, cfg.in_channel, cfg.upsample_initial_channel, 7, weight_norm=True),
        "upsamples": upsamples,
        "noise_convs": noise_convs,
        "mrfs": mrfs,
        "conv_post": init_conv1d_params(rng, ch, 1, 7, bias=True, weight_norm=True),
    }
    if cfg.gin_channels != 0:
        params["cond"] = init_conv1d_params(rng, cfg.gin_channels, cfg.upsample_initial_channel, 1)
    return params


def _mrf_block_forward(
    x: Array,
    params: Mapping[str, Any],
    kernel_size: int,
    dilations: Sequence[int],
    prec: PrecisionConfig,
) -> Array:
    cur = jnp.asarray(x, prec.compute_dtype)
    for l_idx, d in enumerate(dilations):
        layer_p = params["layers"][l_idx]
        y = leaky_relu(cur, LRELU_SLOPE)
        y = conv1d(
            y,
            layer_p["conv1"],
            padding=(kernel_size * int(d) - int(d)) // 2,
            dilation=int(d),
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        y = leaky_relu(y, LRELU_SLOPE)
        y = conv1d(
            y,
            layer_p["conv2"],
            padding=kernel_size // 2,
            dilation=1,
            dtype=prec.compute_dtype,
            precision=prec.compute_precision,
        )
        cur = cur + y
    return cur


def hifigan_mrf_forward(
    x: Array,
    f0: Array,
    params: Mapping[str, Any],
    cfg: HiFiGANMRFConfig,
    rng_key: Array | None = None,
    g: Array | None = None,
    deterministic: bool = False,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """HiFiGAN-MRF forward on NWC input (B, T, C) + f0:(B, T) -> FP32 waveform (B, T*upp, 1)."""
    upp = math.prod(cfg.upsample_rates)
    f0_up = upsample_nearest_1d(jnp.asarray(f0, jnp.float32)[..., None], upp)
    sine_waves = _sine_generator_mrf_forward(
        f0_up,
        cfg.sample_rate,
        rng_key,
        harmonic_num=cfg.harmonic_num,
        deterministic=deterministic,
    )
    har_source = jnp.tanh(linear(sine_waves, params["m_source"]["l_linear"], dtype=jnp.float32, precision=prec.fp32_precision))

    num_kernels = len(cfg.resblock_kernel_sizes)
    stride_f0s = [
        math.prod(cfg.upsample_rates[i + 1 :]) if i + 1 < len(cfg.upsample_rates) else 1
        for i in range(len(cfg.upsample_rates))
    ]

    h = conv1d(x, params["conv_pre"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    if g is not None and "cond" in params:
        h = h + conv1d(g, params["cond"], dtype=prec.compute_dtype, precision=prec.compute_precision)

    for i, (u, uk) in enumerate(zip(cfg.upsample_rates, cfg.upsample_kernel_sizes)):
        up_pad = (uk - u) // 2 if u % 2 == 0 else (u // 2 + u % 2)
        up_out_pad = u % 2
        n_stride = stride_f0s[i]
        n_kernel, n_pad = _noise_conv_shape(n_stride)

        def _stage(
            cur_h: Array,
            up_p: Mapping[str, Any],
            nc_p: Mapping[str, Any],
            mrf_ps: list[Mapping[str, Any]],
        ) -> Array:
            cur_h = leaky_relu(cur_h, LRELU_SLOPE)
            cur_h = conv_transpose1d(
                cur_h,
                up_p,
                stride=u,
                padding=up_pad,
                output_padding=up_out_pad,
                dtype=prec.compute_dtype,
                precision=prec.compute_precision,
            )
            cur_h = cur_h + conv1d(
                har_source,
                nc_p,
                stride=n_stride,
                padding=n_pad,
                dtype=prec.compute_dtype,
                precision=prec.compute_precision,
            )
            acc = None
            for j, (rk, rds) in enumerate(zip(cfg.resblock_kernel_sizes, cfg.resblock_dilations)):
                out_j = _mrf_block_forward(cur_h, mrf_ps[j], rk, rds, prec)
                acc = out_j if acc is None else (acc + out_j)
            return acc / float(num_kernels)

        p_up = params["upsamples"][i]
        p_nc = params["noise_convs"][i]
        p_mrfs = params["mrfs"][i]
        h = jax.checkpoint(_stage)(h, p_up, p_nc, p_mrfs) if cfg.checkpointing else _stage(h, p_up, p_nc, p_mrfs)

    h = leaky_relu(h, 0.01)
    h = conv1d(h, params["conv_post"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    return jnp.tanh(jnp.asarray(h, jnp.float32))


# ---------------------------------------------------------------------------
# RefineGANGenerator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RefineGANConfig:
    sample_rate: int
    upsample_rates: tuple[int, ...]
    leaky_relu_slope: float = 0.2
    num_mels: int = 192
    start_channels: int = 16
    gin_channels: int = 256
    upsample_initial_channel: int = 512
    checkpointing: bool = False


def init_refinegan_params(rng: np.random.Generator, cfg: RefineGANConfig) -> dict[str, Any]:
    downsample_blocks: list[dict[str, np.ndarray]] = []
    ch = cfg.start_channels
    for _ in cfg.upsample_rates:
        new_ch = ch * 2
        downsample_blocks.append(init_conv1d_params(rng, ch, new_ch, 7, weight_norm=True))
        ch = new_ch

    up_ch = cfg.upsample_initial_channel
    upsample_conv_blocks: list[dict[str, Any]] = []
    kernel_sizes = (3, 7, 11)
    dilations = (1, 3, 5)
    for _ in cfg.upsample_rates:
        new_ch = up_ch // 2
        blocks = []
        for ks in kernel_sizes:
            blocks.append(
                {
                    "adain1": {"weight": np.full((new_ch,), 1e-4, dtype=np.float32)},
                    "resblock": init_resblock_params(rng, new_ch, ks, dilations),
                    "adain2": {"weight": np.full((new_ch,), 1e-4, dtype=np.float32)},
                }
            )
        upsample_conv_blocks.append(
            {
                "input_conv": init_conv1d_params(rng, up_ch + up_ch // 4, new_ch, 7, init_std=0.01),
                "blocks": blocks,
            }
        )
        up_ch = new_ch

    params: dict[str, Any] = {
        "m_source": {"merge": {"weight": init_linear_params(rng, 1, 1, bias=False)["weight"]}},
        "pre_conv": init_conv1d_params(rng, 1, 16, 7, weight_norm=True),
        "downsample_blocks": downsample_blocks,
        "mel_conv": init_conv1d_params(rng, cfg.num_mels, cfg.upsample_initial_channel // 2, 7, weight_norm=True, init_std=0.01),
        "upsample_conv_blocks": upsample_conv_blocks,
        "conv_post": init_conv1d_params(rng, up_ch, 1, 7, bias=False, weight_norm=True, init_std=0.01),
    }
    if cfg.gin_channels != 0:
        params["cond"] = init_conv1d_params(rng, 256, cfg.upsample_initial_channel // 2, 1)
    return params


def _adain_forward(x: Array, weight: Array, slope: float, rng_key: Array | None, deterministic: bool) -> Array:
    x_f32 = jnp.asarray(x, jnp.float32)
    if rng_key is not None and not deterministic:
        noise = jax.random.normal(rng_key, shape=x_f32.shape, dtype=jnp.float32)
        x_f32 = x_f32 + noise * jnp.asarray(weight, jnp.float32)[None, None, :]
    return leaky_relu(x_f32, slope)


def _parallel_resblock_forward(
    x: Array,
    params: Mapping[str, Any],
    slope: float,
    rng_key: Array | None,
    deterministic: bool,
    prec: PrecisionConfig,
) -> Array:
    kernel_sizes = (3, 7, 11)
    dilations = (1, 3, 5)
    h = conv1d(x, params["input_conv"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    acc = None
    for j, ks in enumerate(kernel_sizes):
        k1 = k2 = None
        if rng_key is not None and not deterministic:
            rng_key, k1, k2 = jax.random.split(rng_key, 3)
        blk = params["blocks"][j]
        cur = _adain_forward(h, blk["adain1"]["weight"], slope, k1, deterministic)
        cur = resblock_forward(cur, blk["resblock"], kernel_size=ks, dilations=dilations, leaky_relu_slope=slope, prec=prec)
        cur = _adain_forward(cur, blk["adain2"]["weight"], slope, k2, deterministic)
        acc = cur if acc is None else (acc + cur)
    return acc / float(len(kernel_sizes))


def refinegan_forward(
    mel: Array,
    f0: Array,
    params: Mapping[str, Any],
    cfg: RefineGANConfig,
    rng_key: Array | None = None,
    g: Array | None = None,
    deterministic: bool = False,
    prec: PrecisionConfig = PrecisionConfig(),
) -> Array:
    """RefineGAN forward on NWC mel:(B, T, C) + f0:(B, T) -> FP32 waveform (B, T*upp, 1)."""
    upp = math.prod(cfg.upsample_rates)
    if rng_key is not None and not deterministic:
        rng_key, k_sine = jax.random.split(rng_key)
    else:
        k_sine = None

    f0_up = upsample_linear_1d(jnp.asarray(f0, jnp.float32)[..., None], upp)
    sine_waves = _sine_generator_mrf_forward(
        f0_up,
        cfg.sample_rate,
        k_sine,
        harmonic_num=0,
        deterministic=deterministic,
    )
    har_source = jnp.tanh(linear(sine_waves, params["m_source"]["merge"], dtype=jnp.float32, precision=prec.fp32_precision))

    x = conv1d(har_source, params["pre_conv"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)

    # Downsample factor ratios match df0 = [size, new_size] (independent of frame count f0_size).
    size = upp
    downs: list[Array] = []
    for i, u_rev in enumerate(reversed(cfg.upsample_rates)):
        new_size = size // int(u_rev)
        x = leaky_relu(x, cfg.leaky_relu_slope)
        downs.append(x)
        x = kaiser_sinc_resample_1d(x, orig_freq=size, new_freq=new_size)
        size = new_size
        x = conv1d(x, params["downsample_blocks"][i], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)

    mel_h = conv1d(mel, params["mel_conv"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    if g is not None and "cond" in params:
        mel_h = mel_h + conv1d(g, params["cond"], dtype=prec.compute_dtype, precision=prec.compute_precision)

    cur = jnp.concatenate([jnp.asarray(mel_h, prec.compute_dtype), jnp.asarray(x, prec.compute_dtype)], axis=-1)

    for rate, blk_p, down in zip(cfg.upsample_rates, params["upsample_conv_blocks"], reversed(downs)):
        if rng_key is not None and not deterministic:
            rng_key, k_stage = jax.random.split(rng_key)
        else:
            k_stage = None

        def _up_stage(cur_in: Array, down_in: Array, p_blk: Mapping[str, Any], k_sub: Array | None) -> Array:
            h_up = leaky_relu(cur_in, cfg.leaky_relu_slope)
            h_up = jnp.asarray(upsample_linear_1d(h_up, int(rate)), prec.compute_dtype)
            h_cat = jnp.concatenate([h_up, jnp.asarray(down_in, prec.compute_dtype)], axis=-1)
            return _parallel_resblock_forward(h_cat, p_blk, cfg.leaky_relu_slope, k_sub, deterministic, prec)

        cur = (
            jax.checkpoint(_up_stage)(cur, down, blk_p, k_stage)
            if cfg.checkpointing
            else _up_stage(cur, down, blk_p, k_stage)
        )

    cur = leaky_relu(cur, cfg.leaky_relu_slope)
    cur = conv1d(cur, params["conv_post"], padding=3, dtype=prec.compute_dtype, precision=prec.compute_precision)
    return jnp.tanh(jnp.asarray(cur, jnp.float32))


# ---------------------------------------------------------------------------
# PyTorch <-> JAX Vocoder Checkpoints
# ---------------------------------------------------------------------------


def hifigan_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: HiFiGANConfig) -> dict[str, Any]:
    num_ups = len(cfg.upsample_rates)
    num_rbs = num_ups * len(cfg.resblock_kernel_sizes)
    params: dict[str, Any] = {
        "conv_pre": extract_conv_from_torch_state(state, f"{prefix}.conv_pre"),
        "ups": [extract_conv_from_torch_state(state, f"{prefix}.ups.{i}") for i in range(num_ups)],
        "resblocks": [
            resblock_from_torch_state(state, f"{prefix}.resblocks.{i}", len(cfg.resblock_dilation_sizes[i % len(cfg.resblock_kernel_sizes)]))
            for i in range(num_rbs)
        ],
        "conv_post": extract_conv_from_torch_state(state, f"{prefix}.conv_post"),
    }
    if cfg.gin_channels != 0:
        params["cond"] = extract_conv_from_torch_state(state, f"{prefix}.cond")
    return params


def hifigan_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    export_conv_to_torch_state(state, f"{prefix}.conv_pre", params["conv_pre"])
    for i, up in enumerate(params["ups"]):
        export_conv_to_torch_state(state, f"{prefix}.ups.{i}", up, remove_weight_norm=remove_weight_norm)
    for i, rb in enumerate(params["resblocks"]):
        resblock_to_torch_state(state, f"{prefix}.resblocks.{i}", rb, remove_weight_norm=remove_weight_norm)
    export_conv_to_torch_state(state, f"{prefix}.conv_post", params["conv_post"])
    if "cond" in params:
        export_conv_to_torch_state(state, f"{prefix}.cond", params["cond"])


def hifigan_nsf_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: HiFiGANNSFConfig) -> dict[str, Any]:
    num_ups = len(cfg.upsample_rates)
    num_rbs = num_ups * len(cfg.resblock_kernel_sizes)
    params: dict[str, Any] = {
        "m_source": {"l_linear": extract_linear_from_torch_state(state, f"{prefix}.m_source.l_linear")},
        "conv_pre": extract_conv_from_torch_state(state, f"{prefix}.conv_pre"),
        "ups": [extract_conv_from_torch_state(state, f"{prefix}.ups.{i}") for i in range(num_ups)],
        "noise_convs": [extract_conv_from_torch_state(state, f"{prefix}.noise_convs.{i}") for i in range(num_ups)],
        "resblocks": [
            resblock_from_torch_state(state, f"{prefix}.resblocks.{i}", len(cfg.resblock_dilation_sizes[i % len(cfg.resblock_kernel_sizes)]))
            for i in range(num_rbs)
        ],
        "conv_post": extract_conv_from_torch_state(state, f"{prefix}.conv_post"),
    }
    if cfg.gin_channels != 0:
        params["cond"] = extract_conv_from_torch_state(state, f"{prefix}.cond")
    return params


def hifigan_nsf_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    export_linear_to_torch_state(state, f"{prefix}.m_source.l_linear", params["m_source"]["l_linear"])
    export_conv_to_torch_state(state, f"{prefix}.conv_pre", params["conv_pre"])
    for i, up in enumerate(params["ups"]):
        export_conv_to_torch_state(state, f"{prefix}.ups.{i}", up, remove_weight_norm=remove_weight_norm)
    for i, nc in enumerate(params["noise_convs"]):
        export_conv_to_torch_state(state, f"{prefix}.noise_convs.{i}", nc)
    for i, rb in enumerate(params["resblocks"]):
        resblock_to_torch_state(state, f"{prefix}.resblocks.{i}", rb, remove_weight_norm=remove_weight_norm)
    export_conv_to_torch_state(state, f"{prefix}.conv_post", params["conv_post"])
    if "cond" in params:
        export_conv_to_torch_state(state, f"{prefix}.cond", params["cond"])


def hifigan_mrf_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: HiFiGANMRFConfig) -> dict[str, Any]:
    num_ups = len(cfg.upsample_rates)
    mrfs = []
    for i in range(num_ups):
        stage_mrfs = []
        for j, d_list in enumerate(cfg.resblock_dilations):
            layers = [
                {
                    "conv1": extract_conv_from_torch_state(state, f"{prefix}.mrfs.{i}.{j}.layers.{l}.conv1"),
                    "conv2": extract_conv_from_torch_state(state, f"{prefix}.mrfs.{i}.{j}.layers.{l}.conv2"),
                }
                for l in range(len(d_list))
            ]
            stage_mrfs.append({"layers": layers})
        mrfs.append(stage_mrfs)

    params: dict[str, Any] = {
        "m_source": {"l_linear": extract_linear_from_torch_state(state, f"{prefix}.m_source.l_linear")},
        "conv_pre": extract_conv_from_torch_state(state, f"{prefix}.conv_pre"),
        "upsamples": [extract_conv_from_torch_state(state, f"{prefix}.upsamples.{i}") for i in range(num_ups)],
        "noise_convs": [extract_conv_from_torch_state(state, f"{prefix}.noise_convs.{i}") for i in range(num_ups)],
        "mrfs": mrfs,
        "conv_post": extract_conv_from_torch_state(state, f"{prefix}.conv_post"),
    }
    if cfg.gin_channels != 0:
        params["cond"] = extract_conv_from_torch_state(state, f"{prefix}.cond")
    return params


def hifigan_mrf_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    export_linear_to_torch_state(state, f"{prefix}.m_source.l_linear", params["m_source"]["l_linear"])
    export_conv_to_torch_state(state, f"{prefix}.conv_pre", params["conv_pre"], remove_weight_norm=remove_weight_norm)
    for i, up in enumerate(params["upsamples"]):
        export_conv_to_torch_state(state, f"{prefix}.upsamples.{i}", up, remove_weight_norm=remove_weight_norm)
    for i, nc in enumerate(params["noise_convs"]):
        export_conv_to_torch_state(state, f"{prefix}.noise_convs.{i}", nc)
    for i, stage_mrfs in enumerate(params["mrfs"]):
        for j, mrf_b in enumerate(stage_mrfs):
            for l, layer in enumerate(mrf_b["layers"]):
                export_conv_to_torch_state(
                    state, f"{prefix}.mrfs.{i}.{j}.layers.{l}.conv1", layer["conv1"], remove_weight_norm=remove_weight_norm
                )
                export_conv_to_torch_state(
                    state, f"{prefix}.mrfs.{i}.{j}.layers.{l}.conv2", layer["conv2"], remove_weight_norm=remove_weight_norm
                )
    export_conv_to_torch_state(state, f"{prefix}.conv_post", params["conv_post"], remove_weight_norm=remove_weight_norm)
    if "cond" in params:
        export_conv_to_torch_state(state, f"{prefix}.cond", params["cond"])


def refinegan_from_torch_state(state: Mapping[str, Any], prefix: str, cfg: RefineGANConfig) -> dict[str, Any]:
    num_ups = len(cfg.upsample_rates)
    upsample_conv_blocks = []
    for i in range(num_ups):
        up_p = f"{prefix}.upsample_conv_blocks.{i}"
        blocks = []
        for j in range(3):
            b_p = f"{up_p}.blocks.{j}"
            blocks.append(
                {
                    "adain1": {"weight": np.asarray(state[f"{b_p}.0.weight"], dtype=np.float32).copy()},
                    "resblock": resblock_from_torch_state(state, f"{b_p}.1", 3),
                    "adain2": {"weight": np.asarray(state[f"{b_p}.2.weight"], dtype=np.float32).copy()},
                }
            )
        upsample_conv_blocks.append(
            {
                "input_conv": extract_conv_from_torch_state(state, f"{up_p}.input_conv"),
                "blocks": blocks,
            }
        )

    params: dict[str, Any] = {
        "m_source": {"merge": extract_linear_from_torch_state(state, f"{prefix}.m_source.merge.0")},
        "pre_conv": extract_conv_from_torch_state(state, f"{prefix}.pre_conv"),
        "downsample_blocks": [extract_conv_from_torch_state(state, f"{prefix}.downsample_blocks.{i}") for i in range(num_ups)],
        "mel_conv": extract_conv_from_torch_state(state, f"{prefix}.mel_conv"),
        "upsample_conv_blocks": upsample_conv_blocks,
        "conv_post": extract_conv_from_torch_state(state, f"{prefix}.conv_post"),
    }
    if cfg.gin_channels != 0:
        params["cond"] = extract_conv_from_torch_state(state, f"{prefix}.cond")
    return params


def refinegan_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    remove_weight_norm: bool = False,
) -> None:
    export_linear_to_torch_state(state, f"{prefix}.m_source.merge.0", params["m_source"]["merge"])
    export_conv_to_torch_state(state, f"{prefix}.pre_conv", params["pre_conv"], remove_weight_norm=remove_weight_norm)
    for i, db in enumerate(params["downsample_blocks"]):
        export_conv_to_torch_state(state, f"{prefix}.downsample_blocks.{i}", db, remove_weight_norm=remove_weight_norm)
    export_conv_to_torch_state(state, f"{prefix}.mel_conv", params["mel_conv"], remove_weight_norm=remove_weight_norm)
    for i, ucb in enumerate(params["upsample_conv_blocks"]):
        up_p = f"{prefix}.upsample_conv_blocks.{i}"
        export_conv_to_torch_state(state, f"{up_p}.input_conv", ucb["input_conv"])
        for j, blk in enumerate(ucb["blocks"]):
            b_p = f"{up_p}.blocks.{j}"
            state[f"{b_p}.0.weight"] = np.asarray(blk["adain1"]["weight"], dtype=np.float32).copy()
            resblock_to_torch_state(state, f"{b_p}.1", blk["resblock"], remove_weight_norm=remove_weight_norm)
            state[f"{b_p}.2.weight"] = np.asarray(blk["adain2"]["weight"], dtype=np.float32).copy()
    export_conv_to_torch_state(state, f"{prefix}.conv_post", params["conv_post"], remove_weight_norm=remove_weight_norm)
    if "cond" in params:
        export_conv_to_torch_state(state, f"{prefix}.cond", params["cond"])
