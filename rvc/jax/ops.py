"""Core JAX tensor primitives, NWC/NHWC layers, resampling, and PyTorch weight helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

Array = jax.Array
PyTree = Any

DEFAULT_KAISER_ROLLOFF = 0.9475937167399596
DEFAULT_KAISER_BETA = 14.769656459379492
DEFAULT_KAISER_WIDTH = 64


# ---------------------------------------------------------------------------
# Scalar / padding / shape utilities
# ---------------------------------------------------------------------------


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Compute symmetric 1D padding matching rvc.algorithm.commons.get_padding."""
    return int((kernel_size * dilation - dilation) // 2)


def leaky_relu(x: Array, negative_slope: float = 0.1) -> Array:
    """LeakyReLU activation."""
    return jax.nn.leaky_relu(x, negative_slope=negative_slope)


def exact_gelu(x: Array) -> Array:
    """Exact Gaussian Error Linear Unit matching torch.nn.functional.gelu(x, approximate='none')."""
    x_f32 = jnp.asarray(x, jnp.float32)
    return jnp.asarray(jax.nn.gelu(x_f32, approximate=False), x.dtype)


def avg_pool2d(x: Array, kernel_size: tuple[int, int] = (2, 2), stride: tuple[int, int] | None = None) -> Array:
    """2D average pooling on NHWC (B, H, W, C) with VALID padding."""
    kh, kw = int(kernel_size[0]), int(kernel_size[1])
    sh, sw = (kh, kw) if stride is None else (int(stride[0]), int(stride[1]))
    sums = lax.reduce_window(
        x,
        jnp.array(0.0, dtype=x.dtype),
        lax.add,
        window_dimensions=(1, kh, kw, 1),
        window_strides=(1, sh, sw, 1),
        padding="VALID",
    )
    return sums / jnp.asarray(float(kh * kw), dtype=x.dtype)


def conv_transpose2d(
    x: Array,
    params: Mapping[str, Array],
    stride: tuple[int, int] = (2, 2),
    padding: tuple[int, int] = (1, 1),
    output_padding: tuple[int, int] = (1, 1),
    dtype: Any = jnp.float32,
    precision: Any = None,
) -> Array:
    """2D transposed convolution on NHWC tensor with weight shape (Kh, Kw, C_out, C_in)."""
    w = resolve_kernel(params)
    kh, kw = w.shape[0], w.shape[1]
    pad_h = (kh - 1 - int(padding[0]), kh - 1 - int(padding[0]) + int(output_padding[0]))
    pad_w = (kw - 1 - int(padding[1]), kw - 1 - int(padding[1]) + int(output_padding[1]))
    y = lax.conv_transpose(
        lhs=jnp.asarray(x, dtype),
        rhs=jnp.asarray(w, dtype),
        strides=(int(stride[0]), int(stride[1])),
        padding=(pad_h, pad_w),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        transpose_kernel=True,
        precision=precision,
    )
    if "bias" in params and params["bias"] is not None:
        y = y + jnp.asarray(params["bias"], dtype)
    return y


def masked_group_norm_1d(
    x: Array,
    params: Mapping[str, Array],
    num_groups: int,
    mask_bt1: Array | None = None,
    eps: float = 1e-5,
) -> Array:
    """Strictly FP32 masked GroupNorm over NWC (B, T, C) supporting per-channel (G=C) and multi-group."""
    x_f32 = jnp.asarray(x, jnp.float32)
    b, t, c = x_f32.shape
    group_size = c // int(num_groups)
    xg = x_f32.reshape(b, t, int(num_groups), group_size)
    if mask_bt1 is not None:
        m = jnp.asarray(mask_bt1, jnp.float32)[..., None]  # (B, T, 1, 1)
        denom = jnp.maximum(jnp.sum(m, axis=1, keepdims=True) * float(group_size), 1.0)
        mean = jnp.sum(xg * m, axis=(1, 3), keepdims=True) / denom
        var = jnp.sum(jnp.square(xg - mean) * m, axis=(1, 3), keepdims=True) / denom
        normed = ((xg - mean) * lax.rsqrt(var + eps) * m).reshape(b, t, c)
    else:
        mean = jnp.mean(xg, axis=(1, 3), keepdims=True)
        var = jnp.mean(jnp.square(xg - mean), axis=(1, 3), keepdims=True)
        normed = ((xg - mean) * lax.rsqrt(var + eps)).reshape(b, t, c)
    return normed * jnp.asarray(params["weight"], jnp.float32) + jnp.asarray(params["bias"], jnp.float32)


def gelu_1702(x: Array) -> Array:
    """Fast sigmoid GELU approximation x * sigmoid(1.702 * x) matching rvc.algorithm.attentions.FFN."""
    x_f32 = jnp.asarray(x, jnp.float32)
    out = x_f32 * jax.nn.sigmoid(1.702 * x_f32)
    return jnp.asarray(out, x.dtype)


def fused_add_tanh_sigmoid_multiply(
    input_a: Array,
    input_b: Array,
    n_channels: int,
) -> Array:
    """WaveNet gated activation computed safely in float32 for NWC tensors."""
    in_act = jnp.asarray(input_a, jnp.float32) + jnp.asarray(input_b, jnp.float32)
    t_act = jnp.tanh(in_act[..., :n_channels])
    s_act = jax.nn.sigmoid(in_act[..., n_channels:])
    return t_act * s_act


def sequence_mask(length: Array, max_length: int | None = None) -> Array:
    """Boolean-cast float32 mask of shape (B, T) matching PyTorch sequence_mask."""
    length_i = jnp.asarray(length, jnp.int32)
    if max_length is None:
        max_length = int(jnp.max(length_i))
    positions = jnp.arange(max_length, dtype=jnp.int32)[None, :]
    return (positions < length_i[:, None]).astype(jnp.float32)


def slice_segments(x: Array, ids_str: Array, segment_size: int) -> Array:
    """Extract fixed-length contiguous slices along axis 1 for (B, T, ...) arrays."""
    ids = jnp.asarray(ids_str, jnp.int32)

    def _slice_one(arr: Array, start: Array) -> Array:
        return lax.dynamic_slice_in_dim(arr, start, segment_size, axis=0)

    return jax.vmap(_slice_one, in_axes=(0, 0))(x, ids)


def rand_slice_segments(
    x: Array,
    x_lengths: Array | int,
    segment_size: int,
    rng_key: Array,
) -> tuple[Array, Array]:
    """Randomly slice segments of length segment_size along axis 1 in float32/int32."""
    batch_size = x.shape[0]
    ids_str_max = jnp.maximum(jnp.asarray(x_lengths, jnp.float32) - float(segment_size) + 1.0, 1.0)
    rand_u = jax.random.uniform(rng_key, shape=(batch_size,), dtype=jnp.float32)
    ids_str = jnp.floor(rand_u * ids_str_max).astype(jnp.int32)
    ids_str = jnp.clip(ids_str, 0, max(0, x.shape[1] - segment_size))
    return slice_segments(x, ids_str, segment_size), ids_str


def reflect_pad_1d(x: Array, pad_left: int, pad_right: int) -> Array:
    """Static reflect padding along axis 1 for (B, T, C) without dynamic pad lowering issues."""
    if pad_left == 0 and pad_right == 0:
        return x
    pieces = []
    if pad_left > 0:
        if pad_left >= x.shape[1]:
            return jnp.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="reflect")
        pieces.append(x[:, pad_left:0:-1, :])
    pieces.append(x)
    if pad_right > 0:
        if pad_right >= x.shape[1]:
            return jnp.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="reflect")
        pieces.append(x[:, -2 : -2 - pad_right : -1, :])
    return jnp.concatenate(pieces, axis=1)


def constant_pad_1d(x: Array, pad_left: int, pad_right: int, value: float = 0.0) -> Array:
    """Constant padding along axis 1 for (B, T, C)."""
    if pad_left == 0 and pad_right == 0:
        return x
    return jnp.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant", constant_values=value)


def upsample_nearest_1d(x: Array, scale_factor: int) -> Array:
    """Nearest-neighbor 1D upsampling on (B, T, C) along axis 1."""
    if scale_factor == 1:
        return x
    return jnp.repeat(x, int(scale_factor), axis=1)


def upsample_linear_1d(x: Array, scale_factor: int) -> Array:
    """Linear 1D upsampling on (B, T, C) matching F.interpolate(..., mode='linear', align_corners=False)."""
    if scale_factor == 1:
        return x
    b, t, c = x.shape
    return jax.image.resize(jnp.asarray(x, jnp.float32), (b, int(t * scale_factor), c), method="linear", antialias=False)


# ---------------------------------------------------------------------------
# Exact torchaudio-compatible Kaiser-windowed sinc resampling (FP32)
# ---------------------------------------------------------------------------

_RESAMPLE_KERNEL_CACHE: dict[tuple[int, int, int, float, float], tuple[np.ndarray, int, int, int]] = {}


def compute_kaiser_sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = DEFAULT_KAISER_WIDTH,
    rolloff: float = DEFAULT_KAISER_ROLLOFF,
    beta: float = DEFAULT_KAISER_BETA,
) -> tuple[np.ndarray, int, int, int]:
    """Build exact torchaudio sinc_interp_kaiser polyphase FIR kernel in JAX WIO layout (K, 1, new_r)."""
    key = (int(orig_freq), int(new_freq), int(lowpass_filter_width), float(rolloff), float(beta))
    cached = _RESAMPLE_KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig_r = int(orig_freq) // gcd
    new_r = int(new_freq) // gcd

    base_freq = float(min(orig_r, new_r)) * float(rolloff)
    width = int(math.ceil(lowpass_filter_width * orig_r / base_freq))

    idx = np.arange(-width, width + orig_r, dtype=np.float64)[None, None, :] / float(orig_r)
    t = np.arange(0, -new_r, -1, dtype=np.float64)[:, None, None] / float(new_r) + idx
    t = np.clip(t * base_freq, -float(lowpass_filter_width), float(lowpass_filter_width))

    inside = np.maximum(1.0 - (t / float(lowpass_filter_width)) ** 2, 0.0)
    window = np.i0(float(beta) * np.sqrt(inside)) / np.i0(float(beta))

    t_pi = t * math.pi
    scale = base_freq / float(orig_r)
    sinc = np.where(t_pi == 0.0, 1.0, np.sin(t_pi) / np.where(t_pi == 0.0, 1.0, t_pi))
    # PyTorch kernel has shape (new_r, 1, K); JAX NWC conv wants WIO = (K, 1, new_r).
    kernels = (sinc * window * scale).astype(np.float32)
    kernels_wio = np.ascontiguousarray(np.transpose(kernels, (2, 1, 0)))

    result = (kernels_wio, width, orig_r, new_r)
    _RESAMPLE_KERNEL_CACHE[key] = result
    return result


def kaiser_sinc_resample_1d(
    x: Array,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = DEFAULT_KAISER_WIDTH,
    rolloff: float = DEFAULT_KAISER_ROLLOFF,
    beta: float = DEFAULT_KAISER_BETA,
) -> Array:
    """Strictly FP32 sinc_interp_kaiser resampling matching torchaudio.functional.resample on (B, T, C) or (B, T)."""
    if int(orig_freq) == int(new_freq):
        return jnp.asarray(x, jnp.float32)

    kernels_wio, width, orig_r, new_r = compute_kaiser_sinc_resample_kernel(
        orig_freq=orig_freq,
        new_freq=new_freq,
        lowpass_filter_width=lowpass_filter_width,
        rolloff=rolloff,
        beta=beta,
    )
    x_f32 = jnp.asarray(x, jnp.float32)
    squeeze_last = x_f32.ndim == 2
    if squeeze_last:
        x_f32 = x_f32[..., None]

    batch, length, channels = x_f32.shape
    # Flatten channels into batch -> (B*C, T, 1) so every channel receives the same phase FIR filter.
    flat = jnp.transpose(x_f32, (0, 2, 1)).reshape(batch * channels, length, 1)
    target_length = int(math.ceil(new_r * length / orig_r))

    resampled = lax.conv_general_dilated(
        lhs=flat,
        rhs=jnp.asarray(kernels_wio, jnp.float32),
        window_strides=(orig_r,),
        padding=((width, width + orig_r),),
        dimension_numbers=("NWC", "WIO", "NWC"),
        precision=lax.Precision.HIGHEST,
    )
    resampled = resampled.reshape(batch * channels, -1)[:, :target_length]
    out = jnp.transpose(resampled.reshape(batch, channels, target_length), (0, 2, 1))
    return out[..., 0] if squeeze_last else out


# ---------------------------------------------------------------------------
# Weight normalization, spectral normalization, and initialization
# ---------------------------------------------------------------------------


def weight_norm_decompose_np(weight: np.ndarray, axis: int = -1) -> dict[str, np.ndarray]:
    """Decompose kernel into weight_norm (weight_g, weight_v) along `axis`."""
    v = np.asarray(weight, dtype=np.float32)
    pos_axis = axis % v.ndim
    norm_axes = tuple(i for i in range(v.ndim) if i != pos_axis)
    g = np.sqrt(np.maximum(np.sum(v * v, axis=norm_axes, keepdims=True), 1e-24)).astype(np.float32)
    return {"weight_g": g, "weight_v": v}


def weight_norm_compose_np(weight_g: np.ndarray, weight_v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Recompose weight_norm kernel in float32 NumPy."""
    v = np.asarray(weight_v, dtype=np.float32)
    g = np.asarray(weight_g, dtype=np.float32)
    pos_axis = axis % v.ndim
    norm_axes = tuple(i for i in range(v.ndim) if i != pos_axis)
    v_norm = np.sqrt(np.maximum(np.sum(v * v, axis=norm_axes, keepdims=True), 1e-24))
    return (v * (g / v_norm)).astype(np.float32)


def weight_norm_compose(weight_g: Array, weight_v: Array, axis: int = -1) -> Array:
    """Compute normalized kernel g * v / ||v|| strictly in float32."""
    v = jnp.asarray(weight_v, jnp.float32)
    g = jnp.asarray(weight_g, jnp.float32)
    pos_axis = axis % v.ndim
    norm_axes = tuple(i for i in range(v.ndim) if i != pos_axis)
    v_norm = jnp.sqrt(jnp.maximum(jnp.sum(v * v, axis=norm_axes, keepdims=True), 1e-24))
    return v * (g / v_norm)


def spectral_norm_compose(
    weight: Array,
    u: Array | None = None,
    n_power_iterations: int = 1,
    eps: float = 1e-12,
) -> tuple[Array, Array | None]:
    """Apply spectral normalization along the last axis (C_out) in float32."""
    w = jnp.asarray(weight, jnp.float32)
    w_mat = jnp.transpose(w, (w.ndim - 1,) + tuple(range(w.ndim - 1))).reshape(w.shape[-1], -1)
    if u is None:
        u_cur = jnp.ones((w_mat.shape[0],), dtype=jnp.float32) / math.sqrt(max(1, w_mat.shape[0]))
    else:
        u_cur = jnp.asarray(u, jnp.float32)
    for _ in range(max(1, n_power_iterations)):
        v_raw = jnp.matmul(w_mat.T, u_cur, precision=lax.Precision.HIGHEST)
        v_cur = v_raw / jnp.maximum(jnp.linalg.norm(v_raw), eps)
        u_raw = jnp.matmul(w_mat, v_cur, precision=lax.Precision.HIGHEST)
        u_cur = u_raw / jnp.maximum(jnp.linalg.norm(u_raw), eps)
    u_sg = lax.stop_gradient(u_cur)
    v_sg = lax.stop_gradient(v_cur)
    sigma = jnp.dot(u_sg, jnp.matmul(w_mat, v_sg, precision=lax.Precision.HIGHEST), precision=lax.Precision.HIGHEST)
    sigma = jnp.maximum(jnp.abs(sigma), eps)
    return w / sigma, u_sg


def resolve_kernel(params: Mapping[str, Array]) -> Array:
    """Resolve effective float32 kernel from raw weight or weight_norm pair."""
    if "weight_g" in params and "weight_v" in params:
        return weight_norm_compose(params["weight_g"], params["weight_v"], axis=-1)
    w = jnp.asarray(params["weight"], jnp.float32)
    if "sn_u" in params:
        w, _ = spectral_norm_compose(w, params["sn_u"])
    return w


def _kaiming_uniform(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    fan_in: int,
    a: float = math.sqrt(5.0),
) -> np.ndarray:
    gain = math.sqrt(2.0 / (1.0 + a * a))
    std = gain / math.sqrt(max(1, fan_in))
    bound = math.sqrt(3.0) * std
    return rng.uniform(-bound, bound, size=shape).astype(np.float32)


def _init_bias_uniform(rng: np.random.Generator, fan_in: int, out_dim: int) -> np.ndarray:
    bound = 1.0 / math.sqrt(max(1, fan_in)) if fan_in > 0 else 0.0
    return rng.uniform(-bound, bound, size=(out_dim,)).astype(np.float32)


def init_conv1d_params(
    rng: np.random.Generator,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    groups: int = 1,
    bias: bool = True,
    weight_norm: bool = False,
    spectral_norm: bool = False,
    init_std: float | None = None,
    zero_init: bool = False,
) -> dict[str, np.ndarray]:
    """Initialize 1D convolution parameters in JAX WIO shape (K, C_in//g, C_out)."""
    c_in_per_group = in_channels // groups
    shape = (kernel_size, c_in_per_group, out_channels)
    fan_in = c_in_per_group * kernel_size
    if zero_init:
        w = np.zeros(shape, dtype=np.float32)
    elif init_std is not None:
        w = rng.normal(0.0, float(init_std), size=shape).astype(np.float32)
    else:
        w = _kaiming_uniform(rng, shape, fan_in=fan_in)

    out: dict[str, np.ndarray] = weight_norm_decompose_np(w, axis=-1) if weight_norm else {"weight": w}
    if spectral_norm and not weight_norm:
        u = rng.normal(0.0, 1.0, size=(out_channels,)).astype(np.float32)
        u /= max(1e-12, float(np.linalg.norm(u)))
        out["sn_u"] = u
    if bias:
        if zero_init or init_std is not None:
            out["bias"] = np.zeros((out_channels,), dtype=np.float32)
        else:
            out["bias"] = _init_bias_uniform(rng, fan_in, out_channels)
    return out


def init_conv_transpose1d_params(
    rng: np.random.Generator,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    groups: int = 1,
    bias: bool = True,
    weight_norm: bool = False,
    init_std: float | None = None,
) -> dict[str, np.ndarray]:
    """Initialize ConvTranspose1d parameters in JAX shape (K, C_out//g, C_in)."""
    c_out_per_group = out_channels // groups
    shape = (kernel_size, c_out_per_group, in_channels)
    fan_in = in_channels * kernel_size
    if init_std is not None:
        w = rng.normal(0.0, float(init_std), size=shape).astype(np.float32)
    else:
        w = _kaiming_uniform(rng, shape, fan_in=fan_in)
    out: dict[str, np.ndarray] = weight_norm_decompose_np(w, axis=-1) if weight_norm else {"weight": w}
    if bias:
        if init_std is not None:
            out["bias"] = np.zeros((out_channels,), dtype=np.float32)
        else:
            out["bias"] = _init_bias_uniform(rng, c_out_per_group * kernel_size, out_channels)
    return out


def init_conv2d_params(
    rng: np.random.Generator,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    groups: int = 1,
    bias: bool = True,
    weight_norm: bool = False,
    spectral_norm: bool = False,
    init_std: float | None = None,
) -> dict[str, np.ndarray]:
    """Initialize Conv2d parameters in JAX HWIO shape (Kh, Kw, C_in//g, C_out)."""
    kh, kw = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
    c_in_per_group = in_channels // groups
    shape = (kh, kw, c_in_per_group, out_channels)
    fan_in = c_in_per_group * kh * kw
    if init_std is not None:
        w = rng.normal(0.0, float(init_std), size=shape).astype(np.float32)
    else:
        w = _kaiming_uniform(rng, shape, fan_in=fan_in)
    out: dict[str, np.ndarray] = weight_norm_decompose_np(w, axis=-1) if weight_norm else {"weight": w}
    if spectral_norm and not weight_norm:
        u = rng.normal(0.0, 1.0, size=(out_channels,)).astype(np.float32)
        u /= max(1e-12, float(np.linalg.norm(u)))
        out["sn_u"] = u
    if bias:
        out["bias"] = _init_bias_uniform(rng, fan_in, out_channels)
    return out


def init_linear_params(
    rng: np.random.Generator,
    in_features: int,
    out_features: int,
    bias: bool = True,
    xavier: bool = False,
) -> dict[str, np.ndarray]:
    """Initialize Linear parameters in JAX shape (C_in, C_out)."""
    shape = (in_features, out_features)
    if xavier:
        bound = math.sqrt(6.0 / float(in_features + out_features))
        w = rng.uniform(-bound, bound, size=shape).astype(np.float32)
    else:
        w = _kaiming_uniform(rng, shape, fan_in=in_features)
    out: dict[str, np.ndarray] = {"weight": w}
    if bias:
        out["bias"] = np.zeros((out_features,), dtype=np.float32) if xavier else _init_bias_uniform(rng, in_features, out_features)
    return out


def stft_complex(
    y: Array,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: np.ndarray | Array,
    *,
    pad_left: int = 0,
    pad_right: int = 0,
) -> tuple[Array, Array]:
    """Strictly FP32 real-input STFT matching torch.stft(..., center=False) after optional reflect pad."""
    y_f32 = jnp.asarray(y, jnp.float32)
    if y_f32.ndim == 3:
        y_f32 = y_f32[..., 0]

    if pad_left > 0 or pad_right > 0:
        y_f32 = reflect_pad_1d(y_f32[..., None], pad_left, pad_right)[..., 0]

    win_arr = jnp.asarray(window, jnp.float32)
    if win_length < n_fft:
        w_left = (n_fft - win_length) // 2
        w_right = n_fft - win_length - w_left
        win_arr = jnp.pad(win_arr, (w_left, w_right))

    length = y_f32.shape[1]
    num_frames = 1 + (length - n_fft) // hop_length
    starts = jnp.arange(num_frames, dtype=jnp.int32)[:, None] * int(hop_length)
    offsets = jnp.arange(n_fft, dtype=jnp.int32)[None, :]
    frames = y_f32[:, starts + offsets]  # (B, TT, n_fft)
    windowed = frames * win_arr[None, None, :]
    spec = jnp.fft.rfft(windowed, n=int(n_fft), axis=-1)  # (B, TT, F)
    spec_ft = jnp.transpose(spec, (0, 2, 1))  # (B, F, TT)
    return jnp.real(spec_ft), jnp.imag(spec_ft)


def init_embedding_params(
    rng: np.random.Generator,
    num_embeddings: int,
    embedding_dim: int,
    std: float = 1.0,
) -> dict[str, np.ndarray]:
    """Initialize Embedding weight of shape (num_embeddings, embedding_dim)."""
    return {"weight": rng.normal(0.0, float(std), size=(num_embeddings, embedding_dim)).astype(np.float32)}


def init_layer_norm_params(channels: int) -> dict[str, np.ndarray]:
    """Initialize channel LayerNorm parameters (gamma, beta) matching rvc.algorithm.normalization.LayerNorm."""
    return {
        "gamma": np.ones((channels,), dtype=np.float32),
        "beta": np.zeros((channels,), dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Functional NWC / NHWC forward layers
# ---------------------------------------------------------------------------


def conv1d(
    x: Array,
    params: Mapping[str, Array],
    stride: int = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int = 1,
    groups: int = 1,
    dtype: Any = jnp.float32,
    precision: Any = None,
) -> Array:
    """1D convolution on NWC tensor (B, T, C_in) with weight WIO (K, C_in//g, C_out)."""
    w = resolve_kernel(params)
    pad_pair = (padding, padding) if isinstance(padding, int) else tuple(padding)
    x_c = jnp.asarray(x, dtype)
    w_c = jnp.asarray(w, dtype)
    y = lax.conv_general_dilated(
        lhs=x_c,
        rhs=w_c,
        window_strides=(int(stride),),
        padding=(pad_pair,),
        rhs_dilation=(int(dilation),),
        feature_group_count=int(groups),
        dimension_numbers=("NWC", "WIO", "NWC"),
        precision=precision,
    )
    if "bias" in params and params["bias"] is not None:
        y = y + jnp.asarray(params["bias"], dtype)
    return y


_POLYPHASE_BASIS_CACHE: dict[tuple[int, int, int], tuple[np.ndarray, int, int, int]] = {}


def _polyphase_basis(kernel_size: int, stride: int, padding: int) -> tuple[np.ndarray, int, int, int]:
    """Build static zero/one permutation matrix converting strided ConvTranspose1d to dense Conv1d."""
    key = (int(kernel_size), int(stride), int(padding))
    cached = _POLYPHASE_BASIS_CACHE.get(key)
    if cached is not None:
        return cached

    d_min = -(stride - 1 + padding) // stride
    d_max = (kernel_size - 1 - padding) // stride
    u_min, u_max = -d_max, -d_min
    k_eff = u_max - u_min + 1
    mat = np.zeros((k_eff * stride, kernel_size), dtype=np.float32)
    for u_idx, u in enumerate(range(u_min, u_max + 1)):
        d = -u
        for r in range(stride):
            kk = d * stride + r + padding
            if 0 <= kk < kernel_size:
                mat[u_idx * stride + r, kk] = 1.0
    result = (mat, u_min, u_max, k_eff)
    _POLYPHASE_BASIS_CACHE[key] = result
    return result


def conv_transpose1d(
    x: Array,
    params: Mapping[str, Array],
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
    dtype: Any = jnp.float32,
    precision: Any = None,
) -> Array:
    """1D transposed convolution on NWC tensor (B, T, C_in) matching PyTorch ConvTranspose1d."""
    if groups != 1:
        raise ValueError(f"Grouped conv_transpose1d is unsupported (got groups={groups}).")
    w = resolve_kernel(params)
    k, c_out, c_in = w.shape
    s = int(stride)
    p = int(padding)
    op = int(output_padding)
    d = int(dilation)
    x_c = jnp.asarray(x, dtype)

    # Dense polyphase Conv1d avoids zero-interleaved lhs_dilate on TPU/CPU when stride > 1.
    t_in = x_c.shape[1]
    t_out = (t_in - 1) * s - 2 * p + k + op
    if s > 1 and d == 1 and t_out > 0:
        m_count = (t_out + s - 1) // s
        mat_np, u_min, u_max, k_eff = _polyphase_basis(k, s, p)
        w_ci = jnp.asarray(jnp.transpose(w, (0, 2, 1)), dtype).reshape(k, c_in * c_out)
        w_packed = jnp.matmul(jnp.asarray(mat_np, dtype), w_ci, precision=precision).reshape(k_eff, s, c_in, c_out)
        w_packed = jnp.transpose(w_packed, (0, 2, 1, 3)).reshape(k_eff, c_in, s * c_out)
        need_end = (m_count - 1) + u_max + 1
        pad_l = max(0, -u_min)
        pad_r = max(0, need_end - t_in)
        y_blk = lax.conv_general_dilated(
            lhs=x_c,
            rhs=w_packed,
            window_strides=(1,),
            padding=((pad_l, pad_r),),
            dimension_numbers=("NWC", "WIO", "NWC"),
            precision=precision,
        )
        start_idx = max(0, u_min)
        y = y_blk[:, start_idx : start_idx + m_count, :].reshape(x_c.shape[0], m_count * s, c_out)[:, :t_out, :]
    else:
        k_eff = (k - 1) * d + 1
        pad_left = k_eff - 1 - p
        pad_right = k_eff - 1 - p + op
        w_c = jnp.asarray(w, dtype)
        y = lax.conv_transpose(
            lhs=x_c,
            rhs=w_c,
            strides=(s,),
            padding=((pad_left, pad_right),),
            rhs_dilation=(d,),
            dimension_numbers=("NWC", "WIO", "NWC"),
            transpose_kernel=True,
            precision=precision,
        )
    if "bias" in params and params["bias"] is not None:
        y = y + jnp.asarray(params["bias"], dtype)
    return y


def conv2d(
    x: Array,
    params: Mapping[str, Array],
    stride: int | tuple[int, int] = (1, 1),
    padding: int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]] = (0, 0),
    dilation: int | tuple[int, int] = (1, 1),
    groups: int = 1,
    dtype: Any = jnp.float32,
    precision: Any = None,
) -> Array:
    """2D convolution on NHWC tensor (B, H, W, C_in) with weight HWIO."""
    w = resolve_kernel(params)
    stride_2 = (stride, stride) if isinstance(stride, int) else tuple(stride)
    dil_2 = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    if isinstance(padding, int):
        pad_2 = ((padding, padding), (padding, padding))
    elif isinstance(padding[0], int):
        ph, pw = padding  # type: ignore[misc]
        pad_2 = ((int(ph), int(ph)), (int(pw), int(pw)))
    else:
        pad_2 = tuple(tuple(p) for p in padding)  # type: ignore[assignment]

    x_c = jnp.asarray(x, dtype)
    w_c = jnp.asarray(w, dtype)
    y = lax.conv_general_dilated(
        lhs=x_c,
        rhs=w_c,
        window_strides=stride_2,
        padding=pad_2,
        rhs_dilation=dil_2,
        feature_group_count=int(groups),
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        precision=precision,
    )
    if "bias" in params and params["bias"] is not None:
        y = y + jnp.asarray(params["bias"], dtype)
    return y


def linear(
    x: Array,
    params: Mapping[str, Array],
    dtype: Any = jnp.float32,
    precision: Any = None,
) -> Array:
    """Affine linear transformation y = x @ W + b along last axis."""
    x_c = jnp.asarray(x, dtype)
    w_c = jnp.asarray(params["weight"], dtype)
    y = jnp.matmul(x_c, w_c, precision=precision)
    if "bias" in params and params["bias"] is not None:
        y = y + jnp.asarray(params["bias"], dtype)
    return y


def embedding(ids: Array, weight: Array) -> Array:
    """Lookup embeddings in float32."""
    return jnp.asarray(weight, jnp.float32)[jnp.asarray(ids, jnp.int32)]


def layer_norm(x: Array, params: Mapping[str, Array], eps: float = 1e-5) -> Array:
    """Strictly FP32 channel LayerNorm matching rvc.algorithm.normalization.LayerNorm."""
    x_f32 = jnp.asarray(x, jnp.float32)
    mean = jnp.mean(x_f32, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x_f32 - mean), axis=-1, keepdims=True)
    normed = (x_f32 - mean) * lax.rsqrt(var + eps)
    gamma = jnp.asarray(params["gamma"], jnp.float32)
    beta = jnp.asarray(params["beta"], jnp.float32)
    return normed * gamma + beta


def dropout(x: Array, rng_key: Array | None, rate: float, deterministic: bool = True) -> Array:
    """Inverted dropout when rate > 0 and not deterministic."""
    if deterministic or rate <= 0.0 or rng_key is None:
        return x
    keep_prob = 1.0 - float(rate)
    mask = jax.random.bernoulli(rng_key, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, jnp.zeros_like(x))


# ---------------------------------------------------------------------------
# PyTorch <-> JAX tensor & module conversion utilities
# ---------------------------------------------------------------------------


def conv1d_weight_torch_to_jax(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch Conv1d / ConvTranspose1d (A, B, K) to JAX (K, B, A)."""
    return np.ascontiguousarray(np.transpose(np.asarray(w, dtype=np.float32), (2, 1, 0)))


def conv1d_weight_jax_to_torch(w: np.ndarray) -> np.ndarray:
    """Convert JAX Conv1d / ConvTranspose1d (K, B, A) back to PyTorch (A, B, K)."""
    return np.ascontiguousarray(np.transpose(np.asarray(w, dtype=np.float32), (2, 1, 0)))


def conv2d_weight_torch_to_jax(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch Conv2d (O, I, Kh, Kw) to JAX HWIO (Kh, Kw, I, O)."""
    return np.ascontiguousarray(np.transpose(np.asarray(w, dtype=np.float32), (2, 3, 1, 0)))


def conv2d_weight_jax_to_torch(w: np.ndarray) -> np.ndarray:
    """Convert JAX Conv2d HWIO (Kh, Kw, I, O) back to PyTorch (O, I, Kh, Kw)."""
    return np.ascontiguousarray(np.transpose(np.asarray(w, dtype=np.float32), (3, 2, 0, 1)))


def linear_weight_torch_to_jax(w: np.ndarray) -> np.ndarray:
    """Convert PyTorch Linear (O, I) to JAX (I, O)."""
    return np.ascontiguousarray(np.asarray(w, dtype=np.float32).T)


def linear_weight_jax_to_torch(w: np.ndarray) -> np.ndarray:
    """Convert JAX Linear (I, O) back to PyTorch (O, I)."""
    return np.ascontiguousarray(np.asarray(w, dtype=np.float32).T)


def extract_conv_from_torch_state(
    state: Mapping[str, Any],
    prefix: str,
    *,
    is_2d: bool = False,
) -> dict[str, np.ndarray]:
    """Load PyTorch Conv{1d,Transpose1d,2d} (plain, weight_g/v, or parametrizations) into JAX dict."""
    conv_fn = conv2d_weight_torch_to_jax if is_2d else conv1d_weight_torch_to_jax
    out: dict[str, np.ndarray] = {}

    p_g = f"{prefix}.parametrizations.weight.original0"
    p_v = f"{prefix}.parametrizations.weight.original1"
    l_g = f"{prefix}.weight_g"
    l_v = f"{prefix}.weight_v"
    s_o = f"{prefix}.parametrizations.weight.original"
    s_u = f"{prefix}.parametrizations.weight.0._u"

    if p_g in state and p_v in state:
        out["weight_g"] = conv_fn(np.asarray(state[p_g]))
        out["weight_v"] = conv_fn(np.asarray(state[p_v]))
    elif l_g in state and l_v in state:
        out["weight_g"] = conv_fn(np.asarray(state[l_g]))
        out["weight_v"] = conv_fn(np.asarray(state[l_v]))
    elif s_o in state:
        out["weight"] = conv_fn(np.asarray(state[s_o]))
        if s_u in state:
            out["sn_u"] = np.asarray(state[s_u], dtype=np.float32).copy()
    elif f"{prefix}.weight" in state:
        out["weight"] = conv_fn(np.asarray(state[f"{prefix}.weight"]))
    else:
        raise KeyError(f"Missing weight keys for '{prefix}' in PyTorch state dict.")

    bias_key = f"{prefix}.bias"
    if bias_key in state and state[bias_key] is not None:
        out["bias"] = np.asarray(state[bias_key], dtype=np.float32).copy()
    return out


def export_conv_to_torch_state(
    state: dict[str, np.ndarray],
    prefix: str,
    params: Mapping[str, Any],
    *,
    is_2d: bool = False,
    remove_weight_norm: bool = False,
    use_parametrizations: bool = True,
) -> None:
    """Write JAX conv params into PyTorch state dict (either folded weight or parametrizations)."""
    inv_fn = conv2d_weight_jax_to_torch if is_2d else conv1d_weight_jax_to_torch
    if "weight_g" in params and "weight_v" in params:
        wg = np.asarray(params["weight_g"], dtype=np.float32)
        wv = np.asarray(params["weight_v"], dtype=np.float32)
        if remove_weight_norm:
            w = weight_norm_compose_np(wg, wv, axis=-1)
            state[f"{prefix}.weight"] = inv_fn(w)
        elif use_parametrizations:
            state[f"{prefix}.parametrizations.weight.original0"] = inv_fn(wg)
            state[f"{prefix}.parametrizations.weight.original1"] = inv_fn(wv)
        else:
            state[f"{prefix}.weight_g"] = inv_fn(wg)
            state[f"{prefix}.weight_v"] = inv_fn(wv)
    elif "weight" in params:
        state[f"{prefix}.weight"] = inv_fn(np.asarray(params["weight"], dtype=np.float32))
    else:
        raise KeyError(f"Missing JAX weight keys for '{prefix}'.")

    if "bias" in params and params["bias"] is not None:
        state[f"{prefix}.bias"] = np.asarray(params["bias"], dtype=np.float32).copy()


def extract_linear_from_torch_state(state: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    """Load PyTorch Linear state into JAX dict."""
    out = {"weight": linear_weight_torch_to_jax(np.asarray(state[f"{prefix}.weight"]))}
    if f"{prefix}.bias" in state and state[f"{prefix}.bias"] is not None:
        out["bias"] = np.asarray(state[f"{prefix}.bias"], dtype=np.float32).copy()
    return out


def export_linear_to_torch_state(state: dict[str, np.ndarray], prefix: str, params: Mapping[str, Any]) -> None:
    """Write JAX Linear params into PyTorch state dict."""
    state[f"{prefix}.weight"] = linear_weight_jax_to_torch(np.asarray(params["weight"], dtype=np.float32))
    if "bias" in params and params["bias"] is not None:
        state[f"{prefix}.bias"] = np.asarray(params["bias"], dtype=np.float32).copy()


def extract_layer_norm_from_torch_state(state: Mapping[str, Any], prefix: str) -> dict[str, np.ndarray]:
    """Load RVC channel LayerNorm (gamma, beta) into JAX dict."""
    return {
        "gamma": np.asarray(state[f"{prefix}.gamma"], dtype=np.float32).copy(),
        "beta": np.asarray(state[f"{prefix}.beta"], dtype=np.float32).copy(),
    }


def export_layer_norm_to_torch_state(state: dict[str, np.ndarray], prefix: str, params: Mapping[str, Any]) -> None:
    """Write JAX LayerNorm (gamma, beta) into PyTorch state dict."""
    state[f"{prefix}.gamma"] = np.asarray(params["gamma"], dtype=np.float32).copy()
    state[f"{prefix}.beta"] = np.asarray(params["beta"], dtype=np.float32).copy()
