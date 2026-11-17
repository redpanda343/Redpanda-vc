"""Strictly FP32 STFT, Mel Spectrogram, and MultiScaleMelSpectrogramLoss in JAX."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from rvc.jax.ops import Array, stft_complex

_HANN_CACHE: dict[int, np.ndarray] = {}
_MEL_BANK_CACHE: dict[tuple[int, int, int, float, float, bool], np.ndarray] = {}


def _hz_to_mel_htk(freqs: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freqs, dtype=np.float64) / 700.0)


def _mel_to_hz_htk(mels: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mels, dtype=np.float64) / 2595.0) - 1.0)


def hann_window_np(win_length: int) -> np.ndarray:
    """Periodic Hann window matching torch.hann_window(win_length)."""
    n = int(win_length)
    cached = _HANN_CACHE.get(n)
    if cached is not None:
        return cached
    if n <= 1:
        win = np.ones((max(1, n),), dtype=np.float32)
    else:
        idx = np.arange(n, dtype=np.float64)
        win = (0.5 * (1.0 - np.cos(2.0 * math.pi * idx / float(n)))).astype(np.float32)
    _HANN_CACHE[n] = win
    return win


def _hz_to_mel_slaney(freqs: np.ndarray | float) -> np.ndarray:
    f_0 = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_0) / f_sp
    logstep = math.log(6.4) / 27.0
    f = np.asarray(freqs, dtype=np.float64)
    return np.where(
        f >= min_log_hz,
        min_log_mel + np.log(np.maximum(f, min_log_hz) / min_log_hz) / logstep,
        (f - f_0) / f_sp,
    )


def _mel_to_hz_slaney(mels: np.ndarray | float) -> np.ndarray:
    f_0 = 0.0
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_0) / f_sp
    logstep = math.log(6.4) / 27.0
    m = np.asarray(mels, dtype=np.float64)
    return np.where(
        m >= min_log_mel,
        min_log_hz * np.exp(logstep * (m - min_log_mel)),
        f_0 + f_sp * m,
    )


def mel_filterbank_np(
    sample_rate: int,
    n_fft: int,
    num_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
) -> np.ndarray:
    """Pure-NumPy Slaney or HTK mel filterbank (num_mels, 1 + n_fft // 2) matching librosa.filters.mel."""
    fmax_val = float(sample_rate) / 2.0 if fmax is None else float(fmax)
    key = (int(sample_rate), int(n_fft), int(num_mels), float(fmin), fmax_val, bool(htk))
    cached = _MEL_BANK_CACHE.get(key)
    if cached is not None:
        return cached

    n_freqs = 1 + int(n_fft) // 2
    weights = np.zeros((int(num_mels), n_freqs), dtype=np.float64)
    fftfreqs = np.fft.rfftfreq(n=int(n_fft), d=1.0 / float(sample_rate))

    hz_to_mel = _hz_to_mel_htk if htk else _hz_to_mel_slaney
    mel_to_hz = _mel_to_hz_htk if htk else _mel_to_hz_slaney
    min_mel = float(hz_to_mel(float(fmin)))
    max_mel = float(hz_to_mel(fmax_val))
    mel_f = mel_to_hz(np.linspace(min_mel, max_mel, int(num_mels) + 2))

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    for i in range(int(num_mels)):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : int(num_mels) + 2] - mel_f[: int(num_mels)])
    weights *= enorm[:, np.newaxis]
    out = weights.astype(np.float32)
    _MEL_BANK_CACHE[key] = out
    return out


def compute_window_length(n_mels: int, sample_rate: int) -> int:
    """Compute RefineGAN STFT window length matching rvc.train.mel_processing.compute_window_length."""
    f_min = 0.0
    f_max = float(sample_rate) / 2.0
    window_length_seconds = 8.0 * float(n_mels) / (f_max - f_min)
    window_length = int(window_length_seconds * float(sample_rate))
    return 2 ** (window_length.bit_length() - 1)


def spectrogram_jax(
    y: Array,
    n_fft: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
) -> Array:
    """Compute magnitude STFT matching spectrogram_torch, returned in NWC (B, T_frames, F)."""
    pad = int((n_fft - hop_size) / 2)
    win = hann_window_np(win_size)
    # Match spectrogram_torch(y, center=False) which manually reflect-pads before center=False stft.
    total_pad = pad + (n_fft // 2 if center else 0)
    real, imag = stft_complex(y, n_fft, hop_size, win_size, win, pad_left=total_pad, pad_right=total_pad)
    spec = jnp.sqrt(real * real + imag * imag + 2e-7)  # (B, F, TT)
    return jnp.transpose(spec, (0, 2, 1))


def spectrogram_jax_padded(
    y: Array,
    wave_lengths: Array,
    n_fft: int,
    hop_size: int,
    win_size: int,
    max_frames: int | None = None,
) -> Array:
    """Exact variable-length magnitude STFT on bucket-padded audio (B, T_wav, 1) -> (B, TT, F)."""
    pad = int((n_fft - hop_size) // 2)
    y_bt = jnp.asarray(y, jnp.float32)
    if y_bt.ndim == 3:
        y_bt = y_bt[..., 0]
    lengths = jnp.asarray(wave_lengths, jnp.int32)

    def _pad_one(wav: Array, w_len: Array) -> Array:
        left = wav[pad:0:-1]
        ext = jnp.concatenate([left, wav, jnp.zeros((pad,), dtype=jnp.float32)], axis=0)
        start = jnp.maximum(w_len - 1, 0)
        seg = lax.dynamic_slice(ext, (start,), (pad,))[::-1]
        return lax.dynamic_update_slice(ext, seg, (pad + w_len,))

    y_padded = jax.vmap(_pad_one, in_axes=(0, 0))(y_bt, lengths)
    win = hann_window_np(win_size)
    real, imag = stft_complex(y_padded, n_fft, hop_size, win_size, win, pad_left=0, pad_right=0)
    spec = jnp.transpose(jnp.sqrt(real * real + imag * imag + 2e-7), (0, 2, 1))  # (B, TT, F)
    if max_frames is not None:
        spec = spec[:, : int(max_frames), :]
    valid_frames = (lengths // int(hop_size))[:, None, None]
    frame_idx = jnp.arange(spec.shape[1], dtype=jnp.int32)[None, :, None]
    return jnp.where(frame_idx < valid_frames, spec, 0.0)


def spec_to_mel_jax(
    spec_nwc: Array,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    fmin: float,
    fmax: float | None,
) -> Array:
    """Convert NWC magnitude spectrogram (B, TT, F) to log mel spectrogram (B, TT, num_mels) in FP32."""
    basis = jnp.asarray(mel_filterbank_np(sample_rate, n_fft, num_mels, fmin=fmin, fmax=fmax), jnp.float32)
    spec_f32 = jnp.asarray(spec_nwc, jnp.float32)
    melspec = jnp.matmul(spec_f32, basis.T, precision=lax.Precision.HIGHEST)
    return jnp.log(melspec + 3e-4)


def mel_spectrogram_jax(
    y: Array,
    n_fft: int,
    num_mels: int,
    sample_rate: int,
    hop_size: int,
    win_size: int,
    fmin: float,
    fmax: float | None,
    center: bool = False,
) -> Array:
    """Strictly FP32 log-mel spectrogram matching mel_spectrogram_torch in NWC (B, TT, num_mels)."""
    spec = spectrogram_jax(y, n_fft=n_fft, hop_size=hop_size, win_size=win_size, center=center)
    return spec_to_mel_jax(spec, n_fft=n_fft, num_mels=num_mels, sample_rate=sample_rate, fmin=fmin, fmax=fmax)


@dataclass(frozen=True)
class MultiScaleMelLossConfig:
    sample_rate: int = 24000
    n_mels: tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320)
    window_lengths: tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048)


def multi_scale_mel_loss(real: Array, fake: Array, cfg: MultiScaleMelLossConfig) -> Array:
    """Strictly FP32 MultiScaleMelSpectrogramLoss matching rvc.train.mel_processing."""
    log10_base = math.log(10.0)
    real_f32 = jnp.asarray(real, jnp.float32)
    fake_f32 = jnp.asarray(fake, jnp.float32)
    if real_f32.ndim == 3:
        real_f32 = real_f32[..., 0]
    if fake_f32.ndim == 3:
        fake_f32 = fake_f32[..., 0]

    both = jnp.concatenate([real_f32, fake_f32], axis=0)
    batch = real_f32.shape[0]
    total_loss = jnp.array(0.0, dtype=jnp.float32)

    for n_mels, win_len in zip(cfg.n_mels, cfg.window_lengths):
        pad = win_len // 2
        hop = win_len // 4
        win = hann_window_np(win_len)
        real_c, imag_c = stft_complex(both, win_len, hop, win_len, win, pad_left=pad, pad_right=pad)
        mag = jnp.sqrt(real_c * real_c + imag_c * imag_c + 1e-6)  # (2B, F, TT)
        basis = jnp.asarray(mel_filterbank_np(cfg.sample_rate, win_len, n_mels, fmin=0.0, fmax=None), jnp.float32)
        mels = jnp.matmul(basis[None, :, :], mag, precision=lax.Precision.HIGHEST)
        log_mels = jnp.log(jnp.maximum(mels, 1e-5)) / log10_base
        total_loss = total_loss + jnp.mean(jnp.abs(log_mels[:batch] - log_mels[batch:]))

    return total_loss
