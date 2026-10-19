from functools import lru_cache

import numpy as np
import torch
from torch.nn.functional import interpolate


@lru_cache(maxsize=8)
def frequency_transform(length, order, alpha):
    matrix = np.zeros((order + 1, length), dtype=np.float64)
    positions = np.arange(length)
    matrix[0] = alpha**positions
    if order and length > 1:
        matrix[1, 1:] = (1 - alpha**2) * positions[1:] * matrix[0, :-1]
    for row in range(2, order + 1):
        for column in range(1, length):
            matrix[row, column] = matrix[row - 1, column - 1] + alpha * (
                matrix[row, column - 1] - matrix[row - 1, column]
            )
    return matrix.T


def sp2mc(spectrum, order, alpha):
    cepstrum = np.fft.irfft(np.log(np.maximum(spectrum, 1e-20)), axis=-1)
    cepstrum[..., 0] *= 0.5
    return cepstrum @ frequency_transform(cepstrum.shape[-1], order, alpha)


def continuous_f0(f0):
    voiced = f0 > 0
    if not voiced.any():
        return np.zeros_like(f0)
    positions = np.arange(len(f0))
    return np.interp(positions, positions[voiced], f0[voiced])


def dilation_factors(f0, sample_rate, factors, scales, device):
    result = []
    for factor, scale in zip(factors, np.cumprod(scales)):
        dense = np.ones_like(f0, dtype=np.float64)
        voiced = f0 > 0
        dense[voiced] = sample_rate / (factor * f0[voiced])
        result.append(
            torch.tensor(np.repeat(dense, scale), dtype=torch.float32, device=device)[
                None, None
            ]
        )
    return result


def excitation(f0, sample_rate, hop, sine_amp, noise_amp):
    length = f0.shape[-1] * hop
    voiced = interpolate((f0 > 0).float(), size=length)
    cycles = (interpolate(f0.double(), size=length) / sample_rate) % 1
    signal = voiced * torch.sin(torch.cumsum(cycles, dim=-1) * (2 * np.pi)) * sine_amp
    if noise_amp:
        amplitude = voiced * noise_amp + (1 - voiced) * noise_amp / 3
        signal += torch.randn(voiced.shape, device=f0.device) * amplitude
    return signal.float()
