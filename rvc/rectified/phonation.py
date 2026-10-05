import math
from functools import lru_cache

import torch
from torch.nn import functional as F


PHONATION_CHANNELS = 4
PHONATION_VERSION = 1
ANALYSIS_RATE = 8000
WINDOW_SECONDS = 0.08


@lru_cache(maxsize=8)
def _resampler(sample_rate, device):
    from torchaudio.transforms import Resample

    return Resample(sample_rate, ANALYSIS_RATE).to(device)


@torch.no_grad()
def phonation_features(audio, sample_rate, f0, frames, hop):
    audio = audio.float()
    if audio.ndim != 2 or f0.shape != (audio.shape[0], frames) or frames < 1:
        raise ValueError('Phonation requires batched audio and frame-aligned pitch.')
    window = int(round(WINDOW_SECONDS * ANALYSIS_RATE))
    waveform = _resampler(sample_rate, audio.device)(audio)
    length = waveform.shape[-1]
    ends = torch.round(torch.arange(frames, device=audio.device) * hop * ANALYSIS_RATE / sample_rate).long()
    indices = ends[:, None] + torch.arange(window, device=audio.device)[None]
    padding = window - 1
    waveform = F.pad(waveform, (padding, max(0, int(math.ceil(frames * hop * ANALYSIS_RATE / sample_rate)) - length)))
    windows = waveform[:, indices.clamp_max(waveform.shape[-1] - 1)]
    windows = windows - windows.mean(dim=-1, keepdim=True)
    audible = windows.square().mean(dim=-1) > 1e-8
    windows = windows / windows.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    energy = windows.square()
    spectrum = torch.fft.rfft(windows, n=2 ** math.ceil(math.log2(window * 2)))
    correlation = torch.fft.irfft(spectrum.abs().square(), n=2 ** math.ceil(math.log2(window * 2)))[..., :window]
    cumulative = F.pad(energy.cumsum(dim=-1), (1, 0))
    lag = torch.arange(window, device=audio.device)
    norm = ((cumulative[..., window - lag] * (cumulative[..., -1:] - cumulative[..., lag])).clamp_min(1e-12)).sqrt()
    correlation = (correlation / norm).clamp(-1, 1)
    low, high = int(ANALYSIS_RATE / 1000), min(window // 2, int(ANALYSIS_RATE / 30))
    periodicity = correlation[..., low:high + 1].amax(dim=-1).clamp(0, 1)
    period = ANALYSIS_RATE / f0.float().clamp_min(30)
    lags = period[..., None] * torch.arange(1, 5, device=audio.device)
    left = lags.floor().long().clamp(1, window - 2)
    weight = (lags - left).clamp(0, 1)
    cycles = correlation.gather(-1, left) * (1 - weight) + correlation.gather(-1, left + 1) * weight
    differences = (cycles[..., 1:] - cycles[..., :1]).clamp(0, 1)
    differences = differences * ((f0 > 0)[..., None] & (lags[..., 1:] < window // 2))
    return torch.cat((periodicity[..., None], differences), dim=-1) * audible[..., None]


def initialize_phonation_weights(weights, model):
    result = dict(weights)
    for name, value in model.state_dict().items():
        if name.startswith('encoder.phonation.') and name not in result:
            result[name] = value.detach().clone()
    return result
