import math

import numpy as np
import torch
from torch.nn import functional as F

from rvc.rectified.mel import LogMel, denormalize_mel, normalize_mel
from rvc.rectified.vocoder import MEL_KEYS

HEADROOM = math.log(2.0)
MIN_VOICED_FRAMES = 4
EMPTY_LOG_MEL = math.log(1e-5) + 1.0
_MELS = {}


def _log_mel(data, device):
    key = (str(device),) + tuple(float(data[name]) for name in MEL_KEYS)
    if key not in _MELS:
        _MELS[key] = LogMel.from_config(data).to(device)
    return _MELS[key]


def reference_mel(audio, data, frames):
    hop = int(data['hop_length'])
    audio = audio.float().view(1, -1)
    audio = F.pad(audio, (0, max(0, (frames + 1) * hop - audio.shape[-1])))[:, :(frames + 1) * hop]
    return _log_mel(data, audio.device)(audio)[..., :frames]


def source_band(reference):
    active = torch.quantile(reference[0], 0.9, dim=-1) > EMPTY_LOG_MEL
    bins = torch.arange(active.numel(), device=reference.device)
    top = bins[active].max() if active.any() else bins[-1]
    return (bins <= top).view(1, -1, 1)


def guard_unvoiced(mel, f0, audio, voiced, data):
    frames = mel.shape[-1]
    voiced = torch.as_tensor(np.asarray(voiced), device=mel.device).float().view(1, 1, -1)
    voiced = F.pad(voiced, (0, max(0, frames - voiced.shape[-1])))[..., :frames]
    kept = F.max_pool1d(voiced, 3, 1, 1)
    weight = 1 - F.avg_pool1d(kept, 3, 1, 1, count_include_pad=False)
    generated = denormalize_mel(mel.float(), data)
    reference = reference_mel(audio.to(mel.device), data, frames)
    band = source_band(reference)
    gain = generated.new_zeros(())
    core = voiced[0, 0] > 0
    if int(core.sum()) >= MIN_VOICED_FRAMES:
        floor = torch.finfo(generated.dtype).min
        level = lambda value: torch.logsumexp(value.masked_fill(~band, floor), dim=1)[0]
        gain = (level(generated) - level(reference))[core].median()
    excess = F.relu(generated - (reference + gain + HEADROOM)) * band
    guarded = normalize_mel(generated - weight * excess, data).to(mel.dtype)
    return guarded, f0 * kept[:, 0].to(f0.dtype)
