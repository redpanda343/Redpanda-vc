import librosa
import numpy as np
import torch

from .constants import *


def to_local_average_f0(hidden, center=None, thred=0.03):
    idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]
    idx_cents = idx * 20 + CONST
    if center is None:
        center = torch.argmax(hidden, dim=2, keepdim=True)
    start = torch.clip(center - 4, min=0)
    end = torch.clip(center + 5, max=N_CLASS)
    idx_mask = (idx >= start) & (idx < end)
    weights = hidden * idx_mask
    product_sum = torch.sum(weights * idx_cents, dim=2)
    weight_sum = torch.sum(weights, dim=2)
    cents = product_sum / (weight_sum + (weight_sum == 0))
    f0 = 10 * 2 ** (cents / 1200)
    uv = hidden.max(dim=2)[0] < thred
    f0 = f0 * ~uv
    return f0.squeeze(0).cpu().numpy()


def to_viterbi_f0(hidden, thred=0.03):
    if not hasattr(to_viterbi_f0, 'transition'):
        xx, yy = np.meshgrid(range(N_CLASS), range(N_CLASS))
        transition = np.maximum(30 - abs(xx - yy), 0)
        transition = transition / transition.sum(axis=1, keepdims=True)
        to_viterbi_f0.transition = transition

    prob = hidden.squeeze(0).cpu().numpy()
    prob = prob.T
    prob = prob / prob.sum(axis=0)

    path = librosa.sequence.viterbi(prob, to_viterbi_f0.transition).astype(np.int64)
    center = torch.from_numpy(path).unsqueeze(0).unsqueeze(-1).to(hidden.device)

    return to_local_average_f0(hidden, center=center, thred=thred)
