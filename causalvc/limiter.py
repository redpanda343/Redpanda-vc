"""Soft clipping for the final output.

Below the knee (-6 dBFS) samples pass through unchanged.  Above it, the curve
bends smoothly toward full scale (a scaled tanh joined with matching slope),
so loud peaks are rounded instead of being flattened at +/-1.0, which sounds
harsh.  No look-ahead, so no added latency.
"""

import numpy as np
import torch

KNEE_DB = -6.0
KNEE = 10 ** (KNEE_DB / 20)  # about 0.501
_HEADROOM = 1.0 - KNEE


def soft_clip(audio):
    """Soft-clip a torch tensor or numpy array; the result stays within +/-1."""
    if isinstance(audio, torch.Tensor):
        magnitude = audio.abs()
        over = torch.clamp(magnitude - KNEE, min=0.0)
        shaped = KNEE + _HEADROOM * torch.tanh(over / _HEADROOM)
        return torch.where(magnitude > KNEE, torch.sign(audio) * shaped, audio)
    audio = np.asarray(audio)
    magnitude = np.abs(audio)
    over = np.maximum(magnitude - KNEE, 0.0)
    shaped = KNEE + _HEADROOM * np.tanh(over / _HEADROOM)
    return np.where(magnitude > KNEE, np.sign(audio) * shaped, audio).astype(audio.dtype, copy=False)
