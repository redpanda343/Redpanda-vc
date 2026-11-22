"""Batched RMVPE pitch on GPU. Whole-signal and non-causal: this is the reference
tracker used to label training audio, and the teacher of the causal tracker.
The weights are fetched on first use (see causalvc.weights).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def voicing_from_peak(peak: torch.Tensor, thr: float) -> torch.Tensor:
    """Smooth periodicity confidence in 0..1 from RMVPE salience peak: 0 below `thr`,
    rising smoothly across breathy/half-voiced laughs and fry (0.03..0.55) to 1 on clean vowels."""
    v = ((peak - thr) / max(1e-4, 0.55 - thr)).clamp(0.0, 1.0).sqrt()
    return torch.where(peak > thr, v, torch.zeros_like(v))


class Rmvpe:
    hop = 160  # samples at 16 kHz -> 100 frames per second

    def __init__(self, device: str, threshold: float = 0.03):
        from causalvc.rmvpe_model import E2E, N_MELS, MelSpectrogram
        from causalvc.weights import ensure

        model = E2E(4, 1, (2, 2))
        model.load_state_dict(torch.load(ensure("rmvpe.pt"), map_location="cpu", weights_only=True))
        self.model = model.eval().requires_grad_(False).to(device)
        self.mel = MelSpectrogram(N_MELS, 16000, 1024, 160, None, 30, 8000).to(device)
        self.thr = threshold
        cents = 20 * torch.arange(360, dtype=torch.float32) + 1997.3794084376191
        self.cents = F.pad(cents, (4, 4)).to(device)

    @torch.no_grad()
    def salience(self, wav16: torch.Tensor) -> torch.Tensor:
        """[B, N] at 16 kHz -> pitch salience [B, N // 160 + 1, 360] in 0..1."""
        mel = self.mel(wav16.float(), center=True)
        n = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n - 1) // 32 + 1) - n), mode="reflect")
        return self.model(mel)[:, :n].float()

    def decode_with_voicing(self, sal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Salience [B, T, 360] -> (F0 in Hz [B, T], voicing confidence in 0..1 [B, T])."""
        peak, centre = sal.max(dim=-1)
        idx = centre[..., None] + torch.arange(9, device=sal.device)  # window in padded coordinates
        local = F.pad(sal, (4, 4)).gather(-1, idx)
        cents = (local * self.cents[idx]).sum(-1) / local.sum(-1).clamp_min(1e-9)
        f0 = 10.0 * 2.0 ** (cents / 1200.0)
        return torch.where(peak > self.thr, f0, torch.zeros_like(f0)), voicing_from_peak(peak, self.thr)

    def decode(self, sal: torch.Tensor) -> torch.Tensor:
        """Salience [B, T, 360] -> F0 in Hz [B, T], 0 where unvoiced."""
        return self.decode_with_voicing(sal)[0]

    @torch.no_grad()
    def pitch_and_voicing(self, wav16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, N] at 16 kHz -> (F0 in Hz, voicing in 0..1), each [B, N // 160 + 1]."""
        return self.decode_with_voicing(self.salience(wav16))

    @torch.no_grad()
    def __call__(self, wav16: torch.Tensor) -> torch.Tensor:
        """[B, N] at 16 kHz -> F0 in Hz [B, N // 160 + 1], 0 where unvoiced.
        Frame j is centred on sample 160 * j."""
        return self.decode(self.salience(wav16))
