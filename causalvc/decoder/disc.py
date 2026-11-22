"""Discriminators and losses for decoder training: multi-period (HiFi-GAN) and
multi-resolution spectrogram (UnivNet/BigVGAN), least-squares GAN, feature
matching and a multi-scale mel loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.parametrizations import weight_norm


class PeriodD(nn.Module):
    def __init__(self, period: int):
        super().__init__()
        self.p = period
        ch = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(
            weight_norm(nn.Conv2d(ch[i], ch[i + 1], (5, 1), (3, 1), (2, 0))) for i in range(4)
        )
        self.convs.append(weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, (2, 0))))
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x: torch.Tensor):
        b, c, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), mode="reflect")
        x = x.view(b, c, -1, self.p)
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class SpecD(nn.Module):
    def __init__(self, n_fft: int, hop: int, win: int, ch: int = 32):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer("window", torch.hann_window(win), persistent=False)
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(1, ch, (3, 9), padding=(1, 4)))]
            + [weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
            + [weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=1))]
        )
        self.post = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=1))

    def forward(self, x: torch.Tensor):
        with full_precision():
            spec = torch.stft(x.squeeze(1).float(), self.n_fft, self.hop, self.win, self.window, return_complex=True).abs()
        x = spec[:, None]
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class BandD(nn.Module):
    """Complex-spectrogram discriminator split into frequency bands (as in DAC):
    it sees phase as well as magnitude, and each band has its own weights."""

    BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))

    def __init__(self, n_fft: int, ch: int = 32):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        bins = n_fft // 2 + 1
        self.edges = [(int(a * bins), int(b * bins)) for a, b in self.BANDS]
        self.bands = nn.ModuleList(
            nn.ModuleList(
                [weight_norm(nn.Conv2d(2, ch, (3, 9), padding=(1, 4)))]
                + [weight_norm(nn.Conv2d(ch, ch, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
                + [weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=1))]
            )
            for _ in self.BANDS
        )
        self.post = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=1))

    def forward(self, x: torch.Tensor):
        with full_precision():
            spec = torch.stft(x.squeeze(1).float(), self.n_fft, self.n_fft // 4, window=self.window, return_complex=True)
            spec = torch.view_as_real(spec).permute(0, 3, 2, 1)  # [B, 2, frames, bins]
        fmap, outs = [], []
        for (a, b), convs in zip(self.edges, self.bands):
            h = spec[..., a:b]
            for conv in convs:
                h = F.leaky_relu(conv(h), 0.1)
                fmap.append(h)
            outs.append(h)
        x = self.post(torch.cat(outs, dim=-1))
        fmap.append(x)
        return x.flatten(1), fmap


class Discriminators(nn.Module):
    def __init__(self, version: int = 1):
        super().__init__()
        spec = (
            [SpecD(*r) for r in ((2048, 240, 1200), (4096, 480, 2400), (1024, 100, 480))]
            if version == 1
            else [BandD(n) for n in (2048, 1024, 512)]
        )
        extra = [PeriodD(p) for p in (17, 23)] if version >= 3 else []
        self.ds = nn.ModuleList([PeriodD(p) for p in (2, 3, 5, 7, 11)] + spec + extra)

    def forward(self, x: torch.Tensor):
        return [d(x) for d in self.ds]


def full_precision():
    """Turns off mixed precision (train.py --bf16) for a spectrogram: the FFT needs float32."""
    return torch.autocast("cuda", enabled=False)


def d_loss(real, fake) -> torch.Tensor:
    return sum(((1 - r.float()) ** 2).mean() + (f.float() ** 2).mean() for (r, _), (f, _) in zip(real, fake))


def g_loss(real, fake) -> tuple[torch.Tensor, torch.Tensor]:
    adv = sum(((1 - f.float()) ** 2).mean() for f, _ in fake)
    fm = sum(F.l1_loss(a, b.detach()) for (_, fr), (_, ff) in zip(real, fake) for a, b in zip(ff, fr))
    return adv, fm


class MelLoss(nn.Module):
    def __init__(self, sr: int = 48000, f_max: float | None = None):
        super().__init__()
        self.mels = nn.ModuleList(
            torchaudio.transforms.MelSpectrogram(sr, n_fft=n, hop_length=n // 4, n_mels=m, f_max=f_max, power=1.0)
            for n, m in ((512, 40), (1024, 80), (2048, 128))
        )

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for mel in self.mels:
            a, b = mel(fake.squeeze(1)), mel(real.squeeze(1))
            loss = loss + F.l1_loss(torch.log(a.clamp_min(1e-5)), torch.log(b.clamp_min(1e-5)))
        return loss / len(self.mels)


class SpecLoss(nn.Module):
    """L1 on log-magnitude linear-frequency STFTs. Unlike the mel loss, its bins
    stay narrower than the harmonic spacing at every frequency, so noise between
    high harmonics is penalised directly."""

    def __init__(self, ffts: tuple[int, ...] = (1024, 2048, 4096), sr: int = 48000, f_max: float | None = None):
        super().__init__()
        self.ffts, self.sr, self.f_max = ffts, sr, f_max
        for n in ffts:
            self.register_buffer(f"w{n}", torch.hann_window(n), persistent=False)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for n in self.ffts:
            a, b = (torch.stft(x.squeeze(1), n, n // 4, window=getattr(self, f"w{n}"), return_complex=True).abs() for x in (fake, real))
            if self.f_max is not None:
                top = max(8, int(n * self.f_max / self.sr) + 1)
                a, b = a[:, :top], b[:, :top]
            loss = loss + F.l1_loss(torch.log(a.clamp_min(1e-4)), torch.log(b.clamp_min(1e-4)))
        return loss / len(self.ffts)


class EnergyLoss(nn.Module):
    """L1 on the log of each band's power averaged over the segment. The log-spectrum
    losses compare frame by frame, where the cheapest answer for a noise-like band is a
    level a few dB under its true power; averaging power before the log has no such bias,
    so this holds the long-term tone of the output to the recording's."""

    def __init__(self, sr: int = 48000, n_fft: int = 2048, bands: int = 64, f_max: float | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.scale = torchaudio.transforms.MelScale(bands, sr, f_max=f_max, n_stft=n_fft // 2 + 1, norm=None)

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        a, b = (
            self.scale(torch.stft(x.squeeze(1), self.n_fft, self.n_fft // 4, window=self.window, return_complex=True).abs().pow(2)).mean(-1)
            for x in (fake, real)
        )
        return F.l1_loss(torch.log(a.clamp_min(1e-7)), torch.log(b.clamp_min(1e-7)))
