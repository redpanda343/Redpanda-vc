"""Microphone-style degradation for training inputs, on the GPU.

The clean signal stays the training target; only the model's input goes
through this. Each example independently gets a random tone curve, and with
some probability room echo, coloured background noise, a low-pass and clipping.
Everything is time-aligned with the clean signal (the echo starts at lag 0).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def degrade(wav: torch.Tensor, sr: int = 16000, strength: float = 1.0) -> torch.Tensor:
    """[B, N] -> [B, N]."""
    b, n = wav.shape
    dev = wav.device
    rand = lambda lo, hi: torch.empty(b, 1, device=dev).uniform_(lo, hi)  # noqa: E731
    coin = lambda p: (torch.rand(b, 1, device=dev) < p).float()  # noqa: E731
    tail = int(0.6 * sr)
    m = n + tail
    bins = m // 2 + 1
    freq = torch.linspace(0, sr / 2, bins, device=dev)[None]

    # Tone: a smooth random curve of up to +-8 dB over 8 control points.
    curve = torch.empty(b, 1, 8, device=dev).uniform_(-8, 8) * strength
    eq = 10 ** (F.interpolate(curve, size=bins, mode="linear", align_corners=True)[:, 0] / 20)
    # Low-pass (cheap mics, codecs): 12 dB/octave above a random corner.
    corner = rand(3000, 7500)
    lp = coin(0.25) * (1 / (1 + (freq / corner) ** 2) - 1) + 1
    # Room: decaying noise after a unit direct path.
    t = torch.arange(tail, device=dev)[None] / sr
    rir = torch.randn(b, tail, device=dev) * torch.exp(-6.9 * t / rand(0.15, 0.7))
    rir = rir / rir.pow(2).sum(1, keepdim=True).sqrt() * 10 ** (rand(-22, -4) / 20) * coin(0.5 * strength)
    rir[:, 0] = 1.0
    spec = torch.fft.rfft(wav, m) * eq * lp * torch.fft.rfft(rir, m)
    out = torch.fft.irfft(spec, m)[:, :n]

    # Background noise with a random spectral tilt, 8-40 dB below the speech.
    tilt = (freq[:, : n // 2 + 1].clamp_min(20) / 1000) ** (-rand(0.0, 1.5))
    noise = torch.fft.irfft(torch.fft.rfft(torch.randn(b, n, device=dev)) * tilt, n)
    rms = lambda x: x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-5)  # noqa: E731
    out = out + noise / rms(noise) * rms(out) * 10 ** (-rand(8, 40) / 20) * coin(0.7 * strength)

    # Clipping.
    peak = out.abs().amax(1, keepdim=True).clamp_min(1e-5)
    limit = torch.where(coin(0.1 * strength) > 0, rand(0.25, 0.8), torch.full_like(peak, 10.0)) * peak
    return torch.maximum(torch.minimum(out, limit), -limit)


@torch.no_grad()
def add_noise(wav: torch.Tensor, share: float = 0.6, silent: float = 0.12, sr: int = 48000) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-voice sounds a microphone picks up, for teaching a decoder to stay quiet through them.

    [B, N] clean -> (input, target). For `share` of the examples the input gains steady background
    noise and/or short bursts (clicks, snaps, claps, thumps) while the target stays clean; for `silent`
    of them the voice is removed as well, so the input is noise alone and the target is silence."""
    b, n = wav.shape
    dev = wav.device
    inp, target = wav.clone(), wav.clone()
    freq = torch.fft.rfftfreq(n, 1 / sr, device=dev).clamp_min(20)
    u = lambda lo, hi: float(torch.empty(1).uniform_(lo, hi))  # noqa: E731
    for i in range(b):
        r = u(0, 1)
        if r >= share:
            continue
        if r < silent:
            inp[i], target[i] = 0.0, 0.0
        if r < silent or u(0, 1) < 0.6:  # steady noise with a random tilt, far below ordinary speech
            x = torch.fft.irfft(torch.fft.rfft(torch.randn(n, device=dev)) * freq ** (-u(0.0, 1.5)), n)
            inp[i] += x / x.pow(2).mean().sqrt() * 10 ** (u(-65, -35) / 20)
        if u(0, 1) < 0.75:
            for _ in range(int(u(1, 6))):  # bursts: 3-90 ms of band-limited noise with a sharp start and a decaying tail
                k = int(u(0.003, 0.09) * sr)
                at = int(u(0, n - k))
                burst = torch.randn(k, device=dev) * torch.exp(-torch.arange(k, device=dev) / (k / u(3, 8)))
                f = torch.fft.rfftfreq(k, 1 / sr, device=dev)
                lo = 10 ** u(1.6, 3.5)
                spec = torch.fft.rfft(burst)
                spec[(f < lo) | (f > min(lo * u(2, 10), 14000.0))] = 0
                burst = torch.fft.irfft(spec, k)
                inp[i, at : at + k] += burst / burst.abs().max().clamp_min(1e-6) * 10 ** (u(-35, -3) / 20)
    return inp.clamp(-1.0, 1.0), target


@torch.no_grad()
def add_silent_transients(src: torch.Tensor, target: torch.Tensor, sr: int = 48000) -> torch.Tensor:
    """For already-silent target examples in a batch, keep half as pure digital silence and give half a faint
    room floor plus 1-2 short impulsive clicks/thumps (2-18 ms) while `target` stays 0, so dead air and brief
    mic clicks come out silent without dulling whispers or laughs on voiced/breathy examples."""
    rms = target.pow(2).mean(dim=-1).sqrt()
    silent_idx = torch.nonzero(rms < 1e-4, as_tuple=False).flatten().tolist()
    if not silent_idx:
        return src
    out = src.clone()
    n = out.shape[-1]
    dev = out.device
    freq = torch.fft.rfftfreq(n, 1 / sr, device=dev).clamp_min(20)
    u = lambda lo, hi: float(torch.empty(1).uniform_(lo, hi))  # noqa: E731
    for i in silent_idx:
        if u(0, 1) < 0.45:  # pure digital silence input -> exact digital silence output
            out[i].zero_()
            continue
        x = torch.fft.irfft(torch.fft.rfft(torch.randn(n, device=dev)) * freq ** (-u(0.3, 1.4)), n)
        out[i] = x / x.pow(2).mean().sqrt().clamp_min(1e-6) * 10 ** (u(-72, -52) / 20)
        for _ in range(int(u(1, 3))):
            k = int(u(0.002, 0.018) * sr)
            at = int(u(0, max(1, n - k)))
            burst = torch.randn(k, device=dev) * torch.exp(-torch.arange(k, device=dev) / max(1.0, k / u(4, 9)))
            out[i, at : at + k] += burst / burst.abs().max().clamp_min(1e-6) * 10 ** (u(-32, -8) / 20)
    return out.clamp(-1.0, 1.0)


@torch.no_grad()
def colour(wav: torch.Tensor, share: float = 0.5, sr: int = 48000) -> torch.Tensor:
    """Microphone and voice colour on a decoder's input, for teaching it to give the same voice whatever
    the microphone: [B, N] -> [B, N], for `share` of the examples (the rest pass unchanged); the clean
    signal stays the target.

    Each coloured example gets a smooth random tone curve; then, each by chance, a treble shelf (a bright
    or dull microphone), extra breath (high-band noise that follows the voice's own level, as in a
    breathy voice or a close bright microphone), a low-pass (cheap microphones, codecs) and light room
    reflections. Everything stays time-aligned with the target."""
    b, n = wav.shape
    dev = wav.device
    pick = (torch.rand(b, 1, device=dev) < share).float()
    if not pick.any():
        return wav
    rand = lambda lo, hi: torch.empty(b, 1, device=dev).uniform_(lo, hi)  # noqa: E731
    coin = lambda p: (torch.rand(b, 1, device=dev) < p).float()  # noqa: E731
    tail = int(0.25 * sr)
    m = n + tail
    freq = torch.fft.rfftfreq(m, 1 / sr, device=dev)[None]
    octave = torch.log2(freq.clamp_min(50) / 50)  # 0 at 50 Hz, about 8.9 at 24 kHz

    # Tone: up to +-6 dB over 7 smooth control points spread by octave.
    points = torch.empty(b, 1, 7, device=dev).uniform_(-6, 6)
    curve = F.interpolate(points, size=1024, mode="linear", align_corners=True)[:, 0]
    at = (octave / octave.max() * 1023).round().long().expand(b, -1)
    gain_db = torch.gather(curve, 1, at)
    # Treble shelf above 2-7 kHz, -12 to +12 dB: half of the coloured examples.
    corner = rand(2000, 7000)
    shelf = torch.sigmoid(4 * torch.log2(freq.clamp_min(1) / corner))
    gain_db = gain_db + shelf * rand(-12, 12) * coin(0.5)
    spec = torch.fft.rfft(wav, m) * 10 ** (gain_db / 20)
    # Low-pass, 12 dB/octave above 3.5-9 kHz: a fifth of them.
    spec = spec * (coin(0.2) * (1 / (1 + (freq / rand(3500, 9000)) ** 2) - 1) + 1)
    # Room: a short decaying tail after the direct sound, 12-30 dB down: a quarter of them.
    t = torch.arange(tail, device=dev)[None] / sr
    rir = torch.randn(b, tail, device=dev) * torch.exp(-6.9 * t / rand(0.05, 0.25))
    rir = rir / rir.pow(2).sum(1, keepdim=True).sqrt() * 10 ** (rand(-30, -12) / 20) * coin(0.25)
    rir[:, 0] = 1.0
    out = torch.fft.irfft(spec * torch.fft.rfft(rir, m), m)[:, :n]

    # Breath: noise above 3-6 kHz shaped by the voice's own 10 ms level, 15-35 dB under it: a third of them.
    hop = sr // 100
    level = F.avg_pool1d(wav.pow(2)[:, None], hop, hop, ceil_mode=True)[:, 0].sqrt()
    envelope = F.interpolate(level[:, None], size=n, mode="linear", align_corners=False)[:, 0]
    nf = torch.fft.rfftfreq(n, 1 / sr, device=dev)[None]
    noise = torch.fft.irfft(torch.fft.rfft(torch.randn(b, n, device=dev)) * torch.sigmoid(4 * torch.log2(nf.clamp_min(1) / rand(3000, 6000))), n)
    noise = noise / noise.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)
    out = out + noise * envelope * 10 ** (-rand(15, 35) / 20) * coin(0.33)

    # Same overall level as before, so only the colour changes.
    rms = lambda x: x.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-6)  # noqa: E731
    out = out * rms(wav) / rms(out)
    return (pick * out + (1 - pick) * wav).clamp(-1.0, 1.0)
