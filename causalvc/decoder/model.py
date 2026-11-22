"""Causal source-filter decoder: content + pitch + loudness + speaker -> 48 kHz audio.

Every layer only looks backwards, so the waveform it produces for a frame is
final as soon as that frame's features exist: streamed output equals whole-file
output, whatever the block size. The decoder itself adds no lookahead.

Frame clock is 100 Hz (480 samples). Content arrives at 50 Hz and is repeated.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

SR = 48000
FRAME = 480  # samples per 100 Hz frame


def min_phase_lpf(cutoff: float, taps: int = 12, beta: float = 6.0) -> torch.Tensor:
    """Causal minimum-phase Kaiser-windowed sinc low-pass filter (`cutoff` in 0..1 of Nyquist),
    returned in Conv1d correlation order (index -1 multiplies the newest sample)."""
    n_lin = 4 * taps - 1
    m = torch.arange(n_lin, dtype=torch.float64) - (n_lin - 1) / 2
    h = cutoff * torch.sinc(cutoff * m)
    b_t = torch.tensor(beta, dtype=torch.float64)
    w = torch.i0(b_t * torch.sqrt((1.0 - (2.0 * m / (n_lin - 1)) ** 2).clamp_min(0.0))) / torch.i0(b_t)
    h = (h * w) / (h * w).sum()
    H = torch.fft.fft(h, n=1024).abs().clamp_min(1e-7)
    cep = torch.fft.ifft(torch.log(H)).real
    win = torch.zeros(1024, dtype=torch.float64)
    win[0] = win[512] = 1.0
    win[1:512] = 2.0
    mp = torch.fft.ifft(torch.exp(torch.fft.fft(cep * win))).real[:taps]
    return (mp / mp.sum()).flip(0).float()


class CausalLPF(nn.Module):
    """Fixed causal depthwise low-pass filter for suppressing transposed-conv mirror images."""

    def __init__(self, channels: int, cutoff: float, taps: int = 12):
        super().__init__()
        self.channels, self.pad = channels, taps - 1
        h = min_phase_lpf(cutoff, taps=taps)
        self.register_buffer("weight", h[None, None, :].repeat(channels, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(F.pad(x, (self.pad, 0)), self.weight, groups=self.channels)


class AntiAliasAct(nn.Module):
    """Causal 2x polyphase oversampled LeakyReLU: suppresses nonlinear harmonic folding in-band."""

    def __init__(self, channels: int, taps: int = 12):
        super().__init__()
        assert taps % 2 == 0
        self.channels, self.pad_up, self.pad_dn = channels, taps // 2 - 1, taps - 2
        h = min_phase_lpf(0.46, taps=taps, beta=6.5)
        up = torch.stack([2.0 * h[1::2], 2.0 * h[0::2]], dim=0)[:, None, :].repeat(channels, 1, 1)
        dn = h[None, None, :].repeat(channels, 1, 1)
        self.register_buffer("w_up", up, persistent=False)
        self.register_buffer("w_dn", dn, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        u = F.conv1d(F.pad(x, (self.pad_up, 0)), self.w_up, groups=c).view(b, c, 2, t).transpose(2, 3).reshape(b, c, 2 * t)
        u = F.leaky_relu(u, 0.1)
        return F.conv1d(F.pad(u, (self.pad_dn, 0)), self.w_dn, stride=2, groups=c)


class CConv(nn.Module):
    """Causal 1-D convolution (pads on the left only)."""

    def __init__(self, cin: int, cout: int, k: int, dilation: int = 1, groups: int = 1, norm: bool = True, ahead: int = 0):
        super().__init__()
        self.pad, self.ahead = (k - 1) * dilation, ahead  # `ahead` extra taps look at future steps
        conv = nn.Conv1d(cin, cout, k + ahead, dilation=dilation, groups=groups)
        self.conv = weight_norm(conv) if norm else conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.pad, self.ahead)))


class CUp(nn.Module):
    """Causal transposed convolution: output hop t depends on frames t-1 and t."""

    def __init__(self, cin: int, cout: int, stride: int, antialias: bool = False):
        super().__init__()
        self.stride = stride
        self.conv = weight_norm(nn.ConvTranspose1d(cin, cout, 2 * stride, stride))
        self.lpf = CausalLPF(cout, min(0.46, 0.96 / stride), taps=16 if stride >= 4 else 12) if antialias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)[..., : x.shape[-1] * self.stride]
        return self.lpf(y) if self.lpf is not None else y


class ResBlock(nn.Module):
    def __init__(self, ch: int, k: int, dilations: tuple[int, ...] = (1, 3, 5), antialias: bool = False):
        super().__init__()
        self.c1 = nn.ModuleList(CConv(ch, ch, k, d) for d in dilations)
        self.c2 = nn.ModuleList(CConv(ch, ch, k, 1) for _ in dilations)
        self.a1 = nn.ModuleList(AntiAliasAct(ch) for _ in dilations) if antialias else None
        self.a2 = nn.ModuleList(AntiAliasAct(ch) for _ in dilations) if antialias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.a1 is None:
            for c1, c2 in zip(self.c1, self.c2):
                x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1))
            return x
        for c1, c2, a1, a2 in zip(self.c1, self.c2, self.a1, self.a2):
            x = x + c2(a2(c1(a1(x))))
        return x


class NextBlock(nn.Module):
    """Causal ConvNeXt block with a speaker-dependent shift."""

    def __init__(self, dim: int, spk: int, k: int = 7, ahead: int = 0):
        super().__init__()
        self.dw = CConv(dim, dim, k, groups=dim, norm=False, ahead=ahead)
        self.norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, 3 * dim), nn.Linear(3 * dim, dim)
        self.cond = nn.Linear(spk, dim)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:  # x [B, C, T], g [B, S]
        y = self.dw(x).transpose(1, 2) + self.cond(g)[:, None]
        y = self.fc2(F.gelu(self.fc1(self.norm(y))))
        return x + y.transpose(1, 2)


class Source(nn.Module):
    """Harmonic-plus-noise excitation at the sample rate, from frame-rate F0.

    F0 is held for the frame it belongs to; phase is accumulated, so pitch
    steps do not click. Returns [B, harmonics + 1 + bands, N]: the first
    harmonics as separate sines, a voicing flag, then one channel per frequency
    band holding the sum of every harmonic that falls in it (up to Nyquist), so
    the decoder is handed harmonic structure across the whole spectrum.
    """

    EDGES = (0.0, 4000.0, 8000.0, 16000.0, SR / 2)
    MULTIRATE_EDGES = (0.0, 3600.0, 8000.0, 15000.0, 22500.0)

    def __init__(self, harmonics: int = 8, amp: float = 0.1, noise: float = 0.003, bands: int = 0, glide: bool | str = False,
                 with_f0: bool = False, phase_only: bool = False):
        super().__init__()
        assert bands in (0, len(self.EDGES) - 1)
        self.h, self.amp, self.noise, self.bands, self.glide = harmonics, amp, noise, bands, glide
        self.last_phase = None  # phase after the last sample, for the next streamed call
        self.n_in = harmonics + 1 + bands  # channels the decoder's excitation convs read
        self.with_f0 = with_f0  # append per-sample F0 as a last channel, for per-stage band-limiting
        self.phase_only = phase_only  # return [phase % 1, f0] so each stage synthesizes band-limited harmonics natively
        self.channels = 2 if phase_only else self.n_in + int(with_f0)

    @staticmethod
    def _dirichlet(k: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """sum_{j=1..k} sin(j * theta), in closed form."""
        half = torch.sin(theta / 2)
        ok = half.abs() > 1e-7
        return torch.where(ok, torch.sin(k * theta / 2) * torch.sin((k + 1) * theta / 2) / torch.where(ok, half, 1.0), 0.0)

    def forward(
        self,
        f0: torch.Tensor,
        phase0: torch.Tensor | None = None,
        prev: torch.Tensor | None = None,
        nxt: torch.Tensor | None = None,
        voicing: torch.Tensor | None = None,
        nxt_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """f0 [B, T]; phase0 [B] and prev [B] (the F0 of the frame before) carry a stream across calls;
        nxt [B] is the F0 of the frame after the last one, for glide="next". Optional `voicing` [B, T] in
        0..1 blends harmonic amplitude against breath noise smoothly on half-voiced laughs, sighs and fry."""
        f = f0.repeat_interleave(FRAME, dim=1).double()  # [B, N]
        if self.glide == "next":
            # A pitch frame is the pitch at the *start* of its 10 ms (measured on both trackers), so holding
            # it for the frame plays every pitch 5 ms late and in steps. Slide to the next frame's value instead.
            after = torch.cat([f0[:, 1:], f0[:, -1:] if nxt is None else nxt[:, None]], dim=1)
            ramp = torch.arange(FRAME, device=f0.device, dtype=torch.float64) / FRAME
            slide = (f0[..., None] + (after - f0)[..., None] * ramp).flatten(1)
            both = ((after > 0) & (f0 > 0)).repeat_interleave(FRAME, dim=1)
            f = torch.where(both, slide, f)
        elif self.glide:  # slide from the previous frame's pitch (this is 10 ms late; kept for old checkpoints)
            before = torch.cat([f0[:, :1] if prev is None else prev[:, None], f0[:, :-1]], dim=1)
            ramp = torch.arange(1, FRAME + 1, device=f0.device, dtype=torch.float64) / FRAME
            slide = (before[..., None] + (f0 - before)[..., None] * ramp).flatten(1)
            both = ((before > 0) & (f0 > 0)).repeat_interleave(FRAME, dim=1)
            f = torch.where(both, slide, f)
        phase = torch.cumsum(f / SR, dim=1)
        if phase0 is not None:
            phase = phase + phase0[:, None]
        self.last_phase = phase[:, -1] % 1.0
        if voicing is None:
            voiced = (f > 0).float()[:, None]
        else:
            vf = voicing.clamp(0.0, 1.0).float() * (f0 > 0).float()
            if self.glide == "next":
                nv = vf[:, -1:] if nxt_v is None else (nxt_v.clamp(0.0, 1.0).float() * ((f0[:, -1] if nxt is None else nxt) > 0).float())[:, None]
                after_v = torch.cat([vf[:, 1:], nv], dim=1)
                ramp_v = torch.arange(FRAME, device=f0.device, dtype=torch.float32) / FRAME
                v_samp = (vf[..., None] + (after_v - vf)[..., None] * ramp_v).flatten(1)
            else:
                v_samp = vf.repeat_interleave(FRAME, dim=1)
            voiced = (v_samp * (f > 0).float())[:, None]
        if self.phase_only:
            return torch.cat([torch.stack([(phase % 1.0).float(), f.float()], dim=1), voiced], dim=1)
        k = torch.arange(1, self.h + 1, device=f0.device, dtype=torch.float64)[None, :, None]
        sines = torch.sin(2 * math.pi * ((phase[:, None] * k) % 1.0)).float()
        std = voiced * self.noise + (1 - voiced) * self.amp / 3
        out = [sines * (f.float()[:, None] * k.float() < SR / 2) * voiced * self.amp]
        if self.bands:
            theta = 2 * math.pi * (phase % 1.0)
            fs = f.clamp_min(1.0)
            top = [torch.floor(min(e, SR / 2 - 1.0) / fs) for e in self.EDGES]  # harmonics at or below each edge
            for lo, hi in zip(top[:-1], top[1:]):
                band = (self._dirichlet(hi, theta) - self._dirichlet(lo, theta)) / (hi - lo).clamp_min(1.0).sqrt()
                out.append(band.float()[:, None] * voiced * self.amp)
        x = torch.cat(out, dim=1)
        x = x + std * torch.randn_like(x)
        out = [x[:, : self.h], voiced, x[:, self.h :]]
        return torch.cat(out + ([f.float()[:, None]] if self.with_f0 else []), dim=1)

    def stage_excitation(self, phase_f: torch.Tensor, rate: int, rem: int) -> torch.Tensor:
        """Synthesize band-limited harmonics + Dirichlet bands natively at `rate` Hz (`SR // rem`),
        with a smooth raised-cosine roll-off below Nyquist so zero partials fold."""
        with torch.autocast(phase_f.device.type, enabled=False):
            ph = phase_f[:, 0, rem - 1 :: rem].float()
            f = phase_f[:, 1, rem - 1 :: rem].float()
            voiced = phase_f[:, 2:3, rem - 1 :: rem].float() if phase_f.shape[1] > 2 else (f > 0).float()[:, None]
            k = torch.arange(1, self.h + 1, device=ph.device, dtype=torch.float32)[None, :, None]
            ratio = f[:, None] * k / float(rate)
            win = 0.5 * (1.0 + torch.cos(math.pi * ((ratio - 0.40) / 0.07).clamp(0.0, 1.0)))
            sines = torch.sin(2.0 * math.pi * ((ph[:, None] * k) % 1.0)) * win * voiced * self.amp
            out = [sines]
            if self.bands:
                theta = (2.0 * math.pi * (ph % 1.0)).double()
                fs = f.double().clamp_min(1.0)
                nyq = 0.45 * float(rate)
                top = [torch.floor(min(e, nyq) / fs) for e in self.MULTIRATE_EDGES]
                for lo, hi in zip(top[:-1], top[1:]):
                    band = (self._dirichlet(hi, theta) - self._dirichlet(lo, theta)) / (hi - lo).clamp_min(1.0).sqrt()
                    out.append((band.float() * (hi > lo).float())[:, None] * voiced * self.amp)
            x = torch.cat(out, dim=1)
            std = voiced * self.noise + (1.0 - voiced) * (self.amp / 3.0)
            x = x + std * torch.randn_like(x)
            return torch.cat([x[:, : self.h], voiced, x[:, self.h :]], dim=1)


class Decoder(nn.Module):
    def __init__(
        self,
        n_speakers: int,
        content_dim: int = 768,
        dim: int = 256,
        spk_dim: int = 256,
        blocks: int = 6,
        channels: int = 384,
        rates: tuple[int, ...] = (10, 6, 4, 2),
        kernels: tuple[int, ...] = (3, 7, 11),
        harmonics: int = 8,
        bands: int = 0,
        glide: bool | str = False,
        lookahead: int = 0,
        mel_cond: bool = False,
        exc: str = "full",
        smooth_in: str = "",
        tone: int = 0,
        antialias: bool = False,
        content_hz: int = 50,
        expressive: bool = False,
    ):
        super().__init__()
        rates = tuple(rates)
        assert math.prod(rates) == FRAME
        assert content_hz in (50, 100)
        self.rates = rates
        self.antialias = bool(antialias)
        self.content_hz = content_hz
        self.expressive = bool(expressive)
        self.speaker = nn.Embedding(n_speakers, spk_dim)
        self.content = nn.Linear(content_dim, dim)
        self.pitch = nn.Embedding(256, dim)
        self.loud = nn.Linear(1, dim)
        # Optional coarse description of the input's tone per frame (see `tone` below): what content
        # features say least about in non-speech sounds such as laughter. Starts as a no-op.
        self.tone = nn.Linear(tone, dim) if tone else None
        if tone:
            nn.init.zeros_(self.tone.weight), nn.init.zeros_(self.tone.bias)
        # Continuous log-pitch (32-1400 Hz) + periodicity confidence (0..1) for smooth fry/soprano and register shift.
        self.f0_proj = nn.Linear(2, dim) if self.expressive else None
        self.reg_cond = nn.Linear(2, channels) if self.expressive else None
        if self.expressive:
            nn.init.zeros_(self.f0_proj.weight), nn.init.zeros_(self.f0_proj.bias)
            nn.init.zeros_(self.reg_cond.weight), nn.init.zeros_(self.reg_cond.bias)
        # Diagnostic only: condition on the target's own mel spectrogram (copy-synthesis ceiling).
        self.mel = nn.Linear(128, dim) if mel_cond else None
        if mel_cond:
            nn.init.zeros_(self.mel.weight), nn.init.zeros_(self.mel.bias)
        self.lookahead = lookahead  # 100 Hz frames of future input, one per block from the last
        self.blocks = nn.ModuleList(NextBlock(dim, spk_dim, ahead=int(i >= blocks - lookahead)) for i in range(blocks))
        assert exc in ("full", "bandlimit", "fundamental", "multirate")
        self.exc = exc
        # Conditioning without frame-rate steps: "c" interpolates the 50 Hz content,
        # "p" blends neighbouring pitch buckets, "h" averages each hidden frame with the one before.
        self.smooth_in = smooth_in
        self.source = Source(
            harmonics,
            bands=bands,
            glide=glide,
            with_f0=exc in ("bandlimit", "fundamental"),
            phase_only=(exc == "multirate"),
        )
        self._band_edges = None
        if self.source.bands:
            self.register_buffer("_band_hi", torch.tensor(Source.EDGES[1:], dtype=torch.float32)[None, :, None], persistent=False)
        self.stage_rates: list[int] = []
        self.stage_rem: list[int] = []
        self.pre = CConv(dim, channels, 7)
        self.pre_cond = nn.Linear(spk_dim, channels)
        self.ups, self.src, self.res = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        ch, remaining = channels, FRAME
        for idx, r in enumerate(rates):
            aa_up = self.antialias and idx > 0
            aa_res = self.antialias and idx < len(rates) - 1
            self.ups.append(CUp(ch, ch // 2, r, antialias=aa_up))
            ch, remaining = ch // 2, remaining // r
            self.stage_rates.append(SR // remaining)
            self.stage_rem.append(remaining)
            if exc == "multirate":
                self.src.append(CConv(self.source.n_in, ch, 3))
            else:
                # Each output step sees only the excitation of its own hop.
                self.src.append(nn.Conv1d(self.source.n_in, ch, remaining, remaining))
            self.res.append(nn.ModuleList(ResBlock(ch, k, antialias=aa_res) for k in kernels))
        self.post = CConv(ch, 1, 7)

    @staticmethod
    def pitch_position(f0: torch.Tensor) -> torch.Tensor:
        """Fractional bucket position (1..255) on the same scale as pitch_bucket."""
        mel = 1127.0 * torch.log1p(f0.clamp_min(1.0) / 700.0)
        lo, hi = 1127.0 * math.log1p(50 / 700), 1127.0 * math.log1p(1100 / 700)
        return ((mel - lo) / (hi - lo) * 254 + 1).clamp(1, 255)

    @staticmethod
    def pitch_bucket(f0: torch.Tensor) -> torch.Tensor:
        mel = 1127.0 * torch.log1p(f0 / 700.0)
        lo, hi = 1127.0 * math.log1p(50 / 700), 1127.0 * math.log1p(1100 / 700)
        b = ((mel - lo) / (hi - lo) * 254 + 1).round().clamp(1, 255).long()
        return torch.where(f0 > 0, b, torch.zeros_like(b))

    def pitch_embed(self, f0: torch.Tensor) -> torch.Tensor:
        """Pitch embedding for `f0` [B, T] (discrete buckets, or continuous linear blend when 'p' in smooth_in)."""
        if "p" in self.smooth_in:
            pos = self.pitch_position(f0)
            lo = pos.floor().long().clamp(1, 254)
            w = (pos - lo).clamp(0.0, 1.0).to(self.pitch.weight.dtype)[..., None]
            pe = self.pitch(lo) * (1.0 - w) + self.pitch(lo + 1) * w
            return torch.where((f0 > 0)[..., None], pe, self.pitch(torch.zeros_like(lo)))
        return self.pitch(self.pitch_bucket(f0))

    @staticmethod
    def register_vec(f0: torch.Tensor, voicing: torch.Tensor | None = None) -> torch.Tensor:
        """[..., T] -> [..., T, 2]: continuous log2(f0 / 200 Hz) where voiced, and periodicity in 0..1."""
        active = (f0 > 0).float()
        lf = torch.log2(f0.float().clamp_min(32.0) / 200.0) * active
        v = active if voicing is None else (voicing.float().clamp(0.0, 1.0) * active)
        return torch.stack([lf, v], dim=-1)

    @staticmethod
    def tone_gate(loud: torch.Tensor) -> torch.Tensor:
        """Smooth 0..1 gate over -72..-58 dB so dead air never injects a 0 dB ratio while quiet speech stays steady."""
        return ((loud.float() + 72.0) / 14.0).clamp(0.0, 1.0)

    def features(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        loud: torch.Tensor,
        spk: torch.Tensor,
        mel: torch.Tensor | None = None,
        tone: torch.Tensor | None = None,
        content_hz: int | None = None,
        voicing: torch.Tensor | None = None,
    ):
        """content [B, T/2, C] at 50 Hz (or [B, T, C] at 100 Hz); f0 (Hz, 0 unvoiced) and loud (dB) [B, T]
        at 100 Hz; spk [B]. Returns hidden [B, D, T] and speaker vector."""
        g = self.speaker(spk)
        if (content_hz or self.content_hz) == 100:
            t = min(content.shape[1], f0.shape[1], loud.shape[1])
            x = self.content(content[:, :t])
        else:
            t = min(content.shape[1] * 2, f0.shape[1], loud.shape[1])
            x = self.content(content)
            if "c" in self.smooth_in:  # frame 2i is content i, frame 2i+1 the mean of i and i+1 would need the future: use i-1..i instead
                x = torch.stack([(x + torch.cat([x[:, :1], x[:, :-1]], 1)) / 2, x], dim=2).flatten(1, 2)[:, :t]
            else:
                x = x.repeat_interleave(2, dim=1)[:, :t]
        pe = self.pitch_embed(f0[:, :t])
        x = x + pe + self.loud((loud[:, :t, None] + 40.0) / 20.0)
        if self.f0_proj is not None:
            rv = self.register_vec(f0[:, :t], None if voicing is None else voicing[:, :t]).to(self.f0_proj.weight.dtype)
            x = x + self.f0_proj(rv)
        if self.mel is not None:
            x = x + self.mel(mel[:, :t])
        if self.tone is not None and tone is not None:
            tn = self.tone((tone[:, :t] / 10.0).to(self.tone.weight.dtype))
            if self.expressive:
                tn = tn * self.tone_gate(loud[:, :t])[..., None].to(tn.dtype)
            x = x + tn
        x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x, g)
        if "h" in self.smooth_in:
            x = (x + F.pad(x, (1, 0), mode="replicate")[..., :-1]) / 2
        return x, g

    def generate(self, h: torch.Tensor, g: torch.Tensor, excitation: torch.Tensor, reg: torch.Tensor | None = None) -> torch.Tensor:
        """h [B, D, T], excitation [B, C, T * FRAME] -> waveform [B, 1, T * FRAME]."""
        x = self.pre(h) + self.pre_cond(g)[..., None]
        if self.reg_cond is not None and reg is not None:
            x = x + self.reg_cond(reg.to(self.reg_cond.weight.dtype)).transpose(1, 2)
        if self.exc == "multirate":
            for up, src, res, rate, rem in zip(self.ups, self.src, self.res, self.stage_rates, self.stage_rem):
                e_i = self.source.stage_excitation(excitation, rate, rem).to(x.dtype)
                x = up(F.leaky_relu(x, 0.1)) + src(e_i)
                x = sum(r(x) for r in res) / len(res)
            return torch.tanh(self.post(F.leaky_relu(x, 0.1)))
        f = None
        if self.exc != "full":
            excitation, f = excitation[:, :-1], excitation[:, -1:]
        for up, src, res, rate in zip(self.ups, self.src, self.res, self.stage_rates):
            x = up(F.leaky_relu(x, 0.1)) + src(excitation if f is None else self.limit(excitation, f, rate))
            x = sum(r(x) for r in res) / len(res)
        return torch.tanh(self.post(F.leaky_relu(x, 0.1)))

    def limit(self, e: torch.Tensor, f: torch.Tensor, rate: int) -> torch.Tensor:
        """Drop, where voiced, the excitation partials a stage running at `rate` cannot
        represent: its input conv decimates without a filter, so they would fold back
        as inharmonic components. Uses a smooth cosine fade near Nyquist so pitch glides never pop."""
        h, nb = self.source.h, self.source.bands
        k = torch.arange(1, h + 1, device=e.device, dtype=e.dtype)[None, :, None]
        if self.exc == "bandlimit":
            ratio = f * k / float(rate)
            win = 0.5 * (1.0 + torch.cos(math.pi * ((ratio - 0.38) / 0.08).clamp(0.0, 1.0)))
            keep = torch.where(f <= 0, torch.ones_like(win), win)
        else:
            keep = (((k == 1) & (f < 0.45 * rate)) | (f <= 0)).to(e.dtype)
        parts = [e[:, :h] * keep, e[:, h : h + 1]]
        if nb:
            ok = (self._band_hi <= (rate / 2 if self.exc == "bandlimit" else -1.0)) | (f <= 0)
            parts.append(e[:, h + 1 :] * ok.to(e.dtype))
        return torch.cat(parts, dim=1)

    def forward(self, content, f0, loud, spk) -> torch.Tensor:
        h, g = self.features(content, f0, loud, spk)
        return self.generate(h, g, self.source(f0[:, : h.shape[-1]]))


def loudness(wav: torch.Tensor) -> torch.Tensor:
    """[B, N] at 48 kHz -> zero-lag dB RMS of each 10 ms frame [B, N // FRAME] with per-frame DC/rumble mean removed."""
    n = wav.shape[1] // FRAME
    frames = wav[:, : n * FRAME].float().reshape(wav.shape[0], n, FRAME)
    ac = frames - frames.mean(dim=-1, keepdim=True)
    return 20.0 * torch.log10(ac.pow(2).mean(-1).sqrt() + 1e-5)


PS_LOOKBACK = 2 * FRAME  # up to 3 * FRAME (30 ms) causal window -> covers 1 full glottal cycle down to 33.3 Hz
_PS_IDX: dict[str, torch.Tensor] = {}


def _ps_idx(device: torch.device) -> torch.Tensor:
    key = str(device)
    buf = _PS_IDX.get(key)
    if buf is None:
        buf = torch.arange(FRAME + PS_LOOKBACK, device=device, dtype=torch.float32)[None, None, :]
        _PS_IDX[key] = buf
    return buf


def pitch_sync_loudness(wav: torch.Tensor, raw_f0: torch.Tensor, prepadded: bool = False) -> torch.Tensor:
    """Causal pitch-synchronous RMS in dB [B, T]: exact 1.0 glottal cycle (SR / raw_f0) when 0 < raw_f0 < 100 Hz
    (cancelling <100 Hz glottal-pulse beating analytically with zero multi-frame IIR lag on fast speech), and exact
    10 ms (FRAME) when unvoiced or raw_f0 >= 100 Hz. When `prepadded=True`, `wav` carries `PS_LOOKBACK` lead samples."""
    max_w = FRAME + PS_LOOKBACK
    if prepadded:
        n = (wav.shape[1] - PS_LOOKBACK) // FRAME
        x = wav[:, : PS_LOOKBACK + n * FRAME].float().unfold(1, max_w, FRAME)
    else:
        n = wav.shape[1] // FRAME
        x = F.pad(wav[:, : n * FRAME].float(), (PS_LOOKBACK, 0)).unfold(1, max_w, FRAME)
    f = raw_f0[:, :n].float()
    period = torch.where((f > 0) & (f < 100.0), SR / f.clamp_min(33.4), torch.full_like(f, float(FRAME))).clamp(float(FRAME), float(max_w))
    w = (_ps_idx(wav.device) - (float(max_w) - period[..., None]) + 1.0).clamp(0.0, 1.0)
    denom = w.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (x * w).sum(dim=-1, keepdim=True) / denom
    ms = (((x - mean).pow(2) * w).sum(dim=-1, keepdim=True) / denom).clamp_min(0.0)[..., 0]
    return 20.0 * torch.log10(ms.sqrt() + 1e-5)


TONE_BANDS = ((2000, 8000, 250, 2000), (700, 2000, 250, 700))  # brightness, vowel openness
_TONE_BANKS: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _tone_buffers(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = str(device)
    bufs = _TONE_BANKS.get(key)
    if bufs is None:
        win = torch.hann_window(2 * FRAME, device=device)
        f = torch.arange(FRAME + 1, device=device, dtype=torch.float32) * (SR / (2 * FRAME))
        cols = [((f >= lo) & (f < hi)).float() for a, b, c, d in TONE_BANDS for lo, hi in ((a, b), (c, d))]
        bufs = (win, torch.stack(cols, dim=-1))
        _TONE_BANKS[key] = bufs
    return bufs


def tone(wav: torch.Tensor) -> torch.Tensor:
    """[B, N] at 48 kHz -> [B, N // FRAME, 2] in dB: brightness (2-8 kHz against 0.25-2 kHz) and
    openness (0.7-2 kHz against 0.25-0.7 kHz) of each 10 ms frame, from that frame and the one before."""
    n = wav.shape[1] // FRAME
    win, bank = _tone_buffers(wav.device)
    x = F.pad(wav[:, : n * FRAME].float(), (FRAME, 0)).unfold(1, 2 * FRAME, FRAME)  # [B, n, 960]
    S = torch.fft.rfft(x * win).abs().pow(2)
    e = S @ bank + 1e-4  # fixed matmul (no boolean indexing), so CUDA graphs can record it
    return 10.0 * torch.log10(e[..., 0::2] / e[..., 1::2])


class CausalMel(nn.Module):
    """Log-mel at 100 Hz where frame t only uses samples up to the end of frame t."""

    def __init__(self):
        super().__init__()
        import torchaudio

        self.m = torchaudio.transforms.MelSpectrogram(SR, n_fft=1024, hop_length=FRAME, n_mels=128, center=False, power=1.0)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:  # [B, N] -> [B, T, 128]
        return torch.log(self.m(F.pad(wav, (1024 - FRAME, 0))).clamp_min(1e-5)).transpose(1, 2)
