import math
from typing import Callable, Optional

import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F

from rvc.rectified.content_bottleneck import ContentBottleneck


LOG_F0_CENTER = math.log(200.0)
LOG_F0_SCALE = 0.7
SAMPLERS = ("euler", "heun", "rk4", "mean")


SCHEDULES = ("uniform", "sway", "logit-normal")

RESCALE_MODES = ("global", "frame")


def time_grid(schedule: str, steps: int, start: float, device) -> torch.Tensor:
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, not {schedule!r}.")
    u = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if schedule == "sway":
        g = 1.0 - torch.cos(0.5 * math.pi * u)
    elif schedule == "logit-normal":
        g = torch.sigmoid(math.sqrt(2.0) * torch.erfinv((2.0 * u - 1.0).clamp(-1.0, 1.0)))
    else:
        g = u
    g[0], g[-1] = 0.0, 1.0
    return start + (1.0 - start) * g


def pitch_features(f0: torch.Tensor, fourier: int = 0) -> torch.Tensor:
    voiced = (f0 > 0).float()
    log_f0 = (torch.log(f0.clamp_min(1.0)) - LOG_F0_CENTER) / LOG_F0_SCALE
    features = [log_f0 * voiced, voiced]
    for index in range(fourier):
        angle = (2.0**index * math.pi) * log_f0
        features += [torch.sin(angle) * voiced, torch.cos(angle) * voiced]
    return torch.stack(features, dim=1)


class HarmonicPrior(nn.Module):
    def __init__(self, sample_rate, n_fft, n_mels, fmin, fmax):
        super().__init__()
        basis = torch.from_numpy(
            librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        ).float()
        basis = basis / basis.sum(1, keepdim=True).clamp_min(1e-8)
        self.register_buffer("basis", basis, persistent=False)
        self.register_buffer(
            "freqs", torch.fft.rfftfreq(n_fft, 1.0 / sample_rate).float(), persistent=False
        )

        self.sigma = sample_rate / n_fft

    def forward(self, f0: torch.Tensor) -> torch.Tensor:
        f0 = f0.float()
        ratio = self.freqs[None, :, None] / f0.clamp_min(1.0)[:, None, :]
        nearest = ratio.round()
        distance = (ratio - nearest).abs() * f0[:, None, :]
        comb = torch.exp(-0.5 * (distance / self.sigma).square()) * (nearest >= 1)
        with torch.autocast(f0.device.type, enabled=False):
            prior = torch.matmul(self.basis, comb)
        return prior * (f0 > 0).float()[:, None, :]


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, layer_scale: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.up = nn.Linear(channels, channels * 4)
        self.down = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale)) if layer_scale > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x * mask).transpose(1, 2)
        y = self.down(F.gelu(self.up(self.norm(y))))
        if self.gamma is not None:
            y = y * self.gamma
        return (x + self.dropout(y.transpose(1, 2))) * mask


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        content_channels: int,
        hidden_channels: int,
        speaker_count: int,
        speaker_channels: int,
        layers: int,
        content_bottleneck: int = 0,
        pitch_fourier: int = 0,
        harmonic_prior: Optional[HarmonicPrior] = None,
        breathiness: bool = False,
        key_shift: bool = False,
        content_bottleneck_noise: float = 0.0,
        speed: bool = False,
        voicing: bool = False,
        tension: bool = False,
    ):
        super().__init__()
        self.speaker_count = int(speaker_count)
        self.bottleneck = (
            ContentBottleneck(content_channels, content_bottleneck, content_bottleneck_noise)
            if content_bottleneck > 0
            else None
        )
        self.content = nn.Linear(content_channels, hidden_channels)
        self.pitch_fourier = int(pitch_fourier)
        self.pitch = nn.Conv1d(2 + 2 * self.pitch_fourier, hidden_channels, 3, padding=1)
        self.harmonic_prior = harmonic_prior
        if harmonic_prior is not None:
            self.harmonics = nn.Conv1d(harmonic_prior.basis.shape[0], hidden_channels, 1)
        self.energy = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.breathiness = nn.Conv1d(1, hidden_channels, 3, padding=1) if breathiness else None
        self.key_shift = nn.Linear(1, hidden_channels) if key_shift else None
        self.speed = nn.Linear(1, hidden_channels) if speed else None
        self.speaker = nn.Embedding(self.speaker_count + 1, speaker_channels)
        self.speaker_proj = nn.Linear(speaker_channels, hidden_channels)
        self.blocks = nn.ModuleList([ConvNeXtBlock(hidden_channels) for _ in range(layers)])
        self.voicing = nn.Conv1d(1, hidden_channels, 3, padding=1) if voicing else None
        self.tension = nn.Conv1d(1, hidden_channels, 3, padding=1) if tension else None

    def voice(self, speaker: torch.Tensor) -> torch.Tensor:
        return self.speaker_proj(self.speaker(speaker))

    def forward(self, content, f0, energy, speaker, mask, breathiness=None, key_shift=None, speed=None,
                voicing=None, tension=None):
        if self.bottleneck is not None:
            content = self.bottleneck(content)
        x = self.content(content).transpose(1, 2)
        x = x + self.pitch(pitch_features(f0, self.pitch_fourier))
        if self.harmonic_prior is not None:
            x = x + self.harmonics(self.harmonic_prior(f0))
        x = x + self.energy(energy.unsqueeze(1))
        for name, value in (("voicing", voicing), ("tension", tension)):
            projection = getattr(self, name)
            if projection is not None:
                if value is None:
                    raise ValueError(f"This flow checkpoint requires {name} conditioning.")
                x = x + projection(value.unsqueeze(1))
        if self.breathiness is not None:
            if breathiness is None:
                breathiness = torch.ones_like(energy)
            x = x + self.breathiness(breathiness.unsqueeze(1))
        if self.key_shift is not None:
            if key_shift is None:
                key_shift = torch.zeros(content.shape[0], device=content.device)
            x = x + self.key_shift(key_shift.float().view(-1, 1) / 12.0).unsqueeze(-1)
        if self.speed is not None:
            if speed is None:
                speed = torch.ones(content.shape[0], device=content.device)
            x = x + self.speed(speed.float().view(-1, 1)).unsqueeze(-1)
        x = x + self.voice(speaker).unsqueeze(-1)
        x = x * mask
        for block in self.blocks:
            x = block(x, mask)
        return x


def timestep_embedding(t: torch.Tensor, channels: int, scale: float = 1000.0) -> torch.Tensor:
    half = channels // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = (t.float() * scale)[:, None] * frequencies[None]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class _ATanGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate):
        atan_gate = torch.atan(gate)
        ctx.save_for_backward(out / gate.square().add(1.0), atan_gate)
        return out * atan_gate

    @staticmethod
    def backward(ctx, grad):
        decay_out, atan_gate = ctx.saved_tensors
        return grad * atan_gate, grad * decay_out


def atan_glu(x: torch.Tensor, fused: bool = True) -> torch.Tensor:
    out, gate = x.chunk(2, dim=-1)
    if fused and torch.is_grad_enabled():
        return _ATanGLU.apply(out, gate)
    return out * torch.atan(gate)


class LYNXNet2Block(nn.Module):
    def __init__(self, channels, expansion, kernel_size, adaln=False):
        super().__init__()
        inner = int(channels * expansion)
        self.norm = nn.LayerNorm(channels, elementwise_affine=not adaln)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.up = nn.Linear(channels, inner * 2)
        self.mid = nn.Linear(inner, inner * 2)
        self.down = nn.Linear(inner, channels)
        self.modulation = None
        if adaln:
            self.modulation = nn.Linear(channels, channels * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(self, x, mask, embedding=None, fused=True):
        y = self.norm(x)
        gate = None
        if self.modulation is not None:
            shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=-1)
            y = y + y * scale + shift
        y = self.depthwise((y * mask).transpose(1, 2)).transpose(1, 2)
        y = self.down(atan_glu(self.mid(atan_glu(self.up(y), fused)), fused))
        if gate is not None:
            y = y + gate * y
        return (x + y) * mask


class LYNXNet2Backbone(nn.Module):
    def __init__(self, n_mels, cond_channels, channels=1024, layers=6, expansion=1, kernel_size=31,
                 adaln=False, span=False, time_scale=1000.0):
        super().__init__()
        self.channels = int(channels)
        self.time_scale = float(time_scale)
        self.input = nn.Linear(n_mels, channels)
        self.input_cond = nn.Conv1d(cond_channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels)
        )
        self.span_mlp = None
        if span:
            self.span_mlp = nn.Sequential(
                nn.Linear(channels, channels * 4, bias=False), nn.GELU(),
                nn.Linear(channels * 4, channels, bias=False),
            )
            nn.init.zeros_(self.span_mlp[-1].weight)
        self.layers = nn.ModuleList(
            [LYNXNet2Block(channels, expansion, kernel_size, adaln) for _ in range(layers)]
        )
        self.voice = nn.Linear(cond_channels, channels) if adaln else None
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x, t, cond, mask, voice=None, span=None):
        time = self.time_mlp(timestep_embedding(t.reshape(-1), self.channels, self.time_scale))
        time = time.view(t.shape[0], -1, self.channels)
        if span is not None:
            features = timestep_embedding(span.reshape(-1), self.channels, self.time_scale)
            half = self.channels // 2
            features = torch.cat((features[:, :half], 1.0 - features[:, half:]), dim=-1)
            time = time + self.span_mlp(features).view(span.shape[0], -1, self.channels)
        frame_mask = mask.transpose(1, 2)
        with torch.autocast(x.device.type, enabled=False):
            h = self.input(x.transpose(1, 2).to(self.input.weight.dtype))
        h = h + self.input_cond(cond).transpose(1, 2) + time
        h = h * frame_mask
        embedding = None
        if self.voice is not None:
            embedding = time + self.voice(voice)[:, None, :]
        for layer in self.layers:
            h = layer(h, frame_mask, embedding, fused=span is None)
        if span is None:
            h = self.norm(h)
        else:
            h = F.layer_norm(h, (self.channels,), eps=self.norm.eps) * self.norm.weight + self.norm.bias
        return (self.output(h) * frame_mask).transpose(1, 2)


class AuxDecoder(nn.Module):
    def __init__(self, cond_channels, n_mels, channels=512, layers=6, dropout=0.1):
        super().__init__()
        self.input = nn.Conv1d(cond_channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(channels, layer_scale=1e-6, dropout=dropout) for _ in range(layers)]
        )
        self.output = nn.Conv1d(channels, n_mels, 7, padding=3)
        self.output.use_adamw = True

    def forward(self, cond, mask):
        x = self.input(cond) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.output(x) * mask


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        n_mels: int,
        speaker_count: int,
        content_channels: int = 768,
        hidden_channels: int = 384,
        encoder_layers: int = 4,
        content_bottleneck: int = 0,
        content_bottleneck_noise: float = 0.0,
        speaker_channels: int = 256,
        pitch_fourier: int = 0,
        harmonic_prior: Optional[dict] = None,
        breathiness: bool = False,
        key_shift: bool = False,
        speed: bool = False,
        backbone: str = "lynxnet2",
        backbone_args: Optional[dict] = None,
        aux_decoder: Optional[dict] = None,
        t_start: float = 0.0,
        aux_grad: float = 0.1,
        dual_timestep: bool = False,
        voicing: bool = False,
        tension: bool = False,
        sampling_method: str = "euler",
        sampling_steps: int = 16,
        mean_flow: bool = False,
    ):
        super().__init__()
        if backbone != "lynxnet2":
            raise ValueError(f"Only the lynxnet2 backbone is supported, not {backbone!r}.")
        self.n_mels = int(n_mels)
        self.hidden_channels = int(hidden_channels)
        self.encoder = ConditionEncoder(
            content_channels,
            hidden_channels,
            speaker_count,
            speaker_channels,
            encoder_layers,
            content_bottleneck,
            pitch_fourier,
            HarmonicPrior(n_mels=n_mels, **harmonic_prior) if harmonic_prior else None,
            breathiness,
            key_shift,
            content_bottleneck_noise,
            speed,
            voicing,
            tension,
        )
        self.backbone = LYNXNet2Backbone(n_mels, hidden_channels, span=bool(mean_flow), **(backbone_args or {}))


        conditioning = [self.backbone.time_mlp, self.backbone.span_mlp, self.encoder.speaker_proj, self.backbone.voice]
        conditioning += [layer.modulation for layer in self.backbone.layers]
        for module in filter(None, conditioning):
            for child in module.modules():
                child.use_adamw = True
        self.aux = AuxDecoder(hidden_channels, n_mels, **aux_decoder) if aux_decoder else None
        self.t_start = float(t_start) if self.aux is not None else 0.0
        self.aux_grad = float(aux_grad)
        self.dual_timestep = bool(dual_timestep)
        if sampling_method not in SAMPLERS or int(sampling_steps) != sampling_steps or sampling_steps < 1:
            raise ValueError("Invalid flow sampling method or step count.")
        if sampling_method == "mean" and not mean_flow:
            raise ValueError("Mean sampling requires a MeanFlow-trained checkpoint.")
        self.sampling_method = sampling_method
        self.sampling_steps = int(sampling_steps)

    @property
    def speaker_count(self) -> int:
        return self.encoder.speaker_count

    def _drop_speakers(self, speaker, speaker_dropout):
        if speaker_dropout <= 0:
            return speaker
        dropped = torch.rand(speaker.shape, device=speaker.device) < speaker_dropout
        return torch.where(dropped, torch.full_like(speaker, self.speaker_count), speaker)

    def _losses(self, mel, cond, voice, mask, t, noise, backbone):
        mixing = t[:, None, None] if t.ndim == 1 else t[:, None, :]
        x_t = (1.0 - mixing) * noise + mixing * mel
        prediction = backbone(x_t, t, cond, mask, voice)
        error = (prediction.float() - (mel - noise).float()).square() * mask
        count = (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)
        flow = error.sum((1, 2)) / count
        aux = None
        if self.aux is not None:
            aux_cond = cond * self.aux_grad + cond.detach() * (1.0 - self.aux_grad)
            aux_error = (self.aux(aux_cond, mask).float() - mel.float()).abs() * mask
            aux = aux_error.sum() / count.sum()
        return flow, aux

    def _flow_error(self, mel, cond, voice, mask, t, noise, backbone):
        mix = t[:, None, None] if t.dim() == 1 else t[:, None, :]
        x_t = (1.0 - mix) * noise + mix * mel
        prediction = backbone(x_t, t, cond, mask, voice)
        error = (prediction.float() - (mel - noise).float()).square() * mask
        return error.sum((1, 2)) / (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)

    def _aux_loss(self, mel, cond, mask):
        if self.aux is None:
            return None
        aux_cond = cond * self.aux_grad + cond.detach() * (1.0 - self.aux_grad)
        error = (self.aux(aux_cond, mask).float() - mel.float()).abs() * mask
        return error.sum() / (mask.sum() * self.n_mels).clamp_min(1.0)

    def mean_velocity(self, x, t, span, velocity, cond, mask, voice):

        def field(x, t, span):
            return self.backbone(x, t, cond, mask, voice, span)
        tangents = (velocity, torch.ones_like(t), -torch.ones_like(span))
        mean, derivative = torch.func.jvp(field, (x, t, span), tangents)
        return (mean, derivative.detach())

    def _mean_error(self, mel, cond, voice, mask, noise, mean_field, bootstrap=1.0):
        first, second = (self._times(mel.shape[0], mel.device) for _ in range(2))
        t, span = (torch.minimum(first, second), (first - second).abs())
        x_t = (1.0 - t[:, None, None]) * noise + t[:, None, None] * mel
        velocity = mel - noise
        mean, derivative = mean_field(x_t, t, span, velocity, cond, mask, voice)
        correction = span[:, None, None] * derivative.float() * mask
        count = (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)
        size = (correction.square().sum((1, 2)) / count).sqrt()
        ratio = size / ((velocity.square() * mask).sum((1, 2)) / count).sqrt().clamp_min(1e-08)
        correction = correction * (1.0 / ratio.clamp_min(1.0))[:, None, None]
        target = velocity + bootstrap * correction
        error = (mean.float() - target).square() * mask
        return (error.sum((1, 2)) / count, ratio)

    def _mean_forward(self, mel, cond, voice, mask, backbone, mean_ratio, mean_bootstrap):
        if mel.shape[0] < 2 or not 0.0 < mean_ratio < 1.0:
            raise ValueError("MeanFlow training requires batch size at least 2 and a ratio between 0 and 1.")
        mean_field = self.mean_velocity
        noise = torch.randn_like(mel)
        batch = mel.shape[0]
        mean_items = min(batch - 1, max(1, int(round(mean_ratio * batch))))
        rest = slice(mean_items, None)
        t = self._times(batch - mean_items, mel.device)
        if self.dual_timestep:
            other = self._times(batch - mean_items, mel.device)
            swap = torch.rand(batch - mean_items, mel.shape[-1], device=mel.device) < 0.25
            t = torch.where(swap & mask[rest, 0].bool(), other[:, None], t[:, None])
        backbone = self.backbone if backbone is None else backbone
        error = self._flow_error(mel[rest], cond[rest], voice[rest], mask[rest], t, noise[rest], backbone)

        def pooled(error, frames, weighted=False):
            weight = (error.detach() + 1e-3).reciprocal() if weighted else 1.0
            return (weight * error * frames).sum() / frames.sum().clamp_min(1.0)
        frames = mask[rest].sum((1, 2))
        flow = pooled(error, frames)
        head = slice(0, mean_items)
        mean_error, bootstrap_ratio = self._mean_error(
            mel[head], cond[head], voice[head], mask[head], noise[head], mean_field, mean_bootstrap
        )
        mean_frames = mask[head].sum((1, 2))
        mean = torch.stack((
            pooled(mean_error, mean_frames, True), flow.detach(),
            pooled(mean_error, mean_frames).detach(), bootstrap_ratio.mean(),
        ))
        flow = pooled(error, frames, True)
        return (flow, self._aux_loss(mel, cond, mask), mean)

    def _times(self, batch, device):
        u = (torch.arange(batch, device=device) + torch.rand(batch, device=device)) / batch
        u = u[torch.randperm(batch, device=device)].clamp(1e-6, 1.0 - 1e-6)
        t = torch.sigmoid(math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0))
        return self.t_start + (1.0 - self.t_start) * t

    def forward(self, mel, content, f0, energy, speaker, mask, speaker_dropout=0.0,
                breathiness=None, key_shift=None, speed=None, backbone=None, voicing=None, tension=None,
                mean_ratio=0.25, mean_bootstrap=1.0):
        speaker = self._drop_speakers(speaker, speaker_dropout)
        cond = self.encoder(content, f0, energy, speaker, mask, breathiness, key_shift, speed, voicing, tension)
        voice = self.encoder.voice(speaker)
        if self.backbone.span_mlp is not None:
            return self._mean_forward(mel, cond, voice, mask, backbone, mean_ratio, mean_bootstrap)
        t = self._times(mel.shape[0], mel.device)
        if self.dual_timestep:
            t2 = self._times(mel.shape[0], mel.device)
            alternate = torch.rand(mel.shape[0], mel.shape[-1], device=mel.device) < 0.25
            t = torch.where(alternate & mask[:, 0].bool(), t2[:, None], t[:, None])

        backbone = self.backbone if backbone is None else backbone
        flow, aux = self._losses(mel, cond, voice, mask, t, torch.randn_like(mel), backbone)
        return (flow * mask.sum((1, 2))).sum() / mask.sum().clamp_min(1.0), aux

    @torch.no_grad()
    def validation_losses(self, mel, content, f0, energy, speaker, mask, breathiness, key_shift,
                          speed, noise, fractions, voicing=None, tension=None):
        cond = self.encoder(content, f0, energy, speaker, mask, breathiness, key_shift, speed, voicing, tension)
        voice = self.encoder.voice(speaker)
        losses, aux = [], None
        for fraction in fractions:
            t = torch.full((mel.shape[0],), self.t_start + (1.0 - self.t_start) * fraction, device=mel.device)
            flow, aux = self._losses(mel, cond, voice, mask, t, noise, self.backbone)
            losses.append((flow * mask.sum((1, 2))).sum() / mask.sum().clamp_min(1.0))
        return torch.stack(losses), aux

    @torch.no_grad()
    def sample(
        self,
        content,
        f0,
        energy,
        speaker,
        mask,
        steps: Optional[int] = None,
        method: Optional[str] = None,
        cfg_scale: float = 1.0,
        noise: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[], None]] = None,
        breathiness: Optional[torch.Tensor] = None,
        key_shift: Optional[torch.Tensor] = None,
        content_guidance: float = 0.0,
        guidance_rescale: float = 0.0,
        temperature: float = 1.0,
        start: Optional[float] = None,
        guidance_interval: tuple = (0.0, 1.0),
        rescale_mode: str = "global",
        schedule: str = "uniform",
        churn: float = 0.0,
        churn_noise: Optional[Callable[[int], torch.Tensor]] = None,
        voicing: Optional[torch.Tensor] = None,
        tension: Optional[torch.Tensor] = None,
    ):
        method = self.sampling_method if method is None else method
        steps = self.sampling_steps if steps is None else steps
        if method == "mean" and self.backbone.span_mlp is None:
            raise ValueError("Mean sampling requires a MeanFlow-trained checkpoint.")
        if method not in SAMPLERS:
            raise ValueError(f"method must be one of {SAMPLERS}, not {method!r}.")
        if rescale_mode not in RESCALE_MODES:
            raise ValueError(f"rescale_mode must be one of {RESCALE_MODES}, not {rescale_mode!r}.")
        batch = content.shape[0]
        null = torch.full_like(speaker, self.speaker_count)

        variants = [(content, speaker)]
        if cfg_scale != 1.0:
            variants.append((content, null))
        if content_guidance > 0:
            frames = content.shape[1]
            blurred = F.interpolate(content.transpose(1, 2), size=max(1, frames // 4), mode="linear")
            blurred = F.interpolate(blurred, size=frames, mode="linear").transpose(1, 2)
            variants.append((blurred, speaker))
        count = len(variants)

        def repeat(value):
            return None if value is None else value.repeat(count, *([1] * (value.dim() - 1)))

        cond = self.encoder(
            torch.cat([c for c, _ in variants]), repeat(f0), repeat(energy),
            torch.cat([s for _, s in variants]), repeat(mask), repeat(breathiness), repeat(key_shift),
            voicing=repeat(voicing), tension=repeat(tension),
        )
        voice = self.encoder.voice(torch.cat([s for _, s in variants]))
        masks = repeat(mask)
        guide_from, guide_until = (float(value) for value in guidance_interval)

        def spread(y):
            if rescale_mode == "frame":
                return y.square().mean(1, keepdim=True).sqrt()
            frames = mask.sum((1, 2)).clamp_min(1.0) * self.n_mels
            return ((y.square() * mask).sum((1, 2)) / frames).sqrt()[:, None, None]

        def field(x, t, span=None):
            now = float(t[0])

            if count == 1 or not (guide_from <= now and (now < guide_until or guide_until >= 1.0)):
                return self.backbone(x, t, cond[:batch], mask, voice[:batch], span)
            v = self.backbone(repeat(x), repeat(t), cond, masks, voice, repeat(span)).chunk(count)
            guided, index = v[0], 1
            if cfg_scale != 1.0:
                guided = guided + (cfg_scale - 1.0) * (v[0] - v[index])
                index += 1
            if content_guidance > 0:
                guided = guided + content_guidance * (v[0] - v[index])
            if guidance_rescale > 0:
                rescaled = guided * spread(v[0]) / spread(guided).clamp_min(1e-6)
                guided = guidance_rescale * rescaled + (1.0 - guidance_rescale) * guided
            return guided

        shape = (batch, self.n_mels, content.shape[1])
        if noise is None:
            noise = torch.randn(shape, device=content.device)
        noise = noise * float(temperature)
        t0 = 0.0
        if self.t_start > 0:
            t0 = self.t_start if start is None else min(max(self.t_start, float(start)), 0.99)
            x = ((1.0 - t0) * noise + t0 * self.aux(cond[:batch], mask)) * mask
        else:
            x = noise * mask
        times = time_grid(schedule, max(1, int(steps)), t0, x.device)
        for index in range(times.shape[0] - 1):
            now = float(times[index])
            back = max(self.t_start, now - float(churn) * float(times[index + 1] - times[index]))
            if churn > 0 and 0 < back < now:
                fresh = torch.randn_like(x) if churn_noise is None else churn_noise(index)
                scale = back / now
                top_up = math.sqrt(max((1.0 - back) ** 2 - (scale * (1.0 - now)) ** 2, 0.0))
                x = (scale * x + float(temperature) * top_up * fresh) * mask
                now = back
            t = torch.full((batch,), now, device=x.device, dtype=times.dtype)
            dt = times[index + 1] - now
            v = field(x, t, dt.expand(batch) if method == "mean" else None)
            if method == "heun":
                v_next = field(x + dt * v, times[index + 1].expand(batch))
                v = 0.5 * (v + v_next)
            elif method == "rk4":
                middle = t + 0.5 * dt
                k2 = field(x + 0.5 * dt * v, middle)
                k3 = field(x + 0.5 * dt * k2, middle)
                k4 = field(x + dt * k3, times[index + 1].expand(batch))
                v = (v + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            x = x + dt * v
            if callback is not None:
                callback()
        return x * mask


def resize_speakers(state_dict: dict, speaker_count: int) -> dict:
    key = "encoder.speaker.weight"
    table = state_dict[key]
    if table.shape[0] == speaker_count + 1:
        return state_dict
    trained, null = table[:-1], table[-1:]
    rows = trained.mean(0, keepdim=True).expand(speaker_count, -1).clone()
    state_dict = dict(state_dict)
    state_dict[key] = torch.cat((rows, null), dim=0)
    return state_dict


def build_flow(config: dict, speaker_count: int) -> RectifiedFlow:
    model = dict(config["flow"]["model"])
    data = config["data"]
    if model.pop("harmonic_prior", False):
        model["harmonic_prior"] = dict(
            sample_rate=data["sample_rate"], n_fft=data["n_fft"],
            fmin=data["mel_fmin"], fmax=data["mel_fmax"],
        )
    return RectifiedFlow(n_mels=data["n_mels"], speaker_count=speaker_count, **model)
