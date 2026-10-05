import math
from typing import Callable, Optional

import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F

from rvc.rectified.config import resolve_config

LOG_F0_CENTER = math.log(200.0)
LOG_F0_SCALE = 0.7
SAMPLERS = ("euler", "rk2", "rk4", "rk5")


SCHEDULES = ("uniform",)

RESCALE_MODES = ("global", "frame")


def time_grid(schedule: str, steps: int, start: float, device) -> torch.Tensor:
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}, not {schedule!r}.")
    return start + torch.arange(steps + 1, device=device) * ((1.0 - start) / steps)


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


class MixedPrecisionLayerNorm(nn.LayerNorm):
    """LayerNorm that keeps fp16/bf16 activations under AMP autocast"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, enabled=False):
            weight = self.weight
            bias = self.bias
            if weight is not None and weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            if bias is not None and bias.dtype != x.dtype:
                bias = bias.to(x.dtype)
            return F.layer_norm(
                x, self.normalized_shape, weight, bias, self.eps
            )


class ConvNeXtBlock(nn.Module):
    def __init__(self, channels: int, layer_scale: float = 0.0, dropout: float = 0.0,
                 speaker_channels: int = 0):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, 7, padding=3, groups=channels)
        self.norm = MixedPrecisionLayerNorm(channels)
        self.up = nn.Linear(channels, channels * 4)
        self.down = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale)) if layer_scale > 0 else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.speaker_modulation = None
        if speaker_channels:
            self.speaker_modulation = nn.Linear(speaker_channels, channels * 2)
            self.speaker_modulation.use_adamw = True
            nn.init.zeros_(self.speaker_modulation.weight)
            nn.init.zeros_(self.speaker_modulation.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, voice=None) -> torch.Tensor:
        y = self.depthwise(x * mask).transpose(1, 2)
        y = self.norm(y)
        if self.speaker_modulation is not None:
            if voice is None:
                raise ValueError('Speaker-conditioned mel blocks require a speaker embedding.')
            shift, scale = self.speaker_modulation(voice).unsqueeze(1).chunk(2, dim=-1)
            y = y * (1.0 + scale) + shift
        y = self.down(F.gelu(self.up(y)))
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
        conditioning_version: int = 2,
    ):
        super().__init__()
        if conditioning_version not in (2, 3, 4):
            raise ValueError('Obsolete conditioning version. Train a new Multispeaker model.')
        if content_bottleneck or content_bottleneck_noise:
            raise ValueError('Project full content features directly to the encoder width; intermediate bottlenecks are unsupported.')
        if conditioning_version == 2 and speaker_channels != hidden_channels:
            raise ValueError('Conditioning v2 requires hidden-width speaker embeddings.')
        self.conditioning_version = conditioning_version
        self.speaker_count = int(speaker_count)
        self.bottleneck = None
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
        self.has_null_speaker = conditioning_version == 2
        self.speaker = nn.Embedding(self.speaker_count + int(self.has_null_speaker), speaker_channels)
        self.speaker_proj = (nn.Linear(speaker_channels, hidden_channels, bias=False)
                             if conditioning_version >= 3 else nn.Identity())
        self.blocks = nn.ModuleList([ConvNeXtBlock(hidden_channels) for _ in range(layers)])
        self.voicing = nn.Conv1d(1, hidden_channels, 3, padding=1) if voicing else None
        self.tension = nn.Conv1d(1, hidden_channels, 3, padding=1) if tension else None
        self.content_norm = nn.LayerNorm(content_channels)
        nn.init.normal_(self.speaker.weight, std=speaker_channels ** -0.5)
        nn.init.xavier_uniform_(self.content.weight)
        nn.init.zeros_(self.content.bias)

    def voice(self, speaker: torch.Tensor) -> torch.Tensor:
        return self.speaker(speaker)

    def encode_content(self, content, mask):
        x = self.content(self.content_norm(content)).transpose(1, 2) * mask
        for block in self.blocks:
            x = block(x, mask)
        return x

    def forward(self, content, f0, energy, speaker, mask, breathiness=None, key_shift=None, speed=None,
                voicing=None, tension=None, harmonic_prior=None):
        frame_mask = mask[:, 0]
        f0, energy = f0 * frame_mask, energy * frame_mask
        breathiness = None if breathiness is None else breathiness * frame_mask
        voicing = None if voicing is None else voicing * frame_mask
        tension = None if tension is None else tension * frame_mask
        x = self.encode_content(content, mask)
        x = x + self.pitch(pitch_features(f0, self.pitch_fourier))
        if self.harmonic_prior is not None:
            prior = self.harmonic_prior(f0) if harmonic_prior is None else harmonic_prior
            if prior.shape[:2] != (f0.shape[0], self.harmonics.in_channels) or prior.shape[-1] != f0.shape[-1]:
                raise ValueError("Invalid cached harmonic-prior dimensions.")
            x = x + self.harmonics(prior)
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
        x = x + self.speaker_proj(self.voice(speaker)).unsqueeze(-1)
        x = x * mask
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
        self.norm = MixedPrecisionLayerNorm(channels, elementwise_affine=not adaln)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.up = nn.Linear(channels, inner * 2)
        self.mid = nn.Linear(inner, inner * 2)
        self.down = nn.Linear(inner, channels)
        self.modulation = None
        if adaln:
            self.modulation = nn.Linear(channels, channels * 3)
            nn.init.zeros_(self.modulation.weight)
            nn.init.zeros_(self.modulation.bias)

    def forward(self, x, embedding=None, fused=True):
        y = self.norm(x)
        gate = None
        if self.modulation is not None:
            shift, scale, gate = self.modulation(F.silu(embedding)).chunk(3, dim=-1)
            y = y + y * scale + shift
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        y = self.down(atan_glu(self.mid(atan_glu(self.up(y), fused)), fused))
        if gate is not None:
            y = y + gate * y
        return x + y


class LYNXNet2Backbone(nn.Module):
    def __init__(self, n_mels, cond_channels, channels=1024, layers=6, expansion=1, kernel_size=31,
                 adaln=False, time_scale=1000.0, speaker_channels=None):
        super().__init__()
        self.channels = int(channels)
        self.time_scale = float(time_scale)
        self.input = nn.Linear(n_mels, channels)
        self.input_cond = nn.Conv1d(cond_channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels)
        )
        self.layers = nn.ModuleList(
            [LYNXNet2Block(channels, expansion, kernel_size, adaln) for _ in range(layers)]
        )
        self.voice = nn.Linear(cond_channels if speaker_channels is None else speaker_channels, channels) if adaln else None
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def prepare_conditioning(self, cond, voice):
        projected = self.input_cond(cond).transpose(1, 2)
        speaker = self.voice(voice)[:, None, :] if self.voice is not None else None
        return projected, speaker

    def forward(self, x, t, cond, mask, voice=None, prepared=None):
        time = self.time_mlp(timestep_embedding(t.reshape(-1), self.channels, self.time_scale))
        time = time.view(t.shape[0], -1, self.channels)
        h = self.input(x.transpose(1, 2)).float()
        projected, speaker = self.prepare_conditioning(cond, voice) if prepared is None else prepared
        h = h + projected + time
        embedding = None
        if self.voice is not None:
            embedding = time + speaker
        for layer in self.layers:
            h = layer(h, embedding)
        h = self.norm(h)
        return self.output(h).float().transpose(1, 2)


class AuxDecoder(nn.Module):
    def __init__(self, cond_channels, n_mels, channels=512, layers=6, dropout=0.1,
                 speaker_channels=0):
        super().__init__()
        self.input = nn.Conv1d(cond_channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(channels, layer_scale=1e-6, dropout=dropout,
                           speaker_channels=speaker_channels) for _ in range(layers)]
        )
        self.output = nn.Conv1d(channels, n_mels, 7, padding=3)
        self.output.use_adamw = True

    def forward(self, cond, mask, voice=None):
        x = self.input(cond) * mask
        for block in self.blocks:
            x = block(x, mask, voice)
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
        speaker_channels: int = 384,
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
        sampling_steps: int = 20,
        conditioning_version: int = 2,
        flow_conditioning: str = "encoder",
        flow_loss: str = "l2",
        mel_mean: float = 0.0,
        mel_std: float = 1.0,
        direct_speaker_conditioning: bool = False,
    ):
        super().__init__()
        if backbone != "lynxnet2":
            raise ValueError(f"Only the lynxnet2 backbone is supported, not {backbone!r}.")
        if flow_conditioning not in {"encoder", "aux_mel"} or flow_loss not in {"l2", "l2_lognorm"}:
            raise ValueError("Invalid flow conditioning or loss.")
        if not math.isfinite(mel_mean) or not math.isfinite(mel_std) or mel_std <= 0:
            raise ValueError("Mel normalization must be finite with a positive scale.")
        adaln = bool((backbone_args or {}).get("adaln", False))
        if direct_speaker_conditioning and not adaln:
            raise ValueError('Direct speaker conditioning requires standard flow with speaker AdaLN.')
        if conditioning_version == 3 and (
                not direct_speaker_conditioning or flow_conditioning != 'encoder' or aux_decoder or t_start != 0.0):
            raise ValueError('Conditioning v3 requires direct encoder and speaker conditioning, no mel predictor, and t_start=0.')
        if flow_conditioning == "aux_mel" and (
                not aux_decoder or aux_grad != 1.0 or adaln != bool(direct_speaker_conditioning)):
            raise ValueError("Mel-conditioned flow requires standard flow, an auxiliary decoder, full gradients and matching speaker conditioning.")
        if conditioning_version == 4 and (
                not direct_speaker_conditioning or flow_conditioning != 'encoder' or not aux_decoder
                or not 0.0 < t_start < 1.0 or flow_loss != 'l2'):
            raise ValueError('Conditioning v4 requires speaker-conditioned shallow flow, encoder features, a mel predictor and L2 loss.')
        if not 0.0 <= t_start < 1.0 or (t_start > 0 and not aux_decoder):
            raise ValueError('Shallow flow requires a mel predictor and a start time between 0 and 1.')
        self.direct_speaker_conditioning = bool(direct_speaker_conditioning)
        self.flow_conditioning = flow_conditioning
        self.flow_loss = flow_loss
        self.mel_mean = float(mel_mean)
        self.mel_std = float(mel_std)
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
            conditioning_version,
        )
        cond_channels = n_mels if flow_conditioning == "aux_mel" else hidden_channels
        self.backbone = LYNXNet2Backbone(
            n_mels, cond_channels,
            speaker_channels=speaker_channels if self.direct_speaker_conditioning else None,
            **(backbone_args or {}),
        )


        conditioning = [self.backbone.time_mlp, self.encoder.speaker_proj, self.backbone.voice]
        conditioning += [layer.modulation for layer in self.backbone.layers]
        for module in filter(None, conditioning):
            for child in module.modules():
                child.use_adamw = True
        self.aux = AuxDecoder(
            hidden_channels, n_mels,
            speaker_channels=speaker_channels if self.direct_speaker_conditioning else 0,
            **aux_decoder,
        ) if aux_decoder else None
        self.t_start = float(t_start) if self.aux is not None else 0.0
        self.aux_grad = float(aux_grad)
        self.dual_timestep = bool(dual_timestep)
        if sampling_method not in SAMPLERS or int(sampling_steps) != sampling_steps or sampling_steps < 1:
            raise ValueError("Invalid flow sampling method or step count.")
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
        cond, predicted_mel = self._conditioning(cond, mask, voice)
        mixing = t[:, None, None] if t.ndim == 1 else t[:, None, :]
        x_t = (1.0 - mixing) * noise + mixing * mel
        prediction = backbone(x_t, t, cond, mask, voice)
        error = (prediction.float() - (mel - noise).float()).square() * mask
        if self.flow_loss == "l2_lognorm":
            times = t.float().clamp(1e-7, 1.0 - 1e-7)
            weights = 0.398942 / times / (1.0 - times) * torch.exp(-0.5 * torch.log(times / (1.0 - times)).square())
            error = error * (weights[:, None, None] if t.ndim == 1 else weights[:, None, :])
        flow = error.mean()
        count = (mask.sum((1, 2)) * self.n_mels).clamp_min(1.0)
        aux = self._mel_loss(predicted_mel, mel, mask, count.sum())
        return flow, aux

    @property
    def aux_loss_name(self):
        return "aux_mel_mse" if self.flow_conditioning == "aux_mel" else "aux_mel_l1"

    def _conditioning(self, cond, mask, voice=None):
        predicted_mel = None
        if self.aux is not None:
            aux_cond = cond * mask if self.flow_conditioning == "aux_mel" else cond * self.aux_grad + cond.detach() * (1.0 - self.aux_grad)
            predicted_mel = self.aux(aux_cond, mask, voice)
        if self.flow_conditioning == "aux_mel":
            cond = (predicted_mel * self.mel_std + self.mel_mean) * mask
        return cond, predicted_mel

    def _mel_loss(self, prediction, target, mask, denominator=None):
        if prediction is None:
            return None
        error = prediction.float() - target.float()
        error = (error * self.mel_std).square() if self.flow_conditioning == "aux_mel" else error.abs()
        denominator = (mask.sum() * self.n_mels).clamp_min(1.0) if denominator is None else denominator
        return (error * mask).sum() / denominator

    def predict_mel(self, content, f0, energy, speaker, mask, breathiness=None, key_shift=None,
                    speed=None, voicing=None, tension=None):
        if self.aux is None:
            raise ValueError("This model has no direct mel predictor.")
        cond = self.encoder(content, f0, energy, speaker, mask, breathiness, key_shift, speed, voicing, tension)
        return self.aux(cond, mask, self.encoder.voice(speaker))

    def _uniform_times(self, batch, device):
        t = torch.rand(batch, device=device)
        t = self.t_start + (1.0 - self.t_start) * t
        return t.clamp(1e-7, 1.0 - 1e-7) if self.flow_loss == "l2_lognorm" else t

    def forward(self, mel, content, f0, energy, speaker, mask, speaker_dropout=0.0,
                breathiness=None, key_shift=None, speed=None, backbone=None, voicing=None, tension=None,
                harmonic_prior=None):
        if (self.flow_conditioning == "aux_mel" or self.direct_speaker_conditioning) and speaker_dropout != 0:
            raise ValueError("Speaker-conditioned standard flow keeps the speaker ID present during training.")
        speaker = self._drop_speakers(speaker, speaker_dropout)
        cond = self.encoder(content, f0, energy, speaker, mask, breathiness, key_shift, speed, voicing, tension,
                            harmonic_prior=harmonic_prior)
        voice = self.encoder.voice(speaker)
        t = self._uniform_times(mel.shape[0], mel.device)
        if self.dual_timestep:
            t2 = self._uniform_times(mel.shape[0], mel.device)
            alternate = torch.rand(mel.shape[0], mel.shape[-1], device=mel.device) < 0.25
            t = torch.where(alternate & mask[:, 0].bool(), t2[:, None], t[:, None])

        backbone = self.backbone if backbone is None else backbone
        flow, aux = self._losses(mel, cond, voice, mask, t, torch.randn_like(mel), backbone)
        return flow, aux

    @torch.no_grad()
    def validation_losses(self, mel, content, f0, energy, speaker, mask, breathiness, key_shift,
                          speed, noise, fractions, voicing=None, tension=None, harmonic_prior=None):
        cond = self.encoder(content, f0, energy, speaker, mask, breathiness, key_shift, speed, voicing, tension,
                            harmonic_prior=harmonic_prior)
        voice = self.encoder.voice(speaker)
        losses, aux = [], None
        for fraction in fractions:
            t = torch.full((mel.shape[0],), self.t_start + (1.0 - self.t_start) * fraction, device=mel.device)
            flow, aux = self._losses(mel, cond, voice, mask, t, noise, self.backbone)
            losses.append(flow)
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
        voicing: Optional[torch.Tensor] = None,
        tension: Optional[torch.Tensor] = None,
    ):
        method = self.sampling_method if method is None else method
        steps = self.sampling_steps if steps is None else steps
        if method not in SAMPLERS:
            raise ValueError(f"method must be one of {SAMPLERS}, not {method!r}.")
        if rescale_mode not in RESCALE_MODES:
            raise ValueError(f"rescale_mode must be one of {RESCALE_MODES}, not {rescale_mode!r}.")
        batch = content.shape[0]
        if (self.flow_conditioning == "aux_mel" or self.direct_speaker_conditioning) and cfg_scale != 1.0:
            raise ValueError("Speaker-conditioned standard flow trains without speaker dropout; use cfg_scale=1.")
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
        initial_mel = None
        if self.flow_conditioning == "aux_mel":
            cond, initial_mel = self._conditioning(cond, masks, voice)
        guide_from, guide_until = (float(value) for value in guidance_interval)

        def spread(y):
            if rescale_mode == "frame":
                return y.square().mean(1, keepdim=True).sqrt()
            frames = mask.sum((1, 2)).clamp_min(1.0) * self.n_mels
            return ((y.square() * mask).sum((1, 2)) / frames).sqrt()[:, None, None]

        prepared = self.backbone.prepare_conditioning(cond, voice)
        primary = tuple(value[:batch] if value is not None else None for value in prepared)

        def field(x, t):
            if count == 1:
                return self.backbone(x, t, cond[:batch], mask, voice[:batch], prepared=primary)
            now = float(t[0])

            if not (guide_from <= now and (now < guide_until or guide_until >= 1.0)):
                return self.backbone(x, t, cond[:batch], mask, voice[:batch], prepared=primary)
            v = self.backbone(repeat(x), repeat(t), cond, masks, voice, prepared=prepared).chunk(count)
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
            initial = self.aux(cond[:batch], mask, voice[:batch]) if initial_mel is None else initial_mel[:batch]
            x = ((1.0 - t0) * noise + t0 * initial) * mask
        else:
            x = noise * mask
        times = time_grid(schedule, max(1, int(steps)), t0, x.device)
        dt = (1.0 - t0) / (times.shape[0] - 1)
        for index in range(times.shape[0] - 1):
            t = times[index].expand(batch)
            v = field(x, t)
            if method == "rk2":
                v = field(x + 0.5 * dt * v, t + 0.5 * dt)
            elif method == "rk4":
                middle = t + 0.5 * dt
                k2 = field(x + 0.5 * dt * v, middle)
                k3 = field(x + 0.5 * dt * k2, middle)
                k4 = field(x + dt * k3, t + dt)
                v = (v + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            elif method == "rk5":
                k2 = field(x + 0.25 * v * dt, t + 0.25 * dt)
                k3 = field(x + 0.125 * (k2 + v) * dt, t + 0.25 * dt)
                k4 = field(x + 0.5 * (-k2 + 2.0 * k3) * dt, t + 0.5 * dt)
                k5 = field(x + 0.0625 * (3.0 * v + 9.0 * k4) * dt, t + 0.75 * dt)
                k6 = field(x + (-3.0 * v + 2.0 * k2 + 12.0 * k3 - 12.0 * k4 + 8.0 * k5) * dt / 7.0,
                           t + dt)
                v = (7.0 * v + 32.0 * k3 + 12.0 * k4 + 32.0 * k5 + 7.0 * k6) / 90.0
            x = x + dt * v
            if callback is not None:
                callback()
        return x * mask


def resize_speakers(state_dict: dict, speaker_count: int, speaker_init=None, null_speaker: bool = True) -> dict:
    key = "encoder.speaker.weight"
    table = state_dict[key]
    if speaker_init is not None:
        if speaker_init.shape != (speaker_count + int(null_speaker), table.shape[1]):
            raise ValueError('Invalid target speaker initialization.')
        state_dict = dict(state_dict)
        rows = speaker_init.detach().to(table).clone()
        state_dict[key] = torch.cat((rows[:-1], table[-1:]), dim=0) if null_speaker else rows
        return state_dict
    if table.shape[0] == speaker_count + int(null_speaker):
        return state_dict
    trained = table[:-1] if null_speaker else table
    rows = trained.mean(0, keepdim=True).expand(speaker_count, -1).clone()
    state_dict = dict(state_dict)
    state_dict[key] = torch.cat((rows, table[-1:]), dim=0) if null_speaker else rows
    return state_dict


def validate_model_config(model: dict):
    source = model.get('flow_conditioning', 'encoder')
    direct = bool(model.get('direct_speaker_conditioning', False))
    adaln = bool(model.get('backbone_args', {}).get('adaln', False))
    if model.get('mean_flow', False):
        raise ValueError('MeanFlow checkpoints are no longer supported. Train a new standard flow experiment.')
    if direct and not adaln:
        raise ValueError('Direct speaker conditioning requires standard flow with speaker AdaLN.')
    version = model.get('conditioning_version')
    if version == 3 and (
            not direct or source != 'encoder' or model.get('aux_decoder') or model.get('t_start', 0.0) != 0.0):
        raise ValueError('Conditioning v3 requires direct encoder and speaker conditioning, no mel predictor, and t_start=0.')
    if source not in {'encoder', 'aux_mel'} or model.get('flow_loss', 'l2') not in {'l2', 'l2_lognorm'}:
        raise ValueError('Unsupported flow conditioning or loss.')
    if source == 'aux_mel' and (
            not model.get('aux_decoder')
            or model.get('aux_grad', 0.1) != 1.0 or adaln != direct):
        raise ValueError('Mel-conditioned flow requires a standard model with a fully trained mel predictor and matching speaker conditioning.')
    if (version not in (2, 3, 4)
            or any(model.get(name, False) for name in ('voicing', 'tension'))):
        raise ValueError('Unsupported rectified-flow recipe or checkpoint. Use a supported standard-flow recipe.')
    if model.get('content_bottleneck', 0) or model.get('content_bottleneck_noise', 0):
        raise ValueError('Project full content features directly to the encoder width; intermediate bottlenecks are unsupported.')
    if version == 2 and model.get('speaker_channels', 384) != model.get('hidden_channels', 384):
        raise ValueError('Conditioning v2 requires hidden-width speaker embeddings.')
    if model.get('sampling_method', 'euler') not in SAMPLERS:
        raise ValueError(f'Flow sampler must be one of {SAMPLERS}.')
    if version == 4 and (
            not direct or source != 'encoder' or not model.get('aux_decoder')
            or not 0.0 < model.get('t_start', 0.0) < 1.0 or model.get('flow_loss', 'l2') != 'l2'):
        raise ValueError('Conditioning v4 requires speaker-conditioned shallow flow, encoder features, a mel predictor and L2 loss.')



def build_flow(config: dict, speaker_count: int) -> RectifiedFlow:
    config = resolve_config(config)
    model = dict(config["flow"]["model"])
    validate_model_config(model)
    model.pop("mean_flow", None)
    data = config["data"]
    if model.get('flow_conditioning', 'encoder') == 'aux_mel':
        model['mel_mean'] = float(data['mel_mean'])
        model['mel_std'] = float(data['mel_std'])
    if model.pop("harmonic_prior", False):
        model["harmonic_prior"] = dict(
            sample_rate=data["sample_rate"], n_fft=data["n_fft"],
            fmin=data["mel_fmin"], fmax=data["mel_fmax"],
        )
    return RectifiedFlow(n_mels=data["n_mels"], speaker_count=speaker_count, **model)
