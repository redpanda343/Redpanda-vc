import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from rvc.rectified.config import resolve_config
from rvc.rectified.content_encoder import ContentConditionEncoder
from rvc.rectified.variance import variance_names

SAMPLERS = ("euler", "rk2", "rk4", "rk5")
LEGACY_SAMPLER = ("euler", 20)
DEFAULT_SAMPLER = ("rk2", 10)


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
    def __init__(self, channels: int, layer_scale: float = 1e-6, dropout: float = 0.0, kernel_size: int = 7):
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.norm = MixedPrecisionLayerNorm(channels, eps=1e-6)
        self.up = nn.Linear(channels, channels * 4)
        self.down = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.depthwise(x).transpose(1, 2))
        y = self.down(F.gelu(self.up(y))) * self.gamma
        return x + self.dropout(y.transpose(1, 2))


def timestep_embedding(t: torch.Tensor, channels: int, scale: float = 1000.0) -> torch.Tensor:
    half = channels // 2
    frequencies = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000.0) / (half - 1)))
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


class _SoftSignGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, out, gate):
        softsign_gate = F.softsign(gate)
        ctx.save_for_backward(softsign_gate, out * (1.0 - softsign_gate.abs()).square())
        return out * softsign_gate

    @staticmethod
    def backward(ctx, grad):
        softsign_gate, decay_out = ctx.saved_tensors
        return grad * softsign_gate, grad * decay_out


def softsign_glu(x, fused=True):
    out, gate = x.chunk(2, dim=-1)
    if fused and torch.is_grad_enabled():
        return _SoftSignGLU.apply(out, gate)
    return out * F.softsign(gate)


class LYNXNet2Block(nn.Module):
    def __init__(self, channels, expansion, kernel_size, glu_type='atanglu', dropout_rate=0.0):
        super().__init__()
        inner = int(channels * expansion)
        if glu_type not in {'atanglu', 'softsign_glu'}:
            raise ValueError(f'Unsupported flow activation: {glu_type!r}.')
        self.glu_type = glu_type
        self.use_fused_kernels = False
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = MixedPrecisionLayerNorm(channels)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, groups=channels)
        self.up = nn.Linear(channels, inner * 2)
        self.mid = nn.Linear(inner, inner * 2)
        self.down = nn.Linear(inner, channels)

    def forward(self, x, fused=None):
        fused = self.training if fused is None else fused
        y = self.depthwise(self.norm(x).transpose(1, 2)).transpose(1, 2)
        if self.training and self.use_fused_kernels:
            from rvc.rectified.kernels.fused_linear_softsign_glu import fused_linear_softsign_glu

            y = fused_linear_softsign_glu(y, self.up.weight, self.up.bias)
            y = fused_linear_softsign_glu(y, self.mid.weight, self.mid.bias)
        else:
            activation = softsign_glu if self.glu_type == 'softsign_glu' else atan_glu
            y = activation(self.mid(activation(self.up(y), fused)), fused)
        return x + self.dropout(self.down(y))


class LYNXNet2Backbone(nn.Module):
    def __init__(self, n_mels, cond_channels, channels=1024, layers=6, expansion=1, kernel_size=31,
                 glu_type='atanglu', dropout_rate=0.0, use_conditioner_cache=True):
        super().__init__()
        self.channels = int(channels)
        self.use_conditioner_cache = bool(use_conditioner_cache)
        self.input = nn.Linear(n_mels, channels)
        self.input_cond = nn.Conv1d(cond_channels, channels, 1)
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels)
        )
        self.layers = nn.ModuleList(
            [LYNXNet2Block(channels, expansion, kernel_size, glu_type, dropout_rate) for _ in range(layers)]
        )
        self.norm = nn.LayerNorm(channels)
        self.output = nn.Linear(channels, n_mels)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.input.weight)
        nn.init.kaiming_normal_(self.input_cond.weight)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def prepare_conditioning(self, cond):
        return self.input_cond(cond).transpose(1, 2)

    def forward(self, x, t, cond, prepared=None):
        time = self.time_mlp(timestep_embedding(t.reshape(-1), self.channels))
        time = time.view(t.shape[0], -1, self.channels)
        h = self.input(x.transpose(1, 2))
        h = h + (self.prepare_conditioning(cond) if prepared is None else prepared) + time
        for layer in self.layers:
            h = layer(h)
        return self.output(self.norm(h)).transpose(1, 2)


class AuxDecoder(nn.Module):
    def __init__(self, cond_channels, n_mels, channels=512, layers=6, dropout=0.1, kernel_size=7):
        super().__init__()
        self.input = nn.Conv1d(cond_channels, channels, 7, padding=3)
        self.blocks = nn.ModuleList(
            [ConvNeXtBlock(channels, dropout=dropout, kernel_size=kernel_size) for _ in range(layers)]
        )
        self.output = nn.Conv1d(channels, n_mels, 7, padding=3)
        self.output.use_adamw = True
        nn.init.kaiming_normal_(self.output.weight)

    def forward(self, cond):
        x = self.input(cond)
        for block in self.blocks:
            x = block(x)
        return self.output(x)


class RectifiedFlow(nn.Module):
    def __init__(
        self,
        n_mels: int,
        speaker_count: int,
        content_step: float,
        mel_mean: float,
        mel_std: float,
        content_channels: int = 768,
        hidden_channels: int = 384,
        encoder_layers: int = 4,
        enc_ffn_kernel_size: int = 3,
        use_rope: bool = True,
        rope_interleaved: bool = False,
        rope_theta: float = 10000.0,
        use_spk_id: bool = False,
        key_shift: bool = True,
        speed: bool = True,
        use_breathiness_embed: bool = False,
        use_voicing_embed: bool = False,
        backbone_args: Optional[dict] = None,
        aux_decoder: Optional[dict] = None,
        aux_grad: float = 0.1,
        t_start: float = 0.4,
        t_start_infer: float = 0.4,
        dual_timestep: bool = True,
        sampling_method: str = "euler",
        sampling_steps: int = 20,
        train_aux_decoder: bool = True,
        train_diffusion: bool = True,
        val_gt_start: bool = False,
    ):
        super().__init__()
        if not train_diffusion and not train_aux_decoder:
            raise ValueError('Enable at least one training objective.')
        if not 0.0 <= t_start < 1.0 or not 0.0 <= t_start_infer <= 1.0:
            raise ValueError('Shallow flow start times must be between 0 and 1.')
        if sampling_method not in SAMPLERS or int(sampling_steps) != sampling_steps or sampling_steps < 1:
            raise ValueError("Invalid flow sampling method or step count.")
        if not math.isfinite(mel_mean) or not math.isfinite(mel_std) or mel_std <= 0:
            raise ValueError("Mel normalization must be finite with a positive scale.")
        self.n_mels = int(n_mels)
        self.mel_mean = float(mel_mean)
        self.mel_std = float(mel_std)
        self.use_spk_id = bool(use_spk_id)
        self.encoder = ContentConditionEncoder(
            content_channels, hidden_channels, speaker_count, encoder_layers, content_step,
            key_shift=key_shift, speed=speed, use_spk_id=self.use_spk_id, enc_ffn_kernel_size=enc_ffn_kernel_size,
            use_rope=use_rope, rope_interleaved=rope_interleaved, rope_theta=rope_theta,
            variances=variance_names(dict(use_breathiness_embed=use_breathiness_embed,
                                          use_voicing_embed=use_voicing_embed)),
        )
        self.backbone = LYNXNet2Backbone(n_mels, hidden_channels, **(backbone_args or {}))
        self.aux = AuxDecoder(hidden_channels, n_mels, **(aux_decoder or {}))
        self.t_start = float(t_start)
        self.t_start_infer = float(t_start_infer)
        self.train_aux_decoder = bool(train_aux_decoder)
        self.train_diffusion = bool(train_diffusion)
        self.val_gt_start = bool(val_gt_start)
        if not self.train_aux_decoder:
            self.aux.requires_grad_(False)
        if not self.train_diffusion:
            self.backbone.requires_grad_(False)
        self.aux_grad = float(aux_grad)
        self.dual_timestep = bool(dual_timestep)
        self.sampling_method = sampling_method
        self.sampling_steps = int(sampling_steps)

    @property
    def speaker_count(self) -> int:
        return self.encoder.speaker_count

    @property
    def variance_names(self):
        return self.encoder.variance_names

    def _uniform_times(self, batch, device):
        return self.t_start + (1.0 - self.t_start) * torch.rand(batch, device=device)

    def forward(self, mel, content, f0, speaker, mask, content_mask=None, key_shift=None, speed=None, variances=None):
        cond = self.encoder(content, f0, speaker, mask, content_mask, key_shift, speed, variances)
        t = self._uniform_times(mel.shape[0], mel.device)
        if self.dual_timestep:
            t2 = self._uniform_times(mel.shape[0], mel.device)
            alternate = torch.rand(mel.shape[0], mel.shape[-1], device=mel.device) < 0.25
            t = torch.where(alternate, t2[:, None], t[:, None])
        noise = torch.randn_like(mel)
        predicted = None
        if self.train_aux_decoder:
            predicted = self.aux(cond * self.aux_grad + cond.detach() * (1.0 - self.aux_grad))
        aux = F.l1_loss(predicted, mel) if predicted is not None else None
        if not self.train_diffusion:
            return cond.sum() * 0.0, aux
        mixing = t[:, None, None] if t.ndim == 1 else t[:, None, :]
        x_t = (1.0 - mixing) * noise + mixing * mel
        prediction = self.backbone(x_t, t, cond)
        flow = ((prediction.float() - (mel - noise).float()).square() * mask).mean()
        return flow, aux

    def predict_mel(self, content, f0, speaker, mask, content_mask=None, key_shift=None, speed=None, variances=None):
        return self.aux(self.encoder(content, f0, speaker, mask, content_mask, key_shift, speed, variances)) * mask

    @torch.no_grad()
    def sample(
        self,
        content,
        f0,
        speaker,
        mask,
        content_mask: Optional[torch.Tensor] = None,
        key_shift: Optional[torch.Tensor] = None,
        speed: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,
        method: Optional[str] = None,
        noise: Optional[torch.Tensor] = None,
        source_mel: Optional[torch.Tensor] = None,
        variances: Optional[torch.Tensor] = None,
    ):
        method = self.sampling_method if method is None else method
        steps = max(1, int(self.sampling_steps if steps is None else steps))
        if method not in SAMPLERS:
            raise ValueError(f"method must be one of {SAMPLERS}, not {method!r}.")
        cond = self.encoder(content, f0, speaker, mask, content_mask, key_shift, speed, variances)
        prepared = self.backbone.prepare_conditioning(cond) if self.backbone.use_conditioner_cache else None

        def field(x, t):
            return self.backbone(x, t, cond, prepared=prepared)

        batch = content.shape[0]
        if noise is None:
            noise = torch.randn((batch, self.n_mels, mask.shape[-1]), device=content.device)
        t0 = self.t_start_infer
        initial = source_mel if source_mel is not None else self.aux(cond)
        initial = initial * mask + (1.0 - mask) * (-self.mel_mean / self.mel_std)
        x = (1.0 - t0) * noise + t0 * initial
        times = t0 + torch.arange(steps + 1, device=x.device) * ((1.0 - t0) / steps)
        dt = (1.0 - t0) / steps
        for index in range(steps):
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
        return x * mask


def validate_model_config(model: dict):
    for name in ('use_rope', 'rope_interleaved', 'use_spk_id', 'key_shift', 'speed', 'dual_timestep',
                 'train_aux_decoder', 'train_diffusion', 'val_gt_start', 'use_breathiness_embed',
                 'use_voicing_embed'):
        if not isinstance(model[name], bool):
            raise ValueError(f'{name} must be a boolean.')
    for section, key in ((model, 'enc_ffn_kernel_size'), (model['aux_decoder'], 'kernel_size'),
                         (model['backbone_args'], 'kernel_size')):
        value = section[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value % 2 == 0:
            raise ValueError(f'{key} must be a positive odd integer.')
    if not math.isfinite(model['rope_theta']) or model['rope_theta'] <= 0:
        raise ValueError('rope_theta must be finite and positive.')
    if model['backbone_args']['glu_type'] not in {'atanglu', 'softsign_glu'}:
        raise ValueError(f"Unsupported flow activation: {model['backbone_args']['glu_type']!r}.")
    if model['sampling_method'] not in SAMPLERS:
        raise ValueError(f'Flow sampler must be one of {SAMPLERS}.')


def build_flow(config: dict, speaker_count: int) -> RectifiedFlow:
    config = resolve_config(config)
    model = config["flow"]["model"]
    validate_model_config(model)
    if (model["sampling_method"], model["sampling_steps"]) == LEGACY_SAMPLER:
        model = dict(model, sampling_method=DEFAULT_SAMPLER[0], sampling_steps=DEFAULT_SAMPLER[1])
    data = config["data"]
    return RectifiedFlow(
        n_mels=data["n_mels"], speaker_count=speaker_count,
        content_step=data['hop_length'] * 100 / data['sample_rate'],
        mel_mean=float(data['mel_mean']), mel_std=float(data['mel_std']), **model,
    )
