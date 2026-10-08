import math

import torch
from torch import nn
from torch.nn import functional as F

from rectified_flow.variance import VARIANCE_SCALE


def adamw_linear(inputs, outputs):
    layer = nn.Linear(inputs, outputs)
    layer.use_adamw = True
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def expand_content(x, frames, step, lengths):
    positions = torch.arange(frames, device=x.device, dtype=torch.float64)[None] * step.double()[:, None]
    limit = (2 * lengths - 1)[:, None]
    positions = torch.minimum(positions, limit.double())
    left = positions.floor().long()
    right = torch.minimum(left + 1, limit)
    weight = (positions - left).to(x.dtype).unsqueeze(-1)

    def gather(index):
        return x.gather(1, (index // 2).unsqueeze(-1).expand(-1, -1, x.shape[-1]))

    return gather(left) * (1 - weight) + gather(right) * weight


class ContentAttention(nn.Module):
    def __init__(self, channels, heads=2, use_rope=True, rope_interleaved=False, rope_theta=10000.0):
        super().__init__()
        if channels % heads or use_rope and channels % (heads * 2):
            raise ValueError('Content encoder width must be divisible by twice the number of heads.')
        self.use_rope = bool(use_rope)
        self.rope_interleaved = bool(rope_interleaved)
        self.heads = heads
        self.head_dim = channels // heads
        self.in_proj = nn.Linear(channels, channels * 3, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.register_buffer('inv_freq', 1.0 / (rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)), persistent=False)

    def rotate(self, x):
        angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * self.inv_freq.float()[None]
        if self.rope_interleaved:
            angles = angles.repeat_interleave(2, dim=-1)
            rotated = torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)
        else:
            angles = torch.cat((angles, angles), dim=-1)
            first, second = x.chunk(2, dim=-1)
            rotated = torch.cat((-second, first), dim=-1)
        return x * angles.cos().to(x.dtype) + rotated * angles.sin().to(x.dtype)

    def forward(self, x, padding):
        batch, frames, channels = x.shape
        q, k, v = [value.view(batch, frames, self.heads, self.head_dim).transpose(1, 2)
                   for value in self.in_proj(x).split(channels, dim=-1)]
        if self.use_rope:
            q, k = self.rotate(q), self.rotate(k)
        output = F.scaled_dot_product_attention(q, k, v, attn_mask=~padding[:, None, None])
        return self.out_proj(output.transpose(1, 2).contiguous().view(batch, frames, channels))


class ContentEncoderLayer(nn.Module):
    def __init__(self, channels, heads=2, dropout=0.1, kernel_size=3, use_rope=True,
                 rope_interleaved=False, rope_theta=10000.0):
        super().__init__()
        self.dropout = dropout
        self.layer_norm1 = nn.LayerNorm(channels)
        self.self_attn = ContentAttention(channels, heads, use_rope, rope_interleaved, rope_theta)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.ffn_scale = kernel_size ** -0.5
        self.ffn_1 = nn.Conv1d(channels, channels * 4, kernel_size, padding=kernel_size // 2)
        self.ffn_2 = nn.Linear(channels * 4, channels)
        nn.init.xavier_uniform_(self.ffn_2.weight)
        nn.init.zeros_(self.ffn_2.bias)

    def forward(self, x, padding):
        mask = (~padding).unsqueeze(-1).to(x.dtype)
        y = self.self_attn(self.layer_norm1(x), padding)
        x = (x + F.dropout(y, self.dropout, training=self.training)) * mask
        y = self.ffn_1(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2) * self.ffn_scale
        y = F.dropout(F.gelu(y), self.dropout, training=self.training)
        y = self.ffn_2(y)
        return (x + F.dropout(y, self.dropout, training=self.training)) * mask


class ContentConditionEncoder(nn.Module):
    def __init__(self, content_channels, hidden_channels, speaker_count, layers, content_step,
                 key_shift=True, speed=True, use_spk_id=False, enc_ffn_kernel_size=3,
                 use_rope=True, rope_interleaved=False, rope_theta=10000.0, variances=()):
        super().__init__()
        if not content_step > 0:
            raise ValueError('Content encoding requires the mel hop in content frames.')
        self.content_step = float(content_step)
        self.speaker_count = int(speaker_count) if use_spk_id else 1
        self.content = nn.Linear(content_channels, hidden_channels)
        nn.init.xavier_uniform_(self.content.weight)
        nn.init.zeros_(self.content.bias)
        self.blocks = nn.ModuleList([
            ContentEncoderLayer(hidden_channels, kernel_size=enc_ffn_kernel_size, use_rope=use_rope,
                                rope_interleaved=rope_interleaved, rope_theta=rope_theta) for _ in range(layers)
        ])
        self.norm = nn.LayerNorm(hidden_channels)
        self.embed_scale = math.sqrt(hidden_channels)
        self.pitch = adamw_linear(1, hidden_channels)
        self.key_shift = adamw_linear(1, hidden_channels) if key_shift else None
        self.speed = adamw_linear(1, hidden_channels) if speed else None
        self.variance_names = list(variances)
        self.variance_embeds = nn.ModuleDict({name: adamw_linear(1, hidden_channels) for name in self.variance_names})
        self.speaker = nn.Embedding(speaker_count, hidden_channels) if use_spk_id else None
        if self.speaker is not None:
            nn.init.normal_(self.speaker.weight, std=hidden_channels ** -0.5)

    def init_content_scale(self, rms):
        fan_in, fan_out = self.content.in_features, self.content.out_features
        nn.init.normal_(self.content.weight, std=1.0 / (math.sqrt(fan_in * fan_out) * float(rms)))
        nn.init.zeros_(self.content.bias)

    def forward(self, content, f0, speaker, mask, content_mask=None, key_shift=None, speed=None, variances=None):
        if content_mask is None:
            content_mask = mask.new_ones(content.shape[0], 1, content.shape[1])
        padding = ~content_mask[:, 0].bool()
        x = F.dropout(self.content(content) * self.embed_scale, 0.1, training=self.training)
        x = x * content_mask.transpose(1, 2)
        for block in self.blocks:
            x = block(x, padding)
        x = self.norm(x) * content_mask.transpose(1, 2)
        speeds = (f0.new_ones(f0.shape[0]) if speed is None else speed.reshape(-1)).double()
        lengths = content_mask[:, 0].sum(-1).long()
        x = expand_content(x, mask.shape[-1], speeds * self.content_step, lengths) * mask.transpose(1, 2)
        if self.speaker is not None:
            x = x + self.speaker(speaker).unsqueeze(1)
        x = x + self.pitch(torch.log1p(f0 * mask[:, 0] / 700.0).unsqueeze(-1))
        if self.variance_names:
            if variances is None or variances.shape[1] != len(self.variance_names):
                raise ValueError(f'This flow needs {", ".join(self.variance_names)} curves.')
            x = x + torch.stack([self.variance_embeds[name](variances[:, index, :, None] * VARIANCE_SCALE)
                                 for index, name in enumerate(self.variance_names)], dim=-1).sum(-1)
        if self.key_shift is not None:
            values = f0.new_zeros(f0.shape[0]) if key_shift is None else key_shift
            x = x + self.key_shift(values.reshape(-1, 1, 1) / 12.0)
        if self.speed is not None:
            values = f0.new_ones(f0.shape[0]) if speed is None else speed
            x = x + self.speed(values.reshape(-1, 1, 1))
        return x.transpose(1, 2)
