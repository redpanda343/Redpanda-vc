import math

import torch
from torch import nn
from torch.nn import functional as F


def adamw_linear(inputs, outputs):
    layer = nn.Linear(inputs, outputs)
    layer.use_adamw = True
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def interpolate_pitch(f0, mask):
    frames = f0.shape[-1]
    positions = torch.arange(frames, device=f0.device).expand_as(f0)
    voiced = (f0 > 0) & mask[:, 0].bool()
    left = torch.where(voiced, positions, -1).cummax(dim=-1).values
    right = torch.where(voiced, positions, frames).flip(-1).cummin(dim=-1).values.flip(-1)
    left = torch.where(left < 0, right, left).clamp(0, frames - 1)
    right = torch.where(right == frames, left, right).clamp(0, frames - 1)
    logs = f0.clamp_min(1.0).log2()
    weight = (positions - left).float() / (right - left).clamp_min(1)
    pitch = torch.pow(2.0, logs.gather(1, left) + weight * (logs.gather(1, right) - logs.gather(1, left)))
    return pitch * voiced.any(dim=-1, keepdim=True) * mask[:, 0]


class ContentAttention(nn.Module):
    def __init__(self, channels, heads=2):
        super().__init__()
        if channels % (heads * 2):
            raise ValueError('Content encoder width must be divisible by twice the number of heads.')
        self.heads = heads
        self.head_dim = channels // heads
        self.in_proj = nn.Linear(channels, channels * 3, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.register_buffer('inv_freq', 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)), persistent=False)

    def rotate(self, x):
        angles = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * self.inv_freq.float()[None]
        angles = torch.cat((angles, angles), dim=-1)
        first, second = x.chunk(2, dim=-1)
        return x * angles.cos().to(x.dtype) + torch.cat((-second, first), dim=-1) * angles.sin().to(x.dtype)

    def forward(self, x, padding):
        batch, frames, channels = x.shape
        q, k, v = [value.view(batch, frames, self.heads, self.head_dim).transpose(1, 2)
                   for value in self.in_proj(x).split(channels, dim=-1)]
        q, k = self.rotate(q), self.rotate(k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(padding[:, None, None], -torch.inf)
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, frames, channels)
        return self.out_proj(output)


class ContentEncoderLayer(nn.Module):
    def __init__(self, channels, heads=2, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.layer_norm1 = nn.LayerNorm(channels)
        self.self_attn = ContentAttention(channels, heads)
        self.layer_norm2 = nn.LayerNorm(channels)
        self.ffn_1 = nn.Conv1d(channels, channels * 4, 3, padding=1)
        self.ffn_2 = nn.Linear(channels * 4, channels)
        nn.init.xavier_uniform_(self.ffn_2.weight)
        nn.init.zeros_(self.ffn_2.bias)

    def forward(self, x, padding):
        mask = (~padding).unsqueeze(-1).to(x.dtype)
        y = self.self_attn(self.layer_norm1(x), padding)
        x = (x + F.dropout(y, self.dropout, training=self.training)) * mask
        y = self.ffn_1(self.layer_norm2(x).transpose(1, 2)).transpose(1, 2) * (3 ** -0.5)
        y = F.dropout(F.gelu(y), self.dropout, training=self.training)
        y = self.ffn_2(y)
        return (x + F.dropout(y, self.dropout, training=self.training)) * mask


class ContentConditionEncoder(nn.Module):
    def __init__(self, content_channels, hidden_channels, speaker_count, layers,
                 breathiness=False, key_shift=False, speed=False, energy=False):
        super().__init__()
        self.conditioning_version = 5
        self.speaker_count = int(speaker_count)
        self.has_null_speaker = False
        self.content = nn.Linear(content_channels, hidden_channels)
        nn.init.xavier_uniform_(self.content.weight)
        nn.init.zeros_(self.content.bias)
        self.blocks = nn.ModuleList([ContentEncoderLayer(hidden_channels) for _ in range(layers)])
        self.norm = nn.LayerNorm(hidden_channels)
        self.embed_scale = math.sqrt(hidden_channels)
        self.pitch = adamw_linear(1, hidden_channels)
        self.energy = adamw_linear(1, hidden_channels) if energy else None
        self.breathiness = adamw_linear(1, hidden_channels) if breathiness else None
        self.key_shift = adamw_linear(1, hidden_channels) if key_shift else None
        self.speed = adamw_linear(1, hidden_channels) if speed else None
        self.speaker = nn.Embedding(speaker_count, hidden_channels)
        nn.init.normal_(self.speaker.weight, std=hidden_channels ** -0.5)

    def voice(self, speaker):
        return self.speaker(speaker)

    def forward(self, content, f0, energy, speaker, mask, breathiness=None,
                key_shift=None, speed=None, voicing=None, tension=None, harmonic_prior=None):
        padding = ~mask[:, 0].bool()
        x = F.dropout(self.content(content) * self.embed_scale, 0.1, training=self.training)
        x = x * mask.transpose(1, 2)
        for block in self.blocks:
            x = block(x, padding)
        x = self.norm(x) * mask.transpose(1, 2)
        x = x + self.voice(speaker).unsqueeze(1)
        x = x + self.pitch(torch.log1p(interpolate_pitch(f0, mask) / 700.0).unsqueeze(-1))
        for layer, values, scale in ((self.energy, energy, 1.0 / 96),
                                     (self.breathiness, breathiness, 1.0 / 96)):
            if layer is not None:
                if values is None:
                    raise ValueError('Missing required variance conditioning.')
                db = (values - 1.0) * 35.0 if layer is self.energy else values
                x = x + layer(db.unsqueeze(-1) * scale)
        if self.key_shift is not None:
            values = f0.new_zeros(f0.shape[0]) if key_shift is None else key_shift
            x = x + self.key_shift(values.reshape(-1, 1, 1) / 12.0)
        if self.speed is not None:
            values = f0.new_ones(f0.shape[0]) if speed is None else speed
            x = x + self.speed(values.reshape(-1, 1, 1))
        return x.transpose(1, 2)
