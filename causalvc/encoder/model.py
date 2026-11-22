"""Causal streaming content encoder, distilled from ContentVec.

16 kHz waveform -> 768-d content features at 50 Hz, in ContentVec's feature
space (so RVC models and their indexes accept them unchanged).

Causality: output frame t depends on samples up to 320*(t+1) only. With
`lookahead` = k the model's output at step t is trained to be the teacher's
frame t-k, i.e. it has seen 20*k ms of future. Algorithmic latency is therefore
20 ms (one frame) + 20*k ms.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SR = 16000
HOP = 320  # 20 ms: ContentVec's frame rate


def rotary(x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Rotary position embedding on (B, H, T, D); pos is (T,) absolute frame index."""
    d = x.shape[-1] // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(d, device=x.device) / d)
    ang = pos[:, None].float() * freq[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, ffn: int, kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        # Causal depthwise conv: local detail without attention.
        self.kernel = kernel
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim)
        self.conv_out = nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.n1(x)).view(b, t, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(rotary(q, pos), rotary(k, pos), v, attn_mask=mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, c)))
        h = F.pad(self.n2(x).transpose(1, 2), (self.kernel - 1, 0))
        x = x + self.drop(self.conv_out(F.gelu(self.conv(h).transpose(1, 2))))
        return x + self.drop(self.ff(self.n3(x)))


class StreamingEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        layers: int = 8,
        heads: int = 8,
        ffn: int = 2048,
        out_dim: int = 768,
        n_mels: int = 80,
        window: int = 150,  # frames of left context attention may use (3 s at 50 Hz)
        lookahead: int = 2,  # frames of future (hop / 16 samples ms each)
        dropout: float = 0.0,
        hop: int = 320,  # samples per frame at 16 kHz: 320 (50 Hz) or 160 (100 Hz)
    ):
        super().__init__()
        assert hop in (160, 320)
        self.window, self.lookahead, self.hop = window, lookahead, hop
        self.mel = torchaudio.transforms.MelSpectrogram(
            SR, n_fft=400, hop_length=160, n_mels=n_mels, center=False, power=1.0
        )
        self.conv1 = nn.Conv1d(n_mels, dim, 3)
        self.conv2 = nn.Conv1d(dim, dim, 4, stride=hop // 160)
        self.blocks = nn.ModuleList(Block(dim, heads, ffn, dropout=dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def frontend(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, N) -> (B, N // hop, dim); frame t sees samples < hop * (t + 1)."""
        with torch.autocast(wav.device.type, enabled=False):
            m = self.mel(F.pad(wav.float(), (240, 0)))
            m = torch.log(m.clamp_min(1e-5))
        if self.hop == 320:
            m = m[..., : 2 * (m.shape[-1] // 2)]
            h = F.gelu(self.conv1(F.pad(m, (2, 0))))
            h = F.gelu(self.conv2(F.pad(h, (2, 0))))
        else:
            h = F.gelu(self.conv1(F.pad(m, (2, 0))))
            h = F.gelu(self.conv2(F.pad(h, (3, 0))))
        return h.transpose(1, 2)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, N) -> (B, T, out_dim). Output step t is content frame t - lookahead."""
        x = self.frontend(wav)
        t = x.shape[1]
        pos = torch.arange(t, device=x.device)
        rel = pos[:, None] - pos[None, :]
        mask = (rel >= 0) & (rel <= self.window)
        for blk in self.blocks:
            x = blk(x, mask, pos)
        return self.head(self.norm(x))

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Whole-utterance features aligned with the teacher's frames.

        Identical to what frame-by-frame streaming produces: the model is
        causal, so the lookahead is paid by padding the end and shifting.
        """
        k = self.lookahead
        y = self.forward(F.pad(wav, (0, k * self.hop)))
        return y[:, k:]
