from typing import Optional

import torch
from torch import nn


class ContentBottleneck(nn.Module):
    def __init__(self, channels: int, width: int, noise: float = 0.0):
        super().__init__()
        self.down = nn.Linear(channels, width)
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, channels)
        self.noise = float(noise)
        self.code: Optional[torch.Tensor] = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        code = self.norm(self.down(features))
        if self.training and self.noise > 0:
            code = code + torch.randn_like(code) * self.noise
        self.code = code
        return self.up(code)
