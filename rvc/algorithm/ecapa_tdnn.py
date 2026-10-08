import math

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class SEModule(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input_tensor):
        return input_tensor * self.se(input_tensor)


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, dilation, scale=8):
        super().__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        padding = math.floor(kernel_size / 2) * dilation
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                )
                for _ in range(self.nums)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes)

    def forward(self, x):
        residual = x
        out = self.bn1(self.relu(self.conv1(x)))
        split = torch.split(out, self.width, 1)
        pieces = []
        current = None
        for index in range(self.nums):
            current = split[index] if index == 0 else current + split[index]
            current = self.bns[index](self.relu(self.convs[index](current)))
            pieces.append(current)
        out = torch.cat([*pieces, split[self.nums]], 1)
        out = self.bn3(self.relu(self.conv3(out)))
        return self.se(out) + residual


class PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super().__init__()
        self.register_buffer(
            "flipped_filter",
            torch.tensor([-coef, 1.0], dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, input_tensor):
        input_tensor = F.pad(input_tensor.unsqueeze(1), (1, 0), "reflect")
        return F.conv1d(input_tensor, self.flipped_filter).squeeze(1)


class FbankAug(nn.Module):
    def __init__(self, freq_mask_width=(0, 8), time_mask_width=(0, 10)):
        super().__init__()
        self.time_mask_width = time_mask_width
        self.freq_mask_width = freq_mask_width

    def mask_along_axis(self, x, dim):
        original_size = x.shape
        batch, features, time = x.shape
        size = features if dim == 1 else time
        width_range = self.freq_mask_width if dim == 1 else self.time_mask_width
        mask_len = torch.randint(
            width_range[0], width_range[1], (batch, 1), device=x.device
        ).unsqueeze(2)
        maximum_start = max(1, size - int(mask_len.max().item()))
        mask_pos = torch.randint(
            0, maximum_start, (batch, 1), device=x.device
        ).unsqueeze(2)
        positions = torch.arange(size, device=x.device).view(1, 1, -1)
        mask = ((mask_pos <= positions) & (positions < mask_pos + mask_len)).any(dim=1)
        mask = mask.unsqueeze(2) if dim == 1 else mask.unsqueeze(1)
        return x.masked_fill(mask, 0.0).view(*original_size)

    def forward(self, x):
        return self.mask_along_axis(self.mask_along_axis(x, 2), 1)


class ECAPATDNN(nn.Module):
    def __init__(self, channels=1024):
        super().__init__()
        self.torchfbank = nn.Sequential(
            PreEmphasis(),
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=512,
                win_length=400,
                hop_length=160,
                f_min=20,
                f_max=7600,
                window_fn=torch.hamming_window,
                n_mels=80,
            ),
        )
        self.specaug = FbankAug()
        self.conv1 = nn.Conv1d(80, channels, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(channels)
        self.layer1 = Bottle2neck(channels, channels, 3, 2, 8)
        self.layer2 = Bottle2neck(channels, channels, 3, 3, 8)
        self.layer3 = Bottle2neck(channels, channels, 3, 4, 8)
        self.layer4 = nn.Conv1d(3 * channels, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)

    def forward(self, x, aug=False):
        with torch.no_grad():
            x = self.torchfbank(x) + 1e-6
            x = x.log()
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug:
                x = self.specaug(x)
        x = self.bn1(self.relu(self.conv1(x)))
        x1 = self.layer1(x)
        x2 = self.layer2(x + x1)
        x3 = self.layer3(x + x1 + x2)
        x = self.relu(self.layer4(torch.cat((x1, x2, x3), dim=1)))
        time = x.size(-1)
        global_x = torch.cat(
            (
                x,
                torch.mean(x, dim=2, keepdim=True).repeat(1, 1, time),
                torch.sqrt(
                    torch.var(x, dim=2, keepdim=True).clamp(min=1e-4)
                ).repeat(1, 1, time),
            ),
            dim=1,
        )
        weights = self.attention(global_x)
        mean = torch.sum(x * weights, dim=2)
        deviation = torch.sqrt(
            (torch.sum((x**2) * weights, dim=2) - mean**2).clamp(min=1e-4)
        )
        return self.bn6(self.fc6(self.bn5(torch.cat((mean, deviation), dim=1))))


def load_ecapa_tdnn(model_path, device="cpu"):
    model = ECAPATDNN().to(device)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    prefix = "speaker_encoder."
    encoder_state = {
        name[len(prefix) :]: value
        for name, value in checkpoint.items()
        if name.startswith(prefix)
    }
    model.load_state_dict(encoder_state, strict=True)
    model.eval()
    model.requires_grad_(False)
    return model
