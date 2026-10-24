import math

import torch
from torch import nn
from torch.nn import functional as F

ACTIVATION_KERNEL = 12
ACTIVATION_RADIUS = 6.5


def kaiser_sinc_filter(cutoff, half_width, kernel_size):
    half_size = kernel_size // 2
    attenuation = 2.285 * (half_size - 1) * math.pi * 4 * half_width + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if kernel_size % 2 == 0:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    taps = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (taps / taps.sum()).view(1, 1, kernel_size)


def depthwise(x, taps, stride=1):
    return F.conv1d(x, taps.expand(x.shape[1], -1, -1), stride=stride, groups=x.shape[1])


class UpSample(nn.Module):
    def __init__(self, ratio=2, kernel_size=ACTIVATION_KERNEL):
        super().__init__()
        self.ratio = ratio
        self.pad = kernel_size // ratio - 1
        self.pad_left = self.pad * ratio + (kernel_size - ratio) // 2
        self.pad_right = self.pad * ratio + (kernel_size - ratio + 1) // 2
        self.register_buffer('filter', kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x):
        x = F.pad(x, (self.pad, self.pad), mode='replicate')
        x = self.ratio * F.conv_transpose1d(x, self.filter.expand(x.shape[1], -1, -1), stride=self.ratio,
                                            groups=x.shape[1])
        return x[..., self.pad_left:-self.pad_right]


class LowPass(nn.Module):
    def __init__(self, ratio=2, kernel_size=ACTIVATION_KERNEL):
        super().__init__()
        self.ratio = ratio
        self.padding = (kernel_size // 2 - int(kernel_size % 2 == 0), kernel_size // 2)
        self.register_buffer('filter', kaiser_sinc_filter(0.5 / ratio, 0.6 / ratio, kernel_size))

    def forward(self, x):
        return depthwise(F.pad(x, self.padding, mode='replicate'), self.filter, self.ratio)


class DownSample(nn.Module):
    def __init__(self, ratio=2, kernel_size=ACTIVATION_KERNEL):
        super().__init__()
        self.lowpass = LowPass(ratio, kernel_size)

    def forward(self, x):
        return self.lowpass(x)


class Activation(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.upsample = UpSample()
        self.downsample = DownSample()

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class AMPBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations, activation):
        super().__init__()
        self.convs1 = nn.ModuleList(nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                                              padding=(kernel_size - 1) * dilation // 2) for dilation in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
                                    for _ in dilations)
        self.activations = nn.ModuleList(Activation(activation(channels)) for _ in range(2 * len(dilations)))

    def forward(self, x):
        for first, second, act1, act2 in zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]):
            x = x + second(act2(first(act1(x))))
        return x


def amp_radius(kernel_sizes, dilation_sizes):
    return max(sum((kernel_size - 1) * (dilation + 1) / 2 for dilation in dilations)
               + 2 * len(dilations) * ACTIVATION_RADIUS for kernel_size, dilations in zip(kernel_sizes, dilation_sizes))
