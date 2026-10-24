import math

import torch
from torch import nn
from torch.nn import functional as F

from rectified_flow.alias_free import ACTIVATION_RADIUS, Activation, AMPBlock, amp_radius, depthwise
from rectified_flow.openvpi import _fold_weight_norm
from rectified_flow.vocoder import ChunkedInference

MEL_PRESET = 'pupu_44khz'
UPSAMPLE_RATES = (8, 8, 2, 2, 2)
RESBLOCK_DILATIONS = (1, 3, 5)
FILTER_ZEROS = 8


def sinc_lowpass(cutoff):
    half_size = int(FILTER_ZEROS / cutoff / 2)
    window = torch.hann_window(2 * half_size + 1, periodic=False)
    time = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
    taps = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (taps / taps.sum()).view(1, 1, -1)


def zero_stuff(x, factor):
    y = x.new_zeros(*x.shape[:-1], x.shape[-1] * factor)
    y[..., ::factor] = x
    return y


class ADAASnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        previous = F.pad(x[..., :-1], (1, 0))
        total = x + previous
        smooth = torch.cos(alpha * total) * torch.sinc(alpha * (x - previous) / math.pi)
        return total / 2.0 + (1.0 - smooth) / (2.0 * (beta + 1e-9))


class ResampleUpsampler(nn.Module):
    def __init__(self, source_channels, rate, channels, out_channels):
        super().__init__()
        self.rate = rate
        self.convolution_after = nn.Conv1d(channels, out_channels, 1)
        self.convolution_noise = nn.Conv1d(source_channels, channels, 7, padding=3)
        self.register_buffer('lowpass', sinc_lowpass(0.5 / rate), persistent=False)

    def filtered(self, x):
        with torch.autocast(x.device.type, enabled=False):
            half = self.lowpass.shape[-1] // 2
            return depthwise(F.pad(x.float(), (half, half), mode='replicate'), self.lowpass).to(x.dtype)

    def forward(self, x, source, factor):
        noise = self.convolution_noise(zero_stuff(source, factor))
        noise = noise - self.filtered(noise)
        return self.convolution_after(self.filtered(zero_stuff(x, self.rate)) + noise)


class PupuVocoder(nn.Module):
    def __init__(self, num_mels, channels, resblock_kernel_sizes, upsample_rates=UPSAMPLE_RATES,
                 resblock_dilations=RESBLOCK_DILATIONS):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.factors = [math.prod(upsample_rates[:index + 1]) for index in range(len(upsample_rates))]
        self.conv_pre = nn.Conv1d(num_mels, channels, 7, padding=3)
        self.ups = nn.ModuleList(ResampleUpsampler(channels, rate, channels >> index, channels >> (index + 1))
                                 for index, rate in enumerate(upsample_rates))
        self.resblocks = nn.ModuleList(AMPBlock(channels >> (index + 1), kernel_size, resblock_dilations, ADAASnakeBeta)
                                       for index in range(len(upsample_rates)) for kernel_size in resblock_kernel_sizes)
        self.activation_post = Activation(ADAASnakeBeta(channels >> len(upsample_rates)))
        self.conv_post = nn.Conv1d(channels >> len(upsample_rates), 1, 7, padding=3, bias=False)
        block = amp_radius(resblock_kernel_sizes, [resblock_dilations] * self.num_kernels)
        stages = sum((FILTER_ZEROS * rate + block) / factor for rate, factor in zip(upsample_rates, self.factors))
        self.radius = 3 + stages + (ACTIVATION_RADIUS + 3) / self.factors[-1]

    def forward(self, mel, f0=None):
        x = self.conv_pre(mel)
        source = x
        for index, (upsample, factor) in enumerate(zip(self.ups, self.factors)):
            x = upsample(x, source, factor)
            blocks = self.resblocks[index * self.num_kernels:(index + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return self.conv_post(self.activation_post(x)).clamp(-1.0, 1.0)


def pupu_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    state = checkpoint.get('generator', checkpoint)
    if not isinstance(state, dict) or not any(key.startswith('ups.0.convolution_noise.') for key in state):
        return None
    return _fold_weight_norm(state)


def is_pupu(checkpoint):
    return pupu_state(checkpoint) is not None


def build_pupu(path, checkpoint):
    from rectified_flow.config import mel_preset

    state = pupu_state(checkpoint)
    if state is None:
        raise ValueError(f'{path} is not a Pupu-Vocoder.')
    mel = {key: value for key, value in mel_preset(MEL_PRESET).items() if key not in ('mel_mean', 'mel_std')}
    blocks = sorted({int(key.split('.')[1]) for key in state if key.startswith('resblocks.')})
    count = len(blocks) // len(UPSAMPLE_RATES)
    generator = PupuVocoder(
        num_mels=state['conv_pre.weight'].shape[1], channels=state['conv_pre.weight'].shape[0],
        resblock_kernel_sizes=[state[f'resblocks.{index}.convs1.0.weight'].shape[-1] for index in range(count)],
    )
    if generator.conv_pre.in_channels != int(mel['n_mels']):
        raise ValueError(f'{path} takes {generator.conv_pre.in_channels} mel bins, not {mel["n_mels"]}.')
    generator.load_state_dict(state, strict=True)
    return generator, mel


class PupuInference(ChunkedInference):
    name = 'Pupu-Vocoder'
    build = staticmethod(build_pupu)
