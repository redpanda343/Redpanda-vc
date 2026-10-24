import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rectified_flow.openvpi import _fold_weight_norm

MEL_PRESET = 'pupu_44khz'
UPSAMPLE_RATES = (8, 8, 2, 2, 2)
RESBLOCK_DILATIONS = (1, 3, 5)
FILTER_ZEROS = 8
ACTIVATION_KERNEL = 12
ACTIVATION_RADIUS = 6.5
CHUNK_FRAMES = 1024


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


def sinc_lowpass(cutoff):
    half_size = int(FILTER_ZEROS / cutoff / 2)
    window = torch.hann_window(2 * half_size + 1, periodic=False)
    time = torch.arange(-half_size, half_size + 1, dtype=torch.float32)
    taps = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    return (taps / taps.sum()).view(1, 1, -1)


def depthwise(x, taps, stride=1):
    return F.conv1d(x, taps.expand(x.shape[1], -1, -1), stride=stride, groups=x.shape[1])


def zero_stuff(x, factor):
    y = x.new_zeros(*x.shape[:-1], x.shape[-1] * factor)
    y[..., ::factor] = x
    return y


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


class Activation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = ADAASnakeBeta(channels)
        self.upsample = UpSample()
        self.downsample = DownSample()

    def forward(self, x):
        return self.downsample(self.act(self.upsample(x)))


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations=RESBLOCK_DILATIONS):
        super().__init__()
        self.convs1 = nn.ModuleList(nn.Conv1d(channels, channels, kernel_size, dilation=dilation,
                                              padding=(kernel_size - 1) * dilation // 2) for dilation in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(channels, channels, kernel_size, padding=(kernel_size - 1) // 2)
                                    for _ in dilations)
        self.activations = nn.ModuleList(Activation(channels) for _ in range(2 * len(dilations)))

    def forward(self, x):
        for first, second, act1, act2 in zip(self.convs1, self.convs2, self.activations[::2], self.activations[1::2]):
            x = x + second(act2(first(act1(x))))
        return x


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
        self.resblocks = nn.ModuleList(ResBlock(channels >> (index + 1), kernel_size, resblock_dilations)
                                       for index in range(len(upsample_rates)) for kernel_size in resblock_kernel_sizes)
        self.activation_post = Activation(channels >> len(upsample_rates))
        self.conv_post = nn.Conv1d(channels >> len(upsample_rates), 1, 7, padding=3, bias=False)
        block = max(sum((kernel_size - 1) * (dilation + 1) / 2 for dilation in resblock_dilations)
                    for kernel_size in resblock_kernel_sizes) + 2 * len(resblock_dilations) * ACTIVATION_RADIUS
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


class PupuInference:
    def __init__(self, path, checkpoint, device):
        from rectified_flow.mel import LogMel

        self.device = torch.device(device)
        generator, self.data = build_pupu(path, checkpoint)
        self.generator = generator.to(self.device).float().eval().requires_grad_(False)
        self.context = math.ceil(self.generator.radius)
        self.mel = LogMel.from_config(self.data).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, waveform, pitch=0, f0_method='rmvpe', seed=None):
        if float(pitch):
            raise ValueError('Pupu-Vocoder has no pitch input. Set Pitch to 0.')
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError('Select non-empty audio with finite samples.')
        hop, length = int(self.data['hop_length']), len(waveform)
        padded_length = max(math.ceil(length / hop), math.ceil(self.data['n_fft'] / hop)) * hop
        padded = np.pad(np.clip(waveform, -1.0, 1.0), (0, padded_length - length))
        mel = self.mel(torch.from_numpy(padded).to(self.device)[None])
        frames, pieces = mel.shape[-1], []
        for start in range(0, frames, CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, frames)
            left, right = max(0, start - self.context), min(frames, end + self.context)
            pieces.append(self.generator(mel[..., left:right])[0, 0, (start - left) * hop:(end - left) * hop])
        audio = torch.cat(pieces)[:length]
        if not torch.isfinite(audio).all():
            raise FloatingPointError('Non-finite Pupu-Vocoder audio output.')
        return audio.float().cpu().numpy()
