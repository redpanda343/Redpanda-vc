import math

import torch
from torch import nn
from torch.nn import functional as F

from rectified_flow.alias_free import ACTIVATION_RADIUS, Activation, AMPBlock, amp_radius
from rectified_flow.bigvgan import SnakeBeta
from rectified_flow.openvpi import _fold_weight_norm

MEL_PRESET = 'nsf_bigvgan_32khz'
UPSAMPLE_RATES = (5, 4, 2, 2, 2, 2)
UPSAMPLE_KERNEL_SIZES = (15, 8, 4, 4, 4, 4)
RESBLOCK_KERNEL_SIZES = (3, 7, 11)
RESBLOCK_DILATIONS = (1, 3, 5)
HARMONICS = 10
SINE_AMP = 0.1
NOISE_STD = 0.003


class SineSource(nn.Module):
    def __init__(self, sample_rate, harmonics=HARMONICS):
        super().__init__()
        self.sample_rate = sample_rate
        self.register_buffer('orders', torch.arange(1, harmonics + 2, dtype=torch.float32), persistent=False)
        self.register_buffer('merge_w', torch.zeros(1, harmonics + 1))
        self.register_buffer('merge_b', torch.zeros(1))

    def forward(self, f0):
        steps = (f0[..., None] * self.orders / self.sample_rate) % 1
        offsets = torch.rand(f0.shape[0], self.orders.numel(), device=f0.device)
        offsets[:, 0] = 0
        steps[:, 0] = steps[:, 0] + offsets
        phase = torch.cumsum(steps.double(), dim=1) % 1
        voiced = (f0 > 0)[..., None].float()
        noise = (voiced * NOISE_STD + (1.0 - voiced) * SINE_AMP / 3) * torch.randn_like(steps)
        sines = torch.sin(2 * math.pi * phase).float() * SINE_AMP * voiced + noise
        return torch.tanh(F.linear(sines, self.merge_w, self.merge_b))


class NSFBigVGAN(nn.Module):
    zero_unvoiced = True

    def __init__(self, sample_rate, num_mels, channels, upsample_rates=UPSAMPLE_RATES,
                 upsample_kernel_sizes=UPSAMPLE_KERNEL_SIZES, resblock_kernel_sizes=RESBLOCK_KERNEL_SIZES,
                 resblock_dilations=RESBLOCK_DILATIONS):
        super().__init__()
        self.hop = math.prod(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)
        self.m_source = SineSource(sample_rate)
        self.conv_pre = nn.Conv1d(num_mels, channels, 7, padding=3)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(channels >> index, channels >> (index + 1), kernel_size, rate,
                               padding=(kernel_size - rate) // 2)
            for index, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        )
        self.noise_convs = nn.ModuleList()
        for index in range(len(upsample_rates)):
            stride = math.prod(upsample_rates[index + 1:])
            out_channels = channels >> (index + 1)
            self.noise_convs.append(
                nn.Conv1d(1, out_channels, stride * 2, stride, padding=stride // 2)
                if index + 1 < len(upsample_rates) else nn.Conv1d(1, out_channels, 1)
            )
        self.resblocks = nn.ModuleList(
            AMPBlock(channels >> (index + 1), kernel_size, resblock_dilations, SnakeBeta)
            for index in range(len(upsample_rates)) for kernel_size in resblock_kernel_sizes
        )
        self.activation_post = Activation(SnakeBeta(channels >> len(upsample_rates)))
        self.conv_post = nn.Conv1d(channels >> len(upsample_rates), 1, 7, padding=3, bias=False)
        factors = [math.prod(upsample_rates[:index + 1]) for index in range(len(upsample_rates))]
        block = amp_radius(resblock_kernel_sizes, [resblock_dilations] * self.num_kernels)
        stages = sum((kernel_size / 2 + block) / factor for kernel_size, factor in zip(upsample_kernel_sizes, factors))
        self.radius = 3 + stages + (ACTIVATION_RADIUS + 3) / factors[-1]

    def forward(self, mel, f0):
        source = self.m_source(f0.float().repeat_interleave(self.hop, dim=-1)).transpose(1, 2)
        x = F.mish(self.conv_pre(mel))
        for index, (upsample, noise_conv) in enumerate(zip(self.ups, self.noise_convs)):
            x = upsample(x) + noise_conv(source)
            blocks = self.resblocks[index * self.num_kernels:(index + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        return torch.tanh(self.conv_post(self.activation_post(x)))


def nsf_bigvgan_state(checkpoint):
    state = checkpoint.get('model_g') if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or 'm_source.merge_w' not in state or 'noise_convs.0.weight' not in state:
        return None
    return _fold_weight_norm(state)


def is_nsf_bigvgan(checkpoint):
    return nsf_bigvgan_state(checkpoint) is not None


def build_nsf_bigvgan(path, checkpoint):
    from rectified_flow.config import mel_preset

    state = nsf_bigvgan_state(checkpoint)
    if state is None:
        raise ValueError(f'{path} is not an NSF-BigVGAN vocoder.')
    mel = {key: value for key, value in mel_preset(MEL_PRESET).items() if key not in ('mel_mean', 'mel_std')}
    generator = NSFBigVGAN(int(mel['sample_rate']), state['conv_pre.weight'].shape[1], state['conv_pre.weight'].shape[0])
    if generator.hop != int(mel['hop_length']) or generator.conv_pre.in_channels != int(mel['n_mels']):
        raise ValueError(f'{path} does not render the {MEL_PRESET} mel.')
    generator.load_state_dict(state, strict=True)
    return generator, mel
