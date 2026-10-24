import json
import math
import os

import torch
from torch import nn

from rectified_flow.alias_free import ACTIVATION_RADIUS, Activation, AMPBlock, amp_radius
from rectified_flow.openvpi import CONFIG_MEL_KEYS, _fold_weight_norm
from rectified_flow.vocoder import ChunkedInference

MEL_SETTINGS = dict(mel_floor=1e-5, mel_scale='slaney', mel_norm='slaney', mel_center=False)


class SnakeBeta(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        alpha = self.alpha.exp()[None, :, None]
        beta = self.beta.exp()[None, :, None]
        return x + torch.sin(x * alpha).square() / (beta + 1e-9)


class BigVGAN(nn.Module):
    def __init__(self, num_mels, upsample_initial_channel, upsample_rates, upsample_kernel_sizes,
                 resblock_kernel_sizes, resblock_dilation_sizes, use_bias_at_final=True, use_tanh_at_final=True):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.use_tanh_at_final = bool(use_tanh_at_final)
        self.conv_pre = nn.Conv1d(num_mels, upsample_initial_channel, 7, padding=3)
        self.ups = nn.ModuleList(
            nn.ModuleList([nn.ConvTranspose1d(upsample_initial_channel >> index, upsample_initial_channel >> (index + 1),
                                              kernel_size, rate, padding=(kernel_size - rate) // 2)])
            for index, (rate, kernel_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes))
        )
        self.resblocks = nn.ModuleList(
            AMPBlock(upsample_initial_channel >> (index + 1), kernel_size, dilations, SnakeBeta)
            for index in range(len(upsample_rates))
            for kernel_size, dilations in zip(resblock_kernel_sizes, resblock_dilation_sizes)
        )
        channels = upsample_initial_channel >> len(upsample_rates)
        self.activation_post = Activation(SnakeBeta(channels))
        self.conv_post = nn.Conv1d(channels, 1, 7, padding=3, bias=bool(use_bias_at_final))
        factors = [math.prod(upsample_rates[:index + 1]) for index in range(len(upsample_rates))]
        block = amp_radius(resblock_kernel_sizes, resblock_dilation_sizes)
        stages = sum((kernel_size / 2 + block) / factor for kernel_size, factor in zip(upsample_kernel_sizes, factors))
        self.radius = 3 + stages + (ACTIVATION_RADIUS + 3) / factors[-1]

    def forward(self, mel, f0=None):
        x = self.conv_pre(mel)
        for index, upsample in enumerate(self.ups):
            x = upsample[0](x)
            blocks = self.resblocks[index * self.num_kernels:(index + 1) * self.num_kernels]
            x = sum(block(x) for block in blocks) / self.num_kernels
        x = self.conv_post(self.activation_post(x))
        return torch.tanh(x) if self.use_tanh_at_final else x.clamp(-1.0, 1.0)


def bigvgan_state(checkpoint):
    state = checkpoint.get('generator') if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or 'activation_post.act.alpha' not in state:
        return None
    return _fold_weight_norm(state) if any(key.startswith('ups.0.0.') for key in state) else None


def is_bigvgan(checkpoint):
    return bigvgan_state(checkpoint) is not None


def read_bigvgan_config(path):
    config_path = os.path.join(os.path.dirname(path), 'config.json')
    if not os.path.isfile(config_path):
        raise ValueError(f'{path} needs its config.json beside it.')
    with open(config_path, encoding='utf-8') as handle:
        config = json.load(handle)
    if str(config.get('resblock')) != '1' or config.get('activation') != 'snakebeta' or not config.get('snake_logscale'):
        raise ValueError(f'{config_path} describes a BigVGAN variant that is not supported.')
    mel = {target: config[source] for source, target in CONFIG_MEL_KEYS.items()}
    mel.update(MEL_SETTINGS, mel_fmin=float(mel['mel_fmin']),
               mel_fmax=float(mel['mel_fmax'] or mel['sample_rate'] / 2))
    hparams = dict(
        num_mels=int(config['num_mels']), upsample_initial_channel=int(config['upsample_initial_channel']),
        upsample_rates=[int(rate) for rate in config['upsample_rates']],
        upsample_kernel_sizes=[int(size) for size in config['upsample_kernel_sizes']],
        resblock_kernel_sizes=[int(size) for size in config['resblock_kernel_sizes']],
        resblock_dilation_sizes=[[int(dilation) for dilation in row] for row in config['resblock_dilation_sizes']],
        use_bias_at_final=config.get('use_bias_at_final', True), use_tanh_at_final=config.get('use_tanh_at_final', True),
    )
    return hparams, mel


def build_bigvgan(path, checkpoint):
    state = bigvgan_state(checkpoint)
    if state is None:
        raise ValueError(f'{path} is not a BigVGAN vocoder.')
    hparams, mel = read_bigvgan_config(path)
    if math.prod(hparams['upsample_rates']) != int(mel['hop_length']):
        raise ValueError(f"{path} upsamples by {math.prod(hparams['upsample_rates'])}, "
                         f"but its mel hop is {mel['hop_length']}.")
    generator = BigVGAN(**hparams)
    generator.load_state_dict(state, strict=True)
    return generator, mel


class BigVGANInference(ChunkedInference):
    name = 'BigVGAN'
    build = staticmethod(build_bigvgan)
