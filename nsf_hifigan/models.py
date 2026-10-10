import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from rectified_flow.openvpi import LRELU_SLOPE, NSFHiFiGAN, ResBlock1


def _padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


def _init_normal(module, std=0.01):
    module.weight.data.normal_(0.0, std)
    module.bias.data.normal_(0.0, std)


def _normalized(module):
    weight_norm(module)
    _init_normal(module)


class TrainingResBlock1(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.convs1, self.convs2 = block.convs1, block.convs2

    def forward(self, x):
        for first, second in zip(self.convs1, self.convs2):
            residual = first(F.leaky_relu(x, LRELU_SLOPE))
            x = x + second(F.leaky_relu(residual, LRELU_SLOPE, inplace=True))
        return x


class TrainingGenerator(NSFHiFiGAN):
    def forward(self, mel, f0):
        f0 = f0.float()
        source = self._fast_sine(f0) if self.mini_nsf else self.m_source(f0, self.upp).transpose(1, 2)
        x = self.conv_pre(mel)
        if self.noise_sigma > 0:
            x = x + self.noise_sigma * torch.randn_like(x)
        for index, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE, inplace=True))
            if not self.mini_nsf:
                x = x + self.noise_convs[index](source)
            elif index == 1:
                x = x + self.source_conv(source)
            blocks = self.resblocks[index * self.num_kernels:(index + 1) * self.num_kernels]
            combined = blocks[0](x)
            for block in blocks[1:]:
                combined += block(x)
            x = combined / self.num_kernels
        return torch.tanh(self.conv_post(F.leaky_relu(x, inplace=True)))


def build_generator(hparams):
    generator = TrainingGenerator(**hparams)
    weight_norm(generator.conv_pre)
    for layer in generator.ups:
        _normalized(layer)
    for index, block in enumerate(generator.resblocks):
        for layer in ([*block.convs1, *block.convs2] if isinstance(block, ResBlock1) else block.convs):
            _normalized(layer)
        if isinstance(block, ResBlock1):
            generator.resblocks[index] = TrainingResBlock1(block)
    _normalized(generator.conv_post)
    if generator.mini_nsf:
        _init_normal(generator.source_conv)
    return generator


def load_generator_weights(generator, folded):
    state = {}
    for key, target in generator.state_dict().items():
        base = key[:-2] if key.endswith(('.weight_g', '.weight_v')) else key
        if base not in folded:
            raise ValueError(f'The pretrained generator has no weights for {key}.')
        value = folded[base]
        if key.endswith('.weight_g'):
            value = value.flatten(1).norm(dim=1).view(-1, *([1] * (value.dim() - 1)))
        if value.shape != target.shape:
            raise ValueError(f'The pretrained generator stores {key} as {tuple(value.shape)}, not {tuple(target.shape)}.')
        state[key] = value
    generator.load_state_dict(state)


class PeriodDiscriminator(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        channels = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList(
            [weight_norm(nn.Conv2d(source, target, (kernel_size, 1), (stride, 1), padding=(_padding(5), 0)))
             for source, target in zip(channels, channels[1:])]
            + [weight_norm(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0)))]
        )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        features = []
        batch, channels, length = x.shape
        if length % self.period:
            pad = self.period - length % self.period
            x = F.pad(x, (0, pad), 'reflect')
            length += pad
        x = x.view(batch, channels, length // self.period, self.period)
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE, inplace=True)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return torch.flatten(x, 1, -1), features


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(period) for period in periods])

    def forward(self, audio):
        outputs = [discriminator(audio) for discriminator in self.discriminators]
        return [output for output, _ in outputs], [features for _, features in outputs]


class ScaleDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []
        for layer in self.convs:
            x = F.leaky_relu(layer(x), LRELU_SLOPE, inplace=True)
            features.append(x)
        x = self.conv_post(x)
        features.append(x)
        return torch.flatten(x, 1, -1), features


class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [ScaleDiscriminator(use_spectral_norm=True), ScaleDiscriminator(), ScaleDiscriminator()]
        )
        self.meanpools = nn.ModuleList([nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)])

    def forward(self, audio):
        outputs, features = [], []
        for index, discriminator in enumerate(self.discriminators):
            if index:
                audio = self.meanpools[index - 1](audio)
            output, feature = discriminator(audio)
            outputs.append(output)
            features.append(feature)
        return outputs, features


def build_discriminator(periods):
    return nn.ModuleDict(dict(msd=MultiScaleDiscriminator(), mpd=MultiPeriodDiscriminator(periods)))
