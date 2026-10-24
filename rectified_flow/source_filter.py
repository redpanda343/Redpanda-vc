import math

import torch
from torch import nn

from rectified_flow.mel import MEL_DEFAULTS, LogMel

F0_MIN = 25.0
F0_MAX = 3200.0
TABLE_SIZE = 2048
NEIGHBOR_HARMONICS = 2
TEMPLATE_FLOOR = 1e-3


def validate_source_filter_config(model):
    if model.get('mean_flow', False):
        raise ValueError('The source-filter flow trains as a standard rectified flow, not MeanFlow.')
    settings = model['source_filter_args']
    for name in ('envelope_coefficients', 'aperiodicity_coefficients'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'Source-filter {name} must be a positive integer.')


def hann_lobe(bins):
    denominator = 1.0 - bins.square()
    edge = denominator.abs() < 1e-4
    return torch.where(edge, 0.5, torch.sinc(bins) / torch.where(edge, 1.0, denominator)).abs()


def harmonic_table(data, f0_min=F0_MIN, f0_max=F0_MAX, size=TABLE_SIZE):
    sample_rate, n_fft, win = int(data['sample_rate']), int(data['n_fft']), int(data['win_length'])
    basis = LogMel.from_config(data).basis
    f0 = torch.exp2(torch.linspace(math.log2(f0_min), math.log2(f0_max), size))[:, None]
    frequencies = torch.arange(n_fft // 2 + 1, dtype=torch.float32) * (sample_rate / n_fft)
    nearest = torch.round(frequencies / f0)
    comb = torch.zeros(size, frequencies.numel())
    for offset in range(-NEIGHBOR_HARMONICS, NEIGHBOR_HARMONICS + 1):
        harmonic = nearest + offset
        lobe = hann_lobe((frequencies - harmonic * f0) * (win / sample_rate))
        comb += torch.where(harmonic >= 1, lobe, 0.0)
    flat = comb.mean(1, keepdim=True) * basis.sum(1)
    return torch.log((comb @ basis.T / flat).clamp_min(TEMPLATE_FLOOR))


class HarmonicTemplate(nn.Module):
    def __init__(self, data, f0_min=F0_MIN, f0_max=F0_MAX, size=TABLE_SIZE):
        super().__init__()
        self.f0_min, self.f0_max = float(f0_min), float(f0_max)
        self.register_buffer('table', harmonic_table(data, f0_min, f0_max, size), persistent=False)

    def forward(self, f0):
        size = self.table.shape[0]
        low, high = math.log2(self.f0_min), math.log2(self.f0_max)
        position = (torch.log2(f0.float().clamp(self.f0_min, self.f0_max)) - low) * ((size - 1) / (high - low))
        left = position.floor().long().clamp(0, size - 2)
        weight = (position - left).unsqueeze(-1)
        values = self.table[left] * (1.0 - weight) + self.table[left + 1] * weight
        return torch.where((f0 >= self.f0_min).unsqueeze(-1), values, 0.0).transpose(1, 2)


def cosine_basis(coefficients, n_mels):
    bins = torch.arange(n_mels, dtype=torch.float32) + 0.5
    orders = torch.arange(coefficients, dtype=torch.float32)[:, None]
    return torch.cos(math.pi * orders * bins[None] / n_mels) / math.sqrt(coefficients)


class SourceFilterDecoder(nn.Module):
    def __init__(self, trunk, n_mels, data, envelope_coefficients=20, aperiodicity_coefficients=8):
        super().__init__()
        if max(envelope_coefficients, aperiodicity_coefficients) > n_mels:
            raise ValueError('Source-filter coefficients cannot exceed the number of mel bins.')
        self.trunk = trunk
        self.n_mels = int(n_mels)
        self.mel_mean, self.mel_std = float(data['mel_mean']), float(data['mel_std'])
        self.floor = math.log(float(data.get('mel_floor', MEL_DEFAULTS['mel_floor'])))
        self.splits = (int(envelope_coefficients), int(aperiodicity_coefficients))
        self.register_buffer('envelope_basis', cosine_basis(envelope_coefficients, n_mels), persistent=False)
        self.register_buffer('aperiodicity_basis', cosine_basis(aperiodicity_coefficients, n_mels), persistent=False)

    def forward(self, cond):
        envelope, aperiodicity = self.trunk(cond).float().split(self.splits, dim=1)
        envelope = torch.einsum('km,bkt->bmt', self.envelope_basis, envelope)
        aperiodicity = torch.sigmoid(torch.einsum('km,bkt->bmt', self.aperiodicity_basis, aperiodicity))
        template = cond[:, -self.n_mels:].float().exp()
        mel = envelope * self.mel_std + self.mel_mean + torch.log(aperiodicity + (1.0 - aperiodicity) * template)
        return (torch.logaddexp(mel, mel.new_tensor(self.floor)) - self.mel_mean) / self.mel_std
