import numpy as np
import torch
from torch.nn import functional as F

from rectified_flow.pitch import interpolate_f0

VARIANCES = ('breathiness', 'voicing', 'tension')
HNSEP_METHODS = ('vr', 'world')
VARIANCE_SCALES = {'breathiness': 1.0 / 96, 'voicing': 1.0 / 96, 'tension': 0.1}
BASE_HARMONIC_RADIUS = 3.5
STREAMING_CONTEXT = {'vr': 1.0, 'world': 0.3}
STREAMING_REWRITE = 0.1
VR_CHUNK = 15.0
VR_OVERLAP = 1.0
_SEPARATORS = {}
_SMOOTHERS = {}


def variance_names(model):
    return [name for name in VARIANCES if model.get(f'use_{name}_embed', False)]


class SinusoidalSmoothingConv1d(torch.nn.Conv1d):
    def __init__(self, kernel_size):
        super().__init__(in_channels=1, out_channels=1, kernel_size=max(kernel_size, 1), bias=False,
                         padding='same', padding_mode='replicate')
        if kernel_size > 1:
            smooth_kernel = torch.sin(torch.from_numpy(np.linspace(0, 1, kernel_size).astype(np.float32) * np.pi))
            smooth_kernel /= smooth_kernel.sum()
        else:
            smooth_kernel = torch.tensor([1.0], dtype=torch.float32)
        self.weight.data = smooth_kernel[None, None]


def smooth(curve, width, timestep):
    kernel = round(width / timestep)
    if kernel not in _SMOOTHERS:
        _SMOOTHERS[kernel] = SinusoidalSmoothingConv1d(kernel).eval()
    with torch.no_grad():
        return _SMOOTHERS[kernel](torch.from_numpy(np.asarray(curve, dtype=np.float32))[None])[0].numpy()


def rms_energy(waveform, length, hop, win):
    import librosa

    energy = librosa.feature.rms(y=waveform, frame_length=win, hop_length=hop)[0]
    if len(energy) < length:
        energy = np.pad(energy, (0, length - len(energy)))
    return energy[:length]


def energy_db(waveform, length, hop, win):
    import librosa

    return librosa.amplitude_to_db(rms_energy(waveform, length, hop, win), top_db=None)


def base_harmonic(harmonic, f0, uv, sample_rate, hop, win, device):
    waveform = torch.from_numpy(np.ascontiguousarray(harmonic, dtype=np.float32)).to(device)[None]
    samples = waveform.shape[1]
    f0 = np.asarray(f0, dtype=np.float32) * ~np.asarray(uv, dtype=bool)
    pad = samples // hop - len(f0) + 1
    if pad > 0:
        f0 = np.pad(f0, (0, pad), mode='constant', constant_values=(f0[0], f0[-1]))
    f0 = torch.from_numpy(interpolate_f0(f0)).to(device)[None, :, None]
    phase = torch.arange(win, dtype=torch.float32, device=device) / win * 2 * np.pi
    window = (0.355768 - 0.487396 * torch.cos(phase) + 0.144232 * torch.cos(2 * phase)
              - 0.012604 * torch.cos(3 * phase))
    spec = torch.stft(waveform, n_fft=win, win_length=win, hop_length=hop, window=window, center=True,
                      return_complex=True).transpose(1, 2)
    frames, bins = spec.shape[1:]
    index = torch.arange(bins, device=device).to(f0)[None, None]
    center = f0 * win / sample_rate
    mask = ((center >= 1) & (index >= torch.clip(center - BASE_HARMONIC_RADIUS, min=0))
            & (index < torch.clip(center + BASE_HARMONIC_RADIUS, max=bins))).float()
    mask = F.pad(mask, (0, 0, 0, max(0, frames - mask.shape[1])))[:, :frames]
    return torch.istft((spec * mask).transpose(1, 2), n_fft=win, win_length=win, hop_length=hop, window=window,
                       center=True, length=samples)[0].cpu().numpy()


def tension_logit(harmonic, base, length, hop, win):
    full = rms_energy(harmonic, length, hop, win)
    ratio = np.sqrt(np.clip(full ** 2 - rms_energy(base, length, hop, win) ** 2, 0, None)) / (full + 1e-5)
    ratio = np.clip(ratio, 1e-4, 1 - 1e-4)
    return np.log(ratio / (1 - ratio))


def world_separate(waveform, sample_rate, f0, hop, fft_size):
    import pyworld as pw

    noise = np.random.default_rng(0).standard_normal(waveform.shape) * 1e-5
    x = waveform.astype(np.double) + noise
    f0 = np.asarray(f0, dtype=np.double)
    frames = (x.shape[0] + hop - 1) // hop
    if f0.shape[0] < frames:
        f0 = np.pad(f0, (0, frames - f0.shape[0]), mode='constant', constant_values=(f0[0], f0[-1]))
    f0 = f0[:frames]
    t = np.arange(0, frames) * (hop / sample_rate)
    sp = pw.cheaptrick(x, f0, t, sample_rate, fft_size=fft_size)
    ap = pw.d4c(x, f0, t, sample_rate, fft_size=fft_size)
    period = hop / sample_rate * 1000
    harmonic = pw.synthesize(f0, np.clip(sp * (1 - ap * ap), a_min=1e-16, a_max=None), np.zeros_like(ap),
                             sample_rate, frame_period=period).astype(np.float32)
    aperiodic = pw.synthesize(f0, sp * ap * ap, np.ones_like(ap), sample_rate,
                              frame_period=period).astype(np.float32)
    return harmonic, aperiodic


def vr_separator(device):
    key = str(device)
    if key not in _SEPARATORS:
        from shared.predictors.hnsep import load_sep_model
        from rectified_flow.resources import hnsep_model

        _SEPARATORS[key] = load_sep_model(hnsep_model(), device)
    return _SEPARATORS[key]


def _vr_harmonic(model, waveform, device):
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).to(device).reshape(1, 1, -1)
        if not model.is_mono:
            x = x.repeat(1, 2, 1)
        return torch.mean(model.predict_from_audio(x), dim=1).reshape(-1).cpu().numpy()


def vr_separate(waveform, sample_rate, device):
    model = vr_separator(device)
    chunk, overlap = int(VR_CHUNK * sample_rate), int(VR_OVERLAP * sample_rate)
    if len(waveform) <= chunk:
        harmonic = _vr_harmonic(model, waveform, device)
        return harmonic, waveform - harmonic
    harmonic = np.zeros(len(waveform), dtype=np.float64)
    weight = np.zeros(len(waveform), dtype=np.float64)
    fade = np.linspace(0.0, 1.0, overlap + 2)[1:-1]
    start = 0
    while True:
        end = min(start + chunk, len(waveform))
        part = _vr_harmonic(model, waveform[start:end], device)
        window = np.ones(end - start)
        if start:
            window[:overlap] = fade
        if end < len(waveform):
            window[-overlap:] = fade[::-1]
        harmonic[start:end] += part * window
        weight[start:end] += window
        if end == len(waveform):
            break
        start = end - overlap
    harmonic = (harmonic / weight).astype(np.float32)
    return harmonic, waveform - harmonic


def separate(method, waveform, sample_rate, f0, uv, data, device):
    waveform = np.asarray(waveform, dtype=np.float32)
    if method == 'vr':
        return vr_separate(waveform, sample_rate, device)
    if method == 'world':
        return world_separate(waveform, sample_rate, np.asarray(f0) * ~np.asarray(uv), int(data['hop_length']),
                              int(data['n_fft']))
    raise ValueError(f'hnsep must be one of {HNSEP_METHODS}, not {method!r}.')


def variance_curves(names, harmonic, aperiodic, f0, uv, length, data, settings, device):
    rate, hop, win = int(data['sample_rate']), int(data['hop_length']), int(data['win_length'])
    timestep = hop / rate
    curves = []
    for name in names:
        if name == 'tension':
            curve = tension_logit(harmonic, base_harmonic(harmonic, f0, uv, rate, hop, win, device), length, hop, win)
        else:
            curve = energy_db(aperiodic if name == 'breathiness' else harmonic, length, hop, win)
        curves.append(smooth(curve, settings[f'{name}_smooth_width'], timestep))
    return np.stack(curves).astype(np.float32) if curves else np.zeros((0, length), dtype=np.float32)


def extract_variances(names, waveform, f0, uv, length, data, settings, device):
    if not names:
        return np.zeros((0, length), dtype=np.float32)
    sample_rate = int(data['sample_rate'])
    harmonic, aperiodic = separate(settings['hnsep'], waveform, sample_rate, f0, uv, data, device)
    return variance_curves(names, harmonic, aperiodic, f0, uv, length, data, settings, device)


def resample_variances(curves, speed, length, data):
    timestep = int(data['hop_length']) / int(data['sample_rate'])
    positions = np.arange(length) * timestep * speed
    source = np.arange(curves.shape[-1]) * timestep
    return np.stack([np.interp(positions, source, curve) for curve in curves]).astype(np.float32)


class StreamingVariances:
    def __init__(self, names, data, settings, device):
        self.names = list(names)
        self.data, self.settings, self.device = data, settings, device
        self.method = settings['hnsep']
        self.hop = int(data['hop_length'])
        rate = int(data['sample_rate'])
        self.context = int(STREAMING_CONTEXT[self.method] * rate)
        self.rewrite = int(STREAMING_REWRITE * rate)
        self.reset()

    def reset(self):
        self.harmonic = None
        self.aperiodic = None
        self.position = None

    def _separate(self, waveform, f0, uv):
        parts = separate(self.method, waveform, int(self.data['sample_rate']), f0, uv, self.data, self.device)
        return [np.pad(part, (0, max(0, len(waveform) - len(part))))[:len(waveform)] for part in parts]

    def __call__(self, waveform, f0, uv, position, length):
        waveform = np.asarray(waveform, dtype=np.float32)
        total = len(waveform)
        shift = None if self.position is None else position - self.position
        self.position = position
        if (self.harmonic is None or len(self.harmonic) != total or shift is None or shift < 0
                or shift + self.rewrite + self.context >= total):
            self.harmonic, self.aperiodic = self._separate(waveform, f0, uv)
        else:
            if shift:
                self.harmonic = np.concatenate((self.harmonic[shift:], np.zeros(shift, np.float32)))
                self.aperiodic = np.concatenate((self.aperiodic[shift:], np.zeros(shift, np.float32)))
            keep = total - shift - self.rewrite
            start = max(0, keep - self.context)
            start -= start % self.hop
            frame = start // self.hop
            harmonic, aperiodic = self._separate(waveform[start:], f0[frame:], uv[frame:])
            self.harmonic[keep:] = harmonic[keep - start:]
            self.aperiodic[keep:] = aperiodic[keep - start:]
        return variance_curves(self.names, self.harmonic, self.aperiodic, f0, uv, length, self.data, self.settings,
                               self.device)
