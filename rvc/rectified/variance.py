import math

import librosa
import numpy as np
import torch
from torch.nn import functional as F


@torch.no_grad()
def voicing_tension(audio, sample_rate, f0, frames):
    import pyworld

    if audio.ndim != 2 or f0.ndim != 2 or audio.shape[0] != f0.shape[0]:
        raise ValueError("Voicing/tension extraction requires batched waveforms and F0.")
    if frames < 1 or audio.shape[-1] < 1 or f0.shape[-1] < 1:
        raise ValueError("Voicing/tension extraction requires nonempty inputs.")
    if not torch.isfinite(audio).all() or not torch.isfinite(f0).all() or (f0 < 0).any():
        raise ValueError("Voicing/tension extraction requires finite audio and nonnegative F0.")
    hop, window = int(sample_rate) // 100, 2048
    result = []
    for waveform, pitch in zip(audio.float().cpu().numpy(), f0.float().cpu().numpy()):
        length = math.ceil(len(waveform) / hop)
        pitch = np.pad(pitch, (0, max(0, length - len(pitch))), mode="edge")[:length]
        pitch = np.ascontiguousarray(pitch, dtype=np.float64)
        if np.max(np.abs(waveform)) < 1e-5 or not (pitch > 0).any():
            result.append(np.stack((np.full(frames, -100.0 / 96.0),
                                    np.full(frames, math.log(1e-4 / (1 - 1e-4)) * 0.1))))
            continue
        signal = waveform.astype(np.float64) + np.random.default_rng(0).standard_normal(len(waveform)) * 1e-5
        times = np.arange(length, dtype=np.float64) * hop / sample_rate
        spectrum = pyworld.cheaptrick(signal, pitch, times, sample_rate, fft_size=window)
        noise = pyworld.d4c(signal, pitch, times, sample_rate, fft_size=window)
        harmonic = pyworld.synthesize(
            pitch, np.maximum(spectrum * (1.0 - noise ** 2), 1e-16), np.zeros_like(noise),
            sample_rate, frame_period=hop * 1000.0 / sample_rate,
        ).astype(np.float32)
        harmonic = harmonic[:len(waveform)]
        tensor = torch.from_numpy(harmonic).unsqueeze(0)
        phase = torch.arange(window, dtype=torch.float32) / window * (2.0 * math.pi)
        taper = 0.355768 - 0.487396 * phase.cos() + 0.144232 * (2 * phase).cos() - 0.012604 * (3 * phase).cos()
        spec = torch.stft(tensor, window, hop_length=hop, window=taper, center=True,
                          pad_mode="reflect" if len(harmonic) > window // 2 else "constant", return_complex=True)
        voiced = np.flatnonzero(pitch > 0)
        interpolated = np.interp(np.arange(len(pitch)), voiced, pitch[voiced])
        center = torch.from_numpy(interpolated.astype(np.float32)) * window / sample_rate
        center = F.pad(center[None, None], (0, max(0, spec.shape[-1] - len(center))), mode="replicate")
        bins = torch.arange(spec.shape[1])[None, :, None]
        band = (bins - center[..., :spec.shape[-1]]).abs() <= 3.5
        band = band & (center[..., :spec.shape[-1]] >= 1)
        fundamental = torch.istft(spec * band, window, hop_length=hop, window=taper,
                                  center=True, length=len(harmonic))[0].numpy()

        def rms(value):
            curve = librosa.feature.rms(y=value, frame_length=window, hop_length=hop)[0]
            return np.pad(curve, (0, max(0, frames - len(curve))))[:frames]

        harmonic_rms, fundamental_rms = rms(harmonic), rms(fundamental)
        voicing = librosa.amplitude_to_db(harmonic_rms, top_db=None) / 96.0
        ratio = np.sqrt(np.maximum(harmonic_rms ** 2 - fundamental_rms ** 2, 0)) / (harmonic_rms + 1e-5)
        ratio = np.clip(ratio, 1e-4, 1.0 - 1e-4)
        tension = np.log(ratio / (1.0 - ratio)) * 0.1
        result.append(np.stack((voicing, tension)))
    features = torch.from_numpy(np.stack(result).astype(np.float32)).to(audio.device)
    if not torch.isfinite(features).all():
        raise FloatingPointError("Non-finite WORLD voicing/tension features.")
    return features[:, 0], features[:, 1]
