from pathlib import Path

import numpy as np

PARSELMOUTH_F0_MIN = 65.0
PARSELMOUTH_F0_MAX = 1100.0
PITCH_EXTRACTORS = ('parselmouth', 'rmvpe')
RMVPE_PATH = Path(__file__).resolve().parents[1] / 'models' / 'predictors' / 'rmvpe.pt'
_RMVPE = {}


def interpolate_f0(f0):
    f0 = np.asarray(f0, dtype=np.float32).copy()
    if f0.ndim != 1 or not len(f0) or not np.isfinite(f0).all() or (f0 < 0).any():
        raise ValueError('Pitch extraction returned invalid frequencies.')
    voiced = f0 > 0
    if voiced.any():
        positions = np.arange(len(f0))
        f0 = np.exp2(np.interp(positions, positions[voiced], np.log2(f0[voiced]))).astype(np.float32)
    return f0


def resample_f0(f0, source_rate, frames, frame_rate):
    f0 = interpolate_f0(f0)
    if not f0.any():
        return np.zeros(frames, dtype=np.float32)
    positions = np.arange(frames) * (source_rate / frame_rate)
    return np.exp2(np.interp(positions, np.arange(len(f0)), np.log2(f0))).astype(np.float32)


def parselmouth_f0(waveform, sample_rate, hop, frames, f0_min=PARSELMOUTH_F0_MIN, f0_max=PARSELMOUTH_F0_MAX):
    import parselmouth

    waveform = np.asarray(waveform)
    left = int(np.ceil(1.5 / f0_min * sample_rate))
    right = hop * ((len(waveform) - 1) // hop + 1) - len(waveform) + left + 1
    waveform = np.pad(waveform, (left, right))
    contour = parselmouth.Sound(waveform, sampling_frequency=sample_rate).to_pitch_ac(
        time_step=hop / sample_rate, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max,
    ).selected_array['frequency'].astype(np.float32)
    contour = np.pad(contour, (0, max(0, frames - len(contour))))[:frames]
    return interpolate_f0(contour)


def rmvpe_model(device):
    key = str(device)
    if key not in _RMVPE:
        from rvc.lib.predictors.rmvpe import RMVPE

        if not RMVPE_PATH.is_file():
            raise FileNotFoundError(f'RMVPE model not found: {RMVPE_PATH}. Restart the app to download it.')
        _RMVPE[key] = RMVPE(str(RMVPE_PATH), device=device)
    return _RMVPE[key]


def rmvpe_f0(model, waveform, sample_rate, hop, frames):
    f0, uv = model.get_pitch(np.asarray(waveform, dtype=np.float32), sample_rate, frames,
                             hop_size=hop, interp_uv=True)
    return f0.astype(np.float32), not uv.all()


def extract_f0(extractor, waveform, sample_rate, hop, frames, device='cpu'):
    if extractor == 'parselmouth':
        f0 = parselmouth_f0(waveform, sample_rate, hop, frames)
        return f0, bool(f0.any())
    if extractor == 'rmvpe':
        return rmvpe_f0(rmvpe_model(device), waveform, sample_rate, hop, frames)
    raise ValueError(f'Pitch extractor must be one of {PITCH_EXTRACTORS}, not {extractor!r}.')
