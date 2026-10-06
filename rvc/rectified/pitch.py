import numpy as np

PARSELMOUTH_F0_MIN = 65.0
PARSELMOUTH_F0_MAX = 1100.0


def uses_parselmouth(data):
    return data.get('pitch_extractor') == 'parselmouth'


def interpolate_f0(f0):
    f0 = np.asarray(f0, dtype=np.float32).copy()
    if f0.ndim != 1 or not len(f0) or not np.isfinite(f0).all() or (f0 < 0).any():
        raise ValueError('Pitch extraction returned invalid frequencies.')
    voiced = f0 > 0
    if voiced.any():
        positions = np.arange(len(f0))
        f0 = np.exp2(np.interp(positions, positions[voiced], np.log2(f0[voiced]))).astype(np.float32)
    return f0


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
