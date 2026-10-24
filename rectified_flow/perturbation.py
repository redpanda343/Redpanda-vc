import math

import numpy as np
import torch

from rectified_flow.pitch import PARSELMOUTH_F0_MAX, PARSELMOUTH_F0_MIN

FORMANT_RATIO = 1.4
PITCH_RATIO = 2.0
PITCH_RANGE_RATIO = 1.5
PEAKING_FILTERS = 8
SHELF_FREQUENCIES = (60.0, 10000.0)
EQ_GAIN_DB = 12.0
Q_RANGE = (2.0, 5.0)
EQ_HEADROOM = 1.0 / 64


def _ratio(rng, high):
    value = rng.uniform(1.0, high)
    return value if rng.random() < 0.5 else 1.0 / value


def perturbation_parameters(rng):
    filters = PEAKING_FILTERS + 2
    return dict(
        formant=_ratio(rng, FORMANT_RATIO), pitch=_ratio(rng, PITCH_RATIO), pitch_range=_ratio(rng, PITCH_RANGE_RATIO),
        gains=[rng.uniform(-EQ_GAIN_DB, EQ_GAIN_DB) for _ in range(filters)],
        qs=[Q_RANGE[0] * (Q_RANGE[1] / Q_RANGE[0]) ** rng.random() for _ in range(filters)],
    )


def random_equalizer(audio, sample_rate, gains, qs):
    import torchaudio.functional as AF

    low, high = SHELF_FREQUENCIES[0], min(SHELF_FREQUENCIES[1], 0.45 * sample_rate)
    centers = np.geomspace(low, high, PEAKING_FILTERS + 2)[1:-1]
    x = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None] * EQ_HEADROOM
    x = AF.bass_biquad(x, sample_rate, gains[0], low, qs[0])
    for center, gain, q in zip(centers, gains[1:-1], qs[1:-1]):
        x = AF.equalizer_biquad(x, sample_rate, float(center), gain, q)
    x = AF.treble_biquad(x, sample_rate, gains[-1], high, qs[-1])
    return x[0].numpy() / EQ_HEADROOM


def change_gender(audio, sample_rate, formant, pitch, pitch_range):
    import parselmouth
    from parselmouth.praat import call

    sound = parselmouth.Sound(np.asarray(audio, dtype=np.float64), sample_rate)
    contour = sound.to_pitch_ac(pitch_floor=PARSELMOUTH_F0_MIN, pitch_ceiling=PARSELMOUTH_F0_MAX)
    median = call(contour, 'Get quantile', 0.0, 0.0, 0.5, 'Hertz')
    target = median * pitch if math.isfinite(median) else 0.0
    shifted = call(sound, 'Change gender', PARSELMOUTH_F0_MIN, PARSELMOUTH_F0_MAX, formant, target, pitch_range, 1.0)
    values = shifted.values[0].astype(np.float32)
    return np.pad(values, (0, max(0, len(audio) - len(values))))[:len(audio)]


def perturb(audio, sample_rate, parameters):
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(audio).max())
    shaped = random_equalizer(audio, sample_rate, parameters['gains'], parameters['qs'])
    shifted = change_gender(shaped, sample_rate, parameters['formant'], parameters['pitch'], parameters['pitch_range'])
    shifted = np.nan_to_num(shifted)
    level = float(np.abs(shifted).max())
    return shifted * (peak / level) if level > 0 else shifted
