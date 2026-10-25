import math

import numpy as np
from scipy.signal import sosfilt

from rectified_flow.pitch import PARSELMOUTH_F0_MAX, PARSELMOUTH_F0_MIN

FORMANT_RATIO = 1.4
PITCH_RATIO = 2.0
PITCH_RANGE_RATIO = 1.5
PEAKING_FILTERS = 8
SHELF_FREQUENCIES = (60.0, 10000.0)
EQ_GAIN_DB = 12.0
Q_RANGE = (2.0, 5.0)


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


def biquad(kind, frequency, gain, q, sample_rate):
    amplitude = 10.0 ** (gain / 40.0)
    w0 = 2.0 * math.pi * frequency / sample_rate
    cos, alpha = math.cos(w0), math.sin(w0) / (2.0 * q)
    if kind == 'peak':
        b = (1.0 + alpha * amplitude, -2.0 * cos, 1.0 - alpha * amplitude)
        a = (1.0 + alpha / amplitude, -2.0 * cos, 1.0 - alpha / amplitude)
    else:
        sign = 1.0 if kind == 'low' else -1.0
        root = 2.0 * math.sqrt(amplitude) * alpha
        b = (amplitude * ((amplitude + 1) - sign * (amplitude - 1) * cos + root),
             sign * 2.0 * amplitude * ((amplitude - 1) - sign * (amplitude + 1) * cos),
             amplitude * ((amplitude + 1) - sign * (amplitude - 1) * cos - root))
        a = ((amplitude + 1) + sign * (amplitude - 1) * cos + root,
             -sign * 2.0 * ((amplitude - 1) + sign * (amplitude + 1) * cos),
             (amplitude + 1) + sign * (amplitude - 1) * cos - root)
    return [value / a[0] for value in (*b, *a)]


def random_equalizer(audio, sample_rate, gains, qs):
    low, high = SHELF_FREQUENCIES[0], min(SHELF_FREQUENCIES[1], 0.45 * sample_rate)
    centers = np.geomspace(low, high, PEAKING_FILTERS + 2)[1:-1]
    sections = [biquad('low', low, gains[0], qs[0], sample_rate)]
    sections += [biquad('peak', center, gain, q, sample_rate) for center, gain, q in zip(centers, gains[1:-1], qs[1:-1])]
    sections.append(biquad('high', high, gains[-1], qs[-1], sample_rate))
    return sosfilt(np.array(sections), np.asarray(audio, dtype=np.float64)).astype(np.float32)


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
