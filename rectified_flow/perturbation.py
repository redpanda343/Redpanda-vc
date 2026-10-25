import math
import multiprocessing

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
PRAAT_TIMEOUT = 10.0


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
    lowest, median, highest = (call(contour, 'Get quantile', 0.0, 0.0, quantile, 'Hertz') for quantile in (0.0, 0.5, 1.0))
    if math.isfinite(median):
        floor, ceiling = PARSELMOUTH_F0_MIN * max(formant, 1.0), PARSELMOUTH_F0_MAX * min(formant, 1.0)
        target = min(max(median * pitch, floor), ceiling)
        pitch_range = max(0.0, min(pitch_range, (target - floor) / max(median - lowest, 1e-6),
                                   (ceiling - target) / max(highest - median, 1e-6)))
    else:
        target, pitch_range = 0.0, 1.0
    shifted = call(sound, 'Change gender', PARSELMOUTH_F0_MIN, PARSELMOUTH_F0_MAX, formant, target, pitch_range, 1.0)
    values = shifted.values[0].astype(np.float32)
    return np.pad(values, (0, max(0, len(audio) - len(values))))[:len(audio)]


def _praat_worker(connection):
    while True:
        job = connection.recv()
        try:
            connection.send(change_gender(*job))
        except Exception as error:
            connection.send(error)


class PraatWorker:
    def __init__(self):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_praat_worker, args=(child,), daemon=True)
        self.process.start()

    def run(self, job, timeout=PRAAT_TIMEOUT):
        self.connection.send(job)
        if self.connection.poll(timeout):
            result = self.connection.recv()
            if isinstance(result, Exception):
                raise result
            return result
        self.process.kill()
        self.process.join()
        return None


_WORKER = None


def shift_voice(audio, sample_rate, formant, pitch, pitch_range):
    global _WORKER
    if _WORKER is None or not _WORKER.process.is_alive():
        _WORKER = PraatWorker()
    return _WORKER.run((audio, sample_rate, formant, pitch, pitch_range))


def perturb(audio, sample_rate, parameters):
    audio = np.asarray(audio, dtype=np.float32)
    peak = float(np.abs(audio).max())
    shaped = random_equalizer(audio, sample_rate, parameters['gains'], parameters['qs'])
    try:
        shifted = shift_voice(shaped, sample_rate, parameters['formant'], parameters['pitch'], parameters['pitch_range'])
        reason = None if shifted is not None else f'did not finish within {PRAAT_TIMEOUT:g} s'
    except Exception as error:
        shifted, reason = None, str(error).splitlines()[0]
    if shifted is None:
        print(f'Praat could not shift a clip ({reason}); that perturbed copy keeps only the EQ.', flush=True)
        shifted = shaped
    shifted = np.nan_to_num(shifted)
    level = float(np.abs(shifted).max())
    return shifted * (peak / level) if level > 0 else shifted
