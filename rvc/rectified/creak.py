import csv
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

CREAPY_MODEL = Path(__file__).resolve().parent / 'creapy' / 'model_ALL.csv'
CREAPY_FEATURES = ('hnr', 'jitter', 'h1h2', 'shimmer', 'f0mean')
CREAPY_BLOCK = 0.04
CREAPY_HOP = 0.01
CREAPY_ZCR_THRESHOLD = 0.10
CREAPY_STE_THRESHOLD = 1e-5
CREAK_AVERAGE = 0.05
CREAK_THRESHOLD = 0.3
FRY_FLOOR_DB = 30.0
FRY_MANIFEST = 'vocal_fry.json'
FRY_FOLDER = 'vocal_fry'


@lru_cache(maxsize=1)
def creapy_model():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer

    with open(CREAPY_MODEL, newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    features = np.array([[float(row[name]) for name in CREAPY_FEATURES] for row in rows])
    labels = np.array([row['class'] for row in rows])
    imputer = SimpleImputer(strategy='median').fit(features)
    forest = RandomForestClassifier(n_estimators=99, random_state=42).fit(imputer.transform(features), labels)
    return imputer, forest, int(np.flatnonzero(forest.classes_ == 'c')[0])


def _blocks(audio, sample_rate):
    from scipy.signal.windows import hann

    size, hop = int(CREAPY_BLOCK * sample_rate), int(CREAPY_HOP * sample_rate)
    if len(audio) < size:
        audio = np.pad(audio, (0, size - len(audio)))
    count = int(np.ceil((len(audio) - size) / hop + 1))
    window = hann(size)
    blocks = np.zeros((count, size))
    for index in range(count - 1):
        blocks[index] = audio[index * hop:index * hop + size] * window
    tail = audio[(count - 1) * hop:]
    blocks[count - 1, :len(tail)] = tail
    return blocks


def _praat_features(block, sample_rate):
    import parselmouth
    from scipy.interpolate import interp1d

    sound = parselmouth.Sound(values=block, sampling_frequency=sample_rate)
    try:
        harmonicity = sound.to_harmonicity().values
        voiced = harmonicity[harmonicity != -200]
        hnr = voiced.mean() if voiced.size else np.nan
    except parselmouth.PraatError:
        hnr = np.nan
    try:
        points = parselmouth.praat.call(sound, 'To PointProcess (periodic, cc)', 75, 500)
        jitter = parselmouth.praat.call(points, 'Get jitter (local)', 0, 0, 0.0001, 0.02, 1.3)
    except Exception:
        points, jitter = None, np.nan
    try:
        shimmer = parselmouth.praat.call([sound, points], 'Get shimmer (local)', 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        shimmer = np.nan
    try:
        f0 = sound.to_pitch(sound.duration).selected_array[0][0]
    except Exception:
        f0 = np.nan
    try:
        spectrum = sound.to_spectrum()
        amplitude = interp1d(np.arange(spectrum.nf) * spectrum.df,
                             np.sqrt(spectrum.values[0] ** 2 + spectrum.values[1] ** 2), 'quadratic')
        h1h2 = float(amplitude(f0) - amplitude(2 * f0))
    except Exception:
        h1h2 = np.nan
    return [hnr, jitter, h1h2, shimmer, f0]


def creak_probability(audio, sample_rate):
    audio = np.asarray(audio, dtype=np.float64)
    audio = audio.mean(-1) if audio.ndim == 2 else audio
    peak = np.abs(audio).max()
    blocks = _blocks(audio / peak if peak > 0 else audio, sample_rate)
    signs = np.sign(blocks)
    signs[signs == 0] = 1
    zcr = 0.5 * np.abs(np.diff(signs, axis=1)).sum(1) / blocks.shape[1]
    ste = np.square(blocks).sum(1) / blocks.shape[1]
    zcr = zcr / zcr.max() if zcr.max() > 0 else zcr
    included = (zcr < CREAPY_ZCR_THRESHOLD) & (ste > CREAPY_STE_THRESHOLD)
    probability = np.zeros(len(blocks))
    if included.any():
        imputer, forest, column = creapy_model()
        features = np.array([_praat_features(block, sample_rate) for block in blocks[included]], dtype=np.float64)
        probability[included] = forest.predict_proba(imputer.transform(features))[:, column]
    times = CREAPY_BLOCK / 2 + np.arange(len(blocks)) * CREAPY_HOP
    return times, probability


def creak_labels(audio, sample_rate, hop, frames):
    times, probability = creak_probability(audio, sample_rate)
    curve = np.interp((np.arange(frames) + 0.5) * hop / sample_rate, times, probability)
    width = int(round(CREAK_AVERAGE * sample_rate / hop))
    curve = np.convolve(curve, np.ones(2 * width + 1) / (2 * width + 1), mode='same')
    return (curve >= CREAK_THRESHOLD).astype(np.float32)


def fry_labels(audio, hop, frames):
    audio = np.asarray(audio, dtype=np.float64)
    audio = np.pad(audio, (0, max(0, frames * hop - len(audio))))[:frames * hop]
    level = 10 * np.log10(np.square(audio.reshape(frames, hop)).mean(1) + 1e-12)
    return (level > level.max() - FRY_FLOOR_DB).astype(np.float32)


def fry_sources(experiment):
    path = Path(experiment) / FRY_MANIFEST
    if not path.is_file():
        return []
    return sorted(int(value) for value in json.loads(path.read_text(encoding='utf-8'))['sources'])


def is_fry_slice(path, sources):
    parts = Path(path).stem.split('_')
    return len(parts) == 3 and parts[1].isdigit() and int(parts[1]) in sources
