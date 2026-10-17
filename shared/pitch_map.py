import math

import numpy as np

PITCH_OPTIONS = ("pitch", "dataset_hz", "pitch_range")
F0_FLOOR = 50.0
F0_CEILING = 1100.0
TRACK_MARGIN = 3.0
LIMIT_KNEE = 2.0
SOURCE_FRAMES = 1000
SOURCE_WARMUP = 50


def pitch_options(settings):
    return {key: settings[key] for key in PITCH_OPTIONS if key in settings}


class PitchMap:
    def __init__(self, **options):
        self.semitones = 0.0
        self.dataset_hz = 0.0
        self.range = 12.0
        self.configure(**options)
        self.reset()

    def configure(self, pitch=None, dataset_hz=None, pitch_range=None):
        if pitch is not None:
            self.semitones = float(pitch)
        if dataset_hz is not None:
            self.dataset_hz = max(float(dataset_hz), 0.0)
        if pitch_range is not None:
            self.range = min(max(float(pitch_range), 1.0), 36.0)

    def reset(self):
        self.source_mean = None
        self.source_weight = 0

    @property
    def plain(self):
        return self.dataset_hz <= 0

    @property
    def voice_hz(self):
        if self.source_mean is None or self.source_weight < SOURCE_WARMUP:
            return None
        return 2.0 ** self.source_mean

    def observe(self, f0):
        f0 = np.asarray(f0, dtype=np.float64).reshape(-1)
        f0 = f0[f0 > 0]
        if not f0.size:
            return
        count = f0.size
        total = max(min(self.source_weight + count, SOURCE_FRAMES), count)
        mean = 0.0 if self.source_mean is None else self.source_mean
        self.source_mean = mean + (float(np.log2(f0).sum()) - count * mean) / total
        self.source_weight = total

    def band(self):
        if self.plain:
            return None
        width = (self.range + TRACK_MARGIN) / 12
        center = math.log2(self.dataset_hz) - self.semitones / 12
        low = max(2.0 ** (center - width), F0_FLOOR)
        high = min(2.0 ** (center + width), F0_CEILING)
        if high < low * 2.0 ** (1 / 12):
            return None
        return low, high

    def _limit(self, log_f0, center):
        width = self.range / 12
        knee = min(LIMIT_KNEE / 12, width / 2)
        inner = width - knee
        offset = log_f0 - center
        excess = np.abs(offset) - inner
        over = excess > 0
        offset[over] = np.sign(offset[over]) * (inner + knee * np.tanh(excess[over] / knee))
        return center + offset

    def apply(self, f0):
        f0 = np.asarray(f0)
        if self.plain:
            return f0 * 2 ** (self.semitones / 12)
        f0 = f0.astype(np.float64)
        mapped = np.zeros_like(f0)
        voiced = f0 > 0
        if not voiced.any():
            return mapped
        log_f0 = np.log2(f0[voiced]) + self.semitones / 12
        mapped[voiced] = np.exp2(self._limit(log_f0, math.log2(self.dataset_hz)))
        return mapped
