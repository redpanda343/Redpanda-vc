"""Zero-latency smooth envelope noise gate for mono audio blocks."""

from collections import deque
import math

import numpy as np

OFF_DB = -60.0
HYSTERESIS_DB = 3.0
STEP_SECONDS = 0.002
RMS_WINDOW_STEPS = 5  # 5 x 2 ms = 10 ms causal RMS detector
ATTACK_SECONDS = 0.003  # time to reach ~99% open (-40 dB error)
HOLD_SECONDS = 0.040
RELEASE_SECONDS = 0.050  # time to decay by 40 dB (gain <= 0.01)
LN_100 = math.log(100.0)  # 4.605: 1 - exp(-T / tau) = 0.99 when tau = T / LN_100


class NoiseGate:
    """Causal envelope gate with hysteresis, hold, and click-free linear sub-frame ramps."""

    def __init__(self, samplerate: int = 48000):
        self.samplerate = int(samplerate)
        self.step = max(1, round(STEP_SECONDS * self.samplerate))
        self.hold_samples = round(HOLD_SECONDS * self.samplerate)
        self.attack_tau = max(1.0, ATTACK_SECONDS * self.samplerate / LN_100)
        self.release_tau = max(1.0, RELEASE_SECONDS * self.samplerate / LN_100)
        self.reset()

    def reset(self) -> None:
        self.gain = 0.0
        self.is_open = False
        self.hold_remaining = 0
        self.sq_window: deque[float] = deque(maxlen=RMS_WINDOW_STEPS)

    def process(self, block: np.ndarray, threshold_db: float) -> np.ndarray:
        audio = np.asarray(block, dtype=np.float32)
        if threshold_db <= OFF_DB:
            self.gain = 1.0
            self.is_open = True
            self.hold_remaining = self.hold_samples
            self.sq_window.clear()
            return audio
        if audio.size == 0:
            return audio

        out = np.empty_like(audio)
        close_db = float(threshold_db) - HYSTERESIS_DB
        pos = 0
        total = audio.shape[0]
        step = self.step

        while pos < total:
            end = min(pos + step, total)
            chunk = audio[pos:end]
            n = end - pos
            self.sq_window.append(float(np.mean(chunk * chunk)))
            mean_sq = sum(self.sq_window) / len(self.sq_window)
            level_db = 10.0 * math.log10(mean_sq + 1e-12)

            if level_db >= threshold_db:
                self.is_open = True
                self.hold_remaining = self.hold_samples
            elif self.is_open:
                if level_db >= close_db:
                    self.hold_remaining = self.hold_samples
                elif self.hold_remaining > 0:
                    self.hold_remaining = max(0, self.hold_remaining - n)
                else:
                    self.is_open = False

            prev_gain = self.gain
            if self.is_open:
                alpha = 1.0 - math.exp(-n / self.attack_tau)
                new_gain = prev_gain + alpha * (1.0 - prev_gain)
                if new_gain > 0.99999:
                    new_gain = 1.0
            else:
                new_gain = prev_gain * math.exp(-n / self.release_tau)
                if new_gain < 1e-7:
                    new_gain = 0.0
            self.gain = new_gain

            if prev_gain == 1.0 and new_gain == 1.0:
                out[pos:end] = chunk
            elif prev_gain == 0.0 and new_gain == 0.0:
                out[pos:end] = 0.0
            else:
                ramp = np.linspace(prev_gain, new_gain, n, endpoint=True, dtype=np.float32)
                out[pos:end] = chunk * ramp
            pos = end

        return out
