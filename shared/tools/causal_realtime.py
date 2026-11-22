from threading import Lock

import numpy as np
import torch

from causalvc.limiter import soft_clip
from causalvc.noise_gate import NoiseGate
from causalvc.realtime_backend import CausalVC, SAMPLE_RATE
from causalvc.treble_softener import DELAY_SECONDS, TrebleSoftener


class CausalRealtime:
    def __init__(self, settings, device):
        self.settings = settings
        self.device = device
        self.lock = Lock()
        self.voice = CausalVC(
            key=settings["pitch"],
            model_path=settings["model_path"],
            device=device,
            cuda_graphs=True,
            index_rate=settings["index_rate"],
        )
        self.gate = NoiseGate(SAMPLE_RATE)
        self.softener = TrebleSoftener()

    @property
    def delay_seconds(self):
        return self.voice.delay_seconds + DELAY_SECONDS

    def prepare(self, sample_rate):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("Causal Voice Changer requires a 48 kHz audio route.")
        multiple = self.voice.block_multiple
        frames = max(1, round(self.settings["block_time"] * SAMPLE_RATE / multiple)) * multiple
        self.gate.reset()
        self.softener = TrebleSoftener()
        self.voice.warm(frames)
        return frames

    def process(self, samples):
        with self.lock:
            block = self.gate.process(samples, self.settings["threshold"])
            output = self.voice.process(torch.from_numpy(block).to(self.device)).cpu().numpy()
            return self.softener.process(output)

    def change_pitch(self, pitch):
        with self.lock:
            self.voice.change_key(pitch)

    def change_index_rate(self, rate):
        with self.lock:
            self.voice.change_index_rate(rate)

    @staticmethod
    def output_gain(samples, gain):
        return soft_clip(samples * np.float32(gain))
