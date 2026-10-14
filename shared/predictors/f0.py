import math
import os
import torch

from beatrice.inference import (
    BINS_PER_OCTAVE,
    IN_HOP,
    IN_SAMPLE_RATE,
    PITCH_BAND,
    PITCH_BASE_HZ,
    PITCH_WINDOW,
    PitchEstimator,
    Weights,
    checkpoint_pitch_values,
)
from beatrice.resources import ensure_pitch_estimator
from shared.predictors.rmvpe import N_CLASS, RMVPE as RMVPEModel, to_local_average_f0
from shared.predictors.RMVPE import RMVPE0Predictor
from shared.predictors.swift_dependencies import ensure_swift_f0
import numpy as np


class RMVPE:
    def __init__(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPEModel(
            os.path.join("models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03, decoder=None, network=None):
        hidden = self.model.infer_hidden(x, self.sample_rate, network)
        center = None
        if decoder is not None:
            center = np.asarray(decoder(hidden[0].cpu().numpy(), filter_radius), dtype=np.int64)
            if center.shape != (hidden.shape[1],):
                raise ValueError("Pitch decoder returned an invalid shape.")
            center = torch.from_numpy(center).clamp(0, N_CLASS - 1).to(hidden.device).view(1, -1, 1)
        return to_local_average_f0(hidden, center=center, thred=filter_radius)


class RVCRMVPE:
    def __init__(self, device, model_name="rmvpe_rvc.pt", sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPE0Predictor(
            os.path.join("models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03, decoder=None, network=None):
        return self.model.infer_from_audio(x, thred=filter_radius, decoder=decoder, network=network)


class Swift:
    def __init__(
        self, device, sample_rate=16000, hop_size=160, *, threads=None, spin=True
    ):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SwiftF0 inference requires a CUDA device.")
        SwiftF0 = ensure_swift_f0().SwiftF0
        self.model = SwiftF0(threads=threads, spin=spin)
        self.model.session.set_providers(
            [
                ("CUDAExecutionProvider", {"device_id": self.device.index or 0}),
                "CPUExecutionProvider",
            ]
        )
        if "CUDAExecutionProvider" not in self.model.session.get_providers():
            raise RuntimeError("SwiftF0 could not initialize its CUDA execution provider.")
        self.model.session.disable_fallback()

    @staticmethod
    def _repair_subharmonics(pitch, confidence, frame_period=0.016):
        pitch = np.asarray(pitch, dtype=np.float64)
        confidence = np.asarray(confidence, dtype=np.float64)
        corrected = pitch.copy()
        repaired = np.zeros(len(pitch), dtype=bool)
        i = 1
        while i < len(pitch) - 1:
            if confidence[i - 1] < 0.5 or not 0.3 <= confidence[i] < 0.95:
                i += 1
                continue
            ratio = pitch[i - 1] / pitch[i]
            factor = min((2, 3), key=lambda value: abs(np.log2(ratio / value)))
            if abs(1200 * np.log2(ratio / factor)) > 100:
                i += 1
                continue
            j = i
            while j < len(pitch) and (j - i) * frame_period < 1.0:
                if not 0.3 <= confidence[j] < 0.95:
                    break
                previous = pitch[i - 1] if j == i else pitch[j - 1] * factor
                if abs(1200 * np.log2(pitch[j] * factor / previous)) > 100:
                    break
                j += 1
            if j > i and j < len(pitch) and confidence[j] >= 0.5:
                return_error = abs(1200 * np.log2(pitch[j] / (pitch[j - 1] * factor)))
                anchor_error = abs(1200 * np.log2(pitch[j] / pitch[i - 1]))
                if return_error <= 100 and anchor_error <= 150:
                    corrected[i:j] *= factor
                    repaired[i:j] = True
                    i = j + 1
                    continue
            i += 1
        return corrected, repaired

    def get_f0(self, x, p_len=None, f0_min=50.0, f0_max=1100.0, threshold=0.5):
        from swift_f0 import FRAME_PERIOD

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if p_len is None:
            p_len = np.asarray(x).shape[0] // self.hop_size
        if p_len <= 0:
            return np.zeros(0, dtype=np.float64)
        result = self.model.detect(x, self.sample_rate, fmin=f0_min, fmax=f0_max)
        pitch, repaired = self._repair_subharmonics(
            result.pitch_hz, result.confidence, FRAME_PERIOD
        )
        repaired &= (pitch >= f0_min) & (pitch <= f0_max)
        pitch = np.where(repaired, pitch, result.pitch_hz)
        confidence = np.where(
            repaired, np.maximum(result.confidence, threshold), result.confidence
        )
        t_src = result.timestamps
        t_tgt = np.arange(p_len) * self.hop_size / self.sample_rate
        voiced = confidence >= threshold
        if not np.any(voiced):
            return np.zeros(p_len, dtype=np.float64)
        f0 = np.exp2(np.interp(t_tgt, t_src[voiced], np.log2(pitch[voiced])))
        f0[np.interp(t_tgt, t_src, confidence) < threshold] = 0.0
        return f0


class BeatricePitch:
    padding = PITCH_WINDOW // 2
    latency = PITCH_WINDOW + IN_HOP - padding
    lookahead_frames = math.ceil((latency - IN_HOP) / IN_HOP)
    chunk_samples = 10 * IN_SAMPLE_RATE

    def __init__(self, device, sample_rate=16000, hop_size=160, threshold=0.7):
        if sample_rate != IN_SAMPLE_RATE or hop_size != IN_HOP:
            raise ValueError("The Beatrice pitch estimator requires 16 kHz audio with a 160-sample hop.")
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.threshold = threshold
        path = ensure_pitch_estimator()
        self.model = PitchEstimator(Weights(path, self.device, checkpoint_pitch_values(path)), self.device)
        self.offsets = torch.arange(1, PITCH_BAND + 1, device=self.device)
        self.reset()

    def reset(self):
        self.model.reset()
        self.audio = torch.zeros(self.padding, device=self.device)
        self.samples = 0
        self.frames = 0

    def _decode(self, probabilities):
        band = probabilities[:, 1:].unfold(1, PITCH_BAND, 1)
        confidence, top = band.sum(-1).max(1)
        weights = band[torch.arange(band.shape[0], device=self.device), top]
        position = (weights * (top[:, None] + self.offsets)).sum(1) / weights.sum(1)
        f0 = PITCH_BASE_HZ * 2.0 ** (position / BINS_PER_OCTAVE)
        return torch.where(confidence > self.threshold, f0, 0.0)

    @torch.inference_mode()
    def push(self, audio):
        audio = torch.as_tensor(audio, dtype=torch.float32, device=self.device).reshape(-1)
        self.audio = torch.cat([self.audio, audio])
        self.samples += audio.shape[0]
        frames = (self.audio.shape[0] - PITCH_WINDOW) // IN_HOP
        if frames <= 0:
            return torch.zeros(0, device=self.device)
        instfreq, difference, _ = self.model.inputs(self.audio, frames)
        probabilities = self.model.probabilities(instfreq, difference)
        self.audio = self.audio[frames * IN_HOP:]
        self.frames += frames
        return self._decode(probabilities)

    def get_f0(self, x, p_len=None):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if p_len is None:
            p_len = x.shape[0] // self.hop_size
        self.reset()
        x = np.pad(x, (0, self.latency))
        f0 = torch.cat([self.push(x[start:start + self.chunk_samples])
                        for start in range(0, x.shape[0], self.chunk_samples)])
        f0 = f0[:p_len].cpu().numpy()
        return np.pad(f0, (0, p_len - f0.shape[0]))
