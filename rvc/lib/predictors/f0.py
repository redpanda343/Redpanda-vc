import os
import torch

from rvc.lib.predictors.rmvpe import N_CLASS, RMVPE as RMVPEModel, to_local_average_f0
from rvc.lib.predictors.swift_dependencies import ensure_swift_f0
from torchfcpe import spawn_bundled_infer_model
import numpy as np


class RMVPE:
    def __init__(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPEModel(
            os.path.join("rvc", "models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03, decoder=None):
        hidden = self.model.infer_hidden(x, self.sample_rate)
        center = None
        if decoder is not None:
            center = np.asarray(decoder(hidden[0].cpu().numpy(), filter_radius), dtype=np.int64)
            if center.shape != (hidden.shape[1],):
                raise ValueError("Pitch decoder returned an invalid shape.")
            center = torch.from_numpy(center).clamp(0, N_CLASS - 1).to(hidden.device).view(1, -1, 1)
        return to_local_average_f0(hidden, center=center, thred=filter_radius)


class FCPE:
    def __init__(self, device, sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = spawn_bundled_infer_model(self.device)

    def get_f0(self, x, p_len=None, filter_radius=0.006):
        if p_len is None:
            p_len = x.shape[0] // self.hop_size

        if not torch.is_tensor(x):
            x = torch.from_numpy(x)

        f0 = (
            self.model.infer(
                x.float().to(self.device).unsqueeze(0),
                sr=self.sample_rate,
                decoder_mode="local_argmax",
                threshold=filter_radius,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

        return f0


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
