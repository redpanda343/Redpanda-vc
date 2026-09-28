import os
import torch

from rvc.lib.predictors.RMVPE import RMVPE0Predictor
from torchfcpe import spawn_bundled_infer_model
import numpy as np


class RMVPE:
    def __init__(self, device, model_name="rmvpe.pt", sample_rate=16000, hop_size=160):
        self.device = device
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.model = RMVPE0Predictor(
            os.path.join("rvc", "models", "predictors", model_name),
            device=self.device,
        )

    def get_f0(self, x, filter_radius=0.03, decoder=None):
        f0 = self.model.infer_from_audio(
            x, thred=filter_radius, decoder=decoder
        )
        return f0


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
    def __init__(self, device, sample_rate=16000, hop_size=160):
        from swift_f0 import SwiftF0
        import onnxruntime as ort

        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("SwiftF0 inference requires a CUDA device.")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("SwiftF0 requires onnxruntime-gpu with CUDAExecutionProvider.")

        self.model = SwiftF0()
        self.model.session.set_providers(
            [("CUDAExecutionProvider", {"device_id": self.device.index or 0})]
        )
        if "CUDAExecutionProvider" not in self.model.session.get_providers():
            raise RuntimeError("SwiftF0 could not initialize its CUDA execution provider.")
        self.model.session.disable_fallback()

    def get_f0(self, x, p_len=None, f0_min=50.0, f0_max=1100.0, threshold=0.5):
        from swift_f0 import FRAME_PERIOD

        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if p_len is None:
            p_len = np.asarray(x).shape[0] // self.hop_size
        if p_len <= 0:
            return np.zeros(0, dtype=np.float64)

        result = self.model.detect(x, self.sample_rate, fmin=f0_min, fmax=f0_max)
        t_src = np.arange(result.pitch_hz.shape[0]) * FRAME_PERIOD
        t_tgt = np.arange(p_len) * (self.hop_size / self.sample_rate)
        conf_tgt = np.interp(t_tgt, t_src, result.confidence)
        voiced = result.confidence >= threshold
        if not np.any(voiced):
            return np.zeros(p_len, dtype=np.float64)
        f0_tgt = np.power(
            2.0, np.interp(t_tgt, t_src[voiced], np.log2(result.pitch_hz[voiced]))
        )
        f0_tgt[conf_tgt < threshold] = 0.0
        return f0_tgt
