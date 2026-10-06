import dataclasses
import os
import pathlib
import traceback

import numpy as np
import parselmouth

from rvc.configs.config import Config
from rvc.lib.predictors.rmvpe import RMVPE as RMVPEModel
from rvc.lib.utils import load_audio

config = Config()


@dataclasses.dataclass
class F0Extractor:
    wav_path: pathlib.Path
    method: str = "rmvpe"
    sample_rate: int = 16000
    hop_length: int = 160
    f0_min: int = 50
    f0_max: int = 1100

    def __post_init__(self):
        self.sample_rate = 16000
        self.x = load_audio(str(self.wav_path), self.sample_rate)

    def extract_f0(self):
        p_len = self.x.shape[0] // self.hop_length

        if self.method == "pm":
            time_step = self.hop_length / self.sample_rate * 1000
            f0 = (
                parselmouth.Sound(self.x, self.sample_rate)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=self.f0_min,
                    pitch_ceiling=self.f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif self.method == "rmvpe":
            model_rmvpe = RMVPEModel(
                os.path.join("rvc", "models", "predictors", "rmvpe.pt"),
                device=config.device,
            )
            f0 = model_rmvpe.infer_from_audio(self.x, thred=0.03)
        else:
            raise ValueError(f"Unsupported F0 method: {self.method}")

        f0 = np.asarray(f0)
        try:
            uv = f0 == 0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        except Exception:
            traceback.print_exc()
            return None
        return f0
