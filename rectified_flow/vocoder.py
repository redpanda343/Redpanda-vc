import math

import numpy as np
import torch
from torch import nn

from rectified_flow.mel import MEL_DEFAULTS, LogMel

MEL_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels", "mel_fmin", "mel_fmax", *MEL_DEFAULTS)
CHUNK_FRAMES = 1024

class RawMelVocoder(nn.Module):
    def __init__(self, generator: nn.Module, data: dict):
        super().__init__()
        self.generator = generator
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])

    def forward(self, mel, f0):
        return self.generator(mel * self.mel_std + self.mel_mean, f0)


class ChunkedInference:
    name = "vocoder"
    build = None

    def __init__(self, path, checkpoint, device):
        self.device = torch.device(device)
        generator, self.data = self.build(path, checkpoint)
        self.generator = generator.to(self.device).float().eval().requires_grad_(False)
        self.context = math.ceil(self.generator.radius)
        self.mel = LogMel.from_config(self.data).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, waveform, pitch=0, f0_method="rmvpe", seed=None):
        if float(pitch):
            raise ValueError(f"{self.name} has no pitch input. Set Pitch to 0.")
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError("Select non-empty audio with finite samples.")
        hop, length = int(self.data["hop_length"]), len(waveform)
        padded_length = max(math.ceil(length / hop), math.ceil(self.data["n_fft"] / hop)) * hop
        padded = np.pad(np.clip(waveform, -1.0, 1.0), (0, padded_length - length))
        mel = self.mel(torch.from_numpy(padded).to(self.device)[None])
        frames, pieces = mel.shape[-1], []
        for start in range(0, frames, CHUNK_FRAMES):
            end = min(start + CHUNK_FRAMES, frames)
            left, right = max(0, start - self.context), min(frames, end + self.context)
            pieces.append(self.generator(mel[..., left:right])[0, 0, (start - left) * hop:(end - left) * hop])
        audio = torch.cat(pieces)[:length]
        if not torch.isfinite(audio).all():
            raise FloatingPointError(f"Non-finite {self.name} audio output.")
        return audio.float().cpu().numpy()


def _same_mel_value(left, right):
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool):
        return math.isclose(float(left), float(right), rel_tol=1e-9)
    return left == right


def mel_mismatch(data: dict, vocoder_data: dict):
    keys = MEL_KEYS + tuple(k for k in ("mel_mean", "mel_std") if k in vocoder_data)
    for key in keys:
        if key not in vocoder_data:
            continue
        value = data.get(key, MEL_DEFAULTS.get(key))
        if not _same_mel_value(vocoder_data[key], value):
            return f"{key} ({vocoder_data[key]} vs {value})"
    return None


def load_vocoder(path: str, data: dict):
    from rectified_flow.bigvgan import build_bigvgan, is_bigvgan
    from rectified_flow.nsf_bigvgan import build_nsf_bigvgan, is_nsf_bigvgan
    from rectified_flow.openvpi import NSFHiFiGAN, generator_state, onnx_generator_state, openvpi_spec
    from rectified_flow.pupu import build_pupu, is_pupu
    from rectified_flow.vocos import build_vocos, is_vocos

    checkpoint = None
    if str(path).lower().endswith(".onnx"):
        state = onnx_generator_state(path)
    else:
        if str(path).lower().endswith(".safetensors"):
            from safetensors.torch import load_file

            checkpoint = load_file(path)
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = generator_state(checkpoint)
    if is_vocos(checkpoint):
        generator, vocoder_data, _ = build_vocos(path, checkpoint)
    elif is_pupu(checkpoint):
        generator, vocoder_data = build_pupu(path, checkpoint)
    elif is_bigvgan(checkpoint):
        generator, vocoder_data = build_bigvgan(path, checkpoint)
    elif is_nsf_bigvgan(checkpoint):
        generator, vocoder_data = build_nsf_bigvgan(path, checkpoint)
    elif state is None:
        raise ValueError(f"{path} is not an NSF-HiFiGAN, Vocos, Pupu-Vocoder, BigVGAN or NSF-BigVGAN vocoder.")
    else:
        hparams, vocoder_data, weights = openvpi_spec(path, state)
        generator = NSFHiFiGAN(**hparams)
        generator.load_state_dict(weights)
    model = RawMelVocoder(generator, data)
    mismatch = mel_mismatch(data, vocoder_data)
    if mismatch:
        raise ValueError(f"{path} renders another mel than the flow's: {mismatch}.")
    return model.float().eval().requires_grad_(False), vocoder_data
