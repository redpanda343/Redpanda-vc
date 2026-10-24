import math

import torch
from torch import nn

from rectified_flow.mel import MEL_DEFAULTS

MEL_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels", "mel_fmin", "mel_fmax", *MEL_DEFAULTS)

class RawMelVocoder(nn.Module):
    def __init__(self, generator: nn.Module, data: dict):
        super().__init__()
        self.generator = generator
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])

    def forward(self, mel, f0):
        return self.generator(mel * self.mel_std + self.mel_mean, f0)


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
    elif state is None:
        raise ValueError(f"{path} is not an NSF-HiFiGAN, Vocos or Pupu-Vocoder vocoder.")
    else:
        hparams, vocoder_data, weights = openvpi_spec(path, state)
        generator = NSFHiFiGAN(**hparams)
        generator.load_state_dict(weights)
    model = RawMelVocoder(generator, data)
    mismatch = mel_mismatch(data, vocoder_data)
    if mismatch:
        raise ValueError(f"{path} renders another mel than the flow's: {mismatch}.")
    return model.float().eval().requires_grad_(False), vocoder_data
