import torch
from torch import nn

MEL_KEYS = ("sample_rate", "hop_length", "n_fft", "win_length", "n_mels", "mel_fmin", "mel_fmax")

class RawMelVocoder(nn.Module):
    def __init__(self, generator: nn.Module, data: dict):
        super().__init__()
        self.generator = generator
        self.mel_mean = float(data["mel_mean"])
        self.mel_std = float(data["mel_std"])

    def forward(self, mel, f0):
        return self.generator(mel * self.mel_std + self.mel_mean, f0)


def mel_mismatch(data: dict, vocoder_data: dict):
    keys = MEL_KEYS + tuple(k for k in ("mel_mean", "mel_std") if k in vocoder_data)
    for key in keys:
        if key in vocoder_data and float(vocoder_data[key]) != float(data[key]):
            return f"{key} ({vocoder_data[key]} vs {data[key]})"
    return None


def load_vocoder(path: str, data: dict):
    from rectified_flow.openvpi import NSFHiFiGAN, generator_state, onnx_generator_state, openvpi_spec
    from rectified_flow.vocos import build_vocos, is_vocos

    checkpoint = None
    if str(path).lower().endswith(".onnx"):
        state = onnx_generator_state(path)
    else:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        state = generator_state(checkpoint)
    if is_vocos(checkpoint):
        generator, vocoder_data, native = build_vocos(path, checkpoint)
        if native:
            raise ValueError(f"{path} renders Vocos's own mel. Train a Vocos on the flow's mel to use it here.")
    elif state is None:
        raise ValueError(f"{path} is not an NSF-HiFiGAN or Vocos vocoder.")
    else:
        hparams, vocoder_data, weights = openvpi_spec(path, state)
        generator = NSFHiFiGAN(**hparams)
        generator.load_state_dict(weights)
    model = RawMelVocoder(generator, data)
    mismatch = mel_mismatch(data, vocoder_data)
    if mismatch:
        raise ValueError(f"{path} renders another mel than the flow's: {mismatch}.")
    return model.float().eval().requires_grad_(False), vocoder_data
