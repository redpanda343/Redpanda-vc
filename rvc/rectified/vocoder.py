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
    from rvc.rectified.openvpi import ARCHITECTURE, NSFHiFiGAN, generator_state, openvpi_spec

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") == "rectified_vocoder" and checkpoint.get("architecture") == ARCHITECTURE:
        generator = NSFHiFiGAN(**checkpoint["config"]["vocoder"]["model"])
        generator.load_state_dict(checkpoint["model"])
        model = RawMelVocoder(generator, data)
        vocoder_data = checkpoint["config"]["data"]
    else:
        state = generator_state(checkpoint)
        if state is None:
            raise ValueError(f"{path} is not an OpenVPI NSF-HiFiGAN checkpoint.")
        hparams, vocoder_data, weights = openvpi_spec(path, state)
        generator = NSFHiFiGAN(**hparams)
        generator.load_state_dict(weights)
        model = RawMelVocoder(generator, data)
    mismatch = mel_mismatch(data, vocoder_data)
    if mismatch:
        raise ValueError(f"{path} renders another mel than the flow's: {mismatch}.")
    return model.float().eval().requires_grad_(False), vocoder_data
