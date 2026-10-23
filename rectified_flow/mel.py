import torch
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.nn import functional as F


MEL_DEFAULTS = {"mel_floor": 1e-5, "mel_scale": "slaney", "mel_norm": "slaney", "mel_center": False}


class LogMel(nn.Module):
    def __init__(self, sample_rate, n_fft, win_length, hop_length, n_mels, fmin, fmax, floor=1e-5,
                 mel_scale="slaney", norm="slaney", center=False):
        super().__init__()
        self.floor = float(floor)
        self.n_fft = int(n_fft)
        self.win_length = int(win_length)
        self.hop_length = int(hop_length)
        self.center = bool(center)
        basis = librosa_mel_fn(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=mel_scale == "htk", norm=norm
        )
        self.register_buffer("basis", torch.from_numpy(basis).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)

    @classmethod
    def from_config(cls, data: dict) -> "LogMel":
        return cls(
            data["sample_rate"],
            data["n_fft"],
            data["win_length"],
            data["hop_length"],
            data["n_mels"],
            data["mel_fmin"],
            data["mel_fmax"],
            *(data.get(key, MEL_DEFAULTS[key]) for key in MEL_DEFAULTS),
        )

    def forward(self, audio: torch.Tensor, key_shift: float = 0.0, hop_length=None) -> torch.Tensor:
        hop_length = int(hop_length or self.hop_length)
        factor = 2.0 ** (key_shift / 12.0)
        n_fft = int(round(self.n_fft * factor))
        win_length = int(round(self.win_length * factor))
        window = self.window
        if win_length != self.win_length:
            window = torch.hann_window(win_length, device=audio.device)
        pad = win_length - hop_length
        padding = (n_fft // 2, n_fft // 2) if self.center else (pad // 2, (pad + 1) // 2)
        audio = F.pad(audio.float().unsqueeze(1), padding, mode="reflect").squeeze(1)
        spec = torch.stft(
            audio,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=False,
            return_complex=True,
        ).abs()
        if n_fft != self.n_fft:
            bins = self.n_fft // 2 + 1
            spec = F.pad(spec, (0, 0, 0, max(0, bins - spec.shape[1])))[:, :bins]
            spec = spec * (self.win_length / win_length)
        return torch.log(torch.clamp(self.basis @ spec, min=self.floor))


def normalize_mel(mel: torch.Tensor, data: dict) -> torch.Tensor:
    return (mel - data["mel_mean"]) / data["mel_std"]


def denormalize_mel(mel: torch.Tensor, data: dict) -> torch.Tensor:
    return mel * data["mel_std"] + data["mel_mean"]
