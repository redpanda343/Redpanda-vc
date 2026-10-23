import json
import math
import os

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from rectified_flow.openvpi import CONFIG_MEL_KEYS, DEFAULT_MEL

PADDINGS = ("center", "same")
FORK_MEL_ARGS = ("f_min", "f_max", "norm", "mel_scale")


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, intermediate_dim, layer_scale):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(layer_scale)))

    def forward(self, x):
        y = self.dwconv(x).transpose(1, 2)
        y = self.pwconv2(F.gelu(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).transpose(1, 2)


class VocosBackbone(nn.Module):
    def __init__(self, num_mels, dim, intermediate_dim, num_layers):
        super().__init__()
        self.embed = nn.Conv1d(num_mels, dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(ConvNeXtBlock(dim, intermediate_dim, 1 / num_layers) for _ in range(num_layers))
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, mel):
        x = self.norm(self.embed(mel).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        return self.final_layer_norm(x.transpose(1, 2))


class ISTFT(nn.Module):
    def __init__(self, n_fft, hop_length, padding):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.trim = n_fft // 2 if padding == "center" else (n_fft - hop_length) // 2
        self.register_buffer("window", torch.hann_window(n_fft))

    def _overlap_add(self, frames, size):
        return F.fold(frames, (1, size), (1, self.n_fft), stride=(1, self.hop_length))[:, 0, 0]

    def forward(self, spec):
        count = spec.shape[-1]
        size = (count - 1) * self.hop_length + self.n_fft
        audio = self._overlap_add(torch.fft.irfft(spec, self.n_fft, dim=1) * self.window[None, :, None], size)
        envelope = self._overlap_add(self.window.square()[None, :, None].expand(1, -1, count), size)
        audio = audio / envelope.clamp(min=1e-11)
        return audio[:, self.trim:size - self.trim]


class ISTFTHead(nn.Module):
    def __init__(self, dim, n_fft, hop_length, padding):
        super().__init__()
        self.out = nn.Linear(dim, n_fft + 2)
        self.istft = ISTFT(n_fft, hop_length, padding)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            magnitude, phase = self.out(x.float()).transpose(1, 2).chunk(2, dim=1)
            return self.istft(torch.polar(torch.exp(magnitude).clamp(max=1e2), phase))


class Vocos(nn.Module):
    def __init__(self, num_mels, dim, intermediate_dim, num_layers, n_fft, hop_length, padding):
        super().__init__()
        self.backbone = VocosBackbone(num_mels, dim, intermediate_dim, num_layers)
        self.head = ISTFTHead(dim, n_fft, hop_length, padding)
        self.extend = padding == "center"
        self.radius = 3 + 3 * num_layers + n_fft / (2 * hop_length)

    def forward(self, mel, f0=None):
        if self.extend:
            mel = F.pad(mel, (0, 1), mode="replicate")
        return self.head(self.backbone(mel)).unsqueeze(1)


class VocosMel(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, padding, f_min=0.0, f_max=None, norm=None,
                 mel_scale="htk", floor=1e-7):
        super().__init__()
        self.pad = (n_fft - hop_length) // 2 if padding == "same" else 0
        self.floor = floor
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
            center=padding == "center", power=1, f_min=f_min, f_max=f_max, norm=norm, mel_scale=mel_scale,
        )

    def forward(self, audio):
        if self.pad:
            audio = F.pad(audio.unsqueeze(1), (self.pad, self.pad), mode="reflect").squeeze(1)
        return torch.log(torch.clamp(self.mel(audio), min=self.floor))


def vocos_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    state = checkpoint.get("generator", checkpoint)
    lightning = checkpoint.get("state_dict")
    if isinstance(lightning, dict) and "generator.backbone.embed.weight" in lightning:
        state = {key[len("generator."):]: value for key, value in lightning.items() if key.startswith("generator.")}
    if not isinstance(state, dict) or "backbone.embed.weight" not in state or "head.out.weight" not in state:
        return None
    return {key: value for key, value in state.items() if key.startswith(("backbone.", "head."))}


def is_vocos(checkpoint):
    return vocos_state(checkpoint) is not None


def _read_vocos_config(path, checkpoint):
    metadata = checkpoint.get("vocoder")
    if isinstance(metadata, dict) and isinstance(metadata.get("config"), dict):
        config = metadata["config"]
        return dict(config["data"]), config["model"]["padding"], None
    folder = os.path.dirname(path)
    json_path, yaml_path = os.path.join(folder, "config.json"), os.path.join(folder, "config.yaml")
    if os.path.isfile(json_path):
        with open(json_path, encoding="utf-8") as handle:
            config = json.load(handle)
        mel = dict(DEFAULT_MEL)
        for source, target in CONFIG_MEL_KEYS.items():
            if source in config:
                mel[target] = config[source]
        return mel, config.get("padding", "same"), None
    if os.path.isfile(yaml_path):
        import yaml

        with open(yaml_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
        extractor = config.get("feature_extractor", {})
        if not str(extractor.get("class_path", "")).endswith("MelSpectrogramFeatures"):
            raise ValueError(f"{path} is not a mel-spectrogram Vocos.")
        args = extractor["init_args"]
        native = dict(sample_rate=args["sample_rate"], n_fft=args["n_fft"], hop_length=args["hop_length"],
                      n_mels=args["n_mels"], padding=args.get("padding", "center"),
                      f_min=float(args.get("f_min", 0.0)), f_max=args.get("f_max"), norm=args.get("norm"),
                      mel_scale=args.get("mel_scale", "htk"),
                      floor=1e-5 if any(key in args for key in FORK_MEL_ARGS) else 1e-7)
        mel = dict(sample_rate=args["sample_rate"], hop_length=args["hop_length"], n_fft=args["n_fft"],
                   win_length=args["n_fft"], n_mels=args["n_mels"], mel_fmin=native["f_min"],
                   mel_fmax=float(native["f_max"] or args["sample_rate"] / 2), mel_floor=native["floor"],
                   mel_scale=native["mel_scale"], mel_norm=native["norm"],
                   mel_center=native["padding"] == "center")
        return mel, config.get("head", {}).get("init_args", {}).get("padding", "center"), native
    raise ValueError(f"{path} needs its config.json or config.yaml beside it.")


def build_vocos(path, checkpoint):
    state = vocos_state(checkpoint)
    if state is None:
        raise ValueError(f"{path} is not a Vocos vocoder.")
    mel, padding, native = _read_vocos_config(path, checkpoint)
    if padding not in PADDINGS:
        raise ValueError(f"{path} uses unsupported Vocos padding {padding!r}.")
    layers = sum(1 for key in state if key.startswith("backbone.convnext.") and key.endswith(".dwconv.weight"))
    hparams = dict(
        num_mels=state["backbone.embed.weight"].shape[1],
        dim=state["backbone.embed.weight"].shape[0],
        intermediate_dim=state["backbone.convnext.0.pwconv1.weight"].shape[0],
        num_layers=layers,
        n_fft=state["head.out.weight"].shape[0] - 2,
        hop_length=int(mel["hop_length"]),
        padding=padding,
    )
    if hparams["num_mels"] != int(mel["n_mels"]):
        raise ValueError(f"{path} takes {hparams['num_mels']} mel bins, not {mel['n_mels']}.")
    if hparams["n_fft"] != int(mel["n_fft"]):
        raise ValueError(f"{path} synthesizes with n_fft {hparams['n_fft']}, but its mel uses {mel['n_fft']}.")
    generator = Vocos(**hparams)
    generator.load_state_dict(state, strict=True)
    return generator, mel, native


class VocosInference:
    def __init__(self, path, checkpoint, device):
        from rectified_flow.mel import LogMel

        self.device = torch.device(device)
        generator, self.data, native = build_vocos(path, checkpoint)
        self.generator = generator.to(self.device).float().eval().requires_grad_(False)
        self.mel = (VocosMel(**native) if native else LogMel.from_config(self.data)).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, waveform, pitch=0, f0_method="rmvpe", seed=None):
        if float(pitch):
            raise ValueError("Vocos has no pitch input. Set Pitch to 0.")
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError("Select non-empty audio with finite samples.")
        hop, length = int(self.data["hop_length"]), len(waveform)
        padded_length = max(math.ceil(length / hop), math.ceil(self.data["n_fft"] / hop)) * hop
        padded = np.pad(np.clip(waveform, -1.0, 1.0), (0, padded_length - length))
        source = torch.from_numpy(padded).to(self.device)[None]
        audio = self.generator(self.mel(source))[0, 0]
        audio = F.pad(audio, (0, max(0, length - audio.shape[-1])))[:length]
        if not torch.isfinite(audio).all():
            raise FloatingPointError("Non-finite Vocos audio output.")
        return audio.float().cpu().numpy()
