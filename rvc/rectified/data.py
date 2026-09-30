import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from rvc.rectified.energy import frame_energy
from rvc.rectified.aperiodicity import aperiodicity
from rvc.rectified.mel import LogMel

FEATURE_RATE = 100
SMOOTH_SECONDS = 0.06
CONTENT_INTERPOLATIONS = ("nearest", "linear")

def upsample_content(features: torch.Tensor, mode: str = "nearest") -> torch.Tensor:
    if mode not in CONTENT_INTERPOLATIONS:
        raise ValueError(f"content_interpolation must be one of {CONTENT_INTERPOLATIONS}, not {mode!r}.")
    batched = features.dim() == 3
    x = (features if batched else features.unsqueeze(0)).transpose(1, 2)
    if mode == "linear":
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="linear", align_corners=False)
    else:
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode="nearest")
    x = x.transpose(1, 2)
    return x if batched else x[0]

def smooth_curve(curve: torch.Tensor) -> torch.Tensor:
    width = int(round(SMOOTH_SECONDS * FEATURE_RATE))
    kernel = torch.sin(torch.linspace(0, 1, width + 2, device=curve.device)[1:-1] * math.pi)
    kernel = (kernel / kernel.sum()).view(1, 1, -1)
    padded = F.pad(curve.unsqueeze(1), ((width - 1) // 2, width // 2), mode="replicate")
    return F.conv1d(padded, kernel.to(curve.dtype)).squeeze(1)

def split_holdout(entries, count: int, seed: int = 1234):
    candidates = [i for i, entry in enumerate(entries) if "mute" not in os.path.basename(entry[0])]
    held = set(random.Random(seed).sample(candidates, min(count, len(candidates) // 10)))
    train = [entry for i, entry in enumerate(entries) if i not in held]
    return train, [entries[i] for i in sorted(held)]

def mel_frames(feature_frames: int, sample_rate: int, hop: int) -> int:
    return int(feature_frames * sample_rate / (hop * FEATURE_RATE))

def _positions(length: int, frames: int, sample_rate: int, hop: int, device):
    position = torch.arange(frames, device=device, dtype=torch.float64)
    position = (position * hop * FEATURE_RATE / sample_rate).clamp(max=length - 1)
    left = position.floor().long()
    right = (left + 1).clamp(max=length - 1)
    return left, right, (position - left).float()

def to_mel_rate(features: torch.Tensor, frames: int, sample_rate: int, hop: int) -> torch.Tensor:
    left, right, weight = _positions(features.shape[-2], frames, sample_rate, hop, features.device)
    weight = weight.unsqueeze(-1)
    return features[..., left, :] * (1 - weight) + features[..., right, :] * weight

def f0_to_mel_rate(f0: torch.Tensor, frames: int, sample_rate: int, hop: int) -> torch.Tensor:
    left, right, weight = _positions(f0.shape[-1], frames, sample_rate, hop, f0.device)
    a, b = f0[..., left], f0[..., right]
    nearest = torch.where(weight < 0.5, a, b)
    return torch.where((a > 0) & (b > 0), a * (1 - weight) + b * weight, nearest)

class RectifiedDataset(Dataset):
    def __init__(self, entries, config: dict, segment_frames: int, augment: bool = True):
        self.entries = entries
        self.data = config["data"]
        self.segment_frames = int(segment_frames)
        self.hop = int(self.data["hop_length"])
        self.sample_rate = int(self.data["sample_rate"])
        self.mel = LogMel.from_config(self.data)
        self.content_channels = int(config["flow"]["model"]["content_channels"])
        self.key_shift_range = float(config["flow"].get("key_shift_range", 0.0))
        self.key_shift_prob = float(config["flow"].get("key_shift_prob", 0.0))
        self.stretch_range = tuple(config["flow"].get("time_stretch_range", (1.0, 1.0)))
        self.stretch_prob = float(config["flow"].get("time_stretch_prob", 0.0))
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def _audio(self, path):
        data, sample_rate = sf.read(path, dtype="float32")
        audio = torch.from_numpy(data)
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"{path} is {sample_rate} Hz; the rectified models train at "
                f"{self.sample_rate} Hz. Preprocess the dataset at that rate."
            )
        return audio.mean(-1) if audio.dim() == 2 else audio

    def _energy(self, audio, frames, hop):
        feature_frames = audio.shape[-1] // (self.sample_rate // FEATURE_RATE)
        energy = smooth_curve(frame_energy(audio, self.sample_rate, feature_frames))
        return to_mel_rate(energy.unsqueeze(-1), frames, self.sample_rate, hop)[..., 0]

    def _breathiness(self, audio, f0, frames, hop):
        feature_frames = audio.shape[-1] // (self.sample_rate // FEATURE_RATE)
        share = smooth_curve(aperiodicity(audio, self.sample_rate, f0, feature_frames))
        return to_mel_rate(share.unsqueeze(-1), frames, self.sample_rate, hop)[..., 0]

    def __getitem__(self, index):
        wav_path, content_path, _, f0_path, sid = self.entries[index]
        audio = self._audio(wav_path)
        source_f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
        content = upsample_content(
            torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32)),
            self.data["content_interpolation"],
        )
        if content.ndim != 2 or content.shape[1] != self.content_channels:
            raise ValueError(f"Expected {self.content_channels}-wide content features: {content_path}")
        if source_f0.ndim != 1 or source_f0.size(0) == 0 or content.size(0) == 0:
            raise ValueError(f"Empty or invalid features: {wav_path}")
        return self._flow_item(audio, source_f0, content, int(sid))

    def _flow_item(self, audio, source_f0, content, sid):
        key_shift, hop = 0.0, self.hop
        if self.augment and self.key_shift_range > 0 and random.random() < self.key_shift_prob:
            key_shift = random.uniform(-self.key_shift_range, self.key_shift_range)
        if self.augment and self.stretch_prob > 0 and random.random() < self.stretch_prob:
            low, high = self.stretch_range
            hop = int(round(self.hop * low * (high / low) ** random.random()))
        speed = hop / self.hop

        frames = min(
            audio.shape[0] // hop,
            mel_frames(source_f0.shape[0], self.sample_rate, hop),
            mel_frames(content.shape[0], self.sample_rate, hop),
        )
        if frames < 4:
            raise ValueError("Training clips must contain at least four mel frames.")
        audio = audio[: frames * hop]
        content = to_mel_rate(content, frames, self.sample_rate, hop)
        f0 = f0_to_mel_rate(source_f0, frames, self.sample_rate, hop) * 2.0 ** (key_shift / 12.0)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0), key_shift, hop)[0, :, :frames]
        energy = self._energy(audio.unsqueeze(0), frames, hop)[0]
        breathiness = self._breathiness(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, hop)[0]

        length = min(frames, self.segment_frames)
        start = random.randint(0, frames - length) if self.augment else 0
        stop = start + length
        return (
            mel[:, start:stop], content[start:stop], f0[start:stop], energy[start:stop],
            breathiness[start:stop], key_shift, speed, sid,
        )

    def _reference_item(self, audio, content, f0, sid, path, max_frames=None):
        frames = min(
            audio.shape[0] // self.hop,
            mel_frames(min(f0.shape[0], content.shape[0]), self.sample_rate, self.hop),
        )
        if max_frames is not None:
            frames = min(frames, max_frames)
        audio = audio[: frames * self.hop]
        content = to_mel_rate(content, frames, self.sample_rate, self.hop)
        breathiness = self._breathiness(audio.unsqueeze(0), f0.unsqueeze(0), frames, self.hop)
        f0 = f0_to_mel_rate(f0, frames, self.sample_rate, self.hop)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0))[:, :, :frames]
        energy = self._energy(audio.unsqueeze(0), frames, self.hop)
        return (
            mel, content.unsqueeze(0), f0.unsqueeze(0), energy, breathiness,
            audio.unsqueeze(0), int(sid), path,
        )

    def reference(self, max_seconds: float = 10.0):
        ordered = sorted(range(len(self.entries)), key=lambda i: self.entries[i][0])
        for index in ordered:
            wav_path, content_path, _, f0_path, sid = self.entries[index]
            if "mute" in os.path.basename(wav_path):
                continue
            audio = self._audio(wav_path)
            if audio.shape[0] < 2 * self.sample_rate:
                continue
            f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
            content = upsample_content(
                torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32)),
                self.data["content_interpolation"],
            )
            max_frames = int(max_seconds * self.sample_rate) // self.hop
            return self._reference_item(audio, content, f0, sid, wav_path, max_frames)
        return None


def collate_flow(batch, frames=None):
    frames = frames or max(item[0].shape[-1] for item in batch)
    size = len(batch)
    mel = torch.zeros(size, batch[0][0].shape[0], frames)
    content = torch.zeros(size, frames, batch[0][1].shape[-1])
    f0 = torch.zeros(size, frames)
    energy = torch.full((size, frames), -1.0)
    breathiness = torch.ones(size, frames)
    key_shift = torch.zeros(size)
    speed = torch.ones(size)
    mask = torch.zeros(size, 1, frames)
    speaker = torch.zeros(size, dtype=torch.long)
    for i, (m, c, p, e, b, k, v, s) in enumerate(batch):
        n = m.shape[-1]
        mel[i, :, :n] = m
        content[i, :n] = c
        f0[i, :n] = p
        energy[i, :n] = e
        breathiness[i, :n] = b
        key_shift[i] = k
        speed[i] = v
        mask[i, :, :n] = 1.0
        speaker[i] = s
    return mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask

def read_filelist(path, root):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) != 5 or int(fields[4]) < 0:
            raise ValueError(f"Invalid filelist row {number}: expected audio|content|f0|f0_hz|speaker_id")
        for column in range(4):
            fields[column] = os.path.normpath(os.path.join(root, fields[column]))
            if not os.path.isfile(fields[column]):
                raise FileNotFoundError(fields[column])
        rows.append(fields)
    if not rows:
        raise ValueError("The training filelist is empty.")
    return rows
