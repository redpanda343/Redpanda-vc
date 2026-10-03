import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

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

def split_holdout(entries, count: int, seed: int = 1234, stratified: bool = False):
    candidates = [i for i, entry in enumerate(entries) if "mute" not in os.path.basename(entry[0])]
    if stratified:
        rng = random.Random(seed)
        groups = {}
        for index in candidates:
            groups.setdefault(int(entries[index][4]), []).append(index)
        pools = []
        for sid in sorted(groups):
            group = sorted(groups[sid], key=lambda i: entries[i][0])
            rng.shuffle(group)
            if len(group) > 1:
                pools.append(group[1:])
        target = min(sum(map(len, pools)), max(count, len(pools))) if count > 0 else 0
        held = set()
        while len(held) < target:
            rng.shuffle(pools)
            for pool in pools:
                if pool and len(held) < target:
                    held.add(pool.pop())
        return ([entry for i, entry in enumerate(entries) if i not in held],
                [entries[i] for i in sorted(held)])
    held = set(random.Random(seed).sample(candidates, min(count, len(candidates) // 10)))
    train = [entry for i, entry in enumerate(entries) if i not in held]
    return train, [entries[i] for i in sorted(held)]


class SpeakerBalancedSampler(Sampler):
    def __init__(self, entries, seed=1234, rank=0, world=1):
        if not 0 <= rank < world:
            raise ValueError('Invalid sampler rank.')
        self.groups = {}
        for index, entry in enumerate(entries):
            self.groups.setdefault(int(entry[4]), []).append(index)
        if not self.groups:
            raise ValueError('Speaker sampling requires training clips.')
        self.seed, self.rank, self.world = seed, rank, world
        self.epoch = 0
        self.samples = len(entries) // world

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.samples

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        speakers = sorted(self.groups)
        pools = {sid: [] for sid in speakers}
        indices = []
        while len(indices) < self.samples * self.world:
            rng.shuffle(speakers)
            for sid in speakers:
                if len(indices) == self.samples * self.world:
                    break
                if not pools[sid]:
                    pools[sid] = list(self.groups[sid])
                    rng.shuffle(pools[sid])
                indices.append(pools[sid].pop())
        return iter(indices[self.rank::self.world])


def speaker_inventory(entries):
    assignments = {}
    counts = {}
    for entry in entries:
        audio, sid = os.path.normcase(os.path.abspath(entry[0])), int(entry[4])
        if audio in assignments:
            raise ValueError(f'Duplicate audio or conflicting speaker labels: {entry[0]}')
        assignments[audio] = sid
        counts[sid] = counts.get(sid, 0) + 1
    if sorted(counts) != list(range(len(counts))):
        raise ValueError('Multispeaker training requires contiguous speaker IDs starting at 0.')
    return counts

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


def variance_curves(audio, f0, frames, sample_rate, hop):
    from rvc.rectified.variance import voicing_tension

    count = max(1, audio.shape[-1] // (sample_rate // FEATURE_RATE))
    curves = voicing_tension(audio, sample_rate, f0, count)
    return tuple(to_mel_rate(smooth_curve(curve).unsqueeze(-1), frames, sample_rate, hop)[..., 0]
                 for curve in curves)


def unpack_flow(batch, device, non_blocking=False):
    values = tuple(item.to(device, non_blocking=non_blocking) for item in batch)
    if len(values) == 9:
        return (*values, None, None)
    if len(values) != 11:
        raise ValueError("Invalid rectified-flow batch.")
    return values

class RectifiedDataset(Dataset):
    def __init__(self, entries, config: dict, segment_frames: int, augment: bool = True):
        self.entries = entries
        self.data = config["data"]
        self.segment_frames = int(segment_frames)
        self.hop = int(self.data["hop_length"])
        self.sample_rate = int(self.data["sample_rate"])
        self.mel = LogMel.from_config(self.data)
        self.content_channels = int(config["flow"]["model"]["content_channels"])
        self.strict_features = config['flow']['model'].get('conditioning_version', 1) in (2, 3)
        self.key_shift_range = float(config["flow"].get("key_shift_range", 0.0))
        self.key_shift_prob = float(config["flow"].get("key_shift_prob", 0.0))
        self.stretch_range = tuple(config["flow"].get("time_stretch_range", (1.0, 1.0)))
        self.stretch_prob = float(config["flow"].get("time_stretch_prob", 0.0))
        self.augment = augment
        self.use_variances = any(config["flow"]["model"].get(name, False) for name in ("voicing", "tension"))

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
        if self.strict_features:
            if not all(torch.isfinite(value).all() for value in (audio, source_f0, content)) or (source_f0 < 0).any():
                raise ValueError(f'Invalid audio, content or pitch values: {wav_path}')
            seconds = audio.numel() / self.sample_rate
            if any(abs(value.shape[0] / FEATURE_RATE - seconds) > 0.25 for value in (source_f0, content)):
                raise ValueError(f'Content or pitch duration does not match audio; re-extract features: {wav_path}')
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
        item = (
            mel[:, start:stop], content[start:stop], f0[start:stop], energy[start:stop],
            breathiness[start:stop], key_shift, speed, sid,
        )
        if self.use_variances:
            curves = variance_curves(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, self.sample_rate, hop)
            item += tuple(curve[0, start:stop] for curve in curves)
        return item

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
        source_f0 = f0
        f0 = f0_to_mel_rate(f0, frames, self.sample_rate, self.hop)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0))[:, :, :frames]
        energy = self._energy(audio.unsqueeze(0), frames, self.hop)
        item = (
            mel, content.unsqueeze(0), f0.unsqueeze(0), energy, breathiness,
            audio.unsqueeze(0), int(sid), path,
        )
        if self.use_variances:
            item += variance_curves(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, self.sample_rate, self.hop)
        return item

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
    extended = len(batch[0]) == 10
    voicing, tension = torch.zeros(size, frames), torch.zeros(size, frames)
    for i, item in enumerate(batch):
        if len(item) != (10 if extended else 8):
            raise ValueError("Mixed conditioning formats in rectified-flow batch.")
        m, c, p, e, b, k, v, s = item[:8]
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
        if extended:
            voicing[i, :n], tension[i, :n] = item[8:]
    result = (mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask)
    return (*result, voicing, tension) if extended else result

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
