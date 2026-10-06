import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, Sampler

from rvc.rectified.energy import frame_energy
from rvc.rectified.aperiodicity import aperiodicity
from rvc.rectified.mel import LogMel
from rvc.rectified.config import resolve_config
from rvc.rectified.flow_model import HarmonicPrior
from rvc.rectified.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from rvc.rectified.pitch import parselmouth_f0, uses_parselmouth

FEATURE_RATE = 100
SMOOTH_SECONDS = 0.06
CONTENT_INTERPOLATIONS = ("nearest", "linear")
ORIGINAL_CACHE_VERSION = 4

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


def clip_samples(entries, workers: int = 8):
    """Sample count of every training clip, read from the audio headers only."""
    def count(entry):
        if str(entry[1]).endswith('.flow.npz'):
            with np.load(entry[1], allow_pickle=False) as features:
                return int(features['frames']) * int(features['hop'])
        return sf.info(entry[0]).frames
    with ThreadPoolExecutor(workers) as pool:
        return np.fromiter(pool.map(count, entries), dtype=np.int64, count=len(entries))


class FlowBatchSampler(Sampler):
    def __init__(self, dataset, max_frames, max_items, seed=1234, rank=0, world=1,
                 shuffle=True, required_batch_count_multiple=1, disallow_empty_batch=True,
                 pad_batch_assignment=True, sort_by_len=True, frame_count_grid=6):
        if int(max_frames) < 1 or int(max_items) < 1:
            raise ValueError('Batch limits must be positive.')
        if not 0 <= rank < world:
            raise ValueError('Invalid sampler rank.')
        self.dataset = dataset
        self.max_frames, self.max_items = int(max_frames), int(max_items)
        self.seed, self.rank, self.world = seed, rank, world
        self.shuffle = shuffle
        self.required_batch_count_multiple = int(required_batch_count_multiple)
        self.disallow_empty_batch = disallow_empty_batch
        self.pad_batch_assignment = pad_batch_assignment
        self.sort_by_len = sort_by_len
        self.frame_count_grid = int(frame_count_grid)
        if self.required_batch_count_multiple < 1 or self.frame_count_grid < 1:
            raise ValueError('Sampler batch multiple and frame grid must be positive.')
        self.samples = clip_samples(dataset.entries)
        self.frames = np.minimum(np.maximum(1, self.samples // dataset.hop), self.max_frames)
        self.measured_max_frames = None
        self.epoch = 0
        self._formed = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _form(self):
        if self._formed is not None and self._formed[0] == self.epoch:
            return self._formed
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.dataset)) if self.shuffle else np.arange(len(self.dataset))
        if self.shuffle and self.sort_by_len:
            grid = self.frame_count_grid
            sizes = (np.round(self.frames[order] / grid) * grid).clip(grid, None)
            order = order[np.argsort(-sizes, kind='mergesort')]
        maximum = self.measured_max_frames if self.measured_max_frames is not None else self.max_frames
        batches = []
        batch, longest = [], 0
        for index in order.tolist():
            frames = int(self.frames[index])
            if batch and (len(batch) == self.max_items
                          or (len(batch) + 1) * max(longest, frames) > maximum):
                batches.append(batch)
                batch, longest = [], 0
            batch.append(index)
            longest = max(longest, frames)
        if batch:
            batches.append(batch)
        if self.shuffle and self.measured_max_frames is None:
            costs = [len(batch) * max(self.frames[index] for index in batch) for batch in batches]
            self.measured_max_frames = max(costs) if costs else self.max_frames
            batches = [batches[index] for index in sorted(range(len(batches)), key=lambda index: -costs[index])]
        if len(batches) < self.world and self.disallow_empty_batch:
            raise ValueError('The training split is too small for the selected GPUs at these batch limits.')
        floored = len(batches) // self.world * self.world
        leftovers = ((rng.permutation(len(batches) - floored) + floored).tolist()
                     if self.shuffle else list(range(floored, len(batches))))
        assignment = np.arange(floored).reshape(-1, self.world).transpose()
        assignment = (rng.permuted(assignment, axis=0)[self.rank].tolist()
                      if self.shuffle else assignment[self.rank].tolist())
        floored_batch_count = len(assignment)
        if self.rank < len(leftovers):
            assignment.append(leftovers[self.rank])
            floored_batch_count += 1
        elif leftovers and self.pad_batch_assignment:
            if not assignment:
                raise ValueError('Cannot pad an empty batch assignment.')
            assignment.append(assignment[self.epoch % len(assignment)])
        count = len(assignment)
        multiple = self.required_batch_count_multiple
        if count and count % multiple:
            for index in range(multiple - count % multiple):
                assignment.append(assignment[(index + self.epoch * multiple) % floored_batch_count])
        batches = [batches[index] for index in assignment]
        if not batches:
            batches = [[]]
        costs = [len(batch) * max((self.frames[index] for index in batch), default=0) for batch in batches]
        self._formed = (self.epoch, [[(index, self.dataset.hop) for index in batch] for batch in batches], costs)
        return self._formed

    def __len__(self):
        return len(self._form()[1])

    def __iter__(self):
        return iter(self._form()[1])


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

def f0_to_mel_rate(f0: torch.Tensor, frames: int, sample_rate: int, hop: int, continuous=False) -> torch.Tensor:
    if continuous:
        from rvc.rectified.content_encoder import interpolate_pitch

        batched = f0.ndim == 2
        values = f0 if batched else f0.unsqueeze(0)
        values = interpolate_pitch(values, values.new_ones(values.shape[0], 1, values.shape[-1]))
        f0 = values if batched else values[0]
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
    if len(values) == 10:
        return (*values, None, None)
    if len(values) != 12:
        raise ValueError("Invalid rectified-flow batch.")
    return values

class RectifiedDataset(Dataset):
    def __init__(self, entries, config: dict, max_frames: int, augment: bool = True, cache_path=None):
        config = resolve_config(config)
        self.entries = entries
        self.data = config["data"]
        self.max_frames = int(max_frames)
        self.hop = int(self.data["hop_length"])
        self.sample_rate = int(self.data["sample_rate"])
        self.mel = LogMel.from_config(self.data)
        self.content_channels = int(config["flow"]["model"]["content_channels"])
        self.strict_features = config['flow']['model'].get('conditioning_version', 1) in (2, 3, 4, 5)
        self.augment = augment
        self.use_variances = any(config["flow"]["model"].get(name, False) for name in ("voicing", "tension"))
        self.continuous_f0 = config['flow']['model'].get('use_continuous_f0',
                                                       config['flow']['model'].get('conditioning_version') == 5)
        self.parselmouth = uses_parselmouth(self.data)
        self.use_harmonics = bool(config["flow"]["model"].get("harmonic_prior", False))
        self.harmonic_prior = (
            HarmonicPrior(
                sample_rate=self.sample_rate, n_fft=int(self.data["n_fft"]),
                n_mels=int(self.data["n_mels"]), fmin=float(self.data["mel_fmin"]),
                fmax=float(self.data["mel_fmax"]),
            )
            if self.use_harmonics else None
        )
        self._cache_recipe = json.dumps(
            dict(
                version=ORIGINAL_CACHE_VERSION,
                data=self.data,
                content_channels=self.content_channels,
                use_variances=self.use_variances,
                use_harmonics=self.use_harmonics,
                continuous_f0=self.continuous_f0,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        self._cache_recipe_hash = hashlib.sha256(self._cache_recipe.encode()).hexdigest()
        self._original_keys = [self._cache_key(entry) for entry in self.entries]
        self._indexed_cache = IndexedDataset(cache_path, self._cache_recipe_hash) if cache_path is not None else None

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

    def _mel_f0(self, audio, source_f0, frames, hop):
        if self.parselmouth:
            return torch.from_numpy(parselmouth_f0(audio.numpy(), self.sample_rate, hop, frames))
        return f0_to_mel_rate(source_f0, frames, self.sample_rate, hop, self.continuous_f0)

    @torch.no_grad()
    def _harmonics(self, f0):
        if self.harmonic_prior is None:
            return torch.empty((0, f0.shape[-1]), dtype=torch.float32)
        return self.harmonic_prior(f0.unsqueeze(0))[0]

    def _cache_key(self, entry):
        if str(entry[1]).endswith('.flow.npz'):
            return None
        sources = []
        for path in (entry[0], entry[1], entry[3]):
            value = Path(path)
            stat = value.stat()
            sources.append((str(value.resolve()), stat.st_size, stat.st_mtime_ns))
        return hashlib.sha256(
            (self._cache_recipe + json.dumps(sources, separators=(",", ":"))).encode()
        ).hexdigest()

    def _build_original_values(self, index, audio=None, source_f0=None, content=None):
        if self._original_keys[index] is None:
            return None
        wav_path, content_path, _, f0_path, _ = self.entries[index]
        audio = self._audio(wav_path) if audio is None else audio
        source_f0 = (
            torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
            if source_f0 is None else source_f0
        )
        content = (
            upsample_content(
                torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32)),
                self.data["content_interpolation"],
            )
            if content is None else content
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

        frames = min(
            audio.shape[0] // self.hop,
            mel_frames(source_f0.shape[0], self.sample_rate, self.hop),
            mel_frames(content.shape[0], self.sample_rate, self.hop),
        )
        if frames < 4:
            raise ValueError("Training clips must contain at least four mel frames.")
        f0 = self._mel_f0(audio, source_f0, frames, self.hop)
        audio = audio[: frames * self.hop]
        content = to_mel_rate(content, frames, self.sample_rate, self.hop)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0), 0.0, self.hop)[0, :, :frames]
        energy = self._energy(audio.unsqueeze(0), frames, self.hop)[0]
        breathiness = self._breathiness(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, self.hop)[0]

        values = dict(
            mel=mel.numpy(),
            content=content.numpy(),
            f0=f0.numpy(),
            energy=energy.numpy(),
            breathiness=breathiness.numpy(),
            harmonic_prior=self._harmonics(f0).numpy(),
            key_shift=np.float32(0.0),
            speed=np.float32(1.0),
            frames=np.int64(frames),
            hop=np.int64(self.hop),
        )
        if self.use_variances:
            curves = variance_curves(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, self.sample_rate, self.hop)
            values.update(voicing=curves[0][0].numpy(), tension=curves[1][0].numpy())
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError(f'Invalid cached training features: {wav_path}')
        return values

    def __getitem__(self, index):
        index, hop = index if isinstance(index, tuple) else (index, None)
        _, content_path, _, _, sid = self.entries[index]
        if str(content_path).endswith('.flow.npz'):
            return self._npz_item(content_path, int(sid))
        if self._indexed_cache is None:
            values = self._build_original_values(index)
        else:
            key = self._original_keys[index]
            try:
                values = self._indexed_cache[key]
            except (FileNotFoundError, KeyError, OSError, ValueError) as error:
                raise RuntimeError(
                    f'Indexed training feature cache is missing or stale for {self.entries[index][0]}; rebuild it before training.'
                ) from error
        return self._cached_values(values, int(sid), str(self._indexed_cache.path) if self._indexed_cache is not None else self.entries[index][0])

    def _flow_item(self, audio, source_f0, content, sid, hop=None):
        key_shift = 0.0
        hop = self.hop
        speed = hop / self.hop

        frames = min(
            audio.shape[0] // hop,
            mel_frames(source_f0.shape[0], self.sample_rate, hop),
            mel_frames(content.shape[0], self.sample_rate, hop),
        )
        if frames < 4:
            raise ValueError("Training clips must contain at least four mel frames.")
        f0 = self._mel_f0(audio, source_f0, frames, hop) * 2.0 ** (key_shift / 12.0)
        audio = audio[: frames * hop]
        content = to_mel_rate(content, frames, self.sample_rate, hop)
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0), key_shift, hop)[0, :, :frames]
        energy = self._energy(audio.unsqueeze(0), frames, hop)[0]
        breathiness = self._breathiness(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, hop)[0]

        length = min(frames, self.max_frames)
        start = random.randint(0, frames - length) if self.augment else 0
        stop = start + length
        item = (
            mel[:, start:stop], content[start:stop], f0[start:stop], energy[start:stop],
            breathiness[start:stop], key_shift, speed, sid, self._harmonics(f0)[..., start:stop],
        )
        if self.use_variances:
            curves = variance_curves(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, self.sample_rate, hop)
            item += tuple(curve[0, start:stop] for curve in curves)
        return item

    def _npz_item(self, path, sid):
        with np.load(path, allow_pickle=False) as values:
            loaded = {name: values[name] for name in values.files}
        return self._cached_values(loaded, sid, path)

    def _cached_values(self, values, sid, source):
        frames = int(values['frames'])
        length = min(frames, self.max_frames)
        start = random.randint(0, frames - length) if self.augment else 0
        stop = start + length
        mel = torch.from_numpy(np.asarray(values['mel'])[:, start:stop].copy())
        curves = tuple(torch.from_numpy(np.asarray(values[name])[start:stop].copy())
                       for name in ('content', 'f0', 'energy', 'breathiness'))
        if mel.shape != (self.data['n_mels'], length) or curves[0].shape != (length, self.content_channels):
            raise ValueError(f'Invalid augmented feature dimensions: {source}')
        if any(value.shape[0] != length for value in curves[1:]):
            raise ValueError(f'Invalid augmented curve lengths: {source}')
        if not all(torch.isfinite(value).all() for value in (mel, *curves)) or (curves[1] < 0).any():
            raise ValueError(f'Invalid augmented feature values: {source}')
        if self.use_harmonics:
            if 'harmonic_prior' not in values:
                raise ValueError(f'Missing cached harmonic prior; rebuild training features: {source}')
            harmonic = torch.from_numpy(np.asarray(values['harmonic_prior'])[:, start:stop].copy())
            if harmonic.shape != (self.data['n_mels'], length) or not torch.isfinite(harmonic).all():
                raise ValueError(f'Invalid cached harmonic-prior dimensions or values: {source}')
        else:
            harmonic = torch.empty((0, length), dtype=mel.dtype)
        item = (mel, *curves, float(values['key_shift']), float(values['speed']), sid, harmonic)
        if self.use_variances:
            item += tuple(torch.from_numpy(np.asarray(values[name])[start:stop].copy()) for name in ('voicing', 'tension'))
        return item

    def _reference_item(self, audio, content, f0, sid, path, max_frames=None):
        frames = min(
            audio.shape[0] // self.hop,
            mel_frames(min(f0.shape[0], content.shape[0]), self.sample_rate, self.hop),
        )
        if max_frames is not None:
            frames = min(frames, max_frames)
        source_f0 = f0
        f0 = self._mel_f0(audio, source_f0, frames, self.hop)
        audio = audio[: frames * self.hop]
        content = to_mel_rate(content, frames, self.sample_rate, self.hop)
        breathiness = self._breathiness(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, self.hop)
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
        return next(iter(self.references(1, max_seconds)), None)

    def references(self, count: int, max_seconds: float = 10.0):
        result = []
        if count <= 0:
            return result
        ordered = sorted(range(len(self.entries)), key=lambda i: self.entries[i][0])
        for index in ordered:
            wav_path, content_path, _, f0_path, sid = self.entries[index]
            if "mute" in os.path.basename(wav_path) or str(content_path).endswith('.flow.npz'):
                continue
            audio = self._audio(wav_path)
            if audio.shape[0] < self.hop:
                continue
            f0 = torch.from_numpy(np.load(f0_path, allow_pickle=False).astype(np.float32))
            content = upsample_content(
                torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32)),
                self.data["content_interpolation"],
            )
            max_frames = int(max_seconds * self.sample_rate) // self.hop
            result.append(self._reference_item(audio, content, f0, sid, wav_path, max_frames))
            if len(result) >= count:
                break
        return result


def prepare_training_cache(entries, config, cache_path, workers=4):
    if not entries:
        return
    dataset = RectifiedDataset(entries, config, 2 ** 31, augment=False)
    keys = dataset._original_keys
    cache_path = Path(cache_path)
    if IndexedDataset.matches(cache_path, dataset._cache_recipe_hash, keys):
        print(f'Training feature index: {len(entries):,}/{len(entries):,} ready.', flush=True)
        return
    workers = max(1, min(int(workers), os.cpu_count() or 1, len(entries)))
    print(
        f'Training feature index: building {len(entries):,} clip(s) with {workers} worker(s)...',
        flush=True,
    )
    temporary = Path(str(cache_path) + f'.{os.getpid()}.tmp')
    if temporary.exists():
        temporary.unlink()
    builder = IndexedDatasetBuilder(temporary, dataset._cache_recipe_hash, len(entries))
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, previous_threads // workers))
    completed = 0
    last_report = time.monotonic()
    try:
        if workers == 1:
            iterator = map(dataset._build_original_values, range(len(entries)))
            pool = None
        else:
            pool = ThreadPoolExecutor(max_workers=workers)
            iterator = pool.map(dataset._build_original_values, range(len(entries)))
        try:
            for index, values in enumerate(iterator):
                builder.add_item(index, keys[index], values)
                completed += 1
                now = time.monotonic()
                if completed == len(entries) or now - last_report >= 5:
                    print(f'Training feature index: {completed:,}/{len(entries):,}', flush=True)
                    last_report = now
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
        builder.finalize()
        os.replace(temporary, cache_path)
    except Exception:
        builder.abort()
        raise
    finally:
        torch.set_num_threads(previous_threads)

def collate_flow(batch, frames=None):
    frames = frames or max(item[0].shape[-1] for item in batch)
    size = len(batch)
    mel = torch.full((size, batch[0][0].shape[0], frames), math.log(1e-5))
    content = torch.zeros(size, frames, batch[0][1].shape[-1])
    f0 = torch.zeros(size, frames)
    energy = torch.full((size, frames), -1.0)
    breathiness = torch.ones(size, frames)
    key_shift = torch.zeros(size)
    speed = torch.ones(size)
    mask = torch.zeros(size, 1, frames)
    speaker = torch.zeros(size, dtype=torch.long)
    harmonic_channels = batch[0][8].shape[0]
    harmonic_prior = torch.zeros(size, harmonic_channels, frames)
    extended = len(batch[0]) == 11
    voicing, tension = torch.zeros(size, frames), torch.zeros(size, frames)
    for i, item in enumerate(batch):
        if len(item) != (11 if extended else 9):
            raise ValueError("Mixed conditioning formats in rectified-flow batch.")
        m, c, p, e, b, k, v, s, h = item[:9]
        n = m.shape[-1]
        if h.shape != (harmonic_channels, n):
            raise ValueError("Mixed harmonic-prior formats in rectified-flow batch.")
        mel[i, :, :n] = m
        content[i, :n] = c
        f0[i, :n] = p
        energy[i, :n] = e
        breathiness[i, :n] = b
        key_shift[i] = k
        speed[i] = v
        mask[i, :, :n] = 1.0
        speaker[i] = s
        if harmonic_channels:
            harmonic_prior[i, :, :n] = h
        if extended:
            voicing[i, :n], tension[i, :n] = item[9:]
    result = (mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask, harmonic_prior)
    return (*result, voicing, tension) if extended else result

def read_filelist(path, root, originals_only=False):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.strip().split("|")
        if len(fields) != 5 or int(fields[4]) < 0:
            raise ValueError(f"Invalid filelist row {number}: expected audio|content|f0|f0_hz|speaker_id")
        if originals_only and fields[1].endswith('.flow.npz'):
            continue
        for column in range(4):
            fields[column] = os.path.normpath(os.path.join(root, fields[column]))
            if not os.path.isfile(fields[column]):
                raise FileNotFoundError(fields[column])
        rows.append(fields)
    if not rows:
        raise ValueError("The training filelist is empty.")
    return rows
