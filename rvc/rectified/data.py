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
from torch.utils.data import Dataset, Sampler

from rvc.rectified.config import resolve_config
from rvc.rectified.indexed_dataset import IndexedDataset, IndexedDatasetBuilder
from rvc.rectified.mel import LogMel
from rvc.rectified.pitch import parselmouth_f0

CONTENT_RATE = 50
CACHE_VERSION = 5


def split_holdout(entries, count: int, seed: int = 1234):
    candidates = [i for i, entry in enumerate(entries) if "mute" not in os.path.basename(entry[0])]
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
    target = min(sum(map(len, pools)), max(count, len(pools)), len(candidates) // 10) if count > 0 else 0
    held = set()
    while len(held) < target:
        rng.shuffle(pools)
        for pool in pools:
            if pool and len(held) < target:
                held.add(pool.pop())
    return ([entry for i, entry in enumerate(entries) if i not in held],
            [entries[i] for i in sorted(held)])


def clip_samples(entries, workers: int = 8):
    """Sample count of every training clip, read from the audio headers only."""
    def count(entry):
        if is_augmented(entry):
            with np.load(entry[1], allow_pickle=False) as features:
                return int(features['frames']) * int(features['hop'])
        return sf.info(entry[0]).frames
    with ThreadPoolExecutor(workers) as pool:
        return np.fromiter(pool.map(count, entries), dtype=np.int64, count=len(entries))


def is_augmented(entry):
    return str(entry[1]).endswith('.flow.npz')


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
        self._formed = (self.epoch, batches, costs)
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


def mel_frames(content_frames: int, sample_rate: int, hop: int) -> int:
    return int(content_frames * sample_rate / (hop * CONTENT_RATE))


def unpack_flow(batch, device, non_blocking=False):
    return tuple(item.to(device, non_blocking=non_blocking) for item in batch)


class RectifiedDataset(Dataset):
    def __init__(self, entries, config: dict, max_frames: int, cache_path=None):
        config = resolve_config(config)
        self.entries = entries
        self.data = config["data"]
        self.max_frames = int(max_frames)
        self.hop = int(self.data["hop_length"])
        self.sample_rate = int(self.data["sample_rate"])
        self.mel = LogMel.from_config(self.data)
        self.content_channels = int(config["flow"]["model"]["content_channels"])
        self._cache_recipe = json.dumps(
            dict(version=CACHE_VERSION, data=self.data, content_channels=self.content_channels),
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

    def _content(self, path):
        content = torch.from_numpy(np.load(path, allow_pickle=False).astype(np.float32))
        if content.ndim != 2 or content.shape[1] != self.content_channels or not content.shape[0]:
            raise ValueError(f"Expected {self.content_channels}-wide content features: {path}")
        return content

    def _cache_key(self, entry):
        if is_augmented(entry):
            return None
        sources = []
        for path in (entry[0], entry[1]):
            value = Path(path)
            stat = value.stat()
            sources.append((str(value.resolve()), stat.st_size, stat.st_mtime_ns))
        return hashlib.sha256(
            (self._cache_recipe + json.dumps(sources, separators=(",", ":"))).encode()
        ).hexdigest()

    def _frames(self, audio, content, path):
        if not all(torch.isfinite(value).all() for value in (audio, content)):
            raise ValueError(f'Invalid audio or content values: {path}')
        if abs(content.shape[0] / CONTENT_RATE - audio.numel() / self.sample_rate) > 0.25:
            raise ValueError(f'Content duration does not match audio; re-extract features: {path}')
        frames = min(audio.shape[0] // self.hop, mel_frames(content.shape[0], self.sample_rate, self.hop))
        if frames < 4:
            raise ValueError("Training clips must contain at least four mel frames.")
        return frames

    def _build_original_values(self, index):
        if self._original_keys[index] is None:
            return None
        wav_path, content_path = self.entries[index][:2]
        audio, content = self._audio(wav_path), self._content(content_path)
        frames = self._frames(audio, content, wav_path)
        f0 = torch.from_numpy(parselmouth_f0(audio.numpy(), self.sample_rate, self.hop, frames))
        audio = audio[: frames * self.hop]
        with torch.no_grad():
            mel = self.mel(audio.unsqueeze(0), 0.0, self.hop)[0, :, :frames]
        values = dict(
            mel=mel.numpy(),
            content=content.numpy(),
            f0=f0.numpy(),
            key_shift=np.float32(0.0),
            speed=np.float32(1.0),
            frames=np.int64(frames),
            hop=np.int64(self.hop),
        )
        if not all(np.isfinite(value).all() for value in values.values()):
            raise ValueError(f'Invalid cached training features: {wav_path}')
        return values

    def __getitem__(self, index):
        _, content_path, _, _, sid = self.entries[index]
        if is_augmented(self.entries[index]):
            with np.load(content_path, allow_pickle=False) as values:
                return self._item({name: values[name] for name in values.files}, int(sid), content_path)
        if self._indexed_cache is None:
            values = self._build_original_values(index)
        else:
            try:
                values = self._indexed_cache[self._original_keys[index]]
            except (FileNotFoundError, KeyError, OSError, ValueError) as error:
                raise RuntimeError(
                    f'Indexed training feature cache is missing or stale for {self.entries[index][0]}; rebuild it before training.'
                ) from error
        source = str(self._indexed_cache.path) if self._indexed_cache is not None else self.entries[index][0]
        return self._item(values, int(sid), source)

    def _item(self, values, sid, source):
        length = min(int(values['frames']), self.max_frames)
        mel = torch.from_numpy(np.asarray(values['mel'])[:, :length].copy())
        content = torch.from_numpy(np.asarray(values['content']).copy())
        f0 = torch.from_numpy(np.asarray(values['f0'])[:length].copy())
        if mel.shape != (self.data['n_mels'], length) or content.ndim != 2 or not content.shape[0] \
                or content.shape[1] != self.content_channels or f0.shape != (length,):
            raise ValueError(f'Invalid training feature dimensions: {source}')
        if not all(torch.isfinite(value).all() for value in (mel, content, f0)) or (f0 < 0).any():
            raise ValueError(f'Invalid training feature values: {source}')
        return mel, content, f0, float(values['key_shift']), float(values['speed']), sid

    def references(self, count: int, max_seconds: float = 10.0):
        result = []
        for index in sorted(range(len(self.entries)), key=lambda i: self.entries[i][0]):
            if len(result) >= count:
                break
            wav_path, content_path, _, _, sid = self.entries[index]
            if "mute" in os.path.basename(wav_path) or is_augmented(self.entries[index]):
                continue
            audio, content = self._audio(wav_path), self._content(content_path)
            if audio.shape[0] < self.hop * 4:
                continue
            frames = min(self._frames(audio, content, wav_path), int(max_seconds * self.sample_rate) // self.hop)
            f0 = torch.from_numpy(parselmouth_f0(audio.numpy(), self.sample_rate, self.hop, frames))
            audio = audio[: frames * self.hop]
            with torch.no_grad():
                mel = self.mel(audio.unsqueeze(0))[:, :, :frames]
            result.append((mel, content.unsqueeze(0), f0.unsqueeze(0), audio.unsqueeze(0), int(sid), wav_path))
        return result


def prepare_training_cache(entries, config, cache_path, workers=4):
    if not entries:
        return
    dataset = RectifiedDataset(entries, config, 2 ** 31)
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


def collate_flow(batch):
    size = len(batch)
    frames = max(item[0].shape[-1] for item in batch)
    content_frames = max(item[1].shape[0] for item in batch)
    mel = torch.full((size, batch[0][0].shape[0], frames), math.log(1e-5))
    content = torch.zeros(size, content_frames, batch[0][1].shape[-1])
    content_mask = torch.zeros(size, 1, content_frames)
    f0 = torch.zeros(size, frames)
    key_shift = torch.zeros(size)
    speed = torch.ones(size)
    speaker = torch.zeros(size, dtype=torch.long)
    mask = torch.zeros(size, 1, frames)
    for i, (m, c, p, k, v, s) in enumerate(batch):
        n = m.shape[-1]
        mel[i, :, :n] = m
        content[i, :c.shape[0]] = c
        content_mask[i, :, :c.shape[0]] = 1.0
        f0[i, :n] = p
        key_shift[i] = k
        speed[i] = v
        speaker[i] = s
        mask[i, :, :n] = 1.0
    return mel, content, content_mask, f0, key_shift, speed, speaker, mask


def content_rms(entries, limit=64):
    total, count = 0.0, 0
    for entry in [entry for entry in entries if not is_augmented(entry)][:limit]:
        values = np.load(entry[1], allow_pickle=False).astype(np.float64)
        total += float(np.square(values).sum())
        count += values.size
    if not count or not total > 0:
        raise ValueError('Training content features are empty; re-extract features.')
    return math.sqrt(total / count)


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
