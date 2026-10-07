import math
import os
import pickle
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler

from rvc.rectified.indexed_dataset import IndexedDataset

CONTENT_RATE = 50


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
        self.frames = np.asarray(dataset.sizes, dtype=np.int64)
        if len(self.frames) and int(self.frames.max()) > self.max_frames:
            raise ValueError(f'A clip has {int(self.frames.max())} frames, more than the {self.max_frames} max frames '
                             'per batch. Raise the frame limit or slice the dataset into shorter clips.')
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
    def __init__(self, binary_dir, prefix, variances=(), creak=False):
        self.variances = list(variances)
        self.creak = bool(creak)
        with open(Path(binary_dir) / f'{prefix}.meta', 'rb') as handle:
            self.metadata = pickle.load(handle)
        self.sizes = self.metadata['lengths']
        self.indexed_ds = IndexedDataset(Path(binary_dir) / f'{prefix}.data')

    def __len__(self):
        return len(self.sizes)

    def num_frames(self, index):
        return self.sizes[index]

    def __getitem__(self, index):
        item = self.indexed_ds[index]
        return (item['mel'], item['content'], item['f0'], item['key_shift'], item['speed'], item['spk_id'],
                self._variances(item), self._creak(item))

    def _variances(self, item, frames=None):
        if not self.variances:
            return torch.zeros(0, item['mel'].shape[-1] if frames is None else frames)
        missing = [name for name in self.variances if name not in item]
        if missing:
            raise ValueError(f'The binary data has no {", ".join(missing)} curves; delete the binary folder to rebuild it.')
        return torch.stack([item[name][:frames].float() for name in self.variances])

    def _creak(self, item):
        if not self.creak:
            return torch.zeros(0, item['mel'].shape[-1])
        if 'uv' not in item or 'creak' not in item:
            raise ValueError('The binary data has no vocal fry labels; delete the binary folder to rebuild it.')
        return torch.stack((item['uv'].float(), item['creak'].float()))

    def references(self, count, data, max_seconds=10.0):
        hop, sample_rate = int(data['hop_length']), int(data['sample_rate'])
        content_step = hop * 100 / sample_rate
        result = []
        for index, (name, augmented) in enumerate(zip(self.metadata['names'], self.metadata['augmented'])):
            if len(result) >= count:
                break
            if augmented:
                continue
            item = self.indexed_ds[index]
            frames = min(self.sizes[index], int(max_seconds * sample_rate) // hop)
            audio, _ = sf.read(name, dtype='float32')
            audio = torch.from_numpy(audio.mean(-1) if audio.ndim == 2 else audio)[: frames * hop]
            content_frames = min(item['content'].shape[0], int((frames - 1) * content_step) // 2 + 2)
            uv = item['uv'][None, :frames].float() if 'uv' in item else torch.zeros(1, frames)
            result.append((item['mel'][None, :, :frames], item['content'][None, :content_frames],
                           item['f0'][None, :frames], audio[None], int(item['spk_id']), name,
                           self._variances(item, frames)[None], uv))
        return result


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
    variances = torch.zeros(size, batch[0][6].shape[0], frames)
    creak = torch.zeros(size, batch[0][7].shape[0], frames)
    for i, (m, c, p, k, v, s, r, u) in enumerate(batch):
        n = m.shape[-1]
        mel[i, :, :n] = m
        content[i, :c.shape[0]] = c
        content_mask[i, :, :c.shape[0]] = 1.0
        f0[i, :n] = p
        key_shift[i] = k
        speed[i] = v
        speaker[i] = s
        mask[i, :, :n] = 1.0
        variances[i, :, :n] = r
        creak[i, :, :n] = u
    return mel, content, content_mask, f0, key_shift, speed, speaker, mask, variances, creak


def content_rms(entries, limit=64):
    total, count = 0.0, 0
    for entry in entries[:limit]:
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
