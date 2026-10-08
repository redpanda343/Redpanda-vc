import copy
import hashlib
import json
import math
import os
import pickle
import platform
import random
import traceback
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.multiprocessing import Manager, Process, get_context
from tqdm import tqdm

from rvc.rectified.data import CONTENT_RATE, mel_frames
from rvc.rectified.indexed_dataset import IndexedDatasetBuilder
from rvc.rectified.mel import LogMel
from rvc.rectified.pitch import extract_pitch
from rvc.rectified.variance import extract_variances, resample_variances, variance_names

BINARY_VERSION = 2


def augmentation_plan(entries, settings, seed):
    if not entries:
        raise ValueError('Augmentation requires original training examples.')
    args = settings['augmentation_args']
    model = settings['model']
    pitch, stretch = args['random_pitch_shifting'], args['random_time_stretching']
    for name, spec in (('random_pitch_shifting', pitch), ('random_time_stretching', stretch)):
        if not math.isfinite(float(spec['scale'])) or spec['scale'] < 0:
            raise ValueError(f'Invalid augmentation scale: {name}')
    if pitch['enabled']:
        low, high = pitch['range']
        if not model['key_shift'] or not math.isfinite(low + high) or not low < 0 < high:
            raise ValueError('Random pitch shifting requires key-shift conditioning and min < 0 < max.')
    if stretch['enabled']:
        low, high = stretch['range']
        if not model['speed'] or not math.isfinite(low + high) or not 0 < low < 1 < high:
            raise ValueError('Time stretching requires speed conditioning and 0 < min < 1 < max.')
    rng = random.Random(seed)
    names = list(range(len(entries)))
    tasks, total_scale = [], 0.0
    if pitch['enabled']:
        low, high = pitch['range']
        for index in rng.choices(names, k=int(pitch['scale'] * len(names))):
            value = rng.uniform(-1, 1)
            tasks.append({'index': index, 'key_shift': low * abs(value) if value < 0 else high * value})
        total_scale += pitch['scale']
    if stretch['enabled']:
        scale = stretch['scale']
        raw = int(scale / (1 + total_scale) * len(names))
        augmented = int(total_scale * scale / (1 + total_scale) * len(names))
        mutate = int(total_scale * scale / (1 + scale) * len(names))
        kinds = [0] * raw + [1] * augmented + [2] * mutate
        chosen = (rng.choices(names, k=raw) + rng.choices(tasks, k=augmented)
                  + rng.sample(tasks, k=min(mutate, len(tasks))))
        low, high = stretch['range']
        for kind, chosen_item in zip(kinds, chosen):
            speed = low * (high / low) ** rng.random()
            if kind == 0:
                tasks.append({'index': chosen_item, 'speed': speed})
            elif kind == 1:
                item = copy.deepcopy(chosen_item)
                item['speed'] = speed
                tasks.append(item)
            else:
                chosen_item['speed'] = speed
    return tasks


class ItemBuilder:
    def __init__(self, config, device):
        self.data = config['data']
        self.hop = int(self.data['hop_length'])
        self.sample_rate = int(self.data['sample_rate'])
        self.content_channels = int(config['flow']['model']['content_channels'])
        self.pitch_extractor = config['flow']['pitch_extractor']
        self.settings = config['flow']
        self.variances = variance_names(config['flow']['model'])
        self.device = torch.device(device)
        self.mel = LogMel.from_config(self.data).to(self.device)

    def source(self, entry):
        wav_path, content_path = entry[0], entry[1]
        data, sample_rate = sf.read(wav_path, dtype='float32')
        if sample_rate != self.sample_rate:
            raise ValueError(f'{wav_path} is {sample_rate} Hz; the rectified models train at {self.sample_rate} Hz. '
                             'Preprocess the dataset at that rate.')
        audio = torch.from_numpy(data)
        audio = audio.mean(-1) if audio.dim() == 2 else audio
        content = torch.from_numpy(np.load(content_path, allow_pickle=False).astype(np.float32))
        if content.ndim != 2 or content.shape[1] != self.content_channels or not content.shape[0]:
            raise ValueError(f'Expected {self.content_channels}-wide content features: {content_path}')
        if not all(torch.isfinite(value).all() for value in (audio, content)):
            raise ValueError(f'Invalid audio or content values: {wav_path}')
        if abs(content.shape[0] / CONTENT_RATE - audio.numel() / self.sample_rate) > 0.25:
            raise ValueError(f'Content duration does not match audio; re-extract features: {wav_path}')
        frames = min(audio.shape[0] // self.hop, mel_frames(content.shape[0], self.sample_rate, self.hop))
        if frames < 4:
            raise ValueError(f'Training clips must contain at least four mel frames: {wav_path}')
        return audio, content, frames

    @torch.no_grad()
    def original(self, audio, content, frames, sid, name):
        f0, uv = self.f0(audio, self.hop, frames)
        if uv.all():
            return None
        clip = audio[: frames * self.hop].to(self.device)
        mel = self.mel(clip.unsqueeze(0), 0.0, self.hop)[0, :, :frames]
        variances = extract_variances(self.variances, audio.numpy(), f0, uv, frames, self.data, self.settings,
                                      self.device)
        return self._item(mel, content, f0, 0.0, 1.0, sid, variances)

    @torch.no_grad()
    def augmented(self, audio, content, task, sid, original):
        hop = int(round(self.hop * task.get('speed', 1.0)))
        shift = task.get('key_shift', 0.0)
        mel = self.mel(audio.to(self.device).unsqueeze(0), shift, hop)[0]
        f0 = self.f0(audio, hop, mel.shape[-1])[0] * 2 ** (shift / 12)
        curves = np.stack([original[name] for name in self.variances]) if self.variances else None
        variances = resample_variances(curves, hop / self.hop, mel.shape[-1], self.data) if self.variances else None
        return self._item(mel, content, f0, shift, hop / self.hop, sid, variances)

    def f0(self, audio, hop, frames):
        return extract_pitch(self.pitch_extractor, audio.numpy(), self.sample_rate, hop, frames, self.device)

    def _item(self, mel, content, f0, key_shift, speed, sid, variances=None):
        mel = mel.cpu().numpy().astype(np.float32)
        item = dict(mel=mel, content=content.numpy(), f0=np.asarray(f0, dtype=np.float32),
                    key_shift=np.float32(key_shift), speed=np.float32(speed),
                    spk_id=np.int64(sid), length=np.int64(mel.shape[-1]))
        for name, curve in zip(self.variances, variances if variances is not None else ()):
            item[name] = np.asarray(curve, dtype=np.float32)
        if item['length'] < 4 or not all(np.isfinite(value).all() for value in item.values()):
            raise ValueError('Invalid binarized features.')
        return item


_BUILDERS = {}


def item_builder(config, device):
    key = str(device)
    if key not in _BUILDERS:
        _BUILDERS[key] = ItemBuilder(config, device)
    return _BUILDERS[key]


def process_item(config, device, entry):
    builder = item_builder(config, device)
    audio, content, frames = builder.source(entry)
    return builder.original(audio, content, frames, int(entry[4]), entry[0])


def chunked_worker_run(map_func, args, results_queue=None):
    for a in args:
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            results_queue.put(None)


def chunked_multiprocess_run(map_func, args, num_workers, q_max_size=1000):
    num_jobs = len(args)
    if num_jobs < num_workers:
        num_workers = num_jobs

    manager = Manager()
    queues = [manager.Queue(maxsize=q_max_size // num_workers) for _ in range(num_workers)]
    if platform.system().lower() != 'windows':
        process_creation_func = get_context('spawn').Process
    else:
        process_creation_func = Process

    workers = []
    for i in range(num_workers):
        worker = process_creation_func(
            target=chunked_worker_run, args=(map_func, args[i::num_workers], queues[i]), daemon=True
        )
        workers.append(worker)
        worker.start()

    for i in range(num_jobs):
        yield queues[i % num_workers].get()

    for worker in workers:
        worker.join()
        worker.close()


def _recipe(config, training, held, seed):
    def sources(entries):
        result = []
        for entry in entries:
            stats = [(Path(path).stat().st_size, Path(path).stat().st_mtime_ns) for path in entry[:2]]
            result.append([entry[0], entry[1], int(entry[4]), stats])
        return result

    model = config['flow']['model']
    recipe = dict(version=BINARY_VERSION, data=config['data'], content_channels=model['content_channels'],
                  key_shift=model['key_shift'], speed=model['speed'],
                  pitch_extractor=config['flow']['pitch_extractor'],
                  augmentation=config['flow']['augmentation_args'], seed=seed,
                  train=sources(training), valid=sources(held))
    names = variance_names(model)
    if names:
        recipe['variances'] = dict(names=names, hnsep=config['flow']['hnsep'],
                                   widths=[config['flow'][f'{name}_smooth_width'] for name in names])
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def _read_meta(path):
    try:
        with open(path, 'rb') as handle:
            return pickle.load(handle)
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


def process_dataset(binary_dir, prefix, entries, tasks, config, device, num_workers, recipe, progress):
    grouped = {}
    for task in tasks:
        grouped.setdefault(task['index'], []).append(task)
    builder = item_builder(config, device)
    data_path = binary_dir / f'{prefix}.data'
    temporary = data_path.with_suffix('.data.tmp')
    writer = IndexedDatasetBuilder(temporary)
    meta = dict(lengths=[], spk_ids=[], names=[], augmented=[], recipe=recipe)
    seconds = {'original': 0.0, 'total': 0.0}
    skipped = 0

    def add(item, index, augmented):
        writer.add_item(item)
        meta['lengths'].append(int(item['length']))
        meta['spk_ids'].append(int(item['spk_id']))
        meta['names'].append(entries[index][0])
        meta['augmented'].append(augmented)
        duration = int(item['length']) * float(item['speed']) * builder.hop / builder.sample_rate
        seconds['total'] += duration
        if not augmented:
            seconds['original'] += duration

    args = [(config, str(device), entry) for entry in entries]
    if num_workers > 0:
        results = chunked_multiprocess_run(process_item, args, num_workers=num_workers)
    else:
        results = (process_item(*a) for a in args)
    try:
        for index, item in enumerate(results):
            progress.update(1 + len(grouped.get(index, [])))
            if item is None:
                skipped += 1
                continue
            add(item, index, False)
            if index in grouped:
                audio, content, _ = builder.source(entries[index])
                for task in grouped[index]:
                    add(builder.augmented(audio, content, task, int(entries[index][4]), item), index, True)
    finally:
        writer.finalize()
    os.replace(temporary, data_path)
    meta_path = binary_dir / f'{prefix}.meta'
    with open(meta_path.with_suffix('.meta.tmp'), 'wb') as handle:
        pickle.dump(meta, handle)
    os.replace(meta_path.with_suffix('.meta.tmp'), meta_path)
    return len(meta['lengths']), seconds['total'], skipped


def binarize(experiment, config, training, held, seed, device, num_workers):
    binary_dir = Path(experiment) / 'binary'
    recipe = _recipe(config, training, held, seed)
    metas = [_read_meta(binary_dir / f'{prefix}.meta') for prefix in ('valid', 'train')]
    if all(meta is not None and meta.get('recipe') == recipe and (binary_dir / f'{prefix}.data').is_file()
           for meta, prefix in zip(metas, ('valid', 'train'))):
        print(f'| binary data ready: {len(metas[1]["lengths"]):,} train, {len(metas[0]["lengths"]):,} valid', flush=True)
        return binary_dir
    binary_dir.mkdir(parents=True, exist_ok=True)
    for prefix in ('valid', 'train'):
        (binary_dir / f'{prefix}.meta').unlink(missing_ok=True)
    settings = config['flow']
    args = settings['augmentation_args']
    augment = any(args[name]['enabled'] for name in ('random_pitch_shifting', 'random_time_stretching'))
    tasks = augmentation_plan(training, settings, seed) if augment else []
    with tqdm(total=len(held) + len(training) + len(tasks), desc='Binarizing', unit='clip', dynamic_ncols=True) as progress:
        valid, _, skipped_valid = process_dataset(binary_dir, 'valid', held, [], config, device, 0, recipe, progress)
        train, seconds, skipped_train = process_dataset(binary_dir, 'train', training, tasks, config, device,
                                                        int(num_workers), recipe, progress)
    skipped = skipped_valid + skipped_train
    print(f'| binary data ready: {train:,} train, {valid:,} valid, {seconds / 60:.1f} min of training audio'
          + (f', skipped {skipped} clip(s) without voiced frames' if skipped else ''), flush=True)
    return binary_dir
