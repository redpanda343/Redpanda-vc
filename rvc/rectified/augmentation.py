import copy
import hashlib
import json
import math
import os
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from rvc.rectified.config import resolve_config
from rvc.rectified.data import RectifiedDataset, is_augmented, read_filelist
from rvc.rectified.pitch import parselmouth_f0


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


def prepare_source(dataset, entry):
    audio, content = dataset._audio(entry[0]), dataset._content(entry[1])
    dataset._frames(audio, content, entry[0])
    return audio, content


def save_features(path, values):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **values)
    os.replace(temporary, path)


def generate_features(dataset, entry, tasks, paths, device, source=None, writer=None):
    audio, content = prepare_source(dataset, entry) if source is None else source
    mel_extractor = dataset.mel.to(device)
    for task, path in zip(tasks, paths):
        hop = int(round(dataset.hop * task.get('speed', 1.0)))
        shift = task.get('key_shift', 0.0)
        with torch.no_grad():
            mel = mel_extractor(audio.to(device).unsqueeze(0), shift, hop)[0].cpu()
        length = mel.shape[-1]
        f0 = parselmouth_f0(audio.numpy(), dataset.sample_rate, hop, length) * 2 ** (shift / 12)
        values = dict(mel=mel.numpy(), content=content.numpy(), f0=f0.astype(np.float32),
                      key_shift=np.float32(shift), speed=np.float32(hop / dataset.hop),
                      frames=np.int64(length), hop=np.int64(dataset.hop))
        if not all(np.isfinite(value).all() for value in values.values()) or length < 4:
            raise ValueError(f'Invalid augmented features: {entry[0]}')
        if writer is None:
            save_features(path, values)
        else:
            writer(path, values)


def generate_groups(dataset, entries, grouped, device, workers):
    if workers < 2:
        for index, items in grouped.items():
            generate_features(dataset, entries[index], [task for task, _ in items],
                              [path for _, path in items], device)
            yield len(items)
        return
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(max(1, min(previous_threads, (os.cpu_count() or 1) // workers)))
    try:
        with ThreadPoolExecutor(max_workers=workers) as readers, ThreadPoolExecutor(max_workers=workers) as writers:
            jobs = iter(grouped.items())
            pending_sources, pending_writes = deque(), deque()

            def submit_source():
                job = next(jobs, None)
                if job is None:
                    return
                index, items = job
                pending_sources.append((index, items, readers.submit(prepare_source, dataset, entries[index])))

            def submit_write(path, values):
                if len(pending_writes) >= workers * 2:
                    pending_writes.popleft().result()
                pending_writes.append(writers.submit(save_features, path, values))

            for _ in range(workers):
                submit_source()
            while pending_sources:
                index, items, future = pending_sources.popleft()
                source = future.result()
                generate_features(dataset, entries[index], [task for task, _ in items],
                                  [path for _, path in items], device, source=source, writer=submit_write)
                submit_source()
                if not pending_sources:
                    while pending_writes:
                        pending_writes.popleft().result()
                yield len(items)
    finally:
        torch.set_num_threads(previous_threads)


def prepare_augmentation(experiment, root, originals, train_entries, config, seed, device):
    config = resolve_config(config)
    tasks = augmentation_plan(train_entries, config['flow'], seed)
    sources = [[list(entry), [(Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
                              for path in entry[:2]]] for entry in train_entries]
    recipe = dict(version=3, sources=sources, tasks=tasks, data=config['data'])
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:24]
    folder = experiment / 'augmentation' / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    filelist = experiment / 'filelist.txt'
    backup = experiment / 'filelist.original.txt'
    if not backup.exists():
        temporary = backup.with_suffix('.txt.tmp')
        temporary.write_bytes(filelist.read_bytes())
        os.replace(temporary, backup)
    dataset = RectifiedDataset(train_entries, config, 2 ** 31)
    grouped, rows = {}, []
    for number, task in enumerate(tasks):
        entry = train_entries[task['index']]
        path = folder / f'{number:08d}.flow.npz'
        rows.append([entry[0], str(path), str(path), str(path), str(entry[4])])
        if not path.exists():
            grouped.setdefault(task['index'], []).append((task, path))
    if grouped:
        requested = int(config['flow']['augmentation_workers'])
        if requested < 0:
            raise ValueError('Augmentation workers cannot be negative.')
        workers = max(1, min(requested, os.cpu_count() or 1, len(grouped)))
        completed = len(rows) - sum(len(values) for values in grouped.values())
        last_report = time.monotonic()
        for count in generate_groups(dataset, train_entries, grouped, device, workers):
            completed += count
            if completed == len(rows) or time.monotonic() - last_report >= 5:
                print(f'Augmentation: {completed:,}/{len(rows):,}', flush=True)
                last_report = time.monotonic()
    temporary = folder / 'manifest.json.tmp'
    temporary.write_text(json.dumps(dict(fingerprint=fingerprint, examples=len(rows)), indent=2), encoding='utf-8')
    os.replace(temporary, folder / 'manifest.json')
    temporary = filelist.with_suffix('.txt.tmp')
    original_lines = [line for line in filelist.read_text(encoding='utf-8').splitlines()
                      if line.strip() and not line.strip().split('|')[1].endswith('.flow.npz')]
    temporary.write_text(''.join(line + '\n' for line in original_lines)
                         + ''.join('|'.join(map(str, entry)) + '\n' for entry in rows), encoding='utf-8')
    os.replace(temporary, filelist)
    print(f'Training filelist: {len(originals):,} original + {len(rows):,} augmented examples.', flush=True)
    return [entry for entry in read_filelist(filelist, root) if is_augmented(entry)]
