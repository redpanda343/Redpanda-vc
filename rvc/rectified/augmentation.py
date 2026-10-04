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

import librosa
import numpy as np
import torch

from rvc.rectified.data import RectifiedDataset, read_filelist, upsample_content, to_mel_rate
from rvc.rectified.config import STANDARD_PRESET, resolve_config


DEFAULT_AUGMENTATION = copy.deepcopy(STANDARD_PRESET['flow']['augmentation_args'])


def configure_augmentation(settings):
    if 'augmentation_args' not in settings:
        settings['augmentation_args'] = copy.deepcopy(DEFAULT_AUGMENTATION)
    settings.pop('augmentation_max_examples', None)
    for name in ('key_shift_range', 'key_shift_prob', 'time_stretch_range', 'time_stretch_prob'):
        settings.pop(name, None)


def is_augmented(entry):
    return str(entry[1]).endswith('.flow.npz')


def augmentation_plan(entries, settings, seed):
    if not entries:
        raise ValueError('Augmentation requires original training examples.')
    args = settings['augmentation_args']
    model = settings['model']
    for name in DEFAULT_AUGMENTATION:
        spec = args[name]
        if not math.isfinite(float(spec['scale'])) or spec['scale'] < 0:
            raise ValueError(f'Invalid augmentation scale: {name}')
    pitch, fixed, stretch = (args[name] for name in DEFAULT_AUGMENTATION)
    if pitch['enabled']:
        low, high = pitch['range']
        if not model.get('key_shift') or not math.isfinite(low + high) or not low < 0 < high:
            raise ValueError('Random pitch shifting requires key-shift conditioning and min < 0 < max.')
    if fixed['enabled']:
        if pitch['enabled'] or len(set(fixed['targets'])) != len(fixed['targets']) or fixed['scale'] >= 1:
            raise ValueError('Fixed pitch shifting requires unique targets, scale < 1 and random pitch shifting disabled.')
        if not all(math.isfinite(value) for value in fixed['targets']):
            raise ValueError('Fixed pitch targets must be finite.')
    if stretch['enabled']:
        low, high = stretch['range']
        if not model.get('speed') or not math.isfinite(low + high) or not 0 < low < 1 < high:
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
    if fixed['enabled']:
        stride = max(int(entry[4]) for entry in entries) + 1
        for offset, target in enumerate(fixed['targets'], 1):
            for index in rng.choices(names, k=int(fixed['scale'] * len(names))):
                tasks.append({'index': index, 'key_shift': target,
                              'speaker': int(entries[index][4]) + offset * stride})
        total_scale += fixed['scale'] * len(fixed['targets'])
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


def interpolate_f0(f0):
    f0 = np.asarray(f0, dtype=np.float32).copy()
    if f0.ndim != 1 or not len(f0) or not np.isfinite(f0).all() or (f0 < 0).any():
        raise ValueError('Pitch extraction returned invalid frequencies.')
    voiced = f0 > 0
    if voiced.any():
        positions = np.arange(len(f0))
        f0 = np.exp2(np.interp(positions, positions[voiced], np.log2(f0[voiced]))).astype(np.float32)
    return f0


class AugmentationPitch:
    def __init__(self, method, device, root, f0_min=50.0, f0_max=1100.0):
        self.method, self.device = method, device
        self.f0_min, self.f0_max = f0_min, f0_max
        self.last_audio, self.last_contour = None, None
        if method == 'rmvpe':
            from rvc.lib.predictors.RMVPE import RMVPE0Predictor

            self.model = RMVPE0Predictor(str(root / 'rvc/models/predictors/rmvpe.pt'), device=device)
        elif method == 'swift':
            from rvc.lib.predictors.f0 import Swift

            self.model = Swift(device=device)
        elif method != 'pm':
            raise ValueError(f'Unsupported augmentation pitch extractor: {method}')

    def __call__(self, audio, sample_rate, hop, frames):
        waveform = audio.cpu().numpy()
        if self.method == 'pm':
            import parselmouth

            left = int(np.ceil(1.5 / self.f0_min * sample_rate))
            right = hop * ((len(waveform) - 1) // hop + 1) - len(waveform) + left + 1
            waveform = np.pad(waveform, (left, right))
            contour = parselmouth.Sound(waveform, sampling_frequency=sample_rate).to_pitch_ac(
                time_step=hop / sample_rate, voicing_threshold=0.6,
                pitch_floor=self.f0_min, pitch_ceiling=self.f0_max,
            ).selected_array['frequency'].astype(np.float32)
            contour = np.pad(contour, (0, max(0, frames - len(contour))))[:frames]
            return interpolate_f0(contour)
        if audio is not self.last_audio:
            waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=16000)
            if self.method == 'rmvpe':
                contour = self.model.infer_from_audio(waveform, thred=0.03)
            else:
                contour = self.model.get_f0(waveform, f0_min=self.f0_min, f0_max=self.f0_max)
            self.last_contour = interpolate_f0(contour)
            self.last_audio = audio
        contour = self.last_contour
        return np.interp(np.arange(frames) * hop / sample_rate,
                         np.arange(len(contour)) * 0.01, contour).astype(np.float32)


def resample_curve(curve, frames, speed):
    curve = curve.cpu().numpy()
    positions = np.arange(frames) * speed
    return torch.from_numpy(np.interp(positions, np.arange(len(curve)), curve).astype(np.float32))


def prepare_source(dataset, entry):
    audio = dataset._audio(entry[0])
    source_f0 = torch.from_numpy(np.load(entry[3], allow_pickle=False).astype(np.float32))
    content = upsample_content(torch.from_numpy(np.load(entry[1], allow_pickle=False).astype(np.float32)),
                               dataset.data['content_interpolation'])
    if content.ndim != 2 or content.shape[1] != dataset.content_channels:
        raise ValueError(f'Invalid content features: {entry[1]}')
    if not all(torch.isfinite(value).all() for value in (audio, source_f0, content)) or (source_f0 < 0).any():
        raise ValueError(f'Invalid source feature values: {entry[0]}')
    frames = audio.numel() // dataset.hop
    if frames < 4 or abs(source_f0.numel() / 100 - audio.numel() / dataset.sample_rate) > 0.25:
        raise ValueError(f'Invalid audio or pitch duration: {entry[0]}')
    if abs(content.shape[0] / 100 - audio.numel() / dataset.sample_rate) > 0.25:
        raise ValueError(f'Invalid content duration: {entry[0]}')
    energy = dataset._energy(audio.unsqueeze(0), frames, dataset.hop)[0]
    breathiness = dataset._breathiness(audio.unsqueeze(0), source_f0.unsqueeze(0), frames, dataset.hop)[0]
    variances = ()
    if dataset.use_variances:
        from rvc.rectified.data import variance_curves

        variances = variance_curves(audio.unsqueeze(0), source_f0.unsqueeze(0), frames,
                                   dataset.sample_rate, dataset.hop)
    return audio, source_f0, content, energy, breathiness, variances


def save_features(path, values):
    f0_mel = 1127 * np.log1p(values['f0'] / 700)
    low, high = 1127 * np.log1p(np.array([50.0, 1100.0]) / 700)
    coarse = np.rint(np.clip((f0_mel - low) * 254 / (high - low) + 1, 1, 255)).astype(np.int64)
    for destination, value in ((path.with_suffix('.f0.npy'), values['f0']),
                               (path.with_suffix('.coarse.npy'), coarse)):
        temporary = destination.with_suffix(destination.suffix + '.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, value, allow_pickle=False)
        os.replace(temporary, destination)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **values)
    os.replace(temporary, path)


def generate_features(dataset, entry, tasks, paths, pitch, device, speed_embed=True,
                      source=None, writer=None):
    audio, source_f0, content, energy, breathiness, variances = (
        prepare_source(dataset, entry) if source is None else source
    )
    mel_extractor = dataset.mel.to(device)
    for task, path in zip(tasks, paths):
        hop = int(round(dataset.hop * task.get('speed', 1.0)))
        speed = hop / dataset.hop
        shift = task.get('key_shift', 0.0)
        with torch.no_grad():
            mel = mel_extractor(audio.to(device).unsqueeze(0), shift, hop)[0].cpu()
        length = mel.shape[-1]
        if hop != dataset.hop or speed_embed:
            f0 = pitch(audio, dataset.sample_rate, hop, length)
        else:
            contour = interpolate_f0(source_f0.numpy())
            f0 = np.interp(np.arange(length) * hop / dataset.sample_rate,
                           np.arange(len(contour)) * 0.01, contour).astype(np.float32)
        f0 = torch.from_numpy(f0) * 2 ** (shift / 12)
        values = dict(mel=mel, content=to_mel_rate(content, length, dataset.sample_rate, hop),
                      f0=f0, energy=resample_curve(energy, length, speed),
                      breathiness=resample_curve(breathiness, length, speed),
                      key_shift=np.float32(shift if 'speaker' not in task else 0.0),
                      speed=np.float32(speed), frames=np.int64(length), hop=np.int64(dataset.hop))
        for name, curve in zip(('voicing', 'tension'), variances):
            values[name] = resample_curve(curve[0], length, speed)
        values = {name: value.numpy() if torch.is_tensor(value) else value for name, value in values.items()}
        if not all(np.isfinite(value).all() for value in values.values()) or length < 4:
            raise ValueError(f'Invalid augmented features: {entry[0]}')
        if writer is None:
            save_features(path, values)
        else:
            writer(path, values)


def generate_groups(dataset, entries, grouped, pitch, device, speed_embed, workers):
    if workers < 2:
        for index, items in grouped.items():
            generate_features(dataset, entries[index], [task for task, _ in items],
                              [path for _, path in items], pitch, device, speed_embed=speed_embed)
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
                                  [path for _, path in items], pitch, device, speed_embed=speed_embed,
                                  source=source, writer=submit_write)
                submit_source()
                if not pending_sources:
                    while pending_writes:
                        pending_writes.popleft().result()
                yield len(items)
    finally:
        torch.set_num_threads(previous_threads)


def prepare_augmentation(experiment, root, originals, train_entries, config, seed, device, pitch=None):
    config = resolve_config(config)
    tasks = augmentation_plan(train_entries, config['flow'], seed)
    info_path = experiment / 'model_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
    method = info.get('f0_method', 'rmvpe')
    sources = [[list(entry), [(Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
                              for path in entry[:4]]] for entry in train_entries]
    recipe = dict(version=1, sources=sources, tasks=tasks, data=config['data'],
                  method=method, speed_embed=config['flow']['model'].get('speed', False),
                  variances=[config['flow']['model'].get(name, False)
                                           for name in ('voicing', 'tension')])
    checkpoint = root / 'rvc/models/predictors/rmvpe.pt'
    if method == 'rmvpe' and checkpoint.exists():
        recipe['pitch_checkpoint'] = [checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns]
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:24]
    folder = experiment / 'augmentation' / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    filelist = experiment / 'filelist.txt'
    backup = experiment / 'filelist.original.txt'
    if not backup.exists():
        temporary = backup.with_suffix('.txt.tmp')
        temporary.write_bytes(filelist.read_bytes())
        os.replace(temporary, backup)
    dataset = RectifiedDataset(train_entries, config, 2 ** 31, augment=False)
    grouped, rows = {}, []
    for number, task in enumerate(tasks):
        entry = train_entries[task['index']]
        path = folder / f'{number:08d}.flow.npz'
        coarse, f0 = path.with_suffix('.coarse.npy'), path.with_suffix('.f0.npy')
        rows.append([entry[0], str(path), str(coarse), str(f0), str(task.get('speaker', entry[4]))])
        if not all(value.exists() for value in (path, coarse, f0)):
            grouped.setdefault(task['index'], []).append((task, path))
    if grouped:
        requested = int(config['flow'].get('augmentation_workers', config['flow'].get('num_workers', 4)))
        if requested < 0:
            raise ValueError('Augmentation workers cannot be negative.')
        workers = max(1, min(requested, os.cpu_count() or 1, len(grouped)))
        print(f'Generating {len(rows):,} DiffSinger-style augmented examples before training using {method}, '
              f'{workers} preparation workers, {workers} file writers and uncompressed features.', flush=True)
        pitch = pitch or AugmentationPitch(method, device, root)
        completed = len(rows) - sum(len(values) for values in grouped.values())
        last_report = time.monotonic()
        for count in generate_groups(dataset, train_entries, grouped, pitch, device,
                                     config['flow']['model'].get('speed', False), workers):
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
    return read_filelist(filelist, root)
