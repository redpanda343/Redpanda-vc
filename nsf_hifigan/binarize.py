import argparse
import random
import shutil
from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

from nsf_hifigan.config import PITCH_EXTRACTORS, default_data, experiment_paths, write_json
from nsf_hifigan.losses import vocoder_mel
from rectified_flow.distributed import parse_devices
from rectified_flow.pitch import extract_pitch, interpolate_f0, parselmouth_contour

AUDIO_SUFFIXES = ('.wav', '.flac')
VALIDATION_SUFFIXES = ('.wav', '.mp3', '.flac', '.ogg')
_MEL = {}
PARSELMOUTH_SILENCE_THRESHOLD = 0.01
RMVPE_BATCH_SAMPLES = 4096 * 160
RMVPE_PENDING_CLIPS = 512


def read_audio(path, sample_rate):
    path = Path(path)
    if path.suffix.lower() in AUDIO_SUFFIXES:
        audio, rate = sf.read(path, dtype='float32', always_2d=True)
        if rate == sample_rate:
            return audio.mean(axis=1)
    from shared.preprocess.audio import load_audio_ffmpeg

    return np.asarray(load_audio_ffmpeg(str(path), sample_rate), dtype=np.float32)


def load_clip(path, data):
    key = tuple(sorted(data.items()))
    if key not in _MEL:
        _MEL[key] = vocoder_mel(data)
    audio = np.clip(read_audio(path, data['sample_rate']), -1.0, 1.0)
    with torch.no_grad():
        mel = _MEL[key](torch.from_numpy(audio)[None])[0].T.numpy().astype(np.float32)
    return (audio, mel) if len(mel) else None


def clip_features(path, data, extractor, device):
    loaded = load_clip(path, data)
    if loaded is None:
        return None
    audio, mel = loaded
    frames = len(mel)
    if extractor == 'parselmouth':
        f0 = interpolate_f0(parselmouth_contour(audio, data['sample_rate'], data['hop_length'], frames,
                                                silence_threshold=PARSELMOUTH_SILENCE_THRESHOLD))
    else:
        f0, _ = extract_pitch(extractor, audio, data['sample_rate'], data['hop_length'], frames, device)
    return dict(audio=audio, mel=mel, f0=np.asarray(f0, dtype=np.float32))


def write_clip(job):
    source, target, data, extractor, device = job
    torch.set_num_threads(1)
    features = clip_features(source, data, extractor, device)
    if features is None:
        return None
    np.savez(target, **features)
    return target.name, len(features['mel'])


def save_clip(target, audio, mel, f0):
    np.savez(target, audio=audio, mel=mel, f0=np.asarray(f0, dtype=np.float32))
    return target.name, len(mel)


def run_rmvpe_jobs(jobs, workers, device):
    from rectified_flow.pitch import rmvpe_model

    torch.set_num_threads(1)
    model = rmvpe_model(device)
    results = {}
    buckets = {}
    pending = 0
    saving = deque()
    with (ThreadPoolExecutor(workers) as loaders, ThreadPoolExecutor(workers) as savers,
          tqdm(total=len(jobs), desc='Extracting vocoder features', unit='clip') as progress):
        def collect(limit):
            while saving and (len(saving) > limit or saving[0][1].done()):
                source, future = saving.popleft()
                results[source] = future.result()
                progress.update(1)

        def flush(key):
            nonlocal pending
            items = buckets.pop(key)
            pending -= len(items)
            data = items[0][0][2]
            pitches = model.get_pitch_batch([audio for _, audio, _ in items], data['sample_rate'],
                                            [len(mel) for _, _, mel in items], hop_size=data['hop_length'],
                                            interp_uv=True)
            for (job, audio, mel), (f0, _) in zip(items, pitches):
                saving.append((job[0], savers.submit(save_clip, job[1], audio, mel, f0)))
            collect(8 * workers)

        queued = iter(jobs)
        loading = deque((job, loaders.submit(load_clip, job[0], job[2])) for _, job in zip(range(4 * workers), queued))
        while loading:
            job, future = loading.popleft()
            following = next(queued, None)
            if following is not None:
                loading.append((following, loaders.submit(load_clip, following[0], following[2])))
            loaded = future.result()
            if loaded is None:
                results[job[0]] = None
                progress.update(1)
                continue
            audio, mel = loaded
            key = model.padded_length(len(audio), job[2]['sample_rate'])
            buckets.setdefault(key, []).append((job, audio, mel))
            pending += 1
            if len(buckets[key]) * key >= RMVPE_BATCH_SAMPLES:
                flush(key)
            elif pending > RMVPE_PENDING_CLIPS:
                flush(max(buckets, key=lambda name: len(buckets[name]) * name))
            collect(8 * workers)
        for key in list(buckets):
            flush(key)
        collect(0)
    return [results[job[0]] for job in jobs]


def collect_sources(experiment):
    clips = sorted(path for path in (experiment / 'sliced_audios').glob('*')
                   if path.suffix.lower() in AUDIO_SUFFIXES and 'mute' not in path.stem)
    validation_root = experiment / 'validation' / 'audio'
    validation = sorted(path for path in validation_root.rglob('*')
                        if path.suffix.lower() in VALIDATION_SUFFIXES) if validation_root.is_dir() else []
    return clips, validation


def run_jobs(jobs, workers, device):
    if jobs and jobs[0][3] == 'rmvpe':
        return run_rmvpe_jobs(jobs, workers, device)
    if device != 'cpu' or workers <= 1:
        return [write_clip(job) for job in tqdm(jobs, desc='Extracting vocoder features', unit='clip')]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(tqdm(executor.map(write_clip, jobs, chunksize=4), total=len(jobs),
                         desc='Extracting vocoder features', unit='clip'))


def binarize(experiment, extractor, device, workers, valid_clips, seed=1234):
    paths = experiment_paths(experiment)
    clips, validation = collect_sources(paths['experiment'])
    if not clips:
        raise SystemExit(f"No sliced audio found in {paths['experiment'] / 'sliced_audios'}. Slice the dataset first.")
    data = default_data()
    if paths['data'].is_dir():
        shutil.rmtree(paths['data'])
    paths['data'].mkdir(parents=True)
    held = []
    if not validation:
        count = min(int(valid_clips), len(clips) // 10)
        held = random.Random(seed).sample(clips, count) if count > 0 else []
    jobs = [(path, paths['data'] / f'{path.stem}.npz', data, extractor, device) for path in clips]
    jobs += [(path, paths['data'] / f'validation_{index}.npz', data, extractor, device)
             for index, path in enumerate(validation)]
    results = dict(zip([job[0] for job in jobs], run_jobs(jobs, workers, device)))
    valid_sources = validation or held
    train = [list(results[path]) for path in clips if path not in held and results[path]]
    valid = [list(results[path]) for path in valid_sources if results[path]]
    write_json(paths['index'], dict(data=data, pitch_extractor=extractor, train=train, valid=valid))
    print(f'Vocoder features ready: {len(train)} training and {len(valid)} validation clip(s) with {extractor} F0.',
          flush=True)


def main():
    parser = argparse.ArgumentParser(description='Extract mel spectrograms and F0 for NSF-HiFiGAN vocoder training.')
    parser.add_argument('experiment')
    parser.add_argument('--pitch-extractor', choices=PITCH_EXTRACTORS, default='parselmouth')
    parser.add_argument('--device', default='auto', help='auto, cpu or cuda:N (used by RMVPE).')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--valid-clips', type=int, default=5,
                        help='Clips held out for validation when the dataset has no validation folder.')
    args = parser.parse_args()
    device = 'cpu' if args.pitch_extractor == 'parselmouth' else parse_devices(args.device)[0]
    binarize(Path(args.experiment), args.pitch_extractor, device, max(1, args.workers), max(0, args.valid_clips))


if __name__ == '__main__':
    main()
