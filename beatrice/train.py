import argparse
import json
import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from beatrice.resources import ROOT, TRAINER_ROOT, ensure_trainer

AUDIO_SUFFIXES = {'.wav', '.aif', '.aiff', '.fla', '.flac', '.oga', '.ogg', '.opus', '.mp3'}
REPO_ROOT_SOURCE = '''def repo_root() -> Path:
    d = Path.cwd() / "dummy" if is_notebook() else Path(__file__)
    assert d.is_absolute(), d
    for d in d.parents:
        if (d / ".git").is_dir():
            return d
    raise RuntimeError("Repository root is not found.")
'''
LAUNCHER_PATCHES = (
    (REPO_ROOT_SOURCE, 'def repo_root() -> Path:\n    return Path(__file__).resolve().parents[1]\n'),
    ('import torchaudio\n', 'import torchaudio\nfrom beatrice.compat import patch_torchaudio\n\npatch_torchaudio()\n'),
)


def write_launcher():
    source = (TRAINER_ROOT / 'beatrice_trainer' / '__main__.py').read_text(encoding='utf-8')
    for original, replacement in LAUNCHER_PATCHES:
        if source.count(original) != 1:
            raise RuntimeError('This Beatrice Trainer version is not supported: its __main__.py could not be adapted.')
        source = source.replace(original, replacement)
    launcher = TRAINER_ROOT / 'beatrice_trainer' / 'redpanda_main.py'
    launcher.write_text(source, encoding='utf-8')
    return launcher


def link_directory(target, link):
    if os.name == 'nt':
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def dataset_speakers(experiment, dataset):
    dataset = Path(dataset).resolve()
    if not dataset.is_dir():
        raise SystemExit(f'The dataset folder does not exist: {dataset}')
    entries = sorted(dataset.iterdir())
    if any(entry.is_file() and entry.suffix.lower() in AUDIO_SUFFIXES for entry in entries):
        speakers = [(experiment.name, dataset)]
    else:
        speakers = [(entry.name, entry) for entry in entries if entry.is_dir()]
    if not speakers:
        raise SystemExit(f'No audio files or speaker folders found in {dataset}.')
    return speakers


def slicing_stamp(speakers, settings):
    files = [[str(path), stat.st_size, stat.st_mtime_ns] for _, source in speakers
             for path in sorted(source.rglob('*')) if path.is_file() for stat in (path.stat(),)]
    return {'speakers': [[speaker, str(source)] for speaker, source in speakers], 'settings': settings, 'files': files}


def default_config():
    return json.loads((TRAINER_ROOT / 'assets' / 'default_config.json').read_text(encoding='utf-8'))


def slice_dataset(experiment, speakers, args):
    sample_rate = str(default_config()['out_sample_rate'])
    settings = [sample_rate, args.cutting, str(args.chunk_len), str(args.overlap_len), 'none', 'WAV',
                str(args.truncate_silence), str(args.silence_threshold), str(args.silence_to), str(args.silence_minimum), args.silence_action,
                str(args.silence_compress)]
    sliced_root = experiment / 'beatrice_sliced'
    stamp_path = sliced_root / 'slicing.json'
    stamp = slicing_stamp(speakers, settings)
    sliced = [(speaker, sliced_root / speaker / 'sliced_audios') for speaker, _ in speakers]
    if stamp_path.is_file() and json.loads(stamp_path.read_text(encoding='utf-8')) == stamp:
        print(f'Using the sliced dataset in {sliced_root}.', flush=True)
        return sliced
    if sliced_root.is_dir():
        shutil.rmtree(sliced_root)
    workers = str(args.workers or os.cpu_count() or 1)
    for speaker, source in speakers:
        print(f'Slicing {speaker} ({args.cutting}, {sample_rate} Hz)...', flush=True)
        arguments = [str(sliced_root / speaker), str(source), sample_rate, workers, args.cutting, 'False', 'False',
                     '0.0', *settings[2:]]
        if subprocess.run([sys.executable, '-u', '-m', 'shared.preprocess.preprocess', *arguments], cwd=ROOT).returncode:
            raise SystemExit(f'Slicing {speaker} failed.')
    for speaker, directory in sliced:
        if not directory.is_dir() or not any(directory.iterdir()):
            raise SystemExit(f'Slicing produced no audio for {speaker}. Use Skip or check its WAV, FLAC, MP3 and OGG '
                             'files.')
    stamp_path.write_text(json.dumps(stamp), encoding='utf-8')
    return sliced


def prepare_dataset(experiment, speakers):
    data_dir = experiment / 'beatrice_data' / experiment.name
    if data_dir.is_dir():
        for entry in data_dir.iterdir():
            try:
                if os.name == 'nt':
                    os.rmdir(entry)
                elif entry.is_symlink():
                    entry.unlink()
                else:
                    raise OSError
            except OSError:
                raise SystemExit(f'{data_dir} holds {entry.name}, which is not a speaker link. Remove it and try again.')
    data_dir.mkdir(parents=True, exist_ok=True)
    for speaker, source in speakers:
        link_directory(source, data_dir / speaker)
    return data_dir


def write_config(experiment, args):
    config = default_config()
    for key, value in (('n_steps', args.steps), ('batch_size', args.batch_size), ('num_workers', args.workers),
                       ('save_interval', args.save_interval), ('evaluation_interval', args.save_interval)):
        if value is not None:
            config[key] = value
    config['use_amp'] = args.precision != 'fp32'
    path = experiment / 'beatrice_config.json'
    path.write_text(json.dumps(config, indent=4), encoding='utf-8')
    return path


def quiet_trainer_output():
    import functools

    import torch
    import tqdm.auto

    class TrainingProgress(tqdm.auto.tqdm):
        def __init__(self, iterable=None, *args, desc=None, **kwargs):
            if desc != 'Training':
                kwargs['disable'] = True
            elif isinstance(iterable, range) and iterable.step == 1:
                kwargs.setdefault('total', iterable.stop)
                kwargs.setdefault('initial', iterable.start)
            kwargs.setdefault('unit', 'step')
            kwargs.setdefault('dynamic_ncols', True)
            super().__init__(iterable, *args, desc=desc, **kwargs)

    tqdm.auto.tqdm = TrainingProgress
    torch.hub.load = functools.partial(torch.hub.load, verbose=False)
    sys.stdout = open(os.devnull, 'w', encoding='utf-8')


def select_device(device):
    device = device.strip().lower()
    if device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    elif device.startswith('cuda:') and device[5:].isdigit():
        os.environ['CUDA_VISIBLE_DEVICES'] = device[5:]
    elif device != 'auto':
        raise SystemExit('Device must be auto, cpu or cuda:N.')


def main():
    parser = argparse.ArgumentParser(description='Train a Beatrice 2 voice conversion model with Beatrice Trainer.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--save-interval', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--cutting', choices=['Skip', 'Simple', 'Automatic'], default='Skip')
    parser.add_argument('--chunk-len', type=float, default=3.0)
    parser.add_argument('--overlap-len', type=float, default=0.3)
    parser.add_argument('--truncate-silence', action='store_true')
    parser.add_argument('--silence-action', choices=['truncate', 'compress'], default='truncate')
    parser.add_argument('--silence-threshold', type=float, default=-45.0)
    parser.add_argument('--silence-minimum', type=float, default=0.3)
    parser.add_argument('--silence-to', type=float, default=0.3)
    parser.add_argument('--silence-compress', type=float, default=50.0)
    args = parser.parse_args()
    select_device(args.device)
    experiment = ROOT / 'logs' / args.model_name
    ensure_trainer()
    speakers = dataset_speakers(experiment, args.dataset)
    if args.cutting != 'Skip':
        speakers = slice_dataset(experiment, speakers, args)
    data_dir = prepare_dataset(experiment, speakers)
    config = write_config(experiment, args)
    out_dir = experiment / 'beatrice'
    arguments = ['-d', str(data_dir), '-o', str(out_dir), '-c', str(config)]
    if (out_dir / 'checkpoint_latest.pt.gz').is_file():
        print(f'Resuming from {out_dir / "checkpoint_latest.pt.gz"}.', flush=True)
        arguments.append('-r')
    launcher = write_launcher()
    sys.argv = [str(launcher), *arguments]
    quiet_trainer_output()
    runpy.run_path(str(launcher), run_name='__main__')


if __name__ == '__main__':
    main()
