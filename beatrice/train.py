import argparse
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

from beatrice.resources import ROOT, TRAINER_ROOT, ensure_trainer

AUDIO_SUFFIXES = {'.wav', '.aif', '.aiff', '.fla', '.flac', '.oga', '.ogg', '.opus', '.mp3'}
CHECKPOINT = 'checkpoint_latest.pt.gz'
REPO_ROOT_SOURCE = '''def repo_root() -> Path:
    d = Path.cwd() / "dummy" if is_notebook() else Path(__file__)
    assert d.is_absolute(), d
    for d in d.parents:
        if (d / ".git").is_dir():
            return d
    raise RuntimeError("Repository root is not found.")
'''
RUN_FILES_SOURCE = '''    if not resume:
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(dict(h), f, indent=4)
        if not is_notebook():
            shutil.copy(__file__, out_dir)
'''
LAUNCHER_PATCHES = (
    (REPO_ROOT_SOURCE, 'def repo_root() -> Path:\n    return Path(__file__).resolve().parents[1]\n'),
    ('import torchaudio\n', 'import torchaudio\nfrom beatrice.compat import patch_torchaudio\n\npatch_torchaudio()\n'),
    (RUN_FILES_SOURCE, ''),
    ('        writer = SummaryWriter(out_dir)\n',
     '        writer = SummaryWriter(in_wav_dataset_dir.parent / "events")\n'),
    ('                shutil.copy(checkpoint_file_save, out_dir / "checkpoint_latest.pt.gz")\n',
     '                os.replace(checkpoint_file_save, out_dir / "checkpoint_latest.pt.gz")\n'),
    ('                del paraphernalia_dir\n',
     '                for old_paraphernalia_dir in out_dir.glob("paraphernalia_*"):\n'
     '                    if old_paraphernalia_dir != paraphernalia_dir:\n'
     '                        shutil.rmtree(old_paraphernalia_dir, ignore_errors=True)\n'
     '                del paraphernalia_dir\n'),
)
LEGACY_ENTRIES = ('beatrice_data', 'beatrice_sliced', 'beatrice_config.json')


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


def remove_path(path):
    if path.is_symlink() or os.path.isjunction(path):
        if os.name == 'nt':
            os.rmdir(path)
        else:
            path.unlink()
    elif path.is_dir():
        for entry in path.iterdir():
            remove_path(entry)
        path.rmdir()
    elif os.path.lexists(path):
        path.unlink()


def work_directory(experiment):
    return experiment / 'temp'


def remove_work(experiment):
    if os.path.lexists(work_directory(experiment)):
        remove_path(work_directory(experiment))


def model_folders(directory):
    return sorted((path for path in directory.glob('paraphernalia_*') if path.is_dir()),
                  key=lambda path: path.stat().st_mtime)


def tidy_experiment(experiment):
    legacy = experiment / 'beatrice'
    if legacy.is_dir() and not legacy.is_symlink():
        if (legacy / CHECKPOINT).is_file() and not (experiment / CHECKPOINT).exists():
            os.replace(legacy / CHECKPOINT, experiment / CHECKPOINT)
        exports = model_folders(legacy)
        if exports and not model_folders(experiment):
            os.replace(exports[-1], experiment / exports[-1].name)
        remove_path(legacy)
    for name in LEGACY_ENTRIES:
        if os.path.lexists(experiment / name):
            remove_path(experiment / name)
    remove_work(experiment)
    for checkpoint in experiment.glob('checkpoint_*.pt.gz'):
        if checkpoint.name != CHECKPOINT:
            checkpoint.unlink()
    for export in model_folders(experiment)[:-1]:
        remove_path(export)


def dataset_speakers(experiment, dataset):
    dataset = Path(dataset).resolve()
    work = work_directory(experiment).resolve()
    if not dataset.is_dir():
        raise SystemExit(f'The dataset folder does not exist: {dataset}')
    if dataset == work or work in dataset.parents or dataset in work.parents:
        raise SystemExit(f'Choose a dataset folder outside {experiment}.')
    entries = sorted(dataset.iterdir())
    if any(entry.is_file() and entry.suffix.lower() in AUDIO_SUFFIXES for entry in entries):
        speakers = [(experiment.name, dataset)]
    else:
        speakers = [(entry.name, entry) for entry in entries if entry.is_dir()]
    if not speakers:
        raise SystemExit(f'No audio files or speaker folders found in {dataset}.')
    return speakers


def default_config():
    return json.loads((TRAINER_ROOT / 'assets' / 'default_config.json').read_text(encoding='utf-8'))


def build_dataset(experiment, speakers, args):
    work = work_directory(experiment)
    data_dir = work / experiment.name
    data_dir.mkdir(parents=True)
    if args.cutting == 'Skip':
        for speaker, source in speakers:
            link_directory(source, data_dir / speaker)
        return data_dir
    sample_rate = str(default_config()['out_sample_rate'])
    staging = work / 'slicing'
    workers = str(args.workers or os.cpu_count() or 1)
    for speaker, source in speakers:
        print(f'Slicing {speaker} ({args.cutting}, {sample_rate} Hz)...', flush=True)
        arguments = [str(staging / speaker), str(source), sample_rate, workers, args.cutting, 'False', 'False', '0.0',
                     str(args.chunk_len), str(args.overlap_len), 'none', 'WAV', str(args.truncate_silence),
                     str(args.silence_threshold), str(args.silence_to), str(args.silence_minimum), args.silence_action,
                     str(args.silence_compress)]
        if subprocess.run([sys.executable, '-u', '-m', 'shared.preprocess.preprocess', *arguments], cwd=ROOT).returncode:
            raise SystemExit(f'Slicing {speaker} failed.')
        sliced = staging / speaker / 'sliced_audios'
        if not sliced.is_dir() or not any(sliced.iterdir()):
            raise SystemExit(f'Slicing produced no audio for {speaker}. Use Skip or check its WAV, FLAC, MP3 and OGG '
                             'files.')
        os.replace(sliced, data_dir / speaker)
    remove_path(staging)
    return data_dir


def write_config(experiment, args):
    config = default_config()
    for key, value in (('n_steps', args.steps), ('batch_size', args.batch_size), ('num_workers', args.workers),
                       ('save_interval', args.save_interval), ('evaluation_interval', args.save_interval)):
        if value is not None:
            config[key] = value
    config['use_amp'] = args.precision != 'fp32'
    path = work_directory(experiment) / 'config.json'
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
    parser.add_argument('--cutting', choices=['Skip', 'Simple'], default='Skip')
    parser.add_argument('--chunk-len', type=float, default=5.0)
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
    tidy_experiment(experiment)
    speakers = dataset_speakers(experiment, args.dataset)
    data_dir = build_dataset(experiment, speakers, args)
    config = write_config(experiment, args)
    arguments = ['-d', str(data_dir), '-o', str(experiment), '-c', str(config)]
    if (experiment / CHECKPOINT).is_file():
        print(f'Resuming from {experiment / CHECKPOINT}.', flush=True)
        arguments.append('-r')
    launcher = write_launcher()
    sys.argv = [str(launcher), *arguments]
    quiet_trainer_output()
    runpy.run_path(str(launcher), run_name='__main__')


if __name__ == '__main__':
    main()
