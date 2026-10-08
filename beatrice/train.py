import argparse
import json
import os
import runpy
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


def prepare_dataset(experiment, dataset):
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
    config = json.loads((TRAINER_ROOT / 'assets' / 'default_config.json').read_text(encoding='utf-8'))
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
    args = parser.parse_args()
    select_device(args.device)
    experiment = ROOT / 'logs' / args.model_name
    ensure_trainer()
    data_dir = prepare_dataset(experiment, args.dataset)
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
