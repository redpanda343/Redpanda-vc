import os
import re
from pathlib import Path

import torch

from nsf_hifigan.config import DISCRIMINATOR_PERIODS, EXPORT_ROOT, ROOT, openvpi_config, read_json, write_json
from nsf_hifigan.resources import FINETUNE_SOURCE, pretrained_path
from rectified_flow.openvpi import generator_state, openvpi_spec
from rectified_flow.resources import VOCODERS, vocoder_path
from rectified_flow.vocoder import mel_mismatch

CHECKPOINT_PATTERN = re.compile(r'model_ckpt_steps_(\d+)\.ckpt')


def finetune_sources():
    return [FINETUNE_SOURCE]


def resolve_source(source):
    source = str(source).strip().strip('"')
    if source == FINETUNE_SOURCE:
        return pretrained_path()
    if source in VOCODERS:
        return vocoder_path(source)
    path = Path(source)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f'Pretrained vocoder not found: {path}')
    return str(path)


def trained_periods(path):
    folder = Path(path).parent
    if (folder / 'config.yaml').is_file():
        import yaml

        config = yaml.safe_load((folder / 'config.yaml').read_text(encoding='utf-8'))
        model = config.get('model_args') if isinstance(config, dict) else None
        return model.get('discriminator_periods') if isinstance(model, dict) else None
    config = read_json(folder / 'config.json')
    return config.get('discriminator_periods') if isinstance(config, dict) else None


def load_source(source, data):
    path = resolve_source(source)
    discriminator_error = (f'Cannot fine-tune {path}: discriminator weights are missing. '
                           'Use a full NSF-HiFiGAN training checkpoint with both generator and discriminator weights.')
    if path.lower().endswith('.onnx'):
        raise ValueError(discriminator_error)
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = generator_state(checkpoint)
    if state is None:
        raise ValueError(f'{path} is not an NSF-HiFiGAN vocoder.')
    discriminator = None
    if isinstance(checkpoint.get('generator'), dict):
        discriminator = checkpoint.get('discriminator')
    else:
        lightning = checkpoint.get('state_dict')
        if isinstance(lightning, dict):
            discriminator = {key[len('discriminator.'):]: value for key, value in lightning.items()
                             if key.startswith('discriminator.')} or None
    if not isinstance(discriminator, dict) or not discriminator:
        raise ValueError(discriminator_error)
    periods = trained_periods(path)
    if periods is not None and list(periods) != DISCRIMINATOR_PERIODS:
        raise ValueError(f'Cannot fine-tune {path}: it was trained with discriminator periods {list(periods)}, '
                         f'but fine-tuning uses {DISCRIMINATOR_PERIODS}.')
    hparams, mel, _ = openvpi_spec(path, state)
    mismatch = mel_mismatch(data, mel)
    if mismatch:
        raise ValueError(f'{path} uses another mel configuration: {mismatch}.')
    return dict(path=path, hparams=hparams, generator=state, discriminator=discriminator)


def checkpoints(output):
    found = []
    for path in Path(output).glob('model_ckpt_steps_*.ckpt'):
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if match:
            found.append((int(match[1]), path))
    return sorted(found)


def latest_checkpoint(output):
    found = checkpoints(output)
    return found[-1][1] if found else None


def checkpoint_path(output, iteration):
    return Path(output) / f'model_ckpt_steps_{iteration}.ckpt'


def prune_checkpoints(output, keep):
    for _, path in checkpoints(output)[:-keep]:
        path.unlink(missing_ok=True)


def export_vocoder(checkpoint, destination=None, name=None):
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    metadata = state.get('vocoder')
    if not metadata:
        raise ValueError(f'{checkpoint} is not an NSF-HiFiGAN training checkpoint.')
    generator = {key[len('generator.'):]: value for key, value in state['state_dict'].items()
                 if key.startswith('generator.')}
    if destination is None:
        destination = EXPORT_ROOT / f"{name or checkpoint.parents[1].name}_{metadata['iteration']}s"
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / 'model.ckpt.tmp'
    torch.save({'generator': generator}, temporary)
    os.replace(temporary, destination / 'model.ckpt')
    write_json(destination / 'config.json', openvpi_config(metadata['config']))
    return destination / 'model.ckpt'
