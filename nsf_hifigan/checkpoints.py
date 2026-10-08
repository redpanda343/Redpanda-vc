import os
import re
from pathlib import Path

import torch

from nsf_hifigan.config import EXPORT_ROOT, ROOT, openvpi_config, write_json
from rectified_flow.openvpi import generator_state, onnx_generator_state, openvpi_spec
from rectified_flow.resources import VOCODERS, vocoder_path
from rectified_flow.vocoder import mel_mismatch

CHECKPOINT_PATTERN = re.compile(r'model_ckpt_steps_(\d+)\.ckpt')


def finetune_sources():
    exports = sorted(path.relative_to(ROOT).as_posix() for path in EXPORT_ROOT.glob('*/model.ckpt')) \
        if EXPORT_ROOT.is_dir() else []
    return [*VOCODERS, *exports]


def resolve_source(source):
    source = str(source).strip().strip('"')
    if source in VOCODERS:
        return vocoder_path(source)
    path = Path(source)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f'Pretrained vocoder not found: {path}')
    return str(path)


def load_source(source, data):
    path = resolve_source(source)
    discriminator = None
    if path.lower().endswith('.onnx'):
        state = onnx_generator_state(path)
    else:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = generator_state(checkpoint)
        lightning = checkpoint.get('state_dict')
        if isinstance(lightning, dict):
            discriminator = {key[len('discriminator.'):]: value for key, value in lightning.items()
                             if key.startswith('discriminator.')} or None
    if state is None:
        raise ValueError(f'{path} is not an NSF-HiFiGAN vocoder.')
    hparams, mel, folded = openvpi_spec(path, state)
    mismatch = mel_mismatch(data, mel)
    if mismatch:
        raise ValueError(f'{path} uses another mel configuration: {mismatch}.')
    return dict(path=path, hparams=hparams, generator=folded, discriminator=discriminator)


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
