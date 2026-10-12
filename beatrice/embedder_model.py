import hashlib
import importlib.util
import json
import os
import sys
from functools import lru_cache, partial
from pathlib import Path

import torch
import torch.nn.functional as F

from beatrice.embedder_config import LOOKAHEAD_CHOICES
from beatrice.resources import ROOT, TRAINER_ROOT

FORMAT = 1
SAMPLE_LOOKAHEAD_FORMAT = 2
ARCHITECTURE = dict(phone_channels=128, hidden_channels=128, n_blocks=20)
METADATA_FILE = 'redpanda_embedder.json'
CUSTOM_VERSION = '2.0.0-rc.0-redpanda-embedder'
EXPORT_ROOT = ROOT / 'models' / 'embedders' / 'beatrice'


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(value, temporary)
    os.replace(temporary, path)


def validate_metadata(metadata):
    if metadata.get('format') not in (FORMAT, SAMPLE_LOOKAHEAD_FORMAT) or metadata.get('architecture') != ARCHITECTURE:
        raise ValueError('This is not a supported Beatrice embedder export.')
    delay = metadata.get('phone_delay_frames')
    if isinstance(delay, bool) or not isinstance(delay, int) or delay not in (0, 1, 2):
        raise ValueError('Embedder lookahead must be 0, 10, 20 or 25 extra milliseconds.')
    offset = metadata.get('phone_offset_samples', 0)
    if (isinstance(offset, bool) or not isinstance(offset, int)
            or (delay, offset) not in ((0, 0), (1, 0), (2, 0), (2, 80))):
        raise ValueError('Embedder lookahead must be 0, 10, 20 or 25 extra milliseconds.')
    if offset and metadata['format'] != SAMPLE_LOOKAHEAD_FORMAT:
        raise ValueError('Sample lookahead requires embedder metadata format 2.')
    return dict(metadata, phone_offset_samples=offset)


def lookahead_settings(milliseconds):
    if milliseconds not in LOOKAHEAD_CHOICES:
        raise ValueError('Choose 0, 10, 20 or 25 milliseconds of extra lookahead.')
    delay, remainder = divmod(int(milliseconds), 10)
    return dict(phone_delay_frames=delay, phone_offset_samples=remainder * 16)


def embedder_metadata(delay, offset=0, **details):
    return validate_metadata(dict(format=SAMPLE_LOOKAHEAD_FORMAT if offset else FORMAT, architecture=ARCHITECTURE,
                                  phone_delay_frames=delay, phone_offset_samples=offset, **details))


def shift_phone_input(module, arguments, samples):
    return (F.pad(arguments[0][..., samples:], (0, samples)), *arguments[1:])


@lru_cache(maxsize=1)
def trainer_module():
    from beatrice.compat import patch_torchaudio

    patch_torchaudio()
    source = TRAINER_ROOT / 'beatrice_trainer' / '__main__.py'
    if not source.is_file():
        raise FileNotFoundError('Download Beatrice Trainer before training an embedder.')
    spec = importlib.util.spec_from_file_location('beatrice_embedder_architecture', source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_phone_extractor(factory, delay=0, offset=0):
    embedder_metadata(delay, offset)
    model = factory()
    embed = model.backbone.embed
    padding = (embed.kernel_size[0] - 1) * embed.dilation[0]
    embed.padding = (padding - delay,)
    embed.trim = padding - 2 * delay
    if offset:
        model.feature_extractor.conv0.register_forward_pre_hook(partial(shift_phone_input, samples=offset))
    return model


def load_phone_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Embedder checkpoint not found: {path}')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if 'phone_extractor' not in checkpoint:
        raise ValueError('Select a phone_extractor.pt exported from the Embedder training tab.')
    metadata = validate_metadata(checkpoint.get('embedder_config', {}))
    model = make_phone_extractor(trainer_module().PhoneExtractor, metadata['phone_delay_frames'],
                                metadata['phone_offset_samples'])
    model.load_state_dict(checkpoint['phone_extractor'], strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('The embedder checkpoint contains nonfinite weights.')
    return model, metadata


def export_embedder(model, metadata, output):
    metadata = validate_metadata(dict(metadata))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    atomic_torch_save(output / 'phone_extractor.pt', dict(phone_extractor=state, embedder_config=metadata))
    with torch.random.fork_rng(devices=[]):
        binary = make_phone_extractor(trainer_module().PhoneExtractor, metadata['phone_delay_frames'],
                                     metadata['phone_offset_samples'])
    binary.load_state_dict(state, strict=True)
    binary.eval()
    binary.remove_weight_norm()
    binary.merge_weights()
    temporary = output / 'phone_extractor.bin.tmp'
    binary.half().dump(temporary)
    os.replace(temporary, output / 'phone_extractor.bin')
    atomic_json(output / METADATA_FILE, dict(metadata, phone_extractor_sha256=file_hash(output / 'phone_extractor.bin')))
    return output / 'phone_extractor.pt'


def export_choices(lookahead_ms=None):
    choices = []
    for folder in sorted(EXPORT_ROOT.glob('*')):
        checkpoint = folder / 'phone_extractor.pt'
        if not folder.is_dir() or not checkpoint.is_file():
            continue
        if lookahead_ms is not None:
            try:
                metadata = json.loads((folder / METADATA_FILE).read_text(encoding='utf-8'))
                if not isinstance(metadata, dict):
                    continue
                metadata = validate_metadata(metadata)
            except (OSError, ValueError, TypeError):
                continue
            if metadata['phone_delay_frames'] * 10 + metadata['phone_offset_samples'] / 16 != lookahead_ms:
                continue
        choices.append((folder.name, str(checkpoint)))
    return choices


def require_voice_lookahead(metadata, lookahead_ms):
    if lookahead_ms is None:
        return
    if lookahead_ms not in LOOKAHEAD_CHOICES:
        raise ValueError('Choose 0, 10, 20 or 25 milliseconds of extra lookahead.')
    if (metadata is None or metadata['phone_delay_frames'] * 10 + metadata['phone_offset_samples'] / 16
            != lookahead_ms):
        raise ValueError(f'Select an exported embedder with {lookahead_ms} ms of extra lookahead. '
                         'Changing embedders requires a new Beatrice model name.')


def prepare_voice_embedder(experiment, source=None, lookahead_ms=None):
    experiment = Path(experiment)
    saved_path = experiment / 'beatrice_embedder.json'
    resume = (experiment / 'checkpoint_latest.pt.gz').is_file()
    saved = json.loads(saved_path.read_text(encoding='utf-8')) if saved_path.is_file() else None
    source = str(source or '').strip().strip('"')
    if resume:
        if source and (saved is None or saved.get('source_sha256') != file_hash(source)):
            raise ValueError('Changing embedders requires a new Beatrice model name.')
        if saved is None:
            require_voice_lookahead(None, lookahead_ms)
            return {}
        saved = validate_metadata(saved)
        require_voice_lookahead(saved, lookahead_ms)
        snapshot = experiment / 'beatrice_embedder.pt'
        if not snapshot.is_file() or file_hash(snapshot) != saved['source_sha256']:
            raise ValueError('The saved Beatrice embedder is missing or changed. Restore beatrice_embedder.pt.')
        return dict(phone_extractor_file=str(snapshot.resolve()), phone_delay_frames=saved['phone_delay_frames'],
                    phone_offset_samples=saved['phone_offset_samples'],
                    embedder_sha256=saved['source_sha256'])
    if not source:
        if saved is not None:
            raise ValueError('This Beatrice experiment already selected an embedder. Select it again or use a new name.')
        require_voice_lookahead(None, lookahead_ms)
        return {}
    _, metadata = load_phone_checkpoint(source)
    require_voice_lookahead(metadata, lookahead_ms)
    source_hash = file_hash(source)
    snapshot = experiment / 'beatrice_embedder.pt'
    experiment.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_name(snapshot.name + '.tmp')
    temporary.write_bytes(Path(source).read_bytes())
    os.replace(temporary, snapshot)
    atomic_json(saved_path, dict(metadata, source_sha256=source_hash))
    return dict(phone_extractor_file=str(snapshot.resolve()), phone_delay_frames=metadata['phone_delay_frames'],
                phone_offset_samples=metadata['phone_offset_samples'],
                embedder_sha256=source_hash)


def write_voice_metadata(directory, delay=0, source_hash=None, offset=0):
    if source_hash is None:
        return
    directory = Path(directory)
    metadata = embedder_metadata(delay, offset, source_sha256=source_hash,
                                 phone_extractor_sha256=file_hash(directory / 'phone_extractor.bin'))
    atomic_json(directory / METADATA_FILE, metadata)
    if delay:
        for path in directory.glob('beatrice_paraphernalia_*.toml'):
            source = path.read_text(encoding='utf-8')
            source = source.replace('version = "2.0.0-rc.0"', f'version = "{CUSTOM_VERSION}"', 1)
            path.write_text(source, encoding='utf-8')


def read_voice_metadata(directory, version):
    path = Path(directory) / METADATA_FILE
    if not path.is_file():
        if version == CUSTOM_VERSION:
            raise ValueError('This model needs its redpanda_embedder.json lookahead metadata.')
        return 0, 0
    metadata = validate_metadata(json.loads(path.read_text(encoding='utf-8')))
    if file_hash(Path(directory) / 'phone_extractor.bin') != metadata.get('phone_extractor_sha256'):
        raise ValueError('The phone extractor does not match this model\'s lookahead metadata.')
    delay = metadata['phone_delay_frames']
    if bool(delay) != (version == CUSTOM_VERSION):
        raise ValueError('The Beatrice model version does not match its embedder lookahead.')
    return delay, metadata['phone_offset_samples']
