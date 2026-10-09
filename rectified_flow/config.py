import json
from copy import deepcopy
from pathlib import Path


PRESET_DIR = Path(__file__).resolve().parent / 'presets'

PRESETS = ('standard', 'smaller')

FINETUNE_KEY = 'finetune'

FREE_FORM_KEYS = {('flow', 'strategy')}

REMOVED_SHORTCUT_KEYS = ('shortcut', 'shortcut_steps', 'shortcut_bootstrap_every', 'shortcut_ema')


def preset_path(preset):
    if preset not in PRESETS:
        raise ValueError(f'Unknown rectified-flow preset {preset!r}. Choose one of {sorted(PRESETS)}.')
    return PRESET_DIR / f'{preset}.json'


def read_preset(preset):
    path = preset_path(preset)
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        raise ValueError(f'Could not read the rectified-flow preset {path}: {error}') from error
    except json.JSONDecodeError as error:
        raise ValueError(f'The rectified-flow preset {path} is not valid JSON: {error}') from error
    if not isinstance(config, dict):
        raise ValueError(f'The rectified-flow preset {path} must hold a JSON object.')
    return config


def base_config():
    config = read_preset('standard')
    config.pop(FINETUNE_KEY, None)
    return config


def _merge(base, overrides, path=()):
    result = deepcopy(base)
    for key, value in overrides.items():
        if key not in result:
            name = '.'.join((*path, key))
            raise ValueError(f'Unknown rectified-flow setting {name!r}: it is not in {preset_path("standard")}. '
                             'An older version made this experiment or checkpoint, or the preset is missing it.')
        if isinstance(value, dict) and isinstance(result[key], dict) and (*path, key) not in FREE_FORM_KEYS:
            result[key] = _merge(result[key], value, (*path, key))
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(config):
    if not isinstance(config, dict):
        raise ValueError('Rectified-flow config must be a JSON object.')
    flow = config.get('flow')
    model = flow.get('model') if isinstance(flow, dict) else None
    if isinstance(model, dict):
        if model.get('shortcut'):
            raise ValueError('Shortcut (few-step) flows are no longer supported. Retrain this voice as a standard flow.')
        model = {key: value for key, value in model.items() if key not in REMOVED_SHORTCUT_KEYS}
        config = dict(config, flow=dict(flow, model=model))
    return _merge(base_config(), config)


def default_config(finetune=False, preset='standard'):
    config = read_preset(preset)
    overrides = config.pop(FINETUNE_KEY, {})
    config = _merge(base_config(), config)
    return _merge(config, overrides) if finetune else config


def architecture(model):
    return (model['hidden_channels'], model['encoder_layers'],
            model['backbone_args']['channels'], model['backbone_args']['layers'],
            model['aux_decoder']['channels'], model['aux_decoder']['layers'])
