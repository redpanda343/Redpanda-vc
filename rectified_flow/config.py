import json
from copy import deepcopy
from pathlib import Path


PRESET_DIR = Path(__file__).resolve().parent / 'presets'
MEL_PRESET_DIR = PRESET_DIR / 'mel'

PRESETS = ('standard', 'smaller', 'meanflow', 'small_dataset')

MEL_PRESETS = ('openvpi_44khz', 'vocos_44khz', 'vocos_24khz', 'pupu_44khz')

FINETUNE_KEY = 'finetune'

FREE_FORM_KEYS = {('flow', 'strategy')}

REMOVED_MODEL_KEYS = ('shortcut', 'shortcut_steps', 'shortcut_bootstrap_every', 'shortcut_ema', 'use_f0')


def preset_path(preset):
    if preset not in PRESETS:
        raise ValueError(f'Unknown rectified-flow preset {preset!r}. Choose one of {sorted(PRESETS)}.')
    return PRESET_DIR / f'{preset}.json'


def mel_preset_path(mel):
    if mel not in MEL_PRESETS:
        raise ValueError(f'Unknown mel preset {mel!r}. Choose one of {sorted(MEL_PRESETS)}.')
    return MEL_PRESET_DIR / f'{mel}.json'


def _read_json(path):
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except OSError as error:
        raise ValueError(f'Could not read the rectified-flow preset {path}: {error}') from error
    except json.JSONDecodeError as error:
        raise ValueError(f'The rectified-flow preset {path} is not valid JSON: {error}') from error
    if not isinstance(config, dict):
        raise ValueError(f'The rectified-flow preset {path} must hold a JSON object.')
    return config


def read_preset(preset):
    return _read_json(preset_path(preset))


def mel_preset(mel):
    data = _read_json(mel_preset_path(mel)).get('data')
    if not isinstance(data, dict):
        raise ValueError(f'The mel preset {mel_preset_path(mel)} must hold a data object.')
    return data


def base_config():
    config = read_preset('standard')
    config.pop(FINETUNE_KEY, None)
    config['flow']['model'].update(mean_flow=False, mean_flow_args=dict(
        flow_ratio=0.75, time_mu=-0.4, time_sigma=1.0, cfg_ratio=0.2,
        cfg_scale=2.0, loss_p=0.5, loss_eps=1e-3,
    ), source_filter=False, source_filter_args=dict(envelope_coefficients=20, aperiodicity_coefficients=8))
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
        if model.get('use_f0') is False:
            raise ValueError('F0-free flows are no longer supported. Retrain this voice with F0.')
        model = {key: value for key, value in model.items() if key not in REMOVED_MODEL_KEYS}
        model.setdefault('use_tension_embed', False)
        config = dict(config, flow=dict(flow, model=model))
    return _merge(base_config(), config)


def default_config(finetune=False, preset='standard', mel=None):
    config = read_preset(preset)
    overrides = config.pop(FINETUNE_KEY, {})
    config = _merge(base_config(), config)
    if mel:
        config = _merge(config, {'data': mel_preset(mel)})
    return _merge(config, overrides) if finetune else config


def architecture(model):
    size = (model.get('mean_flow', False), model.get('source_filter', False), model['hidden_channels'],
            model['encoder_layers'], model['backbone_args']['channels'], model['backbone_args']['layers'])
    return size if model.get('mean_flow', False) else size + (
        model['aux_decoder']['channels'], model['aux_decoder']['layers'],
    )
