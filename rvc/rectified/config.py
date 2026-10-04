from copy import deepcopy


STANDARD_PRESET = {
    'data': {
        'sample_rate': 44100,
        'hop_length': 512,
        'n_fft': 2048,
        'win_length': 2048,
        'n_mels': 128,
        'mel_fmin': 40.0,
        'mel_fmax': 16000.0,
        'mel_mean': -6.0,
        'mel_std': 6.0,
        'content_interpolation': 'linear',
    },
    'flow': {
        'optimizer': 'muon',
        'learning_rate': 0.0006,
        'min_learning_rate': 0.0,
        'finetune_learning_rate': 0.0001,
        'lr_final_ratio': 0.1,
        'lr_schedule': 'step',
        'decay_step': 5000,
        'gamma': 0.8,
        'step_lr_offset': 0,
        'betas': [0.9, 0.999],
        'weight_decay': 0.1,
        'warmup_steps': 0,
        'max_batch_frames': 50000,
        'max_batch_size': 64,
        'grad_clip': 1.0,
        'ema_decay': 0.9999,
        'finetune_ema_decay': 0.999,
        'speaker_dropout': 0.0,
        'augmentation_args': {
            'random_pitch_shifting': {'enabled': True, 'range': [-5.0, 5.0], 'scale': 0.75},
            'fixed_pitch_shifting': {'enabled': False, 'targets': [-5.0, 5.0], 'scale': 0.5},
            'random_time_stretching': {'enabled': True, 'range': [0.5, 2.0], 'scale': 0.75},
        },
        'preview_interval': 1000,
        'eval_interval': 1000,
        'holdout_clips': 32,
        'num_workers': 4,
        'augmentation_workers': 4,
        'model': {
            'dual_timestep': True,
            'voicing': False,
            'tension': False,
            'sampling_method': 'euler',
            'sampling_steps': 20,
            'flow_conditioning': 'encoder',
            'flow_loss': 'l2',
            'content_channels': 768,
            'hidden_channels': 384,
            'encoder_layers': 4,
            'speaker_channels': 256,
            'pitch_fourier': 6,
            'harmonic_prior': True,
            'breathiness': True,
            'key_shift': True,
            'speed': True,
            'backbone': 'lynxnet2',
            'backbone_args': {
                'channels': 1024, 'layers': 6, 'expansion': 1,
                'kernel_size': 31, 'adaln': True, 'time_scale': 1000.0,
            },
            't_start': 0.4,
            'conditioning_version': 4,
            'direct_speaker_conditioning': True,
            'aux_grad': 0.1,
            'aux_decoder': {'channels': 512, 'layers': 6, 'dropout': 0.1},
        },
        'finetune_preview_interval': 500,
        'aux_mel_weight': 0.2,
    },
}

PRESET_NAME = 'standard-v1'
EDITABLE_FLOW_KEYS = (
    'learning_rate', 'decay_step', 'gamma', 'max_batch_frames', 'max_batch_size',
    'num_workers', 'preview_interval', 'eval_interval', 'holdout_clips',
)


def _merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(config):
    if not isinstance(config, dict):
        raise ValueError('Rectified-flow config must be a JSON object.')
    if 'preset' not in config:
        return deepcopy(config)
    if config['preset'] != PRESET_NAME:
        raise ValueError(f'Unknown rectified-flow preset: {config["preset"]!r}.')
    for key in ('data', 'flow'):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f'Rectified-flow {key} settings must be a JSON object.')
    return _merge(STANDARD_PRESET, {key: value for key, value in config.items() if key != 'preset'})


def _differences(config, defaults):
    result = {}
    for key, value in config.items():
        if key not in defaults or value != defaults[key]:
            if isinstance(value, dict) and isinstance(defaults.get(key), dict):
                result[key] = _differences(value, defaults[key])
            else:
                result[key] = deepcopy(value)
    return result


def compact_config(config):
    resolved = resolve_config(config)
    compact = {'preset': PRESET_NAME, **_differences(resolved, STANDARD_PRESET)}
    if resolve_config(compact) != resolved:
        return resolved
    flow = resolved['flow']
    visible = {key: deepcopy(flow[key]) for key in EDITABLE_FLOW_KEYS}
    if 'finetune_warmup_steps' in flow:
        visible.update({key: deepcopy(flow[key]) for key in (
            'finetune_learning_rate', 'min_learning_rate', 'finetune_warmup_steps',
            'finetune_ema_decay',
            'finetune_preview_interval',
        ) if key in flow})
    visible['augmentation_args'] = {
        key: deepcopy(flow['augmentation_args'][key])
        for key in ('random_pitch_shifting', 'random_time_stretching')
    }
    visible['model'] = {key: flow['model'][key] for key in ('sampling_method', 'sampling_steps')}
    compact['flow'] = _merge(visible, compact.get('flow', {}))
    return compact
