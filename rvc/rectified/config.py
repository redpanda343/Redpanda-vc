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
        'content_interpolation': 'nearest',
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
        'dataloader_prefetch_factor': 2,
        'log_interval': 100,
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

LEGACY_PRESET = deepcopy(STANDARD_PRESET)
STANDARD_PRESET['flow'].update({
    'betas': [0.9, 0.98],
    'adamw_weight_decay': 0.0,
    'muon_min_fan_in': 0,
})
STANDARD_PRESET['flow']['model'].update({
    'conditioning_version': 5,
    'speaker_channels': 384,
    'pitch_fourier': 0,
    'harmonic_prior': False,
    'energy': False,
    'breathiness': False,
    'key_shift': True,
    'speed': True,
    'direct_speaker_conditioning': False,
})
STANDARD_PRESET['flow']['model']['backbone_args']['adaln'] = False
ATAN_PRESET = deepcopy(STANDARD_PRESET)
STANDARD_PRESET['flow']['model']['backbone_args']['glu_type'] = 'softsign_glu'
SOFTSIGN_PRESET = deepcopy(STANDARD_PRESET)
STANDARD_PRESET['flow']['model']['use_spk_id'] = False
NO_SPEAKER_PRESET = deepcopy(STANDARD_PRESET)
STANDARD_PRESET['flow'].update({
    'max_updates': 100000, 'precision': 'fp16', 'devices': 'auto',
    'preview_interval': 4000, 'eval_interval': 4000, 'checkpoint_interval': 4000,
    'num_valid_plots': 10, 'val_with_vocoder': True, 'num_ckpt_keep': 8,
    'permanent_ckpt_start': 60000, 'permanent_ckpt_interval': 10000,
})
STANDARD_PRESET['flow']['model'].update({
    'diffusion_type': 'reflow', 'enc_ffn_kernel_size': 3, 'use_rope': True,
    'rope_interleaved': False, 'rope_theta': 10000.0, 'use_variance_scaling': True,
    'use_shallow_diffusion': True, 't_start_infer': 0.4,
    'train_aux_decoder': True, 'train_diffusion': True, 'val_gt_start': False,
    'aux_decoder_arch': 'convnext',
})
STANDARD_PRESET['flow']['model']['backbone_args'].update({
    'glu_type': 'atanglu', 'dropout_rate': 0.0, 'use_conditioner_cache': True,
})
STANDARD_PRESET['flow']['model']['aux_decoder']['kernel_size'] = 7
PRESET_NAME = 'standard-v5'
PRESETS = {
    'standard-v1': LEGACY_PRESET, 'standard-v2': ATAN_PRESET,
    'standard-v3': SOFTSIGN_PRESET, 'standard-v4': NO_SPEAKER_PRESET, PRESET_NAME: STANDARD_PRESET,
}


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
    if config['preset'] not in PRESETS:
        raise ValueError(f'Unknown rectified-flow preset: {config["preset"]!r}.')
    for key in ('data', 'flow'):
        if key in config and not isinstance(config[key], dict):
            raise ValueError(f'Rectified-flow {key} settings must be a JSON object.')
    return _merge(PRESETS[config['preset']], {key: value for key, value in config.items() if key != 'preset'})


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
    visible = {key: deepcopy(value) for key, value in flow.items() if key != 'model'}
    visible['model'] = deepcopy(flow['model'])
    compact['data'] = deepcopy(resolved['data'])
    compact['flow'] = _merge(visible, compact.get('flow', {}))
    return compact
