from copy import deepcopy


DEFAULT_CONFIG = {
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
    },
    'flow': {
        'optimizer': 'muon',
        'learning_rate': 0.0006,
        'min_learning_rate': 0.0,
        'finetune_learning_rate': 0.0001,
        'finetune_warmup_steps': 0,
        'lr_final_ratio': 0.1,
        'lr_schedule': 'step',
        'decay_step': 5000,
        'gamma': 0.8,
        'step_lr_offset': 0,
        'betas': [0.9, 0.999],
        'weight_decay': 0.1,
        'adamw_weight_decay': 0.0,
        'warmup_steps': 0,
        'max_batch_frames': 50000,
        'max_batch_size': 64,
        'grad_clip': 1.0,
        'pitch_extractor': 'parselmouth',
        'hnsep': 'vr',
        'breathiness_smooth_width': 0.06,
        'voicing_smooth_width': 0.06,
        'aux_mel_weight': 0.2,
        'augmentation_args': {
            'random_pitch_shifting': {'enabled': True, 'range': [-5.0, 5.0], 'scale': 0.75},
            'random_time_stretching': {'enabled': True, 'range': [0.5, 2.0], 'scale': 0.75},
        },
        'augmentation_workers': 0,
        'holdout_clips': 32,
        'num_workers': 4,
        'dataloader_prefetch_factor': 2,
        'log_interval': 100,
        'max_updates': 100000,
        'precision': 'fp16',
        'devices': 'auto',
        'accelerator': 'auto',
        'num_nodes': 1,
        'strategy': {'name': 'auto', 'find_unused_parameters': False},
        'accumulate_grad_batches': 1,
        'num_sanity_val_steps': 1,
        'eval_interval': 4000,
        'preview_interval': 4000,
        'finetune_preview_interval': 500,
        'checkpoint_interval': 4000,
        'num_valid_plots': 10,
        'val_with_vocoder': True,
        'num_ckpt_keep': 8,
        'permanent_ckpt_start': 60000,
        'permanent_ckpt_interval': 10000,
        'max_val_batch_frames': 60000,
        'max_val_batch_size': 1,
        'sort_by_len': True,
        'sampler_frame_count_grid': 6,
        'model': {
            'content_channels': 768,
            'hidden_channels': 384,
            'encoder_layers': 4,
            'enc_ffn_kernel_size': 3,
            'use_rope': True,
            'rope_interleaved': False,
            'rope_theta': 10000.0,
            'use_spk_id': False,
            'key_shift': True,
            'speed': True,
            'use_breathiness_embed': False,
            'use_voicing_embed': False,
            'backbone_args': {
                'channels': 1024,
                'layers': 6,
                'expansion': 1,
                'kernel_size': 31,
                'glu_type': 'atanglu',
                'dropout_rate': 0.0,
                'use_conditioner_cache': True,
            },
            'aux_decoder': {'channels': 512, 'layers': 6, 'dropout': 0.1, 'kernel_size': 7},
            'aux_grad': 0.1,
            't_start': 0.4,
            't_start_infer': 0.4,
            'dual_timestep': True,
            'sampling_method': 'rk2',
            'sampling_steps': 10,
            'shortcut': False,
            'shortcut_steps': 128,
            'shortcut_bootstrap_every': 4,
            'shortcut_ema': 0.999,
            'shortcut_ema_export': True,
            'train_aux_decoder': True,
            'train_diffusion': True,
            'val_gt_start': False,
        },
    },
}

NEW_EXPERIMENT_OVERRIDES = {
    'flow': {
        'model': {
            'use_breathiness_embed': True,
            'use_voicing_embed': True,
        },
    },
}

SHORTCUT_OVERRIDES = {
    'flow': {
        'model': {
            'shortcut': True,
            'sampling_method': 'euler',
            'sampling_steps': 8,
        },
    },
}

FINETUNE_OVERRIDES = {
    'flow': {
        'finetune_learning_rate': 5e-05,
        'finetune_warmup_steps': 500,
        'min_learning_rate': 1e-05,
        'decay_step': 2000,
        'eval_interval': 500,
        'max_batch_size': 32,
        'augmentation_args': {
            'random_pitch_shifting': {'enabled': False},
            'random_time_stretching': {'enabled': False},
        },
    },
}

REALTIME_OVERRIDES = {
    'flow': {
        'model': {
            'hidden_channels': 256,
            'encoder_layers': 6,
            'backbone_args': {'channels': 512, 'layers': 12},
            'aux_decoder': {'channels': 384, 'layers': 8},
        },
    },
}

PRESETS = {'standard': {}, 'realtime': REALTIME_OVERRIDES}

FREE_FORM_KEYS = {('flow', 'strategy')}


def _merge(base, overrides, path=()):
    result = deepcopy(base)
    for key, value in overrides.items():
        if key not in result:
            name = '.'.join((*path, key))
            raise ValueError(f'Unknown rectified-flow setting {name!r}. This experiment or checkpoint was made by '
                             'an older version; start a new experiment.')
        if isinstance(value, dict) and isinstance(result[key], dict) and (*path, key) not in FREE_FORM_KEYS:
            result[key] = _merge(result[key], value, (*path, key))
        else:
            result[key] = deepcopy(value)
    return result


def resolve_config(config):
    if not isinstance(config, dict):
        raise ValueError('Rectified-flow config must be a JSON object.')
    return _merge(DEFAULT_CONFIG, config)


def default_config(finetune=False, preset='standard'):
    if preset not in PRESETS:
        raise ValueError(f'Unknown rectified-flow preset {preset!r}. Choose one of {sorted(PRESETS)}.')
    config = _merge(_merge(DEFAULT_CONFIG, NEW_EXPERIMENT_OVERRIDES), PRESETS[preset])
    return _merge(config, FINETUNE_OVERRIDES) if finetune else config


def architecture(model):
    return (model['hidden_channels'], model['encoder_layers'],
            model['backbone_args']['channels'], model['backbone_args']['layers'],
            model['aux_decoder']['channels'], model['aux_decoder']['layers'])
