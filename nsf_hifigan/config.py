import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from rectified_flow.openvpi import DEFAULT_MEL

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ROOT = ROOT / 'models' / 'pretraineds' / 'rectified' / 'trained'
KINDS = {'pc': 'PC-NSF-HiFiGAN', 'nsf': 'NSF-HiFiGAN'}
PITCH_EXTRACTORS = ('parselmouth', 'rmvpe')
PRECISIONS = {'fp32': '32-true', 'fp16': '16-mixed', 'bf16': 'bf16-mixed'}
DISCRIMINATOR_PERIODS = [3, 5, 7, 11, 17, 23, 37]
GENERATOR = dict(
    upsample_initial_channel=512,
    upsample_rates=[8, 8, 2, 2, 2],
    upsample_kernel_sizes=[16, 16, 4, 4, 4],
    resblock='1',
    resblock_kernel_sizes=[3, 7, 11],
    resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    harmonic_num=8,
    noise_sigma=0.0,
)
TRAINING = dict(
    batch_size=10,
    crop_mel_frames=32,
    learning_rate=1e-4,
    finetune_learning_rate=1e-5,
    betas=[0.8, 0.99],
    weight_decay=0.0,
    grad_clip=None,
    mel_loss_weight=45.0,
    pc_wav_loss_weight=30.0,
    pc_aug_rate=0.4,
    pc_aug_key=12.0,
    volume_aug_prob=0.5,
    key_aug=False,
    key_aug_prob=0.5,
    key_aug_min=0.9,
    key_aug_max=1.4,
    discriminator_warmup=0,
    max_updates=100000,
    finetune_max_updates=10000,
    checkpoint_interval=1000,
    eval_interval=1000,
    log_interval=100,
    num_ckpt_keep=5,
    num_valid_plots=10,
    num_workers=4,
    prefetch_factor=2,
)
OVERRIDABLE = ('batch_size', 'crop_mel_frames', 'learning_rate', 'finetune_learning_rate', 'key_aug', 'max_updates',
               'checkpoint_interval', 'eval_interval', 'num_workers')


def experiment_paths(experiment):
    experiment = Path(experiment)
    output = experiment / 'vocoder'
    return dict(experiment=experiment, output=output, data=output / 'data', index=output / 'data' / 'index.json',
                config=output / 'config.json')


def default_data():
    return dict(DEFAULT_MEL)


def default_model(kind, data):
    return dict(GENERATOR, sample_rate=int(data['sample_rate']), num_mels=int(data['n_mels']), mini_nsf=kind == 'pc')


def max_f0(model):
    if model['mini_nsf']:
        return model['sample_rate'] / int(np.prod(model['upsample_rates'][2:])) / 2
    return model['sample_rate'] / 2


def new_config(kind, data, model=None, pretrained=None):
    if kind not in KINDS:
        raise ValueError(f'Vocoder type must be one of {", ".join(KINDS)}.')
    model = deepcopy(model) if model is not None else default_model(kind, data)
    if bool(model['mini_nsf']) != (kind == 'pc'):
        source = 'PC-NSF (mini-NSF)' if model['mini_nsf'] else 'classic NSF'
        raise ValueError(f'The pretrained vocoder is a {source} model. Choose the matching vocoder type.')
    return dict(kind=kind, data=dict(data), model=model, train=deepcopy(TRAINING), pretrained=pretrained)


def apply_overrides(config, overrides):
    for key, value in overrides.items():
        if key not in OVERRIDABLE:
            raise ValueError(f'{key} cannot be overridden.')
        if value is not None:
            config['train'][key] = value
    return config


def validate(config):
    settings = config['train']
    for name in ('batch_size', 'crop_mel_frames', 'max_updates', 'finetune_max_updates', 'checkpoint_interval',
                 'eval_interval', 'log_interval', 'num_ckpt_keep', 'prefetch_factor'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    for name in ('num_workers', 'num_valid_plots', 'discriminator_warmup'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f'{name} must be a nonnegative integer.')
    if not 0 < settings['pc_aug_rate'] <= 1:
        raise ValueError('pc_aug_rate must be in (0, 1].')
    if not 0 < settings['key_aug_min'] <= settings['key_aug_max']:
        raise ValueError('key_aug_min must be positive and not above key_aug_max.')
    if int(np.prod(config['model']['upsample_rates'])) != int(config['data']['hop_length']):
        raise ValueError('The generator upsampling does not match the mel hop length.')
    return config


def openvpi_config(config):
    model, data = config['model'], config['data']
    return dict(
        discriminator_periods=DISCRIMINATOR_PERIODS,
        resblock=model['resblock'],
        resblock_dilation_sizes=model['resblock_dilation_sizes'],
        resblock_kernel_sizes=model['resblock_kernel_sizes'],
        upsample_initial_channel=model['upsample_initial_channel'],
        upsample_kernel_sizes=model['upsample_kernel_sizes'],
        upsample_rates=model['upsample_rates'],
        sampling_rate=data['sample_rate'],
        num_mels=data['n_mels'],
        hop_size=data['hop_length'],
        n_fft=data['n_fft'],
        win_size=data['win_length'],
        fmin=data['mel_fmin'],
        fmax=data['mel_fmax'],
        mini_nsf=model['mini_nsf'],
        noise_sigma=model['noise_sigma'],
        pc_aug=config['kind'] == 'pc',
    )


def read_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)
