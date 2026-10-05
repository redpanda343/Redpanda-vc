import argparse
import json
import os
import random
import re
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

import numpy as np
import torch

from rvc.rectified.augmentation import configure_augmentation
from rvc.rectified.config import compact_config, resolve_config
from rvc.rectified.flow_model import validate_model_config
from rvc.rectified.mel import normalize_mel

ROOT = Path(__file__).resolve().parents[2]
LIGHTNING_DEFAULTS = dict(accelerator='auto', num_nodes=1,
                          strategy={'name': 'auto', 'find_unused_parameters': False},
                          accumulate_grad_batches=1, num_sanity_val_steps=1,
                          max_val_batch_frames=60000, max_val_batch_size=1,
                          sort_by_len=True, sampler_frame_count_grid=6)


def atomic_save(state, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def random_state(device):
    numpy_state = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                cuda=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_random_state(state, device):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    name, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    if device.type == 'cuda' and state.get('cuda') is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


@contextmanager
def evaluation_model(model):
    device = next(model.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    training = model.training
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            yield
        finally:
            model.train(training)


def configure_fused_backbone(model, enabled, device, amp_dtype, max_frames):
    for layer in model.backbone.layers:
        layer.use_fused_kernels = False
    if not enabled:
        return 0
    if any(layer.glu_type != 'softsign_glu' for layer in model.backbone.layers):
        raise ValueError('Fused kernels require SoftSignGLU. Existing ATanGLU experiments must keep this option disabled.')
    if device.type != 'cuda' or amp_dtype is None:
        print('Fused kernels require CUDA FP16 or BF16; using eager SoftSignGLU.', flush=True)
        return 0
    from rvc.rectified.kernels.fused_linear_softsign_glu import fused_supported, is_triton_available

    if not is_triton_available():
        raise RuntimeError('Fused kernels require a working Triton installation. Install Triton for this platform or disable fused kernels.')
    if not fused_supported(device, amp_dtype):
        print('This GPU, dtype or Triton version cannot run fused kernels; using eager SoftSignGLU.', flush=True)
        return 0
    for layer in model.backbone.layers:
        layer.use_fused_kernels = True
    print(f'Enabled DiffSinger fused Linear + SoftSignGLU in {len(model.backbone.layers)} flow blocks. Warming up forward kernels.', flush=True)
    warmup_fused_backbone(model.backbone, device, amp_dtype, max_frames)
    return len(model.backbone.layers)


@torch.no_grad()
def warmup_fused_backbone(backbone, device, dtype, max_frames):
    from rvc.rectified.kernels.fused_linear_softsign_glu import fused_linear_softsign_glu

    if not backbone.layers:
        return
    layer = backbone.layers[0]
    largest_bucket = 1 << (max(1, int(max_frames)) - 1).bit_length()
    bucket = min(2048, largest_bucket)
    with torch.random.fork_rng(devices=[device.index]), torch.cuda.device(device):
        while bucket <= largest_bucket:
            try:
                x = torch.randn(bucket // 2 + 1, layer.up.in_features, device=device, dtype=dtype)
                x = fused_linear_softsign_glu(x, layer.up.weight, layer.up.bias)
                x = fused_linear_softsign_glu(x, layer.mid.weight, layer.mid.bias)
                del x
            except Exception as error:
                print(f'Fused kernel warmup stopped ({error}); remaining kernels compile on first use.', flush=True)
                break
            bucket *= 2
    torch.cuda.empty_cache()


@torch.no_grad()
def preview(model, vocoder, reference, data, writer, step, index=None):
    if reference is None:
        return
    device = next(model.parameters()).device
    mel, content, f0, energy, breathiness, audio, sid, path = reference[:8]
    extras = reference[8:-1] if model.use_phonation else reference[8:]
    variances = dict(zip(("voicing", "tension"), (value.to(device) for value in extras)))
    phonation = reference[-1].to(device) if model.use_phonation else None
    content, f0, energy = content.to(device), f0.to(device), energy.to(device)
    mask = torch.ones(1, 1, f0.shape[1], device=device)
    speaker = torch.tensor([sid], device=device)
    training = model.training
    suffix = '' if index is None else f'/{index}'
    try:
        with evaluation_model(model):
            model.eval()
            generated = model.sample(content, f0, energy, speaker, mask,
                                     breathiness=breathiness.to(device),
                                     source_mel=normalize_mel(mel.to(device), data) if model.val_gt_start else None,
                                     phonation=phonation,
                                     **variances)
            predicted = model.predict_mel(content, f0, energy, speaker, mask,
                                          breathiness=breathiness.to(device), phonation=phonation, **variances) if model.aux is not None else None
    finally:
        model.train(training)
    if not torch.isfinite(generated).all():
        raise FloatingPointError('Non-finite flow preview.')
    writer.add_image(f'mel/flow{suffix}', (generated[0] / 6.0 + 0.5).clamp(0, 1).cpu(), step, dataformats='HW')
    writer.add_image(f'mel/reference{suffix}', (normalize_mel(mel[0], data) / 6.0 + 0.5).clamp(0, 1).cpu(), step, dataformats='HW')
    if predicted is not None:
        if not torch.isfinite(predicted).all():
            raise FloatingPointError('Non-finite direct mel preview.')
        writer.add_image(f'mel/predictor{suffix}', (predicted[0] / 6.0 + 0.5).clamp(0, 1).cpu(), step, dataformats='HW')
    if vocoder is None:
        return
    generated_audio = vocoder(generated, f0)[0]
    rendered_reference = vocoder(normalize_mel(mel.to(device), data), f0)[0]
    previews = [('flow', generated_audio), ('reference', audio[0]), ('vocoder_on_real_mel', rendered_reference)]
    if predicted is not None:
        previews.append(('predictor', vocoder(predicted, f0)[0]))
    for name, value in previews:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite {name} preview.')
        writer.add_audio(f'audio/{name}{suffix}', value.clamp(-1, 1).cpu(), step, data['sample_rate'])


def select_fused_activation(config, enabled):
    if enabled:
        config['flow']['model'].setdefault('backbone_args', {})['glu_type'] = 'softsign_glu'


def load_training_config(experiment, pretrained_flow=None, use_fused_kernels=False, use_phonation=None):
    config_path = experiment / 'rectified_config.json'
    existing = config_path.exists()
    if not existing:
        filename = '44100_finetune.json' if pretrained_flow else '44100_standard.json'
        config_path = ROOT / 'rvc' / 'configs' / 'rectified' / filename
    config = resolve_config(json.loads(config_path.read_text(encoding='utf-8')))
    if pretrained_flow and not existing:
        checkpoint = torch.load(pretrained_flow, map_location='cpu', weights_only=True)
        source = resolve_config(checkpoint.get('config', {}))
        if 'data' not in source or 'flow' not in source or 'model' not in checkpoint:
            raise ValueError('Choose a Rectified Flow voice checkpoint for fine-tuning.')
        config['data'] = source['data']
        config['flow']['model'] = source['flow']['model']
    select_fused_activation(config, use_fused_kernels)
    if use_phonation is not None:
        config['flow']['model']['use_phonation'] = bool(use_phonation)
    configure_flow(config, existing)
    return config


def train(args):
    from rvc.rectified.lightning_train import fit

    if Path(args.model_name).name != args.model_name or args.model_name in {'.', '..'}:
        raise ValueError('Use a model name, not a path.')
    experiment = ROOT / 'logs' / args.model_name
    config = load_training_config(experiment, args.pretrained_flow, getattr(args, 'use_fused_kernels', False),
                                  getattr(args, 'use_phonation', None))
    configure_arguments(args, config['flow'])
    if config['data']['sample_rate'] != 44100:
        raise ValueError('This recipe requires 44100 Hz audio.')
    fit(args, config, ROOT)


def configure_flow(config, existing_config):
    resolved = resolve_config(config)
    config.clear()
    config.update(resolved)
    settings = config['flow']
    settings['model'].setdefault('use_phonation', False)
    settings['model'].setdefault('use_continuous_f0', settings['model'].get('conditioning_version') == 5)
    validate_model_config(settings['model'])
    settings.setdefault('min_learning_rate', 0.0)
    settings.setdefault('max_batch_frames', 50000)
    settings.setdefault('max_batch_size', 64)
    settings.setdefault('dataloader_prefetch_factor', 2)
    settings.setdefault('log_interval', 100)
    for name, value in LIGHTNING_DEFAULTS.items():
        settings.setdefault(name, deepcopy(value))
    reference = settings['model'].get('conditioning_version') == 5
    settings.setdefault('muon_min_fan_in', 0 if reference else 16)
    settings.setdefault('adamw_weight_decay', 0.0)
    for name in ('use_ema', 'ema_decay', 'finetune_ema_decay', 'ema_update_interval'):
        settings.pop(name, None)
    settings.pop('speaker_balanced_sampling', None)
    for name in ('dataloader_prefetch_factor', 'log_interval', 'accumulate_grad_batches', 'num_nodes',
                 'max_val_batch_frames', 'max_val_batch_size', 'sampler_frame_count_grid'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    warmup = settings.get('finetune_warmup_steps', 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('finetune_warmup_steps must be a nonnegative integer.')
    configure_augmentation(settings)
    for name in ('max_updates', 'checkpoint_interval', 'num_ckpt_keep', 'permanent_ckpt_interval'):
        if name in settings and (isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 1):
            raise ValueError(f'{name} must be a positive integer.')
    for name in ('preview_interval', 'eval_interval', 'num_valid_plots', 'permanent_ckpt_start', 'num_sanity_val_steps'):
        if name in settings and (isinstance(settings[name], bool) or not isinstance(settings[name], int) or settings[name] < 0):
            raise ValueError(f'{name} must be a nonnegative integer.')
    if settings.get('precision', 'fp32') not in {'fp32', 'fp16', 'bf16'}:
        raise ValueError('precision must be fp32, fp16 or bf16.')
    if not isinstance(settings['sort_by_len'], bool):
        raise ValueError('sort_by_len must be a boolean.')
    if settings['accelerator'] not in {'auto', 'cpu', 'gpu', 'cuda'}:
        raise ValueError('accelerator must be auto, cpu, gpu or cuda.')
    if not isinstance(settings['strategy'], (str, dict)):
        raise ValueError('strategy must be a name or a configuration object.')
    if not isinstance(settings.get('val_with_vocoder', True), bool):
        raise ValueError('val_with_vocoder must be a boolean.')


def configure_arguments(args, settings):
    for name in ('batch_size', 'max_batch_frames', 'epochs', 'save_every', 'max_updates', 'checkpoint_interval'):
        value = getattr(args, name, None)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
            raise ValueError(f'{name} must be a positive integer.')
    if getattr(args, 'epochs', None) is None:
        args.max_updates = getattr(args, 'max_updates', None) or settings.get('max_updates')
        if args.max_updates is None:
            args.epochs = 100
    args.save_every = getattr(args, 'save_every', None) or 10
    args.precision = getattr(args, 'precision', None) or settings.get('precision', 'fp32')
    args.device = getattr(args, 'device', None) or 'auto'


def prune_checkpoints(output, model_name, settings):
    keep = settings.get('num_ckpt_keep')
    if keep is None:
        return
    start = settings.get('permanent_ckpt_start', 0)
    interval = settings.get('permanent_ckpt_interval', 10000)
    for suffix in (r'_flow_\d+e_(\d+)s\.pth', r'_trainer_(\d+)s\.pth'):
        pattern = re.compile(re.escape(model_name) + suffix)
        checkpoints = []
        for path in output.iterdir():
            match = pattern.fullmatch(path.name)
            if path.is_file() and match:
                checkpoints.append((int(match[1]), path))
        checkpoints.sort(key=lambda item: item[0], reverse=True)
        for step, path in checkpoints[keep:]:
            permanent = start > 0 and step >= start and (step - start) % interval == 0
            if not permanent:
                path.unlink()


def main():
    parser = argparse.ArgumentParser(description='Train Multispeaker Rectified Flow with an optional frozen OpenVPI NSF-HiFiGAN preview vocoder.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--vocoder', default='')
    parser.add_argument('--batch-size', type=int, help='Maximum clips per batch and GPU (default: max_batch_size in the config).')
    parser.add_argument('--max-batch-frames', type=int, help='Maximum padded frames per batch and GPU (default: max_batch_frames in the config).')
    parser.add_argument('--epochs', type=int, help='Optional epoch limit instead of the configured update limit.')
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--checkpoint-interval', type=int)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--device', default=None,
                        help='cpu, one GPU such as cuda:0, or multiple GPUs such as cuda:0,cuda:1')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default=None)
    parser.add_argument('--pretrained-flow')
    phonation = parser.add_mutually_exclusive_group()
    phonation.add_argument('--use-phonation', dest='use_phonation', action='store_true')
    phonation.add_argument('--no-phonation', dest='use_phonation', action='store_false')
    parser.set_defaults(use_phonation=None)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--use-fused-kernels', action='store_true',
                        help='Override the activation with SoftSignGLU and use DiffSinger Triton kernels during CUDA mixed-precision training.')
    parser.add_argument('--fresh', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
