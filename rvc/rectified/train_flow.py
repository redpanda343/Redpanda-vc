import argparse
import importlib
import json
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch._functorch import config as aot_config
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rvc.rectified.augmentation import configure_augmentation, is_augmented, prepare_augmentation
from rvc.rectified.config import compact_config, resolve_config
from rvc.rectified.data import FlowBatchSampler, RectifiedDataset, collate_flow, read_filelist, speaker_inventory, split_holdout, unpack_flow
from rvc.rectified.distributed import launch
from rvc.rectified.ema import WeightEMA
from rvc.rectified.flow_model import build_flow, resize_speakers, validate_model_config
from rvc.rectified.mel import normalize_mel
from rvc.rectified.muon import MuonAdamW
from rvc.rectified.schedule import learning_rate
from rvc.rectified.vocoder import load_vocoder

ROOT = Path(__file__).resolve().parents[2]


def atomic_save(state, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def precision_setup(precision, device):
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise ValueError(f'Unsupported precision: {precision}')
    if device.type != 'cuda':
        if precision != 'fp32':
            print(f'{precision.upper()} requires CUDA in this trainer; using FP32.', flush=True)
        return None, None
    if precision == 'bf16':
        with torch.cuda.device(device):
            supported = torch.cuda.is_bf16_supported(including_emulation=False)
        if not supported:
            print('BF16 is not supported on this GPU; using FP32.', flush=True)
            return None, None
        return torch.bfloat16, None
    if precision == 'fp16':
        return torch.float16, torch.amp.GradScaler('cuda')
    return None, None


def conditioning_norms(model) -> dict:
    backbone = model.backbone
    norms = {"diag/time_mlp_norm": sum(p.norm().item() ** 2 for p in backbone.time_mlp.parameters()) ** 0.5}
    modulation = [layer.modulation.weight.norm().item()
                  for layer in backbone.layers if layer.modulation is not None]
    if modulation:
        norms["diag/adaln_norm_max"] = max(modulation)
        norms["diag/adaln_norm_mean"] = sum(modulation) / len(modulation)
    if backbone.voice is not None:
        norms["diag/voice_proj_norm"] = backbone.voice.weight.norm().item()
    return norms


def compiled_backbone(model, enabled: bool, mode: str, device):
    if not enabled:
        return None
    if sys.platform != "linux" or device.type != "cuda" or not torch.cuda.is_available():
        print("Flow backbone compilation requires Linux and CUDA; training uncompiled.", flush=True)
        return None
    if not torch.version.hip and torch.cuda.get_device_capability(device) < (8, 0):
        print("Triton 3.6 requires NVIDIA compute capability 8.0 or newer; training uncompiled.", flush=True)
        return None
    try:
        importlib.import_module("triton")
    except (ImportError, OSError) as error:
        print(f"Triton is unavailable ({error}); training uncompiled.", flush=True)
        return None
    print(f"Flow backbone compilation requested (Inductor, {mode}). The first training step compiles forward and backward graphs.", flush=True)
    return torch.compile(model.backbone, backend="inductor", mode=mode, dynamic=True)


def train_step(model, optimizer, ema, batch, data, settings, device, speaker_dropout, amp_dtype=None, scaler=None, backbone=None, ranks=None):
    mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask, voicing, tension = unpack_flow(batch, device, True)
    mel = normalize_mel(mel, data) * mask
    optimizer.zero_grad(set_to_none=True)
    rng_state = torch.get_rng_state() if amp_dtype == torch.float16 else None
    cuda_rng_state = torch.cuda.get_rng_state(device) if rng_state is not None and device.type == 'cuda' else None

    def forward_loss(dtype):
        backward_context = aot_config.patch(backward_pass_autocast="off") if backbone is not None else nullcontext()
        with backward_context, torch.autocast(device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
            result = model(
                mel, content, f0, energy, speaker, mask,
                speaker_dropout=speaker_dropout, breathiness=breathiness,
                key_shift=key_shift, speed=speed, backbone=backbone,
                voicing=voicing, tension=tension,
            )
            flow, auxiliary = result
            loss = flow if auxiliary is None else flow + settings['aux_mel_weight'] * auxiliary
        return flow, auxiliary, loss

    flow, auxiliary, loss = forward_loss(amp_dtype)
    finite = bool(torch.isfinite(loss))
    if ranks is not None:
        finite = ranks.all_true(finite)
    retried = not finite and amp_dtype == torch.float16
    if retried:
        del flow, auxiliary, loss
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
        if ranks is None or ranks.main:
            print('FP16 forward overflow: retrying this batch in FP32 on all training ranks.', flush=True)
        flow, auxiliary, loss = forward_loss(None)
        finite = bool(torch.isfinite(loss))
        if ranks is not None:
            finite = ranks.all_true(finite)
    if not finite:
        inputs = dict(mel=mel, content=content, f0=f0, energy=energy, breathiness=breathiness,
                      key_shift=key_shift, speed=speed, mask=mask)
        inputs.update({name: value for name, value in (("voicing", voicing), ("tension", tension)) if value is not None})
        invalid = [not bool(torch.isfinite(value).all()) for value in inputs.values()]
        invalid.append(any(not bool(torch.isfinite(param).all()) for param in model.parameters()))
        invalid = torch.tensor(invalid, device=device, dtype=torch.float32)
        if ranks is not None and ranks.world > 1:
            invalid = ranks.sum(invalid)
        names = list(inputs) + ['model parameters']
        details = ', '.join(name for name, flag in zip(names, invalid.tolist()) if flag) or 'none in inputs or model parameters'
        retry = ' after an FP32 retry' if retried else ''
        raise FloatingPointError(f'Non-finite rectified-flow loss{retry}. Non-finite values: {details}.')
    frames = mask.sum().detach()
    padded_frames = frames.new_tensor(mask.numel())
    if ranks is not None and ranks.world > 1:
        loss = flow * (padded_frames * ranks.world / ranks.sum(padded_frames).clamp_min(1.0))
        if auxiliary is not None:
            loss = loss + settings['aux_mel_weight'] * auxiliary * (frames * ranks.world / ranks.sum(frames).clamp_min(1.0))
    if scaler is None:
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings['grad_clip'], error_if_nonfinite=True
        )
        optimizer.step()
        ema.update(model)
    else:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'])
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= previous_scale:
            ema.update(model)
        else:
            print('FP16 overflow: skipped optimizer and EMA update; reduced gradient scale.', flush=True)
    if ranks is not None and ranks.world > 1:
        stats = torch.stack([flow.detach() * padded_frames,
                             auxiliary.detach() * frames if auxiliary is not None else frames * 0,
                             padded_frames, frames])
        stats = ranks.sum(stats)
        return float(stats[0] / stats[2].clamp_min(1.0)), float(stats[1] / stats[3].clamp_min(1.0)), float(norm)
    return float(flow.detach()), float(auxiliary.detach()) if auxiliary is not None else 0.0, float(norm)


@torch.no_grad()
def preview(model, ema, vocoder, reference, data, writer, step):
    if reference is None:
        return
    device = next(model.parameters()).device
    mel, content, f0, energy, breathiness, audio, sid, path = reference[:8]
    variances = dict(zip(("voicing", "tension"), (value.to(device) for value in reference[8:])))
    content, f0, energy = content.to(device), f0.to(device), energy.to(device)
    mask = torch.ones(1, 1, f0.shape[1], device=device)
    speaker = torch.tensor([sid], device=device)
    training = model.training
    try:
        with ema.applied(model):
            model.eval()
            generated = model.sample(content, f0, energy, speaker, mask,
                                     breathiness=breathiness.to(device), **variances)
            predicted = model.predict_mel(content, f0, energy, speaker, mask,
                                          breathiness=breathiness.to(device), **variances) if model.aux is not None else None
    finally:
        model.train(training)
    if not torch.isfinite(generated).all():
        raise FloatingPointError('Non-finite flow preview.')
    writer.add_image('mel/flow', (generated[0] / 6.0 + 0.5).clamp(0, 1).cpu(), step, dataformats='HW')
    writer.add_image('mel/reference', (normalize_mel(mel[0], data) / 6.0 + 0.5).clamp(0, 1).cpu(), step, dataformats='HW')
    if predicted is not None:
        if not torch.isfinite(predicted).all():
            raise FloatingPointError('Non-finite direct mel preview.')
        writer.add_image('mel/predictor', (predicted[0] / 6.0 + 0.5).clamp(0, 1).cpu(), step, dataformats='HW')
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
        writer.add_audio(f'audio/{name}', value.clamp(-1, 1).cpu(), step, data['sample_rate'])


@torch.no_grad()
def evaluate(model, ema, loader, data, writer, step):
    device = next(model.parameters()).device
    fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
    totals = torch.zeros(len(fractions), device=device)
    aux_total, count, aux_count = 0.0, 0, 0
    with ema.applied(model):
        model.eval()
        for index, batch in enumerate(loader):
            mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask, voicing, tension = unpack_flow(batch, device)
            mel = normalize_mel(mel, data) * mask
            generator = torch.Generator(device=device).manual_seed(index)
            noise = torch.randn(mel.shape, device=device, generator=generator)
            losses, auxiliary = model.validation_losses(
                mel, content, f0, energy, speaker, mask, breathiness, key_shift, speed,
                noise, fractions,
                voicing=voicing, tension=tension,
            )
            weight = mask.numel()
            aux_weight = float(mask.sum())
            totals += losses.float() * weight
            aux_total += (float(auxiliary) if auxiliary is not None else 0.0) * aux_weight
            count += weight
            aux_count += aux_weight
        model.train()
    totals /= max(1, count)
    writer.add_scalar('val/flow', float(totals.mean()), step)
    for fraction, value in zip(fractions, totals.tolist()):
        writer.add_scalar(f'val/flow_t{fraction:g}', value, step)
    if model.aux is not None:
        writer.add_scalar(f'val/{model.aux_loss_name}', aux_total / max(1, aux_count), step)


def load_training_config(experiment, pretrained_flow=None):
    config_path = experiment / 'rectified_config.json'
    existing = config_path.exists()
    if not existing:
        filename = '44100_finetune.json' if pretrained_flow else '44100_standard.json'
        config_path = ROOT / 'rvc' / 'configs' / 'rectified' / filename
    config = json.loads(config_path.read_text(encoding='utf-8'))
    if pretrained_flow and not existing:
        checkpoint = torch.load(pretrained_flow, map_location='cpu', weights_only=True)
        source = resolve_config(checkpoint.get('config', {}))
        if 'data' not in source or 'flow' not in source or 'model' not in checkpoint:
            raise ValueError('Choose a Rectified Flow voice checkpoint for fine-tuning.')
        config['data'] = source['data']
        config['flow']['model'] = source['flow']['model']
    configure_flow(config, existing)
    return config


def train(args):
    if Path(args.model_name).name != args.model_name or args.model_name in {'.', '..'}:
        raise ValueError('Use a model name, not a path.')
    if any(value is not None and value < 1 for value in (args.batch_size, args.max_batch_frames)) or args.epochs < 1 or args.save_every < 1:
        raise ValueError('Batch limits, epochs and save interval must be positive.')
    experiment = ROOT / 'logs' / args.model_name
    config = load_training_config(experiment, args.pretrained_flow)
    if config['data']['sample_rate'] != 44100:
        raise ValueError('This recipe requires 44100 Hz audio.')
    originals = read_filelist(experiment / 'filelist.txt', ROOT, originals_only=True)
    multispeaker = config['flow']['model'].get('conditioning_version', 1) in (2, 3, 4)
    if multispeaker:
        speaker_inventory(originals)
    entries, _ = split_holdout(originals, int(config['flow'].get('holdout_clips', 0)), stratified=multispeaker)
    from rvc.rectified.distributed import parse_devices

    devices = parse_devices(args.device)
    device = torch.device(devices[0])
    if device.type == 'cuda' and (not torch.cuda.is_available() or any(int(value[5:]) >= torch.cuda.device_count() for value in devices)):
        raise ValueError('A selected CUDA device is unavailable.')
    prepare_augmentation(experiment, ROOT, originals, entries, config, args.seed, device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    launch(train_rank, args)


def configure_flow(config, existing_config):
    resolved = resolve_config(config)
    config.clear()
    config.update(resolved)
    settings = config['flow']
    validate_model_config(settings['model'])
    settings.setdefault('min_learning_rate', 0.0)
    settings.setdefault('max_batch_frames', 50000)
    settings.setdefault('max_batch_size', 64)
    settings.pop('speaker_balanced_sampling', None)
    warmup = settings.get('finetune_warmup_steps', 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError('finetune_warmup_steps must be a nonnegative integer.')
    configure_augmentation(settings)


def train_rank(args, ranks):
    if any(value is not None and value < 1 for value in (args.batch_size, args.max_batch_frames)) or args.epochs < 1 or args.save_every < 1:
        raise ValueError('Batch limits, epochs and save interval must be positive.')
    if Path(args.model_name).name != args.model_name or args.model_name in {'.', '..'}:
        raise ValueError('Use a model name, not a path.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = ROOT / 'logs' / args.model_name
    config = load_training_config(experiment, args.pretrained_flow)
    validate_model_config(config['flow']['model'])
    settings, data = config['flow'], config['data']
    multispeaker = settings['model'].get('conditioning_version', 1) in (2, 3, 4)
    if data['sample_rate'] != 44100:
        raise ValueError('This recipe requires 44100 Hz audio and a matching NSF-HiFiGAN vocoder.')
    device = ranks.device
    amp_dtype, scaler = precision_setup(args.precision, device)
    if ranks.world > 1 and args.precision == 'bf16' and not ranks.all_true(amp_dtype == torch.bfloat16):
        amp_dtype, scaler = None, None
        if ranks.main:
            print('At least one selected GPU cannot use BF16; all ranks will use FP32.', flush=True)
    precision_label = str(amp_dtype).split(".")[-1].upper() if amp_dtype is not None else "FP32"
    vocoder = None
    with ranks.main_work('preview vocoder loading') as main:
        if main:
            if args.vocoder:
                vocoder, _ = load_vocoder(args.vocoder, data)
                vocoder = vocoder.to(device)
            else:
                print('No vocoder selected: previews will show mel images only.', flush=True)
    entries = read_filelist(experiment / 'filelist.txt', ROOT)
    originals = [entry for entry in entries if not is_augmented(entry)]
    inventory = speaker_inventory(originals) if multispeaker else None
    speakers = max(int(entry[4]) for entry in entries) + 1
    originals, held = split_holdout(originals, int(settings.get('holdout_clips', 0)), stratified=multispeaker)
    entries = originals + [entry for entry in entries if is_augmented(entry)]
    max_items = int(args.batch_size or settings['max_batch_size'])
    max_frames = int(args.max_batch_frames or settings['max_batch_frames'])
    dataset = RectifiedDataset(entries, config, max_frames)
    workers = int(settings.get('num_workers', 4))
    if ranks.main:
        print(f'Whole-utterance batching: up to {max_items} clips and {max_frames} padded frames per batch and GPU. Measuring {len(entries):,} clip lengths...', flush=True)
    batcher = FlowBatchSampler(dataset, max_frames, max_items, args.seed, ranks.rank, ranks.world)
    loader = DataLoader(dataset, batch_sampler=batcher,
                        num_workers=workers, collate_fn=collate_flow,
                        pin_memory=device.type == 'cuda', persistent_workers=workers > 0,
                        multiprocessing_context='spawn' if workers > 0 else None)
    held_dataset = RectifiedDataset(held, config, max_frames, augment=False)
    held_loader = DataLoader(held_dataset,
                             batch_sampler=FlowBatchSampler(held_dataset, max_frames, max_items, args.seed, shuffle=False),
                             collate_fn=collate_flow) if held and ranks.main else None
    reference = None
    with ranks.main_work('preview reference preparation') as main:
        if main:
            reference = dataset.reference()
    info_path = experiment / 'model_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
    embedder = info.get('embedder_model', 'contentvec')
    feature_metadata = {key: info[key] for key in ('embedder_model', 'version', 'feature_dim', 'feature_output', 'feature_fingerprint') if key in info}
    if multispeaker and (info.get('version', 'v2') != 'v2' or int(info.get('feature_dim', settings['model']['content_channels'])) != settings['model']['content_channels']):
        raise ValueError('Extraction metadata does not match the content encoder. Re-extract v2 features.')
    model = build_flow(config, speakers).to(device).float()
    output = experiment / 'flow'
    resume_path = output / 'checkpoint.pth'
    state = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path.exists() and not args.fresh else None
    finetune = bool(state.get('finetune', False)) if state else bool(args.pretrained_flow)
    dropout = float(settings['speaker_dropout'])
    lr = args.learning_rate or settings['finetune_learning_rate' if finetune else 'learning_rate']
    if settings['optimizer'] == 'muon':
        optimizer = MuonAdamW(model, lr, muon_weight_decay=settings['weight_decay'],
                              betas=tuple(settings['betas']), iteration_dtype=amp_dtype or torch.float32)
    elif settings['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=tuple(settings['betas']), weight_decay=settings['weight_decay'])
    else:
        raise ValueError('Optimizer must be muon or adamw.')
    decay = settings.get('finetune_ema_decay', settings['ema_decay']) if finetune else settings['ema_decay']
    ema = WeightEMA(model, decay)
    first_epoch, step = 1, 0
    if state:
        configure_flow(state['config'], True)
        if multispeaker and (state.get('speaker_ids') != sorted(inventory) or state.get('feature_metadata') != feature_metadata):
            raise ValueError('Speaker IDs or extracted feature metadata changed. Use a new experiment.')
        if state['config'] != config or state['embedder_model'] != embedder:
            raise ValueError('Resume config or embedder differs from the saved checkpoint.')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'], model)
        if scaler is not None and state.get('scaler'):
            scaler.load_state_dict(state['scaler'])
        first_epoch, step = state['epoch'] + 1, state['step']
        del state
    elif args.pretrained_flow:
        state = torch.load(args.pretrained_flow, map_location='cpu', weights_only=True)
        state['config'] = resolve_config(state.get('config', {}))
        if state.get('embedder_model', embedder) != embedder:
            raise ValueError('Pretrained flow uses a different content embedder.')
        pretrained_model = state.get('config', {}).get('flow', {}).get('model', {})
        validate_model_config(pretrained_model)
        if pretrained_model.get('flow_conditioning', 'encoder') != settings['model'].get('flow_conditioning', 'encoder'):
            raise ValueError('Pretrained flow uses different conditioning. Use a pretrained with the same conditioning path or start a new model from scratch.')
        if pretrained_model.get('conditioning_version', 1) != settings['model'].get('conditioning_version', 1):
            raise ValueError('Pretrained conditioning version differs. Start a new model or use a matching pretrained.')
        for name in ('voicing', 'tension', 'direct_speaker_conditioning'):
            if bool(pretrained_model.get(name, False)) != bool(settings['model'].get(name, False)):
                raise ValueError(f'Pretrained flow differs in {name}. Use a matching pretrained or train a new model from scratch.')
        source_scale = float(pretrained_model.get('backbone_args', {}).get('time_scale', 1000.0))
        target_scale = float(settings['model'].get('backbone_args', {}).get('time_scale', 1000.0))
        if source_scale != target_scale:
            raise ValueError('Pretrained flow uses a different time embedding scale. Use a matching pretrained.')
        pretrained_data = state.get('config', {}).get('data', {})
        for name in ('mel_mean', 'mel_std'):
            if name in pretrained_data and float(pretrained_data[name]) != float(data[name]):
                raise ValueError(
                    f'Pretrained flow uses different mel normalization ({name}={pretrained_data[name]} vs {data[name]}). '
                    'Use a matching pretrained or start from scratch.'
                )
        weights = state['ema']['shadow'] if state.get('ema') else state['model']
        speaker_init = model.encoder.speaker.weight if multispeaker else None
        model.load_state_dict(resize_speakers(
            weights, speakers, speaker_init, null_speaker=model.encoder.has_null_speaker,
        ), strict=True)
        ema.reseed(model)
        del state, weights
    with ranks.main_work('training output setup') as main:
        if main:
            output.mkdir(parents=True, exist_ok=True)
            (experiment / 'rectified_config.json').write_text(json.dumps(compact_config(config), indent=2) + '\n', encoding='utf-8')
    backbone = compiled_backbone(model, getattr(args, 'compile', False),
                                 getattr(args, 'torch_compile_mode', 'default'), device)
    train_model = ranks.wrap(model)
    if ranks.world > 1:
        random.seed(args.seed + ranks.rank)
        np.random.seed(args.seed + ranks.rank)
        torch.manual_seed(args.seed + ranks.rank)
    preview_interval = int(settings.get('finetune_preview_interval', 500) if finetune
                           else settings.get('preview_interval', 1000))
    total = args.epochs * len(loader)
    warmup = settings.get('finetune_warmup_steps', 0) if finetune else settings['warmup_steps']
    if ranks.main:
        print(f'Rectified flow: {sum(p.numel() for p in model.parameters()):,} parameters, {precision_label}, {device}, whole utterances, up to {max_items} clips / {max_frames} frames per batch, {len(loader)} batches per epoch and GPU, {ranks.world} device(s)', flush=True)
        if finetune:
            print(f'Fine-tuning: fresh-run LR {lr:g}, warmup {warmup} steps, {settings.get("lr_schedule", "cosine")} decay, minimum LR {settings.get("min_learning_rate", 0.0):g}.', flush=True)
        if multispeaker:
            print(f'Multispeaker conditioning v{model.encoder.conditioning_version}: {speakers} speakers, {len(held)} held-out clips. Clips per speaker: {inventory}', flush=True)
            if settings['model'].get('direct_speaker_conditioning', False):
                predictor = str(model.aux.input.out_channels) if model.aux is not None else 'none'
                print(f'Standard flow: direct speaker conditioning in every flow block. Content {model.encoder.content.in_features} -> {model.hidden_channels}; speaker {model.encoder.speaker.embedding_dim}; mel predictor {predictor}; flow {model.backbone.channels}.', flush=True)
            if finetune and first_epoch == 1:
                print('Initialized independent speaker embeddings for the new dataset.', flush=True)
    with SummaryWriter(str(output)) if ranks.main else nullcontext(None) as writer:
        model.train()
        for epoch in range(first_epoch, args.epochs + 1):
            batcher.set_epoch(epoch - 1)
            for batch in loader:
                current_lr = learning_rate(
                    lr, step, warmup, total, settings['lr_final_ratio'],
                    schedule=settings.get('lr_schedule', 'cosine'),
                    decay_step=settings.get('decay_step', 4000),
                    gamma=settings.get('gamma', 0.9),
                    step_offset=settings.get('step_lr_offset', 1),
                    min_lr=float(settings.get('min_learning_rate', 0.0)),
                    step_warmup=finetune,
                )
                for group in optimizer.param_groups:
                    group['lr'] = current_lr
                flow, auxiliary, norm = train_step(train_model, optimizer, ema, batch, data, settings, device, dropout, amp_dtype, scaler, backbone, ranks)
                step += 1
                media_step = bool((preview_interval and step % preview_interval == 0) or
                                  (held and settings.get('eval_interval', 0) and step % settings['eval_interval'] == 0))
                if ranks.main:
                    for tag, value in [('loss/flow', flow), ('grad_norm', norm), ('lr', current_lr)]:
                        writer.add_scalar(tag, value, step)
                    if model.aux is not None:
                        writer.add_scalar(f'loss/{model.aux_loss_name}', auxiliary, step)
                    aux_status = f' aux={auxiliary:.6f}' if model.aux is not None else ''
                    print(f'epoch={epoch} step={step}/{total} flow={flow:.6f}{aux_status} grad_norm={norm:.6f}', flush=True)
                    if step % 50 == 0:
                        for tag, value in conditioning_norms(model).items():
                            writer.add_scalar(tag, value, step)
                        if scaler is not None:
                            writer.add_scalar("amp/scale", scaler.get_scale(), step)
                if media_step:
                    with ranks.main_work(f'preview/validation at step {step}') as main:
                        if main:
                            if preview_interval and step % preview_interval == 0:
                                print(f'Rank 0: starting preview at step {step}.', flush=True)
                                preview(model, ema, vocoder, reference, data, writer, step)
                                print(f'Rank 0: preview finished at step {step}.', flush=True)
                            if held_loader is not None and settings.get('eval_interval', 0) and step % settings['eval_interval'] == 0:
                                print(f'Rank 0: starting validation at step {step}.', flush=True)
                                evaluate(model, ema, held_loader, data, writer, step)
                                print(f'Rank 0: validation finished at step {step}.', flush=True)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                with ranks.main_work(f'checkpoint/preview at epoch {epoch}') as main:
                    if main:
                        print(f'Rank 0: saving checkpoint at epoch {epoch}.', flush=True)
                        if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                            raise FloatingPointError('Non-finite trained model weights.')
                        metadata = dict(config=config, speaker_count=speakers, embedder_model=embedder,
                                        epoch=epoch, step=step)
                        if multispeaker:
                            metadata.update(speaker_ids=sorted(inventory), feature_metadata=feature_metadata)
                        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                         ema=ema.state_dict(), finetune=finetune,
                                         scaler=scaler.state_dict() if scaler is not None else None,
                                         precision=args.precision, **metadata), resume_path)
                        atomic_save(dict(kind='rectified_flow', model=ema.cpu_state_dict(), **metadata),
                                    output / f'{args.model_name}_flow_{epoch}e_{step}s.pth')
                        print(f'Rank 0: checkpoint saved; starting preview at step {step}.', flush=True)
                        preview(model, ema, vocoder, reference, data, writer, step)
                        writer.flush()
                        print(f'Rank 0: epoch {epoch} checkpoint and preview finished.', flush=True)
            ranks.barrier()
    if ranks.main:
        print(f'Finished at step {step}. Checkpoints and TensorBoard previews: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train Multispeaker Rectified Flow with an optional frozen OpenVPI NSF-HiFiGAN preview vocoder.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--vocoder', default='')
    parser.add_argument('--batch-size', type=int, help='Maximum clips per batch and GPU (default: max_batch_size in the config).')
    parser.add_argument('--max-batch-frames', type=int, help='Maximum padded frames per batch and GPU (default: max_batch_frames in the config).')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                        help='cpu, one GPU such as cuda:0, or multiple GPUs such as cuda:0,cuda:1')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--pretrained-flow')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--torch-compile-mode', choices=['default', 'reduce-overhead', 'max-autotune'], default='default')
    parser.add_argument('--fresh', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
