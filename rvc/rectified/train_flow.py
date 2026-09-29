import argparse
import json
import os
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rvc.rectified.data import RectifiedDataset, collate_flow, read_filelist, split_holdout
from rvc.rectified.ema import WeightEMA
from rvc.rectified.flow_model import build_flow, resize_speakers
from rvc.rectified.mel import normalize_mel
from rvc.rectified.muon import MuonAdamW
from rvc.rectified.schedule import freeze_voice, learning_rate
from rvc.rectified.vocoder import load_vocoder

ROOT = Path(__file__).resolve().parents[2]


def atomic_save(state, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_step(model, optimizer, ema, batch, data, settings, device, speaker_dropout):
    mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask = (
        item.to(device, non_blocking=True) for item in batch
    )
    mel = normalize_mel(mel, data) * mask
    optimizer.zero_grad(set_to_none=True)
    flow, auxiliary = model(
        mel, content, f0, energy, speaker, mask,
        speaker_dropout=speaker_dropout, breathiness=breathiness,
        key_shift=key_shift, speed=speed,
    )
    loss = flow if auxiliary is None else flow + settings['aux_mel_weight'] * auxiliary
    if not torch.isfinite(loss):
        raise FloatingPointError('Non-finite rectified-flow loss.')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), settings['grad_clip'], error_if_nonfinite=True
    )
    optimizer.step()
    ema.update(model)
    return float(flow.detach()), float(auxiliary.detach()) if auxiliary is not None else 0.0, float(norm)


@torch.no_grad()
def preview(model, ema, vocoder, batch, data, writer, step):
    device = next(model.parameters()).device
    mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask = (
        item[:1].to(device) for item in batch
    )
    with ema.applied(model):
        model.eval()
        generated = model.sample(content, f0, energy, speaker, mask, steps=16,
                                 breathiness=breathiness, key_shift=key_shift)
        model.train()
    frames = int(mask.sum())
    audio = vocoder(generated[:, :, :frames], f0[:, :frames])[0]
    reference = vocoder(normalize_mel(mel[:, :, :frames], data), f0[:, :frames])[0]
    for name, value in [('flow', audio), ('vocoder_on_real_mel', reference)]:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f'Non-finite {name} preview.')
        writer.add_audio(f'audio/{name}', value.clamp(-1, 1).cpu(), step, data['sample_rate'])


@torch.no_grad()
def evaluate(model, ema, loader, data, writer, step):
    device = next(model.parameters()).device
    total, count = 0.0, 0
    with ema.applied(model):
        model.eval()
        for index, batch in enumerate(loader):
            mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask = (
                item.to(device) for item in batch
            )
            mel = normalize_mel(mel, data) * mask
            generator = torch.Generator(device=device).manual_seed(index)
            noise = torch.randn(mel.shape, device=device, generator=generator)
            losses, auxiliary = model.validation_losses(
                mel, content, f0, energy, speaker, mask, breathiness, key_shift, speed,
                noise, (0.1, 0.3, 0.5, 0.7, 0.9),
            )
            total += float(losses.mean()) * mel.shape[0]
            count += mel.shape[0]
        model.train()
    writer.add_scalar('val/flow', total / count, step)


def train(args):
    if args.batch_size < 1 or args.epochs < 1 or args.save_every < 1:
        raise ValueError('Batch size, epochs and save interval must be positive.')
    if Path(args.model_name).name != args.model_name or args.model_name in {'.', '..'}:
        raise ValueError('Use a model name, not a path.')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    experiment = ROOT / 'logs' / args.model_name
    config_path = experiment / 'rectified_config.json'
    if not config_path.exists():
        config_path = ROOT / 'rvc' / 'configs' / 'rectified' / '44100.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    settings, data = config['flow'], config['data']
    if data['sample_rate'] != 44100:
        raise ValueError('This recipe requires 44100 Hz audio and a matching NSF-HiFiGAN vocoder.')
    device = torch.device(args.device)
    vocoder, _ = load_vocoder(args.vocoder, data)
    vocoder = vocoder.to(device)
    entries = read_filelist(experiment / 'filelist.txt', ROOT)
    speakers = max(int(entry[4]) for entry in entries) + 1
    entries, held = split_holdout(entries, int(settings.get('holdout_clips', 0)))
    segment = int(settings['segment_frames'])
    dataset = RectifiedDataset(entries, config, segment)
    if len(dataset) < args.batch_size:
        raise ValueError('The training split has fewer clips than one full batch.')
    workers = int(settings.get('num_workers', 4))
    collate = partial(collate_flow, frames=segment)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=workers, collate_fn=collate,
                        pin_memory=device.type == 'cuda', persistent_workers=workers > 0)
    held_loader = DataLoader(RectifiedDataset(held, config, segment, augment=False),
                             batch_size=args.batch_size, collate_fn=collate) if held else None
    reference = collate([RectifiedDataset(entries[:1], config, segment, augment=False)[0]])
    info_path = experiment / 'model_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
    embedder = info.get('embedder_model', 'contentvec')
    model = build_flow(config, speakers).to(device).float()
    output = experiment / 'flow'
    resume_path = output / 'checkpoint.pth'
    state = torch.load(resume_path, map_location='cpu', weights_only=True) if resume_path.exists() and not args.fresh else None
    finetune = bool(state.get('finetune', False)) if state else bool(args.pretrained_flow)
    dropout = float(settings['speaker_dropout'])
    if finetune and speakers == 1 and settings.get('finetune_freeze_voice', True):
        freeze_voice(model)
        dropout = 0.0
    lr = args.learning_rate or settings['finetune_learning_rate' if finetune else 'learning_rate']
    if settings['optimizer'] == 'muon':
        optimizer = MuonAdamW(model, lr, muon_weight_decay=settings['weight_decay'], betas=tuple(settings['betas']))
    elif settings['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=tuple(settings['betas']), weight_decay=settings['weight_decay'])
    else:
        raise ValueError('Optimizer must be muon or adamw.')
    decay = settings.get('finetune_ema_decay', settings['ema_decay']) if finetune else settings['ema_decay']
    ema = WeightEMA(model, decay)
    first_epoch, step = 1, 0
    if state:
        if state['config'] != config or state['embedder_model'] != embedder:
            raise ValueError('Resume config or embedder differs from the saved checkpoint.')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'], model)
        first_epoch, step = state['epoch'] + 1, state['step']
        del state
    elif args.pretrained_flow:
        state = torch.load(args.pretrained_flow, map_location='cpu', weights_only=True)
        if state.get('embedder_model', embedder) != embedder:
            raise ValueError('Pretrained flow uses a different content embedder.')
        weights = state['ema']['shadow'] if state.get('ema') else state['model']
        model.load_state_dict(resize_speakers(weights, speakers), strict=True)
        ema.reseed(model)
        del state, weights
    output.mkdir(parents=True, exist_ok=True)
    (experiment / 'rectified_config.json').write_text(json.dumps(config, indent=4) + '\n', encoding='utf-8')
    total = args.epochs * len(loader)
    warmup = 0 if finetune else settings['warmup_steps']
    print(f'Rectified flow: {sum(p.numel() for p in model.parameters()):,} parameters, FP32, {device}, batch {args.batch_size}, {segment} frames', flush=True)
    with SummaryWriter(str(output)) as writer:
        model.train()
        for epoch in range(first_epoch, args.epochs + 1):
            for batch in loader:
                current_lr = learning_rate(lr, step, warmup, total, settings['lr_final_ratio'])
                for group in optimizer.param_groups:
                    group['lr'] = current_lr
                flow, auxiliary, norm = train_step(model, optimizer, ema, batch, data, settings, device, dropout)
                step += 1
                for tag, value in [('loss/flow', flow), ('loss/aux_mel_l1', auxiliary), ('grad_norm', norm), ('lr', current_lr)]:
                    writer.add_scalar(tag, value, step)
                print(f'epoch={epoch} step={step}/{total} flow={flow:.6f} aux={auxiliary:.6f} grad_norm={norm:.6f}', flush=True)
                if settings.get('preview_interval', 0) and step % settings['preview_interval'] == 0:
                    preview(model, ema, vocoder, reference, data, writer, step)
                if held_loader is not None and settings.get('eval_interval', 0) and step % settings['eval_interval'] == 0:
                    evaluate(model, ema, held_loader, data, writer, step)
            if epoch % args.save_every == 0 or epoch == args.epochs:
                if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                    raise FloatingPointError('Non-finite trained model weights.')
                metadata = dict(config=config, speaker_count=speakers, embedder_model=embedder,
                                vocoder=str(Path(args.vocoder).resolve()), epoch=epoch, step=step)
                atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                 ema=ema.state_dict(), finetune=finetune, **metadata), resume_path)
                atomic_save(dict(kind='rectified_flow', model=ema.cpu_state_dict(), **metadata),
                            output / f'{args.model_name}_flow_{epoch}e_{step}s.pth')
                preview(model, ema, vocoder, reference, data, writer, step)
                writer.flush()
    print(f'Finished at step {step}. Checkpoints and TensorBoard previews: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train Shiro rectified flow with a frozen OpenVPI NSF-HiFiGAN vocoder in FP32.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--vocoder', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--precision', choices=['fp32'], default='fp32')
    parser.add_argument('--pretrained-flow')
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--fresh', action='store_true')
    train(parser.parse_args())


if __name__ == '__main__':
    main()
