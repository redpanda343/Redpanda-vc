import argparse
import gc
import hashlib
import json
import math
import random
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from itertools import islice
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from beatrice.embedder_config import LOOKAHEAD_CHOICES, default_config
from beatrice.embedder_model import (ARCHITECTURE, EXPORT_ROOT, FORMAT, atomic_json, atomic_torch_save,
                                    embedder_metadata, export_embedder, file_hash, lookahead_settings,
                                    make_phone_extractor, trainer_module)
from beatrice.resources import ROOT, TRAINER_ROOT, ensure_trainer
from beatrice.train import AUDIO_SUFFIXES

CONTENTVEC_DIR = ROOT / 'models' / 'embedders' / 'contentvec'
CHECKPOINT = 'checkpoint_latest.pt'
SEED = 42


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Enter a positive whole number.')
    return number


def experiment_directory(name):
    if not name or Path(name).name != name or name in ('.', '..') or any(char in name for char in '/\\:'):
        raise ValueError('Enter an embedder name without folders or path separators.')
    return ROOT / 'logs' / name / 'embedder'


def geometry(kernels, strides):
    receptive_field, hop = 1, 1
    for kernel, stride in zip(kernels, strides, strict=True):
        receptive_field += (int(kernel) - 1) * hop
        hop *= int(stride)
    return receptive_field, hop


def audio_manifest(dataset):
    dataset = Path(dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise ValueError('The dataset folder does not exist.')
    files = sorted(path for path in dataset.rglob('*') if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES)
    if len(files) < 2:
        raise ValueError('Embedder training needs at least two audio files so validation can stay separate.')
    validation = [path for path in files if path.relative_to(dataset).parts[0].lower() in ('validation', 'valid')]
    if not validation:
        groups = {}
        for path in files:
            relative = path.relative_to(dataset)
            group = relative.parts[0] if len(relative.parts) > 1 else ''
            groups.setdefault(group, []).append(path)
        rng = random.Random(SEED)
        if len(groups) > 1:
            names = sorted(groups)
            rng.shuffle(names)
            held_out = set(names[:max(1, len(names) // 10)])
            validation = [path for group, paths in groups.items() if group in held_out for path in paths]
        else:
            shuffled = list(files)
            rng.shuffle(shuffled)
            validation = shuffled[:max(1, len(files) // 10)]
    validation_set = set(validation)
    training = [path for path in files if path not in validation_set]
    if not training or not validation:
        raise ValueError('Provide both training audio and separate validation audio.')
    manifest = []
    for split, paths in (('train', training), ('validation', validation)):
        for path in sorted(paths):
            stat = path.stat()
            manifest.append(dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, split=split))
    return str(dataset), manifest


def teacher_config():
    config = json.loads((CONTENTVEC_DIR / 'config.json').read_text(encoding='utf-8'))
    receptive_field, hop = geometry(config['conv_kernel'], config['conv_stride'])
    if hop != 320 or config['hidden_size'] != 768:
        raise ValueError('This training option requires the standard 768-channel, 20 ms ContentVec teacher.')
    return dict(receptive_field=receptive_field, hop=hop, channels=config['hidden_size'])


def recipe_for(dataset, manifest, clip_samples, teacher_hashes=None):
    teacher_hashes = teacher_hashes or dict(weights=file_hash(CONTENTVEC_DIR / 'pytorch_model.bin'),
                                          config=file_hash(CONTENTVEC_DIR / 'config.json'))
    return dict(format=FORMAT, dataset=dataset, files=manifest, clip_samples=clip_samples, seed=SEED,
                teacher='contentvec', teacher_layer='last_hidden_state',
                teacher_sha256=teacher_hashes['weights'],
                teacher_config_sha256=teacher_hashes['config'],
                original_sha256=file_hash(TRAINER_ROOT / 'assets' / 'pretrained' / '122_checkpoint_03000000.pt'),
                geometry=teacher_config())


def write_feature_batch(batch, targets, anchors, data_dir, recipe_hash):
    clips = []
    for piece, target, anchor in zip(batch, targets, anchors, strict=True):
        key = hashlib.sha256(f'{recipe_hash}:{piece["source"]}:{piece["offset"]}'.encode()).hexdigest()[:24]
        filename = f'{key}.npz'
        temporary = data_dir / f'{filename}.tmp'
        values = dict(audio=piece['audio'], teacher=target, anchor=anchor, valid_samples=piece['valid_samples'])
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as handle:
            for name, value in values.items():
                with handle.open(f'{name}.npy', 'w', force_zip64=True) as entry:
                    np.lib.format.write_array(entry, np.asanyarray(value), allow_pickle=False)
        temporary.replace(data_dir / filename)
        clips.append(dict(file=filename, source=piece['source'], offset=piece['offset'], split=piece['split']))
    return clips


def save_feature_batch(batch, teacher, original, device, data_dir, recipe_hash, writer=None):
    audio = np.stack([piece['audio'] for piece in batch])
    lengths = torch.tensor([piece['valid_samples'] for piece in batch], device=device)
    inputs = torch.from_numpy(audio).to(device)
    attention_mask = torch.arange(inputs.shape[1], device=device)[None] < lengths[:, None]
    with torch.inference_mode():
        targets = teacher(inputs, attention_mask=attention_mask.long(), return_dict=True).last_hidden_state
        anchors = original(inputs[:, None], return_stats=False).transpose(1, 2)
    targets = targets.to(device='cpu', dtype=torch.float16).numpy()
    anchors = anchors.to(device='cpu', dtype=torch.float16).numpy()
    if not np.isfinite(targets).all() or not np.isfinite(anchors).all():
        raise ValueError(f'Nonfinite teacher features in the batch beginning with: {batch[0]["source"]}')
    values = (tuple(batch), targets, anchors, data_dir, recipe_hash)
    return writer.submit(write_feature_batch, *values) if writer is not None else write_feature_batch(*values)


def prepare_data(directory, dataset, clip_samples, device, extraction_batch_size=None, workers=0):
    from shared.utils import (CONTENTVEC_CONFIG_SHA256, CONTENTVEC_SHA256, load_audio_16k, load_embedding)

    batch_size = positive_int(default_config()['extraction_batch_size'] if extraction_batch_size is None
                              else extraction_batch_size)
    dataset, manifest = audio_manifest(dataset)
    teacher = None
    teacher_files = dict(weights=CONTENTVEC_DIR / 'pytorch_model.bin', config=CONTENTVEC_DIR / 'config.json')
    teacher_hashes = ({key: file_hash(path) for key, path in teacher_files.items()}
                      if all(path.is_file() for path in teacher_files.values()) else None)
    if teacher_hashes != dict(weights=CONTENTVEC_SHA256, config=CONTENTVEC_CONFIG_SHA256):
        teacher = load_embedding('contentvec', 'v2')
        teacher_hashes = {key: file_hash(path) for key, path in teacher_files.items()}
    recipe = recipe_for(dataset, manifest, clip_samples, teacher_hashes)
    index_path = directory / 'data' / 'index.json'
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding='utf-8'))
        if index.get('recipe') == recipe and all((index_path.parent / row['file']).is_file() for row in index['clips']):
            print('Using the prepared ContentVec features.', flush=True)
            return index
    if (directory / CHECKPOINT).is_file():
        raise ValueError('The dataset, teacher or clip length changed. Restore the original data or use a new name.')
    teacher = (teacher if teacher is not None else load_embedding('contentvec', 'v2')).to(device).float().eval()
    teacher.requires_grad_(False)
    original = trainer_module().PhoneExtractor().to(device).eval().requires_grad_(False)
    original.load_state_dict(torch.load(TRAINER_ROOT / 'assets' / 'pretrained' / '122_checkpoint_03000000.pt',
                                        map_location='cpu', weights_only=True)['phone_extractor'], strict=True)
    data_dir = index_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    recipe_hash = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:16]
    clips = []
    pending = []
    benchmark = torch.backends.cudnn.benchmark
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f'Extracting teacher features in batches of {batch_size}.'
          + (' cuDNN benchmark is enabled.' if device.type == 'cuda' else ''), flush=True)
    writes = deque()
    pool = ThreadPoolExecutor(max_workers=min(workers, 4)) if workers > 0 else nullcontext()
    try:
        with pool as writer:
            def save_pending():
                result = save_feature_batch(pending, teacher, original, device, data_dir, recipe_hash, writer)
                if writer is None:
                    clips.extend(result)
                else:
                    writes.append(result)
                    if len(writes) >= 2:
                        clips.extend(writes.popleft().result())
                pending.clear()

            for entry in tqdm(manifest, desc='Extracting teacher features', unit='file', dynamic_ncols=True):
                audio = load_audio_16k(entry['path'])
                if not np.isfinite(audio).all():
                    raise ValueError(f'Audio contains nonfinite samples: {entry["path"]}')
                if len(audio) < 16000:
                    print(f'Skipping audio shorter than one second: {entry["path"]}', flush=True)
                    continue
                for offset in range(0, len(audio), clip_samples):
                    piece = audio[offset:offset + clip_samples]
                    if len(piece) < 16000:
                        continue
                    valid_samples = len(piece)
                    padded = np.pad(piece, (0, clip_samples - valid_samples)).astype(np.float32)
                    pending.append(dict(audio=padded, valid_samples=valid_samples, source=entry['path'],
                                        offset=offset, split=entry['split']))
                    if len(pending) == batch_size:
                        save_pending()
            if pending:
                save_pending()
            while writes:
                clips.extend(writes.popleft().result())
    finally:
        torch.backends.cudnn.benchmark = benchmark
        del teacher, original
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    if not all(any(row['split'] == split for row in clips) for split in ('train', 'validation')):
        raise ValueError('No usable clips in one split. Training and validation each need audio at least one second long.')
    index = dict(recipe=recipe, clips=clips)
    atomic_json(index_path, index)
    print(f'Prepared {sum(row["split"] == "train" for row in clips)} training clips and '
          f'{sum(row["split"] == "validation" for row in clips)} validation clips.', flush=True)
    return index


class FeatureDataset(Dataset):
    def __init__(self, directory, clips, split):
        self.paths = [directory / row['file'] for row in clips if row['split'] == split]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with np.load(self.paths[index], allow_pickle=False) as cache:
            audio, teacher, anchor = cache['audio'], cache['teacher'], cache['anchor']
            valid_samples = int(cache['valid_samples'])
        if (audio.ndim != 1 or len(audio) % 160 or not 16000 <= valid_samples <= len(audio)
                or teacher.shape != ((len(audio) - 400) // 320 + 1, 768)
                or anchor.shape != (len(audio) // 160, 128)):
            raise ValueError(f'The prepared feature clip is invalid: {self.paths[index]}')
        return dict(audio=torch.from_numpy(audio), teacher=torch.from_numpy(teacher),
                    anchor=torch.from_numpy(anchor), valid_samples=valid_samples)


class TrainingBatchSampler(Sampler):
    def __init__(self, size, batch_size, state=None):
        if size < 1 or batch_size < 1:
            raise ValueError('Training needs usable clips and a positive batch size.')
        self.size, self.batch_size = size, batch_size
        self.generator = torch.Generator().manual_seed(SEED)
        self.order, self.position = [], 0
        if state is not None:
            if (state['size'] != size or sorted(state['order']) != list(range(size))
                    or not 0 <= state['position'] <= size):
                raise ValueError('The saved data order does not match the prepared training clips.')
            self.order, self.position = state['order'], state['position']
            self.generator.set_state(state['generator'])

    def __iter__(self):
        if self.position >= len(self.order):
            self.new_epoch()
        for position in range(self.position, self.size, self.batch_size):
            yield self.order[position:position + self.batch_size]

    def __len__(self):
        return math.ceil((self.size - self.position) / self.batch_size)

    def advance(self, count):
        self.position += count

    def new_epoch(self):
        self.order = torch.randperm(self.size, generator=self.generator).tolist()
        self.position = 0

    def state_dict(self):
        if not self.order:
            self.new_epoch()
        return dict(size=self.size, order=list(self.order), position=self.position,
                    generator=self.generator.get_state())


class Distiller(nn.Module):
    def __init__(self, delay, offset=0):
        super().__init__()
        self.phone_extractor = make_phone_extractor(trainer_module().PhoneExtractor, delay, offset)
        self.projection = nn.Linear(128, 768)

    def forward(self, audio):
        phone = self.phone_extractor(audio[:, None], return_stats=False).transpose(1, 2)
        phone = phone.float()
        phone = phone * torch.rsqrt(phone.square().mean(-1, keepdim=True) + torch.finfo(torch.float32).eps)
        return phone, self.projection(phone)


def student_positions(frames, geometry_settings, device):
    centers = torch.arange(frames, device=device) * geometry_settings['hop']
    centers = centers + (geometry_settings['receptive_field'] - 1) / 2
    return (centers - 79.5) / 160


def weighted_mean(values, mask):
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def aligned_frames(values, batch, geometry_settings, delay, offset=0):
    targets = batch['teacher'].float()
    positions = student_positions(targets.shape[1], geometry_settings, values.device)
    left = positions.floor().long().clamp(0, values.shape[1] - 1)
    right = (left + 1).clamp_max(values.shape[1] - 1)
    fraction = (positions - left).clamp(0, 1)[None, :, None]
    aligned = torch.lerp(values[:, left].float(), values[:, right].float(), fraction)
    targets = F.layer_norm(targets, (targets.shape[-1],))
    lengths = batch['valid_samples']
    max_phone = (lengths - 200 - offset).div(160, rounding_mode='floor') - delay
    teacher_end = torch.arange(targets.shape[1], device=values.device) * geometry_settings['hop']
    teacher_end = teacher_end + geometry_settings['receptive_field']
    valid = (right[None] <= max_phone[:, None]) & (teacher_end[None] <= lengths[:, None])
    return aligned, targets, valid, max_phone


def initialize_projection(model, dataset, geometry_settings, delay, offset, device, clips, batch_size):
    generator = torch.Generator().manual_seed(SEED + 3)
    indices = torch.randperm(len(dataset), generator=generator)[:clips].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, generator=torch.Generator())
    channels = model.projection.in_features
    gram = torch.zeros(channels + 1, channels + 1, dtype=torch.float64, device=device)
    cross = torch.zeros(channels + 1, model.projection.out_features, dtype=torch.float64, device=device)
    target_energy, count = torch.zeros((), dtype=torch.float64, device=device), 0
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = move_batch(batch, device)
                phone, _ = model(batch['audio'])
                inputs, targets, valid, _ = aligned_frames(phone, batch, geometry_settings, delay, offset)
                inputs = F.pad(inputs[valid].double(), (0, 1), value=1.0)
                targets = targets[valid].double()
                gram += inputs.T @ inputs
                cross += inputs.T @ targets
                target_energy += targets.square().sum()
                count += len(targets)
    finally:
        model.train(training)
    if not count:
        raise ValueError('No valid aligned frames to fit the teacher projection. Use longer audio clips.')
    ridge = 1e-3 * gram.diagonal()[:channels].mean() * torch.eye(channels + 1, dtype=torch.float64, device=device)
    solution = torch.linalg.solve(gram + ridge, cross)
    residual = (target_energy - 2 * (solution * cross).sum() + (solution * (gram @ solution)).sum()) / (
        count * model.projection.out_features)
    if not torch.isfinite(solution).all() or not torch.isfinite(residual):
        raise FloatingPointError('Fitting the teacher projection produced nonfinite values.')
    with torch.no_grad():
        model.projection.weight.copy_(solution[:channels].T.float())
        model.projection.bias.copy_(solution[channels].float())
    print(f'Fitted the ContentVec projection on {count} frames from {len(indices)} clips '
          f'(feature loss {residual.item():.4f}).', flush=True)


def distillation_loss(phone, projected, batch, geometry_settings, delay, offset=0, config=None, check_valid=True):
    predicted, targets, valid, max_phone = aligned_frames(projected, batch, geometry_settings, delay, offset)
    predicted = F.layer_norm(predicted, (predicted.shape[-1],))
    if check_valid and not valid.any():
        raise ValueError('No valid aligned frames in the batch. Use longer audio clips.')
    feature = weighted_mean((predicted - targets).square().mean(-1), valid)
    transitions = weighted_mean((torch.diff(predicted, dim=1) - torch.diff(targets, dim=1)).abs().mean(-1),
                                valid[:, 1:] & valid[:, :-1])
    anchor = batch['anchor'].float()
    anchor_mask = torch.arange(phone.shape[1], device=phone.device)[None] <= max_phone[:, None]
    consistency = weighted_mean(1 - F.cosine_similarity(phone.float(), anchor, dim=-1), anchor_mask)
    config = config or {}
    loss = feature + config.get('transition_weight', 0.25) * transitions + config.get('anchor_weight', 0.1) * consistency
    return loss, dict(feature=feature, transitions=transitions, anchor=consistency)


def learning_rate_for_step(config, step):
    peak = config['learning_rate']
    if config.get('lr_schedule') != 'warmup_cosine':
        return peak
    total = config['schedule_steps']
    warmup = min(config['warmup_steps'], max(0, total - 1))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, max(0.0, (step - warmup + 1) / max(1, total - warmup)))
    floor = config['minimum_learning_rate_ratio']
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def precision_context(device, precision):
    if device.type != 'cuda' or precision == 'fp32':
        return nullcontext()
    return torch.autocast('cuda', dtype=torch.float16 if precision == 'fp16' else torch.bfloat16)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=device.type == 'cuda') for key, value in batch.items()}


def validate(model, loader, device, precision, geometry_settings, delay, offset=0, config=None):
    training = model.training
    model.eval()
    keys = ('loss', 'feature', 'transitions', 'anchor')
    totals, count = torch.zeros(len(keys), dtype=torch.float64, device=device), 0
    try:
        with torch.inference_mode():
            for batch in islice(loader, 32):
                batch = move_batch(batch, device)
                with precision_context(device, precision):
                    phone, projected = model(batch['audio'])
                loss, components = distillation_loss(phone, projected, batch, geometry_settings, delay, offset,
                                                     config)
                values = dict(loss=loss, **components)
                totals += torch.stack([values[key] for key in keys]).double() * len(batch['audio'])
                count += len(batch['audio'])
        if not count:
            raise ValueError('There are no validation clips.')
        return dict(zip(keys, (totals / count).cpu().tolist(), strict=True))
    finally:
        model.train(training)


def save_progress(directory, name, model, optimizer, scaler, config, step, metrics, sampler=None):
    state = dict(model={key: value.detach().cpu() for key, value in model.state_dict().items()},
                 optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), config=config, step=step,
                 metrics=metrics, torch_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state(next(model.parameters()).device)
                 if next(model.parameters()).is_cuda else None)
    if sampler is not None:
        state['data_sampler'] = sampler.state_dict()
    atomic_torch_save(directory / CHECKPOINT, state)
    metadata = embedder_metadata(config['phone_delay_frames'], config.get('phone_offset_samples', 0),
                                 teacher='contentvec', teacher_sha256=config['recipe']['teacher_sha256'], step=step,
                                 validation=metrics)
    path = export_embedder(model.phone_extractor, metadata, EXPORT_ROOT / name)
    atomic_json(directory / 'metrics.json', dict(step=step, validation=metrics,
                                                learning_rate=optimizer.param_groups[0]['lr']))
    print(f'Saved step {step}. Beatrice embedder: {path}', flush=True)


def train(directory, name, index, args, device):
    checkpoint_path = directory / CHECKPOINT
    config_path = directory / 'config.json'
    defaults = default_config()
    saved_parameters = ('warmup_steps', 'minimum_learning_rate_ratio', 'adam_betas', 'weight_decay', 'max_grad_norm',
                        'transition_weight', 'anchor_weight', 'gain_range', 'noise_relative_std',
                        'projection_learning_rate_ratio', 'projection_init_clips')
    config = dict(format=FORMAT, architecture=ARCHITECTURE, **lookahead_settings(args.lookahead_ms),
                  recipe=index['recipe'], learning_rate=args.learning_rate, precision=args.precision,
                  lr_schedule='warmup_cosine', schedule_steps=args.steps,
                  **{key: defaults[key] for key in saved_parameters})
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True) if checkpoint_path.is_file() else None
    if checkpoint is not None:
        saved = checkpoint['config']
        if (any(saved[key] != config[key] for key in ('format', 'architecture', 'phone_delay_frames', 'recipe'))
                or saved.get('phone_offset_samples', 0) != config['phone_offset_samples']):
            raise ValueError('Lookahead or training data changed. Resume with the saved settings or use a new name.')
        config = saved
    else:
        atomic_json(config_path, config)
    precision = config['precision']
    if device.type == 'cuda' and precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16. Choose fp16 or fp32 in Settings → Training → Precision.')
    if device.type == 'cpu' and precision != 'fp32':
        print('CPU training uses fp32.', flush=True)
    delay = config['phone_delay_frames']
    offset = config.get('phone_offset_samples', 0)
    torch.manual_seed(SEED)
    model = Distiller(delay, offset).to(device)
    if checkpoint is None:
        original = torch.load(TRAINER_ROOT / 'assets' / 'pretrained' / '122_checkpoint_03000000.pt',
                              map_location='cpu', weights_only=True)
        model.phone_extractor.load_state_dict(original['phone_extractor'], strict=True)
        del original
    else:
        model.load_state_dict(checkpoint['model'], strict=True)
    projection_ratio = config.get('projection_learning_rate_ratio')
    parameter_groups = ([dict(params=model.phone_extractor.parameters()), dict(params=model.projection.parameters())]
                        if projection_ratio else model.parameters())
    learning_rate_scales = (1.0, projection_ratio) if projection_ratio else (1.0,)
    optimizer = torch.optim.AdamW(parameter_groups, lr=config['learning_rate'],
                                  betas=tuple(config.get('adam_betas', (0.9, 0.99))),
                                  weight_decay=config.get('weight_decay', 0.01), fused=device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and precision == 'fp16', init_scale=1024)
    step = 0
    if checkpoint is not None:
        for group in checkpoint['optimizer']['param_groups']:
            group.update(fused=device.type == 'cuda', foreach=None)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng') is not None:
            torch.cuda.set_rng_state(checkpoint['cuda_rng'], device=device)
        step = checkpoint['step']
        print(f'Resuming embedder from step {step}.', flush=True)
    train_set = FeatureDataset(directory / 'data', index['clips'], 'train')
    valid_set = FeatureDataset(directory / 'data', index['clips'], 'validation')
    if not len(valid_set):
        raise ValueError('Training needs separate usable validation clips.')
    if checkpoint is None and config.get('projection_init_clips'):
        initialize_projection(model, train_set, index['recipe']['geometry'], delay, offset, device,
                              config['projection_init_clips'], args.batch_size)
    sampler = TrainingBatchSampler(len(train_set), args.batch_size,
                                   checkpoint.get('data_sampler') if checkpoint is not None else None)
    loader_generator = torch.Generator().manual_seed(SEED + 1)
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.workers,
                              persistent_workers=args.workers > 0, pin_memory=device.type == 'cuda',
                              generator=loader_generator)
    valid_generator = torch.Generator().manual_seed(SEED + 2)
    valid_indices = torch.randperm(len(valid_set), generator=valid_generator)[:32 * args.batch_size].tolist()
    valid_loader = DataLoader(Subset(valid_set, valid_indices), batch_size=args.batch_size, num_workers=0,
                              pin_memory=device.type == 'cuda', generator=valid_generator)
    writer = SummaryWriter(directory / 'events')
    iterator = iter(train_loader) if step < args.steps else None
    model.train()
    metrics = checkpoint.get('metrics', {}) if checkpoint is not None else {}
    saved_step = step
    del checkpoint
    overflow_attempts = 0
    benchmark = torch.backends.cudnn.benchmark
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    try:
        with tqdm(total=args.steps, initial=min(step, args.steps), desc='Training embedder', unit='step',
                  dynamic_ncols=True) as progress:
            while step < args.steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                sampler.advance(len(batch['audio']))
                batch = move_batch(batch, device)
                audio = batch['audio']
                gain = torch.empty(len(audio), 1, device=device).uniform_(*config.get('gain_range', (0.7, 1.3)))
                noise = (torch.randn_like(audio) * audio.square().mean(-1, keepdim=True).sqrt()
                         * config.get('noise_relative_std', 0.005))
                audio = audio * gain + noise
                current_lr = learning_rate_for_step(config, step)
                for group, scale in zip(optimizer.param_groups, learning_rate_scales, strict=True):
                    group['lr'] = current_lr * scale
                optimizer.zero_grad(set_to_none=True)
                with precision_context(device, precision):
                    phone, projected = model(audio)
                loss, components = distillation_loss(phone, projected, batch, index['recipe']['geometry'], delay,
                                                     offset, config, check_valid=False)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Embedder training produced a nonfinite loss. Try fp32 or a lower learning rate.')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('max_grad_norm', 1.0),
                                                          error_if_nonfinite=False)
                if not torch.isfinite(grad_norm):
                    if not scaler.is_enabled():
                        raise FloatingPointError('Nonfinite gradients. Try a lower learning rate.')
                    scaler.step(optimizer)
                    scaler.update()
                    overflow_attempts += 1
                    if overflow_attempts > 16:
                        raise FloatingPointError('Repeated fp16 gradient overflow. Choose fp32 in training settings.')
                    continue
                overflow_attempts = 0
                scaler.step(optimizer)
                scaler.update()
                step += 1
                progress.update(1)
                if step == 1 or step % 10 == 0 or step == args.steps:
                    values = torch.stack([loss.detach(), *[value.detach() for value in components.values()]])
                    recorded = dict(zip(('loss', *components), values.cpu().tolist(), strict=True))
                    progress.set_postfix(loss=f'{recorded["loss"]:.4f}', lr=f'{current_lr:.2e}', refresh=False)
                    if step == 1 or step % 50 == 0:
                        writer.add_scalar('train/learning_rate', current_lr, step)
                        for key, value in recorded.items():
                            writer.add_scalar(f'train/{key}', value, step)
                if step % args.save_interval == 0 or step == args.steps:
                    metrics = validate(model, valid_loader, device, precision, index['recipe']['geometry'], delay,
                                       offset, config)
                    if not all(math.isfinite(value) for value in metrics.values()):
                        raise FloatingPointError('Nonfinite validation results. Try fp32 or a lower learning rate.')
                    for key, value in metrics.items():
                        writer.add_scalar(f'validation/{key}', value, step)
                    save_progress(directory, name, model, optimizer, scaler, config, step, metrics, sampler)
        if step >= args.steps and step == saved_step and checkpoint_path.is_file():
            save_progress(directory, name, model, optimizer, scaler, config, step, metrics, sampler)
    finally:
        writer.close()
        torch.backends.cudnn.benchmark = benchmark


def main():
    defaults = default_config()
    parser = argparse.ArgumentParser(description='Distill ContentVec into a small CPU Beatrice embedder.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=positive_int, default=defaults['steps'])
    parser.add_argument('--batch-size', type=positive_int, default=defaults['batch_size'])
    parser.add_argument('--extraction-batch-size', type=positive_int, default=defaults['extraction_batch_size'])
    parser.add_argument('--save-interval', type=positive_int, default=defaults['save_interval'])
    parser.add_argument('--workers', type=int, default=defaults['workers'])
    parser.add_argument('--clip-seconds', type=float, default=defaults['clip_seconds'])
    parser.add_argument('--lookahead-ms', type=int, choices=LOOKAHEAD_CHOICES, default=defaults['lookahead_ms'])
    parser.add_argument('--learning-rate', type=float, default=defaults['learning_rate'])
    parser.add_argument('--precision', choices=('fp32', 'fp16', 'bf16'), default=defaults['precision'])
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    directory = experiment_directory(args.model_name)
    if args.workers < 0 or not 1 <= args.clip_seconds <= 4:
        raise ValueError('Workers must be nonnegative and clip length must be between one and four seconds.')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be positive and finite.')
    ensure_trainer()
    if args.export_only:
        checkpoint = torch.load(directory / CHECKPOINT, map_location='cpu', weights_only=True)
        config = checkpoint['config']
        model = Distiller(config['phone_delay_frames'], config.get('phone_offset_samples', 0))
        model.load_state_dict(checkpoint['model'], strict=True)
        metadata = embedder_metadata(config['phone_delay_frames'], config.get('phone_offset_samples', 0),
                                     teacher='contentvec', teacher_sha256=config['recipe']['teacher_sha256'],
                                     step=checkpoint['step'], validation=checkpoint.get('metrics', {}))
        print(export_embedder(model.phone_extractor, metadata, EXPORT_ROOT / args.model_name), flush=True)
        return
    device_name = args.device.strip().lower()
    if device_name != 'auto' and device_name != 'cpu' and not (device_name.startswith('cuda:') and device_name[5:].isdigit()):
        raise ValueError('Device must be auto, cpu or cuda:N.')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else torch.device(device_name)
    if device.type == 'cuda' and (not torch.cuda.is_available() or device.index >= torch.cuda.device_count()):
        raise ValueError('The selected CUDA device is unavailable.')
    if not args.dataset:
        raise ValueError('Choose the folder containing your English training audio.')
    directory.mkdir(parents=True, exist_ok=True)
    clip_samples = round(args.clip_seconds * 100) * 160
    index = prepare_data(directory, args.dataset, clip_samples, device, args.extraction_batch_size, args.workers)
    if not args.prepare_only:
        train(directory, args.model_name, index, args, device)


if __name__ == '__main__':
    main()
