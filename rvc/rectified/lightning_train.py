import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy, StrategyRegistry
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from rvc.rectified.augmentation import is_augmented, prepare_augmentation
from rvc.rectified.config import compact_config, resolve_config
from rvc.rectified.data import (
    FlowBatchSampler, RectifiedDataset, collate_flow, prepare_training_cache,
    read_filelist, speaker_inventory, split_holdout, unpack_flow,
)
from rvc.rectified.distributed import parse_devices
from rvc.rectified.flow_model import build_flow, resize_speakers
from rvc.rectified.mel import normalize_mel
from rvc.rectified.muon import MuonAdamW
from rvc.rectified.schedule import learning_rate
from rvc.rectified.train_flow import (
    atomic_save, configure_fused_backbone, preview, prune_checkpoints,
    random_state, restore_random_state, select_fused_activation,
)
from rvc.rectified.vocoder import load_vocoder


def validation_collate(batch):
    return collate_flow(batch) if batch else None


def latest_checkpoint(output):
    candidates = []
    for path in Path(output).glob('model_ckpt_steps_*.ckpt'):
        match = re.fullmatch(r'model_ckpt_steps_(\d+)(?:-v\d+)?\.ckpt', path.name)
        if match:
            candidates.append((int(match[1]), path.stat().st_mtime_ns, path))
    return str(max(candidates)[2]) if candidates else None


def prune_lightning_checkpoints(output, settings):
    candidates = []
    for path in Path(output).glob('model_ckpt_steps_*.ckpt'):
        match = re.fullmatch(r'model_ckpt_steps_(\d+)(?:-v\d+)?\.ckpt', path.name)
        if match:
            candidates.append((int(match[1]), path.stat().st_mtime_ns, path))
    candidates.sort(reverse=True)
    start, interval = settings['permanent_ckpt_start'], settings['permanent_ckpt_interval']
    for step, _, path in candidates[settings['num_ckpt_keep']:]:
        if not (start > 0 and step >= start and (step - start) % interval == 0):
            path.unlink()


class FlowDataModule(pl.LightningDataModule):
    def __init__(self, experiment, root, config, args):
        super().__init__()
        self.experiment, self.root, self.config, self.args = experiment, root, config, args
        self.settings = config['flow']
        self.originals = read_filelist(experiment / 'filelist.txt', root, originals_only=True)
        self.multispeaker = self.settings['model'].get('conditioning_version', 1) in (2, 3, 4, 5)
        self.inventory = speaker_inventory(self.originals) if self.multispeaker else None
        self.speaker_count = max(int(entry[4]) for entry in self.originals) + 1
        self.training_entries, self.held = split_holdout(
            self.originals, int(self.settings.get('holdout_clips', 0)), stratified=self.multispeaker,
        )
        self.max_items = int(args.batch_size or self.settings['max_batch_size'])
        self.max_frames = int(args.max_batch_frames or self.settings['max_batch_frames'])
        info_path = experiment / 'model_info.json'
        info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.exists() else {}
        self.embedder = info.get('embedder_model', 'contentvec')
        self.feature_metadata = {key: info[key] for key in (
            'embedder_model', 'version', 'feature_dim', 'feature_output', 'feature_fingerprint',
        ) if key in info}
        if self.multispeaker and (info.get('version', 'v2') != 'v2' or
                                 int(info.get('feature_dim', self.settings['model']['content_channels'])) != self.settings['model']['content_channels']):
            raise ValueError('Extraction metadata does not match the content encoder. Re-extract v2 features.')
        self.training_sampler = None
        self.resume_sampler_cap = None

    def prepare_data(self):
        prepare_training_cache(self.originals, self.config, self.experiment / 'rectified-flow.data',
                               self.settings.get('num_workers', 4))
        prepare_augmentation(self.experiment, self.root, self.originals, self.training_entries,
                             self.config, self.args.seed, self.trainer.strategy.root_device)

    def setup(self, stage):
        entries = read_filelist(self.experiment / 'filelist.txt', self.root)
        entries = self.training_entries + [entry for entry in entries if is_augmented(entry)]
        cache_path = self.experiment / 'rectified-flow.data'
        self.train_dataset = RectifiedDataset(entries, self.config, self.max_frames, cache_path=cache_path)
        self.valid_dataset = RectifiedDataset(
            self.held, self.config, self.settings['max_val_batch_frames'], augment=False, cache_path=cache_path,
        )
        self.references = []
        if self.trainer.is_global_zero and self.settings.get('preview_interval', 0):
            self.references = (self.valid_dataset if self.held else self.train_dataset).references(
                self.settings.get('num_valid_plots', 10),
            )

    def loader(self, dataset, sampler, validation=False):
        workers = int(self.settings.get('num_workers', 4))
        kwargs = dict(num_workers=workers, pin_memory=not validation,
                      persistent_workers=workers > 0, collate_fn=validation_collate if validation else collate_flow)
        if workers:
            kwargs.update(prefetch_factor=self.settings['dataloader_prefetch_factor'], multiprocessing_context='spawn')
        return DataLoader(dataset, batch_sampler=sampler, **kwargs)

    def train_dataloader(self):
        self.training_sampler = FlowBatchSampler(
            self.train_dataset, self.max_frames, self.max_items, self.args.seed,
            self.trainer.global_rank, self.trainer.world_size,
            required_batch_count_multiple=self.settings['accumulate_grad_batches'],
            sort_by_len=self.settings['sort_by_len'], frame_count_grid=self.settings['sampler_frame_count_grid'],
        )
        self.training_sampler.measured_max_frames = self.resume_sampler_cap
        return self.loader(self.train_dataset, self.training_sampler)

    def val_dataloader(self):
        if not self.held:
            return []
        sampler = FlowBatchSampler(
            self.valid_dataset, self.settings['max_val_batch_frames'], self.settings['max_val_batch_size'],
            self.args.seed, self.trainer.global_rank, self.trainer.world_size, shuffle=False,
            disallow_empty_batch=False, pad_batch_assignment=False,
        )
        return self.loader(self.valid_dataset, sampler, validation=True)

    def state_dict(self):
        cap = self.training_sampler.measured_max_frames if self.training_sampler else None
        return dict(measured_max_frames=int(cap) if cap is not None else None)

    def load_state_dict(self, state):
        self.resume_sampler_cap = state.get('measured_max_frames')
        if self.training_sampler:
            self.training_sampler.measured_max_frames = self.resume_sampler_cap
            self.training_sampler._formed = None


class FlowTask(pl.LightningModule):
    def __init__(self, config, args, datamodule, finetune=False):
        super().__init__()
        self.config, self.args, self.data_module = config, args, datamodule
        self.settings, self.data = config['flow'], config['data']
        self.finetune = finetune
        self.model = build_flow(config, datamodule.speaker_count).float()
        self.base_lr = args.learning_rate or self.settings['finetune_learning_rate' if finetune else 'learning_rate']
        self.valid_losses = torch.nn.ModuleDict({name: MeanMetric() for name in ('total_loss', 'mel_loss', 'aux_mel_loss')})
        self.skip_immediate_validation = False
        self.skip_immediate_ckpt_save = False
        self.resume_random = None
        self.trained_epoch = 1

    def run_model(self, batch):
        (mel, content, f0, energy, breathiness, key_shift, speed, speaker, mask,
         harmonic_prior, voicing, tension, phonation) = unpack_flow(batch, self.device, True)
        mel = normalize_mel(mel, self.data)
        if not self.model.reference:
            mel = mel * mask
        flow, auxiliary = self.model(
            mel, content, f0, energy, speaker, mask,
            speaker_dropout=self.settings['speaker_dropout'] if self.training else 0.0,
            breathiness=breathiness, key_shift=key_shift, speed=speed, voicing=voicing, tension=tension,
            harmonic_prior=harmonic_prior if harmonic_prior.shape[1] else None,
            phonation=phonation,
        )
        aux = flow.new_zeros(()) if auxiliary is None else auxiliary * self.settings['aux_mel_weight']
        return dict(mel_loss=flow, aux_mel_loss=aux, total_loss=flow + aux)

    def training_step(self, batch, batch_idx):
        self.trained_epoch = self.current_epoch + 1
        losses = self.run_model(batch)
        if not torch.isfinite(losses['total_loss']):
            raise FloatingPointError('Non-finite rectified-flow training loss.')
        log_outputs = dict(mel_loss=losses['mel_loss'], aux_mel_loss=losses['aux_mel_loss'], batch_size=float(batch[0].shape[0]))
        self.log_dict(log_outputs, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        lr = self.lr_schedulers().get_last_lr()[0]
        self.log('lr', lr, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        if self.global_step % self.settings['log_interval'] == 0 and self.trainer.is_global_zero:
            self.logger.log_metrics({**{f'training/{key}': value.detach() if torch.is_tensor(value) else value
                                       for key, value in log_outputs.items()}, 'training/lr': lr}, step=self.global_step)
        return losses['total_loss']

    def on_train_epoch_start(self):
        self.data_module.training_sampler.set_epoch(self.current_epoch)

    def on_validation_start(self):
        if not self.skip_immediate_validation:
            for metric in self.valid_losses.values():
                metric.reset()

    def validation_step(self, batch, batch_idx):
        if self.skip_immediate_validation or batch is None:
            return
        with torch.autocast(self.device.type, enabled=False):
            losses = self.run_model(batch)
        for name, loss in losses.items():
            self.valid_losses[name].update(loss, weight=batch[0].shape[0])

    def on_validation_epoch_end(self):
        if self.skip_immediate_validation:
            self.skip_immediate_validation = False
            self.skip_immediate_ckpt_save = True
            return
        losses = {name: metric.compute() for name, metric in self.valid_losses.items()}
        self.log('val_loss', losses['total_loss'], on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        if self.trainer.is_global_zero:
            self.logger.log_metrics({f'validation/{key}': value for key, value in losses.items()}, step=self.global_step)

    def build_optimizer(self):
        settings = self.settings
        if settings['optimizer'] == 'muon':
            return MuonAdamW(self.model, self.base_lr, muon_weight_decay=settings['weight_decay'],
                             adamw_weight_decay=settings['adamw_weight_decay'], min_fan_in=settings['muon_min_fan_in'],
                             betas=tuple(settings['betas']), iteration_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32)
        if settings['optimizer'] == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=self.base_lr, betas=tuple(settings['betas']),
                                     weight_decay=settings['weight_decay'])
        raise ValueError('Optimizer must be muon or adamw.')

    def build_scheduler(self, optimizer):
        settings = self.settings
        warmup = settings.get('finetune_warmup_steps', 0) if self.finetune else settings['warmup_steps']
        if settings['lr_schedule'] == 'step' and not (warmup or settings['min_learning_rate'] or settings['step_lr_offset']):
            return torch.optim.lr_scheduler.StepLR(optimizer, step_size=settings['decay_step'], gamma=settings['gamma'])
        total = self.args.max_updates or self.trainer.estimated_stepping_batches
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: learning_rate(
            self.base_lr, step, warmup, total, settings['lr_final_ratio'], settings['lr_schedule'],
            settings['decay_step'], settings['gamma'], settings['step_lr_offset'],
            settings['min_learning_rate'], self.finetune,
        ) / self.base_lr)

    def configure_optimizers(self):
        optimizer = self.build_optimizer()
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=self.build_scheduler(optimizer), interval='step', frequency=1))

    def on_fit_start(self):
        dtype = {'16-mixed': torch.float16, 'bf16-mixed': torch.bfloat16}.get(
            getattr(self.trainer.precision_plugin, 'precision', None),
        )
        configure_fused_backbone(self.model, getattr(self.args, 'use_fused_kernels', False),
                                 self.device, dtype, self.data_module.max_frames)
        if self.resume_random:
            if len(self.resume_random) != self.trainer.world_size:
                raise ValueError('Resume requires the same number of training devices.')
            restore_random_state(self.resume_random[self.global_rank], self.device)

    def metadata(self):
        data = self.data_module
        result = dict(config=self.config, speaker_count=self.model.speaker_count,
                      embedder_model=data.embedder, step=self.global_step,
                      finetune=self.finetune, precision=self.args.precision, trained_epoch=self.trained_epoch)
        if data.multispeaker:
            result.update(speaker_ids=sorted(data.inventory), feature_metadata=data.feature_metadata)
        return result

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(self.metadata())
        checkpoint['trainer_stage'] = self.trainer.state.stage.value
        states = [random_state(self.device)]
        if self.trainer.world_size > 1:
            states = [None] * self.trainer.world_size
            torch.distributed.all_gather_object(states, random_state(self.device))
        checkpoint['random_states'] = states

    def on_load_checkpoint(self, checkpoint):
        saved = resolve_config(checkpoint['config'])
        saved['flow']['model'].setdefault('use_phonation', False)
        saved['flow']['model'].setdefault('use_continuous_f0', saved['flow']['model'].get('conditioning_version') == 5)
        select_fused_activation(saved, getattr(self.args, 'use_fused_kernels', False))
        if saved['data'] != self.data or saved['flow']['model'] != self.settings['model']:
            raise ValueError('Resume architecture or audio configuration differs from the checkpoint. Enable phonation in a new experiment and fine-tune from the voice .pth export.')
        if saved['flow']['optimizer'] != self.settings['optimizer']:
            raise ValueError('Resume requires the same optimizer type. Use the voice export to fine-tune with a different optimizer.')
        data = self.data_module
        if checkpoint.get('embedder_model') != data.embedder:
            raise ValueError('Resume content embedder differs from the checkpoint.')
        if data.multispeaker and (checkpoint.get('speaker_ids') != sorted(data.inventory) or
                                 checkpoint.get('feature_metadata') != data.feature_metadata):
            raise ValueError('Speaker IDs or extraction metadata changed. Use a new experiment.')
        self.skip_immediate_validation = checkpoint.get('trainer_stage') == 'validate'
        self.resume_random = checkpoint.get('random_states')
        self.trained_epoch = checkpoint.get('trained_epoch', checkpoint['epoch'] + 1)
        if checkpoint.get('optimizer_states'):
            optimizer = self.build_optimizer()
            scheduler = self.build_scheduler(optimizer)
            step = checkpoint['global_step']
            scheduler.last_epoch = step
            scheduler._step_count = step + 1
            if isinstance(scheduler, torch.optim.lr_scheduler.StepLR):
                scheduler._last_lr = scheduler._get_closed_form_lr()
            else:
                scheduler._last_lr = [base * fn(step) for base, fn in zip(scheduler.base_lrs, scheduler.lr_lambdas)]
            for old, new, lr in zip(checkpoint['optimizer_states'][0]['param_groups'], optimizer.param_groups, scheduler.get_last_lr()):
                for key, value in new.items():
                    if key != 'params':
                        old[key] = value
                old.update(lr=lr, initial_lr=self.base_lr)
            checkpoint['lr_schedulers'] = [scheduler.state_dict()]

    def load_pretrained(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        source = resolve_config(state.get('config', {}))
        select_fused_activation(source, getattr(self.args, 'use_fused_kernels', False))
        source_model, target_model = deepcopy(source['flow']['model']), deepcopy(self.settings['model'])
        add_phonation = target_model.get('use_phonation', False) and not source_model.get('use_phonation', False)
        for candidate in (source_model, target_model):
            candidate.pop('use_phonation', None)
            candidate.pop('use_continuous_f0', None)
        if source['data'] != self.data or source_model != target_model:
            raise ValueError('Pretrained architecture or audio configuration differs from the experiment.')
        if state.get('embedder_model', self.data_module.embedder) != self.data_module.embedder:
            raise ValueError('Pretrained flow uses a different content embedder.')
        speaker_init = self.model.encoder.speaker.weight if self.model.use_spk_id else None
        weights = state['model']
        if add_phonation:
            from rvc.rectified.phonation import initialize_phonation_weights

            weights = initialize_phonation_weights(weights, self.model)
        self.model.load_state_dict(resize_speakers(weights, self.model.speaker_count, speaker_init,
                                                  null_speaker=self.model.encoder.has_null_speaker), strict=True)


class FlowCheckpoint(ModelCheckpoint):
    def __init__(self, output, args, settings, has_validation):
        self.args, self.settings = args, settings
        interval = args.checkpoint_interval or settings['checkpoint_interval']
        self.at_validation = has_validation and interval == settings['eval_interval']
        super().__init__(dirpath=output, filename='model_ckpt_steps_{step}', auto_insert_metric_name=False,
                         monitor='step', mode='max', save_top_k=settings['num_ckpt_keep'], save_last=True,
                         every_n_train_steps=0 if self.at_validation else interval,
                         every_n_epochs=1 if self.at_validation else 0, save_on_train_epoch_end=False)

    def on_validation_end(self, trainer, pl_module):
        if pl_module.skip_immediate_ckpt_save:
            pl_module.skip_immediate_ckpt_save = False
            return
        if self.at_validation and not self._should_skip_saving_checkpoint(trainer):
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            self._save_last_checkpoint(trainer, candidates)

    def on_train_end(self, trainer, pl_module):
        if trainer.global_step != self._last_global_step_saved:
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            self._save_last_checkpoint(trainer, candidates)

    def _save_checkpoint(self, trainer, filepath):
        task = trainer.lightning_module
        if any(not torch.isfinite(value).all() for value in task.model.state_dict().values()):
            raise FloatingPointError('Non-finite trained model weights.')
        super()._save_checkpoint(trainer, filepath)
        if not trainer.is_global_zero or Path(filepath).name == 'last.ckpt':
            return
        metadata = task.metadata()
        epoch = task.trained_epoch
        path = Path(self.dirpath) / f'{self.args.model_name}_flow_{epoch}e_{trainer.global_step}s.pth'
        atomic_save(dict(kind='rectified_flow', model={key: value.detach().cpu() for key, value in task.model.state_dict().items()},
                         epoch=epoch, **metadata), path)
        prune_checkpoints(Path(self.dirpath), self.args.model_name, self.settings)
        prune_lightning_checkpoints(self.dirpath, self.settings)

    def _remove_checkpoint(self, trainer, filepath):
        if not Path(filepath).is_file():
            return
        match = re.search(r'steps_(\d+)', Path(filepath).name)
        start, interval = self.settings['permanent_ckpt_start'], self.settings['permanent_ckpt_interval']
        if match and start > 0 and int(match[1]) >= start and (int(match[1]) - start) % interval == 0:
            return
        super()._remove_checkpoint(trainer, filepath)


class FlowPreview(pl.Callback):
    def __init__(self, args):
        self.args = args
        self.vocoder = None
        self.last_step = -1

    def on_fit_start(self, trainer, task):
        if trainer.is_global_zero and self.args.vocoder and task.settings.get('val_with_vocoder', True):
            self.vocoder, _ = load_vocoder(self.args.vocoder, task.data)
            self.vocoder = self.vocoder.to(task.device)

    def render(self, trainer, task):
        if not trainer.is_global_zero or self.last_step == trainer.global_step:
            return
        with torch.autocast(task.device.type, enabled=False):
            for index, reference in enumerate(task.data_module.references):
                preview(task.model, self.vocoder, reference, task.data, trainer.logger.experiment, trainer.global_step, index)
        self.last_step = trainer.global_step

    def on_validation_epoch_end(self, trainer, task):
        if not task.skip_immediate_validation:
            self.render(trainer, task)

    def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx):
        interval = task.settings.get('finetune_preview_interval', 500) if task.finetune else task.settings.get('preview_interval', 0)
        if interval and trainer.global_step and trainer.global_step % interval == 0:
            self.render(trainer, task)


def trainer_device_options(args, settings):
    selected = args.device
    accelerator = settings['accelerator']
    if str(selected).lower() == 'auto' or isinstance(selected, (int, list)):
        accelerator = ('gpu' if torch.cuda.is_available() else 'cpu') if accelerator == 'auto' else accelerator
        devices = selected if isinstance(selected, (int, list)) else settings.get('devices', 'auto')
        if devices == 'auto' or isinstance(devices, (int, list)):
            return dict(accelerator=accelerator, devices=devices)
        selected = devices
    devices = parse_devices(selected)
    if devices == ['cpu']:
        return dict(accelerator='cpu', devices=1)
    if not torch.cuda.is_available() or any(int(value[5:]) >= torch.cuda.device_count() for value in devices):
        raise ValueError('A selected CUDA device is unavailable.')
    return dict(accelerator='gpu', devices=[int(value[5:]) for value in devices])


def migrate_legacy_checkpoint(path, trainer):
    legacy = torch.load(path, map_location='cpu', weights_only=True)
    loops = trainer.fit_loop.state_dict()
    step = legacy['step']
    epoch = legacy['epoch'] - 1
    completed = bool(legacy.get('epoch_complete', True))
    for values in loops['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step'].values():
        values.update(ready=step, completed=step)
    loops['epoch_loop.state_dict']['_batches_that_stepped'] = step
    for values in loops['epoch_progress'].values():
        values.update(ready=epoch + 1, started=epoch + 1, processed=epoch + int(completed), completed=epoch + int(completed))
    if not completed:
        for values in loops['epoch_loop.batch_progress'].values():
            if isinstance(values, dict):
                values.update({key: legacy.get('batch_in_epoch', 0) for key in values})
    checkpoint = dict(state_dict={f'model.{key}': value for key, value in legacy['model'].items()},
                      optimizer_states=[legacy['optimizer']], lr_schedulers=[], epoch=epoch,
                      global_step=step, loops={'fit_loop': loops}, **{'pytorch-lightning_version': pl.__version__})
    for key in ('config', 'embedder_model', 'speaker_count', 'speaker_ids', 'feature_metadata', 'random_states', 'finetune'):
        if key in legacy:
            checkpoint[key] = legacy[key]
    if legacy.get('scaler'):
        checkpoint['MixedPrecision'] = legacy['scaler']
    converted = path.with_name('legacy_lightning.ckpt')
    if trainer.is_global_zero:
        atomic_save(checkpoint, converted)
        print('Migrated legacy trainer checkpoint to Lightning. Mid-epoch replay follows Lightning resume behavior.', flush=True)
    trainer.strategy.barrier()
    return str(converted)


def fit(args, config, root):
    pl.seed_everything(args.seed, workers=True)
    experiment = root / 'logs' / args.model_name
    output = experiment / 'flow'
    settings = config['flow']
    data = FlowDataModule(experiment, root, config, args)
    checkpoint_path = None if args.fresh else latest_checkpoint(output)
    legacy_path = output / 'checkpoint.pth'
    resume_state = torch.load(checkpoint_path or legacy_path, map_location='cpu', weights_only=True) if (
        not args.fresh and (checkpoint_path or legacy_path.exists())
    ) else None
    finetune = bool(resume_state.get('finetune', False)) if resume_state else bool(args.pretrained_flow)
    del resume_state
    task = FlowTask(config, args, data, finetune)
    if not checkpoint_path and (args.fresh or not legacy_path.exists()) and args.pretrained_flow:
        task.load_pretrained(args.pretrained_flow)
    options = trainer_device_options(args, settings)
    strategy = settings['strategy']
    strategy = dict(name=strategy) if isinstance(strategy, str) else dict(strategy)
    name = strategy.pop('name', 'auto')
    devices = options['devices']
    count = torch.cuda.device_count() if devices == 'auto' and options['accelerator'] in {'gpu', 'cuda'} else (len(devices) if isinstance(devices, list) else (devices if isinstance(devices, int) else 1))
    if name == 'ddp' or (name == 'auto' and (count > 1 or settings['num_nodes'] > 1)):
        strategy.setdefault('process_group_backend', 'gloo' if os.name == 'nt' or options['accelerator'] == 'cpu' else 'nccl')
        strategy = DDPStrategy(**strategy)
    elif name == 'auto':
        strategy = name
    else:
        registration = StrategyRegistry[name]
        parameters = dict(registration['init_params'])
        parameters.update(strategy)
        if issubclass(registration['strategy'], DDPStrategy):
            parameters.setdefault('process_group_backend', 'gloo' if os.name == 'nt' or options['accelerator'] == 'cpu' else 'nccl')
        strategy = registration['strategy'](**parameters)
    precision = {'fp32': '32-true', 'fp16': '16-mixed', 'bf16': 'bf16-mixed'}[args.precision]
    has_validation = bool(data.held and settings.get('eval_interval', 0))
    trainer = pl.Trainer(
        **options, num_nodes=settings['num_nodes'], strategy=strategy, precision=precision,
        callbacks=[FlowCheckpoint(output, args, settings, has_validation), FlowPreview(args)],
        logger=TensorBoardLogger(save_dir=str(output), name='lightning_logs', version='latest'),
        gradient_clip_val=settings['grad_clip'],
        val_check_interval=max(1, settings.get('eval_interval', 4000)) * settings['accumulate_grad_batches'],
        check_val_every_n_epoch=None, log_every_n_steps=1,
        max_steps=args.max_updates if args.epochs is None else -1,
        max_epochs=-1 if args.epochs is None else args.epochs,
        use_distributed_sampler=False, num_sanity_val_steps=settings['num_sanity_val_steps'] if has_validation else 0,
        limit_val_batches=1.0 if has_validation else 0,
        accumulate_grad_batches=settings['accumulate_grad_batches'], default_root_dir=str(output),
    )
    if trainer.is_global_zero:
        output.mkdir(parents=True, exist_ok=True)
        if args.fresh:
            old = list(output.glob('*.ckpt')) + list(output.glob(f'{args.model_name}_flow_*.pth'))
            old += list(output.glob(f'{args.model_name}_trainer_*.pth'))
            if legacy_path.exists():
                old.append(legacy_path)
            if old:
                archive = output / f'previous-run-{time.time_ns()}'
                archive.mkdir()
                for path in old:
                    path.rename(archive / path.name)
        (experiment / 'rectified_config.json').write_text(json.dumps(compact_config(config), indent=2) + '\n', encoding='utf-8')
    if not checkpoint_path and legacy_path.exists() and not args.fresh:
        checkpoint_path = migrate_legacy_checkpoint(legacy_path, trainer)
    trainer.fit(task, datamodule=data, ckpt_path=checkpoint_path)
    if trainer.is_global_zero:
        print(f'Finished at Lightning step {trainer.global_step}. Checkpoints and TensorBoard previews: {output}', flush=True)
