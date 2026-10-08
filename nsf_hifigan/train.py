import argparse
import logging
import os
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from tqdm import tqdm

from nsf_hifigan.checkpoints import (
    checkpoint_path, export_vocoder, latest_checkpoint, load_source, prune_checkpoints,
)
from nsf_hifigan.config import (
    KINDS, PRECISIONS, ROOT, apply_overrides, experiment_paths, new_config, read_json, validate, write_json,
)
from nsf_hifigan.data import VocoderData
from nsf_hifigan.task import VocoderTask
from rectified_flow.distributed import parse_devices

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))
for _name in ('lightning.pytorch', 'lightning.fabric'):
    logging.getLogger(_name).setLevel(logging.WARNING)


class VocoderSchedule(pl.Callback):
    def __init__(self, output, model_name, max_updates):
        self.output, self.model_name, self.max_updates = Path(output), model_name, max_updates
        self.last_saved = None
        self.bar = None

    def save(self, trainer, task):
        if self.last_saved == task.iteration:
            return
        if any(not torch.isfinite(value).all() for value in task.generator.state_dict().values()):
            raise FloatingPointError('Non-finite generator weights. Lower the learning rate or train in fp32.')
        trainer.save_checkpoint(checkpoint_path(self.output, task.iteration))
        self.last_saved = task.iteration
        if trainer.is_global_zero:
            prune_checkpoints(self.output, task.settings['num_ckpt_keep'])

    def on_train_start(self, trainer, task):
        self.last_saved = task.iteration
        if trainer.is_global_zero:
            self.bar = tqdm(total=self.max_updates, initial=min(task.iteration, self.max_updates), desc='Training',
                            unit='step', dynamic_ncols=True)
        if task.iteration >= self.max_updates:
            trainer.should_stop = True

    def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx):
        if self.bar is not None:
            metrics = trainer.progress_bar_metrics
            self.bar.set_postfix({name: f'{float(value):.4f}' for name, value in metrics.items()}, refresh=False)
            self.bar.update(1)
        if task.iteration % task.settings['checkpoint_interval'] == 0:
            self.save(trainer, task)
        if task.iteration >= self.max_updates:
            trainer.should_stop = True

    def on_train_end(self, trainer, task):
        self.save(trainer, task)
        self.close()
        if trainer.is_global_zero and task.iteration >= self.max_updates:
            exported = export_vocoder(checkpoint_path(self.output, task.iteration), name=self.model_name)
            print(f'Exported the vocoder to {exported.parent}. Select it as a Rectified Flow vocoder.', flush=True)

    def on_exception(self, trainer, task, exception):
        self.close()

    def close(self):
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def device_options(device):
    if str(device).strip().lower() == 'auto':
        return dict(accelerator='gpu' if torch.cuda.is_available() else 'cpu', devices='auto')
    devices = parse_devices(device)
    if devices == ['cpu']:
        return dict(accelerator='cpu', devices=1)
    if not torch.cuda.is_available() or any(int(value[5:]) >= torch.cuda.device_count() for value in devices):
        raise ValueError('A selected CUDA device is unavailable.')
    return dict(accelerator='gpu', devices=[int(value[5:]) for value in devices])


def device_count(options):
    devices = options['devices']
    if devices == 'auto':
        return torch.cuda.device_count() if options['accelerator'] == 'gpu' else 1
    return len(devices) if isinstance(devices, list) else devices


def prepare_config(paths, args, resuming):
    index = read_json(paths['index'])
    if index is None:
        raise ValueError('Vocoder features are missing. Preprocess the dataset first.')
    source = None
    if resuming:
        config = read_json(paths['config'])
        if config is None:
            raise ValueError(f"{paths['config']} is missing; it is needed to resume training.")
        if config['kind'] != args.kind:
            raise ValueError(f"This experiment trains a {KINDS[config['kind']]} vocoder. Choose that type to resume, "
                             'or use another model name.')
    else:
        if args.pretrained:
            source = load_source(args.pretrained, index['data'])
        config = new_config(args.kind, index['data'], source['hparams'] if source else None, args.pretrained or None)
        if source:
            config['train']['max_updates'] = config['train']['finetune_max_updates']
            if source['discriminator'] is None:
                config['train']['discriminator_warmup'] = max(0, args.discriminator_warmup)
    learning_rate = 'finetune_learning_rate' if config['pretrained'] else 'learning_rate'
    apply_overrides(config, {'batch_size': args.batch_size, 'crop_mel_frames': args.crop_mel_frames,
                             learning_rate: args.learning_rate, 'key_aug': args.key_aug,
                             'max_updates': args.max_updates, 'checkpoint_interval': args.checkpoint_interval,
                             'eval_interval': args.checkpoint_interval, 'num_workers': args.workers})
    return validate(config), source


def fit(args):
    if Path(args.model_name).name != args.model_name or args.model_name in {'.', '..'}:
        raise ValueError('Use a model name, not a path.')
    paths = experiment_paths(ROOT / 'logs' / args.model_name)
    resume = latest_checkpoint(paths['output'])
    config, source = prepare_config(paths, args, resume is not None)
    finetune = bool(config['pretrained'])
    pl.seed_everything(args.seed, workers=True)
    data = VocoderData(paths, config, args.seed)
    task = VocoderTask(config, finetune)
    if source is not None:
        task.load_pretrained(source)
        print(f"Fine-tuning from {source['path']}"
              + ('.' if source['discriminator'] is not None else
                 f" (generator only; the discriminator warms up for {config['train']['discriminator_warmup']} steps)."),
              flush=True)
    options = device_options(args.device)
    strategy = 'auto'
    if device_count(options) > 1:
        backend = 'gloo' if os.name == 'nt' or options['accelerator'] == 'cpu' else 'nccl'
        strategy = DDPStrategy(process_group_backend=backend, find_unused_parameters=True)
    has_validation = len(data.valid_dataset) > 0
    max_updates = config['train']['max_updates']
    trainer = pl.Trainer(
        **options, strategy=strategy, precision=PRECISIONS[args.precision], min_epochs=0,
        callbacks=[VocoderSchedule(paths['output'], args.model_name, max_updates)],
        enable_progress_bar=False, enable_model_summary=False, enable_checkpointing=False,
        logger=TensorBoardLogger(save_dir=str(paths['output']), name='lightning_logs', version='latest'),
        val_check_interval=config['train']['eval_interval'], check_val_every_n_epoch=None, log_every_n_steps=1,
        max_steps=-1, max_epochs=-1, num_sanity_val_steps=1 if has_validation else 0,
        limit_val_batches=1.0 if has_validation else 0, default_root_dir=str(paths['output']),
    )
    if trainer.is_global_zero:
        paths['output'].mkdir(parents=True, exist_ok=True)
        write_json(paths['config'], config)
    trainer.fit(task, datamodule=data, ckpt_path=str(resume) if resume else None)
    if trainer.is_global_zero:
        print(f'Finished at step {task.iteration}. Checkpoints and TensorBoard logs: {paths["output"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description='Train or fine-tune an OpenVPI NSF-HiFiGAN or PC-NSF-HiFiGAN vocoder.')
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--kind', choices=sorted(KINDS), default='pc',
                        help='pc trains PC-NSF-HiFiGAN (mini-NSF with pitch-cycle augmentation); nsf trains the '
                             'classic NSF-HiFiGAN.')
    parser.add_argument('--pretrained', default='',
                        help='Fine-tune source for a new experiment: a Rectified Flow vocoder name or a '
                             '.ckpt/.onnx path. Ignored when resuming.')
    parser.add_argument('--discriminator-warmup', type=int, default=1000,
                        help='Steps that train only the discriminator when the source has no discriminator weights.')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--crop-mel-frames', type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--max-updates', type=int)
    parser.add_argument('--checkpoint-interval', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--key-aug', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--precision', choices=sorted(PRECISIONS), default='fp32')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda:0 or cuda:0,cuda:1')
    parser.add_argument('--seed', type=int, default=1234)
    fit(parser.parse_args())


if __name__ == '__main__':
    main()
