import math
from contextlib import nullcontext

import lightning.pytorch as pl
import torch
from lightning.pytorch.strategies import ParallelStrategy
from torch.nn import functional as F
from torchmetrics import MeanMetric

from nsf_hifigan.config import DISCRIMINATOR_PERIODS, max_f0
from nsf_hifigan.losses import (
    discriminate, discriminator_loss, generator_adversarial_loss, mel_l1, vocoder_mel,
)
from nsf_hifigan.models import build_discriminator, build_generator, load_generator_weights

VALIDATION_LOSSES = ('mel_loss', 'stft_loss')


def log_stft(audio, n_fft=2048, hop=512):
    window = torch.hann_window(n_fft, device=audio.device)
    audio = F.pad(audio[:, None], ((n_fft - hop) // 2, (n_fft - hop + 1) // 2), mode='reflect')[:, 0]
    spec = torch.stft(audio, n_fft, hop_length=hop, win_length=n_fft, window=window, center=False,
                      return_complex=True).abs()
    return torch.log10(spec.clamp(min=1e-7))


def spectrogram_image(reference, generated):
    image = torch.cat([reference, generated, (generated - reference).abs() + reference.min()], dim=0).flip(0)
    return ((image - image.min()) / (image.max() - image.min()).clamp(min=1e-5)).cpu()


class VocoderTask(pl.LightningModule):
    def __init__(self, config, finetune=False):
        super().__init__()
        self.config, self.finetune = config, finetune
        self.data, self.settings = config['data'], config['train']
        self.automatic_optimization = False
        self.generator = build_generator(config['model'])
        self.discriminator = build_discriminator(DISCRIMINATOR_PERIODS)
        self.input_mel = vocoder_mel(self.data)
        self.loss_mel = vocoder_mel(self.data, fmax=None)
        self.pc_aug = config['kind'] == 'pc'
        self.max_f0 = max_f0(config['model'])
        self.learning_rate = self.settings['finetune_learning_rate' if finetune else 'learning_rate']
        self.iteration = 0
        self.logged_references = set()
        self.valid_losses = torch.nn.ModuleDict({name: MeanMetric() for name in VALIDATION_LOSSES})

    def load_pretrained(self, source):
        load_generator_weights(self.generator, source['generator'])
        if source['discriminator'] is not None:
            self.discriminator.load_state_dict(source['discriminator'])

    def configure_optimizers(self):
        options = dict(lr=self.learning_rate, betas=tuple(self.settings['betas']),
                       weight_decay=self.settings['weight_decay'])
        return [torch.optim.AdamW(self.generator.parameters(), **options),
                torch.optim.AdamW(self.discriminator.parameters(), **options)]

    def shifted(self, f0, keys):
        return torch.clamp(f0 * 2 ** (keys / 12), max=self.max_f0)

    def render_mel(self, audio):
        with torch.autocast(audio.device.type, enabled=False):
            return self.input_mel(audio.squeeze(1).float())

    def pitch_cycle(self, batch, count):
        mel, f0 = batch['mel'], batch['f0']
        span = self.settings['pc_aug_key']
        key_c = (2 * torch.rand(count, 1, device=f0.device) - 1) * span
        f0_c = self.shifted(f0[:count], key_c)
        mixed = self.generator(mel, torch.cat((f0_c, f0[count:])))
        shift_c, unshifted = mixed[:count], mixed[count:]
        shift_back = self.generator(self.render_mel(shift_c), f0[:count])
        key_min = -span + key_c.clamp(min=0)
        key_max = span + key_c.clamp(max=0)
        key_a = (key_max - key_min) * torch.rand(count, 1, device=f0.device) + key_min
        shift_a = self.generator(mel[:count], self.shifted(f0[:count], key_a))
        shift_ab = self.generator(self.render_mel(shift_a), f0_c)
        return dict(audio=torch.cat((shift_back, unshifted)), shift_c=shift_c, shift_a=shift_a, shift_ab=shift_ab)

    def generate(self, batch):
        count = math.ceil(batch['audio'].shape[0] * self.settings['pc_aug_rate']) if self.pc_aug else 0
        if count:
            outputs = self.pitch_cycle(batch, count)
            fake = torch.cat((outputs['audio'], outputs['shift_c'], outputs['shift_a'], outputs['shift_ab']))
            return outputs, fake
        outputs = dict(audio=self.generator(batch['mel'], batch['f0']))
        return outputs, outputs['audio']

    def discriminator_step(self, optimizer, real, fake):
        optimizer.zero_grad(set_to_none=True)
        logs = {}
        strategy = getattr(self._trainer, 'strategy', None)
        for is_real, audio in ((True, real), (False, fake.detach())):
            sync = strategy.block_backward_sync() if is_real and isinstance(strategy, ParallelStrategy) else nullcontext()
            with sync:
                outputs = discriminate(self.discriminator, audio)
                loss, values = discriminator_loss(outputs if is_real else None, None if is_real else outputs)
                self.manual_backward(loss)
            logs.update(values)
            del outputs, loss
        self.clip(optimizer)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return logs

    def generator_step(self, optimizer, real, outputs, fake):
        optimizer.zero_grad(set_to_none=True)
        parameters = [parameter for parameter in self.discriminator.parameters() if parameter.requires_grad]
        for parameter in parameters:
            parameter.requires_grad = False
        try:
            with torch.no_grad():
                real_output = discriminate(self.discriminator, real)
            adversarial, logs = generator_adversarial_loss(real_output, discriminate(self.discriminator, fake))
            generated, reference = outputs['audio'], real
            pc_loss = 0
            if 'shift_ab' in outputs:
                pc_loss = F.l1_loss(outputs['shift_ab'], outputs['shift_c']) * self.settings['pc_wav_loss_weight']
                logs['generator/pc_wav_loss'] = pc_loss.detach()
                generated = torch.cat((generated, outputs['shift_ab']))
                reference = torch.cat((reference, outputs['shift_c']))
            mel_loss = mel_l1(self.loss_mel, generated, reference) * self.settings['mel_loss_weight']
            logs['generator/mel_loss'] = mel_loss.detach()
            self.manual_backward(adversarial + mel_loss + pc_loss)
            self.clip(optimizer)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        finally:
            for parameter in parameters:
                parameter.requires_grad = True
        return logs

    def clip(self, optimizer):
        if self.settings['grad_clip']:
            self.clip_gradients(optimizer, gradient_clip_val=self.settings['grad_clip'], gradient_clip_algorithm='norm')

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()
        optimizer_g.zero_grad(set_to_none=True)
        warming = self.iteration < self.settings['discriminator_warmup']
        with torch.set_grad_enabled(not warming):
            outputs, fake = self.generate(batch)
        logs = self.discriminator_step(optimizer_d, batch['audio'], fake)
        if not warming:
            logs.update(self.generator_step(optimizer_g, batch['audio'], outputs, fake))
        self.iteration += 1
        self.log_dict({key.split('/')[-1]: value for key, value in logs.items()
                       if key in ('generator/mel_loss', 'discriminator/mpd_fake')},
                      prog_bar=True, logger=False, on_step=True, on_epoch=False)
        if self.iteration % self.settings['log_interval'] == 0 and self.trainer.is_global_zero:
            self.logger.log_metrics({f'training/{key}': float(value) for key, value in logs.items()},
                                    step=self.iteration)

    def on_validation_start(self):
        for metric in self.valid_losses.values():
            metric.reset()

    def validation_step(self, batch, batch_idx):
        with torch.autocast(self.device.type, enabled=False):
            generated = self.generator(batch['mel'].float(), batch['f0'].float())
            real = batch['audio']
            mel_loss = mel_l1(self.loss_mel, generated, real)
            generated_stft, real_stft = log_stft(generated[:, 0].float()), log_stft(real[:, 0].float())
            stft_loss = F.l1_loss(generated_stft, real_stft)
        self.valid_losses['mel_loss'].update(mel_loss)
        self.valid_losses['stft_loss'].update(stft_loss)
        if self.trainer.sanity_checking or not self.trainer.is_global_zero:
            return
        if batch_idx < self.settings['num_valid_plots']:
            writer, rate = self.logger.experiment, self.data['sample_rate']
            writer.add_audio(f'audio/generated_{batch_idx}', generated[0].clamp(-1, 1).cpu(), self.iteration, rate)
            if batch_idx not in self.logged_references:
                writer.add_audio(f'audio/reference_{batch_idx}', real[0].cpu(), self.iteration, rate)
                self.logged_references.add(batch_idx)
            writer.add_image(f'spectrogram/{batch_idx}', spectrogram_image(real_stft[0], generated_stft[0]),
                             self.iteration, dataformats='HW')

    def on_validation_epoch_end(self):
        losses = {name: metric.compute() for name, metric in self.valid_losses.items()}
        self.log('val_loss', losses['mel_loss'], prog_bar=True, logger=False, sync_dist=True)
        if not self.trainer.sanity_checking and self.trainer.is_global_zero:
            self.logger.log_metrics({f'validation/{key}': float(value) for key, value in losses.items()},
                                    step=self.iteration)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['vocoder'] = dict(config=self.config, iteration=self.iteration, finetune=self.finetune)

    def on_load_checkpoint(self, checkpoint):
        saved = checkpoint['vocoder']
        if saved['config']['model'] != self.config['model'] or saved['config']['kind'] != self.config['kind']:
            raise ValueError('The checkpoint uses another vocoder architecture than this experiment.')
        self.iteration = int(saved['iteration'])
