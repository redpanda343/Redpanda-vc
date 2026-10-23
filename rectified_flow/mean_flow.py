import math

import numpy as np
import torch
from torch import nn

from rectified_flow.content_encoder import ContentConditionEncoder
from rectified_flow.flow_model import LYNXNet2Backbone, timestep_embedding
from rectified_flow.variance import variance_names


def validate_mean_flow_config(model):
    steps = model['sampling_steps']
    if model['sampling_method'] != 'mean' or isinstance(steps, bool) or not isinstance(steps, int) or steps not in (1, 2):
        raise ValueError('MeanFlow requires the mean sampler with one or two steps.')
    if (model['dual_timestep'] or model['train_aux_decoder'] or not model['train_diffusion'] or
            model['val_gt_start'] or model['t_start'] != 0.0 or model['t_start_infer'] != 0.0):
        raise ValueError('MeanFlow requires noise-to-mel training without shallow starts or an auxiliary decoder.')
    settings = model['mean_flow_args']
    for name in ('flow_ratio', 'cfg_ratio', 'loss_p'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'MeanFlow {name} must be between zero and one.')
    for name in ('time_sigma', 'cfg_scale', 'loss_eps'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'MeanFlow {name} must be finite and positive.')
    value = settings['time_mu']
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('MeanFlow time_mu must be finite.')


class MeanFlowBackbone(LYNXNet2Backbone):
    def __init__(self, n_mels, cond_channels, **settings):
        super().__init__(n_mels, cond_channels, **settings)
        channels = self.channels
        self.r_time_mlp = nn.Sequential(
            nn.Linear(channels, channels * 4), nn.GELU(), nn.Linear(channels * 4, channels),
        )

    def forward(self, x, t, r, cond, mask, prepared=None, unconditional=False, cfg_mask=None):
        if unconditional:
            cond = torch.zeros_like(cond)
            prepared = None
        elif cfg_mask is not None:
            cond = torch.where(cfg_mask[:, None, None], torch.zeros_like(cond), cond)
            prepared = None
        time = self.time_mlp(timestep_embedding(t, self.channels))
        time = time + self.r_time_mlp(timestep_embedding(r, self.channels))
        valid = mask.transpose(1, 2)
        h = self.input((x * mask).transpose(1, 2))
        h = (h + (self.prepare_conditioning(cond) if prepared is None else prepared) + time[:, None]) * valid
        for layer in self.layers:
            h = layer(h, fused=False) * valid
        return self.output(self.norm(h)).transpose(1, 2) * mask


class MeanFlow(nn.Module):
    def __init__(self, n_mels, speaker_count, content_step, mel_mean, mel_std, mean_flow_args,
                 hidden_channels=256, encoder_layers=4, backbone_args=None, **settings):
        super().__init__()
        if not math.isfinite(mel_mean) or not math.isfinite(mel_std) or mel_std <= 0:
            raise ValueError('Mel normalization must be finite with a positive scale.')
        self.n_mels = int(n_mels)
        self.mel_mean = float(mel_mean)
        self.mel_std = float(mel_std)
        self.mean_flow = True
        self.use_spk_id = settings['use_spk_id']
        self.encoder = ContentConditionEncoder(
            settings['content_channels'], hidden_channels, speaker_count, encoder_layers, content_step,
            key_shift=settings['key_shift'], speed=settings['speed'], use_spk_id=self.use_spk_id,
            enc_ffn_kernel_size=settings['enc_ffn_kernel_size'], use_rope=settings['use_rope'],
            rope_interleaved=settings['rope_interleaved'], rope_theta=settings['rope_theta'],
            variances=variance_names(settings),
        )
        self.backbone = MeanFlowBackbone(n_mels, hidden_channels, **(backbone_args or {}))
        self.mean_flow_args = dict(mean_flow_args)
        self.sampling_method = settings['sampling_method']
        self.sampling_steps = int(settings['sampling_steps'])
        self.t_start = self.t_start_infer = 0.0
        self.train_aux_decoder = False
        self.train_diffusion = True
        self.val_gt_start = False

    @property
    def speaker_count(self):
        return self.encoder.speaker_count

    @property
    def variance_names(self):
        return self.encoder.variance_names

    def sample_t_r(self, batch_size, device, return_mean_indices=False):
        settings = self.mean_flow_args
        normal = np.random.randn(batch_size, 2).astype(np.float32) * settings['time_sigma'] + settings['time_mu']
        samples = 1.0 / (1.0 + np.exp(-normal))
        t = np.maximum(samples[:, 0], samples[:, 1])
        r = np.minimum(samples[:, 0], samples[:, 1])
        indices = np.random.permutation(batch_size)[:int(settings['flow_ratio'] * batch_size)]
        r[indices] = t[indices]
        times = torch.tensor(t, device=device), torch.tensor(r, device=device)
        if return_mean_indices:
            return (*times, torch.tensor(np.flatnonzero(t != r), device=device))
        return times

    def velocity_jvp(self, z, t, r, direction, cond, mask, cfg_mask):
        with torch.autocast(device_type=z.device.type, enabled=False):
            return torch.func.jvp(
                lambda value, time, end: self.backbone(value, time, end, cond.float(), mask, cfg_mask=cfg_mask),
                (z, t, r), (direction, torch.ones_like(t), torch.zeros_like(r)),
            )

    def loss(self, mel, cond, mask):
        settings = self.mean_flow_args
        t, r, mean_indices = self.sample_t_r(mel.shape[0], mel.device, return_mean_indices=True)
        noise = torch.randn_like(mel)
        z = (1.0 - t[:, None, None]) * mel + t[:, None, None] * noise
        velocity = noise - mel
        with torch.no_grad():
            if settings['cfg_scale'] == 1.0:
                direction = velocity
            else:
                unconditional = self.backbone(z, t, t, cond, mask, unconditional=True)
                direction = settings['cfg_scale'] * velocity + (1.0 - settings['cfg_scale']) * unconditional.float()
        cfg_mask = torch.rand(mel.shape[0], device=mel.device) < settings['cfg_ratio']
        if any(layer.dropout.p for layer in self.backbone.layers):
            prediction, derivative = self.velocity_jvp(z, t, r, direction, cond, mask, cfg_mask)
            target = direction - (t - r)[:, None, None] * derivative.detach()
        else:
            prediction = self.backbone(z, t, r, cond, mask, cfg_mask=cfg_mask)
            target = direction.clone()
            if mean_indices.numel():
                with torch.no_grad():
                    _, derivative = self.velocity_jvp(
                        z[mean_indices], t[mean_indices], r[mean_indices], direction[mean_indices],
                        cond[mean_indices], mask[mean_indices], cfg_mask[mean_indices],
                    )
                    target[mean_indices] -= (t - r)[mean_indices, None, None] * derivative
        error = prediction.float() - target.detach()
        squared = error.square().mean(dim=1)
        weight = (squared + settings['loss_eps']).pow(-settings['loss_p']).detach()
        valid = mask[:, 0].bool()
        return torch.where(valid, weight * squared, 0.0).sum() / valid.sum()

    def forward(self, mel, content, f0, speaker, mask, content_mask=None, key_shift=None, speed=None, variances=None):
        cond = self.encoder(content.float(), f0.float(), speaker, mask, content_mask, key_shift, speed, variances)
        return self.loss(mel.float(), cond, mask), None

    @torch.no_grad()
    def sample(self, content, f0, speaker, mask, content_mask=None, key_shift=None, speed=None,
               steps=None, method=None, noise=None, source_mel=None, variances=None, start=0):
        method = self.sampling_method if method is None else method
        steps = self.sampling_steps if steps is None else steps
        if method != 'mean' or isinstance(steps, bool) or not isinstance(steps, int) or steps not in (1, 2):
            raise ValueError('MeanFlow sampling requires the mean sampler and one or two steps.')
        if source_mel is not None:
            raise ValueError('MeanFlow samples from noise; source-mel starts are unavailable.')
        steps = int(steps)
        with torch.autocast(device_type=content.device.type, enabled=False):
            cond = self.encoder(content.float(), f0.float(), speaker, mask, content_mask, key_shift, speed, variances)
            if start:
                cond, mask = cond[..., start:], mask[..., start:]
            prepared = self.backbone.prepare_conditioning(cond) if self.backbone.use_conditioner_cache else None
            shape = (content.shape[0], self.n_mels, mask.shape[-1])
            x = torch.randn(shape, device=content.device) if noise is None else noise.float()
            if x.shape != shape:
                raise ValueError('MeanFlow noise must match the batch, mel channels and sampled frame count.')
            times = torch.linspace(1.0, 0.0, steps + 1, device=content.device)
            for index in range(steps):
                t = times[index].expand(content.shape[0])
                r = times[index + 1].expand(content.shape[0])
                velocity = self.backbone(x, t, r, cond, mask, prepared=prepared)
                x = x - (times[index] - times[index + 1]) * velocity
            return x * mask
