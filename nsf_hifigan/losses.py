import torch
from torch.nn import functional as F

from rectified_flow.mel import LogMel

DISCRIMINATORS = ('msd', 'mpd')


def vocoder_mel(data, fmax='config'):
    return LogMel(data['sample_rate'], data['n_fft'], data['win_length'], data['hop_length'], data['n_mels'],
                  data['mel_fmin'], data['mel_fmax'] if fmax == 'config' else fmax)


def discriminate(discriminator, audio):
    return {name: discriminator[name](audio) for name in DISCRIMINATORS}


def discriminator_loss(real, fake):
    loss = 0
    logs = {}
    for name in DISCRIMINATORS:
        real_loss = sum(torch.mean((1 - output) ** 2) for output in real[name][0])
        fake_loss = sum(torch.mean(output ** 2) for output in fake[name][0])
        loss = loss + real_loss + fake_loss
        logs[f'discriminator/{name}_real'] = real_loss.detach()
        logs[f'discriminator/{name}_fake'] = fake_loss.detach()
    return loss, logs


def feature_loss(real_features, fake_features):
    loss = 0
    for real_maps, fake_maps in zip(real_features, fake_features):
        for real_map, fake_map in zip(real_maps, fake_maps):
            count = min(real_map.shape[0], fake_map.shape[0])
            loss = loss + torch.mean(torch.abs(real_map[:count] - fake_map[:count]))
    return loss * 2


def generator_adversarial_loss(real, fake):
    loss = 0
    logs = {}
    for name in DISCRIMINATORS:
        adversarial = sum(torch.mean((1 - output) ** 2) for output in fake[name][0])
        features = feature_loss(real[name][1], fake[name][1])
        loss = loss + adversarial + features
        logs[f'generator/{name}_adversarial'] = adversarial.detach()
        logs[f'generator/{name}_feature'] = features.detach()
    return loss, logs


def mel_l1(mel, generated, reference):
    count = min(generated.shape[0], reference.shape[0])
    with torch.autocast(generated.device.type, enabled=False):
        return F.l1_loss(mel(generated[:count].squeeze(1).float()), mel(reference[:count].squeeze(1).float()))
