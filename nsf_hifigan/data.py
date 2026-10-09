import math
import random

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import resample

from nsf_hifigan.config import max_f0, read_json
from nsf_hifigan.losses import vocoder_mel

MIN_LOG_MEL = math.log(1e-5)
KEY_AUG_MARGIN = 2


class VocoderDataset(Dataset):
    def __init__(self, directory, entries, config, training):
        super().__init__()
        self.directory = directory
        self.data, self.settings = config['data'], config['train']
        self.training = training
        self.crop = self.settings['crop_mel_frames']
        self.entries = [name for name, frames in entries if not training or frames >= self.crop]
        self.max_f0 = max_f0(config['model'])
        self.mel = vocoder_mel(self.data) if training and self.settings['key_aug'] else None

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        for _ in range(10):
            sample = self.load(index)
            if sample['f0'].max() < self.max_f0:
                return sample
            index = random.randrange(len(self))
        sample['f0'] = np.minimum(sample['f0'], self.max_f0 - 1)
        return sample

    def load(self, index):
        with np.load(self.directory / self.entries[index]) as item:
            sample = dict(audio=item['audio'], mel=item['mel'], f0=item['f0'])
        if self.mel is not None and random.random() < self.settings['key_aug_prob']:
            return self.speed_augment(sample)
        return sample

    def speed_augment(self, sample):
        hop = self.data['hop_length']
        speed = random.uniform(self.settings['key_aug_min'], self.settings['key_aug_max'])
        frames = int(math.ceil((self.crop + 2 * KEY_AUG_MARGIN) * speed))
        available = len(sample['mel'])
        if frames >= available:
            return sample
        start = random.randint(0, available - frames)
        audio = torch.from_numpy(np.ascontiguousarray(sample['audio'][start * hop:(start + frames) * hop]))
        if len(audio) < frames * hop:
            return sample
        audio = resample(audio, int(round(hop * speed)), hop, lowpass_filter_width=128)
        with torch.no_grad():
            mel = self.mel(audio[None])[0].T.numpy()
        positions = start + (np.arange(len(mel)) + 0.5) * speed - 0.5
        f0 = np.interp(positions, np.arange(available), sample['f0']).astype(np.float32) * speed
        margin = KEY_AUG_MARGIN
        return dict(audio=audio[margin * hop:-margin * hop].numpy(), mel=mel[margin:-margin], f0=f0[margin:-margin])

    def collate(self, batch):
        hop = self.data['hop_length']
        records = []
        for sample in batch:
            frames = len(sample['mel'])
            if self.training:
                if frames < self.crop:
                    continue
                start = random.randint(0, frames - 1 - self.crop) if frames > self.crop else 0
                end = start + self.crop
            else:
                start, end = 0, frames
            audio = sample['audio'][start * hop:end * hop]
            audio = np.pad(audio, (0, (end - start) * hop - len(audio))).astype(np.float32)
            mel = sample['mel'][start:end].T.astype(np.float32)
            if self.training and random.random() < self.settings['volume_aug_prob']:
                peak = float(np.max(np.abs(audio))) + 1e-5
                shift = random.uniform(-3.0, min(3.0, float(np.log(1 / peak))))
                audio = audio * np.exp(shift)
                mel = mel + shift
            records.append(dict(audio=audio, mel=np.maximum(mel, MIN_LOG_MEL), f0=sample['f0'][start:end]))
        if not records:
            raise ValueError('Every clip in the batch is shorter than the crop length.')
        stacked = {key: torch.from_numpy(np.stack([record[key] for record in records]).astype(np.float32))
                   for key in ('audio', 'mel', 'f0')}
        stacked['audio'] = stacked['audio'][:, None]
        return stacked


class VocoderData(pl.LightningDataModule):
    def __init__(self, paths, config, seed):
        super().__init__()
        index = read_json(paths['index'])
        if index is None:
            raise ValueError('Vocoder features are missing. Extract the features first.')
        if index['data'] != config['data']:
            raise ValueError('The extracted features use another mel configuration. Extract the features again.')
        self.directory, self.config, self.seed = paths['data'], config, seed
        self.pitch_extractor = index['pitch_extractor']
        self.train_dataset = VocoderDataset(self.directory, index['train'], config, True)
        self.valid_dataset = VocoderDataset(self.directory, index['valid'], config, False)
        if not len(self.train_dataset):
            raise ValueError(f"No training clip is at least {config['train']['crop_mel_frames']} mel frames long.")

    def loader(self, dataset, training):
        settings = self.config['train']
        workers = settings['num_workers']
        options = dict(num_workers=workers, persistent_workers=workers > 0, collate_fn=dataset.collate)
        if workers:
            options['prefetch_factor'] = settings['prefetch_factor']
        if training:
            return DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True, drop_last=True,
                              pin_memory=True, **options)
        return DataLoader(dataset, batch_size=1, shuffle=False, **options)

    def train_dataloader(self):
        if len(self.train_dataset) < self.config['train']['batch_size']:
            raise ValueError(f"The dataset has {len(self.train_dataset)} usable clip(s), fewer than the batch size "
                             f"{self.config['train']['batch_size']}. Lower the batch size or add audio.")
        return self.loader(self.train_dataset, True)

    def val_dataloader(self):
        return self.loader(self.valid_dataset, False) if len(self.valid_dataset) else []
