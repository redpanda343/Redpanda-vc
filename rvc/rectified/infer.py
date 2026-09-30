import math

import numpy as np
import soxr
import torch
from torch.nn import functional as F

from rvc.infer.pipeline import Pipeline, _INFERENCE_RNG_LOCK
from rvc.lib.utils import extract_embedding_features
from rvc.rectified.aperiodicity import aperiodicity
from rvc.rectified.data import f0_to_mel_rate, smooth_curve, to_mel_rate, upsample_content
from rvc.rectified.energy import frame_energy
from rvc.rectified.resources import default_vocoder
from rvc.rectified.vocoder import load_vocoder


def is_rectified(checkpoint):
    config = checkpoint.get('config')
    return checkpoint.get('kind') == 'rectified_flow' or (
        isinstance(config, dict) and 'flow' in config and 'data' in config and 'model' in checkpoint
    )


class RectifiedPipeline(Pipeline):
    def __init__(self, sample_rate, config, checkpoint, vocoder_path=''):
        super().__init__(sample_rate, config)
        self.data = checkpoint['config']['data']
        self.content_channels = int(checkpoint['config']['flow']['model']['content_channels'])
        self.vocoder_path = vocoder_path
        self.checkpoint_vocoder = checkpoint.get('vocoder', '')
        self.vocoder_model = None
        self.pitch_shift = 0.0

    def pipeline(self, *args, **kwargs):
        self.pitch_shift = float(kwargs.get('pitch', args[4] if len(args) > 4 else 0))
        return super().pipeline(*args, **kwargs)

    def set_vocoder(self, path):
        path = str(path or '').strip().strip('"')
        if path != self.vocoder_path:
            self.vocoder_model = None
            self.vocoder_path = path

    @torch.inference_mode()
    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index,
                         big_npy, index_rate, version, protect, inference_rng=None):
        if self.vocoder_model is None:
            path = self.vocoder_path or default_vocoder(self.checkpoint_vocoder)
            vocoder, _ = load_vocoder(path, self.data)
            self.vocoder_model = vocoder.to(self.device).float()
        speaker = int(sid.item())
        if not 0 <= speaker < net_g.speaker_count:
            raise ValueError(f'Speaker ID must be between 0 and {net_g.speaker_count - 1}.')
        source = torch.from_numpy(np.asarray(audio0, dtype=np.float32)).view(1, -1).to(self.device)
        if getattr(model, 'audio_requires_normalization', False):
            source = F.layer_norm(source, source.shape[-1:])
        content = extract_embedding_features(model, source, 'v2').float()
        if content.shape[-1] != self.content_channels:
            raise ValueError(f'The flow expects {self.content_channels} content channels, got {content.shape[-1]}.')
        original = content
        if index is not None:
            content = self._retrieve_speaker_embeddings(content, index, big_npy, index_rate)
        mode = self.data['content_interpolation']
        content = upsample_content(content, mode)
        if index is not None and protect < .5:
            original = upsample_content(original, mode)
            voiced = (pitchf > 0).float()
            voiced = F.interpolate(voiced.unsqueeze(1), size=content.shape[1], mode='nearest').transpose(1, 2)
            amount = voiced + (1 - voiced) * float(protect)
            content = content * amount + original * (1 - amount)
        rate, hop = int(self.data['sample_rate']), int(self.data['hop_length'])
        waveform = soxr.resample(np.asarray(audio0, dtype=np.float32), self.sample_rate, rate, quality='HQ')
        waveform = torch.from_numpy(waveform).view(1, -1).to(self.device)
        length = waveform.shape[-1]
        if len(audio0) == (pitchf.shape[-1] + 1) * self.window:
            length = round(pitchf.shape[-1] * self.window * rate / self.sample_rate)
            waveform = waveform[..., :length]
        frames = math.ceil(length / hop)
        feature_frames = max(1, length // (rate // 100))
        source_f0 = pitchf.float() / (2 ** (self.pitch_shift / 12))
        energy = smooth_curve(frame_energy(waveform, rate, feature_frames))
        breathiness = smooth_curve(aperiodicity(waveform, rate, source_f0, feature_frames))
        content = to_mel_rate(content, frames, rate, hop)
        f0 = f0_to_mel_rate(pitchf.float(), frames, rate, hop)
        energy = to_mel_rate(energy.unsqueeze(-1), frames, rate, hop)[..., 0]
        breathiness = to_mel_rate(breathiness.unsqueeze(-1), frames, rate, hop)[..., 0]
        mask = torch.ones(1, 1, frames, device=self.device)
        with _INFERENCE_RNG_LOCK:
            if inference_rng is not None:
                inference_rng.seed_next_segment()
            mel = net_g.sample(content, f0, energy, sid, mask, steps=16, breathiness=breathiness)
            audio = self.vocoder_model(mel, f0)[0, 0, :length]
        if not torch.isfinite(audio).all():
            raise FloatingPointError('Non-finite Rectified Flow audio output.')
        return audio.float().cpu().numpy()
