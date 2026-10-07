import math

import numpy as np
import soxr
import torch
from torch.nn import functional as F

from rvc.infer.pipeline import Pipeline, _INFERENCE_RNG_LOCK
from rvc.lib.utils import extract_embedding_features
from rvc.lib.predictors.f0 import RMVPE
from rvc.rectified.pitch import parselmouth_pitch, resample_f0, resample_voicing, rmvpe_pitch
from rvc.rectified.resources import default_vocoder
from rvc.rectified.variance import extract_variances
from rvc.rectified.vocoder import load_vocoder


def is_rectified(checkpoint):
    return checkpoint.get('kind') == 'rectified_flow'


class RectifiedPipeline(Pipeline):
    high_pass = False

    def __init__(self, sample_rate, config, checkpoint, vocoder_path=''):
        super().__init__(sample_rate, config)
        self.data = checkpoint['config']['data']
        self.settings = checkpoint['config']['flow']
        self.content_channels = int(checkpoint['config']['flow']['model']['content_channels'])
        self.vocoder_path = vocoder_path
        self.vocoder_model = None
        self.pitch_shift = 0.0
        self.f0_method = 'pm'
        self.source_pad = None

    def pipeline(self, *args, source_audio=None, **kwargs):
        self.pitch_shift = float(kwargs.get('pitch', args[4] if len(args) > 4 else 0))
        self.f0_method = kwargs.get('f0_method', args[5] if len(args) > 5 else 'pm')
        if source_audio is not None:
            pad = self.t_pad * int(self.data['sample_rate']) // self.sample_rate
            self.source_pad = np.pad(np.asarray(source_audio, dtype=np.float32), (pad, pad), mode='reflect')
        try:
            return super().pipeline(*args, **kwargs)
        finally:
            self.source_pad = None

    def set_vocoder(self, path):
        path = str(path or '').strip().strip('"')
        if path != self.vocoder_path:
            self.vocoder_model = None
            self.vocoder_path = path

    @torch.inference_mode()
    def voice_conversion(self, model, net_g, sid, audio0, pitch, pitchf, index,
                         big_npy, index_rate, version, protect, inference_rng=None, segment_start=0):
        if self.vocoder_model is None:
            vocoder, _ = load_vocoder(self.vocoder_path or default_vocoder(), self.data)
            self.vocoder_model = vocoder.to(self.device).float()
        speaker = int(sid.item())
        if net_g.use_spk_id and not 0 <= speaker < net_g.speaker_count:
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
            if protect < .5:
                voiced = (pitchf > 0).float()
                voiced = F.interpolate(voiced.unsqueeze(1), size=content.shape[1], mode='nearest').transpose(1, 2)
                amount = voiced + (1 - voiced) * float(protect)
                content = content * amount + original * (1 - amount)
        rate, hop = int(self.data['sample_rate']), int(self.data['hop_length'])
        if self.source_pad is not None:
            length = round(len(audio0) * rate / self.sample_rate)
        else:
            waveform = soxr.resample(np.asarray(audio0, dtype=np.float32), self.sample_rate, rate, quality='HQ')
            length = len(waveform)
        if len(audio0) == (pitchf.shape[-1] + 1) * self.window:
            length = round(pitchf.shape[-1] * self.window * rate / self.sample_rate)
        if self.source_pad is not None:
            start = round(segment_start * rate / self.sample_rate)
            waveform = self.source_pad[start:start + length]
            waveform = np.pad(waveform, (0, length - len(waveform)))
        else:
            waveform = waveform[:length]
        frames = math.ceil(length / hop)
        if self.f0_method in {'pm', 'rmvpe'}:
            if self.f0_method == 'pm':
                f0, uv = parselmouth_pitch(waveform, rate, hop, frames)
            else:
                if not hasattr(self, 'model_rmvpe'):
                    self.model_rmvpe = RMVPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
                f0, uv = rmvpe_pitch(self.model_rmvpe.model, waveform, rate, hop, frames)
        else:
            source_f0 = pitchf[0].float().cpu().numpy()
            uv = ~resample_voicing(source_f0 > 0, self.sample_rate / self.window, frames, rate / hop)
            f0 = resample_f0(source_f0, self.sample_rate / self.window, frames, rate / hop) / 2 ** (self.pitch_shift / 12)
        variances = None
        if net_g.variance_names:
            variances = extract_variances(net_g.variance_names, waveform, f0, uv, frames, self.data, self.settings,
                                          self.device)
            variances = torch.from_numpy(variances).to(self.device)[None]
        f0 = torch.from_numpy(f0).to(self.device)[None] * 2 ** (self.pitch_shift / 12)
        mask = torch.ones(1, 1, frames, device=self.device)
        with _INFERENCE_RNG_LOCK:
            if inference_rng is not None:
                inference_rng.seed_next_segment()
            mel = net_g.sample(content, f0, sid, mask, variances=variances)
            audio = self.vocoder_model(mel, f0)[0, 0, :length]
        if not torch.isfinite(audio).all():
            raise FloatingPointError('Non-finite Rectified Flow audio output.')
        return audio.float().cpu().numpy()
