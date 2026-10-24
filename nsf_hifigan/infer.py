from contextlib import nullcontext
import math

import numpy as np
import torch

from nsf_hifigan.config import openvpi_config
from rectified_flow.mel import LogMel
from rectified_flow.openvpi import NSFHiFiGAN, generator_state, openvpi_spec
from rectified_flow.pitch import extract_pitch


def load_checkpoint(path):
    if str(path).lower().endswith('.safetensors'):
        from safetensors.torch import load_file

        return load_file(path)
    try:
        return torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    except RuntimeError as error:
        if 'mmap can only be used with files saved with' not in str(error):
            raise
        return torch.load(path, map_location='cpu', weights_only=True)


def is_nsf_hifigan(checkpoint):
    state = generator_state(checkpoint) if isinstance(checkpoint, dict) else None
    return isinstance(state, dict) and 'conv_pre.bias' in state and (
        'conv_pre.weight' in state or 'conv_pre.weight_v' in state
    )


class VocoderInference:
    def __init__(self, path, checkpoint, device):
        metadata = checkpoint.get('vocoder')
        saved = metadata.get('config') if isinstance(metadata, dict) else None
        config = openvpi_config(saved) if isinstance(saved, dict) else None
        hparams, self.data, weights = openvpi_spec(path, generator_state(checkpoint), config=config)
        self.device = torch.device(device)
        self.generator = NSFHiFiGAN(**hparams)
        self.generator.load_state_dict(weights, strict=True)
        self.generator = self.generator.to(self.device).float().eval().requires_grad_(False)
        self.mel = LogMel.from_config(self.data).to(self.device).eval()

    @torch.inference_mode()
    def infer(self, waveform, pitch=0, f0_method='rmvpe', seed=None):
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError('Select non-empty audio with finite samples.')
        extractor = 'parselmouth' if f0_method == 'pm' else f0_method
        if extractor not in {'rmvpe', 'parselmouth'}:
            raise ValueError('NSF-HiFiGAN supports RMVPE or parselmouth pitch extraction.')
        rate, hop = int(self.data['sample_rate']), int(self.data['hop_length'])
        length = len(waveform)
        frames = math.ceil(length / hop)
        padded_length = max(frames, math.ceil(self.data['n_fft'] / hop)) * hop
        padded = np.pad(np.clip(waveform, -1.0, 1.0), (0, padded_length - length))
        source = torch.from_numpy(padded).to(self.device)[None]
        mel = self.mel(source)[..., :frames]
        f0, _ = extract_pitch(extractor, waveform, rate, hop, frames, self.device)
        f0 = torch.from_numpy(f0).to(self.device)[None] * 2 ** (float(pitch) / 12)
        devices = []
        if self.device.type == 'cuda':
            devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()]
        scope = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
        with scope:
            if seed is not None:
                torch.random.default_generator.manual_seed(int(seed))
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(int(seed))
            audio = self.generator(mel, f0)[0, 0, :length]
        if not torch.isfinite(audio).all():
            raise FloatingPointError('Non-finite NSF-HiFiGAN audio output.')
        return audio.float().cpu().numpy()
