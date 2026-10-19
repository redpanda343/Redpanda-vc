from contextlib import nullcontext
from pathlib import Path
import warnings

import numpy as np
import torch
import yaml

from rectified_flow.pitch import extract_pitch
from sifigan.features import continuous_f0, dilation_factors, excitation, sp2mc
from sifigan.model import SiFiGANGenerator


def generator_state(checkpoint):
    if not isinstance(checkpoint, dict):
        return None
    model = checkpoint.get('model')
    if isinstance(model, dict) and isinstance(model.get('generator'), dict):
        return model['generator']
    if isinstance(checkpoint.get('generator'), dict):
        return checkpoint['generator']
    state = checkpoint.get('state_dict', checkpoint)
    if not isinstance(state, dict):
        return None
    for prefix in ('generator.', 'model.generator.'):
        if any(key.startswith(prefix + 'sn.emb.') for key in state):
            return {
                key[len(prefix) :]: value
                for key, value in state.items()
                if key.startswith(prefix)
            }
    return state


def is_sifigan(checkpoint):
    state = generator_state(checkpoint)
    return isinstance(state, dict) and all(
        prefix + '.weight_v' in state or prefix + '.weight' in state
        for prefix in (
            'input_conv',
            'sn.emb',
            'sn.blocks.0.convsC.0',
            'fn.output_conv.1',
        )
    )


def load_config(path):
    directory = Path(path).resolve().parent
    with Path(__file__).with_name('config.yaml').open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    for name in ('sifigan.yaml', 'config.yaml', 'config.yml'):
        candidate = directory / name
        if not candidate.is_file():
            continue
        with candidate.open(encoding='utf-8') as stream:
            saved = yaml.safe_load(stream)
        if not isinstance(saved, dict):
            raise ValueError(f'Invalid SiFiGAN configuration: {candidate}')
        config.update(saved)
        break
    generator = dict(config['generator'])
    target = generator.pop('_target_', 'sifigan.models.SiFiGANGenerator')
    if target != 'sifigan.models.SiFiGANGenerator':
        raise ValueError(
            'Only the standard SiFiGAN generator is supported for offline inference.'
        )
    config['generator'] = generator
    return config


def load_statistics(path, config):
    from joblib import load
    from sklearn.exceptions import InconsistentVersionWarning

    directory = Path(path).resolve().parent
    saved = config.get('stats')
    candidates = []
    if saved:
        candidates.extend(
            (
                directory / saved,
                directory / str(saved).replace('\\', '/').split('/')[-1],
            )
        )
    else:
        candidates.extend(sorted(directory.glob('*.joblib')))
    candidates = list(
        dict.fromkeys(
            candidate.resolve() for candidate in candidates if candidate.is_file()
        )
    )
    if not candidates:
        raise FileNotFoundError(
            f'SiFiGAN statistics not found: {saved}'
            if saved
            else 'Keep the SiFiGAN statistics .joblib file beside its checkpoint.'
        )
    if not saved and len(candidates) > 1:
        raise ValueError(
            'Multiple SiFiGAN statistics files found. Set stats in the adjacent config.yaml.'
        )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', InconsistentVersionWarning)
        scalers = load(candidates[0])
    result = {}
    for feature in config['aux_feats']:
        if feature not in scalers:
            raise ValueError(f'SiFiGAN statistics are missing {feature}.')
        mean = np.asarray(scalers[feature].mean_, dtype=np.float64)
        scale = np.asarray(scalers[feature].scale_, dtype=np.float64)
        if (
            mean.ndim != 1
            or mean.shape != scale.shape
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError(f'Invalid SiFiGAN statistics for {feature}.')
        result[feature] = mean, scale
    return result


class SiFiGANInference:
    def __init__(self, path, checkpoint, device):
        self.config = load_config(path)
        config = self.config
        self.device = torch.device(device)
        rate, hop = int(config['sample_rate']), int(config['hop_size'])
        generator = config['generator']
        if int(np.prod(generator['upsample_scales'])) != hop:
            raise ValueError('SiFiGAN upsampling scales must match its hop size.')
        if len(config['dense_factors']) != len(generator['upsample_scales']):
            raise ValueError('SiFiGAN requires one dense factor per upsampling stage.')
        if (
            config['signal_types'] != ['sine']
            or config['sine_f0_type'] not in {'f0', 'cf0'}
            or config['df_f0_type'] not in {'f0', 'cf0'}
        ):
            raise ValueError('Unsupported SiFiGAN excitation configuration.')
        if not set(config['aux_feats']) <= {'f0', 'cf0', 'uv', 'mcep', 'mcap', 'bap'}:
            raise ValueError('Unsupported SiFiGAN acoustic features.')
        self.statistics = load_statistics(path, config)
        if (
            sum(len(mean) for mean, _ in self.statistics.values())
            != generator['in_channels']
        ):
            raise ValueError(
                'SiFiGAN statistics do not match the generator input channels.'
            )
        state = generator_state(checkpoint)
        parameters = dict(generator)
        parameters['use_weight_norm'] = any(key.endswith('.weight_g') for key in state)
        self.generator = SiFiGANGenerator(**parameters)
        self.generator.load_state_dict(state, strict=True)
        if parameters['use_weight_norm']:
            self.generator.remove_weight_norm()
        self.generator = (
            self.generator.to(self.device).float().eval().requires_grad_(False)
        )
        self.data = {
            'sample_rate': rate,
            'hop_length': hop,
            'n_fft': int(config['fft_size']),
        }

    def analyze(self, waveform, f0_method):
        import pyworld

        config = self.config
        rate, hop = self.data['sample_rate'], self.data['hop_length']
        source = np.ascontiguousarray(waveform, dtype=np.float64)
        source = np.pad(source, (0, max(0, self.data['n_fft'] - len(source))))
        frames = 1 + len(source) // hop
        f0, unvoiced = extract_pitch(f0_method, source, rate, hop, frames, self.device)
        f0 = np.ascontiguousarray(np.where(unvoiced, 0, f0), dtype=np.float64)
        times = np.arange(frames, dtype=np.float64) * hop / rate
        spectrum = pyworld.cheaptrick(
            source, f0, times, rate, fft_size=self.data['n_fft']
        )
        aperiodicity = pyworld.d4c(source, f0, times, rate, fft_size=self.data['n_fft'])
        alpha = float(config['alpha'])
        features = {
            'mcep': sp2mc(spectrum, int(config['mcep_dim']), alpha),
            'bap': pyworld.code_aperiodicity(aperiodicity, rate),
        }
        if 'mcap' in config['aux_feats']:
            features['mcap'] = sp2mc(aperiodicity, int(config['mcap_dim']), alpha)
        return f0, features

    @torch.inference_mode()
    def infer(self, waveform, pitch=0, f0_method='rmvpe', seed=None):
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim != 1 or not len(waveform) or not np.isfinite(waveform).all():
            raise ValueError('Select non-empty audio with finite samples.')
        extractor = 'parselmouth' if f0_method == 'pm' else f0_method
        if extractor not in {'rmvpe', 'parselmouth'}:
            raise ValueError('SiFiGAN supports RMVPE or parselmouth pitch extraction.')
        f0, features = self.analyze(np.clip(waveform, -1, 1), extractor)
        f0 *= 2 ** (float(pitch) / 12)
        cf0 = continuous_f0(f0)
        features.update(
            f0=f0[:, None], cf0=cf0[:, None], uv=(f0 > 0).astype(np.float64)[:, None]
        )
        normalized = []
        for name in self.config['aux_feats']:
            mean, scale = self.statistics[name]
            if features[name].shape[1] != len(mean):
                raise ValueError(
                    f'SiFiGAN {name} dimensions do not match its statistics.'
                )
            normalized.append((features[name] - mean) / scale)
        conditioning = torch.tensor(
            np.concatenate(normalized, axis=1).T,
            dtype=torch.float32,
            device=self.device,
        )[None]
        config = self.config
        dense_f0 = cf0 if config['df_f0_type'] == 'cf0' else f0
        dense = dilation_factors(
            dense_f0,
            self.data['sample_rate'],
            config['dense_factors'],
            config['generator']['upsample_scales'],
            self.device,
        )
        signal_f0 = cf0 if config['sine_f0_type'] == 'cf0' else f0
        signal_f0 = torch.tensor(signal_f0, dtype=torch.float32, device=self.device)[
            None, None
        ]
        devices = []
        if self.device.type == 'cuda':
            devices = [
                (
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                )
            ]
        scope = (
            torch.random.fork_rng(devices=devices)
            if seed is not None
            else nullcontext()
        )
        with scope:
            if seed is not None:
                torch.random.default_generator.manual_seed(int(seed))
                if self.device.type == 'cuda':
                    with torch.cuda.device(self.device):
                        torch.cuda.manual_seed(int(seed))
            signal = excitation(
                signal_f0,
                self.data['sample_rate'],
                self.data['hop_length'],
                float(config['sine_amp']),
                float(config['noise_amp']),
            )
            output = self.generator(signal, conditioning, dense)[0][
                0, 0, : len(waveform)
            ]
        if len(output) != len(waveform) or not torch.isfinite(output).all():
            raise FloatingPointError('Invalid SiFiGAN audio output.')
        return output.float().cpu().numpy()
