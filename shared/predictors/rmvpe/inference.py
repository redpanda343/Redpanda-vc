import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from .constants import *
from .model import E2E0
from .spec import MelSpectrogram
from .utils import to_local_average_f0, to_viterbi_f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = np.log2(f0 + uv)
    f0[uv] = -np.inf
    if uv.any() and not uv.all():
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return 2 ** f0, uv


def resample_align_curve(points, original_timestep, target_timestep, align_length):
    return np.interp(
        np.arange(align_length) * target_timestep,
        original_timestep * np.arange(len(points)),
        points
    ).astype(points.dtype)


class RMVPE:
    def __init__(self, model_path, hop_length=160, device=None):
        self.resample_kernel = {}
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = E2E0(4, 1, (2, 2)).eval().to(self.device)
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        if 'model' not in ckpt:
            raise ValueError(f'{model_path} is not the RMVPE 230917 checkpoint. Delete it and restart to download it.')
        self.model.load_state_dict(ckpt['model'], strict=False)
        self.hop_length = hop_length
        self.seg_length = 32 * hop_length
        self.mel_extractor = MelSpectrogram(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        ).to(self.device)

    @torch.no_grad()
    def mel2hidden(self, mel):
        n_frames = mel.shape[-1]
        mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode='reflect')
        hidden = self.model(mel)
        return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03, use_viterbi=False):
        if use_viterbi:
            f0 = to_viterbi_f0(hidden, thred=thred)
        else:
            f0 = to_local_average_f0(hidden, thred=thred)
        return f0

    def _resample(self, audio, sample_rate):
        audio = torch.from_numpy(np.asarray(audio)).float().unsqueeze(0).to(self.device)
        if sample_rate == 16000:
            return audio
        key_str = str(sample_rate)
        if key_str not in self.resample_kernel:
            self.resample_kernel[key_str] = Resample(sample_rate, 16000, lowpass_filter_width=128)
        self.resample_kernel[key_str] = self.resample_kernel[key_str].to(self.device)
        return self.resample_kernel[key_str](audio)

    def _padded_length(self, length):
        return self.seg_length * ((length + self.hop_length - 1) // self.seg_length + 1) - self.hop_length

    def padded_length(self, num_samples, sample_rate):
        return self._padded_length(-(-num_samples * 16000 // sample_rate))

    @torch.no_grad()
    def infer_hidden(self, audio, sample_rate=16000, network=None):
        audio_res = self._resample(audio, sample_rate)
        B, T = audio_res.shape
        n_frames = T // self.hop_length + 1
        audio_res = F.pad(audio_res, (0, self._padded_length(T) - T))
        mel = self.mel_extractor(audio_res, center=True)
        hidden = (self.model if network is None else network)(mel)
        return hidden[:, :n_frames]

    def infer_from_audio(self, audio, sample_rate=16000, thred=0.03, use_viterbi=False):
        hidden = self.infer_hidden(audio, sample_rate)
        return self.decode(hidden, thred=thred, use_viterbi=use_viterbi)

    def get_pitch(
            self, waveform, samplerate, length,
            *, hop_size, f0_min=65, f0_max=1100,
            speed=1, interp_uv=False
    ):
        f0 = self.infer_from_audio(waveform, sample_rate=samplerate)
        return self._align_pitch(f0, samplerate, length, hop_size, speed, interp_uv)

    @torch.no_grad()
    def get_pitch_batch(self, waveforms, samplerate, lengths, *, hop_size, speed=1, interp_uv=False):
        resampled = [self._resample(waveform, samplerate)[0] for waveform in waveforms]
        groups = {}
        for index, audio in enumerate(resampled):
            groups.setdefault(self._padded_length(audio.shape[0]), []).append(index)
        results = [None] * len(resampled)
        for padded, indices in groups.items():
            batch = torch.stack([F.pad(resampled[index], (0, padded - resampled[index].shape[0]))
                                 for index in indices])
            hidden = self.model(self.mel_extractor(batch, center=True))
            f0s = self.decode(hidden).reshape(len(indices), -1)
            for f0, index in zip(f0s, indices):
                f0 = f0[:resampled[index].shape[0] // self.hop_length + 1]
                results[index] = self._align_pitch(f0, samplerate, lengths[index], hop_size, speed, interp_uv)
        return results

    def _align_pitch(self, f0, samplerate, length, hop_size, speed, interp_uv):
        uv = f0 == 0
        f0, uv = interp_f0(f0, uv)

        hop_size = int(np.round(hop_size * speed))
        time_step = hop_size / samplerate
        f0_res = resample_align_curve(f0, 0.01, time_step, length)
        uv_res = resample_align_curve(uv.astype(np.float32), 0.01, time_step, length) > 0.5
        if not interp_uv:
            f0_res[uv_res] = 0
        return f0_res, uv_res
