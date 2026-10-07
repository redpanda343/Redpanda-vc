import os
import sys
import threading
import torch
import torch.nn.functional as F
import faiss
import numpy as np
from scipy import signal
from torch import Tensor

now_dir = os.getcwd()
sys.path.append(now_dir)

from rvc.lib.predictors.f0 import RMVPE, Swift
from rvc.infer.pm import extract_pm
from rvc.lib.utils import extract_embedding_features

import logging

logging.getLogger("faiss").setLevel(logging.WARNING)

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48  # Hz
SAMPLE_RATE = 16000  # Hz
bh, ah = signal.butter(
    N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE
)
_INFERENCE_RNG_LOCK = threading.Lock()


class InferenceRNG:
    """Provides a reproducible, distinct seed for each synthesized segment."""

    _MAX_SEED = 2**63 - 1

    def __init__(self, seed: int):
        self.seed = int(seed)
        self.segment_index = 0

    def seed_next_segment(self):
        segment_seed = (self.seed + self.segment_index) % self._MAX_SEED
        torch.manual_seed(segment_seed)
        self.segment_index += 1


class Pipeline:
    """
    The main pipeline class for performing voice conversion, including preprocessing, F0 estimation,
    voice conversion using a model, and post-processing.
    """

    high_pass = True

    def __init__(self, tgt_sr, config):
        """
        Initializes the Pipeline class with target sampling rate and configuration parameters.

        Args:
            tgt_sr: The target sampling rate for the output audio.
            config: A configuration object containing various parameters for the pipeline.
        """
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.tgt_sr = tgt_sr
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = config.device

    def get_f0(
        self,
        x,
        p_len,
        f0_method: str = "rmvpe",
        pitch: int = 0,
        f0_decoder=None,
    ):
        """
        Estimates the fundamental frequency (F0) of a given audio signal using various methods.

        Args:
            x: The input audio signal as a NumPy array.
            p_len: Desired length of the F0 output.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation (e.g., "pm", "rmvpe").
        """
        if f0_method == "pm":
            f0 = extract_pm(
                audio=x,
                sample_rate=self.sample_rate,
                p_len=p_len,
                time_step=self.time_step / 1000,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                self.model_rmvpe = RMVPE(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.model_rmvpe.get_f0(
                x, filter_radius=0.03, decoder=f0_decoder
            )
        elif f0_method == "swift":
            if not hasattr(self, "model_swift"):
                self.model_swift = Swift(
                    device=self.device,
                    sample_rate=self.sample_rate,
                    hop_size=self.window,
                )
            f0 = self.model_swift.get_f0(x, p_len)
        else:
            raise ValueError(f"Unsupported pitch extraction method: {f0_method}")

        # Preserve zero F0 frames so the existing pitch inputs carry an explicit
        # unvoiced state: coarse bin 1 for the encoder and zero F0 for the vocoder.
        f0 = np.asarray(f0).copy()
        f0 *= pow(2, pitch / 12)
        # quantizing f0 to 255 buckets to make coarse f0
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak

    def voice_conversion(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
        inference_rng=None,
        segment_start=0,
    ):
        """
        Performs voice conversion on a given audio segment.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio0: The input audio segment.
            pitch: Quantized F0 contour for pitch guidance.
            pitchf: Original F0 contour for pitch guidance.
            index: FAISS index for speaker embedding retrieval.
            big_npy: Speaker embeddings stored in a NumPy array.
            index_rate: Blending rate for speaker embedding retrieval.
            version: Model version (Keep to support old models).
            protect: Protection level for preserving the original pitch.
            inference_rng: Seed sequence shared by all segments in the input audio.
        """
        with torch.inference_mode():
            pitch_guidance = pitch != None and pitchf != None
            # prepare source audio
            feats = torch.from_numpy(audio0).float()
            feats = feats.mean(-1) if feats.dim() == 2 else feats
            assert feats.dim() == 1, feats.dim()
            if getattr(model, "audio_requires_normalization", False):
                feats = F.layer_norm(feats, feats.shape)
            feats = feats.view(1, -1).to(self.device)
            # extract features
            feats = extract_embedding_features(model, feats, version)
            expected_dim = int(net_g.enc_p.emb_phone.in_features)
            if feats.shape[-1] != expected_dim:
                raise RuntimeError(
                    f"The selected embedder outputs {feats.shape[-1]} channels, but "
                    f"this voice model expects {expected_dim}. Select the embedder used "
                    "during training."
                )
            # make a copy for pitch guidance and protection
            feats0 = feats.clone() if pitch_guidance else None
            if (
                index
            ):  # set by parent function, only true if index is available, loaded, and index rate > 0
                feats = self._retrieve_speaker_embeddings(
                    feats, index, big_npy, index_rate
                )
            # feature upsampling
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
            # adjust the length if the audio is short
            p_len = min(audio0.shape[0] // self.window, feats.shape[1])
            if pitch_guidance:
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                    0, 2, 1
                )
                pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len].float()
                # Pitch protection blending
                if protect < 0.5:
                    pitchff = pitchf.clone()
                    pitchff[pitchf > 0] = 1
                    pitchff[pitchf < 1] = protect
                    feats = feats * pitchff.unsqueeze(-1) + feats0 * (
                        1 - pitchff.unsqueeze(-1)
                    )
                    feats = feats.to(feats0.dtype)
            else:
                pitch, pitchf = None, None
            p_len = torch.tensor([p_len], device=self.device).long()
            with _INFERENCE_RNG_LOCK:
                if inference_rng is not None:
                    inference_rng.seed_next_segment()
                audio1 = net_g.infer(feats.float(), p_len, pitch, pitchf, sid)[0][
                    0, 0
                ]
            audio1 = audio1.data.cpu().float().numpy()
            # clean up
            del feats, feats0, p_len
        return audio1

    def _retrieve_speaker_embeddings(self, feats, index, big_npy, index_rate):
        if int(index.d) != feats.shape[-1]:
            raise RuntimeError(
                f"The selected index has {index.d} channels, but the embedder outputs "
                f"{feats.shape[-1]}. Select the index created for this voice model."
            )
        npy = feats[0].cpu().numpy()
        neighbor_count = min(8, int(index.ntotal))
        if neighbor_count < 1:
            raise RuntimeError("The selected feature index is empty.")
        score, ix = index.search(npy, k=neighbor_count)
        valid = ix >= 0
        if not valid.any(axis=1).all():
            raise RuntimeError("The selected feature index returned no valid neighbors.")
        ix = np.where(valid, ix, 0)
        score = np.maximum(score, 1e-6)
        weight = np.square(1 / score)
        weight = np.where(valid, weight, 0.0)
        weight /= weight.sum(axis=1, keepdims=True)
        npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        feats = (
            torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
            + (1 - index_rate) * feats
        )
        return feats

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        version,
        protect,
        inference_rng=None,
    ):
        """
        The main pipeline function for performing voice conversion.

        Args:
            model: The feature extractor model.
            net_g: The generative model for synthesizing speech.
            sid: Speaker ID for the target voice.
            audio: The input audio signal.
            input_audio_path: Path to the input audio file.
            pitch: Key to adjust the pitch of the F0 contour.
            f0_method: Method to use for F0 estimation.
            file_index: Path to the FAISS index file for speaker embedding retrieval.
            index_rate: Blending rate for speaker embedding retrieval.
            pitch_guidance: Whether to use pitch guidance during voice conversion.
            tgt_sr: Target sampling rate for the output audio.
            resample_sr: Resampling rate for the output audio.
            version: Model version.
            protect: Protection level for preserving the original pitch.
            inference_rng: Seed sequence shared by all segments in the input audio.
            hop_length: Hop length for F0 estimation methods.
        """
        if file_index != "" and os.path.exists(file_index) and index_rate > 0:
            index = faiss.read_index(file_index)
            if index.ntotal < 1:
                raise RuntimeError("The selected feature index is empty.")
            big_npy = index.reconstruct_n(0, index.ntotal)
        else:
            index = big_npy = None
        if self.high_pass:
            audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if pitch_guidance:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                f0_method,
                pitch,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        for t in opt_ts:
            t = t // self.window * self.window
            if pitch_guidance:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        inference_rng,
                        segment_start=s,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.voice_conversion(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                        inference_rng,
                        segment_start=s,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if pitch_guidance:
            audio_opt.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    inference_rng,
                    segment_start=t or 0,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.voice_conversion(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                    inference_rng,
                    segment_start=t or 0,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if pitch_guidance:
            del pitch, pitchf
        del sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
