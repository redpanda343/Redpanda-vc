import math
import os
import threading
import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from rvc.infer.infer import VoiceConverter, deterministic_torch_scope
from rvc.lib.utils import extract_embedding_features


SUPPORTED_VOCODERS = {"HiFi-GAN", "RefineGAN"}
SUPPORTED_EMBEDDERS = {"contentvec", "spin-v2"}


class RMVPEViterbi:
    def __init__(
        self,
        max_jump_bins=12,
        transition_scale_bins=6.0,
        octave_jump_penalty=4.0,
        bridge_frames=2,
    ):
        self.max_jump_bins = int(max_jump_bins)
        self.transition_scale_bins = float(transition_scale_bins)
        self.octave_jump_penalty = float(octave_jump_penalty)
        self.bridge_frames = int(bridge_frames)
        self.state_count = 0
        self.predecessors = None
        self.log_transition = None
        self.state_indices = None
        self.previous_path = None
        self.previous_tracking_voiced = None

    def reset(self):
        self.previous_path = None
        self.previous_tracking_voiced = None

    def _prepare(self, state_count):
        if state_count == self.state_count:
            return
        offsets = np.unique(
            np.concatenate(
                (
                    np.arange(
                        -self.max_jump_bins,
                        self.max_jump_bins + 1,
                        dtype=np.int32,
                    ),
                    np.array([-60, 60], dtype=np.int32),
                )
            )
        )
        distances = np.abs(offsets).astype(np.float32)
        costs = distances / self.transition_scale_bins
        costs += self.octave_jump_penalty * (distances == 60.0)
        weights = np.exp(-costs).astype(np.float32)
        source = np.arange(state_count, dtype=np.int32)[:, None]
        targets = source + offsets[None, :]
        source_valid = (targets >= 0) & (targets < state_count)
        row_sums = np.sum(weights[None, :] * source_valid, axis=1)
        destination = np.arange(state_count, dtype=np.int32)[:, None]
        predecessors = destination + offsets[None, :]
        valid = (predecessors >= 0) & (predecessors < state_count)
        safe_predecessors = np.clip(predecessors, 0, state_count - 1)
        log_transition = np.log(weights[None, :]) - np.log(
            row_sums[safe_predecessors]
        )
        log_transition[~valid] = -np.inf
        self.state_count = state_count
        self.predecessors = safe_predecessors
        self.log_transition = log_transition.astype(np.float32)
        self.state_indices = np.arange(state_count)

    def _decode_segment(self, salience, initial_center=None):
        frame_count, state_count = salience.shape
        self._prepare(state_count)
        emissions = np.log(np.maximum(salience, np.finfo(np.float32).tiny))
        score = emissions[0].copy()
        if initial_center is not None:
            distances = np.abs(self.state_indices - int(initial_center))
            score -= distances / self.transition_scale_bins
            score -= self.octave_jump_penalty * (distances == 60)
        back = np.empty((frame_count, state_count), dtype=np.int16)
        for frame in range(1, frame_count):
            candidates = score[self.predecessors] + self.log_transition
            choices = np.argmax(candidates, axis=1)
            previous = self.predecessors[self.state_indices, choices]
            score = candidates[self.state_indices, choices] + emissions[frame]
            back[frame] = previous
        path = np.empty(frame_count, dtype=np.int64)
        path[-1] = int(np.argmax(score))
        for frame in range(frame_count - 1, 0, -1):
            path[frame - 1] = back[frame, path[frame]]
        return path

    def _tracking_mask(self, voiced, previous_voiced):
        tracking = voiced.copy()
        unvoiced = ~voiced
        changes = np.flatnonzero(
            np.diff(np.concatenate(([False], unvoiced, [False])).astype(np.int8))
        ).reshape(-1, 2)
        for start, end in changes:
            left_voiced = voiced[start - 1] if start else previous_voiced
            right_voiced = voiced[end] if end < voiced.size else left_voiced
            if end - start <= self.bridge_frames and left_voiced and right_voiced:
                tracking[start:end] = True
        return tracking

    def decode(self, salience, threshold, advance_frames=0):
        salience = np.asarray(salience, dtype=np.float32)
        if salience.ndim != 2 or salience.shape[0] == 0 or salience.shape[1] == 0:
            raise ValueError("RMVPE salience must contain frames and pitch bins.")
        centers = np.argmax(salience, axis=1).astype(np.int64)
        voiced = np.max(salience, axis=1) > threshold
        initial_center = None
        previous_voiced = False
        if self.previous_path is not None:
            previous_index = min(
                max(int(advance_frames), 0), self.previous_path.shape[0] - 1
            )
            previous_voiced = bool(self.previous_tracking_voiced[previous_index])
            if previous_voiced:
                initial_center = int(self.previous_path[previous_index])
        tracking_voiced = self._tracking_mask(voiced, previous_voiced)
        changes = np.flatnonzero(
            np.diff(
                np.concatenate(([False], tracking_voiced, [False])).astype(np.int8)
            )
        ).reshape(-1, 2)
        for start, end in changes:
            segment_initial = initial_center if start == 0 else None
            centers[start:end] = self._decode_segment(
                salience[start:end], segment_initial
            )
        self.previous_path = centers.copy()
        self.previous_tracking_voiced = tracking_voiced.copy()
        return centers


class RealTimeRVC:
    def __init__(
        self,
        model_path,
        index_path="",
        index_rate=0.0,
        pitch=0,
        speaker_id=0,
        embedder_model="contentvec",
        seed=0,
        rectified_vocoder_path="",
        rectified_steps=0,
        rectified_cuda_graph=True,
    ):
        self.converter = VoiceConverter()
        self.converter.get_vc(model_path, speaker_id)
        if self.converter.cpt is None:
            raise FileNotFoundError(f"Voice model not found: {model_path}")
        self.vocoder = self.converter.vocoder
        self.is_rectified = isinstance(self.converter.cpt.get("config"), dict)
        if not self.is_rectified and self.vocoder not in SUPPORTED_VOCODERS:
            supported = ", ".join(sorted(SUPPORTED_VOCODERS))
            raise ValueError(
                f"Real-time mode supports {supported}; this model uses {self.vocoder}."
            )
        if not self.converter.use_f0:
            raise ValueError("Real-time mode currently requires a pitch-guided model.")
        if speaker_id < 0 or speaker_id >= self.converter.n_spk:
            raise ValueError(
                f"Speaker ID must be between 0 and {self.converter.n_spk - 1}."
            )
        self.device = self.converter.config.device
        self.model = self.converter.net_g.to(
            device=self.device, dtype=torch.float32
        )
        self.pipeline = self.converter.vc
        self.sample_rate = self.converter.tgt_sr
        self.version = self.converter.version
        self.speaker_id = speaker_id
        self.pitch = pitch
        self.index_rate = float(index_rate)
        self.index = None
        self.big_npy = None
        self.lock = threading.RLock()
        self.seed = int(seed) % (2**63 - 1)
        self.last_f0_method = None
        self.pitch_sample_count = 0
        self.prepared_f0_methods = set()
        self.rmvpe_viterbi = RMVPEViterbi()
        self.cache_pitch = torch.zeros(4096, device=self.device, dtype=torch.long)
        self.cache_pitchf = torch.zeros(
            4096, device=self.device, dtype=torch.float32
        )
        expected_embedder = self.converter.cpt.get("embedder_model")
        if embedder_model not in SUPPORTED_EMBEDDERS:
            raise ValueError(f"Unsupported embedder model: {embedder_model}")
        if expected_embedder and expected_embedder not in SUPPORTED_EMBEDDERS:
            raise ValueError(
                f"This voice model uses unsupported embedder {expected_embedder}."
            )
        if expected_embedder and embedder_model != expected_embedder:
            raise ValueError(
                f"This voice model was trained with {expected_embedder}, but "
                f"{embedder_model} is selected."
            )
        self.embedder_name = embedder_model
        self.converter.load_hubert(embedder_model)
        self.embedder = self.converter.hubert_model.to(
            device=self.device, dtype=torch.float32
        )
        self.expected_feature_dim = (
            self.pipeline.content_channels
            if self.is_rectified
            else int(self.model.enc_p.emb_phone.in_features)
        )
        if self.is_rectified:
            from torchaudio.transforms import Resample
            from rvc.rectified.resources import default_vocoder
            from rvc.rectified.vocoder import load_vocoder
            from rvc.rectified.realtime import RealtimeFlowSampler

            if int(rectified_steps) != rectified_steps or not 0 <= rectified_steps <= 1000:
                raise ValueError("Flow steps must be an integer between 0 and 1000.")
            self.rectified_steps = int(rectified_steps) or None
            self.flow_sampler = RealtimeFlowSampler(self.model, enabled=rectified_cuda_graph)
            self.pipeline.set_vocoder(rectified_vocoder_path)
            path = self.pipeline.vocoder_path or default_vocoder(self.pipeline.checkpoint_vocoder)
            vocoder, _ = load_vocoder(path, self.pipeline.data)
            self.pipeline.vocoder_model = vocoder.to(self.device).float()
            self.flow_resampler = Resample(16000, self.sample_rate).to(self.device)
            generator = vocoder.generator
            radius = generator.conv_pre.kernel_size[0] // 2
            scale = 1
            for stage, up in enumerate(generator.ups):
                scale *= up.stride[0]
                radius += (up.kernel_size[0] - 1) / scale
                blocks = generator.resblocks[
                    stage * generator.num_kernels:(stage + 1) * generator.num_kernels
                ]
                radius += max(
                    sum((layer.kernel_size[0] - 1) * layer.dilation[0] / 2
                        for layer in block.modules() if isinstance(layer, torch.nn.Conv1d))
                    for block in blocks
                ) / scale
            radius += (generator.conv_post.kernel_size[0] // 2) / scale
            self.vocoder_context_frames = math.ceil(radius) + 2
        if index_path:
            self._load_index(index_path)
        if self.index_rate > 0 and self.index is None:
            raise ValueError("Select a valid feature index or set Index Rate to 0.")

    @staticmethod
    def pitch_lookahead_frames(method):
        return 20 if method == "swift" else 0

    @property
    def speaker_count(self):
        return self.converter.n_spk

    def _load_index(self, index_path):
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Feature index not found: {index_path}")
        index = faiss.read_index(index_path)
        if int(index.d) != self.expected_feature_dim:
            raise ValueError(
                f"The index has {index.d} channels, but this model expects "
                f"{self.expected_feature_dim}."
            )
        if index.ntotal < 1:
            raise ValueError("The selected feature index is empty.")
        self.index = index
        self.big_npy = index.reconstruct_n(0, index.ntotal)

    def change_pitch(self, pitch):
        with self.lock:
            self.pitch = int(pitch)

    def change_index_rate(self, index_rate):
        rate = float(index_rate)
        if rate > 0 and self.index is None:
            raise ValueError("No feature index is loaded.")
        with self.lock:
            self.index_rate = rate

    def reset_caches(self):
        self.cache_pitch.zero_()
        self.cache_pitchf.zero_()
        self.rmvpe_viterbi.reset()
        self.last_f0_method = None
        self.pitch_sample_count = 0

    def _extract_features(self, input_wav):
        source = input_wav.float().view(1, -1)
        if getattr(self.embedder, "audio_requires_normalization", False):
            source = F.layer_norm(source, source.shape)
        features = extract_embedding_features(self.embedder, source, self.version)
        if features.shape[-1] != self.expected_feature_dim:
            raise RuntimeError(
                f"{self.embedder_name} outputs {features.shape[-1]} channels, but "
                f"this model expects {self.expected_feature_dim}."
            )
        if self.is_rectified:
            return features
        return torch.cat((features, features[:, -1:, :]), dim=1)

    def _apply_index(self, features, skip_head):
        if self.index is None or self.index_rate <= 0:
            return features
        start = skip_head // 2
        query = features[0, start:].detach().cpu().numpy().astype("float32")
        score, indices = self.index.search(query, k=min(8, self.index.ntotal))
        valid = indices >= 0
        if not valid.any(axis=1).all():
            raise RuntimeError("The selected index returned no valid neighbors.")
        indices = np.where(valid, indices, 0)
        score = np.maximum(score, 1e-6)
        weight = np.where(valid, np.square(1.0 / score), 0.0)
        weight /= weight.sum(axis=1, keepdims=True)
        retrieved = np.sum(
            self.big_npy[indices] * np.expand_dims(weight, axis=2), axis=1
        )
        replacement = torch.from_numpy(retrieved).unsqueeze(0).to(self.device)
        features[0, start:] = (
            replacement * self.index_rate
            + features[0, start:] * (1.0 - self.index_rate)
        )
        return features

    def _prepare_pitch_predictor(self, method):
        predictor = getattr(self.pipeline, f"model_{method}", None)
        if predictor is None and method == "swift":
            from rvc.lib.predictors.f0 import Swift

            predictor = Swift(
                device=self.device,
                sample_rate=16000,
                hop_size=160,
                threads=1,
                spin=False,
            )
            self.pipeline.model_swift = predictor
        if predictor is None:
            return
        current = predictor
        for _ in range(4):
            if isinstance(current, torch.nn.Module):
                current.float()
            mel_extractor = getattr(current, "mel_extractor", None)
            if isinstance(mel_extractor, torch.nn.Module):
                mel_extractor.float()
            current = getattr(current, "model", None)
            if current is None:
                break
        self.prepared_f0_methods.add(method)

    def _update_pitch(self, input_wav, block_frame_16k, method):
        if method != self.last_f0_method:
            self.rmvpe_viterbi.reset()
            self.last_f0_method = method
        if method not in self.prepared_f0_methods:
            self._prepare_pitch_predictor(method)
        self.pitch_sample_count += int(block_frame_16k)
        extractor_frame = block_frame_16k + 800
        if method == "swift":
            extractor_frame = min(block_frame_16k + 19200, input_wav.shape[0])
            start = self.pitch_sample_count - extractor_frame
            aligned_start = start - start % 1280
            if self.pitch_sample_count - aligned_start > input_wav.shape[0]:
                aligned_start += 1280
            extractor_frame = self.pitch_sample_count - aligned_start
        if method == "rmvpe":
            extractor_frame = 5120 * ((extractor_frame - 1) // 5120 + 1) - 160
        source = input_wav[-extractor_frame:].detach().cpu().numpy()
        shift = max(1, block_frame_16k // 160)
        decoder = None
        if method == "rmvpe":
            decoder = lambda salience, threshold: self.rmvpe_viterbi.decode(
                salience, threshold, shift
            )
        pitch, pitchf = self.pipeline.get_f0(
            source,
            source.shape[0] // 160,
            f0_method=method,
            pitch=self.pitch,
            f0_decoder=decoder,
        )
        if method not in self.prepared_f0_methods:
            self._prepare_pitch_predictor(method)
        pitch = torch.as_tensor(pitch, device=self.device, dtype=torch.long).flatten()
        pitchf = torch.as_tensor(
            pitchf, device=self.device, dtype=torch.float32
        ).flatten()
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
        if method == "swift":
            usable_pitch = pitch
            usable_pitchf = pitchf
        else:
            usable_pitch = pitch[3:-1] if pitch.numel() > 4 else pitch
            usable_pitchf = pitchf[3:-1] if pitchf.numel() > 4 else pitchf
        count = min(usable_pitch.numel(), self.cache_pitch.numel())
        if count:
            self.cache_pitch[-count:] = usable_pitch[-count:]
            self.cache_pitchf[-count:] = usable_pitchf[-count:]

    def _infer_rectified(self, features, input_wav, speaker, skip_head, return_length):
        from rvc.rectified.aperiodicity import aperiodicity
        from rvc.rectified.data import (
            f0_to_mel_rate, smooth_curve, to_mel_rate, upsample_content, variance_curves,
        )
        from rvc.rectified.energy import frame_energy
        from rvc.rectified.pitch import parselmouth_f0, uses_parselmouth

        rate = self.sample_rate
        hop = int(self.pipeline.data["hop_length"])
        waveform = self.flow_resampler(input_wav.float()).view(1, -1)
        length = waveform.shape[-1]
        start = round(skip_head * rate / 100)
        count = round(return_length * rate / 100)
        if start < 0 or count < 1 or start + count > length:
            raise ValueError("The requested flow output exceeds the input context.")
        frames = math.ceil(length / hop)
        feature_frames = max(1, input_wav.shape[0] // 160)
        if feature_frames > self.cache_pitchf.numel():
            raise ValueError("Flow input context exceeds the pitch cache capacity.")
        pitchf = self.cache_pitchf[None, -feature_frames:]
        source_f0 = pitchf / (2 ** (self.pitch / 12))
        content = features.float()
        if not self.model.native_content_rate:
            content = upsample_content(content, self.pipeline.data["content_interpolation"])
            content = to_mel_rate(content, frames, rate, hop)
        if uses_parselmouth(self.pipeline.data):
            f0 = parselmouth_f0(waveform[0].cpu().numpy(), rate, hop, frames)
            f0 = torch.from_numpy(f0).to(waveform.device)[None] * 2 ** (self.pitch / 12)
        else:
            f0 = f0_to_mel_rate(pitchf, frames, rate, hop, self.model.use_continuous_f0)
        energy = smooth_curve(frame_energy(waveform, rate, feature_frames))
        energy = to_mel_rate(energy.unsqueeze(-1), frames, rate, hop)[..., 0]
        breathiness = smooth_curve(aperiodicity(waveform, rate, source_f0, feature_frames))
        breathiness = to_mel_rate(breathiness.unsqueeze(-1), frames, rate, hop)[..., 0]
        variances = {}
        if any(getattr(self.model.encoder, name, None) is not None for name in ('voicing', 'tension')):
            variances = dict(zip(
                ("voicing", "tension"), variance_curves(waveform, source_f0, frames, rate, hop)
            ))
        mask = torch.ones(1, 1, frames, device=self.device)
        mel = self.flow_sampler(
            content, f0, energy, speaker, mask, steps=self.rectified_steps,
            breathiness=breathiness, **variances,
        )
        first_frame = max(0, start // hop - self.vocoder_context_frames)
        audio = self.pipeline.vocoder_model(mel[..., first_frame:], f0[..., first_frame:])[0, 0]
        start -= first_frame * hop
        audio = audio[start:start + count]
        if audio.numel() != count:
            raise RuntimeError("The flow vocoder returned an incomplete audio block.")
        if not torch.isfinite(audio).all():
            raise FloatingPointError("Non-finite Rectified Flow audio output.")
        return audio

    @torch.inference_mode()
    def infer(
        self,
        input_wav,
        block_frame_16k,
        skip_head,
        return_length,
        f0_method,
    ):
        lookahead = self.pitch_lookahead_frames(f0_method)
        if skip_head < lookahead:
            raise ValueError("SwiftF0 realtime requires at least 0.2 seconds of extra context.")
        skip_head = int(skip_head) - lookahead
        started = time.perf_counter()
        with deterministic_torch_scope():
            with self.lock:
                torch.manual_seed(self.seed)
                features = self._extract_features(input_wav)
                features = self._apply_index(features, skip_head)
                self._update_pitch(input_wav, block_frame_16k, f0_method)
                speaker = torch.tensor(
                    [self.speaker_id], device=self.device, dtype=torch.long
                )
                if self.is_rectified:
                    audio = self._infer_rectified(
                        features, input_wav, speaker, skip_head, int(return_length)
                    )
                    if torch.device(self.device).type == "cuda":
                        torch.cuda.synchronize(self.device)
                    return audio.float(), time.perf_counter() - started
                p_len = min(input_wav.shape[0] // 160, features.shape[1] * 2)
                features = F.interpolate(
                    features.permute(0, 2, 1), scale_factor=2
                ).permute(0, 2, 1)
                features = features[:, :p_len]
                lengths = torch.tensor(
                    [p_len], device=self.device, dtype=torch.long
                )
                coarse = self.cache_pitch[None, -p_len:]
                continuous = self.cache_pitchf[None, -p_len:]
                audio = self.model.infer(
                    features.float(),
                    lengths,
                    coarse,
                    continuous,
                    speaker,
                    int(skip_head),
                    int(return_length),
                )[0]
                if torch.device(self.device).type == "cuda":
                    torch.cuda.synchronize(self.device)
        return audio.squeeze().float(), time.perf_counter() - started
