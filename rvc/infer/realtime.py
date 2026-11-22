import os
import threading
import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from numba import njit

from rvc.infer.feature_index import FeatureIndex
from rvc.infer.infer import VoiceConverter
from rvc.infer.realtime_graph import RealtimeGraph
from shared.utils import V1_FEATURE_LAYER, extract_embedding_features


SUPPORTED_VOCODERS = {"HiFi-GAN", "RefineGAN"}
SUPPORTED_EMBEDDERS = {"contentvec", "spin-v2"}
PITCH_PREDICTORS = ("rmvpe", "swift")
RMVPE_THRESHOLD = 0.03


def inference_dtype(device, use_fp32=False):
    device = torch.device(device)
    if use_fp32 or device.type != "cuda":
        return torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    name = torch.cuda.get_device_name(device).upper().replace(" ", "")
    unsupported = any(token in name for token in ("GTX16", "RADEON", "ZLUDA"))
    return torch.float16 if major >= 7 and not unsupported else torch.float32


@njit(cache=True)
def viterbi_path(emissions, score, predecessors, log_transition):
    frames, states = emissions.shape
    width = predecessors.shape[1]
    back = np.empty((frames, states), dtype=np.int16)
    fresh = np.empty(states, dtype=np.float32)
    for frame in range(1, frames):
        for state in range(states):
            best = score[predecessors[state, 0]] + log_transition[state, 0]
            choice = 0
            for offset in range(1, width):
                value = score[predecessors[state, offset]] + log_transition[state, offset]
                if value > best:
                    best = value
                    choice = offset
            back[frame, state] = predecessors[state, choice]
            fresh[state] = best + emissions[frame, state]
        score, fresh = fresh, score
    path = np.empty(frames, dtype=np.int64)
    path[-1] = np.argmax(score)
    for frame in range(frames - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]
    return path


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
        self._prepare(salience.shape[1])
        emissions = np.log(np.maximum(salience, np.finfo(np.float32).tiny))
        score = emissions[0].copy()
        if initial_center is not None:
            distances = np.abs(self.state_indices - int(initial_center))
            score -= distances / self.transition_scale_bins
            score -= self.octave_jump_penalty * (distances == 60)
        return viterbi_path(emissions, score, self.predecessors, self.log_transition)

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


class MatmulGroupedConv1d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.groups = conv.groups
        self.kernel_size = conv.kernel_size[0]
        self.padding = conv.padding[0]
        weight = conv.weight.detach()
        out_channels, group_channels, _ = weight.shape
        self.register_buffer(
            "weight",
            weight.reshape(self.groups, out_channels // self.groups, group_channels * self.kernel_size)
            .transpose(1, 2)
            .contiguous(),
        )
        self.register_buffer("bias", conv.bias.detach().clone())

    def forward(self, x):
        batch, channels, _ = x.shape
        group_channels = channels // self.groups
        windows = F.pad(x, (self.padding, self.padding)).unfold(2, self.kernel_size, 1)
        windows = windows.reshape(batch, self.groups, group_channels, -1, self.kernel_size).permute(0, 1, 3, 2, 4)
        windows = windows.reshape(batch * self.groups, -1, group_channels * self.kernel_size)
        output = torch.bmm(windows, self.weight.repeat(batch, 1, 1))
        output = output.reshape(batch, self.groups, -1, self.weight.shape[2]).permute(0, 1, 3, 2)
        return output.reshape(batch, -1, output.shape[-1]) + self.bias[:, None]


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
        cuda_graph=True,
        use_fp32=False,
    ):
        self.converter = VoiceConverter()
        self.converter.get_vc(model_path, speaker_id)
        if self.converter.cpt is None:
            raise FileNotFoundError(f"Voice model not found: {model_path}")
        self.vocoder = self.converter.vocoder
        if self.vocoder not in SUPPORTED_VOCODERS:
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
        self.dtype = inference_dtype(self.device, use_fp32)
        device = torch.device(self.device)
        self.cuda = device.type == "cuda"
        self.generators = [torch.default_generator]
        if self.cuda:
            torch.backends.cudnn.benchmark = True
            index = torch.cuda.current_device() if device.index is None else device.index
            self.generators.append(torch.cuda.default_generators[index])
        self.model = self.converter.net_g.to(
            device=self.device, dtype=self.dtype
        )
        self.converter.cpt.pop("weight", None)
        self.pipeline = self.converter.vc
        self.sample_rate = self.converter.tgt_sr
        self.version = self.converter.version
        self.speaker_id = speaker_id
        self.pitch = pitch
        self.index_rate = float(index_rate)
        self.index = None
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
            device=self.device, dtype=self.dtype
        )
        positional = getattr(getattr(self.embedder, "encoder", None), "pos_conv_embed", None)
        if positional is not None and isinstance(positional.conv, torch.nn.Conv1d):
            positional.conv = MatmulGroupedConv1d(positional.conv)
        if self.version == "v1" and not self.embedder.config.do_stable_layer_norm:
            self.embedder.encoder.layers = self.embedder.encoder.layers[:V1_FEATURE_LAYER]
        self.expected_feature_dim = int(self.model.enc_p.emb_phone.in_features)
        self.cuda_graph = bool(cuda_graph)
        self.feature_graph = RealtimeGraph(
            lambda source: extract_embedding_features(self.embedder, source, self.version),
            "content feature", self.cuda_graph,
        )
        self.rmvpe_graph = None
        self.salience_graph = None
        self.salience_model = None
        self.synthesizer_graphs = {}
        self.speaker = torch.tensor([speaker_id], device=self.device, dtype=torch.long)
        self.lengths = {}
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
        self.index = FeatureIndex(index, device=self.device)

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
        if self.index is not None:
            self.index.reset()

    def _seed(self):
        for generator in self.generators:
            generator.manual_seed(self.seed)

    def _lengths(self, frames):
        lengths = self.lengths.get(frames)
        if lengths is None:
            lengths = torch.tensor([frames], device=self.device, dtype=torch.long)
            self.lengths[frames] = lengths
        return lengths

    @torch.inference_mode()
    def skip_block(self, block_frame_16k):
        with self.lock:
            shift = max(1, block_frame_16k // 160)
            self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
            self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
            self.cache_pitch[-shift:] = 1
            self.cache_pitchf[-shift:] = 0.0
            self.rmvpe_viterbi.reset()
            self.pitch_sample_count += int(block_frame_16k)

    def _extract_features(self, input_wav):
        source = input_wav.float().view(1, -1)
        if getattr(self.embedder, "audio_requires_normalization", False):
            source = F.layer_norm(source, source.shape)
        features = self.feature_graph(source.to(dtype=self.dtype))
        self._check_feature_dim(features)
        return torch.cat((features, features[:, -1:, :]), dim=1)

    def _check_feature_dim(self, features):
        if features.shape[-1] != self.expected_feature_dim:
            raise RuntimeError(
                f"{self.embedder_name} outputs {features.shape[-1]} channels, but "
                f"this model expects {self.expected_feature_dim}."
            )

    def _apply_index(self, features, start):
        if self.index is None or self.index_rate <= 0:
            return features
        query = features[0, start:].detach()
        blended = self.index.blend(query, self.index_rate)
        features[0, start:] = blended
        return features

    def _prepare_pitch_predictor(self, method):
        predictor = getattr(self.pipeline, f"model_{method}", None)
        if predictor is None and method == "rmvpe":
            predictor = self.pipeline.rmvpe_class(
                device=self.device, sample_rate=16000, hop_size=160
            )
            self.pipeline.model_rmvpe = predictor
        if predictor is None and method == "swift":
            from shared.predictors.f0 import Swift

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
        if method == "rmvpe":
            viterbi_path(
                np.zeros((2, 1), dtype=np.float32), np.zeros(1, dtype=np.float32),
                np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=np.float32),
            )
        current = predictor
        for _ in range(4):
            if isinstance(current, torch.nn.Module):
                current.to(dtype=self.dtype if method == "rmvpe" else torch.float32)
            mel_extractor = getattr(current, "mel_extractor", None)
            if isinstance(mel_extractor, torch.nn.Module):
                mel_extractor.float()
            current = getattr(current, "model", None)
            if current is None:
                break
        self.prepared_f0_methods.add(method)

    def _release_pitch_predictors(self, method):
        released = False
        for name in PITCH_PREDICTORS:
            if name != method and hasattr(self.pipeline, f"model_{name}"):
                delattr(self.pipeline, f"model_{name}")
                self.prepared_f0_methods.discard(name)
                released = True
        if method != "rmvpe":
            released = released or self.rmvpe_graph is not None or self.salience_graph is not None
            self.rmvpe_graph = None
            self.salience_graph = None
        if released and torch.device(self.device).type == "cuda":
            torch.cuda.empty_cache()

    def _graphs(self):
        graphs = [self.feature_graph, self.rmvpe_graph, self.salience_graph, *self.synthesizer_graphs.values()]
        return [graph for graph in graphs if graph is not None]

    @property
    def graphs_failed(self):
        return any(graph.failed for graph in self._graphs())

    def reset_graphs(self):
        for graph in self._graphs():
            graph.reset()

    def use_full_precision(self):
        self.dtype = torch.float32
        self.model.float()
        self.embedder.float()
        self.prepared_f0_methods.clear()
        self.reset_graphs()

    def _rmvpe_network(self, model):
        if self.rmvpe_graph is None or self.rmvpe_graph.function is not model:
            self.rmvpe_graph = RealtimeGraph(model, "RMVPE", self.cuda_graph)
        return self.rmvpe_graph

    def _salience_network(self, model):
        if self.salience_graph is None or self.salience_model is not model:
            def salience(audio):
                with torch.autocast("cuda", enabled=False):
                    mel = model.mel_extractor(audio)
                return model.mel2hidden(mel)

            self.salience_model = model
            self.salience_graph = RealtimeGraph(salience, "RMVPE", self.cuda_graph)
        return self.salience_graph

    def _rmvpe_pitch(self, source, shift):
        model = self.pipeline.model_rmvpe.model
        salience = self._salience_network(model)(source[None])[0].float().cpu().numpy()
        center = self.rmvpe_viterbi.decode(salience, RMVPE_THRESHOLD, shift)
        f0 = model.decode(salience, thred=RMVPE_THRESHOLD, center=center)
        return self.pipeline.quantize_f0(f0, self.pitch)

    def _synthesizer_graph(self, skip_head, return_length):
        key = (skip_head, return_length)
        if key not in self.synthesizer_graphs:
            self.synthesizer_graphs[key] = RealtimeGraph(
                lambda *inputs: self.model.infer(*inputs, skip_head, return_length)[0],
                "synthesizer", self.cuda_graph,
            )
        return self.synthesizer_graphs[key]

    def _update_pitch(self, input_wav, block_frame_16k, method):
        if method != self.last_f0_method:
            self.rmvpe_viterbi.reset()
            self._release_pitch_predictors(method)
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
        source = input_wav[-extractor_frame:].detach()
        shift = max(1, block_frame_16k // 160)
        if method == "rmvpe":
            pitch, pitchf = self._rmvpe_pitch(source, shift)
        else:
            source = source.cpu().numpy()
            pitch, pitchf = self.pipeline.get_f0(source, source.shape[0] // 160, f0_method=method, pitch=self.pitch)
            if method not in self.prepared_f0_methods:
                self._prepare_pitch_predictor(method)
        values = torch.from_numpy(np.stack((np.ravel(pitch), np.ravel(pitchf))).astype(np.float32)).to(self.device)
        pitch = values[0].long()
        pitchf = values[1]
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
            raise ValueError(
                f"Extra context must be at least {lookahead / 100:g} seconds for this pitch method."
            )
        skip_head = int(skip_head) - lookahead
        started = time.perf_counter()
        with self.lock, torch.autocast("cuda", dtype=torch.float16, enabled=self.dtype == torch.float16):
            self._seed()
            features = self._extract_features(input_wav)
            features = self._apply_index(features, skip_head // 2)
            with torch.autocast("cuda", dtype=torch.float16,
                                enabled=self.dtype == torch.float16 and f0_method == "rmvpe"):
                self._update_pitch(input_wav, block_frame_16k, f0_method)
            features = F.interpolate(
                features.permute(0, 2, 1), scale_factor=2
            ).permute(0, 2, 1)
            p_len = min(input_wav.shape[0] // 160, features.shape[1])
            features = features[:, :p_len]
            coarse = self.cache_pitch[None, -p_len:]
            continuous = self.cache_pitchf[None, -p_len:]
            audio = self._synthesizer_graph(skip_head, int(return_length))(
                features.to(dtype=self.dtype), self._lengths(p_len), coarse, continuous, self.speaker
            )
            if self.cuda:
                torch.cuda.synchronize(self.device)
        return audio.squeeze().float(), time.perf_counter() - started
