import gc
import logging
import sys
import time

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from noisereduce.torchgate import TorchGate
from noisereduce.torchgate.utils import amp_to_db
from torchaudio.transforms import Resample

from beatrice.inference import OUT_SAMPLE_RATE, find_paraphernalia, is_beatrice_checkpoint
from beatrice.realtime import BeatriceRealtime
from rectified_flow.realtime import RealtimeGraph
from rvc.infer.realtime import RealTimeRVC


BEATRICE_BLOCK_TIME = 0.02
AUTOTUNE_HEADROOM = 1 << 30
RESAMPLE_FILTER = dict(
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="sinc_interp_kaiser",
    beta=14.769656459379492,
)
logger = logging.getLogger(__name__)


class BlockTorchGate(TorchGate):
    def forward(self, x, xn=None):
        window = torch.hann_window(self.win_length, device=x.device)
        spectrum = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                              return_complex=True, pad_mode="constant", center=True, window=window)
        if self.nonstationary:
            mask = self._nonstationary_mask(spectrum.abs())
        else:
            mask = self._stationary_mask(amp_to_db(spectrum), xn)
        mask = self.prop_decrease * (mask.float() - 1.0) + 1.0
        if self.smoothing_filter is not None:
            mask = F.conv2d(mask.unsqueeze(1), self.smoothing_filter.to(mask.dtype), padding="same")
        y = torch.istft(spectrum * mask.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length,
                        win_length=self.win_length, center=True, window=window)
        return y.to(dtype=x.dtype)


DEVICE_BUFFERS = (
    "input_wav", "input_wav_denoise", "input_wav_res", "sola_buffer", "sola_den_kernel", "nr_buffer",
    "output_buffer", "fade_in_window", "fade_out_window", "resampler", "resampler2", "tg",
)


def release_flow_models():
    for module_name, cache_name in (("rectified_flow.pitch", "_RMVPE"), ("rectified_flow.variance", "_SEPARATORS")):
        cache = getattr(sys.modules.get(module_name), cache_name, None)
        if cache:
            cache.clear()


def db_to_linear(db):
    return 10.0 ** (float(db) / 20.0)


class RealtimeProcessor:
    def __init__(self):
        self.rvc = None
        self.beatrice = None
        self.settings = {}
        self.sample_rate = 0
        self.channels = 1
        self.device = torch.device("cpu")
        self.last_infer_ms = 0
        self.last_block_ms = 0
        self.input_level = 0.0
        self.output_level = 0.0

    def load_model(self, settings):
        self.settings = settings
        beatrice_path = find_paraphernalia(settings["model_path"])
        if beatrice_path is None and is_beatrice_checkpoint(settings["model_path"]):
            raise ValueError(
                "This is a Beatrice training checkpoint, not a model. Select the paraphernalia_* model folder "
                "or its .toml file from the same training run."
            )
        if beatrice_path is None:
            self.rvc = RealTimeRVC(
                model_path=settings["model_path"],
                index_path=settings["index_path"],
                index_rate=settings["index_rate"],
                pitch=settings["pitch"],
                speaker_id=settings["speaker_id"],
                embedder_model=settings["embedder_model"],
                rectified_vocoder_path=settings.get("rectified_vocoder_path", ""),
                rectified_steps=settings.get("rectified_steps", 0),
                rectified_flow_window=settings.get("rectified_flow_window", True),
                lookahead_frames=round(settings.get("lookahead_time", 0.0) * 100),
                seamless=settings.get("seamless", False),
                cached_context=settings.get("cached_context", False),
                use_fp32=settings.get("use_fp32", False),
            )
            model_rate = self.rvc.sample_rate
            self.device = self.rvc.device
        else:
            model_rate = OUT_SAMPLE_RATE
            self.device = torch.device("cpu")
        return model_rate, beatrice_path

    def prepare(self, beatrice_path=None):
        settings = self.settings
        self._prepare_buffers()
        if beatrice_path is not None:
            self.beatrice = BeatriceRealtime(
                beatrice_path,
                self.sample_rate,
                speaker=settings["speaker_id"],
                pitch=settings["pitch"],
                formant=settings["formant_shift"],
                vq_neighbors=settings["vq_neighbors"],
            )
        self._prewarm()

    def _prepare_buffers(self):
        settings = self.settings
        rate = self.sample_rate
        device = self.device
        self.zc = rate // 100
        block_time = BEATRICE_BLOCK_TIME if self.rvc is None else settings["block_time"]
        self.block_frame = int(np.round(block_time * rate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        if self.rvc is None:
            self.crossfade_frame = self.zc
        else:
            self.crossfade_frame = min(
                int(np.round(settings["crossfade_time"] * rate / self.zc)) * self.zc, self.block_frame
            )
        self.safety_frame = int(np.round(settings.get("safety_buffer", 0.0) * rate / self.zc)) * self.zc
        self.sola_buffer_frame = self.crossfade_frame
        self.sola_search_frame = 0 if self.rvc is not None and self.rvc.seamless else self.zc
        self.extra_frame = int(np.round(settings["extra_time"] * rate / self.zc)) * self.zc
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=device,
            dtype=torch.float32,
        )
        self.input_wav_denoise = self.input_wav.clone()
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc, device=device, dtype=torch.float32
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.nr_buffer = self.sola_buffer.clone()
        self.output_buffer = self.input_wav.clone()
        self.skip_head = self.extra_frame // self.zc
        self.return_length = (self.block_frame + self.sola_buffer_frame + self.sola_search_frame) // self.zc
        self.fade_in_window = (
            torch.sin(
                0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.sola_buffer_frame, device=device,
                                             dtype=torch.float32)
            )
            ** 2
        )
        self.fade_out_window = 1 - self.fade_in_window
        graph_enabled = self.rvc is not None and self.rvc.cuda_graph
        self.resampler = RealtimeGraph(
            Resample(orig_freq=rate, new_freq=16000, dtype=torch.float32, **RESAMPLE_FILTER).to(device),
            "input resample", graph_enabled,
        )
        self.resampler2 = None
        if self.rvc is not None and self.rvc.sample_rate != rate:
            self.resampler2 = RealtimeGraph(
                Resample(orig_freq=self.rvc.sample_rate, new_freq=rate, dtype=torch.float32, **RESAMPLE_FILTER).to(device),
                "output resample", graph_enabled,
            )
        self.tg = BlockTorchGate(sr=rate, n_fft=4 * self.zc, prop_decrease=0.9).to(device)

    def _reset_buffers(self):
        for buffer in (self.input_wav, self.input_wav_denoise, self.input_wav_res, self.output_buffer,
                       self.sola_buffer, self.nr_buffer):
            buffer.zero_()
        self.rms_buffer[:] = 0
        if self.rvc is not None:
            self.rvc.reset_caches()
        if self.beatrice is not None:
            self.beatrice.reset()

    def _warm_up(self):
        silence = np.zeros((self.block_frame, self.channels), dtype=np.float32)
        for _ in range(2 if self.beatrice is None else 3):
            self.process(silence, warm_up=True)

    def _budgeted_warm_up(self):
        device = torch.device(self.device)
        if self.rvc is None or device.type != "cuda" or not torch.backends.cudnn.benchmark:
            return False
        limit = torch.cuda.memory_reserved(device) + AUTOTUNE_HEADROOM
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, limit / total), device)
        try:
            self._warm_up()
            succeeded = not self.rvc.graphs_failed
        except torch.OutOfMemoryError:
            succeeded = False
        finally:
            torch.cuda.set_per_process_memory_fraction(1.0, device)
        if not succeeded:
            self._reset_buffers()
            self.rvc.reset_graphs()
            torch.cuda.empty_cache()
        return succeeded

    def _prewarm(self):
        try:
            if not self._budgeted_warm_up():
                self._warm_up()
        except (RuntimeError, FloatingPointError) as error:
            if self.rvc is None or self.rvc.dtype != torch.float16:
                raise
            logger.warning("Realtime FP16 warm-up failed; retrying with FP32: %s", error)
            self.rvc.use_full_precision()
            for graph in (self.resampler, self.resampler2):
                if graph is not None:
                    graph.reset()
            self._reset_buffers()
            if not self._budgeted_warm_up():
                self._warm_up()
        finally:
            self._reset_buffers()

    def _release_models(self):
        self.rvc = None
        self.beatrice = None
        for name in DEVICE_BUFFERS:
            setattr(self, name, None)
        release_flow_models()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_pitch(self, pitch):
        if self.beatrice is not None:
            self.beatrice.model.set_pitch_shift(pitch)
        elif self.rvc is not None:
            self.rvc.change_pitch(pitch)

    def update_index_rate(self, index_rate):
        if self.rvc is not None:
            self.rvc.change_index_rate(index_rate)

    def update_beatrice(self, speaker, formant_shift, vq_neighbors):
        if self.beatrice is not None:
            self.beatrice.model.set_speaker(speaker)
            self.beatrice.model.set_formant_shift(formant_shift)
            self.beatrice.model.set_vq_neighbors(vq_neighbors)

    def _gate(self, indata):
        indata = np.append(self.rms_buffer, indata)
        rms = librosa.feature.rms(y=indata, frame_length=4 * self.zc, hop_length=self.zc)[:, 2:]
        self.rms_buffer[:] = indata[-4 * self.zc :]
        indata = indata[2 * self.zc - self.zc // 2 :]
        quiet = librosa.amplitude_to_db(rms, ref=1.0)[0] < self.settings["threshold"]
        for index in range(quiet.shape[0]):
            if quiet[index]:
                indata[index * self.zc : (index + 1) * self.zc] = 0
        return indata[self.zc // 2 :]

    def _mix_volume(self, converted, source):
        rate = self.settings["rms_mix_rate"]
        lookahead = self.rvc.lookahead_frames(self.settings["f0_method"])
        start = self.extra_frame - lookahead * self.zc
        source = source[start : start + converted.shape[0]]
        rms_source = librosa.feature.rms(
            y=source.cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc
        )
        rms_source = torch.from_numpy(rms_source).to(self.device)
        rms_source = F.interpolate(
            rms_source.unsqueeze(0), size=converted.shape[0] + 1, mode="linear", align_corners=True
        )[0, 0, :-1]
        rms_converted = librosa.feature.rms(
            y=converted[:].cpu().numpy(), frame_length=4 * self.zc, hop_length=self.zc
        )
        rms_converted = torch.from_numpy(rms_converted).to(self.device)
        rms_converted = F.interpolate(
            rms_converted.unsqueeze(0), size=converted.shape[0] + 1, mode="linear", align_corners=True
        )[0, 0, :-1]
        rms_converted = torch.max(rms_converted, torch.zeros_like(rms_converted) + 1e-3)
        return converted * torch.pow(rms_source / rms_converted, 1.0 - rate)

    def _apply_sola(self, converted):
        needed = self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        if converted.shape[0] < needed:
            converted = F.pad(converted, (0, needed - converted.shape[0]))
        if self.sola_search_frame:
            conv_input = converted[None, None, : self.sola_buffer_frame + self.sola_search_frame]
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
            cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
            sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
            converted = converted[sola_offset:]
        converted[: self.sola_buffer_frame] *= self.fade_in_window
        converted[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = converted[self.block_frame : self.block_frame + self.sola_buffer_frame]
        return converted[: self.block_frame]

    def process(self, indata, warm_up=False):
        started = time.perf_counter()
        settings = self.settings
        indata = librosa.to_mono(indata.T).astype(np.float32)
        indata *= np.float32(db_to_linear(settings["input_gain_db"]))
        np.clip(indata, -1.0, 1.0, out=indata)
        self.input_level = float(np.abs(indata).max(initial=0.0))
        if settings["threshold"] > -60:
            indata = self._gate(indata)
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        denoise = settings["input_noise_reduce"]
        if denoise:
            self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[self.block_frame :].clone()
            input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]
            input_wav = self.tg(input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)).squeeze(0)
            input_wav[: self.sola_buffer_frame] *= self.fade_in_window
            input_wav[: self.sola_buffer_frame] += self.nr_buffer * self.fade_out_window
            self.input_wav_denoise[-self.block_frame :] = input_wav[: self.block_frame]
            self.nr_buffer[:] = input_wav[self.block_frame :]
            if self.rvc is not None:
                resample_input = self.input_wav_denoise[-self.block_frame - 2 * self.zc :]
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(resample_input)[160:]
        elif self.rvc is not None:
            resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
            self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = self.resampler(resample_input)[160:]
        source = self.input_wav_denoise if denoise else self.input_wav
        passthrough = settings["passthrough"] and not warm_up
        infer_seconds = 0.0
        silent = passthrough
        if self.beatrice is not None:
            block = source[-self.block_frame :]
            if passthrough:
                infer_wav = torch.zeros_like(block)
            else:
                infer_started = time.perf_counter()
                infer_wav = torch.from_numpy(self.beatrice.process(block.cpu().numpy())).to(self.device)
                infer_seconds = time.perf_counter() - infer_started
            if settings["output_noise_reduce"] and not passthrough:
                self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
                self.output_buffer[-self.block_frame :] = infer_wav
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
            output_block = infer_wav[: self.block_frame]
        else:
            lookahead = self.rvc.lookahead_frames(settings["f0_method"])
            quiet = not warm_up and not bool(self.input_wav[max(0, self.extra_frame - lookahead * self.zc) :].any())
            silent = passthrough or quiet
            if silent:
                self.rvc.skip_block(self.block_frame_16k, self.input_wav_res, not quiet)
                infer_wav = torch.zeros_like(source[self.extra_frame :])
            else:
                infer_wav, infer_seconds = self.rvc.infer(
                    self.input_wav_res,
                    self.block_frame_16k,
                    self.skip_head,
                    self.return_length,
                    settings["f0_method"],
                    source,
                    self.sample_rate,
                )
                if self.resampler2 is not None:
                    infer_wav = self.resampler2(infer_wav)
                else:
                    infer_wav = infer_wav.clone()
            if settings["output_noise_reduce"] and not silent:
                self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
            if settings["rms_mix_rate"] < 1 and not silent:
                infer_wav = self._mix_volume(infer_wav, source)
            output_block = self._apply_sola(infer_wav)
        gain = db_to_linear(settings["output_gain_db"])
        if warm_up and not bool(torch.isfinite(output_block).all()):
            raise FloatingPointError("Realtime warm-up produced non-finite audio.")
        output = torch.clamp(output_block * gain, -1.0, 1.0).cpu().numpy().astype(np.float32)
        if not passthrough:
            self.output_level = float(np.abs(output).max(initial=0.0))
        if not silent:
            self.last_infer_ms = round(infer_seconds * 1000)
            self.last_block_ms = round((time.perf_counter() - started) * 1000)
        self._refresh_latency()
        return output

    def _refresh_latency(self):
        pass
