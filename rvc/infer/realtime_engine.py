import gc
import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from rvc.infer.realtime import RealTimeRVC
from rvc.infer.realtime_graph import RealtimeGraph
from shared.tools.causal_voices import SAMPLE_RATE as CAUSAL_SAMPLE_RATE, is_causal_voice


AUTOTUNE_HEADROOM = 1 << 30
PITCH_METHODS = ("rmvpe", "swift", "pm")
RESAMPLE_FILTER = dict(
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="sinc_interp_kaiser",
    beta=14.769656459379492,
)
logger = logging.getLogger(__name__)


def frame_rms(signal, frame_length, hop_length):
    power = np.pad(np.square(signal), frame_length // 2)
    frames = np.lib.stride_tricks.sliding_window_view(power, frame_length)[::hop_length]
    return np.sqrt(frames.mean(axis=1))


DEVICE_BUFFERS = (
    "input_wav", "input_wav_res", "sola_buffer", "sola_den_kernel", "sola_positions",
    "fade_in_window", "fade_out_window", "resampler", "resampler2",
)


def db_to_linear(db):
    return 10.0 ** (float(db) / 20.0)


class RealtimeProcessor:
    def __init__(self):
        self.rvc = None
        self.causal = None
        self.rnnoise = None
        self._rnnoise_active = False
        self.settings = {}
        self.sample_rate = 0
        self.channels = 1
        self.device = torch.device("cpu")
        self.last_infer_ms = 0
        self.last_block_ms = 0
        self.input_level = 0.0
        self.output_level = 0.0

    def ensure_rnnoise(self):
        if self.rnnoise is not None and self.rnnoise.sample_rate == self.sample_rate:
            return self.rnnoise
        if self.rnnoise is not None:
            self.rnnoise.close()
            self.rnnoise = None
        from shared.tools.rnnoise import RNNoiseDenoiser

        self.rnnoise = RNNoiseDenoiser(self.sample_rate)
        self._rnnoise_active = False
        return self.rnnoise

    def load_model(self, settings):
        self.settings = settings
        if is_causal_voice(settings["model_path"]):
            from rvc.configs.config import Config
            from shared.tools.causal_realtime import CausalRealtime

            self.device = torch.device(Config().device)
            self.causal = CausalRealtime(settings, self.device)
            return CAUSAL_SAMPLE_RATE
        self.rvc = RealTimeRVC(
            model_path=settings["model_path"],
            index_path=settings["index_path"],
            index_rate=settings["index_rate"],
            pitch=settings["pitch"],
            speaker_id=settings["speaker_id"],
            embedder_model=settings["embedder_model"],
            use_fp32=settings.get("use_fp32", False),
        )
        self.device = self.rvc.device
        return self.rvc.sample_rate

    @torch.inference_mode()
    def prepare(self):
        self._rnnoise_active = False
        if self.rnnoise is not None and self.rnnoise.sample_rate != self.sample_rate:
            self.rnnoise.close()
            self.rnnoise = None
        if self.settings.get("input_noise_reduce"):
            self.ensure_rnnoise().reset()
        if self.causal is not None:
            self.block_frame = self.causal.prepare(self.sample_rate)
            self.zc = self.sample_rate // 100
            self.sola_buffer_frame = 0
            return
        self._prepare_buffers()
        self._prewarm()

    @property
    def conversion_delay(self):
        if self.causal is not None:
            return self.causal.delay_seconds
        lookahead = self.rvc.pitch_lookahead_frames(self.settings["f0_method"])
        return (self.crossfade_frame + self.sola_search_frame) / self.sample_rate + lookahead / 100

    def _prepare_buffers(self):
        settings = self.settings
        rate = self.sample_rate
        device = self.device
        self.zc = rate // 100
        block_time = settings["block_time"]
        self.block_frame = int(np.round(block_time * rate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = min(
            int(np.round(settings["crossfade_time"] * rate / self.zc)) * self.zc, self.block_frame
        )
        self.sola_buffer_frame = self.crossfade_frame
        self.sola_search_frame = self.zc
        self.extra_frame = int(np.round(settings["extra_time"] * rate / self.zc)) * self.zc
        self.input_wav = torch.zeros(
            self.extra_frame + self.crossfade_frame + self.sola_search_frame + self.block_frame,
            device=device,
            dtype=torch.float32,
        )
        self.input_wav_res = torch.zeros(
            160 * self.input_wav.shape[0] // self.zc, device=device, dtype=torch.float32
        )
        lookahead = max(RealTimeRVC.pitch_lookahead_frames(method) for method in PITCH_METHODS)
        self.recent_input = np.zeros(
            self.input_wav.shape[0] - max(0, self.extra_frame - lookahead * self.zc), dtype=np.float32
        )
        self.rms_buffer = np.zeros(4 * self.zc, dtype="float32")
        self.sola_buffer = torch.zeros(self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.sola_den_kernel = torch.ones(1, 1, self.sola_buffer_frame, device=device, dtype=torch.float32)
        self.sola_positions = torch.arange(self.block_frame + self.sola_buffer_frame, device=device)
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
        graph_enabled = self.rvc.cuda_graph
        self.resampler = RealtimeGraph(
            Resample(orig_freq=rate, new_freq=16000, dtype=torch.float32, **RESAMPLE_FILTER).to(device),
            "input resample", graph_enabled,
        )
        self.resampler2 = None
        if self.rvc.sample_rate != rate:
            self.resampler2 = RealtimeGraph(
                Resample(orig_freq=self.rvc.sample_rate, new_freq=rate, dtype=torch.float32, **RESAMPLE_FILTER).to(device),
                "output resample", graph_enabled,
            )

    def _reset_buffers(self):
        for buffer in (self.input_wav, self.input_wav_res, self.sola_buffer):
            buffer.zero_()
        self.rms_buffer[:] = 0
        self.recent_input[:] = 0
        self._rnnoise_active = False
        if self.rnnoise is not None:
            self.rnnoise.reset()
        if self.rvc is not None:
            self.rvc.reset_caches()

    def _warm_up(self):
        silence = np.zeros((self.block_frame, self.channels), dtype=np.float32)
        for _ in range(2):
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
        self.causal = None
        self._rnnoise_active = False
        if self.rnnoise is not None:
            self.rnnoise.close()
            self.rnnoise = None
        for name in DEVICE_BUFFERS:
            setattr(self, name, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_pitch(self, pitch):
        if self.causal is not None:
            self.causal.change_pitch(pitch)
        elif self.rvc is not None:
            self.rvc.change_pitch(pitch)

    def update_index_rate(self, index_rate):
        if self.causal is not None:
            self.causal.change_index_rate(index_rate)
        elif self.rvc is not None:
            self.rvc.change_index_rate(index_rate)

    def update_pitch_map(self):
        if self.causal is not None:
            self.causal.change_pitch_map()

    def _gate(self, indata):
        indata = np.append(self.rms_buffer, indata)
        rms = frame_rms(indata, 4 * self.zc, self.zc)[2:]
        self.rms_buffer[:] = indata[-4 * self.zc :]
        indata = indata[2 * self.zc - self.zc // 2 :]
        quiet = 20 * np.log10(np.maximum(rms, 1e-5)) < self.settings["threshold"]
        indata[np.repeat(quiet, self.zc)[: indata.shape[0]]] = 0
        return indata[self.zc // 2 :]

    def _rms(self, audio, size):
        frame = 4 * self.zc
        power = F.pad(audio.square()[None, None], (frame // 2, frame // 2))
        rms = F.avg_pool1d(power, frame, self.zc).sqrt_()
        return F.interpolate(rms, size=size + 1, mode="linear", align_corners=True)[0, 0, :-1]

    def _mix_volume(self, converted, source):
        rate = self.settings["rms_mix_rate"]
        start = self.extra_frame - self.rvc.pitch_lookahead_frames(self.settings["f0_method"]) * self.zc
        size = converted.shape[0]
        rms_source = self._rms(source[start : start + size], size)
        rms_converted = self._rms(converted, size).clamp_min_(1e-3)
        return converted * torch.pow(rms_source / rms_converted, 1.0 - rate)

    def _apply_sola(self, converted):
        needed = self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        if converted.shape[0] < needed:
            converted = F.pad(converted, (0, needed - converted.shape[0]))
        conv_input = converted[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        converted = converted[self.sola_positions + torch.argmax(cor_nom[0, 0] / cor_den[0, 0])]
        converted[: self.sola_buffer_frame] *= self.fade_in_window
        converted[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = converted[self.block_frame :]
        return converted[: self.block_frame]

    @torch.inference_mode()
    def process(self, indata, warm_up=False):
        started = time.perf_counter()
        settings = self.settings
        indata = indata.mean(axis=1, dtype=np.float32)
        indata *= np.float32(db_to_linear(settings["input_gain_db"]))
        np.clip(indata, -1.0, 1.0, out=indata)
        self.input_level = float(np.abs(indata).max(initial=0.0))
        if settings.get("input_noise_reduce") and not warm_up:
            denoiser = self.ensure_rnnoise()
            if not self._rnnoise_active:
                denoiser.reset()
                self._rnnoise_active = True
            indata = denoiser.process(indata)
        elif self._rnnoise_active:
            self._rnnoise_active = False
            if self.rnnoise is not None:
                self.rnnoise.reset()
        if self.causal is not None:
            return self._process_causal(indata, started)
        if settings["threshold"] > -60:
            indata = self._gate(indata)
        self.input_wav[: -self.block_frame] = self.input_wav[self.block_frame :].clone()
        self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(self.device)
        recent = self.recent_input
        recent[: -self.block_frame] = recent[self.block_frame :]
        tail = indata[-recent.shape[0] :]
        recent[-tail.shape[0] :] = tail
        self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[self.block_frame_16k :].clone()
        resample_input = self.input_wav[-indata.shape[0] - 2 * self.zc :]
        self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = self.resampler(resample_input)[160:]
        passthrough = settings["passthrough"] and not warm_up
        infer_seconds = 0.0
        start = max(0, self.extra_frame - self.rvc.pitch_lookahead_frames(settings["f0_method"]) * self.zc)
        quiet = not warm_up and not self.recent_input[start - self.input_wav.shape[0] :].any()
        silent = passthrough or quiet
        if silent:
            self.rvc.skip_block(self.block_frame_16k)
            infer_wav = torch.zeros_like(self.input_wav[self.extra_frame :])
        else:
            infer_wav, infer_seconds = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                settings["f0_method"],
            )
            if self.resampler2 is not None:
                infer_wav = self.resampler2(infer_wav)
            else:
                infer_wav = infer_wav.clone()
        if settings["rms_mix_rate"] < 1 and not silent:
            infer_wav = self._mix_volume(infer_wav, self.input_wav)
        output_block = self._apply_sola(infer_wav)
        gain = db_to_linear(settings["output_gain_db"])
        if warm_up and not bool(torch.isfinite(output_block).all()):
            raise FloatingPointError("Realtime warm-up produced non-finite audio.")
        output = torch.clamp(output_block * gain, -1.0, 1.0).cpu().numpy()
        if not passthrough:
            self.output_level = float(np.abs(output).max(initial=0.0))
        if not silent:
            self.last_infer_ms = round(infer_seconds * 1000)
            self.last_block_ms = round((time.perf_counter() - started) * 1000)
        self._refresh_latency()
        return output

    def _process_causal(self, samples, started):
        output = self.causal.process(samples)
        self.last_infer_ms = round((time.perf_counter() - started) * 1000)
        if self.settings["passthrough"]:
            # The desktop audio driver supplies the unconverted monitoring signal.
            output = np.zeros_like(output)
        else:
            output = self.causal.output_gain(output, db_to_linear(self.settings["output_gain_db"]))
            self.output_level = float(np.abs(output).max(initial=0.0))
        self.last_block_ms = round((time.perf_counter() - started) * 1000)
        self._refresh_latency()
        return output

    def _refresh_latency(self):
        pass
