import json
import os
import queue
import threading
import time
import traceback
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from noisereduce.torchgate import TorchGate
from noisereduce.torchgate.utils import amp_to_db
from torchaudio.transforms import Resample

from rvc.beatrice.inference import OUT_SAMPLE_RATE, find_paraphernalia
from rvc.beatrice.realtime import BeatriceRealtime
from rvc.infer.realtime import RealTimeRVC


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "assets" / "realtime_config.json"
MONITOR_DISABLED = "Disabled"
MAIN_HOST_APIS = ("ASIO", "Windows WASAPI")


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


def db_to_linear(db):
    return 10.0 ** (float(db) / 20.0)


def enqueue_latest(block_queue, block):
    try:
        block_queue.put_nowait(block)
        return
    except queue.Full:
        pass
    try:
        block_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        block_queue.put_nowait(block)
    except queue.Full:
        pass


class AudioFrameFifo:
    def __init__(self, channels=1, max_frames=None):
        self.channels = int(channels)
        self.max_frames = None if max_frames is None else int(max_frames)
        self._chunks = deque()
        self._frames = 0
        self._lock = threading.Lock()

    def write(self, data):
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if not array.shape[0]:
            return
        with self._lock:
            self._chunks.append(array.copy())
            self._frames += array.shape[0]
            if self.max_frames is not None and self._frames > self.max_frames:
                self._discard_locked(self._frames - self.max_frames)

    def read(self, frames):
        with self._lock:
            take = min(int(frames), self._frames)
            if take <= 0:
                return None
            parts = []
            remaining = take
            while remaining:
                chunk = self._chunks[0]
                count = min(remaining, chunk.shape[0])
                parts.append(chunk[:count])
                if count == chunk.shape[0]:
                    self._chunks.popleft()
                else:
                    self._chunks[0] = chunk[count:]
                self._frames -= count
                remaining -= count
        return parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)

    def _discard_locked(self, frames):
        remaining = int(frames)
        while remaining > 0 and self._chunks:
            chunk = self._chunks[0]
            count = min(remaining, chunk.shape[0])
            if count == chunk.shape[0]:
                self._chunks.popleft()
            else:
                self._chunks[0] = chunk[count:]
            self._frames -= count
            remaining -= count


def friendly_device_label(api_name, device_name, direction):
    if "voicemeeter" in device_name.lower():
        if api_name == "ASIO":
            return f"[Voicemeeter ASIO] {device_name}"
        arrow = "→ RVC input" if direction == "input" else "← RVC output"
        return f"[Voicemeeter] {device_name} {arrow}"
    api_label = "WASAPI" if api_name == "Windows WASAPI" else api_name
    return f"[{api_label}] {device_name}"


def device_sort_key(label):
    lower_label = label.lower()
    if "voicemeeter" in lower_label:
        priority = 0
    elif label.startswith("[ASIO]"):
        priority = 1
    elif label.startswith("[WASAPI]"):
        priority = 2
    else:
        priority = 3
    return priority, lower_label


class AudioEngine:
    def __init__(self, error_queue):
        self.error_queue = error_queue
        self.input_stream = None
        self.output_stream = None
        self.monitor_stream = None
        self.output_queue = None
        self.monitor_queue = None
        self.rvc = None
        self.beatrice = None
        self.running = False
        self.settings = {}
        self.sample_rate = 0
        self.last_infer_ms = 0
        self.last_block_ms = 0
        self.algorithm_latency_ms = 0
        self.base_latency_ms = 0.0
        self.reported_statuses = set()
        self.shared_fallbacks = []

    def start(self, settings):
        self.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.settings = settings
        beatrice_path = find_paraphernalia(settings["model_path"])
        self.rvc = None
        self.beatrice = None
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
            )
            model_rate = self.rvc.sample_rate
            self.device = self.rvc.device
        else:
            model_rate = OUT_SAMPLE_RATE
            self.device = torch.device("cpu")
        self.input_device = settings["input_device"]
        self.output_device = settings["output_device"]
        self.monitor_device = settings.get("monitor_device")
        if self.monitor_device == self.output_device:
            self.monitor_device = None
        self.channels = self._channels(self.input_device, "max_input_channels")
        self.output_channels = self._channels(self.output_device, "max_output_channels")
        self.monitor_channels = (
            self._channels(self.monitor_device, "max_output_channels")
            if self.monitor_device is not None
            else 0
        )
        self.sample_rate = self._routing_samplerate(model_rate)
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
        self.reported_statuses = set()
        self.output_queue = queue.Queue(maxsize=3)
        self.output_stream = sd.OutputStream(
            callback=self._output_callback,
            blocksize=self.block_frame,
            samplerate=self.sample_rate,
            channels=self.output_channels,
            device=self.output_device,
            dtype="float32",
            extra_settings=self.output_extra,
        )
        self.input_stream = sd.InputStream(
            callback=self._input_callback,
            blocksize=self.block_frame,
            samplerate=self.sample_rate,
            channels=self.channels,
            device=self.input_device,
            dtype="float32",
            extra_settings=self.input_extra,
        )
        self.running = True
        self.output_stream.start()
        self.input_stream.start()
        self._start_monitor_stream()
        input_latency = self.input_stream.latency
        output_latency = self.output_stream.latency
        if self.beatrice is not None:
            model_latency = self.beatrice.latency_seconds
        else:
            model_latency = settings["crossfade_time"] + 0.01
        self.base_latency_ms = (
            input_latency + output_latency + settings["block_time"] + model_latency
        ) * 1000
        self._refresh_latency()

    def _channels(self, device, key):
        return max(1, min(int(sd.query_devices(device)[key]), 2))

    def _routing_samplerate(self, model_rate):
        exclusive = self.settings.get("wasapi_exclusive", False)

        def check(checker, device, channels, rate):
            api = sd.query_hostapis(sd.query_devices(device)["hostapi"])["name"]
            if "WASAPI" in api and exclusive:
                settings = sd.WasapiSettings(exclusive=True)
                try:
                    checker(device=device, channels=channels, dtype="float32",
                            samplerate=rate, extra_settings=settings)
                    return settings, False
                except sd.PortAudioError:
                    pass
                checker(device=device, channels=channels, dtype="float32", samplerate=rate)
                return None, True
            checker(device=device, channels=channels, dtype="float32", samplerate=rate)
            return None, False

        candidates = []
        for rate in (
            sd.query_devices(self.input_device)["default_samplerate"],
            sd.query_devices(self.output_device)["default_samplerate"],
            48000,
            44100,
            model_rate,
            40000,
        ):
            rate = int(round(rate))
            if rate > 0 and rate not in candidates:
                candidates.append(rate)
        errors = []
        for rate in candidates:
            try:
                self.input_extra, input_shared = check(
                    sd.check_input_settings, self.input_device, self.channels, rate
                )
                self.output_extra, output_shared = check(
                    sd.check_output_settings, self.output_device, self.output_channels, rate
                )
                self.monitor_extra, monitor_shared = None, False
                if self.monitor_device is not None:
                    self.monitor_extra, monitor_shared = check(
                        sd.check_output_settings, self.monitor_device, self.monitor_channels, rate
                    )
                self.shared_fallbacks = [
                    sd.query_devices(device)["name"]
                    for shared, device in (
                        (input_shared, self.input_device),
                        (output_shared, self.output_device),
                        (monitor_shared, self.monitor_device),
                    )
                    if shared and device is not None
                ]
                return rate
            except sd.PortAudioError as error:
                errors.append(f"{rate} Hz: {error}")
        raise ValueError(
            "The selected input, output and monitor devices have no common sample rate. "
            "Set them to the same rate (usually 48 kHz or 44.1 kHz).\n" + "\n".join(errors[-3:])
        )

    def _prepare_buffers(self):
        settings = self.settings
        rate = self.sample_rate
        device = self.device
        self.zc = rate // 100
        self.block_frame = int(np.round(settings["block_time"] * rate / self.zc)) * self.zc
        self.block_frame_16k = 160 * self.block_frame // self.zc
        self.crossfade_frame = int(np.round(settings["crossfade_time"] * rate / self.zc)) * self.zc
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)
        self.sola_search_frame = self.zc
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
        self.resampler = Resample(orig_freq=rate, new_freq=16000, dtype=torch.float32).to(device)
        self.resampler2 = None
        if self.rvc is not None and self.rvc.sample_rate != rate:
            self.resampler2 = Resample(
                orig_freq=self.rvc.sample_rate, new_freq=rate, dtype=torch.float32
            ).to(device)
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

    def _prewarm(self):
        try:
            silence = np.zeros((self.block_frame, self.channels), dtype=np.float32)
            for _ in range(2 if self.beatrice is None else 3):
                self._process(silence)
        finally:
            self._reset_buffers()

    def _start_monitor_stream(self):
        self.monitor_stream = None
        self.monitor_queue = None
        if self.monitor_device is None:
            return
        try:
            self.monitor_queue = AudioFrameFifo(1, max_frames=self.block_frame * 4)
            self.monitor_stream = sd.OutputStream(
                device=self.monitor_device,
                callback=self._monitor_callback,
                blocksize=0,
                samplerate=self.sample_rate,
                channels=self.monitor_channels,
                dtype="float32",
                extra_settings=self.monitor_extra,
            )
            self.monitor_stream.start()
        except Exception as error:
            if self.monitor_stream is not None:
                self.monitor_stream.close()
            self.monitor_stream = None
            self.monitor_queue = None
            self.error_queue.put_nowait(
                f"The monitor device could not start; continuing with the main output only. {error}"
            )

    def _refresh_latency(self):
        lookahead_ms = 0
        if self.rvc is not None and not self.settings["passthrough"]:
            lookahead_ms = 10 * self.rvc.pitch_lookahead_frames(self.settings["f0_method"])
        noise_reduction_ms = 0
        if self.settings["input_noise_reduce"]:
            noise_reduction_ms = 1000 * min(self.settings["crossfade_time"], 0.04)
        self.algorithm_latency_ms = round(
            self.base_latency_ms + lookahead_ms + noise_reduction_ms + self.last_block_ms
        )

    def stop(self):
        self.running = False
        for name in ("input_stream", "output_stream", "monitor_stream"):
            stream = getattr(self, name)
            if stream is None:
                continue
            try:
                if stream.active:
                    stream.abort()
            except sd.PortAudioError:
                pass
            finally:
                try:
                    stream.close()
                except sd.PortAudioError:
                    pass
                setattr(self, name, None)
        self.output_queue = None
        self.monitor_queue = None

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
        lookahead = self.rvc.pitch_lookahead_frames(self.settings["f0_method"])
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
        conv_input = converted[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, self.sola_den_kernel) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]).item())
        converted = converted[sola_offset:]
        converted[: self.sola_buffer_frame] *= self.fade_in_window
        converted[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out_window
        self.sola_buffer[:] = converted[self.block_frame : self.block_frame + self.sola_buffer_frame]
        return converted[: self.block_frame]

    def _process(self, indata):
        started = time.perf_counter()
        settings = self.settings
        indata = librosa.to_mono(indata.T).astype(np.float32)
        indata *= np.float32(db_to_linear(settings["input_gain_db"]))
        np.clip(indata, -1.0, 1.0, out=indata)
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
        passthrough = settings["passthrough"]
        infer_seconds = 0.0
        if self.beatrice is not None:
            block = source[-self.block_frame :]
            if passthrough:
                infer_wav = block.clone()
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
            if passthrough:
                self.rvc.pitch_sample_count += self.block_frame_16k
                infer_wav = source[self.extra_frame :].clone()
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
            if settings["output_noise_reduce"] and not passthrough:
                self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
            if settings["rms_mix_rate"] < 1 and not passthrough:
                infer_wav = self._mix_volume(infer_wav, source)
            output_block = self._apply_sola(infer_wav)
        gain = db_to_linear(settings["output_gain_db"])
        output = torch.clamp(output_block * gain, -1.0, 1.0).cpu().numpy().astype(np.float32)
        self.last_infer_ms = round(infer_seconds * 1000)
        self.last_block_ms = round((time.perf_counter() - started) * 1000)
        self._refresh_latency()
        return output

    def _report_status(self, status):
        text = str(status)
        if text not in self.reported_statuses:
            self.reported_statuses.add(text)
            self.error_queue.put_nowait(text)

    def _input_callback(self, indata, frames, times, status):
        if not self.running:
            return
        try:
            if status:
                self._report_status(status)
            output = self._process(indata)
            output_queue = self.output_queue
            if output_queue is not None:
                enqueue_latest(output_queue, output)
            monitor_queue = self.monitor_queue
            if monitor_queue is not None:
                monitor_queue.write(
                    np.clip(output * np.float32(db_to_linear(self.settings["monitor_gain_db"])), -1.0, 1.0)
                )
        except Exception:
            self.running = False
            self.error_queue.put_nowait(traceback.format_exc())
            raise sd.CallbackAbort

    def _output_callback(self, outdata, frames, times, status):
        outdata.fill(0)
        output_queue = self.output_queue
        if output_queue is None:
            return
        try:
            block = output_queue.get_nowait()
        except queue.Empty:
            return
        count = min(frames, block.shape[0])
        outdata[:count] = block[:count, None]

    def _monitor_callback(self, outdata, frames, times, status):
        outdata.fill(0)
        monitor_queue = self.monitor_queue
        block = monitor_queue.read(frames) if monitor_queue is not None else None
        if block is None:
            return
        outdata[: block.shape[0]] = block[:, :1]


class RealtimeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Applio Real-Time Voice Conversion")
        self.root.minsize(760, 620)
        self.error_queue = queue.Queue()
        self.engine = AudioEngine(self.error_queue)
        self.input_devices = {}
        self.output_devices = {}
        self.device_names = {}
        self.saved = self._load_config()
        self._make_variables()
        self._build()
        self._load_devices()
        for variable in (
            self.pitch, self.index_rate, self.rms_mix_rate, self.threshold, self.f0_method,
            self.passthrough, self.speaker_id, self.formant_shift, self.vq_neighbors,
            self.input_gain_db, self.output_gain_db, self.monitor_gain_db,
            self.input_noise_reduce, self.output_noise_reduce,
        ):
            variable.trace_add("write", self._hot_update)
        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _load_config(self):
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _make_variables(self):
        value = self.saved
        self.model_path = tk.StringVar(value=value.get("model_path", ""))
        self.index_path = tk.StringVar(value=value.get("index_path", ""))
        self.rectified_vocoder_path = tk.StringVar(value=value.get("rectified_vocoder_path", ""))
        self.rectified_steps = tk.IntVar(value=value.get("rectified_steps", 0))
        self.rectified_flow_window = tk.BooleanVar(value=value.get("rectified_flow_window", True))
        self.embedder_model = tk.StringVar(
            value=value.get("embedder_model", "contentvec")
        )
        self.input_device = tk.StringVar(value=value.get("input_device", ""))
        self.output_device = tk.StringVar(value=value.get("output_device", ""))
        self.monitor_device = tk.StringVar(value=value.get("monitor_device", MONITOR_DISABLED))
        self.show_legacy_devices = tk.BooleanVar(value=value.get("show_legacy_devices", False))
        self.wasapi_exclusive = tk.BooleanVar(
            value=value.get("wasapi_exclusive", False)
        )
        self.pitch = tk.IntVar(value=value.get("pitch", 0))
        self.speaker_id = tk.IntVar(value=value.get("speaker_id", 0))
        self.index_rate = tk.DoubleVar(value=value.get("index_rate", 0.0))
        self.rms_mix_rate = tk.DoubleVar(value=value.get("rms_mix_rate", 0.0))
        self.threshold = tk.IntVar(value=value.get("threshold", -60))
        self.input_gain_db = tk.DoubleVar(value=value.get("input_gain_db", 0.0))
        self.output_gain_db = tk.DoubleVar(value=value.get("output_gain_db", 0.0))
        self.monitor_gain_db = tk.DoubleVar(value=value.get("monitor_gain_db", 0.0))
        self.input_noise_reduce = tk.BooleanVar(value=value.get("input_noise_reduce", False))
        self.output_noise_reduce = tk.BooleanVar(value=value.get("output_noise_reduce", False))
        self.formant_shift = tk.DoubleVar(value=value.get("formant_shift", 0.0))
        self.vq_neighbors = tk.IntVar(value=value.get("vq_neighbors", 0))
        f0_method = value.get("f0_method", "rmvpe")
        self.f0_method = tk.StringVar(value=f0_method if f0_method in {"rmvpe", "swift", "pm"} else "rmvpe")
        self.block_time = tk.DoubleVar(value=value.get("block_time", 0.25))
        self.crossfade_time = tk.DoubleVar(
            value=value.get("crossfade_time", 0.05)
        )
        self.extra_time = tk.DoubleVar(value=value.get("extra_time", 2.5))
        self.passthrough = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")
        self.latency = tk.StringVar(value="Estimated latency: 0 ms")
        self.infer_time = tk.StringVar(value="Processing: 0 ms (Model: 0 ms)")

    def _build(self):
        root = ttk.Frame(self.root, padding=12)
        root.pack(fill="both", expand=True)
        model = ttk.LabelFrame(root, text="Model", padding=10)
        model.pack(fill="x", pady=(0, 8))
        ttk.Label(model, text="Voice model").grid(row=0, column=0, sticky="w")
        ttk.Entry(model, textvariable=self.model_path).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(model, text="Browse", command=self._browse_model).grid(row=0, column=2)
        ttk.Label(model, text="Feature index").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(model, textvariable=self.index_path).grid(
            row=1, column=1, sticky="ew", padx=8, pady=(8, 0)
        )
        ttk.Button(model, text="Browse", command=self._browse_index).grid(
            row=1, column=2, pady=(8, 0)
        )
        ttk.Label(model, text="Embedder").grid(row=2, column=0, sticky="w", pady=(8, 0))
        embedders = ttk.Frame(model)
        embedders.grid(row=2, column=1, columnspan=2, sticky="w", padx=8, pady=(8, 0))
        for embedder in ("contentvec", "spin-v2"):
            ttk.Radiobutton(
                embedders,
                text=embedder,
                variable=self.embedder_model,
                value=embedder,
            ).pack(side="left", padx=(0, 10))
        ttk.Label(model, text="Flow vocoder (optional)").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(model, textvariable=self.rectified_vocoder_path).grid(
            row=3, column=1, sticky="ew", padx=8, pady=(8, 0)
        )
        ttk.Button(model, text="Browse", command=self._browse_vocoder).grid(
            row=3, column=2, pady=(8, 0)
        )
        ttk.Label(model, text="Flow steps").grid(row=4, column=0, sticky="w", pady=(8, 0))
        flow_settings = ttk.Frame(model)
        flow_settings.grid(row=4, column=1, columnspan=2, sticky="w", padx=8, pady=(8, 0))
        ttk.Spinbox(flow_settings, from_=0, to=1000, textvariable=self.rectified_steps, width=6).pack(side="left")
        ttk.Label(flow_settings, text="0 uses model settings; fewer steps process faster").pack(side="left", padx=8)
        ttk.Checkbutton(
            model,
            text="Flow window: generate only the newest audio plus 0.5 s of context (faster, same output)",
            variable=self.rectified_flow_window,
        ).grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))
        model.columnconfigure(1, weight=1)
        devices = ttk.LabelFrame(root, text="Audio devices", padding=10)
        devices.pack(fill="x", pady=(0, 8))
        ttk.Label(devices, text="Input").grid(row=0, column=0, sticky="w")
        self.input_combo = ttk.Combobox(
            devices, textvariable=self.input_device, state="readonly"
        )
        self.input_combo.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(devices, text="Reload", command=self._load_devices).grid(row=0, column=2)
        ttk.Label(devices, text="Output").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.output_combo = ttk.Combobox(
            devices, textvariable=self.output_device, state="readonly"
        )
        self.output_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Label(devices, text="Monitor").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.monitor_combo = ttk.Combobox(
            devices, textvariable=self.monitor_device, state="readonly"
        )
        self.monitor_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        options = ttk.Frame(devices)
        options.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            options,
            text="WASAPI exclusive",
            variable=self.wasapi_exclusive,
        ).pack(side="left")
        ttk.Checkbutton(
            options,
            text="Show MME / DirectSound / WDM-KS devices",
            variable=self.show_legacy_devices,
            command=self._load_devices,
        ).pack(side="left", padx=(12, 0))
        devices.columnconfigure(1, weight=1)
        settings = ttk.LabelFrame(root, text="Conversion", padding=10)
        settings.pack(fill="both", expand=True, pady=(0, 8))
        self._scale(settings, "Pitch", self.pitch, -24, 24, 0)
        self._scale(settings, "Speaker ID", self.speaker_id, 0, 255, 1)
        self._scale(settings, "Index rate", self.index_rate, 0, 1, 2, 0.01)
        self._scale(settings, "RMS mix", self.rms_mix_rate, 0, 1, 3, 0.01)
        self._scale(settings, "Gate threshold", self.threshold, -60, 0, 4)
        self._scale(settings, "Input gain (dB)", self.input_gain_db, -24, 24, 5, 0.5)
        self._scale(settings, "Output gain (dB)", self.output_gain_db, -24, 24, 6, 0.5)
        self._scale(settings, "Monitor gain (dB)", self.monitor_gain_db, -24, 24, 7, 0.5)
        ttk.Label(settings, text="Pitch extraction").grid(row=8, column=0, sticky="w")
        pitch_methods = ttk.Frame(settings)
        pitch_methods.grid(row=8, column=1, sticky="w", pady=4)
        for label, method in (
            ("rmvpe", "rmvpe"),
            ("swift", "swift"),
            ("parselmouth", "pm"),
        ):
            ttk.Radiobutton(
                pitch_methods,
                text=label,
                variable=self.f0_method,
                value=method,
            ).pack(side="left", padx=(0, 10))
        self._scale(settings, "Block time", self.block_time, 0.02, 1.5, 9, 0.01)
        self._scale(
            settings, "Crossfade", self.crossfade_time, 0.01, 0.15, 10, 0.01
        )
        self._scale(settings, "Extra context", self.extra_time, 0.05, 5.0, 11, 0.01)
        self._scale(settings, "Formant shift (Beatrice)", self.formant_shift, -2.0, 2.0, 12, 0.5)
        self._scale(settings, "VQ neighbors (Beatrice)", self.vq_neighbors, 0, 8, 13)
        ttk.Label(
            settings,
            text=(
                "Beatrice models (the beatrice_paraphernalia_*.toml file) run on the CPU in FP32 "
                "with 37.5 ms of model latency; use a block time of 0.02-0.05 s. Index rate, "
                "embedder, pitch extraction, RMS mix, crossfade and extra context apply to RVC "
                "and Rectified Flow models."
            ),
            wraplength=700,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(4, 0))
        toggles = ttk.Frame(settings)
        toggles.grid(row=15, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            toggles,
            text="Input noise reduction",
            variable=self.input_noise_reduce,
        ).pack(side="left")
        ttk.Checkbutton(
            toggles,
            text="Output noise reduction",
            variable=self.output_noise_reduce,
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            toggles,
            text="Passthrough",
            variable=self.passthrough,
        ).pack(side="left", padx=(12, 0))
        settings.columnconfigure(1, weight=1)
        actions = ttk.Frame(root)
        actions.pack(fill="x")
        self.start_button = ttk.Button(actions, text="Start conversion", command=self._start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(8, 0))
        ttk.Label(actions, textvariable=self.latency).pack(side="left", padx=(18, 0))
        ttk.Label(actions, textvariable=self.infer_time).pack(side="left", padx=(18, 0))
        ttk.Label(root, textvariable=self.status, wraplength=720).pack(
            fill="x", pady=(8, 0)
        )

    def _scale(self, parent, label, variable, minimum, maximum, row, resolution=1):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        scale = tk.Scale(
            parent,
            variable=variable,
            from_=minimum,
            to=maximum,
            resolution=resolution,
            orient="horizontal",
            showvalue=True,
            highlightthickness=0,
        )
        scale.grid(row=row, column=1, sticky="ew")

    def _browse_model(self):
        path = filedialog.askopenfilename(
            initialdir=ROOT / "logs",
            filetypes=(
                ("RVC / Rectified Flow / Beatrice model", "*.pth *.toml"),
                ("All files", "*.*"),
            ),
        )
        if path:
            self.model_path.set(path)

    def _browse_vocoder(self):
        path = filedialog.askopenfilename(
            initialdir=ROOT / "rvc" / "models" / "pretraineds" / "rectified",
            filetypes=(("Flow vocoder", "*.pth *.ckpt *.onnx"), ("All files", "*.*")),
        )
        if path:
            self.rectified_vocoder_path.set(path)

    def _browse_index(self):
        path = filedialog.askopenfilename(
            initialdir=ROOT / "logs",
            filetypes=(("FAISS index", "*.index"), ("All files", "*.*")),
        )
        if path:
            self.index_path.set(path)

    def _load_devices(self):
        try:
            sd._terminate()
            sd._initialize()
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
            show_legacy = self.show_legacy_devices.get()
            self.input_devices = {}
            self.output_devices = {}
            self.device_names = {}
            for index, device in enumerate(devices):
                api_name = hostapis[device["hostapi"]]["name"]
                if not show_legacy and api_name not in MAIN_HOST_APIS:
                    continue
                self.device_names[index] = device["name"]
                if device["max_input_channels"] > 0:
                    self.input_devices[friendly_device_label(api_name, device["name"], "input")] = index
                if device["max_output_channels"] > 0:
                    self.output_devices[friendly_device_label(api_name, device["name"], "output")] = index
            self.input_devices = dict(sorted(self.input_devices.items(), key=lambda item: device_sort_key(item[0])))
            self.output_devices = dict(sorted(self.output_devices.items(), key=lambda item: device_sort_key(item[0])))
            defaults = self._default_devices(hostapis)
            self.input_combo["values"] = list(self.input_devices)
            self.output_combo["values"] = list(self.output_devices)
            self.monitor_combo["values"] = [MONITOR_DISABLED, *self.output_devices]
            self.input_device.set(self._normalize_choice(self.input_device.get(), self.input_devices, defaults[0]))
            self.output_device.set(self._normalize_choice(self.output_device.get(), self.output_devices, defaults[1]))
            monitor = self.monitor_device.get()
            if monitor != MONITOR_DISABLED:
                monitor = self._normalize_choice(monitor, self.output_devices, None, MONITOR_DISABLED)
            self.monitor_device.set(monitor)
        except Exception as error:
            self.status.set(f"Audio device error: {error}")

    @staticmethod
    def _default_devices(hostapis):
        for hostapi in hostapis:
            if hostapi["name"] == "Windows WASAPI":
                return hostapi["default_input_device"], hostapi["default_output_device"]
        return sd.default.device[0], sd.default.device[1]

    def _normalize_choice(self, saved, choices, default_index, fallback=None):
        if saved in choices:
            return saved
        if saved:
            name = saved.split("] ", 1)[-1].removesuffix(" → RVC input").removesuffix(" ← RVC output")
            match = next((label for label, index in choices.items() if self.device_names.get(index) == name), None)
            if match:
                return match
        if fallback is not None:
            return fallback
        return next(
            (label for label, index in choices.items() if index == default_index),
            next(iter(choices), ""),
        )

    def _settings(self):
        if not self.model_path.get().strip():
            raise ValueError("Select a voice model.")
        if self.input_device.get() not in self.input_devices:
            raise ValueError("Select an input device.")
        if self.output_device.get() not in self.output_devices:
            raise ValueError("Select an output device.")
        monitor = self.monitor_device.get()
        return {
            "model_path": self.model_path.get().strip(),
            "index_path": self.index_path.get().strip(),
            "rectified_vocoder_path": self.rectified_vocoder_path.get().strip(),
            "rectified_steps": self.rectified_steps.get(),
            "rectified_flow_window": self.rectified_flow_window.get(),
            "embedder_model": self.embedder_model.get(),
            "input_device": self.input_devices[self.input_device.get()],
            "output_device": self.output_devices[self.output_device.get()],
            "monitor_device": self.output_devices.get(monitor),
            "input_device_label": self.input_device.get(),
            "output_device_label": self.output_device.get(),
            "monitor_device_label": monitor,
            "show_legacy_devices": self.show_legacy_devices.get(),
            "wasapi_exclusive": self.wasapi_exclusive.get(),
            "pitch": self.pitch.get(),
            "speaker_id": self.speaker_id.get(),
            "index_rate": self.index_rate.get(),
            "rms_mix_rate": self.rms_mix_rate.get(),
            "threshold": self.threshold.get(),
            "input_gain_db": self.input_gain_db.get(),
            "output_gain_db": self.output_gain_db.get(),
            "monitor_gain_db": self.monitor_gain_db.get(),
            "input_noise_reduce": self.input_noise_reduce.get(),
            "output_noise_reduce": self.output_noise_reduce.get(),
            "formant_shift": self.formant_shift.get(),
            "vq_neighbors": self.vq_neighbors.get(),
            "f0_method": self.f0_method.get(),
            "block_time": self.block_time.get(),
            "crossfade_time": self.crossfade_time.get(),
            "extra_time": self.extra_time.get(),
            "passthrough": self.passthrough.get(),
        }

    def _save_config(self, settings):
        saved = dict(settings)
        saved["input_device"] = saved.pop("input_device_label")
        saved["output_device"] = saved.pop("output_device_label")
        saved["monitor_device"] = saved.pop("monitor_device_label")
        saved.pop("passthrough", None)
        CONFIG_PATH.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _start(self):
        try:
            settings = self._settings()
            self.status.set("Loading model and starting audio streams...")
            self.root.update_idletasks()
            self.engine.start(settings)
            self._save_config(settings)
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            if self.engine.beatrice is not None:
                status = (
                    f"Running Beatrice model {self.engine.beatrice.model.name} "
                    f"on the CPU in FP32 at {self.engine.sample_rate} Hz"
                )
            else:
                status = (
                    f"Running {self.engine.rvc.vocoder} with "
                    f"{self.engine.rvc.embedder_name} in FP32 at "
                    f"{self.engine.sample_rate} Hz"
                )
            if self.engine.shared_fallbacks:
                status += "; WASAPI exclusive unavailable, shared mode for " + ", ".join(
                    self.engine.shared_fallbacks
                )
            self.status.set(status)
            self.latency.set(
                f"Estimated latency: {self.engine.algorithm_latency_ms} ms"
            )
        except Exception as error:
            self.engine.stop()
            self.status.set(f"Start failed: {error}")
            messagebox.showerror("Applio Real-Time", str(error))

    def _stop(self):
        self.engine.stop()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status.set("Stopped")

    def _hot_update(self, *args):
        if not self.engine.running:
            return
        try:
            self.engine.update_pitch(self.pitch.get())
            self.engine.update_index_rate(self.index_rate.get())
            self.engine.update_beatrice(
                self.speaker_id.get(),
                self.formant_shift.get(),
                self.vq_neighbors.get(),
            )
            for key, variable in (
                ("rms_mix_rate", self.rms_mix_rate),
                ("threshold", self.threshold),
                ("f0_method", self.f0_method),
                ("passthrough", self.passthrough),
                ("input_gain_db", self.input_gain_db),
                ("output_gain_db", self.output_gain_db),
                ("monitor_gain_db", self.monitor_gain_db),
                ("input_noise_reduce", self.input_noise_reduce),
                ("output_noise_reduce", self.output_noise_reduce),
            ):
                self.engine.settings[key] = variable.get()
        except (tk.TclError, ValueError) as error:
            self.status.set(str(error))

    def _poll(self):
        self.infer_time.set(
            f"Processing: {self.engine.last_block_ms} ms "
            f"(Model: {self.engine.last_infer_ms} ms)"
        )
        self.latency.set(
            f"Estimated latency: {self.engine.algorithm_latency_ms} ms"
        )
        try:
            while True:
                error = self.error_queue.get_nowait()
                self.status.set(error.strip().splitlines()[-1])
                if not self.engine.running:
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _close(self):
        self.engine.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    RealtimeGUI().run()
