import ctypes
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import sv_ttk
except ImportError:
    sv_ttk = None

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("SD_ENABLE_ASIO", "1")

import librosa
import numpy as np
import sounddevice as sd
import torch
import torch.nn.functional as F
from noisereduce.torchgate import TorchGate
from noisereduce.torchgate.utils import amp_to_db
from torchaudio.transforms import Resample

from beatrice.inference import OUT_SAMPLE_RATE, find_paraphernalia, is_beatrice_checkpoint
from beatrice.realtime import BeatriceRealtime
from rvc.infer.realtime import RealTimeRVC
from shared.platform import migrate_models_folder


ROOT = Path(__file__).resolve().parent
migrate_models_folder(ROOT)
CONFIG_PATH = ROOT / "assets" / "realtime_config.json"
MONITOR_DISABLED = "Disabled"
MAIN_HOST_APIS = ("ASIO", "Windows WASAPI")
EMBEDDERS = ("contentvec", "spin-v2")
PITCH_METHODS = {"rmvpe": "rmvpe", "swift": "swift", "parselmouth": "pm"}
METER_FLOOR_DB = -60.0


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


class AudioFrameFifo:
    def __init__(self, channels=1, max_frames=None):
        self.channels = int(channels)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.prefill = 0
        self._chunks = deque()
        self._frames = 0
        self._starved = True
        self._lock = threading.Lock()

    def write(self, data):
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if not array.shape[0]:
            return
        with self._lock:
            if self._starved and self.prefill > 0:
                self._chunks.append(np.zeros((self.prefill, array.shape[1]), dtype=np.float32))
                self._frames += self.prefill
            self._starved = False
            self._chunks.append(array.copy())
            self._frames += array.shape[0]
            if self.max_frames is not None and self._frames > self.max_frames:
                self._discard_locked(self._frames - self.max_frames)

    def read(self, frames, exact=False):
        with self._lock:
            if exact and self._frames < int(frames):
                return None
            take = min(int(frames), self._frames)
            if take < int(frames):
                self._starved = True
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


def asio_channel_choices(count, prefix, mono):
    choices = {}
    if mono:
        for channel in range(count):
            choices[f"{prefix} {channel + 1}"] = [channel]
    for first in range(0, count - 1, 2):
        choices[f"{prefix} {first + 1}+{first + 2}"] = [first, first + 1]
    if count % 2:
        choices[f"{prefix} {count}"] = [count - 1]
    return choices


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
    parts = re.split(r"(\d+)", lower_label)
    return priority, [int(part) if index % 2 else part for index, part in enumerate(parts)]


class AudioEngine:
    def __init__(self, error_queue):
        self.error_queue = error_queue
        self.input_stream = None
        self.output_stream = None
        self.monitor_stream = None
        self.asio_stream = None
        self.output_fifo = None
        self.monitor_queue = None
        self.input_fifo = None
        self.worker = None
        self.worker_stop = None
        self.worker_wakeup = None
        self.asio_device = None
        self.asio_main_channels = 0
        self.asio_callback_frames = 0
        self.rvc = None
        self.beatrice = None
        self.running = False
        self.settings = {}
        self.sample_rate = 0
        self.last_infer_ms = 0
        self.last_block_ms = 0
        self.algorithm_latency_ms = 0
        self.base_latency_ms = 0.0
        self.input_level = 0.0
        self.output_level = 0.0
        self.reported_statuses = set()
        self.shared_fallbacks = []

    def start(self, settings):
        self.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.settings = settings
        beatrice_path = find_paraphernalia(settings["model_path"])
        if beatrice_path is None and is_beatrice_checkpoint(settings["model_path"]):
            raise ValueError(
                "This is a Beatrice training checkpoint, not a model. Select the paraphernalia_* model folder "
                "or its .toml file from the same training run."
            )
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
        self.input_selectors = settings.get("input_selectors")
        self.output_selectors = settings.get("output_selectors")
        self.monitor_selectors = settings.get("monitor_selectors")
        if (self.monitor_device, self.monitor_selectors) == (self.output_device, self.output_selectors):
            self.monitor_device = None
            self.monitor_selectors = None
        asio_devices = {
            device
            for device, selectors in (
                (self.input_device, self.input_selectors),
                (self.output_device, self.output_selectors),
                (self.monitor_device, self.monitor_selectors),
            )
            if device is not None and selectors
        }
        if len(asio_devices) > 1:
            raise ValueError(
                "Only one ASIO driver can be open at a time. Pick channels of the same ASIO driver "
                "for every ASIO device, or use WASAPI devices for the rest."
            )
        self.asio_device = next(iter(asio_devices), None)
        self.channels = self._channels(self.input_device, "max_input_channels", self.input_selectors)
        self.output_channels = self._channels(self.output_device, "max_output_channels", self.output_selectors)
        self.monitor_channels = (
            self._channels(self.monitor_device, "max_output_channels", self.monitor_selectors)
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
        self.asio_callback_frames = 0
        self.prefill_margin = max(self.block_frame // 4, self.zc)
        self.output_fifo = AudioFrameFifo(1, max_frames=self.block_frame * 4)
        if self.output_selectors or self.input_selectors:
            self._resize_asio_fifo(self.output_fifo)
        else:
            self.output_fifo.prefill = min(self.block_frame, self.sola_buffer_frame)
        if not self.output_selectors:
            self.output_stream = sd.OutputStream(
                callback=self._fifo_output_callback,
                blocksize=0,
                latency="low",
                samplerate=self.sample_rate,
                channels=self.output_channels,
                device=self.output_device,
                dtype="float32",
                extra_settings=self.output_extra,
            )
        if not self.input_selectors:
            self.input_stream = sd.InputStream(
                callback=self._input_callback,
                blocksize=0,
                latency="low",
                samplerate=self.sample_rate,
                channels=self.channels,
                device=self.input_device,
                dtype="float32",
                extra_settings=self.input_extra,
            )
        self.running = True
        self._start_worker()
        if self.output_stream is not None:
            self.output_stream.start()
        self._start_asio_stream()
        if self.input_stream is not None:
            self.input_stream.start()
        if not self.monitor_selectors:
            self._start_monitor_stream()
        input_latency = self._stream_latency(self.input_stream or self.asio_stream, 0)
        if self.output_stream is not None:
            output_latency = self.output_stream.latency
        else:
            output_latency = self._stream_latency(self.asio_stream, 1)
        if self.beatrice is not None:
            model_latency = self.beatrice.latency_seconds
        else:
            model_latency = (self.crossfade_frame + self.sola_search_frame) / self.sample_rate
        self.base_latency_ms = (
            input_latency + output_latency + self.block_frame / self.sample_rate + model_latency
        ) * 1000
        self._refresh_latency()

    def _channels(self, device, key, selectors=None):
        if selectors:
            return len(selectors)
        return max(1, min(int(sd.query_devices(device)[key]), 2))

    @staticmethod
    def _stream_latency(stream, index):
        latency = stream.latency
        return latency[index] if isinstance(latency, tuple) else latency

    def _routing_samplerate(self, model_rate):
        exclusive = self.settings.get("wasapi_exclusive", False)

        def check(checker, device, channels, rate, selectors=None):
            if selectors:
                settings = sd.AsioSettings(channel_selectors=selectors)
                checker(device=device, channels=channels, dtype="float32",
                        samplerate=rate, extra_settings=settings)
                return settings, False
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
                    sd.check_input_settings, self.input_device, self.channels, rate, self.input_selectors
                )
                self.output_extra, output_shared = check(
                    sd.check_output_settings, self.output_device, self.output_channels, rate,
                    self.output_selectors,
                )
                self.monitor_extra, monitor_shared = None, False
                if self.monitor_device is not None:
                    self.monitor_extra, monitor_shared = check(
                        sd.check_output_settings, self.monitor_device, self.monitor_channels, rate,
                        self.monitor_selectors,
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
        if self.rvc is None:
            self.crossfade_frame = self.zc
        else:
            self.crossfade_frame = min(
                int(np.round(settings["crossfade_time"] * rate / self.zc)) * self.zc, 2 * self.zc
            )
        self.sola_buffer_frame = self.crossfade_frame
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
            if self.input_selectors:
                self._resize_asio_fifo(self.monitor_queue)
            else:
                self.monitor_queue.prefill = min(self.block_frame, self.sola_buffer_frame)
            self.monitor_stream = sd.OutputStream(
                device=self.monitor_device,
                callback=self._monitor_callback,
                blocksize=0,
                latency="low",
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

    def _resize_asio_fifo(self, fifo):
        frames = self.asio_callback_frames
        fifo.prefill = 2 * frames + max(0, frames - self.block_frame) + self.prefill_margin
        fifo.max_frames = 2 * (self.block_frame + fifo.prefill)

    def _track_asio_frames(self, frames):
        if frames <= self.asio_callback_frames:
            return
        self.asio_callback_frames = frames
        output_fifo = self.output_fifo if self.output_selectors or self.input_selectors else None
        monitor_queue = self.monitor_queue if self.monitor_selectors or self.input_selectors else None
        for fifo in (output_fifo, monitor_queue):
            if fifo is not None:
                self._resize_asio_fifo(fifo)
        input_fifo = self.input_fifo
        if input_fifo is not None and self.input_selectors:
            input_fifo.max_frames = 4 * self.block_frame + 2 * frames

    def _start_asio_stream(self):
        if self.asio_device is None:
            return
        output_selectors = list(self.output_selectors or [])
        self.asio_main_channels = len(output_selectors)
        if self.monitor_selectors:
            self.monitor_queue = AudioFrameFifo(1)
            self._resize_asio_fifo(self.monitor_queue)
            output_selectors += self.monitor_selectors
        common = dict(samplerate=self.sample_rate, dtype="float32", blocksize=0)
        if self.input_selectors:
            input_settings = sd.AsioSettings(channel_selectors=self.input_selectors)
            if output_selectors:
                self.asio_stream = sd.Stream(
                    callback=self._asio_duplex_callback,
                    device=(self.asio_device, self.asio_device),
                    channels=(self.channels, len(output_selectors)),
                    extra_settings=(input_settings, sd.AsioSettings(channel_selectors=output_selectors)),
                    **common,
                )
            else:
                self.asio_stream = sd.InputStream(
                    callback=self._asio_input_callback,
                    device=self.asio_device,
                    channels=self.channels,
                    extra_settings=input_settings,
                    **common,
                )
        else:
            self.asio_stream = sd.OutputStream(
                callback=self._asio_output_callback,
                device=self.asio_device,
                channels=len(output_selectors),
                extra_settings=sd.AsioSettings(channel_selectors=output_selectors),
                **common,
            )
        self.asio_stream.start()

    def _start_worker(self):
        self.input_fifo = AudioFrameFifo(self.channels, max_frames=self.block_frame * 4)
        self.worker_stop = threading.Event()
        self.worker_wakeup = threading.Event()
        self.worker = threading.Thread(
            target=self._worker,
            args=(self.input_fifo, self.worker_wakeup, self.worker_stop),
            name="realtime-inference",
            daemon=True,
        )
        self.worker.start()

    def _worker(self, fifo, wakeup, stop):
        while not stop.is_set():
            wakeup.wait(0.05)
            wakeup.clear()
            while not stop.is_set() and self.running:
                block = fifo.read(self.block_frame, exact=True)
                if block is None:
                    break
                try:
                    self._handle_block(block)
                except Exception:
                    self.running = False
                    self.error_queue.put_nowait(traceback.format_exc())
                    return

    def _refresh_latency(self):
        lookahead_ms = 0
        if self.rvc is not None and not self.settings["passthrough"]:
            lookahead_ms = 10 * self.rvc.pitch_lookahead_frames(self.settings["f0_method"])
        noise_reduction_ms = 0
        if self.settings["input_noise_reduce"]:
            noise_reduction_ms = 1000 * self.sola_buffer_frame / self.sample_rate
        prefill_ms = 0
        output_fifo = self.output_fifo
        if output_fifo is not None and self.sample_rate:
            prefill_ms = 1000 * output_fifo.prefill / self.sample_rate
        self.algorithm_latency_ms = round(
            self.base_latency_ms + lookahead_ms + noise_reduction_ms + prefill_ms + self.last_block_ms
        )

    def stop(self):
        self.running = False
        self.input_level = 0.0
        self.output_level = 0.0
        if self.worker_stop is not None:
            self.worker_stop.set()
            self.worker_wakeup.set()
        for name in ("asio_stream", "input_stream", "output_stream", "monitor_stream"):
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
        if self.worker is not None and self.worker is not threading.current_thread():
            self.worker.join(timeout=5)
        self.worker = None
        self.worker_stop = None
        self.worker_wakeup = None
        self.output_fifo = None
        self.monitor_queue = None
        self.input_fifo = None

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
                else:
                    infer_wav = infer_wav.clone()
            if settings["output_noise_reduce"] and not passthrough:
                self.output_buffer[: -self.block_frame] = self.output_buffer[self.block_frame :].clone()
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]
                infer_wav = self.tg(infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)).squeeze(0)
            if settings["rms_mix_rate"] < 1 and not passthrough:
                infer_wav = self._mix_volume(infer_wav, source)
            output_block = self._apply_sola(infer_wav)
        gain = db_to_linear(settings["output_gain_db"])
        output = torch.clamp(output_block * gain, -1.0, 1.0).cpu().numpy().astype(np.float32)
        self.output_level = float(np.abs(output).max(initial=0.0))
        self.last_infer_ms = round(infer_seconds * 1000)
        self.last_block_ms = round((time.perf_counter() - started) * 1000)
        self._refresh_latency()
        return output

    def _report_status(self, status):
        text = str(status)
        if text not in self.reported_statuses:
            self.reported_statuses.add(text)
            self.error_queue.put_nowait(text)

    def _handle_block(self, indata):
        output = self._process(indata)
        output_fifo = self.output_fifo
        if output_fifo is not None:
            output_fifo.write(output)
        monitor_queue = self.monitor_queue
        if monitor_queue is not None:
            monitor_queue.write(
                np.clip(output * np.float32(db_to_linear(self.settings["monitor_gain_db"])), -1.0, 1.0)
            )

    def _input_callback(self, indata, frames, times, status):
        if not self.running:
            return
        try:
            if status:
                self._report_status(status)
            fifo = self.input_fifo
            wakeup = self.worker_wakeup
            if fifo is not None and wakeup is not None:
                fifo.write(indata)
                wakeup.set()
        except Exception:
            self.running = False
            self.error_queue.put_nowait(traceback.format_exc())
            raise sd.CallbackAbort

    def _asio_input_callback(self, indata, frames, times, status):
        if not self.running:
            return
        self._track_asio_frames(frames)
        self._input_callback(indata, frames, times, status)

    def _asio_output_callback(self, outdata, frames, times, status):
        outdata.fill(0)
        self._track_asio_frames(frames)
        output_fifo = self.output_fifo if self.output_selectors else None
        monitor_queue = self.monitor_queue if self.monitor_selectors else None
        main_channels = self.asio_main_channels
        block = output_fifo.read(frames) if output_fifo is not None else None
        if block is not None:
            outdata[: block.shape[0], :main_channels] = block[:, :1]
        block = monitor_queue.read(frames) if monitor_queue is not None else None
        if block is not None:
            outdata[: block.shape[0], main_channels:] = block[:, :1]

    def _asio_duplex_callback(self, indata, outdata, frames, times, status):
        self._asio_output_callback(outdata, frames, times, status)
        self._asio_input_callback(indata, frames, times, status)

    @staticmethod
    def _play_fifo(outdata, fifo, frames):
        outdata.fill(0)
        block = fifo.read(frames) if fifo is not None else None
        if block is not None:
            outdata[: block.shape[0]] = block[:, :1]

    def _fifo_output_callback(self, outdata, frames, times, status):
        self._play_fifo(outdata, self.output_fifo, frames)

    def _monitor_callback(self, outdata, frames, times, status):
        self._play_fifo(outdata, self.monitor_queue, frames)


def system_theme():
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
        ) as key:
            return "light" if winreg.QueryValueEx(key, "AppsUseLightTheme")[0] else "dark"
    except (ImportError, OSError):
        return "light"


def level_percent(level):
    if level <= 0:
        return 0.0
    db = 20.0 * np.log10(level)
    return max(0.0, min(100.0, 100.0 * (db - METER_FLOOR_DB) / -METER_FLOOR_DB))


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.window = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None):
        if self.window is not None:
            return
        self.window = tk.Toplevel(self.widget)
        self.window.wm_overrideredirect(True)
        self.window.wm_geometry(
            f"+{self.widget.winfo_rootx() + 12}+{self.widget.winfo_rooty() + self.widget.winfo_height() + 4}"
        )
        frame = ttk.Frame(self.window, padding=1, style="Tooltip.TFrame")
        frame.pack()
        ttk.Label(frame, text=self.text, wraplength=340, padding=(8, 5)).pack()

    def _hide(self, _event=None):
        if self.window is not None:
            self.window.destroy()
            self.window = None


class RealtimeGUI:
    def __init__(self):
        if sys.platform == "win32":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass
        self.root = tk.Tk()
        self.root.title("Applio Real-Time")
        try:
            self.root.iconbitmap(str(ROOT / "assets" / "red_panda_favicon.ico"))
        except tk.TclError:
            pass
        self.error_queue = queue.Queue()
        self.engine = AudioEngine(self.error_queue)
        self.input_devices = {}
        self.output_devices = {}
        self.input_selectors = {}
        self.output_selectors = {}
        self.device_names = {}
        self.rvc_widgets = set()
        self.locked_widgets = set()
        self.meter_values = {"input": 0.0, "output": 0.0, "monitor": 0.0}
        self.saved = self._load_config()
        self._make_variables()
        self._apply_theme(self.saved.get("theme") or system_theme())
        self._build()
        self._load_devices()
        self._refresh_states()
        for variable in (
            self.pitch, self.index_rate, self.rms_mix_rate, self.threshold, self.f0_method,
            self.passthrough, self.speaker_id, self.formant_shift, self.vq_neighbors,
            self.input_gain_db, self.output_gain_db, self.monitor_gain_db,
            self.input_noise_reduce, self.output_noise_reduce,
        ):
            variable.trace_add("write", self._hot_update)
        self.model_path.trace_add("write", self._refresh_states)
        self.root.update_idletasks()
        self.root.minsize(self.root.winfo_reqwidth(), self.root.winfo_reqheight())
        self.root.after(100, self._poll)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _load_config(self):
        try:
            return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _write_config(self, data):
        CONFIG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _make_variables(self):
        value = self.saved
        self.model_path = tk.StringVar(value=value.get("model_path", ""))
        self.index_path = tk.StringVar(value=value.get("index_path", ""))
        self.rectified_vocoder_path = tk.StringVar(value=value.get("rectified_vocoder_path", ""))
        self.rectified_steps = tk.IntVar(value=value.get("rectified_steps", 0))
        self.rectified_flow_window = tk.BooleanVar(value=value.get("rectified_flow_window", True))
        self.show_flow_options = tk.BooleanVar(
            value=value.get(
                "show_flow_options",
                bool(value.get("rectified_vocoder_path") or value.get("rectified_steps")),
            )
        )
        embedder = value.get("embedder_model", "contentvec")
        self.embedder_model = tk.StringVar(value=embedder if embedder in EMBEDDERS else "contentvec")
        self.input_device = tk.StringVar(value=value.get("input_device", ""))
        self.output_device = tk.StringVar(value=value.get("output_device", ""))
        self.monitor_device = tk.StringVar(value=value.get("monitor_device", MONITOR_DISABLED))
        self.show_legacy_devices = tk.BooleanVar(value=value.get("show_legacy_devices", False))
        self.wasapi_exclusive = tk.BooleanVar(value=value.get("wasapi_exclusive", False))
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
        self.f0_method = tk.StringVar(value=f0_method if f0_method in PITCH_METHODS.values() else "rmvpe")
        self.block_times = {
            "rvc": value.get("block_time", 0.3),
            "beatrice": value.get("beatrice_block_time", 0.02),
        }
        self.block_kind = "rvc"
        self.block_time = tk.DoubleVar(value=self.block_times["rvc"])
        self.crossfade_time = tk.DoubleVar(value=min(value.get("crossfade_time", 0.01), 0.02))
        self.extra_time = tk.DoubleVar(value=value.get("extra_time", 3.0))
        self.passthrough = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")
        self.stats = tk.StringVar()
        self.sample_rate_text = tk.StringVar()

    def _apply_theme(self, theme):
        self.theme = "dark" if theme == "dark" else "light"
        style = ttk.Style(self.root)
        if sv_ttk is not None:
            sv_ttk.set_theme(self.theme, self.root)
        else:
            if "vista" in style.theme_names():
                style.theme_use("vista")
            style.layout("Accent.TButton", style.layout("TButton"))
            style.layout("Switch.TCheckbutton", style.layout("TCheckbutton"))
        muted = "#9a9a9a" if self.theme == "dark" else "#6b6b6b"
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 14))
        style.configure("Muted.TLabel", foreground=muted)
        style.configure("Value.TLabel", foreground=muted, anchor="e")
        style.configure("Tooltip.TFrame", background=muted)

    def _toggle_theme(self):
        self._apply_theme("light" if self.theme == "dark" else "dark")
        self.theme_button.configure(text="Light mode" if self.theme == "dark" else "Dark mode")
        data = self._load_config()
        data["theme"] = self.theme
        try:
            self._write_config(data)
        except OSError:
            pass

    def _build(self):
        root = ttk.Frame(self.root, padding=(16, 12, 16, 14))
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1, uniform="half")
        root.columnconfigure(1, weight=1, uniform="half")
        root.rowconfigure(3, weight=1)
        header = ttk.Frame(root)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        ttk.Label(header, text="Real-Time Voice Conversion", style="Title.TLabel").pack(side="left")
        self.theme_button = ttk.Button(
            header,
            text="Light mode" if self.theme == "dark" else "Dark mode",
            command=self._toggle_theme,
            width=11,
        )
        self.theme_button.pack(side="right")
        if sv_ttk is None:
            self.theme_button.pack_forget()
        self._build_model(root).grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        self._build_devices(root).grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(0, 10))
        self._build_voice(root).grid(row=3, column=0, sticky="nsew", padx=(0, 5), pady=(0, 12))
        self._build_performance(root).grid(row=3, column=1, sticky="nsew", padx=(5, 0), pady=(0, 12))
        self._build_actions(root).grid(row=4, column=0, columnspan=2, sticky="ew")

    def _card(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=(12, 6, 12, 10))
        frame.columnconfigure(1, weight=1)
        return frame

    def _build_model(self, parent):
        frame = self._card(parent, "Model")
        self.locked_widgets.update(self._path_row(frame, 0, "Voice model", self.model_path, self._browse_model))
        index_row = self._path_row(frame, 1, "Feature index", self.index_path, self._browse_index)
        self.rvc_widgets.update(index_row)
        self.locked_widgets.update(index_row)
        embedder_label = ttk.Label(frame, text="Embedder")
        embedder_label.grid(row=2, column=0, sticky="w", padx=(0, 12), pady=3)
        options = ttk.Frame(frame)
        options.grid(row=2, column=1, columnspan=2, sticky="ew", pady=3)
        options.columnconfigure(3, weight=1)
        embedder = ttk.Combobox(options, textvariable=self.embedder_model, values=EMBEDDERS,
                                state="readonly", width=12)
        embedder.grid(row=0, column=0, sticky="w")
        self.rvc_widgets.update((embedder_label, embedder))
        self.locked_widgets.add(embedder)
        ttk.Label(options, text="Speaker ID").grid(row=0, column=1, sticky="w", padx=(20, 8))
        ttk.Spinbox(options, from_=0, to=255, textvariable=self.speaker_id, width=5).grid(row=0, column=2, sticky="w")
        self.flow_toggle = ttk.Checkbutton(
            options,
            text="Rectified Flow options",
            variable=self.show_flow_options,
            command=self._refresh_states,
            style="Switch.TCheckbutton",
        )
        self.flow_toggle.grid(row=0, column=3, sticky="e")
        self.flow_rows = self._path_row(frame, 3, "Flow vocoder", self.rectified_vocoder_path, self._browse_vocoder)
        Tooltip(self.flow_rows[1], "Optional vocoder for Rectified Flow models. Leave empty to use the model's own.")
        steps_label = ttk.Label(frame, text="Flow steps")
        steps_label.grid(row=4, column=0, sticky="w", padx=(0, 12), pady=3)
        steps = ttk.Frame(frame)
        steps.grid(row=4, column=1, columnspan=2, sticky="ew", pady=3)
        steps_box = ttk.Spinbox(steps, from_=0, to=1000, textvariable=self.rectified_steps, width=6)
        steps_box.pack(side="left")
        ttk.Label(steps, text="0 = model default, fewer = faster", style="Muted.TLabel").pack(side="left", padx=(8, 0))
        flow_window = ttk.Checkbutton(steps, text="Flow window", variable=self.rectified_flow_window)
        flow_window.pack(side="right")
        Tooltip(flow_window, "Generate only the newest audio plus 0.5 s of context. Faster, same output.")
        self.flow_rows += [steps_label, steps]
        self.locked_widgets.update((*self.flow_rows[:3], steps_box, flow_window))
        beatrice_label = ttk.Label(frame, text="Beatrice")
        beatrice_label.grid(row=5, column=0, sticky="w", padx=(0, 12), pady=3)
        beatrice = ttk.Frame(frame)
        beatrice.grid(row=5, column=1, columnspan=2, sticky="ew", pady=3)
        beatrice.columnconfigure(1, weight=1)
        beatrice.columnconfigure(4, weight=1)
        self._slider(beatrice, 0, "Formant", self.formant_shift, -2.0, 2.0, 0.5)
        self._slider(beatrice, 0, "VQ neighbors", self.vq_neighbors, 0, 8, 1, column=3, padx=(20, 10))
        self.beatrice_rows = [beatrice_label, beatrice]
        return frame

    def _path_row(self, parent, row, label, variable, command):
        text = ttk.Label(parent, text=label)
        text.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=3)
        button = ttk.Button(parent, text="Browse", command=command, width=8)
        button.grid(row=row, column=2, padx=(8, 0), pady=3)
        return [text, entry, button]

    def _slider(self, parent, row, label, variable, minimum, maximum, resolution=1, column=0,
                padx=(0, 12), tooltip=None):
        digits = 0 if resolution >= 1 else len(f"{resolution:g}".split(".")[1])
        widgets = []
        if label is not None:
            text = ttk.Label(parent, text=label)
            text.grid(row=row, column=column, sticky="w", padx=padx, pady=3)
            widgets.append(text)
            if tooltip:
                Tooltip(text, tooltip)

        def moved(raw):
            snapped = round(round(float(raw) / resolution) * resolution, digits)
            if isinstance(variable, tk.IntVar):
                snapped = int(snapped)
            try:
                current = variable.get()
            except tk.TclError:
                current = None
            if current != snapped:
                variable.set(snapped)

        try:
            initial = variable.get()
        except tk.TclError:
            initial = minimum
        scale = ttk.Scale(parent, from_=minimum, to=maximum, value=initial, command=moved)
        scale.grid(row=row, column=column + 1, sticky="ew", pady=3)
        value = ttk.Label(parent, width=5, style="Value.TLabel")
        value.grid(row=row, column=column + 2, sticky="e", padx=(6, 0), pady=3)

        def show(*_):
            try:
                value.configure(text=f"{variable.get():.{digits}f}")
            except tk.TclError:
                pass

        variable.trace_add("write", show)
        show()
        return widgets + [scale, value]

    def _build_devices(self, parent):
        frame = self._card(parent, "Audio")
        frame.columnconfigure(1, weight=3)
        frame.columnconfigure(2, weight=1, minsize=110)
        for column, text in ((1, "Device"), (2, "Gain (dB)"), (4, "Level")):
            ttk.Label(frame, text=text, style="Muted.TLabel").grid(
                row=0, column=column, sticky="w", padx=(0 if column == 1 else 12, 0)
            )
        self.meters = {}
        combos = []
        for row, (label, device, gain, key) in enumerate(
            (
                ("Input", self.input_device, self.input_gain_db, "input"),
                ("Output", self.output_device, self.output_gain_db, "output"),
                ("Monitor", self.monitor_device, self.monitor_gain_db, "monitor"),
            ),
            start=1,
        ):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
            combo = ttk.Combobox(frame, textvariable=device, state="readonly", width=38)
            combo.grid(row=row, column=1, sticky="ew", pady=3, padx=(0, 12))
            self._slider(frame, row, None, gain, -24, 24, 0.5, column=1)
            meter = ttk.Progressbar(frame, maximum=100, length=90)
            meter.grid(row=row, column=4, sticky="ew", padx=(12, 0), pady=3)
            self.meters[key] = meter
            combos.append(combo)
        self.input_combo, self.output_combo, self.monitor_combo = combos
        Tooltip(self.monitor_combo, "Optional second output to hear yourself, e.g. headphones.")
        options = ttk.Frame(frame)
        options.grid(row=4, column=0, columnspan=5, sticky="ew", pady=(8, 0))
        reload_button = ttk.Button(options, text="Reload devices", command=self._load_devices)
        reload_button.pack(side="left")
        exclusive = ttk.Checkbutton(options, text="WASAPI exclusive", variable=self.wasapi_exclusive)
        exclusive.pack(side="left", padx=(16, 0))
        legacy = ttk.Checkbutton(
            options, text="Legacy drivers", variable=self.show_legacy_devices, command=self._load_devices
        )
        legacy.pack(side="left", padx=(16, 0))
        Tooltip(legacy, "Also list MME, DirectSound and WDM-KS devices.")
        ttk.Label(options, textvariable=self.sample_rate_text, style="Muted.TLabel").pack(side="right")
        self.locked_widgets.update((*combos, reload_button, exclusive, legacy))
        return frame

    def _build_voice(self, parent):
        frame = self._card(parent, "Voice")
        self._slider(frame, 0, "Pitch", self.pitch, -24, 24, 1, tooltip="Shift in semitones. +12 is one octave up.")
        self.rvc_widgets.update(self._slider(
            frame, 1, "Index rate", self.index_rate, 0, 1, 0.01,
            tooltip="How strongly the feature index pulls the voice toward the training data.",
        ))
        self.rvc_widgets.update(self._slider(
            frame, 2, "RMS mix", self.rms_mix_rate, 0, 1, 0.01,
            tooltip="0 follows your input loudness, 1 keeps the model's own loudness.",
        ))
        self._slider(
            frame, 3, "Noise gate", self.threshold, -60, 0, 1,
            tooltip="Input quieter than this (dB) is muted. -60 turns the gate off.",
        )
        toggles = ttk.Frame(frame)
        toggles.grid(row=4, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(toggles, text="Input noise reduction", variable=self.input_noise_reduce).pack(side="left")
        ttk.Checkbutton(toggles, text="Output noise reduction", variable=self.output_noise_reduce).pack(
            side="left", padx=(16, 0)
        )
        return frame

    def _build_performance(self, parent):
        frame = self._card(parent, "Performance")
        self.locked_widgets.update(self._slider(
            frame, 0, "Block time", self.block_time, 0.02, 1.5, 0.01,
            tooltip=(
                "Seconds of audio per step. Lower is less latency but more load. Beatrice models keep their "
                "own value; 0.02-0.05 s works best for them."
            ),
        ))
        crossfade = self._slider(frame, 1, "Crossfade", self.crossfade_time, 0.01, 0.02, 0.01)
        extra = self._slider(
            frame, 2, "Extra context", self.extra_time, 0.05, 5.0, 0.01,
            tooltip="Seconds of past audio the model sees. More is smoother but slower.",
        )
        self.rvc_widgets.update(crossfade + extra)
        self.locked_widgets.update(crossfade + extra)
        pitch_label = ttk.Label(frame, text="Pitch extraction")
        pitch_label.grid(row=3, column=0, sticky="w", padx=(0, 12), pady=(6, 3))
        methods = ttk.Frame(frame)
        methods.grid(row=3, column=1, columnspan=2, sticky="w", pady=(6, 3))
        self.rvc_widgets.add(pitch_label)
        for label, method in PITCH_METHODS.items():
            radio = ttk.Radiobutton(methods, text=label, variable=self.f0_method, value=method)
            radio.pack(side="left", padx=(0, 12))
            self.rvc_widgets.add(radio)
        return frame

    def _build_actions(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(2, weight=1)
        self.start_button = ttk.Button(frame, text="Start", style="Accent.TButton", width=12, command=self._toggle)
        self.start_button.grid(row=0, column=0, sticky="w")
        passthrough = ttk.Checkbutton(frame, text="Passthrough", variable=self.passthrough,
                                      style="Switch.TCheckbutton")
        passthrough.grid(row=0, column=1, sticky="w", padx=(16, 0))
        Tooltip(passthrough, "Send the input straight to the output without conversion.")
        ttk.Label(frame, textvariable=self.stats, style="Muted.TLabel").grid(row=0, column=2, sticky="e")
        status = ttk.Label(frame, textvariable=self.status, style="Muted.TLabel")
        status.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        frame.bind("<Configure>", lambda event: status.configure(wraplength=max(200, event.width)))
        return frame

    def _refresh_states(self, *_):
        path = self.model_path.get().strip()
        beatrice = bool(path) and find_paraphernalia(path) is not None
        kind = "beatrice" if beatrice else "rvc"
        if kind != self.block_kind:
            self.block_times[self.block_kind] = self.block_time.get()
            self.block_kind = kind
            self.block_time.set(self.block_times[kind])
        running = self.engine.running
        for widget in self.rvc_widgets | self.locked_widgets:
            disabled = (beatrice and widget in self.rvc_widgets) or (running and widget in self.locked_widgets)
            widget.state(["disabled"] if disabled else ["!disabled"])
        show_flow = not beatrice and self.show_flow_options.get()
        for widget in self.flow_rows:
            widget.grid() if show_flow else widget.grid_remove()
        for widget in self.beatrice_rows:
            widget.grid() if beatrice else widget.grid_remove()
        self.flow_toggle.grid_remove() if beatrice else self.flow_toggle.grid()

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
            initialdir=ROOT / "models" / "pretraineds" / "rectified",
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
            self.input_selectors = {}
            self.output_selectors = {}
            self.device_names = {}
            for index, device in enumerate(devices):
                api_name = hostapis[device["hostapi"]]["name"]
                if not show_legacy and api_name not in MAIN_HOST_APIS:
                    continue
                self.device_names[index] = device["name"]
                if api_name == "ASIO":
                    base = friendly_device_label(api_name, device["name"], "input")
                    for devices_map, selectors_map, count, prefix, mono in (
                        (self.input_devices, self.input_selectors, device["max_input_channels"], "In", True),
                        (self.output_devices, self.output_selectors, device["max_output_channels"], "Out", False),
                    ):
                        for channels, selectors in asio_channel_choices(int(count), prefix, mono).items():
                            label = f"{base}: {channels}"
                            devices_map[label] = index
                            selectors_map[label] = selectors
                    continue
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
            "input_selectors": self.input_selectors.get(self.input_device.get()),
            "output_selectors": self.output_selectors.get(self.output_device.get()),
            "monitor_selectors": self.output_selectors.get(monitor),
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
        for key in ("passthrough", "input_selectors", "output_selectors", "monitor_selectors"):
            saved.pop(key, None)
        self.block_times[self.block_kind] = self.block_time.get()
        saved["block_time"] = self.block_times["rvc"]
        saved["beatrice_block_time"] = self.block_times["beatrice"]
        saved["theme"] = self.theme
        saved["show_flow_options"] = self.show_flow_options.get()
        self._write_config(saved)

    def _toggle(self):
        if self.engine.running:
            self._stop()
        else:
            self._start()

    def _set_running(self, running):
        self.start_button.configure(text="Stop" if running else "Start")
        self.sample_rate_text.set(f"{self.engine.sample_rate / 1000:g} kHz" if running else "")
        self._refresh_states()

    def _start(self):
        try:
            settings = self._settings()
            self.status.set("Loading model and starting audio streams...")
            self.start_button.state(["disabled"])
            self.root.update_idletasks()
            self.engine.start(settings)
            self._save_config(settings)
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
        except Exception as error:
            self.engine.stop()
            self.status.set(f"Start failed: {error}")
            messagebox.showerror("Applio Real-Time", str(error))
        finally:
            self.start_button.state(["!disabled"])
            self._set_running(self.engine.running)

    def _stop(self):
        self.engine.stop()
        self._set_running(False)
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

    def _update_meters(self):
        engine = self.engine
        monitor = 0.0
        if engine.running and engine.monitor_queue is not None:
            try:
                monitor = engine.output_level * db_to_linear(self.monitor_gain_db.get())
            except tk.TclError:
                pass
        for key, level in (("input", engine.input_level), ("output", engine.output_level), ("monitor", monitor)):
            value = max(level_percent(level), self.meter_values[key] - 6.0)
            self.meter_values[key] = value
            self.meters[key]["value"] = value

    def _poll(self):
        if self.engine.running:
            self.stats.set(
                f"Latency {self.engine.algorithm_latency_ms} ms   ·   "
                f"Processing {self.engine.last_block_ms} ms   ·   "
                f"Model {self.engine.last_infer_ms} ms"
            )
        else:
            self.stats.set("")
        self._update_meters()
        try:
            while True:
                error = self.error_queue.get_nowait()
                self.status.set(error.strip().splitlines()[-1])
                if not self.engine.running:
                    self.engine.stop()
                    self._set_running(False)
        except queue.Empty:
            pass
        self.root.after(50, self._poll)

    def _close(self):
        self.engine.stop()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    from shared.tools.prerequisites_download import prequisites_download_pipeline

    prequisites_download_pipeline(pretraineds_hifigan=False, models=True, exe=False)
    RealtimeGUI().run()
