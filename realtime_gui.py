import ctypes
import json
import math
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

if __name__ == "__main__":
    from shared.updater import update_and_restart

    update_and_restart(__file__, "realtime GUI")

try:
    import sv_ttk
except ImportError:
    sv_ttk = None

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_BLOCKTIME", "1")
os.environ.setdefault("SD_ENABLE_ASIO", "1")

from shared.cuda_sync import use_blocking_cuda_sync

use_blocking_cuda_sync()

import torch
import numpy as np
import sounddevice as sd

from beatrice.inference import find_paraphernalia
from rvc.infer.realtime_engine import BEATRICE_BLOCK_TIME, RealtimeProcessor, db_to_linear
from shared.platform import migrate_models_folder


ROOT = Path(__file__).resolve().parent
migrate_models_folder(ROOT)
CONFIG_PATH = ROOT / "assets" / "realtime_config.json"
BEATRICE_THREADS = 1
MAX_CROSSFADE_TIME = 0.1
MONITOR_DISABLED = "Disabled"
MAIN_HOST_APIS = ("ASIO", "Windows WASAPI")
EMBEDDERS = ("contentvec", "spin-v2")
PITCH_METHODS = {"rmvpe": "rmvpe", "swift": "swift", "beatrice": "beatrice", "parselmouth": "pm"}
METER_FLOOR_DB = -60.0
HOTKEYS = sys.platform == "win32"
HOTKEY_POLL_MS = 15
HOTKEY_MODIFIERS = (0x11, 0x10, 0x12)
HOTKEY_SIDE_MODIFIERS = {0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0x5B, 0x5C}
HOTKEY_CANDIDATES = [vk for vk in range(0x02, 0xFF) if vk not in HOTKEY_MODIFIERS]
KEY_NAMES = {0x02: "Right mouse", 0x04: "Middle mouse", 0x05: "Mouse 4", 0x06: "Mouse 5", 0x13: "Pause", 0x2C: "Print Screen"}
EXTENDED_KEYS = {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x2D, 0x2E, 0x6F, 0x90}
VK_ESCAPE = 0x1B


def fade_ramp(start, target, frames, length):
    if start == target:
        return np.full(frames, target, dtype=np.float32)
    steps = np.arange(1, frames + 1, dtype=np.float32) / max(1, length)
    return np.clip(start + (target - start) * steps, 0.0, 1.0).astype(np.float32)


def key_down(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def keys_down():
    return {vk for vk in HOTKEY_CANDIDATES if key_down(vk)}


def key_name(vk):
    if vk in KEY_NAMES:
        return KEY_NAMES[vk]
    user32 = ctypes.windll.user32
    scan = user32.MapVirtualKeyW(vk, 4)
    extended = 1 << 24 if vk in EXTENDED_KEYS or scan >> 8 == 0xE0 else 0
    buffer = ctypes.create_unicode_buffer(64)
    if user32.GetKeyNameTextW(((scan & 0xFF) << 16) | extended, buffer, len(buffer)):
        return buffer.value.title()
    return f"Key {vk}"


class AudioFrameFifo:
    def __init__(self, channels=1, max_frames=None):
        self.channels = int(channels)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.prefill = 0
        self.dropouts = 0
        self.fade = 0
        self.trim_threshold = 0
        self._chunks = deque()
        self._frames = 0
        self._starved = True
        self._levels = None
        self._largest_read = 0
        self._lock = threading.Lock()

    def drain_after(self, writes, fade, threshold):
        self._levels = deque(maxlen=max(1, int(writes)))
        self.fade = int(fade)
        self.trim_threshold = int(threshold)

    def write(self, data):
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]
        if not array.shape[0]:
            return
        with self._lock:
            if self._levels is not None:
                self._levels.append(self._frames)
            if self._starved and self.prefill > 0:
                self._chunks.append(np.zeros((self.prefill, array.shape[1]), dtype=np.float32))
                self._frames += self.prefill
            self._starved = False
            self._chunks.append(array.copy())
            self._frames += array.shape[0]
            if self.max_frames is not None and self._frames > self.max_frames:
                self._discard_locked(self._frames - self.max_frames)
            if self._levels is not None and len(self._levels) == self._levels.maxlen:
                excess = min(self._levels) - self.prefill - self._largest_read
                if excess > self.trim_threshold:
                    self._trim_locked(excess)
                    self._levels.clear()

    def read(self, frames, exact=False):
        with self._lock:
            self._largest_read = max(self._largest_read, int(frames))
            if exact and self._frames < int(frames):
                return None
            take = min(int(frames), self._frames)
            if take < int(frames):
                if not self._starved:
                    self.dropouts += 1
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

    def _trim_locked(self, frames):
        data = np.concatenate(self._chunks, axis=0)
        kept = data[frames:].copy()
        fade = min(self.fade, frames, kept.shape[0])
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)[:, None]
            kept[:fade] = data[:fade] * (1.0 - ramp) + kept[:fade] * ramp
        self._chunks = deque([kept])
        self._frames = kept.shape[0]

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


class AudioEngine(RealtimeProcessor):
    def __init__(self, error_queue):
        super().__init__()
        self.error_queue = error_queue
        self.input_stream = None
        self.output_stream = None
        self.monitor_stream = None
        self.asio_stream = None
        self.output_fifo = None
        self.monitor_queue = None
        self.direct_output = None
        self.direct_monitor = None
        self.dry_gains = {}
        self.direct_history = None
        self.direct_gate_gain = 1.0
        self.input_fifo = None
        self.worker = None
        self.worker_stop = None
        self.worker_wakeup = None
        self.asio_device = None
        self.asio_main_channels = 0
        self.asio_callback_frames = 0
        self.running = False
        self.algorithm_latency_ms = 0
        self.base_latency_ms = 0.0
        self.device_latency_ms = 0.0
        self.reported_statuses = set()
        self.shared_fallbacks = []

    def start(self, settings):
        self.stop()
        model_rate, beatrice_path = self.load_model(settings)
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
        self.prepare(beatrice_path)
        self.reported_statuses = set()
        self.asio_callback_frames = 0
        self.prefill_margin = max(self.block_frame // 4, self.zc)
        self.output_fifo = AudioFrameFifo(1, max_frames=self.block_frame * 4)
        self._drain(self.output_fifo)
        if self.output_selectors or self.input_selectors:
            self._resize_asio_fifo(self.output_fifo)
        else:
            self.output_fifo.prefill = min(self.block_frame, self.sola_buffer_frame)
        self.dry_gains = dict.fromkeys(("output", "monitor"), float(settings["passthrough"]))
        self.direct_history = np.zeros(4 * self.zc, dtype=np.float32)
        self.direct_gate_gain = 1.0
        self.direct_output = self._direct_fifo(self.output_selectors)
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
        self.device_latency_ms = (input_latency + output_latency) * 1000
        self.base_latency_ms = self.device_latency_ms + (self.block_frame / self.sample_rate + model_latency) * 1000
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
        def check(checker, device, channels, rate, selectors=None, exclusive=False):
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
                    sd.check_input_settings, self.input_device, self.channels, rate, self.input_selectors,
                    self.settings.get("exclusive_input", False),
                )
                self.output_extra, output_shared = check(
                    sd.check_output_settings, self.output_device, self.output_channels, rate,
                    self.output_selectors, self.settings.get("exclusive_output", False),
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

    def _start_monitor_stream(self):
        self.monitor_stream = None
        self.monitor_queue = None
        self.direct_monitor = None
        if self.monitor_device is None:
            return
        try:
            self.monitor_queue = AudioFrameFifo(1, max_frames=self.block_frame * 4)
            self._drain(self.monitor_queue)
            if self.input_selectors:
                self._resize_asio_fifo(self.monitor_queue)
            else:
                self.monitor_queue.prefill = min(self.block_frame, self.sola_buffer_frame)
            self.direct_monitor = self._direct_fifo(self.monitor_selectors)
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
            self.direct_monitor = None
            self.error_queue.put_nowait(
                f"The monitor device could not start; continuing with the main output only. {error}"
            )

    def _drain(self, fifo):
        fifo.drain_after(math.ceil(5 * self.sample_rate / self.block_frame), self.zc, 5 * self.zc)

    def _resize_asio_fifo(self, fifo):
        frames = self.asio_callback_frames
        fifo.prefill = 2 * frames + max(0, frames - self.block_frame) + self.prefill_margin
        fifo.max_frames = 2 * (self.block_frame + fifo.prefill)

    def _direct_fifo(self, selectors):
        fifo = AudioFrameFifo(1)
        fifo.drain_after(math.ceil(5 * self.sample_rate / self.zc), self.zc, 2 * self.zc)
        self._resize_direct_fifo(fifo, selectors)
        return fifo

    def _resize_direct_fifo(self, fifo, selectors):
        frames = self.asio_callback_frames
        fifo.prefill = 0 if self.input_selectors and selectors else self.zc + frames
        fifo.max_frames = 8 * self.zc + 4 * frames

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
        for fifo, selectors in (
            (self.direct_output, self.output_selectors),
            (self.direct_monitor, self.monitor_selectors),
        ):
            if fifo is not None:
                self._resize_direct_fifo(fifo, selectors)

    def _start_asio_stream(self):
        if self.asio_device is None:
            return
        output_selectors = list(self.output_selectors or [])
        self.asio_main_channels = len(output_selectors)
        if self.monitor_selectors:
            self.monitor_queue = AudioFrameFifo(1)
            self._drain(self.monitor_queue)
            self._resize_asio_fifo(self.monitor_queue)
            self.direct_monitor = self._direct_fifo(self.monitor_selectors)
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
        if self.beatrice is not None:
            torch.set_num_threads(BEATRICE_THREADS)
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
        if self.settings["passthrough"]:
            direct = self.direct_output
            buffer_ms = 1000 * direct.prefill / self.sample_rate if direct is not None else 0
            self.algorithm_latency_ms = round(self.device_latency_ms + buffer_ms)
            return
        pitch_delay_ms = 0
        if self.rvc is not None:
            pitch_delay_ms = 10 * self.rvc.pitch_lookahead_frames(self.settings["f0_method"])
        noise_reduction_ms = 0
        if self.settings["input_noise_reduce"]:
            noise_reduction_ms = 1000 * self.sola_buffer_frame / self.sample_rate
        prefill_ms = 0
        output_fifo = self.output_fifo
        if output_fifo is not None and self.sample_rate:
            prefill_ms = 1000 * output_fifo.prefill / self.sample_rate
        self.algorithm_latency_ms = round(
            self.base_latency_ms + pitch_delay_ms + noise_reduction_ms + prefill_ms + self.last_block_ms
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
        self.direct_output = None
        self.direct_monitor = None
        self.input_fifo = None
        self._release_models()

    @property
    def dropouts(self):
        fifo = self.output_fifo
        return 0 if fifo is None else fifo.dropouts

    def _report_status(self, status):
        text = str(status)
        if text not in self.reported_statuses:
            self.reported_statuses.add(text)
            self.error_queue.put_nowait(text)

    def _handle_block(self, indata):
        output = self.process(indata)
        output_fifo = self.output_fifo
        if output_fifo is not None:
            output_fifo.write(output)
        monitor_queue = self.monitor_queue
        if monitor_queue is not None:
            monitor_queue.write(
                np.clip(output * np.float32(db_to_linear(self.settings["monitor_gain_db"])), -1.0, 1.0)
            )

    def _gate_direct(self, dry):
        threshold = self.settings["threshold"]
        self.direct_history = np.concatenate((self.direct_history, dry))[-4 * self.zc :]
        target = 1.0
        if threshold > -60:
            rms = float(np.sqrt(np.mean(np.square(self.direct_history))))
            target = float(20.0 * math.log10(max(rms, 1e-5)) >= threshold)
        gain = fade_ramp(self.direct_gate_gain, target, dry.shape[0], self.zc)
        self.direct_gate_gain = float(gain[-1])
        return dry * gain

    def _write_direct(self, indata):
        direct = self.direct_output
        if direct is None:
            return
        settings = self.settings
        dry = indata.mean(axis=1, dtype=np.float32)
        dry *= np.float32(db_to_linear(settings["input_gain_db"]))
        np.clip(dry, -1.0, 1.0, out=dry)
        dry = self._gate_direct(dry)
        dry *= np.float32(db_to_linear(settings["output_gain_db"]))
        np.clip(dry, -1.0, 1.0, out=dry)
        if settings["passthrough"]:
            self.output_level = float(np.abs(dry).max(initial=0.0))
        direct.write(dry)
        monitor = self.direct_monitor
        if monitor is not None:
            monitor.write(np.clip(dry * np.float32(db_to_linear(settings["monitor_gain_db"])), -1.0, 1.0))

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
            self._write_direct(indata)
        except Exception:
            self.running = False
            self.error_queue.put_nowait(traceback.format_exc())
            raise sd.CallbackAbort

    def _asio_input_callback(self, indata, frames, times, status):
        if not self.running:
            return
        self._track_asio_frames(frames)
        self._input_callback(indata, frames, times, status)

    def _mix(self, key, fifo, direct, frames):
        target = 1.0 if self.settings["passthrough"] else 0.0
        gain = fade_ramp(self.dry_gains.get(key, target), target, frames, self.crossfade_frame)
        self.dry_gains[key] = float(gain[-1])
        mixed = np.zeros(frames, dtype=np.float32)
        for source, weight in ((fifo, 1.0 - gain), (direct, gain)):
            block = source.read(frames) if source is not None else None
            if block is not None:
                mixed[: block.shape[0]] += block[:, 0] * weight[: block.shape[0]]
        return mixed[:, None]

    def _asio_output_callback(self, outdata, frames, times, status):
        outdata.fill(0)
        self._track_asio_frames(frames)
        main_channels = self.asio_main_channels
        if self.output_selectors:
            outdata[:, :main_channels] = self._mix("output", self.output_fifo, self.direct_output, frames)
        if self.monitor_selectors:
            outdata[:, main_channels:] = self._mix("monitor", self.monitor_queue, self.direct_monitor, frames)

    def _asio_duplex_callback(self, indata, outdata, frames, times, status):
        self._asio_input_callback(indata, frames, times, status)
        self._asio_output_callback(outdata, frames, times, status)

    def _fifo_output_callback(self, outdata, frames, times, status):
        outdata[:] = self._mix("output", self.output_fifo, self.direct_output, frames)

    def _monitor_callback(self, outdata, frames, times, status):
        outdata[:] = self._mix("monitor", self.monitor_queue, self.direct_monitor, frames)


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
        self._load_devices(rescan=False)
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
        if HOTKEYS:
            self.root.after(HOTKEY_POLL_MS, self._poll_hotkey)
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
        legacy_exclusive = value.get("wasapi_exclusive", False)
        self.exclusive_input = tk.BooleanVar(value=value.get("exclusive_input", legacy_exclusive))
        self.exclusive_output = tk.BooleanVar(value=value.get("exclusive_output", legacy_exclusive))
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
            "beatrice": BEATRICE_BLOCK_TIME,
        }
        self.block_kind = "rvc"
        self.block_time = tk.DoubleVar(value=self.block_times["rvc"])
        self.crossfade_time = tk.DoubleVar(value=min(value.get("crossfade_time", 0.01), MAX_CROSSFADE_TIME))
        self.extra_time = tk.DoubleVar(value=value.get("extra_time", 3.0))
        self.use_fp32 = tk.BooleanVar(value=value.get("use_fp32", False))
        self.passthrough = tk.BooleanVar(value=False)
        self.passthrough_hold = tk.BooleanVar(value=value.get("passthrough_hold", False))
        self.hotkey = list(value.get("passthrough_hotkey", [])) if HOTKEYS else []
        self.hotkey_down = False
        self.hotkey_capture = None
        self.hotkey_text = tk.StringVar()
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
        self._slider(beatrice, 0, "Formant", self.formant_shift, -2.0, 2.0, 0.5)
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
        exclusive_input = ttk.Checkbutton(options, text="Exclusive input", variable=self.exclusive_input)
        exclusive_input.pack(side="left", padx=(16, 0))
        Tooltip(exclusive_input, "WASAPI exclusive mode for the input. Lower delay, but other apps "
                                 "cannot record from this device while running.")
        exclusive_output = ttk.Checkbutton(options, text="Exclusive output", variable=self.exclusive_output)
        exclusive_output.pack(side="left", padx=(16, 0))
        Tooltip(exclusive_output, "WASAPI exclusive mode for the output. Lower delay, but other apps "
                                  "cannot play to this device while running. The monitor always stays shared.")
        legacy = ttk.Checkbutton(
            options, text="Legacy drivers", variable=self.show_legacy_devices, command=self._load_devices
        )
        legacy.pack(side="left", padx=(16, 0))
        Tooltip(legacy, "Also list MME, DirectSound and WDM-KS devices.")
        ttk.Label(options, textvariable=self.sample_rate_text, style="Muted.TLabel").pack(side="right")
        self.locked_widgets.update((*combos, reload_button, exclusive_input, exclusive_output, legacy))
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
        block_widgets = self._slider(
            frame, 0, "Block time", self.block_time, 0.02, 1.5, 0.01,
            tooltip=(
                "Seconds of audio per step. Lower is less latency but more load. "
                "Beatrice always uses 0.02 s."
            ),
        )
        self.block_time_slider = block_widgets[-2]
        self.block_time_slider.configure(variable=self.block_time)
        self.rvc_widgets.update(block_widgets)
        self.locked_widgets.update(block_widgets)
        crossfade = self._slider(
            frame, 1, "Crossfade", self.crossfade_time, 0.01, MAX_CROSSFADE_TIME, 0.01,
            tooltip=(
                "Seconds each block blends into the next. Longer hides block joins better but adds the same "
                "delay. Capped at the block time."
            ),
        )
        extra = self._slider(
            frame, 2, "Extra context", self.extra_time, 0.05, 5.0, 0.01,
            tooltip="Seconds of past audio the model sees. More is smoother but slower.",
        )
        precision = ttk.Checkbutton(frame, text="FP32 inference", variable=self.use_fp32)
        precision.grid(row=3, column=0, columnspan=3, sticky="w", pady=3)
        Tooltip(precision, "Use full precision. Otherwise compatible NVIDIA GPUs use FP16 for RVC. "
                           "CPU, GTX 16-series and Flow models use FP32. Applies on the next Start.")
        self.rvc_widgets.update(crossfade + extra + [precision])
        self.locked_widgets.update(crossfade + extra + [precision])
        pitch_label = ttk.Label(frame, text="Pitch extraction")
        pitch_label.grid(row=4, column=0, sticky="w", padx=(0, 12), pady=(6, 3))
        methods = ttk.Frame(frame)
        methods.grid(row=4, column=1, columnspan=2, sticky="w", pady=(6, 3))
        self.rvc_widgets.add(pitch_label)
        for label, method in PITCH_METHODS.items():
            radio = ttk.Radiobutton(methods, text=label, variable=self.f0_method, value=method)
            radio.pack(side="left", padx=(0, 12))
            self.rvc_widgets.add(radio)
        return frame

    def _build_actions(self, parent):
        frame = ttk.Frame(parent)
        frame.columnconfigure(3, weight=1)
        self.start_button = ttk.Button(frame, text="Start", style="Accent.TButton", width=12, command=self._toggle)
        self.start_button.grid(row=0, column=0, sticky="w")
        passthrough = ttk.Checkbutton(frame, text="Passthrough", variable=self.passthrough,
                                      style="Switch.TCheckbutton")
        passthrough.grid(row=0, column=1, sticky="w", padx=(16, 0))
        Tooltip(passthrough, "Send the input straight to the output without conversion or its delay. "
                             "Input noise reduction does not apply.")
        if HOTKEYS:
            hotkey = ttk.Frame(frame)
            hotkey.grid(row=0, column=2, sticky="w", padx=(12, 0))
            hotkey_button = ttk.Button(hotkey, textvariable=self.hotkey_text, command=self._capture_hotkey)
            hotkey_button.pack(side="left")
            Tooltip(hotkey_button, "Optional global hotkey for Passthrough that works while other apps are "
                                   "focused. Click, then press a key, key combination or mouse button. "
                                   "Esc removes it.")
            hold = ttk.Checkbutton(hotkey, text="Hold", variable=self.passthrough_hold, command=self._store_hotkey)
            hold.pack(side="left", padx=(8, 0))
            Tooltip(hold, "On: Passthrough only while the hotkey is held. Off: the hotkey toggles Passthrough.")
            self._show_hotkey()
        ttk.Label(frame, textvariable=self.stats, style="Muted.TLabel").grid(row=0, column=3, sticky="e")
        status = ttk.Label(frame, textvariable=self.status, style="Muted.TLabel")
        status.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(8, 0))
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
        if beatrice:
            self.block_time.set(BEATRICE_BLOCK_TIME)
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

    def _load_devices(self, rescan=True):
        try:
            if rescan:
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
            "exclusive_input": self.exclusive_input.get(),
            "exclusive_output": self.exclusive_output.get(),
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
            "block_time": BEATRICE_BLOCK_TIME if self.block_kind == "beatrice" else self.block_time.get(),
            "crossfade_time": self.crossfade_time.get(),
            "extra_time": self.extra_time.get(),
            "use_fp32": self.use_fp32.get(),
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
        saved["beatrice_block_time"] = BEATRICE_BLOCK_TIME
        saved["theme"] = self.theme
        saved["show_flow_options"] = self.show_flow_options.get()
        saved["passthrough_hotkey"] = self.hotkey
        saved["passthrough_hold"] = self.passthrough_hold.get()
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
            self.engine._refresh_latency()
        except (tk.TclError, ValueError) as error:
            self.status.set(str(error))

    def _show_hotkey(self):
        if self.hotkey_capture is not None:
            text = "Press a key..."
        elif self.hotkey:
            text = "Hotkey: " + "+".join(key_name(vk) for vk in self.hotkey)
        else:
            text = "Set hotkey"
        self.hotkey_text.set(text)

    def _store_hotkey(self):
        data = self._load_config()
        data["passthrough_hotkey"] = self.hotkey
        data["passthrough_hold"] = self.passthrough_hold.get()
        try:
            self._write_config(data)
        except OSError:
            pass

    def _capture_hotkey(self):
        if self.hotkey_capture is None:
            self.hotkey_capture = {"held": keys_down(), "modifier": None}
            self.root.focus_set()
        else:
            self.hotkey_capture = None
        self._show_hotkey()

    def _set_hotkey(self, keys):
        self.hotkey_capture = None
        self.hotkey = keys
        self.hotkey_down = bool(keys) and all(key_down(vk) for vk in keys)
        self._show_hotkey()
        self._store_hotkey()

    def _capture_step(self):
        capture = self.hotkey_capture
        held = keys_down()
        pressed = held - capture["held"]
        released = capture["held"] - held
        capture["held"] = held
        for vk in sorted(pressed):
            if vk == VK_ESCAPE:
                self._set_hotkey([])
                return
            if vk in HOTKEY_SIDE_MODIFIERS:
                capture["modifier"] = vk
            else:
                self._set_hotkey([modifier for modifier in HOTKEY_MODIFIERS if key_down(modifier)] + [vk])
                return
        if capture["modifier"] in released:
            self._set_hotkey([capture["modifier"]])

    def _poll_hotkey(self):
        if self.hotkey_capture is not None:
            self._capture_step()
        elif self.hotkey:
            down = all(key_down(vk) for vk in self.hotkey)
            if down != self.hotkey_down:
                self.hotkey_down = down
                if self.passthrough_hold.get():
                    self.passthrough.set(down)
                elif down:
                    self.passthrough.set(not self.passthrough.get())
        self.root.after(HOTKEY_POLL_MS, self._poll_hotkey)

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
        if self.engine.running and self.engine.settings["passthrough"]:
            self.stats.set(f"Latency {self.engine.algorithm_latency_ms} ms   ·   Passthrough")
        elif self.engine.running:
            stats = (
                f"Latency {self.engine.algorithm_latency_ms} ms   ·   "
                f"Processing {self.engine.last_block_ms} ms   ·   "
                f"Model {self.engine.last_infer_ms} ms"
            )
            dropouts = self.engine.dropouts
            if dropouts:
                stats += f"   ·   Dropouts {dropouts}"
            self.stats.set(stats)
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
