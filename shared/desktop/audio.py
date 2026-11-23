"""Local PortAudio streams and buffering shared by the desktop realtime controls."""

import math
import os
import threading
import traceback
from collections import deque

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("KMP_BLOCKTIME", "1")
os.environ.setdefault("SD_ENABLE_ASIO", "1")

from shared.cuda_sync import use_blocking_cuda_sync

use_blocking_cuda_sync()

import numpy as np
import sounddevice as sd

from rvc.infer.realtime_engine import RealtimeProcessor, db_to_linear
from shared.desktop.devices import (
    asio_channel_choices,
    audio_routes,
    device_sort_key,
    friendly_device_label,
)
from shared.tools.causal_voices import SAMPLE_RATE as CAUSAL_SAMPLE_RATE


def fade_ramp(start, target, frames, length):
    if start == target:
        return np.full(frames, target, dtype=np.float32)
    steps = np.arange(1, frames + 1, dtype=np.float32) / max(1, length)
    return np.clip(start + (target - start) * steps, 0.0, 1.0).astype(np.float32)


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
        model_rate = self.load_model(settings)
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
        self.prepare()
        if self.causal is not None:
            self.crossfade_frame = self.zc
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
        self.device_latency_ms = (input_latency + output_latency) * 1000
        self.base_latency_ms = self.device_latency_ms + 1000 * self.block_frame / self.sample_rate
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
            if self.causal is not None and rate != CAUSAL_SAMPLE_RATE:
                continue
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
        if self.causal is not None:
            raise ValueError("Casual Voice Changer needs a 48 kHz input and output route.\n" + "\n".join(errors))
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
        conversion_delay_ms = 1000 * self.conversion_delay
        noise_reduction_ms = 0
        if self.settings.get("input_noise_reduce") and self.rnnoise is not None:
            noise_reduction_ms = self.rnnoise.latency_ms
        prefill_ms = 0
        output_fifo = self.output_fifo
        if output_fifo is not None and self.sample_rate:
            prefill_ms = 1000 * output_fifo.prefill / self.sample_rate
        self.algorithm_latency_ms = round(
            self.base_latency_ms + conversion_delay_ms + noise_reduction_ms + prefill_ms + self.last_block_ms
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
