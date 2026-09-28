import json
import os
import queue
import threading
import time
import traceback
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
from torchaudio.transforms import Resample

from rvc.infer.realtime import RealTimeRVC


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "assets" / "realtime_config.json"


class AudioRingBuffer:
    def __init__(self, capacity, channels):
        self.capacity = int(capacity)
        self.channels = int(channels)
        self.data = np.zeros((self.capacity, self.channels), dtype=np.float32)
        self.read_position = 0
        self.write_position = 0

    @property
    def available(self):
        return self.write_position - self.read_position

    @property
    def free(self):
        return self.capacity - self.available

    def write(self, values):
        count = min(int(values.shape[0]), self.free)
        if count <= 0:
            return 0
        start = self.write_position % self.capacity
        first = min(count, self.capacity - start)
        self.data[start : start + first] = values[:first]
        remaining = count - first
        if remaining:
            self.data[:remaining] = values[first : first + remaining]
        self.write_position += count
        return count

    def read_into(self, target):
        count = min(int(target.shape[0]), self.available)
        if count <= 0:
            return 0
        start = self.read_position % self.capacity
        first = min(count, self.capacity - start)
        target[:first] = self.data[start : start + first]
        remaining = count - first
        if remaining:
            target[first : first + remaining] = self.data[:remaining]
        self.read_position += count
        return count


class AudioEngine:
    SOLA_SILENCE_RMS = 1e-4
    SOLA_MIN_CORRELATION = 0.2
    SOLA_MIN_IMPROVEMENT = 0.05

    def __init__(self, error_queue):
        self.error_queue = error_queue
        self.stream = None
        self.rvc = None
        self.running = False
        self.last_infer_ms = 0
        self.last_block_ms = 0
        self.algorithm_latency_ms = 0
        self.worker_thread = None
        self.worker_event = threading.Event()
        self.worker_stop = threading.Event()
        self.input_ring = None
        self.output_ring = None
        self.worker_input = None
        self.prime_frames_remaining = 0
        self.output_primed = False
        self.input_overflow_reported = False
        self.output_underflow_reported = False
        self.base_latency_ms = 0.0

    def start(self, settings):
        self.stop()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.rvc = RealTimeRVC(
            model_path=settings["model_path"],
            index_path=settings["index_path"],
            index_rate=settings["index_rate"],
            pitch=settings["pitch"],
            speaker_id=settings["speaker_id"],
            embedder_model=settings["embedder_model"],
        )
        input_info = sd.query_devices(settings["input_device"])
        output_info = sd.query_devices(settings["output_device"])
        stream_rate = (
            self.rvc.sample_rate
            if settings["sample_rate_mode"] == "model"
            else int(input_info["default_samplerate"])
        )
        channels = min(
            int(input_info["max_input_channels"]),
            int(output_info["max_output_channels"]),
            2,
        )
        if channels < 1:
            raise ValueError("The selected devices do not support full-duplex audio.")
        sd.check_input_settings(
            device=settings["input_device"],
            channels=channels,
            dtype="float32",
            samplerate=stream_rate,
        )
        sd.check_output_settings(
            device=settings["output_device"],
            channels=channels,
            dtype="float32",
            samplerate=stream_rate,
        )
        self.settings = settings
        self.sample_rate = stream_rate
        self.channels = channels
        self.zero_crossing = stream_rate // 100
        self.block_frame = (
            round(settings["block_time"] * stream_rate / self.zero_crossing)
            * self.zero_crossing
        )
        self.block_frame_16k = 160 * self.block_frame // self.zero_crossing
        self.crossfade_frame = (
            round(settings["crossfade_time"] * stream_rate / self.zero_crossing)
            * self.zero_crossing
        )
        self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zero_crossing)
        self.sola_search_frame = self.zero_crossing
        self.extra_frame = (
            round(settings["extra_time"] * stream_rate / self.zero_crossing)
            * self.zero_crossing
        )
        device = self.rvc.device
        total_input = (
            self.extra_frame
            + self.crossfade_frame
            + self.sola_search_frame
            + self.block_frame
        )
        self.input_wav = torch.zeros(total_input, device=device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(
            160 * total_input // self.zero_crossing,
            device=device,
            dtype=torch.float32,
        )
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=device, dtype=torch.float32
        )
        self.sola_den_kernel = torch.ones(
            1, 1, self.sola_buffer_frame, device=device, dtype=torch.float32
        )
        self.skip_head = self.extra_frame // self.zero_crossing
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zero_crossing
        phase = torch.linspace(
            0.0,
            1.0,
            steps=self.sola_buffer_frame,
            device=device,
            dtype=torch.float32,
        )
        self.fade_in = torch.sin(0.5 * np.pi * phase) ** 2
        self.fade_out = 1.0 - self.fade_in
        self.input_resampler = Resample(
            orig_freq=stream_rate,
            new_freq=16000,
            dtype=torch.float32,
        ).to(device)
        self.output_resampler = None
        if self.rvc.sample_rate != stream_rate:
            self.output_resampler = Resample(
                orig_freq=self.rvc.sample_rate,
                new_freq=stream_rate,
                dtype=torch.float32,
            ).to(device)
        extra_settings = None
        if settings["wasapi_exclusive"] and "WASAPI" in settings["host_api"]:
            extra_settings = sd.WasapiSettings(exclusive=True)
        ring_capacity = self.block_frame * 4
        self.input_ring = AudioRingBuffer(ring_capacity, channels)
        self.output_ring = AudioRingBuffer(ring_capacity, channels)
        self.worker_input = np.empty(
            (self.block_frame, channels), dtype=np.float32
        )
        self.prime_frames_remaining = 5 * self.zero_crossing
        self.output_primed = False
        self.input_overflow_reported = False
        self.output_underflow_reported = False
        self.worker_stop.clear()
        self.worker_event.clear()
        self.stream = sd.Stream(
            device=(settings["input_device"], settings["output_device"]),
            samplerate=stream_rate,
            blocksize=0,
            channels=channels,
            dtype="float32",
            latency="low",
            extra_settings=extra_settings,
            callback=self._callback,
        )
        self.rvc.reset_caches()
        self.rvc.infer(
            self.input_wav_res,
            self.block_frame_16k,
            self.skip_head,
            self.return_length,
            settings["f0_method"],
        )
        self.rvc.reset_caches()
        self._process_block(
            np.zeros((self.block_frame, channels), dtype=np.float32)
        )
        self.rvc.reset_caches()
        self.input_wav.zero_()
        self.input_wav_res.zero_()
        self.sola_buffer.zero_()
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ApplioRealtimeWorker",
            daemon=True,
        )
        self.worker_thread.start()
        self.stream.start()
        stream_latency = self.stream.latency
        if isinstance(stream_latency, tuple):
            input_latency, output_latency = stream_latency
        else:
            input_latency = output_latency = stream_latency
        self.base_latency_ms = (
            input_latency
            + output_latency
            + settings["block_time"]
            + settings["crossfade_time"]
            + 0.01
            + self.prime_frames_remaining / stream_rate
        ) * 1000
        self._refresh_latency()

    def _refresh_latency(self):
        self.algorithm_latency_ms = round(
            self.base_latency_ms + self.last_block_ms
        )

    def stop(self):
        self.running = False
        self.worker_stop.set()
        self.worker_event.set()
        if self.stream is not None:
            try:
                if self.stream.active:
                    self.stream.abort()
            except sd.PortAudioError:
                pass
            finally:
                try:
                    self.stream.close()
                except sd.PortAudioError:
                    pass
                self.stream = None
        if (
            self.worker_thread is not None
            and self.worker_thread is not threading.current_thread()
        ):
            self.worker_thread.join(timeout=5)
        self.worker_thread = None

    def update_pitch(self, pitch):
        if self.rvc is not None:
            self.rvc.change_pitch(pitch)

    def update_index_rate(self, index_rate):
        if self.rvc is not None:
            self.rvc.change_index_rate(index_rate)

    def _gate_silence(self, mono):
        threshold = self.settings["threshold"]
        if threshold <= -60:
            return mono
        gated = mono.copy()
        complete = gated.shape[0] // self.zero_crossing
        if complete:
            frames = gated[: complete * self.zero_crossing].reshape(
                complete, self.zero_crossing
            )
            rms = np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)
            frames[20 * np.log10(rms) < threshold] = 0
        return gated

    def _resample_input(self, source):
        converted = self.input_resampler(source)
        expected = self.block_frame_16k + 160
        converted = converted[160:]
        if converted.shape[0] < expected:
            converted = F.pad(converted, (expected - converted.shape[0], 0))
        return converted[-expected:]

    def _mix_volume(self, converted):
        rate = self.settings["rms_mix_rate"]
        if rate >= 1:
            return converted
        source = self.input_wav[self.extra_frame :]
        source = source[: converted.shape[0]]
        rms_source = librosa.feature.rms(
            y=source.detach().cpu().numpy(),
            frame_length=4 * self.zero_crossing,
            hop_length=self.zero_crossing,
        )
        rms_converted = librosa.feature.rms(
            y=converted.detach().cpu().numpy(),
            frame_length=4 * self.zero_crossing,
            hop_length=self.zero_crossing,
        )
        rms_source = torch.from_numpy(rms_source).to(converted.device)
        rms_converted = torch.from_numpy(rms_converted).to(converted.device)
        rms_source = F.interpolate(
            rms_source.unsqueeze(0),
            size=converted.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms_converted = F.interpolate(
            rms_converted.unsqueeze(0),
            size=converted.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms_converted = torch.clamp(rms_converted, min=1e-3)
        return converted * torch.pow(
            rms_source / rms_converted, 1.0 - rate
        )

    def _find_sola_offset(self, converted):
        search = converted[
            None, None, : self.sola_buffer_frame + self.sola_search_frame
        ]
        reference = self.sola_buffer
        reference_centered = reference - reference.mean()
        reference_energy = reference_centered.square().sum()
        candidate_sum = F.conv1d(search, self.sola_den_kernel)
        candidate_square_sum = F.conv1d(search.square(), self.sola_den_kernel)
        candidate_energy = torch.clamp(
            candidate_square_sum
            - candidate_sum.square() / self.sola_buffer_frame,
            min=0.0,
        )
        numerator = F.conv1d(search, reference_centered[None, None])
        denominator = torch.sqrt(reference_energy * candidate_energy).clamp_min(
            torch.finfo(converted.dtype).eps
        )
        correlation = (numerator / denominator)[0, 0]
        best_correlation, best_offset = torch.max(correlation, dim=0)
        reference_rms = torch.sqrt(reference_energy / self.sola_buffer_frame)
        candidate_rms = torch.sqrt(
            candidate_energy[0, 0, best_offset] / self.sola_buffer_frame
        )
        confident = (
            (reference_rms >= self.SOLA_SILENCE_RMS)
            & (candidate_rms >= self.SOLA_SILENCE_RMS)
            & torch.isfinite(best_correlation)
            & (best_correlation >= self.SOLA_MIN_CORRELATION)
            & (
                best_correlation - correlation[0]
                >= self.SOLA_MIN_IMPROVEMENT
            )
        )
        selected_offset = torch.where(
            confident, best_offset, torch.zeros_like(best_offset)
        )
        return int(selected_offset.item())

    def _apply_sola(self, converted):
        needed = self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        if converted.shape[0] < needed:
            converted = F.pad(converted, (0, needed - converted.shape[0]))
        offset = self._find_sola_offset(converted)
        converted = converted[offset:]
        converted[: self.sola_buffer_frame] *= self.fade_in
        converted[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out
        self.sola_buffer[:] = converted[
            self.block_frame : self.block_frame + self.sola_buffer_frame
        ]
        return converted[: self.block_frame]

    def _process_block(self, indata):
        started = time.perf_counter()
        mono = librosa.to_mono(indata.T).astype(np.float32, copy=False)
        mono = self._gate_silence(mono)
        self.input_wav[:-self.block_frame] = self.input_wav[
            self.block_frame:
        ].clone()
        self.input_wav[-self.block_frame:] = torch.from_numpy(mono).to(
            self.rvc.device
        )
        self.input_wav_res[:-self.block_frame_16k] = self.input_wav_res[
            self.block_frame_16k:
        ].clone()
        resample_source = self.input_wav[
            -self.block_frame - 2 * self.zero_crossing :
        ]
        resampled = self._resample_input(resample_source)
        self.input_wav_res[-resampled.shape[0] :] = resampled
        if self.settings["monitor_input"]:
            converted = self.input_wav[self.extra_frame :].clone()
            infer_seconds = 0.0
        else:
            converted, infer_seconds = self.rvc.infer(
                self.input_wav_res,
                self.block_frame_16k,
                self.skip_head,
                self.return_length,
                self.settings["f0_method"],
            )
            if self.output_resampler is not None:
                converted = self.output_resampler(converted)
            converted = self._mix_volume(converted)
        output = self._apply_sola(converted)
        output = output.repeat(self.channels, 1).t().detach().cpu().numpy()
        np.clip(output, -1.0, 1.0, out=output)
        self.last_infer_ms = round(infer_seconds * 1000)
        self.last_block_ms = round((time.perf_counter() - started) * 1000)
        self._refresh_latency()
        return output

    def _worker_loop(self):
        try:
            while not self.worker_stop.is_set():
                self.worker_event.wait(0.1)
                self.worker_event.clear()
                while (
                    not self.worker_stop.is_set()
                    and self.input_ring.available >= self.block_frame
                    and self.output_ring.free >= self.block_frame
                ):
                    count = self.input_ring.read_into(self.worker_input)
                    if count != self.block_frame:
                        break
                    output = self._process_block(self.worker_input)
                    if self.worker_stop.is_set():
                        break
                    written = self.output_ring.write(output)
                    if written != self.block_frame:
                        raise RuntimeError("The real-time output buffer is full.")
        except Exception:
            self.running = False
            self.worker_stop.set()
            self.error_queue.put_nowait(traceback.format_exc())

    def _callback(self, indata, outdata, frames, timing, status):
        try:
            if not self.running:
                outdata.fill(0)
                raise sd.CallbackStop
            if status:
                self.error_queue.put_nowait(str(status))
            written = self.input_ring.write(indata)
            if written != frames and not self.input_overflow_reported:
                self.input_overflow_reported = True
                self.error_queue.put_nowait("Real-time input buffer overflow.")
            outdata.fill(0)
            if not self.output_primed:
                if self.output_ring.available >= self.block_frame:
                    self.prime_frames_remaining -= frames
                    if self.prime_frames_remaining <= 0:
                        self.output_primed = True
                self.worker_event.set()
                return
            count = self.output_ring.read_into(outdata)
            if count != frames and not self.output_underflow_reported:
                self.output_underflow_reported = True
                self.error_queue.put_nowait("Real-time output buffer underflow.")
            self.worker_event.set()
        except sd.CallbackStop:
            raise
        except Exception:
            outdata.fill(0)
            self.running = False
            self.worker_stop.set()
            self.worker_event.set()
            self.error_queue.put_nowait(traceback.format_exc())
            raise sd.CallbackAbort


class RealtimeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Applio Real-Time Voice Conversion")
        self.root.minsize(760, 620)
        self.error_queue = queue.Queue()
        self.engine = AudioEngine(self.error_queue)
        self.devices = {}
        self.input_devices = {}
        self.output_devices = {}
        self.saved = self._load_config()
        self._make_variables()
        self._build()
        self._load_devices()
        self.pitch.trace_add("write", self._hot_update)
        self.index_rate.trace_add("write", self._hot_update)
        self.rms_mix_rate.trace_add("write", self._hot_update)
        self.threshold.trace_add("write", self._hot_update)
        self.f0_method.trace_add("write", self._hot_update)
        self.monitor_input.trace_add("write", self._hot_update)
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
        self.embedder_model = tk.StringVar(
            value=value.get("embedder_model", "contentvec")
        )
        self.host_api = tk.StringVar(value=value.get("host_api", ""))
        self.input_device = tk.StringVar(value=value.get("input_device", ""))
        self.output_device = tk.StringVar(value=value.get("output_device", ""))
        self.sample_rate_mode = tk.StringVar(
            value=value.get("sample_rate_mode", "device")
        )
        self.wasapi_exclusive = tk.BooleanVar(
            value=value.get("wasapi_exclusive", False)
        )
        self.pitch = tk.IntVar(value=value.get("pitch", 0))
        self.speaker_id = tk.IntVar(value=value.get("speaker_id", 0))
        self.index_rate = tk.DoubleVar(value=value.get("index_rate", 0.0))
        self.rms_mix_rate = tk.DoubleVar(value=value.get("rms_mix_rate", 0.0))
        self.threshold = tk.IntVar(value=value.get("threshold", -60))
        f0_method = value.get("f0_method", "rmvpe")
        self.f0_method = tk.StringVar(
            value="swift" if f0_method == "pm" else f0_method
        )
        self.block_time = tk.DoubleVar(value=value.get("block_time", 0.25))
        self.crossfade_time = tk.DoubleVar(
            value=value.get("crossfade_time", 0.05)
        )
        self.extra_time = tk.DoubleVar(value=value.get("extra_time", 2.5))
        self.monitor_input = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Ready")
        self.latency = tk.StringVar(value="Estimated latency: 0 ms")
        self.infer_time = tk.StringVar(value="Processing: 0 ms (RVC: 0 ms)")

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
        model.columnconfigure(1, weight=1)
        devices = ttk.LabelFrame(root, text="Audio devices", padding=10)
        devices.pack(fill="x", pady=(0, 8))
        ttk.Label(devices, text="Host API").grid(row=0, column=0, sticky="w")
        self.host_combo = ttk.Combobox(
            devices, textvariable=self.host_api, state="readonly"
        )
        self.host_combo.grid(row=0, column=1, sticky="ew", padx=8)
        self.host_combo.bind("<<ComboboxSelected>>", self._host_changed)
        ttk.Button(devices, text="Reload", command=self._load_devices).grid(row=0, column=2)
        ttk.Label(devices, text="Input").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.input_combo = ttk.Combobox(
            devices, textvariable=self.input_device, state="readonly"
        )
        self.input_combo.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Label(devices, text="Output").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.output_combo = ttk.Combobox(
            devices, textvariable=self.output_device, state="readonly"
        )
        self.output_combo.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 0), pady=(8, 0))
        rate_frame = ttk.Frame(devices)
        rate_frame.grid(row=3, column=1, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Radiobutton(
            rate_frame,
            text="Device sample rate",
            variable=self.sample_rate_mode,
            value="device",
        ).pack(side="left")
        ttk.Radiobutton(
            rate_frame,
            text="Model sample rate",
            variable=self.sample_rate_mode,
            value="model",
        ).pack(side="left", padx=(12, 0))
        ttk.Checkbutton(
            rate_frame,
            text="WASAPI exclusive",
            variable=self.wasapi_exclusive,
        ).pack(side="left", padx=(12, 0))
        devices.columnconfigure(1, weight=1)
        settings = ttk.LabelFrame(root, text="Conversion", padding=10)
        settings.pack(fill="both", expand=True, pady=(0, 8))
        self._scale(settings, "Pitch", self.pitch, -24, 24, 0)
        self._scale(settings, "Speaker ID", self.speaker_id, 0, 255, 1)
        self._scale(settings, "Index rate", self.index_rate, 0, 1, 2, 0.01)
        self._scale(settings, "RMS mix", self.rms_mix_rate, 0, 1, 3, 0.01)
        self._scale(settings, "Gate threshold", self.threshold, -60, 0, 4)
        ttk.Label(settings, text="Pitch extraction").grid(row=5, column=0, sticky="w")
        pitch_methods = ttk.Frame(settings)
        pitch_methods.grid(row=5, column=1, sticky="w", pady=4)
        for method in ("rmvpe", "fcpe", "swift"):
            ttk.Radiobutton(
                pitch_methods,
                text=method,
                variable=self.f0_method,
                value=method,
            ).pack(side="left", padx=(0, 10))
        self._scale(settings, "Block time", self.block_time, 0.02, 1.5, 6, 0.01)
        self._scale(
            settings, "Crossfade", self.crossfade_time, 0.01, 0.15, 7, 0.01
        )
        self._scale(settings, "Extra context", self.extra_time, 0.5, 5.0, 8, 0.1)
        toggles = ttk.Frame(settings)
        toggles.grid(row=9, column=0, columnspan=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            toggles,
            text="Monitor input",
            variable=self.monitor_input,
        ).pack(side="left")
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
            filetypes=(("RVC model", "*.pth"), ("All files", "*.*")),
        )
        if path:
            self.model_path.set(path)

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
            device_list = sd.query_devices()
            host_list = sd.query_hostapis()
            self.devices = {index: dict(device) for index, device in enumerate(device_list)}
            host_names = [host["name"] for host in host_list]
            self.host_combo["values"] = host_names
            if self.host_api.get() not in host_names:
                default_input = sd.default.device[0]
                default_host = (
                    self.devices[default_input]["hostapi"]
                    if default_input in self.devices
                    else 0
                )
                self.host_api.set(host_names[default_host] if host_names else "")
            self._populate_devices()
        except Exception as error:
            self.status.set(f"Audio device error: {error}")

    def _host_changed(self, event=None):
        self._populate_devices()

    def _populate_devices(self):
        host_list = sd.query_hostapis()
        host_index = next(
            (
                index
                for index, host in enumerate(host_list)
                if host["name"] == self.host_api.get()
            ),
            0,
        )
        self.input_devices = {
            f"{index}: {device['name']}": index
            for index, device in self.devices.items()
            if device["hostapi"] == host_index and device["max_input_channels"] > 0
        }
        self.output_devices = {
            f"{index}: {device['name']}": index
            for index, device in self.devices.items()
            if device["hostapi"] == host_index and device["max_output_channels"] > 0
        }
        self.input_combo["values"] = list(self.input_devices)
        self.output_combo["values"] = list(self.output_devices)
        if self.input_device.get() not in self.input_devices:
            default = sd.default.device[0]
            selected = next(
                (label for label, index in self.input_devices.items() if index == default),
                next(iter(self.input_devices), ""),
            )
            self.input_device.set(selected)
        if self.output_device.get() not in self.output_devices:
            default = sd.default.device[1]
            selected = next(
                (label for label, index in self.output_devices.items() if index == default),
                next(iter(self.output_devices), ""),
            )
            self.output_device.set(selected)

    def _settings(self):
        if not self.model_path.get().strip():
            raise ValueError("Select a voice model.")
        if self.input_device.get() not in self.input_devices:
            raise ValueError("Select an input device.")
        if self.output_device.get() not in self.output_devices:
            raise ValueError("Select an output device.")
        index_path = self.index_path.get().strip()
        return {
            "model_path": self.model_path.get().strip(),
            "index_path": index_path,
            "embedder_model": self.embedder_model.get(),
            "host_api": self.host_api.get(),
            "input_device": self.input_devices[self.input_device.get()],
            "output_device": self.output_devices[self.output_device.get()],
            "input_device_label": self.input_device.get(),
            "output_device_label": self.output_device.get(),
            "sample_rate_mode": self.sample_rate_mode.get(),
            "wasapi_exclusive": self.wasapi_exclusive.get(),
            "pitch": self.pitch.get(),
            "speaker_id": self.speaker_id.get(),
            "index_rate": self.index_rate.get(),
            "rms_mix_rate": self.rms_mix_rate.get(),
            "threshold": self.threshold.get(),
            "f0_method": self.f0_method.get(),
            "block_time": self.block_time.get(),
            "crossfade_time": self.crossfade_time.get(),
            "extra_time": self.extra_time.get(),
            "monitor_input": self.monitor_input.get(),
        }

    def _save_config(self, settings):
        saved = dict(settings)
        saved["input_device"] = saved.pop("input_device_label")
        saved["output_device"] = saved.pop("output_device_label")
        saved.pop("monitor_input", None)
        CONFIG_PATH.write_text(
            json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _start(self):
        try:
            settings = self._settings()
            self.status.set("Loading model and starting audio stream...")
            self.root.update_idletasks()
            self.engine.start(settings)
            self._save_config(settings)
            self.start_button.configure(state="disabled")
            self.stop_button.configure(state="normal")
            self.status.set(
                f"Running {self.engine.rvc.vocoder} with "
                f"{self.engine.rvc.embedder_name} in FP32 at "
                f"{self.engine.sample_rate} Hz"
            )
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
            self.engine.settings["rms_mix_rate"] = self.rms_mix_rate.get()
            self.engine.settings["threshold"] = self.threshold.get()
            self.engine.settings["f0_method"] = self.f0_method.get()
            self.engine.settings["monitor_input"] = self.monitor_input.get()
        except (tk.TclError, ValueError) as error:
            self.status.set(str(error))

    def _poll(self):
        self.infer_time.set(
            f"Processing: {self.engine.last_block_ms} ms "
            f"(RVC: {self.engine.last_infer_ms} ms)"
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
