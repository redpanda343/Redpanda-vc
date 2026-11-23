"""Streaming input-audio noise reduction backed by Xiph RNNoise."""

from ctypes import CDLL, POINTER, c_float, c_int, c_void_p
from ctypes.util import find_library
import importlib.util
import os
import platform
from threading import Lock

import numpy as np

NATIVE_SAMPLE_RATE = 48000
_PCM_SCALE = np.float32(32768.0)
_PCM_INV_SCALE = np.float32(1.0 / 32768.0)

_LIB_LOCK = Lock()
_LIB = None


def _locate_rnnoise_library():
    system = platform.system()
    lib_name = {
        "Windows": "rnnoise.dll",
        "Linux": "librnnoise.so",
        "Darwin": "librnnoise.dylib",
    }.get(system)
    if lib_name:
        spec = importlib.util.find_spec("pyrnnoise")
        if spec and spec.submodule_search_locations:
            for location in spec.submodule_search_locations:
                candidate = os.path.join(location, lib_name)
                if os.path.isfile(candidate):
                    return candidate
    system_lib = find_library("rnnoise")
    if system_lib:
        return system_lib
    raise RuntimeError(
        "RNNoise library not found. Run the installer or install pyrnnoise==0.4.6 (--no-deps)."
    )


def _load_library():
    global _LIB
    with _LIB_LOCK:
        if _LIB is not None:
            return _LIB
        lib = CDLL(_locate_rnnoise_library())
        lib.rnnoise_create.argtypes = [c_void_p]
        lib.rnnoise_create.restype = c_void_p
        lib.rnnoise_destroy.argtypes = [c_void_p]
        lib.rnnoise_destroy.restype = None
        lib.rnnoise_get_frame_size.argtypes = []
        lib.rnnoise_get_frame_size.restype = c_int
        lib.rnnoise_process_frame.argtypes = [
            c_void_p,
            POINTER(c_float),
            POINTER(c_float),
        ]
        lib.rnnoise_process_frame.restype = c_float
        _LIB = lib
        return _LIB


class RNNoiseDenoiser:
    """Real-time mono float32 denoiser using Xiph RNNoise (48 kHz, 10 ms frames)."""

    def __init__(self, sample_rate: int):
        self.sample_rate = int(sample_rate)
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample rate for RNNoise: {sample_rate}")
        self._lib = _load_library()
        self.frame_size = int(self._lib.rnnoise_get_frame_size())
        self._lock = Lock()
        self._state = None
        self._in_frame = np.empty(self.frame_size, dtype=np.float32)
        self._out_frame = np.empty(self.frame_size, dtype=np.float32)
        self._in_ptr = self._in_frame.ctypes.data_as(POINTER(c_float))
        self._out_ptr = self._out_frame.ctypes.data_as(POINTER(c_float))
        self._pending_in = np.empty(0, dtype=np.float32)
        self._pending_out = np.empty(0, dtype=np.float32)
        self._up = None
        self._down = None
        self.reset()

    @property
    def latency_ms(self) -> float:
        base = 1000.0 * self.frame_size / NATIVE_SAMPLE_RATE
        return base if self.sample_rate == NATIVE_SAMPLE_RATE else base + 12.5

    def _create_state(self):
        state = self._lib.rnnoise_create(None)
        if not state:
            raise RuntimeError("Failed to allocate RNNoise state.")
        return state

    def _reset_unlocked(self):
        if self._state:
            self._lib.rnnoise_destroy(self._state)
            self._state = None
        self._state = self._create_state()
        self._pending_in = np.empty(0, dtype=np.float32)
        self._pending_out = np.empty(0, dtype=np.float32)
        if self.sample_rate != NATIVE_SAMPLE_RATE:
            import soxr

            self._up = soxr.ResampleStream(
                self.sample_rate, NATIVE_SAMPLE_RATE, 1, dtype="float32", quality="MQ"
            )
            self._down = soxr.ResampleStream(
                NATIVE_SAMPLE_RATE, self.sample_rate, 1, dtype="float32", quality="MQ"
            )
        else:
            self._up = None
            self._down = None

    def reset(self):
        with self._lock:
            self._reset_unlocked()

    def close(self):
        with self._lock:
            if self._state:
                self._lib.rnnoise_destroy(self._state)
                self._state = None
            self._up = None
            self._down = None
            self._pending_in = np.empty(0, dtype=np.float32)
            self._pending_out = np.empty(0, dtype=np.float32)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _denoise_48k_frames(self, samples_48k: np.ndarray) -> np.ndarray:
        frame = self.frame_size
        count = samples_48k.shape[0]
        output = np.empty(count, dtype=np.float32)
        in_frame = self._in_frame
        out_frame = self._out_frame
        in_ptr = self._in_ptr
        out_ptr = self._out_ptr
        process_frame = self._lib.rnnoise_process_frame
        state = self._state
        for offset in range(0, count, frame):
            np.multiply(samples_48k[offset : offset + frame], _PCM_SCALE, out=in_frame)
            process_frame(state, out_ptr, in_ptr)
            np.multiply(out_frame, _PCM_INV_SCALE, out=output[offset : offset + frame])
        return output

    def process(self, samples: np.ndarray) -> np.ndarray:
        audio = np.ascontiguousarray(samples, dtype=np.float32)
        target = audio.shape[0]
        if target == 0:
            return audio
        with self._lock:
            if not self._state:
                self._reset_unlocked()
            frame = self.frame_size
            if (
                self.sample_rate == NATIVE_SAMPLE_RATE
                and self._pending_in.size == 0
                and self._pending_out.size == 0
                and target % frame == 0
            ):
                out = self._denoise_48k_frames(audio)
                np.clip(out, -1.0, 1.0, out=out)
                return out

            feed = self._up.resample_chunk(audio) if self._up is not None else audio
            if self._pending_in.size:
                feed = np.concatenate((self._pending_in, feed))
            usable = (feed.shape[0] // frame) * frame
            if usable:
                denoised_48k = self._denoise_48k_frames(feed[:usable])
                self._pending_in = feed[usable:].copy()
                produced = (
                    self._down.resample_chunk(denoised_48k)
                    if self._down is not None
                    else denoised_48k
                )
                if produced.size:
                    self._pending_out = (
                        np.concatenate((self._pending_out, produced))
                        if self._pending_out.size
                        else produced
                    )
            else:
                self._pending_in = feed.copy()

            available = self._pending_out.shape[0]
            if available >= target:
                out = self._pending_out[:target].copy()
                self._pending_out = self._pending_out[target:].copy()
            else:
                out = np.zeros(target, dtype=np.float32)
                if available:
                    out[target - available :] = self._pending_out
                    self._pending_out = np.empty(0, dtype=np.float32)
            np.clip(out, -1.0, 1.0, out=out)
            return out
