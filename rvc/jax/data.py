"""Pure-NumPy NWC dataset loader, bucket sampler, and threaded prefetcher for JAX training.

This module intentionally avoids importing `jax` or `torch` so worker threads/subprocesses
never touch TPU/PJRT or GPU contexts.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import queue
import re
import struct
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import soundfile as sf

from rvc.train.backend import effective_cpu_count


def _available_ram_bytes() -> int:
    """Return available host physical memory in bytes for bounded dataset sample caching."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        pass
    if os.name == "nt":
        try:
            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64),
                ]

            stat = _MemoryStatusEx()
            stat.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullAvailPhys)
        except Exception:
            pass
    return 4 * (1024 ** 3)


def _sample_nbytes(item: tuple[Any, ...]) -> int:
    return sum(x.nbytes for x in item if isinstance(x, np.ndarray))


_NPY_DESCR_RE = re.compile(r"['\"]descr['\"]\s*:\s*(['\"])([^'\"]+)\1")
_NPY_FORTRAN_RE = re.compile(r"['\"]fortran_order['\"]\s*:\s*(True|False)")
_NPY_SHAPE_RE = re.compile(r"['\"]shape['\"]\s*:\s*\(([^)]*)\)")


def _fast_load_npy(path: str, max_rows: int | None = None) -> np.ndarray:
    """Parse C-contiguous .npy header directly and read up to `max_rows` without ast.literal_eval."""
    try:
        with open(path, "rb") as f:
            prefix = f.read(8)
            if len(prefix) != 8 or prefix[:6] != b"\x93NUMPY":
                raise ValueError("Not an NPY file")
            major = prefix[6]
            if major == 1:
                hlen = int.from_bytes(f.read(2), "little")
            elif major == 2:
                hlen = int.from_bytes(f.read(4), "little")
            else:
                raise ValueError(f"Unsupported NPY version {major}")
            hdr = f.read(hlen).decode("latin1")
            m_descr = _NPY_DESCR_RE.search(hdr)
            m_fort = _NPY_FORTRAN_RE.search(hdr)
            m_shape = _NPY_SHAPE_RE.search(hdr)
            if not (m_descr and m_fort and m_shape) or m_fort.group(1) != "False":
                raise ValueError("Non-standard NPY header")
            dtype = np.dtype(m_descr.group(2))
            if dtype.hasobject:
                raise ValueError("Object arrays require np.load validation")
            dims = [int(tok.strip()) for tok in m_shape.group(1).split(",") if tok.strip()]
            if not dims:
                raise ValueError("Empty scalar shape")
            rows = min(dims[0], int(max_rows)) if max_rows is not None else dims[0]
            tail_shape = tuple(dims[1:])
            row_stride = math.prod(tail_shape) if tail_shape else 1
            count = rows * row_stride
            arr = np.fromfile(f, dtype=dtype, count=count)
            if arr.size != count:
                raise ValueError("Short NPY read")
            return arr.reshape((rows,) + tail_shape)
    except Exception:
        return np.load(path, allow_pickle=False)


_DEFAULT_SF_READ = sf.read
_PCM16_INV = np.float32(1.0 / 32768.0)


def _fast_load_wav(
    path: str,
    max_samples: int | None = None,
    *,
    allow_int16: bool = False,
) -> tuple[np.ndarray, int]:
    """Read uncompressed mono PCM-16/float32 WAV directly without libsndfile seek overhead."""
    if sf.read is not _DEFAULT_SF_READ or not path.lower().endswith(".wav"):
        return sf.read(path, dtype="float32")
    try:
        with open(path, "rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                raise ValueError("Not RIFF/WAVE")
            fmt_code = ch = sr = bits = 0
            for _ in range(8):
                chdr = f.read(8)
                if len(chdr) < 8:
                    break
                cid = chdr[:4]
                csize = int.from_bytes(chdr[4:8], "little")
                if cid == b"fmt ":
                    if csize < 16 or csize > 4096:
                        raise ValueError("Invalid fmt chunk")
                    fbody = f.read(csize)
                    if len(fbody) < csize:
                        raise ValueError("Short fmt chunk")
                    if csize & 1:
                        f.read(1)
                    fmt_code, ch, sr, _, _, bits = struct.unpack("<HHIIHH", fbody[:16])
                    if ch != 1 or not ((fmt_code == 1 and bits == 16) or (fmt_code == 3 and bits == 32)):
                        raise ValueError("Unsupported WAV encoding for fast path")
                elif cid == b"data":
                    if sr <= 0 or bits <= 0:
                        raise ValueError("Missing fmt before data")
                    bw = bits // 8
                    n_avail = csize // bw
                    n_read = min(n_avail, int(max_samples)) if max_samples is not None else n_avail
                    n_bytes = n_read * bw
                    raw = f.read(n_bytes)
                    if len(raw) != n_bytes:
                        raise ValueError("Short WAV data")
                    if fmt_code == 1:
                        pcm = np.frombuffer(raw, dtype="<i2")
                        if allow_int16:
                            return np.array(pcm, dtype=np.int16, copy=True), int(sr)
                        audio = pcm.astype(np.float32)
                        audio *= _PCM16_INV
                    else:
                        audio = np.array(np.frombuffer(raw, dtype="<f4"), dtype=np.float32, copy=True)
                    return audio, int(sr)
                else:
                    if csize < 0:
                        raise ValueError("Invalid chunk size")
                    f.seek(csize + (csize & 1), os.SEEK_CUR)
    except Exception:
        pass
    return sf.read(path, dtype="float32")


def _hann_window_np(win_length: int) -> np.ndarray:
    if win_length <= 1:
        return np.ones((max(1, win_length),), dtype=np.float32)
    idx = np.arange(win_length, dtype=np.float64)
    return (0.5 * (1.0 - np.cos(2.0 * math.pi * idx / float(win_length)))).astype(np.float32)


def spectrogram_np(y: np.ndarray, n_fft: int, hop_size: int, win_size: int) -> np.ndarray:
    """Pure-NumPy linear STFT magnitude matching spectrogram_torch(center=False) -> (TT, F)."""
    y_1d = np.asarray(y, dtype=np.float32).reshape(-1)
    pad = int((n_fft - hop_size) / 2)
    y_pad = np.pad(y_1d, (pad, pad), mode="reflect")

    win = _hann_window_np(win_size)
    if win_size < n_fft:
        w_left = (n_fft - win_size) // 2
        w_right = n_fft - win_size - w_left
        win = np.pad(win, (w_left, w_right))

    frames = np.lib.stride_tricks.sliding_window_view(y_pad, window_shape=n_fft)[::hop_size]
    windowed = frames * win[None, :]
    spec_c = np.fft.rfft(windowed, n=n_fft, axis=-1)
    mag = np.sqrt(spec_c.real * spec_c.real + spec_c.imag * spec_c.imag + 2e-7)
    return np.ascontiguousarray(mag.astype(np.float32))


def load_filepaths_and_text(filename: str, split: str = "|") -> list[list[str]]:
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip().split(split) for line in f if line.strip()]


def _probe_audio_frames(audiopath: str, hop_length: int, max_sample_frames: int) -> int:
    try:
        info = sf.info(audiopath)
        if getattr(info, "frames", 0) > 0:
            return min(int(info.frames) // hop_length, max_sample_frames)
    except Exception:
        pass
    return min(os.path.getsize(audiopath) // (3 * hop_length), max_sample_frames)


class TextAudioDataset:
    """Loads WAV audio, optional linear STFT, phoneme features, and pitch for NWC JAX training."""

    def __init__(
        self,
        data_cfg,
        *,
        compute_spec: bool = True,
        repeat_phone: bool = True,
        phone_dtype: Any = np.float32,
    ) -> None:
        self.training_files = str(data_cfg.training_files)
        self.audiopaths_and_text = load_filepaths_and_text(self.training_files)
        self.max_wav_value = float(data_cfg.max_wav_value)
        self.sample_rate = int(data_cfg.sample_rate)
        self.filter_length = int(data_cfg.filter_length)
        self.hop_length = int(data_cfg.hop_length)
        self.win_length = int(data_cfg.win_length)
        self.max_sample_frames = 896
        self.compute_spec = bool(compute_spec)
        self.repeat_phone = bool(repeat_phone)
        self.phone_dtype = np.dtype(phone_dtype)
        self.min_text_len = int(getattr(data_cfg, "min_text_len", 1))
        self.max_text_len = int(getattr(data_cfg, "max_text_len", 5000))
        self._filter()

    @classmethod
    def from_cached_entries(cls, cfg_dict: Mapping[str, Any]) -> "TextAudioDataset":
        """Reconstruct dataset in worker processes using pre-filtered file lists and lengths."""
        obj = cls.__new__(cls)
        obj.training_files = str(cfg_dict.get("training_files", ""))
        obj.audiopaths_and_text = [list(row) for row in cfg_dict["audiopaths_and_text"]]
        obj.lengths = [int(x) for x in cfg_dict.get("lengths", ())]
        obj.max_wav_value = float(cfg_dict.get("max_wav_value", 32768.0))
        obj.sample_rate = int(cfg_dict["sample_rate"])
        obj.filter_length = int(cfg_dict["filter_length"])
        obj.hop_length = int(cfg_dict["hop_length"])
        obj.win_length = int(cfg_dict["win_length"])
        obj.max_sample_frames = int(cfg_dict.get("max_sample_frames", 896))
        obj.compute_spec = bool(cfg_dict.get("compute_spec", False))
        obj.repeat_phone = bool(cfg_dict.get("repeat_phone", True))
        obj.phone_dtype = np.dtype(cfg_dict.get("phone_dtype", "float32"))
        obj.min_text_len = int(cfg_dict.get("min_text_len", 1))
        obj.max_text_len = int(cfg_dict.get("max_text_len", 5000))
        return obj

    def worker_config(self, hop_length: int, allowed_frames: Sequence[int] = ()) -> dict[str, Any]:
        """Serialize picklable dataset metadata for multi-process batch workers."""
        return {
            "training_files": getattr(self, "training_files", ""),
            "audiopaths_and_text": self.audiopaths_and_text,
            "lengths": self.lengths,
            "max_wav_value": self.max_wav_value,
            "sample_rate": self.sample_rate,
            "filter_length": self.filter_length,
            "hop_length": int(hop_length),
            "win_length": self.win_length,
            "max_sample_frames": self.max_sample_frames,
            "compute_spec": self.compute_spec,
            "repeat_phone": self.repeat_phone,
            "phone_dtype": self.phone_dtype.name,
            "min_text_len": self.min_text_len,
            "max_text_len": self.max_text_len,
            "allowed_frames": tuple(int(x) for x in allowed_frames),
        }

    def _filter(self) -> None:
        kept: list[list[str]] = []
        for entry in self.audiopaths_and_text:
            audiopath, text, pitch, pitchf, dv = entry[:5]
            if self.min_text_len <= len(text) <= self.max_text_len:
                kept.append([audiopath, text, pitch, pitchf, dv])

        training_files = getattr(self, "training_files", "")
        cache_path = f"{training_files}.lengths.json" if training_files else ""
        src_stat = None
        if cache_path and os.path.isfile(training_files):
            try:
                src_stat = os.stat(training_files)
                if os.path.isfile(cache_path):
                    with open(cache_path, "r", encoding="utf-8") as cf:
                        meta = json.load(cf)
                    cached = meta.get("lengths")
                    if (
                        meta.get("mtime_ns") == src_stat.st_mtime_ns
                        and meta.get("size") == src_stat.st_size
                        and meta.get("count") == len(kept)
                        and meta.get("hop_length") == self.hop_length
                        and meta.get("max_sample_frames") == self.max_sample_frames
                        and isinstance(cached, list)
                        and len(cached) == len(kept)
                    ):
                        self.audiopaths_and_text = kept
                        self.lengths = [int(x) for x in cached]
                        return
            except (OSError, ValueError, TypeError):
                pass

        if len(kept) > 32:
            workers = min(32, max(4, effective_cpu_count() * 4))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                lengths = list(
                    pool.map(
                        lambda row: _probe_audio_frames(row[0], self.hop_length, self.max_sample_frames),
                        kept,
                    )
                )
        else:
            lengths = [_probe_audio_frames(row[0], self.hop_length, self.max_sample_frames) for row in kept]
        self.audiopaths_and_text = kept
        self.lengths = lengths

        if cache_path and src_stat is not None:
            try:
                tmp_path = f"{cache_path}.tmp.{os.getpid()}"
                with open(tmp_path, "w", encoding="utf-8") as cf:
                    json.dump(
                        {
                            "mtime_ns": src_stat.st_mtime_ns,
                            "size": src_stat.st_size,
                            "count": len(kept),
                            "hop_length": self.hop_length,
                            "max_sample_frames": self.max_sample_frames,
                            "lengths": lengths,
                        },
                        cf,
                    )
                os.replace(tmp_path, cache_path)
            except OSError:
                pass

    def __len__(self) -> int:
        return len(self.audiopaths_and_text)

    def load_labels(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.int32, int]:
        """Load trimmed 50Hz/100Hz phoneme and F0 labels for sample `index`."""
        _, phone_path, pitch_path, pitchf_path, dv_str = self.audiopaths_and_text[index]
        phone_raw = _fast_load_npy(phone_path, max_rows=(self.max_sample_frames + 1) // 2 + 1)
        pitch = _fast_load_npy(pitch_path, max_rows=self.max_sample_frames)
        pitchf = _fast_load_npy(pitchf_path, max_rows=self.max_sample_frames)

        n_num = min(int(phone_raw.shape[0]) * 2, int(pitch.shape[0]), int(pitchf.shape[0]), self.max_sample_frames)
        n_phone_half = (n_num + 1) // 2
        phone_half = np.asarray(phone_raw[:n_phone_half, :], dtype=self.phone_dtype)
        pitch_arr = np.asarray(pitch[:n_num], dtype=np.int32)
        pitchf_arr = np.asarray(pitchf[:n_num], dtype=np.float32)
        try:
            sid = np.int32(int(dv_str))
        except ValueError:
            sid = np.int32(0)
        return phone_half, pitch_arr, pitchf_arr, sid, n_num

    def load_item_with_cached_labels(
        self,
        index: int,
        labels: tuple[np.ndarray, np.ndarray, np.ndarray, np.int32, int],
        *,
        allow_pcm16: bool = False,
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
        """Load WAV waveform and align with pre-loaded labels."""
        audiopath = self.audiopaths_and_text[index][0]
        phone_half, pitch, pitchf, sid, n_num = labels

        keep_i16 = bool(allow_pcm16 and not self.compute_spec)
        audio, sr = _fast_load_wav(audiopath, max_samples=n_num * self.hop_length, allow_int16=keep_i16)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if int(sr) != self.sample_rate:
            raise ValueError(f"{sr} SR doesn't match target {self.sample_rate} SR in '{audiopath}'.")

        n_frames_audio = min(len(audio) // self.hop_length, self.max_sample_frames)
        len_min = min(n_num, n_frames_audio)
        len_wav = len_min * self.hop_length
        if keep_i16 and audio.dtype == np.int16:
            wav = np.ascontiguousarray(audio[:len_wav], dtype=np.int16)[:, None]
        else:
            audio_f32 = np.asarray(audio[:len_wav], dtype=np.float32)
            wav = audio_f32[:, None]  # (T_wav, 1) NWC

        if len_min < n_num:
            phone_use = phone_half[: (len_min + 1) // 2, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]
        else:
            phone_use = phone_half

        phone = np.repeat(phone_use, 2, axis=0)[:len_min, :] if self.repeat_phone else phone_use
        spec = (
            spectrogram_np(np.ascontiguousarray(wav[:, 0], dtype=np.float32), self.filter_length, self.hop_length, self.win_length)
            if self.compute_spec
            else None
        )
        return spec, wav, phone, pitch, pitchf, sid

    def __getitem__(self, index: int) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
        return self.load_item_with_cached_labels(index, self.load_labels(index))


class BucketBatchSampler:
    """Length-bucketed batch sampler emitting (bucket_target_frames, [sample_indices]) pairs."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        boundaries: Sequence[int],
        *,
        shuffle: bool = True,
        max_frames: int | None = None,
        batch_multiple: int = 1,
        static_bucket_padding: bool = True,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.boundaries = list(boundaries)
        self.shuffle = bool(shuffle)
        self.max_frames = max_frames
        self.batch_multiple = max(1, int(batch_multiple))
        self.static_bucket_padding = bool(static_bucket_padding)
        self.epoch = 0
        self.start_batch = 0

        self.buckets, self.bucket_upper_bounds, self.bucket_batch_sizes, self.num_samples_per_bucket = (
            self._create_buckets()
        )
        self.allowed_frames: tuple[int, ...] = (
            tuple(sorted(set(self.bucket_upper_bounds))) if self.static_bucket_padding else ()
        )

    def unique_batch_shapes(self) -> list[tuple[int, int]]:
        """Return sorted unique (batch_size, target_frames) static bucket shapes."""
        if not self.static_bucket_padding:
            return []
        return sorted(set(zip(self.bucket_batch_sizes, self.bucket_upper_bounds)))

    def _bisect(self, x: int) -> int:
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] < x <= self.boundaries[i + 1]:
                return i
        return -1

    def _align_batch_size(self, raw_bs: int) -> int:
        if raw_bs <= self.batch_multiple:
            return self.batch_multiple
        return (raw_bs // self.batch_multiple) * self.batch_multiple

    def _create_buckets(self) -> tuple[list[list[int]], list[int], list[int], list[int]]:
        raw_buckets: list[list[int]] = [[] for _ in range(len(self.boundaries) - 1)]
        for idx, length in enumerate(self.lengths):
            b_idx = self._bisect(length)
            if b_idx != -1:
                raw_buckets[b_idx].append(idx)

        buckets: list[list[int]] = []
        upper_bounds: list[int] = []
        for i, b in enumerate(raw_buckets):
            if b:
                buckets.append(b)
                upper_bounds.append(int(self.boundaries[i + 1]))

        batch_sizes: list[int] = []
        num_samples_per_bucket: list[int] = []
        for b in buckets:
            if self.max_frames:
                longest = max(self.lengths[idx] for idx in b)
                bs = max(self.batch_multiple, min(self.batch_size, self.max_frames // max(1, longest)))
                bs = self._align_batch_size(bs)
            else:
                bs = self._align_batch_size(self.batch_size)
            batch_sizes.append(bs)
            rem = (bs - (len(b) % bs)) % bs
            num_samples_per_bucket.append(len(b) + rem)
        return buckets, upper_bounds, batch_sizes, num_samples_per_bucket

    def set_epoch(self, epoch: int, start_batch: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)

    def __len__(self) -> int:
        return sum(n // bs for n, bs in zip(self.num_samples_per_bucket, self.bucket_batch_sizes))

    def __iter__(self) -> Iterator[tuple[int | None, list[int]]]:
        rng = np.random.default_rng(self.epoch)
        batches: list[tuple[int | None, list[int]]] = []
        for i, bucket in enumerate(self.buckets):
            n_b = len(bucket)
            perm = rng.permutation(n_b).tolist() if self.shuffle else list(range(n_b))
            n_target = self.num_samples_per_bucket[i]
            rem = n_target - n_b
            perm = perm + perm * (rem // n_b) + perm[: (rem % n_b)]

            bs = self.bucket_batch_sizes[i]
            target_frames = self.bucket_upper_bounds[i] if self.static_bucket_padding else None
            for j in range(len(perm) // bs):
                batch_idx = [bucket[p] for p in perm[j * bs : (j + 1) * bs]]
                batches.append((target_frames, batch_idx))

        if self.shuffle and batches:
            order = rng.permutation(len(batches)).tolist()
            batches = [batches[k] for k in order]

        return iter(batches[self.start_batch :])


def _snap_frames_to_bucket(needed: int, target_frames: int | None, allowed_frames: Sequence[int] | None) -> int:
    if target_frames is None:
        return needed
    target = int(target_frames)
    if needed <= target:
        return target
    if allowed_frames:
        for bound in allowed_frames:
            if bound >= needed:
                return int(bound)
    return needed


def collate_nwc_batch(
    rows: Sequence[tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]],
    hop_length: int,
    target_frames: int | None = None,
    allowed_frames: Sequence[int] | None = None,
    phone_half: bool = False,
) -> tuple[np.ndarray, ...]:
    """Collate sample rows into NWC NumPy arrays with optional static XLA bucket padding."""
    has_spec = rows[0][0] is not None
    # Half-rate phoneme rows store 50Hz frames; pitch (index 3) always tracks full 100Hz frame count.
    len_idx = 3 if phone_half else 2
    sorted_rows = sorted(rows, key=lambda r: r[len_idx].shape[0], reverse=True)
    batch_size = len(sorted_rows)

    phone_lengths = np.asarray([r[len_idx].shape[0] for r in sorted_rows], dtype=np.int32)
    wave_lengths = np.asarray([r[1].shape[0] for r in sorted_rows], dtype=np.int32)
    spec_lengths = (
        np.asarray([r[0].shape[0] for r in sorted_rows], dtype=np.int32)
        if has_spec
        else phone_lengths.copy()
    )

    max_spec = int(spec_lengths.max())
    max_phone = int(phone_lengths.max())
    frames = _snap_frames_to_bucket(max(max_spec, max_phone), target_frames, allowed_frames)
    wave_frames = max(frames * int(hop_length), int(wave_lengths.max()))

    phone_dim = sorted_rows[0][2].shape[1]
    phone_frames = (frames + 1) // 2 if phone_half else frames

    phone_padded = np.empty((batch_size, phone_frames, phone_dim), dtype=sorted_rows[0][2].dtype)
    pitch_padded = np.empty((batch_size, frames), dtype=np.int32)
    pitchf_padded = np.empty((batch_size, frames), dtype=np.float32)
    wave_padded = np.empty((batch_size, wave_frames, 1), dtype=np.float32)
    sid = np.asarray([int(r[5]) for r in sorted_rows], dtype=np.int32)
    spec_padded = (
        np.empty((batch_size, frames, sorted_rows[0][0].shape[1]), dtype=np.float32)
        if has_spec
        else None
    )

    for i, (spec, wav, phone, pitch, pitchf, _) in enumerate(sorted_rows):
        w_len = wav.shape[0]
        p_len = phone.shape[0]
        pt_len = pitch.shape[0]
        pf_len = pitchf.shape[0]
        if has_spec and spec_padded is not None and spec is not None:
            s_len = spec.shape[0]
            spec_padded[i, :s_len, :] = spec
            if s_len < frames:
                spec_padded[i, s_len:, :] = 0.0
        if wav.dtype == np.int16:
            np.multiply(wav.astype(np.float32), _PCM16_INV, out=wave_padded[i, :w_len, :])
        else:
            wave_padded[i, :w_len, :] = wav
        if w_len < wave_frames:
            wave_padded[i, w_len:, :] = 0.0
        phone_padded[i, :p_len, :] = phone
        if p_len < phone_frames:
            phone_padded[i, p_len:, :] = 0.0
        pitch_padded[i, :pt_len] = pitch
        if pt_len < frames:
            pitch_padded[i, pt_len:] = 0
        pitchf_padded[i, :pf_len] = pitchf
        if pf_len < frames:
            pitchf_padded[i, pf_len:] = 0.0

    if has_spec and spec_padded is not None:
        return (
            phone_padded,
            phone_lengths,
            pitch_padded,
            pitchf_padded,
            spec_padded,
            spec_lengths,
            wave_padded,
            wave_lengths,
            sid,
        )
    return (
        phone_padded,
        phone_lengths,
        pitch_padded,
        pitchf_padded,
        spec_lengths,
        wave_padded,
        wave_lengths,
        sid,
    )


_PACK_TOC_STRUCT = struct.Struct("<qqiiiiiBBH")
_DISK_PACK_MIN_CLIPS = 64
_HAS_PREAD = hasattr(os, "pread")


def _dataset_pack_fingerprint(dataset: Any) -> dict[str, Any] | None:
    """Return metadata fingerprint describing dataset pack compatibility, or None if ineligible."""
    tf = getattr(dataset, "training_files", "")
    if not tf or not os.path.isfile(tf) or bool(getattr(dataset, "compute_spec", False)):
        return None
    try:
        st = os.stat(tf)
    except OSError:
        return None
    return {
        "version": 1,
        "n_items": len(dataset),
        "repeat_phone": bool(getattr(dataset, "repeat_phone", True)),
        "phone_dtype": np.dtype(getattr(dataset, "phone_dtype", np.float32)).name,
        "filelist_size": int(st.st_size),
        "filelist_mtime_ns": int(st.st_mtime_ns),
    }


def _pack_payload_nbytes(
    phone_rows: int,
    phone_dim: int,
    wav_len: int,
    frames_len: int,
    wav_is_int16: int,
    phone_is_f16: int,
) -> int:
    ph_item = 2 if phone_is_f16 else 4
    wv_item = 2 if wav_is_int16 else 4
    return (
        int(phone_rows) * int(phone_dim) * ph_item
        + int(wav_len) * wv_item
        + int(frames_len) * 8  # int32 pitch + float32 pitchf
    )


class _DatasetPackStore:
    """Append-only contiguous mmap/pread binary cache eliminating loose overlayfs small-file reads."""

    def __init__(self, base_path: str, fingerprint: Mapping[str, Any]) -> None:
        self.bin_path = f"{base_path}.pack.bin"
        self.toc_path = f"{base_path}.pack.toc"
        self.meta_path = f"{base_path}.pack.meta.json"
        self._fingerprint = dict(fingerprint)
        self._entries: dict[int, tuple[int, int, int, int, int, int, int, int]] = {}
        self._queued: set[int] = set()
        self._lock = threading.Lock()
        self._read_lock = threading.Lock()
        self._closed = False
        self._read_fd = -1
        self._append_offset = 0
        self._stop_sentinel = object()
        self._write_q: queue.Queue = queue.Queue(maxsize=512)

        valid_entries: dict[int, tuple[int, int, int, int, int, int, int, int]] = {}
        valid_toc_chunks: list[bytes] = []
        max_end = 0
        reuse = False
        n_items = int(fingerprint["n_items"])

        try:
            if (
                os.path.isfile(self.meta_path)
                and os.path.isfile(self.bin_path)
                and os.path.isfile(self.toc_path)
            ):
                with open(self.meta_path, "r", encoding="utf-8") as mf:
                    existing_meta = json.load(mf)
                if existing_meta == self._fingerprint:
                    reuse = True
                    bin_size = os.path.getsize(self.bin_path)
                    with open(self.toc_path, "rb") as tf:
                        toc_raw = tf.read()
                    usable = (len(toc_raw) // _PACK_TOC_STRUCT.size) * _PACK_TOC_STRUCT.size
                    need_rewrite = usable != len(toc_raw)
                    for pos in range(0, usable, _PACK_TOC_STRUCT.size):
                        rec_bytes = toc_raw[pos : pos + _PACK_TOC_STRUCT.size]
                        idx_i, off, pr, pd, wl, fl, sid_i, w16, p16, _ = _PACK_TOC_STRUCT.unpack(rec_bytes)
                        if idx_i < 0 or idx_i >= n_items or off < 0 or pr < 0 or pd <= 0 or wl < 0 or fl < 0:
                            need_rewrite = True
                            continue
                        nbytes = _pack_payload_nbytes(pr, pd, wl, fl, w16, p16)
                        if off + nbytes > bin_size:
                            need_rewrite = True
                            continue
                        valid_entries[idx_i] = (off, pr, pd, wl, fl, sid_i, w16, p16)
                        valid_toc_chunks.append(rec_bytes)
                        if off + nbytes > max_end:
                            max_end = off + nbytes
                    if need_rewrite:
                        with open(self.toc_path, "wb") as tf:
                            if valid_toc_chunks:
                                tf.write(b"".join(valid_toc_chunks))
                        with open(self.bin_path, "r+b") as bf:
                            bf.truncate(max_end)
        except Exception:
            reuse = False
            valid_entries.clear()
            max_end = 0

        if not reuse:
            for p in (self.bin_path, self.toc_path, self.meta_path):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except OSError:
                    pass
            with open(self.meta_path, "w", encoding="utf-8") as mf:
                json.dump(self._fingerprint, mf)
            with open(self.bin_path, "wb"):
                pass
            with open(self.toc_path, "wb"):
                pass

        self._entries = valid_entries
        self._append_offset = max_end
        self._read_fd = os.open(self.bin_path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
        self._write_bin = open(self.bin_path, "r+b")
        self._write_bin.seek(self._append_offset)
        self._write_toc = open(self.toc_path, "r+b")
        self._write_toc.seek(0, os.SEEK_END)
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def _read_exact_at(self, offset: int, nbytes: int) -> bytes | None:
        fd = self._read_fd
        if fd < 0 or self._closed:
            return None
        if _HAS_PREAD:
            chunks: list[bytes] = []
            got = 0
            while got < nbytes:
                try:
                    chunk = os.pread(fd, nbytes - got, offset + got)
                except OSError:
                    return None
                if not chunk:
                    return None
                chunks.append(chunk)
                got += len(chunk)
            return chunks[0] if len(chunks) == 1 else b"".join(chunks)

        with self._read_lock:
            fd = self._read_fd
            if fd < 0 or self._closed:
                return None
            try:
                os.lseek(fd, offset, os.SEEK_SET)
                chunks = []
                got = 0
                while got < nbytes:
                    chunk = os.read(fd, nbytes - got)
                    if not chunk:
                        return None
                    chunks.append(chunk)
                    got += len(chunk)
                return chunks[0] if len(chunks) == 1 else b"".join(chunks)
            except OSError:
                return None

    def read_item(self, idx: int) -> tuple[None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32] | None:
        entry = self._entries.get(idx)
        if entry is None or self._closed:
            return None
        off, pr, pd, wl, fl, sid_i, w16, p16 = entry
        total_bytes = _pack_payload_nbytes(pr, pd, wl, fl, w16, p16)
        buf = self._read_exact_at(off, total_bytes)
        if buf is None or len(buf) != total_bytes:
            return None

        ph_dt = np.float16 if p16 else np.float32
        wv_dt = np.int16 if w16 else np.float32
        ph_bytes = pr * pd * (2 if p16 else 4)
        wv_bytes = wl * (2 if w16 else 4)
        pt_bytes = fl * 4

        phone = np.array(np.frombuffer(buf, dtype=ph_dt, count=pr * pd, offset=0).reshape(pr, pd), copy=True)
        wav = np.array(np.frombuffer(buf, dtype=wv_dt, count=wl, offset=ph_bytes).reshape(wl, 1), copy=True)
        pitch = np.array(np.frombuffer(buf, dtype=np.int32, count=fl, offset=ph_bytes + wv_bytes), copy=True)
        pitchf = np.array(np.frombuffer(buf, dtype=np.float32, count=fl, offset=ph_bytes + wv_bytes + pt_bytes), copy=True)
        return None, wav, phone, pitch, pitchf, np.int32(sid_i)

    def enqueue(self, idx: int, item: tuple[Any, ...]) -> None:
        if self._closed or idx in self._entries:
            return
        with self._lock:
            if self._closed or idx in self._entries or idx in self._queued:
                return
            self._queued.add(idx)
        try:
            self._write_q.put_nowait((idx, item))
        except queue.Full:
            with self._lock:
                self._queued.discard(idx)

    def _writer_loop(self) -> None:
        while True:
            first = self._write_q.get()
            if first is self._stop_sentinel:
                break
            batch = [first]
            while len(batch) < 32:
                try:
                    nxt = self._write_q.get_nowait()
                except queue.Empty:
                    break
                if nxt is self._stop_sentinel:
                    self._flush_records(batch)
                    return
                batch.append(nxt)
            self._flush_records(batch)

    def _flush_records(self, batch: list[tuple[int, tuple[Any, ...]]]) -> None:
        data_chunks: list[bytes] = []
        toc_chunks: list[bytes] = []
        staged: list[tuple[int, tuple[int, int, int, int, int, int, int, int]]] = []
        cursor = self._append_offset

        for idx, item in batch:
            if idx in self._entries:
                with self._lock:
                    self._queued.discard(idx)
                continue
            _, wav, phone, pitch, pitchf, sid = item
            ph_c = np.ascontiguousarray(phone)
            wv_c = np.ascontiguousarray(wav)
            pt_c = np.ascontiguousarray(pitch, dtype=np.int32)
            pf_c = np.ascontiguousarray(pitchf, dtype=np.float32)

            p16 = 1 if ph_c.dtype == np.float16 else 0
            w16 = 1 if wv_c.dtype == np.int16 else 0
            if not p16 and ph_c.dtype != np.float32:
                ph_c = np.ascontiguousarray(ph_c, dtype=np.float32)
            if not w16 and wv_c.dtype != np.float32:
                wv_c = np.ascontiguousarray(wv_c, dtype=np.float32)

            pr, pd = int(ph_c.shape[0]), int(ph_c.shape[1])
            wl = int(wv_c.shape[0])
            fl = int(pt_c.shape[0])
            sid_i = int(sid)

            raw_bytes = ph_c.tobytes() + wv_c.tobytes() + pt_c.tobytes() + pf_c.tobytes()
            data_chunks.append(raw_bytes)
            toc_chunks.append(_PACK_TOC_STRUCT.pack(idx, cursor, pr, pd, wl, fl, sid_i, w16, p16, 0))
            staged.append((idx, (cursor, pr, pd, wl, fl, sid_i, w16, p16)))
            cursor += len(raw_bytes)

        if data_chunks:
            self._write_bin.write(b"".join(data_chunks))
            self._write_bin.flush()
            self._write_toc.write(b"".join(toc_chunks))
            self._write_toc.flush()
            self._append_offset = cursor
            with self._lock:
                for idx, entry in staged:
                    self._entries[idx] = entry
                    self._queued.discard(idx)
        else:
            with self._lock:
                for idx, _ in batch:
                    self._queued.discard(idx)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while True:
            try:
                self._write_q.put(self._stop_sentinel, timeout=0.05)
                break
            except queue.Full:
                try:
                    dropped = self._write_q.get_nowait()
                    if dropped is self._stop_sentinel:
                        break
                except queue.Empty:
                    pass
        self._writer_thread.join(timeout=5.0)
        try:
            self._write_bin.flush()
            self._write_bin.close()
        except OSError:
            pass
        try:
            self._write_toc.flush()
            self._write_toc.close()
        except OSError:
            pass
        with self._read_lock:
            if self._read_fd >= 0:
                try:
                    os.close(self._read_fd)
                except OSError:
                    pass
                self._read_fd = -1


class PrefetchBatchLoader:
    """Multi-process / threaded NumPy batch prefetcher + optional RAM/mmap pack cache for high TPU throughput."""

    def __init__(
        self,
        dataset: TextAudioDataset,
        sampler: BucketBatchSampler,
        hop_length: int,
        num_workers: int = 4,
        prefetch_batches: int = 8,
        cache_in_ram: bool = False,
        use_multiprocessing: bool | None = None,
        enable_disk_pack: bool | None = None,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.hop_length = int(hop_length)
        self.num_workers = max(0, int(num_workers))
        self.prefetch_batches = max(2, int(prefetch_batches))
        self.cache_in_ram = bool(cache_in_ram)
        self.allowed_frames = getattr(sampler, "allowed_frames", ())
        self.phone_half = not bool(getattr(dataset, "repeat_phone", True))
        self.use_multiprocessing = bool(use_multiprocessing) and (not self.cache_in_ram) and (self.num_workers > 0)
        self._cache: list[tuple[np.ndarray, ...]] | None = None
        avail = _available_ram_bytes()
        self._sample_cache: dict[int, tuple[Any, ...]] = {}
        self._sample_cache_bytes = 0
        self._sample_cache_lock = threading.Lock()
        self._sample_cache_limit = max(0, min(int(avail * 0.65), avail - 4 * (1024 ** 3)))
        self._label_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.int32, int]] = {}
        self._label_cache_bytes = 0
        self._label_cache_limit = max(0, min(int(avail * 0.72), avail - 3 * (1024 ** 3)))
        self._inflight: dict[int, Future] = {}
        self._warmup_stop = threading.Event()
        self._warmup_thread: threading.Thread | None = None
        self._closed = False
        self._mp_pool = None
        # Overlayfs random-read IOPS peaks at ~8 threads; higher thread counts degrade throughput.
        self._pool_workers = (
            min(8, max(4, self.num_workers))
            if self.num_workers > 0 and not self.use_multiprocessing
            else 0
        )
        self._pool = (
            ThreadPoolExecutor(max_workers=self._pool_workers)
            if self._pool_workers > 0
            else None
        )
        self._pack_store: _DatasetPackStore | None = None
        if not self.use_multiprocessing:
            fp = _dataset_pack_fingerprint(dataset)
            should_pack = (
                fp is not None
                and (bool(enable_disk_pack) if enable_disk_pack is not None else len(dataset) >= _DISK_PACK_MIN_CLIPS)
            )
            if should_pack and fp is not None:
                try:
                    self._pack_store = _DatasetPackStore(str(dataset.training_files), fp)
                except OSError:
                    self._pack_store = None

    def _get_item(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        cached = self._sample_cache.get(idx)
        if cached is not None:
            return cached
        if self._pack_store is not None:
            packed = self._pack_store.read_item(idx)
            if packed is not None:
                if self._sample_cache_limit > 0 and self._sample_cache_bytes < self._sample_cache_limit:
                    item_bytes = _sample_nbytes(packed)
                    with self._sample_cache_lock:
                        if idx not in self._sample_cache and self._sample_cache_bytes + item_bytes <= self._sample_cache_limit:
                            self._sample_cache[idx] = packed
                            self._sample_cache_bytes += item_bytes
                return packed

        cached_lbl = self._label_cache.get(idx)
        if cached_lbl is not None:
            return self.dataset.load_item_with_cached_labels(idx, cached_lbl, allow_pcm16=True)

        if hasattr(self.dataset, "load_labels"):
            labels = self.dataset.load_labels(idx)
            item = self.dataset.load_item_with_cached_labels(idx, labels, allow_pcm16=True)
            if self._pack_store is not None:
                self._pack_store.enqueue(idx, item)
            if (self._sample_cache_bytes + self._label_cache_bytes) < self._label_cache_limit:
                item_bytes = _sample_nbytes(item)
                lbl_bytes = _sample_nbytes(labels[:3])
                with self._sample_cache_lock:
                    if idx not in self._sample_cache and (
                        self._sample_cache_bytes + item_bytes <= self._sample_cache_limit
                    ):
                        self._sample_cache[idx] = item
                        self._sample_cache_bytes += item_bytes
                    elif idx not in self._label_cache and (
                        self._sample_cache_bytes + self._label_cache_bytes + lbl_bytes <= self._label_cache_limit
                    ):
                        self._label_cache[idx] = labels
                        self._label_cache_bytes += lbl_bytes
            return item

        item = self.dataset[idx]
        if self._sample_cache_limit > 0 and self._sample_cache_bytes < self._sample_cache_limit:
            item_bytes = _sample_nbytes(item)
            with self._sample_cache_lock:
                if idx not in self._sample_cache and self._sample_cache_bytes + item_bytes <= self._sample_cache_limit:
                    self._sample_cache[idx] = item
                    self._sample_cache_bytes += item_bytes
        return item

    def _load_sample_tracked(self, idx: int):
        try:
            return self._get_item(idx)
        finally:
            with self._sample_cache_lock:
                self._inflight.pop(idx, None)

    def _submit_sample(self, idx: int) -> tuple[bool, Any]:
        if self._cache is not None:
            return True, self._cache[idx]
        cached = self._sample_cache.get(idx)
        if cached is not None:
            return True, cached
        with self._sample_cache_lock:
            cached = self._sample_cache.get(idx)
            if cached is not None:
                return True, cached
            if self._closed or self._pool is None:
                return True, self._get_item(idx)
            fut = self._inflight.get(idx)
            if fut is None:
                fut = self._pool.submit(self._load_sample_tracked, idx)
                self._inflight[idx] = fut
            return False, fut

    def start_warmup(
        self,
        epoch: int = 1,
        start_batch: int = 0,
        max_samples: int | None = None,
    ) -> None:
        """Pre-populate sample/label RAM caches during XLA compile in exact Epoch batch order."""
        if (
            self._sample_cache_limit <= 0
            or self.cache_in_ram
            or self.use_multiprocessing
            or self._pool is None
            or self._closed
        ):
            return
        if self._warmup_thread is not None and self._warmup_thread.is_alive():
            return
        self.sampler.set_epoch(epoch, start_batch)
        specs = list(iter(self.sampler))
        if not specs:
            return

        order: list[int] = []
        seen: set[int] = set()
        limit = int(max_samples) if max_samples is not None else len(self.dataset)
        for _, indices in specs:
            for raw_idx in indices:
                idx_i = int(raw_idx)
                if idx_i not in seen:
                    seen.add(idx_i)
                    order.append(idx_i)
                    if len(order) >= limit:
                        break
            if len(order) >= limit:
                break
        if not order:
            return

        self._warmup_stop.clear()

        def _warmup_loop() -> None:
            in_flight: deque[Future] = deque()
            max_pending = max(4, min(24, self._pool_workers * 2))
            for idx_i in order:
                if self._warmup_stop.is_set() or self._closed:
                    break
                if self._sample_cache_bytes >= int(self._sample_cache_limit * 0.90):
                    break
                ready, item = self._submit_sample(idx_i)
                if not ready:
                    in_flight.append(item)
                while len(in_flight) >= max_pending:
                    fut = in_flight.popleft()
                    try:
                        fut.result()
                    except BaseException:
                        pass
            while in_flight and not self._warmup_stop.is_set() and not self._closed:
                fut = in_flight.popleft()
                try:
                    fut.result()
                except BaseException:
                    pass

        self._warmup_thread = threading.Thread(target=_warmup_loop, daemon=True)
        self._warmup_thread.start()

    def stop_warmup(self, wait: bool = False) -> None:
        self._warmup_stop.set()
        if wait and self._warmup_thread is not None:
            self._warmup_thread.join(timeout=2.0)
            self._warmup_thread = None

    def _prepare_batch(self, spec_pair: tuple[int | None, list[int]]):
        target_frames, indices = spec_pair
        rows = [self._get_item(i) for i in indices]
        return collate_nwc_batch(
            rows,
            self.hop_length,
            target_frames=target_frames,
            allowed_frames=self.allowed_frames,
            phone_half=self.phone_half,
        )

    def _load_for_cache(self, idx: int):
        if hasattr(self.dataset, "load_labels"):
            return self.dataset.load_item_with_cached_labels(idx, self.dataset.load_labels(idx), allow_pcm16=True)
        return self.dataset[idx]

    def _ensure_cache(self) -> None:
        if not self.cache_in_ram or self._cache is not None:
            return
        indices = list(range(len(self.dataset)))
        print(f"Caching {len(indices)} training samples into host RAM...", flush=True)
        if self._pool is not None:
            self._cache = list(self._pool.map(self._load_for_cache, indices))
        else:
            self._cache = [self._load_for_cache(i) for i in indices]

    def __iter__(self):
        self._ensure_cache()
        specs = list(iter(self.sampler))
        if not specs:
            return

        if self.use_multiprocessing and not self.cache_in_ram and self.num_workers > 0:
            if self._mp_pool is None:
                from rvc.train.jax_data_worker import SpawnBatchPool

                cfg_dict = self.dataset.worker_config(self.hop_length, self.allowed_frames)
                self._mp_pool = SpawnBatchPool(
                    cfg_dict,
                    num_workers=self.num_workers,
                    prefetch_batches=self.prefetch_batches,
                )
            yield from self._mp_pool.iter_batches(specs)
            return

        if self._pool is None or self.num_workers == 0:
            for sp in specs:
                yield self._prepare_batch(sp)
            return

        self.stop_warmup(wait=False)

        out_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        sentinel = object()
        max_sample_budget = max(192, self._pool_workers * 24)
        batch_window = max(self.prefetch_batches, 6)

        def _producer() -> None:
            pending_batches: deque[tuple[int | None, list[tuple[bool, Any]]]] = deque()
            spec_iter = iter(specs)
            in_flight_futs = 0
            try:
                while True:
                    while len(pending_batches) < batch_window and (
                        not pending_batches or in_flight_futs < max_sample_budget
                    ):
                        sp = next(spec_iter, None)
                        if sp is None:
                            break
                        target_frames, indices = sp
                        slots = [self._submit_sample(int(i)) for i in indices]
                        in_flight_futs += sum(1 for ready, _ in slots if not ready)
                        pending_batches.append((target_frames, slots))

                    if not pending_batches or stop_event.is_set():
                        break

                    target_frames, slots = pending_batches.popleft()
                    rows = []
                    for ready, item in slots:
                        if ready:
                            rows.append(item)
                        else:
                            rows.append(item.result())
                            in_flight_futs -= 1
                    if stop_event.is_set():
                        break
                    batch = collate_nwc_batch(
                        rows,
                        self.hop_length,
                        target_frames=target_frames,
                        allowed_frames=self.allowed_frames,
                        phone_half=self.phone_half,
                    )
                    while not stop_event.is_set():
                        try:
                            out_queue.put(batch, timeout=0.05)
                            break
                        except queue.Full:
                            continue
            except BaseException as exc:
                while not stop_event.is_set():
                    try:
                        out_queue.put(exc, timeout=0.05)
                        break
                    except queue.Full:
                        continue
            finally:
                while not stop_event.is_set():
                    try:
                        out_queue.put(sentinel, timeout=0.05)
                        break
                    except queue.Full:
                        continue

        thread = threading.Thread(target=_producer, daemon=True)
        thread.start()
        try:
            while True:
                item = out_queue.get()
                if item is sentinel:
                    break
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop_event.set()
            while True:
                try:
                    if out_queue.get_nowait() is sentinel:
                        break
                except queue.Empty:
                    break
            thread.join(timeout=2.0)

    def close(self) -> None:
        self.stop_warmup(wait=True)
        with self._sample_cache_lock:
            self._closed = True
            self._inflight.clear()
        self._sample_cache.clear()
        self._sample_cache_bytes = 0
        self._label_cache.clear()
        self._label_cache_bytes = 0
        if self._pack_store is not None:
            self._pack_store.close()
            self._pack_store = None
        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool = None
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
