"""Pure-NumPy NWC dataset loader, bucket sampler, and threaded prefetcher for JAX training.

This module intentionally avoids importing `jax` or `torch` so worker threads/subprocesses
never touch TPU/PJRT or GPU contexts.
"""

from __future__ import annotations

import ctypes
import math
import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

    def __init__(self, data_cfg, *, compute_spec: bool = True, repeat_phone: bool = True) -> None:
        self.audiopaths_and_text = load_filepaths_and_text(data_cfg.training_files)
        self.max_wav_value = float(data_cfg.max_wav_value)
        self.sample_rate = int(data_cfg.sample_rate)
        self.filter_length = int(data_cfg.filter_length)
        self.hop_length = int(data_cfg.hop_length)
        self.win_length = int(data_cfg.win_length)
        self.max_sample_frames = 896
        self.compute_spec = bool(compute_spec)
        self.repeat_phone = bool(repeat_phone)
        self.min_text_len = int(getattr(data_cfg, "min_text_len", 1))
        self.max_text_len = int(getattr(data_cfg, "max_text_len", 5000))
        self._filter()

    @classmethod
    def from_cached_entries(cls, cfg_dict: Mapping[str, Any]) -> "TextAudioDataset":
        """Reconstruct dataset in worker processes using pre-filtered file lists and lengths."""
        obj = cls.__new__(cls)
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
        obj.min_text_len = int(cfg_dict.get("min_text_len", 1))
        obj.max_text_len = int(cfg_dict.get("max_text_len", 5000))
        return obj

    def worker_config(self, hop_length: int, allowed_frames: Sequence[int] = ()) -> dict[str, Any]:
        """Serialize picklable dataset metadata for multi-process batch workers."""
        return {
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

        if len(kept) > 32:
            workers = min(8, max(2, effective_cpu_count()))
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

    def __len__(self) -> int:
        return len(self.audiopaths_and_text)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]:
        audiopath, phone_path, pitch_path, pitchf_path, dv_str = self.audiopaths_and_text[index]

        phone_raw = np.load(phone_path, allow_pickle=False)
        pitch = np.load(pitch_path, allow_pickle=False)
        pitchf = np.load(pitchf_path, allow_pickle=False)

        n_num = min(int(phone_raw.shape[0]) * 2, int(pitch.shape[0]), int(pitchf.shape[0]), self.max_sample_frames)
        audio, sr = sf.read(audiopath, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if int(sr) != self.sample_rate:
            raise ValueError(f"{sr} SR doesn't match target {self.sample_rate} SR in '{audiopath}'.")

        n_frames_audio = min(len(audio) // self.hop_length, self.max_sample_frames)
        len_min = min(n_num, n_frames_audio)
        len_wav = len_min * self.hop_length
        audio = np.asarray(audio[:len_wav], dtype=np.float32)
        wav = audio[:, None]  # (T_wav, 1) NWC

        n_phone_half = (len_min + 1) // 2
        phone_half = np.asarray(phone_raw[:n_phone_half, :], dtype=np.float32)
        phone = np.repeat(phone_half, 2, axis=0)[:len_min, :] if self.repeat_phone else phone_half
        pitch = np.asarray(pitch[:len_min], dtype=np.int32)
        pitchf = np.asarray(pitchf[:len_min], dtype=np.float32)

        spec = (
            spectrogram_np(np.ascontiguousarray(audio, dtype=np.float32), self.filter_length, self.hop_length, self.win_length)
            if self.compute_spec
            else None
        )

        try:
            sid = np.int32(int(dv_str))
        except ValueError:
            sid = np.int32(0)

        return spec, wav, phone, pitch, pitchf, sid


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

    phone_padded = np.empty((batch_size, phone_frames, phone_dim), dtype=np.float32)
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


class PrefetchBatchLoader:
    """Multi-process / threaded NumPy batch prefetcher + optional RAM cache for high TPU throughput."""

    def __init__(
        self,
        dataset: TextAudioDataset,
        sampler: BucketBatchSampler,
        hop_length: int,
        num_workers: int = 4,
        prefetch_batches: int = 8,
        cache_in_ram: bool = False,
        use_multiprocessing: bool | None = None,
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
        self._sample_cache_limit = max(0, min(int(avail * 0.35), avail - 3 * (1024 ** 3)))
        self._mp_pool = None
        self._pool = (
            ThreadPoolExecutor(max_workers=max(1, self.num_workers))
            if self.num_workers > 0 and not self.use_multiprocessing
            else None
        )

    def _get_item(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        cached = self._sample_cache.get(idx)
        if cached is not None:
            return cached
        item = self.dataset[idx]
        if self._sample_cache_limit > 0 and self._sample_cache_bytes < self._sample_cache_limit:
            item_bytes = _sample_nbytes(item)
            with self._sample_cache_lock:
                if idx not in self._sample_cache and self._sample_cache_bytes + item_bytes <= self._sample_cache_limit:
                    self._sample_cache[idx] = item
                    self._sample_cache_bytes += item_bytes
        return item

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

    def _ensure_cache(self) -> None:
        if not self.cache_in_ram or self._cache is not None:
            return
        indices = list(range(len(self.dataset)))
        print(f"Caching {len(indices)} training samples into host RAM...", flush=True)
        if self._pool is not None:
            self._cache = list(self._pool.map(self.dataset.__getitem__, indices))
        else:
            self._cache = [self.dataset[i] for i in indices]

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

        out_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        sentinel = object()
        max_in_flight = self.prefetch_batches

        def _producer() -> None:
            pending: deque = deque()
            spec_iter = iter(specs)
            try:
                while len(pending) < max_in_flight:
                    sp = next(spec_iter, None)
                    if sp is None:
                        break
                    pending.append(self._pool.submit(self._prepare_batch, sp))

                while pending and not stop_event.is_set():
                    fut = pending.popleft()
                    res = fut.result()
                    if stop_event.is_set():
                        break
                    out_queue.put(res)
                    sp = next(spec_iter, None)
                    if sp is not None and not stop_event.is_set():
                        pending.append(self._pool.submit(self._prepare_batch, sp))
            except BaseException as exc:
                out_queue.put(exc)
            finally:
                for fut in pending:
                    fut.cancel()
                out_queue.put(sentinel)

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
        self._sample_cache.clear()
        self._sample_cache_bytes = 0
        if self._mp_pool is not None:
            self._mp_pool.close()
            self._mp_pool = None
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
