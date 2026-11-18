"""Pure-NumPy NWC dataset loader, bucket sampler, and threaded prefetcher for JAX training.

This module intentionally avoids importing `jax` or `torch` so worker threads/subprocesses
never touch TPU/PJRT or GPU contexts.
"""

from __future__ import annotations

import math
import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Sequence

import numpy as np
import soundfile as sf


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

    def __init__(self, data_cfg, *, compute_spec: bool = True) -> None:
        self.audiopaths_and_text = load_filepaths_and_text(data_cfg.training_files)
        self.max_wav_value = float(data_cfg.max_wav_value)
        self.sample_rate = int(data_cfg.sample_rate)
        self.filter_length = int(data_cfg.filter_length)
        self.hop_length = int(data_cfg.hop_length)
        self.win_length = int(data_cfg.win_length)
        self.max_sample_frames = 896
        self.compute_spec = bool(compute_spec)
        self.min_text_len = int(getattr(data_cfg, "min_text_len", 1))
        self.max_text_len = int(getattr(data_cfg, "max_text_len", 5000))
        self._filter()

    def _filter(self) -> None:
        kept: list[list[str]] = []
        for entry in self.audiopaths_and_text:
            audiopath, text, pitch, pitchf, dv = entry[:5]
            if self.min_text_len <= len(text) <= self.max_text_len:
                kept.append([audiopath, text, pitch, pitchf, dv])

        if len(kept) > 32:
            workers = min(16, max(4, os.cpu_count() or 4))
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

        phone = np.load(phone_path, allow_pickle=False)
        phone = np.repeat(phone, 2, axis=0)
        pitch = np.load(pitch_path, allow_pickle=False)
        pitchf = np.load(pitchf_path, allow_pickle=False)

        n_num = min(phone.shape[0], pitch.shape[0], pitchf.shape[0], self.max_sample_frames)
        phone = np.ascontiguousarray(phone[:n_num, :], dtype=np.float32)
        pitch = np.ascontiguousarray(pitch[:n_num], dtype=np.int32)
        pitchf = np.ascontiguousarray(pitchf[:n_num], dtype=np.float32)

        audio, sr = sf.read(audiopath, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        if int(sr) != self.sample_rate:
            raise ValueError(f"{sr} SR doesn't match target {self.sample_rate} SR in '{audiopath}'.")
        audio = np.ascontiguousarray(audio[: self.max_sample_frames * self.hop_length], dtype=np.float32)

        n_frames_audio = len(audio) // self.hop_length
        len_min = min(phone.shape[0], n_frames_audio)
        len_wav = len_min * self.hop_length
        audio = np.ascontiguousarray(audio[:len_wav], dtype=np.float32)
        wav = audio[:, None]  # (T_wav, 1) NWC
        if len_min < n_num:
            phone = np.ascontiguousarray(phone[:len_min, :], dtype=np.float32)
            pitch = np.ascontiguousarray(pitch[:len_min], dtype=np.int32)
            pitchf = np.ascontiguousarray(pitchf[:len_min], dtype=np.float32)

        spec = (
            spectrogram_np(audio, self.filter_length, self.hop_length, self.win_length)
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
) -> tuple[np.ndarray, ...]:
    """Collate sample rows into NWC NumPy arrays with optional static XLA bucket padding."""
    has_spec = rows[0][0] is not None
    sorted_rows = sorted(rows, key=lambda r: r[2].shape[0], reverse=True)
    batch_size = len(sorted_rows)

    phone_lengths = np.asarray([r[2].shape[0] for r in sorted_rows], dtype=np.int32)
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

    phone_padded = np.zeros((batch_size, frames, phone_dim), dtype=np.float32)
    pitch_padded = np.zeros((batch_size, frames), dtype=np.int32)
    pitchf_padded = np.zeros((batch_size, frames), dtype=np.float32)
    wave_padded = np.zeros((batch_size, wave_frames, 1), dtype=np.float32)
    sid = np.asarray([int(r[5]) for r in sorted_rows], dtype=np.int32)
    spec_padded = (
        np.zeros((batch_size, frames, sorted_rows[0][0].shape[1]), dtype=np.float32)
        if has_spec
        else None
    )

    for i, (spec, wav, phone, pitch, pitchf, _) in enumerate(sorted_rows):
        w_len = wav.shape[0]
        p_len = phone.shape[0]
        if has_spec and spec_padded is not None and spec is not None:
            spec_padded[i, : spec.shape[0], :] = spec
        wave_padded[i, :w_len, :] = wav
        phone_padded[i, :p_len, :] = phone
        pitch_padded[i, : pitch.shape[0]] = pitch
        pitchf_padded[i, : pitchf.shape[0]] = pitchf

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
    """Threaded NumPy batch prefetcher + optional RAM dataset cache for high TPU throughput."""

    def __init__(
        self,
        dataset: TextAudioDataset,
        sampler: BucketBatchSampler,
        hop_length: int,
        num_workers: int = 4,
        prefetch_batches: int = 8,
        cache_in_ram: bool = False,
    ) -> None:
        self.dataset = dataset
        self.sampler = sampler
        self.hop_length = int(hop_length)
        self.num_workers = max(0, int(num_workers))
        self.prefetch_batches = max(2, int(prefetch_batches))
        self.cache_in_ram = bool(cache_in_ram)
        self.allowed_frames = getattr(sampler, "allowed_frames", ())
        self._cache: list[tuple[np.ndarray, ...]] | None = None
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.num_workers)) if self.num_workers > 0 else None

    def _get_item(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        return self.dataset[idx]

    def _prepare_batch(self, spec_pair: tuple[int | None, list[int]]):
        target_frames, indices = spec_pair
        rows = [self._get_item(i) for i in indices]
        return collate_nwc_batch(
            rows,
            self.hop_length,
            target_frames=target_frames,
            allowed_frames=self.allowed_frames,
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

        if self._pool is None or self.num_workers == 0:
            for sp in specs:
                yield self._prepare_batch(sp)
            return

        out_queue: queue.Queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_event = threading.Event()
        sentinel = object()
        max_in_flight = max(self.prefetch_batches * 2, self.num_workers * 2)

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
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None
