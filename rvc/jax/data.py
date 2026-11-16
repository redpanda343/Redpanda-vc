"""Pure-NumPy NWC dataset loader, bucket sampler, and threaded prefetcher for JAX training.

This module intentionally avoids importing `jax` or `torch` so worker threads/subprocesses
never touch TPU/PJRT or GPU contexts.
"""

from __future__ import annotations

import math
import os
import queue
import threading
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


class TextAudioDataset:
    """Loads WAV audio, linear STFT, phoneme features, and pitch for NWC JAX training."""

    def __init__(self, data_cfg) -> None:
        self.audiopaths_and_text = load_filepaths_and_text(data_cfg.training_files)
        self.max_wav_value = float(data_cfg.max_wav_value)
        self.sample_rate = int(data_cfg.sample_rate)
        self.filter_length = int(data_cfg.filter_length)
        self.hop_length = int(data_cfg.hop_length)
        self.win_length = int(data_cfg.win_length)
        self.max_sample_frames = 900
        self.min_text_len = int(getattr(data_cfg, "min_text_len", 1))
        self.max_text_len = int(getattr(data_cfg, "max_text_len", 5000))
        self._filter()

    def _filter(self) -> None:
        kept: list[list[str]] = []
        lengths: list[int] = []
        for entry in self.audiopaths_and_text:
            audiopath, text, pitch, pitchf, dv = entry[:5]
            if self.min_text_len <= len(text) <= self.max_text_len:
                kept.append([audiopath, text, pitch, pitchf, dv])
                info = sf.info(audiopath)
                if info.format == "FLAC" or info.subtype in {"FLOAT", "PCM_16"}:
                    length = info.frames // self.hop_length
                else:
                    length = os.path.getsize(audiopath) // (3 * self.hop_length)
                lengths.append(min(int(length), self.max_sample_frames))
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
        spec = spectrogram_np(audio, self.filter_length, self.hop_length, self.win_length)
        wav = audio[:, None]  # (T_wav, 1) NWC

        len_phone = phone.shape[0]
        len_spec = spec.shape[0]
        if len_phone != len_spec:
            len_min = min(len_phone, len_spec)
            len_wav = len_min * self.hop_length
            spec = spec[:len_min, :]
            wav = wav[:len_wav, :]
            phone = phone[:len_min, :]
            pitch = pitch[:len_min]
            pitchf = pitchf[:len_min]

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


def collate_nwc_batch(
    rows: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.int32]],
    hop_length: int,
    target_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collate sample rows into NWC NumPy arrays with optional static XLA bucket padding."""
    sorted_rows = sorted(rows, key=lambda r: r[0].shape[0], reverse=True)
    batch_size = len(sorted_rows)

    spec_lengths = np.asarray([r[0].shape[0] for r in sorted_rows], dtype=np.int32)
    wave_lengths = np.asarray([r[1].shape[0] for r in sorted_rows], dtype=np.int32)
    phone_lengths = np.asarray([r[2].shape[0] for r in sorted_rows], dtype=np.int32)

    max_spec = int(spec_lengths.max())
    max_phone = int(phone_lengths.max())
    frames = max(int(target_frames), max_spec, max_phone) if target_frames is not None else max(max_spec, max_phone)
    wave_frames = max(frames * int(hop_length), int(wave_lengths.max()))

    spec_bins = sorted_rows[0][0].shape[1]
    phone_dim = sorted_rows[0][2].shape[1]

    phone_padded = np.zeros((batch_size, frames, phone_dim), dtype=np.float32)
    pitch_padded = np.zeros((batch_size, frames), dtype=np.int32)
    pitchf_padded = np.zeros((batch_size, frames), dtype=np.float32)
    spec_padded = np.zeros((batch_size, frames, spec_bins), dtype=np.float32)
    wave_padded = np.zeros((batch_size, wave_frames, 1), dtype=np.float32)
    sid = np.asarray([int(r[5]) for r in sorted_rows], dtype=np.int32)

    for i, (spec, wav, phone, pitch, pitchf, _) in enumerate(sorted_rows):
        s_len = spec.shape[0]
        w_len = wav.shape[0]
        p_len = phone.shape[0]
        spec_padded[i, :s_len, :] = spec
        wave_padded[i, :w_len, :] = wav
        phone_padded[i, :p_len, :] = phone
        pitch_padded[i, : pitch.shape[0]] = pitch
        pitchf_padded[i, : pitchf.shape[0]] = pitchf

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
        self._cache: list[tuple[np.ndarray, ...]] | None = None
        self._pool = ThreadPoolExecutor(max_workers=max(1, self.num_workers)) if self.num_workers > 0 else None

    def _get_item(self, idx: int):
        if self._cache is not None:
            return self._cache[idx]
        return self.dataset[idx]

    def _prepare_batch(self, spec_pair: tuple[int | None, list[int]]):
        target_frames, indices = spec_pair
        rows = [self._get_item(i) for i in indices]
        return collate_nwc_batch(rows, self.hop_length, target_frames=target_frames)

    def _ensure_cache(self) -> None:
        if not self.cache_in_ram or self._cache is not None:
            return
        indices = list(range(len(self.dataset)))
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

        def _producer() -> None:
            try:
                futures = [self._pool.submit(self._prepare_batch, sp) for sp in specs]
                for fut in futures:
                    if stop_event.is_set():
                        break
                    out_queue.put(fut.result())
            except BaseException as exc:
                out_queue.put(exc)
            finally:
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
            thread.join(timeout=2.0)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=False)
            self._pool = None
