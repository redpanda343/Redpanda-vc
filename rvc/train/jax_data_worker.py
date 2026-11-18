"""Isolated spawn-safe NumPy batch loader workers for JAX training (zero JAX/TPU or PyTorch imports)."""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import queue
import threading
from typing import Any

_DATA_MOD: Any = None


def _load_data_module() -> Any:
    global _DATA_MOD
    if _DATA_MOD is None:
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "jax",
            "data.py",
        )
        spec = importlib.util.spec_from_file_location("_redpanda_jax_data_worker", data_path)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _DATA_MOD = mod
    return _DATA_MOD


def _worker_main(cfg_dict: dict[str, Any], shared_gen: Any, task_q: Any, res_q: Any) -> None:
    """Decode and collate NWC training batches in a clean child process free of CPython GIL."""
    data_mod = _load_data_module()
    ds = data_mod.TextAudioDataset.from_cached_entries(cfg_dict)
    hop_length = int(cfg_dict["hop_length"])
    allowed_frames = tuple(cfg_dict.get("allowed_frames") or ())

    while True:
        item = task_q.get()
        if item is None:
            break
        gen, target_frames, indices = item
        if gen != shared_gen.value:
            continue
        try:
            rows = [ds[i] for i in indices]
            if gen != shared_gen.value:
                continue
            batch = data_mod.collate_nwc_batch(
                rows,
                hop_length,
                target_frames=target_frames,
                allowed_frames=allowed_frames,
            )
            res_q.put((gen, True, batch))
        except BaseException as exc:
            res_q.put((gen, False, f"{type(exc).__name__}: {exc}"))


class SpawnBatchPool:
    """Persistent spawned worker pool streaming collated NumPy batches across epochs."""

    def __init__(self, cfg_dict: dict[str, Any], num_workers: int, prefetch_batches: int) -> None:
        self._num_workers = max(1, int(num_workers))
        self._prefetch = max(2, int(prefetch_batches))
        ctx = mp.get_context("spawn")
        self._shared_gen = ctx.Value("i", 0)
        q_cap = max(16, self._prefetch * 3)
        self._task_q = ctx.Queue(maxsize=q_cap)
        self._res_q = ctx.Queue(maxsize=q_cap)
        self._procs = [
            ctx.Process(
                target=_worker_main,
                args=(cfg_dict, self._shared_gen, self._task_q, self._res_q),
                daemon=True,
                name=f"jax-data-{idx}",
            )
            for idx in range(self._num_workers)
        ]
        for proc in self._procs:
            proc.start()

    def iter_batches(self, specs: list[tuple[int | None, list[int]]]):
        if not specs:
            return
        self._shared_gen.value += 1
        cur_gen = int(self._shared_gen.value)
        total = len(specs)
        stop_event = threading.Event()
        sem = threading.Semaphore(self._prefetch)

        def _feed() -> None:
            for sp in specs:
                if stop_event.is_set():
                    break
                while not sem.acquire(timeout=0.1):
                    if stop_event.is_set():
                        return
                if stop_event.is_set():
                    break
                target_frames, indices = sp
                while not stop_event.is_set():
                    try:
                        self._task_q.put((cur_gen, target_frames, indices), timeout=0.1)
                        break
                    except queue.Full:
                        continue

        feeder = threading.Thread(target=_feed, daemon=True)
        feeder.start()
        delivered = 0
        try:
            while delivered < total:
                try:
                    gen, ok, payload = self._res_q.get(timeout=0.5)
                except queue.Empty:
                    if not any(proc.is_alive() for proc in self._procs):
                        raise RuntimeError("JAX data loader worker processes exited unexpectedly.")
                    continue
                if gen != cur_gen:
                    continue
                sem.release()
                delivered += 1
                if not ok:
                    raise RuntimeError(f"Data worker failed: {payload}")
                yield payload
        finally:
            stop_event.set()
            # Advance generation so in-flight workers drop cancelled work without decoding.
            self._shared_gen.value += 1
            while True:
                try:
                    self._res_q.get_nowait()
                except queue.Empty:
                    break
            feeder.join(timeout=1.0)

    def close(self) -> None:
        if not self._procs:
            return
        self._shared_gen.value += 1
        for _ in self._procs:
            try:
                self._task_q.put_nowait(None)
            except queue.Full:
                pass
        # Drain any queued worker results so children blocked on res_q.put can exit cleanly.
        while True:
            try:
                self._res_q.get_nowait()
            except queue.Empty:
                break
        for proc in self._procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)
        self._procs.clear()
        for q in (self._task_q, self._res_q):
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
