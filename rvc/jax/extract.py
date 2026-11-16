"""Pipelined multi-chip JAX/TPU extraction for RMVPE F0 and ContentVec/Spin-v2 features."""

from __future__ import annotations

import math
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from rvc.jax import PrecisionConfig, resolve_jax_devices, supports_bfloat16
from rvc.jax.hubert import HubertConfig, hubert_forward, load_hubert_from_dir
from rvc.jax.rmvpe import (
    HOP_LENGTH as RMVPE_HOP,
    rmvpe_f0_forward,
    rmvpe_params_from_torch_file,
)
from shared.utils import (
    compute_coarse_f0,
    ensure_embedding_files,
    load_audio_16k,
    readable_npy,
    save_npy,
)

# Exact RMVPE bucket lengths: T_wav = 5120 * K - 160 produces T_mel = 32 * K frames (multiple of 32).
_BUCKET_KS = (4, 7, 10, 15, 20, 30)
_RMVPE_SEG_SAMPLES = 32 * RMVPE_HOP  # 5120


def jax_extraction_available(accelerator: str = "auto") -> bool:
    """Return True if JAX TPU (or GPU when explicitly requested) extraction is available."""
    mode = os.environ.get("REDPANDA_EXTRACT_BACKEND", "auto").strip().lower()
    if mode == "torch":
        return False
    try:
        devs = resolve_jax_devices(accelerator, "all")
        if mode == "jax":
            return len(devs) > 0
        return bool(devs and devs[0].platform == "tpu")
    except Exception:
        return False


def _estimate_16k_samples(wav_path: str) -> int:
    try:
        info = sf.info(wav_path)
        if info.frames > 0 and info.samplerate > 0:
            return int(math.ceil(info.frames * 16000.0 / float(info.samplerate))) + 64
    except Exception:
        pass
    # Conservative fallback for headers without frame counts (assume ~16k PCM bytes heuristic)
    return max(16000, os.path.getsize(wav_path) // 2 + 64)


def _bucket_length_for_samples(n_samples: int) -> int:
    for k in _BUCKET_KS:
        cap = _RMVPE_SEG_SAMPLES * k - RMVPE_HOP
        if n_samples <= cap:
            return cap
    k_dyn = max(1, math.ceil((n_samples + RMVPE_HOP) / _RMVPE_SEG_SAMPLES))
    return _RMVPE_SEG_SAMPLES * k_dyn - RMVPE_HOP


def _batch_per_device_for_bucket(bucket_samples: int) -> int:
    if bucket_samples <= 35680:
        return 32
    if bucket_samples <= 51040:
        return 24
    if bucket_samples <= 102240:
        return 12
    return 8


def _snap_per_device_batch(n_items: int, num_devices: int, max_per_dev: int) -> int:
    per_dev_needed = max(1, (n_items + num_devices - 1) // num_devices)
    if per_dev_needed >= max_per_dev:
        return max_per_dev
    for tier in (1, 2, 4, 8, 12, 16, 24, 32):
        if per_dev_needed <= tier <= max_per_dev:
            return tier
    return max_per_dev


def _decode_one(item: tuple[int, Sequence[str], bool, bool]) -> tuple[int, Sequence[str], np.ndarray, bool, bool]:
    idx, file_info, do_f0, do_feat = item
    audio = load_audio_16k(file_info[0])
    return idx, file_info, audio, do_f0, do_feat


def _write_outputs(
    tasks_meta: list[tuple[Sequence[str], int, bool, bool]],
    f0_batch: np.ndarray | None,
    feats_batch: np.ndarray | None,
    hubert_cfg: HubertConfig | None,
    expected_dim: int | None,
    updates: Any,
    progress_fallback: Any,
) -> None:
    count_updates = 0
    for b_idx, (file_info, wav_len, do_f0, do_feat) in enumerate(tasks_meta):
        wav_path, opt_coarse, opt_full, feat_path = file_info[:4]
        if do_f0 and f0_batch is not None:
            n_frames = wav_len // RMVPE_HOP + 1
            f0 = np.ascontiguousarray(f0_batch[b_idx, :n_frames], dtype=np.float64)
            if f0.ndim != 1 or not f0.size or not np.isfinite(f0).all() or np.any(f0 < 0):
                raise ValueError(f"RMVPE returned invalid F0 contour for {wav_path}.")
            coarse = compute_coarse_f0(f0)
            save_npy(opt_full, f0)
            save_npy(opt_coarse, coarse)
            count_updates += 1

        if do_feat and feats_batch is not None and hubert_cfg is not None:
            n_feat = int(hubert_cfg.output_length(wav_len))
            feats = np.ascontiguousarray(feats_batch[b_idx, :n_feat], dtype=np.float32)
            if feats.ndim != 2 or (expected_dim is not None and feats.shape[1] != expected_dim):
                raise RuntimeError(f"{wav_path} produced shape {feats.shape}; expected [frames, {expected_dim}].")
            if not np.isfinite(feats).all():
                raise RuntimeError(f"{wav_path} produced non-finite embedding values.")
            save_npy(feat_path, feats)
            count_updates += 1

    if count_updates:
        if updates is not None:
            for _ in range(count_updates):
                updates.put(1)
        elif progress_fallback is not None:
            progress_fallback.update(count_updates)


def run_jax_extraction(
    files: Sequence[Sequence[str]],
    *,
    extract_f0: bool = True,
    f0_force: bool = False,
    rectified: bool = False,
    extract_features: bool = True,
    embedder_model: str = "contentvec",
    version: str = "v2",
    accelerator: str = "auto",
    filter_radius: float = 0.03,
    progress: Any = None,
    updates: Any = None,
    predecoded_audio: dict[str, np.ndarray] | None = None,
) -> dict[str, int]:
    """Run pipelined multi-chip JAX extraction for RMVPE F0 and/or HuBERT features.

    Reads/resamples each clip once, executes batched BF16-MXU / FP32-critical forward passes
    on static bucket shapes across all JAX devices, and writes .npy outputs asynchronously.
    """
    cache_dir = os.path.join(os.getcwd(), "assets", "xla_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception:
        pass

    devices = resolve_jax_devices(accelerator, "all")
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), ("data",))
    replicated = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    prec = PrecisionConfig.bfloat16() if supports_bfloat16(devices) else PrecisionConfig.float32()

    # Stage 0: Filter work & mark skipped steps cleanly on progress bar
    pending: list[tuple[int, Sequence[str], bool, bool]] = []
    skipped_updates = 0
    for idx, f_info in enumerate(files):
        _, f0_c, f0_v, feat_p = f_info[:4]
        need_f0 = bool(extract_f0 and (f0_force or not (readable_npy(f0_c) and readable_npy(f0_v))))
        need_feat = bool(extract_features and not readable_npy(feat_p))
        if extract_f0 and not need_f0:
            skipped_updates += 1
        if extract_features and not need_feat:
            skipped_updates += 1
        if need_f0 or need_feat:
            pending.append((idx, f_info, need_f0, need_feat))

    if skipped_updates and progress is not None:
        progress.update(skipped_updates)
    if not pending:
        return {"processed": 0, "skipped_steps": skipped_updates}

    need_any_f0 = any(x[2] for x in pending)
    need_any_feat = any(x[3] for x in pending)

    cpu_count = os.cpu_count() or 4
    decode_workers = min(32, max(4, cpu_count * 2))
    writer_workers = min(16, max(4, cpu_count))
    threshold_f32 = float(filter_radius)

    batch_queue: queue.Queue = queue.Queue(maxsize=4)
    sentinel = object()
    spillover: list[tuple[int, Sequence[str], np.ndarray, bool, bool]] = []
    decode_errors: list[BaseException] = []

    def _pack_batch(rows, global_bs: int, bucket_samples: int, do_f0: bool, do_feat: bool):
        audio_arr = np.zeros((global_bs, bucket_samples), dtype=np.float32)
        lens_arr = np.full((global_bs,), max(512, min(bucket_samples, RMVPE_HOP * 4)), dtype=np.int32)
        meta = []
        for bi, (fi, wav, df, dft) in enumerate(rows):
            n = len(wav)
            audio_arr[bi, :n] = wav
            lens_arr[bi] = n
            meta.append((fi, n, df, dft))
        return do_f0, do_feat, audio_arr, lens_arr, meta

    def _produce_batches() -> None:
        try:
            with ThreadPoolExecutor(max_workers=decode_workers) as dec_pool:
                if predecoded_audio:
                    est_lens = [
                        len(predecoded_audio[item[1][0]])
                        if item[1][0] in predecoded_audio
                        else _estimate_16k_samples(item[1][0])
                        for item in pending
                    ]
                else:
                    est_lens = list(dec_pool.map(lambda it: _estimate_16k_samples(it[1][0]), pending))

                # Bucket key: (bucket_T, do_f0, do_feat) so every item in a batch executes identical XLA graphs
                grouped: dict[tuple[int, bool, bool], list[tuple[int, Sequence[str], bool, bool]]] = {}
                for item, est_n in zip(pending, est_lens):
                    b_len = _bucket_length_for_samples(est_n)
                    grouped.setdefault((b_len, item[2], item[3]), []).append(item)

                def _get_audio(it):
                    if predecoded_audio and it[1][0] in predecoded_audio:
                        return (*it, predecoded_audio[it[1][0]])
                    idx, fi, audio, df, dft = _decode_one(it)
                    return idx, fi, df, dft, audio

                for (bucket_samples, do_f0, do_feat), items in sorted(grouped.items()):
                    per_dev = _snap_per_device_batch(len(items), num_devices, _batch_per_device_for_bucket(bucket_samples))
                    global_bs = per_dev * num_devices
                    cur_rows: list[tuple[Sequence[str], np.ndarray, bool, bool]] = []

                    for idx, fi, df, dft, audio in dec_pool.map(_get_audio, items):
                        if len(audio) > bucket_samples:
                            spillover.append((idx, fi, audio, df, dft))
                            continue
                        cur_rows.append((fi, audio, df, dft))
                        if len(cur_rows) == global_bs:
                            batch_queue.put(_pack_batch(cur_rows, global_bs, bucket_samples, do_f0, do_feat))
                            cur_rows = []
                    if cur_rows:
                        batch_queue.put(_pack_batch(cur_rows, global_bs, bucket_samples, do_f0, do_feat))
        except BaseException as exc:
            decode_errors.append(exc)
        finally:
            batch_queue.put(sentinel)

    # Start host probing + decode in parallel with TPU model loading so CPU/disk hides weight load.
    producer = threading.Thread(target=_produce_batches, daemon=True)
    producer.start()

    rmvpe_params = None
    if need_any_f0:
        model_name = "rmvpe.pt" if rectified else "rmvpe_rvc.pt"
        model_path = os.path.join("models", "predictors", model_name)
        rmvpe_params = jax.device_put(
            jax.tree.map(jnp.asarray, rmvpe_params_from_torch_file(model_path)),
            replicated,
        )

    hubert_cfg: HubertConfig | None = None
    hubert_params = None
    expected_dim: int | None = None
    if need_any_feat:
        embed_dir = ensure_embedding_files(embedder_model)
        hubert_cfg, h_host = load_hubert_from_dir(embed_dir)
        expected_dim = 256 if version == "v1" else int(hubert_cfg.hidden_size)
        hubert_params = jax.device_put(jax.tree.map(jnp.asarray, h_host), replicated)

    @partial(jax.jit, static_argnums=(0, 1))
    def _jit_infer(run_f0: bool, run_feat: bool, p_rmvpe, p_hubert, audio_bt, lens_b):
        f0 = rmvpe_f0_forward(audio_bt, lens_b, p_rmvpe, threshold=threshold_f32, prec=prec) if run_f0 else None
        feats = (
            hubert_forward(audio_bt, lens_b, p_hubert, hubert_cfg, version=version, prec=prec)
            if run_feat and hubert_cfg is not None
            else None
        )
        return f0, feats

    write_futures: list[Any] = []
    pending_result: tuple[Any, Any, list[tuple[Sequence[str], int, bool, bool]]] | None = None

    def _flush_pending(prev_f0, prev_feat, prev_meta, pool):
        f0_h, feat_h = jax.device_get((prev_f0, prev_feat))
        nv = len(prev_meta)
        fut = pool.submit(
            _write_outputs,
            prev_meta,
            np.asarray(f0_h[:nv]) if f0_h is not None else None,
            np.asarray(feat_h[:nv]) if feat_h is not None else None,
            hubert_cfg,
            expected_dim,
            updates,
            progress if updates is None else None,
        )
        write_futures.append(fut)
        while len(write_futures) > 8:
            write_futures.pop(0).result()

    with ThreadPoolExecutor(max_workers=writer_workers) as write_pool:
        while True:
            item = batch_queue.get()
            if item is sentinel:
                break
            do_f0, do_feat, audio_np, lens_np, meta = item
            audio_dev = jax.device_put(audio_np, data_sharding if num_devices > 1 else devices[0])
            lens_dev = jax.device_put(lens_np, data_sharding if num_devices > 1 else devices[0])

            # Enqueue next TPU batch asynchronously, then transfer/write previous batch while TPU runs.
            f0_dev, feat_dev = _jit_infer(do_f0, do_feat, rmvpe_params, hubert_params, audio_dev, lens_dev)
            if pending_result is not None:
                _flush_pending(*pending_result, write_pool)
            pending_result = (f0_dev, feat_dev, meta)

        if pending_result is not None:
            _flush_pending(*pending_result, write_pool)
            pending_result = None

        producer.join()
        if decode_errors:
            raise decode_errors[0]

        # Handle any unusually long spillover clips whose headers underestimated length
        for _idx, fi, audio, do_f0, do_feat in spillover:
            b_len = _bucket_length_for_samples(len(audio) + 64)
            do_f0_s, do_feat_s, audio_np, lens_np, meta = _pack_batch([(fi, audio, do_f0, do_feat)], num_devices, b_len, do_f0, do_feat)
            audio_dev = jax.device_put(audio_np, data_sharding if num_devices > 1 else devices[0])
            lens_dev = jax.device_put(lens_np, data_sharding if num_devices > 1 else devices[0])
            f0_dev, feat_dev = _jit_infer(do_f0_s, do_feat_s, rmvpe_params, hubert_params, audio_dev, lens_dev)
            _flush_pending(f0_dev, feat_dev, meta, write_pool)

        for fut in write_futures:
            fut.result()

    return {"processed": len(pending), "skipped_steps": skipped_updates}
