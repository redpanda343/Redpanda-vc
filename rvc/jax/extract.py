"""Pipelined multi-chip JAX/TPU extraction for RMVPE F0 and ContentVec/Spin-v2 features."""

from __future__ import annotations

import math
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from rvc.train.backend import configure_jax_env, has_jax_cuda_plugin, has_jax_tpu_runtime

configure_jax_env("auto")

import jax
import jax.numpy as jnp
import numpy as np
import soundfile as sf
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from rvc.jax import PrecisionConfig, init_jax_compilation_cache, resolve_jax_devices, resolve_shard_map, supports_bfloat16
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
# Keep 3 coarse standard buckets (3.2s, 6.4s, 9.6s) + 15-step alignment above 9.6s to prevent XLA recompile storms.
_BUCKET_KS = (10, 20, 30)
_RMVPE_SEG_SAMPLES = 32 * RMVPE_HOP  # 5120


def jax_extraction_available(accelerator: str = "auto", *, allow_gpu: bool = False) -> bool:
    """Return True if JAX TPU (or CUDA GPU when requested/needed) extraction is available."""
    mode = os.environ.get("REDPANDA_EXTRACT_BACKEND", "auto").strip().lower()
    if mode == "torch":
        return False
    if mode != "jax":
        if not (has_jax_tpu_runtime() or (allow_gpu and has_jax_cuda_plugin())):
            return False
    try:
        devs = resolve_jax_devices(accelerator, "all")
        if mode == "jax":
            return len(devs) > 0
        if not devs:
            return False
        return devs[0].platform == "tpu" or (allow_gpu and devs[0].platform == "gpu")
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
    k_dyn = max(30, int(math.ceil((n_samples + RMVPE_HOP) / (_RMVPE_SEG_SAMPLES * 15))) * 15)
    return _RMVPE_SEG_SAMPLES * k_dyn - RMVPE_HOP


def _batch_per_device_for_bucket(bucket_samples: int) -> int:
    if bucket_samples <= 51040:
        return 16
    if bucket_samples <= 102240:
        return 12
    if bucket_samples <= 153440:
        return 8
    return 4


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
    init_jax_compilation_cache()

    devices = resolve_jax_devices(accelerator, "all")
    num_devices = len(devices)
    mesh = Mesh(np.array(devices), ("data",))
    replicated = NamedSharding(mesh, PartitionSpec())
    data_sharding = NamedSharding(mesh, PartitionSpec("data"))
    prec = PrecisionConfig.bfloat16() if supports_bfloat16(devices) else PrecisionConfig.float32()

    # Stage 0: Filter work & mark skipped steps cleanly on progress bar
    def _check_pending(pair):
        idx, f_info = pair
        _, f0_c, f0_v, feat_p = f_info[:4]
        need_f0 = bool(extract_f0 and (f0_force or not (readable_npy(f0_c) and readable_npy(f0_v))))
        need_feat = bool(extract_features and not readable_npy(feat_p))
        return idx, f_info, need_f0, need_feat

    cpu_count = os.cpu_count() or 4
    decode_workers = min(32, max(4, cpu_count * 2))
    writer_workers = min(16, max(4, cpu_count))

    if len(files) > 32:
        with ThreadPoolExecutor(max_workers=decode_workers) as check_pool:
            checked = list(check_pool.map(_check_pending, enumerate(files)))
    else:
        checked = [_check_pending(pair) for pair in enumerate(files)]

    pending: list[tuple[int, Sequence[str], bool, bool]] = []
    skipped_updates = 0
    for idx, f_info, need_f0, need_feat in checked:
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
    threshold_f32 = float(filter_radius)

    # Stage 1: Probe lengths upfront so active buckets are known for AOT warmup while decoding runs.
    with ThreadPoolExecutor(max_workers=decode_workers) as est_pool:
        if predecoded_audio:
            est_lens = [
                len(predecoded_audio[item[1][0]])
                if item[1][0] in predecoded_audio
                else _estimate_16k_samples(item[1][0])
                for item in pending
            ]
        else:
            est_lens = list(est_pool.map(lambda it: _estimate_16k_samples(it[1][0]), pending))

    # Bucket key: (bucket_T, do_f0, do_feat) using one canonical per-device batch size per bucket.
    grouped: dict[tuple[int, bool, bool], list[tuple[int, Sequence[str], bool, bool]]] = {}
    for item, est_n in zip(pending, est_lens):
        b_len = _bucket_length_for_samples(est_n)
        grouped.setdefault((b_len, item[2], item[3]), []).append(item)

    batch_queue: queue.Queue = queue.Queue(maxsize=24)
    sentinel = object()
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
            overflow: dict[tuple[int, bool, bool], list[tuple[Sequence[str], np.ndarray, bool, bool]]] = {}

            def _get_audio(it):
                if predecoded_audio and it[1][0] in predecoded_audio:
                    return (*it, predecoded_audio[it[1][0]])
                idx, fi, audio, df, dft = _decode_one(it)
                return idx, fi, df, dft, audio

            with ThreadPoolExecutor(max_workers=decode_workers) as dec_pool:
                keys_ordered = sorted(grouped.keys())
                idx_cursor = 0
                while idx_cursor < len(keys_ordered) or overflow:
                    if idx_cursor < len(keys_ordered):
                        key = keys_ordered[idx_cursor]
                        idx_cursor += 1
                    else:
                        key = min(overflow.keys())
                    bucket_samples, do_f0, do_feat = key
                    items = grouped.get(key, [])
                    per_dev = _batch_per_device_for_bucket(bucket_samples)
                    global_bs = per_dev * num_devices
                    cur_rows: list[tuple[Sequence[str], np.ndarray, bool, bool]] = overflow.pop(key, [])

                    while len(cur_rows) >= global_bs:
                        batch_queue.put(_pack_batch(cur_rows[:global_bs], global_bs, bucket_samples, do_f0, do_feat))
                        cur_rows = cur_rows[global_bs:]

                    futures = [dec_pool.submit(_get_audio, it) for it in items]
                    for fut in as_completed(futures):
                        _idx, fi, df, dft, audio = fut.result()
                        if len(audio) > bucket_samples:
                            target_b = _bucket_length_for_samples(len(audio) + 64)
                            overflow.setdefault((target_b, df, dft), []).append((fi, audio, df, dft))
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

    # Start host decoding in parallel with TPU weight load + AOT bucket compilation.
    producer = threading.Thread(target=_produce_batches, daemon=True)
    producer.start()

    rmvpe_params = None
    if need_any_f0:
        model_name = "rmvpe.pt" if rectified else "rmvpe_rvc.pt"
        model_path = os.path.join("models", "predictors", model_name)
        rmvpe_params = jax.device_put(rmvpe_params_from_torch_file(model_path), replicated)

    hubert_cfg: HubertConfig | None = None
    hubert_params = None
    expected_dim: int | None = None
    if need_any_feat:
        embed_dir = ensure_embedding_files(embedder_model)
        hubert_cfg, h_host = load_hubert_from_dir(embed_dir)
        expected_dim = 256 if version == "v1" else int(hubert_cfg.hidden_size)
        hubert_params = jax.device_put(h_host, replicated)

    shard_map_fn, smap_check_kwargs = resolve_shard_map() if num_devices > 1 else (None, {})
    p_data = PartitionSpec("data")
    p_empty = PartitionSpec()
    jit_modes: dict[tuple[bool, bool], Any] = {}

    def _call_sharded(local_fn, out_spec, audio_bt, lens_b, *params):
        if shard_map_fn is not None:
            return shard_map_fn(
                local_fn,
                mesh=mesh,
                in_specs=(p_data, p_data) + (p_empty,) * len(params),
                out_specs=out_spec,
                **smap_check_kwargs,
            )(audio_bt, lens_b, *params)
        return local_fn(audio_bt, lens_b, *params)

    def _get_jit_infer(run_f0: bool, run_feat: bool):
        mode = (bool(run_f0), bool(run_feat))
        fn = jit_modes.get(mode)
        if fn is not None:
            return fn

        @jax.jit
        def _infer(p_rmvpe, p_hubert, audio_bt, lens_b):
            if run_f0 and run_feat and hubert_cfg is not None:
                return _call_sharded(
                    lambda a_bt, l_b, p_r, p_h: (
                        rmvpe_f0_forward(a_bt, l_b, p_r, threshold=threshold_f32, prec=prec),
                        hubert_forward(a_bt, l_b, p_h, hubert_cfg, version=version, prec=prec),
                    ),
                    (p_data, p_data),
                    audio_bt,
                    lens_b,
                    p_rmvpe,
                    p_hubert,
                )
            if run_f0:
                f0 = _call_sharded(
                    lambda a_bt, l_b, p_r: rmvpe_f0_forward(a_bt, l_b, p_r, threshold=threshold_f32, prec=prec),
                    p_data,
                    audio_bt,
                    lens_b,
                    p_rmvpe,
                )
                return f0, None
            if run_feat and hubert_cfg is not None:
                feats = _call_sharded(
                    lambda a_bt, l_b, p_h: hubert_forward(a_bt, l_b, p_h, hubert_cfg, version=version, prec=prec),
                    p_data,
                    audio_bt,
                    lens_b,
                    p_hubert,
                )
                return None, feats
            return None, None

        jit_modes[mode] = _infer
        return _infer

    def _to_sds(tree: Any) -> Any:
        return jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=a.sharding), tree)

    p_rmvpe_sds = _to_sds(rmvpe_params) if rmvpe_params is not None else None
    p_hubert_sds = _to_sds(hubert_params) if hubert_params is not None else None
    compiled_infer: dict[tuple[bool, bool, int, int], Any] = {}

    warmup_keys = sorted(grouped.keys())
    if warmup_keys:
        print(f"Pre-compiling XLA extraction kernels for {len(warmup_keys)} bucket shape(s)...", flush=True)
        for idx_w, (b_len, do_f0_w, do_feat_w) in enumerate(warmup_keys, start=1):
            g_bs = _batch_per_device_for_bucket(b_len) * num_devices
            t0 = time.perf_counter()
            audio_sds = jax.ShapeDtypeStruct((g_bs, b_len), jnp.float32, sharding=data_sharding)
            lens_sds = jax.ShapeDtypeStruct((g_bs,), jnp.int32, sharding=data_sharding)
            compiled = _get_jit_infer(do_f0_w, do_feat_w).lower(
                p_rmvpe_sds,
                p_hubert_sds,
                audio_sds,
                lens_sds,
            ).compile()
            compiled_infer[(do_f0_w, do_feat_w, g_bs, b_len)] = compiled
            dt = time.perf_counter() - t0
            print(
                f"  [{idx_w}/{len(warmup_keys)}] compiled (batch={g_bs}, samples={b_len}, f0={do_f0_w}, feat={do_feat_w}) in {dt:.1f}s",
                flush=True,
            )

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
        while len(write_futures) > 16:
            write_futures.pop(0).result()

    with ThreadPoolExecutor(max_workers=writer_workers) as write_pool:
        while True:
            item = batch_queue.get()
            if item is sentinel:
                break
            do_f0, do_feat, audio_np, lens_np, meta = item
            audio_dev, lens_dev = jax.device_put((audio_np, lens_np), data_sharding)

            infer_key = (do_f0, do_feat, int(audio_np.shape[0]), int(audio_np.shape[1]))
            infer_fn = compiled_infer.get(infer_key) or _get_jit_infer(do_f0, do_feat)
            f0_dev, feat_dev = infer_fn(rmvpe_params, hubert_params, audio_dev, lens_dev)
            if pending_result is not None:
                _flush_pending(*pending_result, write_pool)
            pending_result = (f0_dev, feat_dev, meta)

        if pending_result is not None:
            _flush_pending(*pending_result, write_pool)
            pending_result = None

        producer.join()
        if decode_errors:
            raise decode_errors[0]

        for fut in write_futures:
            fut.result()

    return {"processed": len(pending), "skipped_steps": skipped_updates}
