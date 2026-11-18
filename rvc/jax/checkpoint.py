"""Native JAX checkpointing (.npz + JSON), PyTorch pretrain import, and .pth inference export."""

from __future__ import annotations

import glob
import json
import math
import os
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from rvc.jax.discriminators import (
    MultiPeriodDiscriminatorConfig,
    mpd_from_torch_state,
    mpd_to_torch_state,
)
from rvc.jax.synthesizers import (
    SynthesizerConfig,
    synthesizer_from_torch_state,
    synthesizer_to_torch_state,
)

LATEST_CHECKPOINT_SUFFIX = 2333333
_META_KEY = "__meta_json__"
_BLOB_KEY = "__packed_blob__"


def _normalize_torch_wn_keys(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize legacy .weight_g/.weight_v keys to parametrizations.weight.original0/1."""
    out: dict[str, Any] = {}
    for k, v in state.items():
        nk = k
        if isinstance(nk, str):
            nk = nk.replace(".weight_v", ".parametrizations.weight.original1").replace(
                ".weight_g", ".parametrizations.weight.original0"
            )
        out[nk] = v
    return out


def _collect_named_leaves(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Collect (key_path, leaf) pairs from a nested dict/list/tuple PyTree in deterministic order."""
    entries: list[tuple[str, Any]] = []

    def _walk(node: Any, cur: str) -> None:
        if isinstance(node, Mapping):
            for k, v in node.items():
                nxt = f"{cur}/{k}" if cur else str(k)
                _walk(v, nxt)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                nxt = f"{cur}/{i}" if cur else str(i)
                _walk(v, nxt)
        else:
            entries.append((cur, node))

    _walk(tree, prefix)
    return entries


def _needs_device_get(x: Any) -> bool:
    jax_arr = getattr(jax, "Array", None)
    if jax_arr is not None and isinstance(x, jax_arr):
        return True
    return not isinstance(x, (np.ndarray, np.generic, int, float, bool))


def _flatten_named_pytree(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten nested dict/list PyTree into stable string-keyed NumPy arrays via a single batched D2H copy."""
    entries = _collect_named_leaves(tree, prefix)
    if not entries:
        return {}
    raw_leaves = [v for _, v in entries]
    host_leaves = jax.device_get(raw_leaves) if any(_needs_device_get(x) for x in raw_leaves) else raw_leaves
    return {k: np.asarray(h) for (k, _), h in zip(entries, host_leaves)}


def _unflatten_named_pytree(template: Any, flat: Mapping[str, np.ndarray], prefix: str = "") -> Any:
    """Restore a PyTree matching `template`'s structure from string-keyed arrays."""

    def _walk(node: Any, cur: str) -> Any:
        if isinstance(node, Mapping):
            return {k: _walk(v, f"{cur}/{k}" if cur else str(k)) for k, v in node.items()}
        if isinstance(node, tuple):
            return tuple(_walk(v, f"{cur}/{i}" if cur else str(i)) for i, v in enumerate(node))
        if isinstance(node, list):
            return [_walk(v, f"{cur}/{i}" if cur else str(i)) for i, v in enumerate(node)]
        if cur not in flat:
            raise KeyError(f"Missing checkpoint array key '{cur}'.")
        return np.asarray(flat[cur])

    return _walk(template, prefix)


def save_jax_checkpoint(
    path: str,
    params: Mapping[str, Any],
    opt_state: Any,
    learning_rate: float,
    epoch: int,
    extra: Mapping[str, Any] | None = None,
    ema_params: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save JAX model params, Optax state, optional EMA, and metadata to .npz."""
    param_entries = _collect_named_leaves(params, prefix="model")
    opt_leaves = jax.tree_util.tree_leaves(opt_state)
    ema_entries = _collect_named_leaves(ema_params, prefix="ema") if ema_params is not None else []

    all_raw = [v for _, v in param_entries] + opt_leaves + [v for _, v in ema_entries]
    all_host = jax.device_get(all_raw) if any(_needs_device_get(x) for x in all_raw) else all_raw

    n_p = len(param_entries)
    n_o = len(opt_leaves)
    arrays: dict[str, np.ndarray] = {}
    for (k, _), val in zip(param_entries, all_host[:n_p]):
        arrays[k] = np.asarray(val)
    for i, val in enumerate(all_host[n_p : n_p + n_o]):
        arrays[f"opt/{i}"] = np.asarray(val)
    for (k, _), val in zip(ema_entries, all_host[n_p + n_o :]):
        arrays[k] = np.asarray(val)

    meta: dict[str, Any] = {
        "format": "redpanda_jax_v1",
        "epoch": int(epoch),
        "learning_rate": float(learning_rate),
        "opt_leaf_count": n_o,
        "has_ema": ema_params is not None,
    }
    if extra:
        meta["extra"] = dict(extra)
    if "model/emb_g/weight" in arrays:
        meta["spk_embed_dim"] = int(arrays["model/emb_g/weight"].shape[0])
    meta["model_keys"] = [k[len("model/") :] for k, _ in param_entries]

    save_payload: dict[str, np.ndarray]
    if len(arrays) > 16:
        # Pack hundreds of parameter/optimizer arrays into one contiguous ZIP entry to avoid overlayfs metadata stalls.
        manifest: dict[str, list[Any]] = {}
        chunks: list[tuple[int, np.ndarray]] = []
        cursor = 0
        for k, arr in arrays.items():
            orig_shape = list(arr.shape)
            arr_c = np.ascontiguousarray(arr)
            nbytes = int(arr_c.nbytes)
            if nbytes > 0:
                offset = (cursor + 15) & ~15
                cursor = offset + nbytes
                chunks.append((offset, arr_c))
            else:
                offset = 0
            manifest[k] = [arr_c.dtype.str, orig_shape, int(offset), nbytes]
        blob = bytearray(cursor)
        blob_mv = memoryview(blob)
        for offset, arr_c in chunks:
            raw = np.frombuffer(np.ascontiguousarray(arr_c), dtype=np.uint8)
            blob_mv[offset : offset + len(raw)] = raw
        meta["format"] = "redpanda_jax_v2_packed"
        meta["manifest"] = manifest
        save_payload = {
            _META_KEY: np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8),
            _BLOB_KEY: np.frombuffer(blob, dtype=np.uint8),
        }
    else:
        meta_bytes = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        arrays[_META_KEY] = meta_bytes
        save_payload = arrays

    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp.npz"
    np.savez(tmp_path, **save_payload)
    os.replace(tmp_path, path)


def read_jax_checkpoint_meta(path: str) -> dict[str, Any]:
    """Read JSON metadata header and speaker count without materializing full optimizer state."""
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(bytes(archive[_META_KEY]).decode("utf-8"))
        if "manifest" not in meta:
            if "model/emb_g/weight" in archive:
                meta["spk_embed_dim"] = int(archive["model/emb_g/weight"].shape[0])
            meta["model_keys"] = [k[len("model/") :] for k in archive.files if k.startswith("model/")]
    meta.pop("manifest", None)
    return meta


def load_jax_checkpoint(
    path: str,
    params_template: Mapping[str, Any],
    optimizer: Any = None,
    opt_state_template: Any = None,
    load_opt: bool = True,
    reset_state: bool = False,
) -> tuple[Any, Any, float, int, dict[str, Any], Any]:
    """Load native JAX .npz checkpoint into host PyTrees (params, opt_state, lr, epoch, extra, ema_params)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(bytes(archive[_META_KEY]).decode("utf-8"))
        if _BLOB_KEY in archive and "manifest" in meta:
            blob = np.asarray(archive[_BLOB_KEY], dtype=np.uint8)
            raw_buf = memoryview(blob)
            flat_arrays = {}
            for k, (dt_str, shape_list, offset, nbytes) in meta["manifest"].items():
                shape = tuple(int(d) for d in shape_list)
                dt = np.dtype(dt_str)
                if int(nbytes) == 0:
                    flat_arrays[k] = np.empty(shape, dtype=dt)
                else:
                    flat_arrays[k] = np.array(
                        np.frombuffer(
                            raw_buf, dtype=dt, count=math.prod(shape), offset=int(offset)
                        ).reshape(shape),
                        copy=True,
                    )
        else:
            flat_arrays = {k: np.asarray(archive[k]) for k in archive.files if k != _META_KEY}

    params = _unflatten_named_pytree(params_template, flat_arrays, prefix="model")

    opt_state = opt_state_template
    if load_opt and not reset_state and opt_state_template is not None:
        template_leaves, treedef = jax.tree_util.tree_flatten(opt_state_template)
        expected_count = int(meta.get("opt_leaf_count", -1))
        if expected_count == len(template_leaves):
            restored_leaves = []
            for i, tmpl_leaf in enumerate(template_leaves):
                arr = flat_arrays[f"opt/{i}"]
                tmpl_shape = tuple(tmpl_leaf.shape) if hasattr(tmpl_leaf, "shape") else ()
                if tuple(arr.shape) != tmpl_shape:
                    raise ValueError(
                        f"Optimizer state leaf {i} shape mismatch in '{path}': "
                        f"got {arr.shape}, expected {tmpl_shape}."
                    )
                if isinstance(tmpl_leaf, (bool, np.bool_)):
                    tmpl_dtype = np.bool_
                elif isinstance(tmpl_leaf, int):
                    tmpl_dtype = np.int32
                else:
                    tmpl_dtype = getattr(tmpl_leaf, "dtype", np.float32)
                    if tmpl_dtype == np.float64:
                        tmpl_dtype = np.float32
                restored_leaves.append(arr.astype(tmpl_dtype, copy=False))
            opt_state = jax.tree_util.tree_unflatten(treedef, restored_leaves)
        else:
            print(f"Warning: optimizer state structure in '{path}' did not match; starting optimizer fresh.")

    ema_params = None
    if meta.get("has_ema"):
        ema_template = {k: v for k, v in params_template.items() if k != "enc_q"}
        if all(k in flat_arrays for k in _flatten_named_pytree(ema_template, prefix="ema")):
            ema_params = _unflatten_named_pytree(ema_template, flat_arrays, prefix="ema")

    return (
        params,
        opt_state,
        float(meta.get("learning_rate", 0.0)),
        int(meta.get("epoch", 1)),
        dict(meta.get("extra", {})),
        ema_params,
    )


def load_pretrained_torch_g(
    path: str,
    cfg: SynthesizerConfig,
    params_template: Mapping[str, Any],
    match_speakers: bool = True,
) -> dict[str, Any]:
    """Load PyTorch pretrained or G_*.pth weights into a JAX Synthesizer param PyTree."""
    import torch

    ckpt_obj = torch.load(path, map_location="cpu", weights_only=True)
    raw_state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    state = _normalize_torch_wn_keys(
        {k: v.detach().cpu().numpy() if hasattr(v, "detach") else v for k, v in raw_state.items()}
    )
    if match_speakers and "emb_g.weight" in state:
        speakers = np.asarray(params_template["emb_g"]["weight"], dtype=np.float32).copy()
        saved_spk = np.asarray(state["emb_g.weight"], dtype=np.float32)
        rows = min(speakers.shape[0], saved_spk.shape[0])
        speakers[:rows] = saved_spk[:rows]
        state["emb_g.weight"] = speakers
    return synthesizer_from_torch_state(state, cfg)


def load_pretrained_torch_d(path: str, cfg: MultiPeriodDiscriminatorConfig) -> dict[str, Any]:
    """Load PyTorch pretrained or D_*.pth weights into a JAX MultiPeriodDiscriminator param PyTree."""
    import torch

    ckpt_obj = torch.load(path, map_location="cpu", weights_only=True)
    raw_state = ckpt_obj["model"] if isinstance(ckpt_obj, dict) and "model" in ckpt_obj else ckpt_obj
    state = _normalize_torch_wn_keys(
        {k: v.detach().cpu().numpy() if hasattr(v, "detach") else v for k, v in raw_state.items()}
    )
    return mpd_from_torch_state(state, cfg)


def export_jax_to_inference_pth(
    g_or_ema_params: Mapping[str, Any],
    cfg: SynthesizerConfig,
    *,
    sr: int,
    name: str,
    model_path: str,
    epoch: int,
    step: int,
    hps: Any,
    vocoder: str,
    pitch_guidance: bool = True,
    version: str | None = None,
    export_dtype_str: str = "float16",
    verbose: bool = False,
) -> str:
    """Export JAX generator/EMA parameters into a standard main-compatible PyTorch .pth model."""
    import torch
    from rvc.train.process.extract_model import extract_model

    # Keep weight_norm parametrizations intact because rvc/infer/infer.py loads into
    # a weight-normed Synthesizer first and calls remove_weight_norm() afterward.
    np_state = synthesizer_to_torch_state(
        g_or_ema_params,
        cfg,
        remove_weight_norm=False,
        inference_only=True,
    )
    torch_state = {
        k: torch.from_numpy(np.array(v, dtype=np.float32, copy=True))
        for k, v in np_state.items()
    }
    export_dtype = torch.float16 if export_dtype_str == "float16" else torch.float32
    extract_model(
        ckpt=torch_state,
        sr=sr,
        name=name,
        model_path=model_path,
        epoch=epoch,
        step=step,
        hps=hps,
        vocoder=vocoder,
        pitch_guidance=pitch_guidance,
        version=version,
        export_dtype=export_dtype,
        verbose=verbose,
    )
    return model_path


def latest_checkpoint_pair(experiment_dir: str) -> tuple[str | None, str | None]:
    """Find newest resumable G/D pair in native .npz (preferred) or legacy .pth format."""
    for ext in ("npz", "pth"):
        generators = sorted(
            glob.glob(os.path.join(experiment_dir, f"G_*.{ext}")),
            key=os.path.getmtime,
            reverse=True,
        )
        for gen in generators:
            disc = os.path.join(experiment_dir, "D_" + os.path.basename(gen)[2:])
            if os.path.isfile(disc):
                return gen, disc
        if generators:
            return generators[0], None
        discs = sorted(
            glob.glob(os.path.join(experiment_dir, f"D_*.{ext}")),
            key=os.path.getmtime,
            reverse=True,
        )
        if discs:
            return None, discs[0]
    return None, None


def checkpoint_step_from_path(path: str) -> int:
    match = re.fullmatch(r"G_(\d+)\.(?:npz|pth)", os.path.basename(path))
    if not match:
        return 0
    step = int(match.group(1))
    return 0 if step == LATEST_CHECKPOINT_SUFFIX else step


class _AsyncSingleWorker:
    """Shared single-thread background executor lifecycle for async exports and checkpoint writes."""

    def __init__(self, thread_name_prefix: str) -> None:
        self.executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name_prefix
        )
        self._future: Future | None = None

    def _submit_fn(self, fn: Any, *args: Any, **kwargs: Any) -> None:
        self.wait_for_completion()
        if self.executor is None:
            return
        self._future = self.executor.submit(fn, *args, **kwargs)

    def raise_if_failed(self) -> None:
        if self._future is not None and self._future.done():
            self.wait_for_completion()

    def wait_for_completion(self) -> None:
        fut = self._future
        if fut is None:
            return
        try:
            fut.result()
        finally:
            self._future = None

    def close(self) -> None:
        if self.executor is None:
            return
        try:
            self.wait_for_completion()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None


class AsyncJaxInferenceExporter(_AsyncSingleWorker):
    """Stage JAX weights onto host CPU and export inference .pth on a background thread."""

    def __init__(self) -> None:
        super().__init__("jax-inference-export")

    def submit(self, params: Mapping[str, Any], cfg: SynthesizerConfig, **export_kwargs: Any) -> None:
        self.wait_for_completion()
        if self.executor is None:
            return
        # Pull arrays onto host CPU cleanly while excluding enc_q before launching worker.
        filtered = {k: v for k, v in params.items() if k != "enc_q"}
        leaves = jax.tree_util.tree_leaves(filtered)
        host_snapshot = (
            jax.device_get(filtered) if any(not isinstance(x, np.ndarray) for x in leaves) else filtered
        )
        self.submit_host(host_snapshot, cfg, **export_kwargs)

    def submit_host(self, host_snapshot: Mapping[str, Any], cfg: SynthesizerConfig, **export_kwargs: Any) -> None:
        """Queue an inference .pth export using an already host-resident parameter PyTree."""
        filtered = {k: v for k, v in host_snapshot.items() if k != "enc_q"}
        self._submit_fn(export_jax_to_inference_pth, filtered, cfg, **export_kwargs)


class AsyncJaxCheckpointWriter(_AsyncSingleWorker):
    """Write host-staged JAX .npz checkpoints on a background thread without blocking TPU steps."""

    def __init__(self) -> None:
        super().__init__("jax-checkpoint-write")

    def submit(self, jobs: list[tuple[tuple[Any, ...], dict[str, Any]]]) -> None:
        def _run_jobs() -> None:
            for pos_args, kw_args in jobs:
                save_jax_checkpoint(*pos_args, **kw_args)

        self._submit_fn(_run_jobs)
