"""Native JAX/XLA implementation of RVC models and training."""

from dataclasses import dataclass
import os
from typing import Any, Sequence

import jax
import jax.numpy as jnp
from jax import lax

# 128-lane / 16-frame aligned bucket upper boundaries for TPU MXU compilation stability.
# At upsample_rates=(10, 8, 2, 2) or (10, 10, 2, 2), L % 16 == 0 ensures (frames * 8) % 128 == 0.
JAX_BUCKET_BOUNDARIES: tuple[int, ...] = (50, 304, 608, 912)


def init_jax_compilation_cache(base_dir: str | None = None) -> None:
    """Configure persistent XLA/JAX compilation cache before backend/device execution."""
    root = base_dir or os.getcwd()
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR") or os.path.join(root, "assets", "xla_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    except Exception as exc:
        print(f"Warning: could not configure persistent XLA cache at '{cache_dir}': {exc}")
        return
    for key, val in (
        ("jax_persistent_cache_min_entry_size_bytes", -1),
        ("jax_persistent_cache_enable_xla_caches", "all"),
    ):
        try:
            jax.config.update(key, val)
        except Exception:
            pass


@dataclass(frozen=True)
class PrecisionConfig:
    """Precision policy enforcing FP32 storage/math where stability requires and BF16 on MXU ops."""

    compute_dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    compute_precision: Any = lax.Precision.HIGHEST
    fp32_precision: Any = lax.Precision.HIGHEST

    @classmethod
    def bfloat16(cls) -> "PrecisionConfig":
        """Mixed precision: BF16 heavy convolutions/linears with FP32 parameters, losses, and utils."""
        return cls(
            compute_dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            compute_precision=lax.Precision.DEFAULT,
            fp32_precision=lax.Precision.HIGHEST,
        )

    @classmethod
    def float32(cls) -> "PrecisionConfig":
        """Full FP32 precision with Precision.HIGHEST."""
        return cls(
            compute_dtype=jnp.float32,
            param_dtype=jnp.float32,
            compute_precision=lax.Precision.HIGHEST,
            fp32_precision=lax.Precision.HIGHEST,
        )


def resolve_jax_devices(accelerator: str = "auto", tpu_cores: str = "all") -> list[jax.Device]:
    """Select JAX devices for training based on CLI flags."""
    acc = (accelerator or "auto").strip().lower()
    if acc in ("cuda", "jax-cuda", "jax_cuda", "jax-gpu", "jax_gpu", "jax"):
        acc = "gpu"
    try:
        if acc == "tpu":
            devices = jax.devices("tpu")
        elif acc == "gpu":
            devices = jax.devices("gpu")
        elif acc == "cpu":
            devices = jax.devices("cpu")
        else:
            for backend in ("tpu", "gpu", "cpu"):
                try:
                    devices = jax.devices(backend)
                    if devices:
                        break
                except RuntimeError:
                    continue
            else:
                devices = jax.devices()
    except RuntimeError as exc:
        if acc == "gpu":
            raise RuntimeError(
                'No CUDA-enabled JAX GPU devices found. On Linux/WSL2 install "pip install -r requirements-jax-cuda.txt" '
                '(or "jax[cuda12]"/"jax[cuda13]"). On Windows with an NVIDIA GPU, CUDA JAX wheels are unavailable so PyTorch CUDA is used instead.'
            ) from exc
        if acc == "tpu":
            raise RuntimeError(
                'No JAX TPU devices found. Ensure libtpu is installed ("pip install jax[tpu]") on a TPU VM.'
            ) from exc
        raise

    if not devices:
        raise RuntimeError(f"No JAX devices found for accelerator='{accelerator}'.")

    platform = devices[0].platform
    cores_mode = str(tpu_cores or "all").strip().lower()
    if platform == "tpu" and cores_mode == "1":
        return [devices[0]]
    return list(devices)


def supports_bfloat16(devices: Sequence[jax.Device]) -> bool:
    """Return True if the target JAX devices natively accelerate bfloat16."""
    if not devices:
        return False
    platform = devices[0].platform
    if platform == "tpu":
        return True
    if platform == "gpu":
        kind = getattr(devices[0], "device_kind", "").lower()
        # NVIDIA Volta/Turing (T4/V100) lack native tensor-core BF16; Ampere+ and MI200+ support it.
        non_bf16_kinds = ("t4", "v100", "p100", "p4", "k80", "gtx", "rtx 20", "quadro rtx")
        return not any(tok in kind for tok in non_bf16_kinds)
    return False
