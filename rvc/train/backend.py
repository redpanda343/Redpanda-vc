"""Lightweight backend discovery and environment setup before importing JAX or PyTorch."""

from __future__ import annotations

import glob
import importlib.metadata
import importlib.util
import os
import sys


def _has_spec(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def has_jax_packages() -> bool:
    """Return True if JAX and Optax are installed in the Python environment."""
    return _has_spec("jax") and _has_spec("optax")


def has_jax_tpu_runtime() -> bool:
    """Return True if a TPU runtime / PJRT plugin appears available without initializing XLA."""
    if not has_jax_packages():
        return False
    if _has_spec("libtpu") or os.path.exists("/dev/accel0"):
        return True
    if os.environ.get("COLAB_TPU_ADDR") or os.environ.get("TPU_NAME"):
        return True
    try:
        for ep in importlib.metadata.entry_points(group="jax_plugins"):
            if "tpu" in ep.name.lower():
                return True
    except Exception:
        pass
    return False


def has_jax_cuda_plugin() -> bool:
    """Return True if a CUDA-enabled JAX PJRT plugin appears installed without calling jax.devices()."""
    if not has_jax_packages():
        return False
    if (
        _has_spec("jax_cuda12_plugin")
        or _has_spec("jax_cuda13_plugin")
        or _has_spec("jaxlib.cuda")
    ):
        return True
    try:
        for ep in importlib.metadata.entry_points(group="jax_plugins"):
            text = f"{ep.name} {ep.value}".lower()
            if "cuda" in text or "gpu" in text:
                return True
    except Exception:
        pass
    if sys.platform != "win32":
        try:
            jaxlib_spec = importlib.util.find_spec("jaxlib")
            if jaxlib_spec and jaxlib_spec.submodule_search_locations:
                root = jaxlib_spec.submodule_search_locations[0]
                for pattern in ("**/*pjrt*cuda*.so*", "**/*pjrt*gpu*.so*"):
                    if glob.glob(os.path.join(root, pattern), recursive=True):
                        return True
        except Exception:
            pass
    return False


def torch_cuda_info(gpus: str = "0") -> tuple[bool, str]:
    """Lazily query PyTorch CUDA availability and primary selected GPU name."""
    try:
        import torch

        if not torch.cuda.is_available():
            return False, ""
        count = torch.cuda.device_count()
        idx = 0
        for part in str(gpus or "0").split("-"):
            if part.strip().isdigit():
                cand = int(part)
                if 0 <= cand < count:
                    idx = cand
                break
        return True, str(torch.cuda.get_device_name(idx))
    except Exception:
        return False, ""


def torch_mps_available() -> bool:
    """Lazily check PyTorch Apple MPS availability."""
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def normalize_accelerator(accelerator: str) -> str:
    """Normalize CLI/UI accelerator names into canonical backend selectors."""
    raw = (accelerator or "auto").strip().lower()
    if raw in ("jax-cuda", "jax_cuda", "jax-gpu", "jax_gpu", "jax"):
        return "jax-cuda"
    if raw in ("torch-cuda", "torch_cuda", "pytorch-cuda", "pytorch_cuda", "torch", "pytorch"):
        return "torch-cuda"
    if raw == "cuda":
        return "gpu"
    return raw


def configure_jax_env(accelerator: str = "auto", gpus: str | None = None) -> None:
    """Apply JAX/XLA environment variables safely prior to importing jax."""
    acc = normalize_accelerator(accelerator)
    gpu_str = (gpus or "").strip()

    if acc == "cpu" or (acc == "auto" and gpu_str == "-"):
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif acc in ("auto", "gpu", "jax-cuda") and gpu_str and gpu_str != "-" and all(
        part.isdigit() for part in gpu_str.split("-")
    ):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_str.replace("-", ","))

    # Prevent XLA from preallocating 75% GPU VRAM when PyTorch CUDA ops may coexist in the process.
    if not has_jax_tpu_runtime():
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def select_training_backend(accelerator: str, gpus: str) -> tuple[str, str | None]:
    """Select ('jax' | 'torch', optional user notice) before initializing heavy runtimes."""
    acc = normalize_accelerator(accelerator)
    gpu_str = (gpus or "0").strip()

    override = os.environ.get("REDPANDA_TRAIN_BACKEND", "").strip().lower()
    if override in {"jax", "torch"}:
        return override, f"Using {override} training backend (REDPANDA_TRAIN_BACKEND={override})."

    if acc == "tpu":
        if not has_jax_packages():
            raise RuntimeError(
                'TPU training requested (--accelerator tpu), but JAX/Optax are not installed. Install with: pip install "jax[tpu]" -r requirements-jax.txt'
            )
        return "jax", None

    if acc == "jax-cuda":
        if gpu_str == "-":
            raise RuntimeError("JAX CUDA requested (--accelerator jax-cuda), but --gpus is '-' (CPU).")
        if not has_jax_packages():
            raise RuntimeError(
                'JAX CUDA requested (--accelerator jax-cuda), but JAX/Optax are not installed. '
                'Install CUDA JAX via: pip install -r requirements-jax-cuda.txt'
            )
        if has_jax_cuda_plugin():
            return "jax", None
        if sys.platform == "win32":
            raise RuntimeError(
                "JAX CUDA requested (--accelerator jax-cuda), but Google JAX does not publish native Windows CUDA wheels. "
                "Select 'PyTorch CUDA' (--accelerator torch-cuda) or 'Auto' on Windows, or run JAX CUDA on Linux/WSL2."
            )
        raise RuntimeError(
            "JAX CUDA requested (--accelerator jax-cuda), but no CUDA-enabled jaxlib plugin is installed. "
            'Install CUDA JAX via "pip install -r requirements-jax-cuda.txt", or select "PyTorch CUDA" (--accelerator torch-cuda).'
        )

    if acc == "torch-cuda":
        if gpu_str == "-":
            raise RuntimeError("PyTorch CUDA requested (--accelerator torch-cuda), but --gpus is '-' (CPU).")
        cuda_ok, gpu_name = torch_cuda_info(gpu_str)
        if cuda_ok:
            return "torch", f"Using PyTorch CUDA backend on {gpu_name}."
        raise RuntimeError(
            "PyTorch CUDA requested (--accelerator torch-cuda), but torch.cuda.is_available() is False. "
            "Install CUDA-enabled PyTorch (cu128/cu130) or select another accelerator."
        )

    if acc == "cpu" or (acc == "auto" and gpu_str == "-"):
        return ("jax" if has_jax_packages() else "torch"), None

    if acc == "gpu":
        if gpu_str == "-":
            raise RuntimeError("GPU training requested (--accelerator gpu), but --gpus is '-' (CPU).")
        if has_jax_cuda_plugin():
            return "jax", None
        cuda_ok, gpu_name = torch_cuda_info(gpu_str)
        if cuda_ok:
            if sys.platform == "win32":
                return (
                    "torch",
                    f"CUDA-enabled jaxlib is unavailable on native Windows; training with PyTorch CUDA on {gpu_name}.",
                )
            return (
                "torch",
                f"CUDA-enabled jaxlib plugin is not installed; training with PyTorch CUDA on {gpu_name}. "
                'Install CUDA JAX via "pip install -r requirements-jax-cuda.txt" to run JAX on GPU.',
            )
        if sys.platform == "win32":
            raise RuntimeError(
                "GPU training requested (--accelerator gpu), but neither PyTorch CUDA nor CUDA JAX is available on Windows. "
                "Re-run run-install.bat or install CUDA-enabled PyTorch (cu128/cu130)."
            )
        raise RuntimeError(
            "GPU training requested (--accelerator gpu), but no CUDA runtime was found in JAX or PyTorch. "
            "Install CUDA JAX and PyTorch via: pip install -r requirements-jax-cuda.txt"
        )

    # acc == "auto" and gpu_str != "-"
    if has_jax_tpu_runtime() or has_jax_cuda_plugin():
        return "jax", None

    cuda_ok, gpu_name = torch_cuda_info(gpu_str)
    if cuda_ok:
        if sys.platform == "win32":
            return (
                "torch",
                f"Using PyTorch CUDA backend on {gpu_name} (CUDA-enabled jaxlib is unavailable on native Windows).",
            )
        return (
            "torch",
            f"Using PyTorch CUDA backend on {gpu_name} (CUDA-enabled jaxlib plugin is not installed).",
        )

    if sys.platform == "darwin" and torch_mps_available():
        return "torch", "Using PyTorch MPS backend."

    return ("jax" if has_jax_packages() else "torch"), None
