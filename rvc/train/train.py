"""Backend-aware training entrypoint for Redpanda-vc (JAX TPU/CUDA and PyTorch CUDA/MPS/CPU)."""

from __future__ import annotations

import importlib
import os
import sys

now_dir = os.getcwd()
_repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _path in (_repo_root, now_dir):
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)

from rvc.train.backend import configure_jax_env, normalize_accelerator, select_training_backend, torch_cuda_info

_COMPAT_EXPORTS = {
    "BUCKET_BOUNDARIES": "rvc.train.train_jax",
    "JaxDeviceInitError": "rvc.train.train_jax",
    "JaxTrainer": "rvc.train.train_jax",
    "LEGACY_BUCKET_BOUNDARIES": "rvc.train.train_jax",
    "LR_SCHEDULES": "rvc.train.train_jax",
    "VOCODERS": "rvc.train.train_jax",
    "check_dataset_sample_rate": "rvc.train.train_jax",
    "load_experiment_config": "rvc.train.train_jax",
    "parse_arguments": "rvc.train.train_jax",
    "read_model_info": "rvc.train.train_jax",
    "remove_previous_attempt": "rvc.train.train_jax",
    "resolve_precision": "rvc.train.train_jax",
}


def __getattr__(name: str):
    mod_name = _COMPAT_EXPORTS.get(name)
    if mod_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(mod_name), name)


def _extract_routing_flags(argv: list[str] | None) -> tuple[str, str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    acc = "auto"
    gpus = "0"
    for i, tok in enumerate(raw):
        if tok == "--accelerator" and i + 1 < len(raw):
            acc = raw[i + 1].lower()
        elif tok.startswith("--accelerator="):
            acc = tok.split("=", 1)[1].lower()
        elif tok == "--gpus" and i + 1 < len(raw):
            gpus = raw[i + 1]
        elif tok.startswith("--gpus="):
            gpus = tok.split("=", 1)[1]
    return acc, gpus


def main(argv: list[str] | None = None) -> None:
    acc, gpus = _extract_routing_flags(argv)
    if "--help" in (sys.argv[1:] if argv is None else argv) or "-h" in (sys.argv[1:] if argv is None else argv):
        from rvc.train.train_torch import parse_arguments as torch_parse

        torch_parse(argv)
        return

    try:
        backend, notice = select_training_backend(acc, gpus)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if notice:
        print(notice)

    if backend == "torch":
        from rvc.train import train_torch

        train_torch.main(argv)
        return

    had_cvd = "CUDA_VISIBLE_DEVICES" in os.environ
    prev_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    configure_jax_env(acc, gpus)

    from rvc.train import train_jax

    try:
        train_jax.main(argv)
    except train_jax.JaxDeviceInitError as exc:
        override = os.environ.get("REDPANDA_TRAIN_BACKEND", "").strip().lower()
        if not had_cvd:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif prev_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cvd

        norm_acc = normalize_accelerator(acc)
        if override != "jax" and norm_acc in ("auto", "gpu") and gpus.strip() != "-":
            cuda_ok, gpu_name = torch_cuda_info(gpus)
            if cuda_ok:
                print(
                    f"JAX GPU initialization failed ({exc}); falling back to PyTorch CUDA on {gpu_name}."
                )
                from rvc.train import train_torch

                train_torch.main(argv)
                return

        if norm_acc == "jax-cuda":
            cuda_ok, gpu_name = torch_cuda_info(gpus)
            if cuda_ok:
                print(
                    f"Error: {exc}\nHint: PyTorch CUDA is working on {gpu_name}; you can select 'PyTorch CUDA' (--accelerator torch-cuda) in the WebUI or CLI.",
                    file=sys.stderr,
                )
                sys.exit(1)

        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
