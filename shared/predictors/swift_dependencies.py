import importlib
import subprocess
import sys
import threading


SWIFT_REQUIREMENT = "swift-f0==0.3.0"
_INSTALL_LOCK = threading.Lock()


def ensure_swift_f0():
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as error:
        if error.name != "onnxruntime":
            raise
        raise RuntimeError(
            "SwiftF0 requires onnxruntime-gpu. Install the project's requirements first."
        ) from error
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "SwiftF0 requires onnxruntime-gpu with CUDAExecutionProvider. "
            "If onnxruntime CPU is installed, remove it and reinstall onnxruntime-gpu."
        )

    with _INSTALL_LOCK:
        try:
            return importlib.import_module("swift_f0")
        except ModuleNotFoundError as error:
            if error.name != "swift_f0":
                raise

        command = [
            sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--no-deps", "--only-binary=:all:", SWIFT_REQUIREMENT,
        ]
        print(
            "SwiftF0 is missing. Installing swift-f0==0.3.0 with --no-deps; "
            "the existing ONNX Runtime GPU installation will be preserved.",
            flush=True,
        )
        try:
            subprocess.run(command, check=True, timeout=180)
        except (subprocess.SubprocessError, OSError) as error:
            raise RuntimeError(
                "Could not safely install SwiftF0. Run "
                f'"{sys.executable}" -m pip install --no-deps {SWIFT_REQUIREMENT} '
                "in the training environment, then retry extraction."
            ) from error
        importlib.invalidate_caches()
        return importlib.import_module("swift_f0")
