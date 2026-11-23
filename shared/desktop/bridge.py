"""Local stdio bridge for the Rust desktop client. Audio stays in the Python engine."""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
import time

from shared.desktop.assets import Assets
from shared.desktop.jobs import Jobs
from shared.desktop.paths import (
    CONFIG_PATH,
    DATASETS,
    LOGS,
    MODELS,
    ROOT,
    catalog,
    match_index,
    read_settings,
    require_path,
    write_settings,
)
from shared.desktop.realtime import Realtime
from shared.tools.causal_voices import voice_settings


def _iter_stdin_lines():
    """Read UTF-8 lines from stdin without blocking Windows DLL loading.

    On Windows, a synchronous ReadFile on an anonymous pipe holds a kernel lock
    that blocks GetFileType calls made inside MSVC CRT DllMain during LoadLibrary,
    deadlocking background threads importing numpy/torch/sounddevice. Peeking
    before reading ensures no thread stays inside ReadFile while idle.
    """
    if os.name != "nt":
        yield from sys.stdin
        return
    import ctypes
    import ctypes.wintypes as wt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetStdHandle.argtypes = [wt.DWORD]
    kernel32.GetStdHandle.restype = wt.HANDLE
    kernel32.GetFileType.argtypes = [wt.HANDLE]
    kernel32.GetFileType.restype = wt.DWORD
    kernel32.PeekNamedPipe.argtypes = [
        wt.HANDLE,
        ctypes.c_void_p,
        wt.DWORD,
        ctypes.POINTER(wt.DWORD),
        ctypes.POINTER(wt.DWORD),
        ctypes.POINTER(wt.DWORD),
    ]
    kernel32.PeekNamedPipe.restype = wt.BOOL

    std_input_handle = wt.DWORD(-10).value
    file_type_pipe = 3
    invalid_handle = ctypes.c_void_p(-1).value
    handle = kernel32.GetStdHandle(std_input_handle)
    if not handle or handle == invalid_handle or kernel32.GetFileType(handle) != file_type_pipe:
        yield from sys.stdin
        return

    fd = sys.stdin.fileno()
    buf = bytearray()
    avail = wt.DWORD()
    while True:
        while (newline := buf.find(b"\n")) != -1:
            line = bytes(buf[: newline + 1])
            del buf[: newline + 1]
            yield line.decode("utf-8", errors="replace")
        if not kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None):
            break
        if avail.value > 0:
            chunk = os.read(fd, avail.value)
            if not chunk:
                break
            buf.extend(chunk)
        else:
            time.sleep(0.01)
    if buf:
        yield buf.decode("utf-8", errors="replace")


def main():
    os.chdir(ROOT)
    # Preserve a dedicated protocol stream even when a model or subprocess prints progress.
    outgoing = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    write_lock = threading.Lock()

    def write(message):
        with write_lock:
            outgoing.write(json.dumps(message, ensure_ascii=False, allow_nan=False) + "\n")

    def emit(event, data):
        write({"event": event, "data": data})

    def reply(ident, *, result=None, error=None):
        write({"id": ident, "result": result, "error": error})

    from shared.platform import migrate_models_folder, platform_config

    packages = ("click", "faiss", "librosa", "psutil", "sounddevice", "soundfile", "tensorboard", "torch", "torchaudio")
    missing = [name for name in packages if importlib.util.find_spec(name) is None]
    if missing:
        message = "This Python environment is missing " + ", ".join(missing) + ". Install the runtime or choose another Python."
        emit("runtime_error", message)
        return
    platform_config()
    migrate_models_folder(ROOT)
    LOGS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    DATASETS.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        template = CONFIG_PATH.with_name("config_template.json")
        if template.is_file():
            shutil.copyfile(template, CONFIG_PATH)
        else:
            write_settings({"precision": "fp16", "model_author": ""})

    assets = Assets(emit)
    jobs = Jobs(emit, assets)
    realtime = Realtime(emit, reply, assets)
    emit(
        "ready",
        {
            "root": str(ROOT),
            "python": sys.executable,
            "cpu": max(1, min(32, (os.cpu_count() or 2) // 2)),
            "settings": read_settings(),
        },
    )
    threading.Thread(target=assets.startup, daemon=True).start()
    try:
        for line in _iter_stdin_lines():
            if not line.strip():
                continue
            ident = None
            try:
                request = json.loads(line)
                ident, action = request["id"], request["action"]
                args = request.get("args") or {}
                if action.startswith("audio_"):
                    realtime.request(ident, action, args)
                    continue
                if action == "catalog":
                    output = catalog()
                elif action == "match_index":
                    output = {"index_path": match_index(args.get("path"))}
                elif action == "voice_settings":
                    output = voice_settings(require_path(args.get("path"), "a voice model"))
                elif action == "settings_get":
                    output = read_settings()
                elif action == "settings_set":
                    output = write_settings(args)
                elif action == "tensorboard_start":
                    from shared.tools.launch_tensorboard import PINNED_CARDS, get_tb_error, launch_tensorboard

                    url = launch_tensorboard(quiet=True)
                    if not url:
                        raise RuntimeError(get_tb_error() or "TensorBoard could not be started.")
                    output = {"url": f"{url}{PINNED_CARDS}"}
                elif action == "job_start":
                    output = jobs.start(args["kind"], args["options"])
                elif action == "job_control":
                    command = args["command"]
                    if command not in {"pause", "resume", "stop"}:
                        raise ValueError("Unknown training command.")
                    output = jobs.control(command)
                elif action == "devices":
                    from causalvc.trainer import torch_gpus

                    output = torch_gpus()
                else:
                    raise ValueError(f"Unknown desktop command: {action}")
                reply(ident, result=output)
            except (Exception, SystemExit) as error:
                reply(ident, error=str(error) or type(error).__name__)
    finally:
        realtime.close()
        jobs.close()
        outgoing.close()


if __name__ == "__main__":
    main()
