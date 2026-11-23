"""Building voices from a window: the build process, its queue and its progress.

    python -m causalvc.trainer

The training windows (training/ui on Linux, training/winui on Windows) start
this as a child process and talk to it in newline-delimited JSON: commands on
stdin, events on stdout. Closing stdin (the window closing) stops a build in
progress the way Stop does: it saves a checkpoint first, so starting the same
voice again resumes it. The small Tk window (causalvc.gui) uses ``Job`` from here directly.

Commands ({"cmd": ..., "args": {...}}):
    start  {config}   start a build now (refused while one runs)
    queue  {config}   add a build to the queue; it starts at once when idle
    remove {"index"}  drop a queued build
    move   {"from", "to"}  move a queued build to another position (indices into the queue)
    build_now {"index"}  start a queued build now: a running one saves a checkpoint, stops and
                      goes back to the front of the queue, so it resumes after this one
    stop              first: save a checkpoint and stop; second: end at once

A config is {"name", "audio", "batch", "steps", "save_every", "base", "device",
"restart", "fixes", "noise", "app_models"}; it is checked, and the form is remembered
for next time (runs/gui.json, shared with the Tk window).

Events ({"event": ..., "data": {...}}):
    hello  {"form", "bases", "gpus"}       once, at start
    gpus   {"gpus": [{"id", "label"}]}     when the GPU list is known (PyTorch loads slowly)
    state  {"phase", "current", "queue", "progress", "exit_code", "next"}
                                           next: the build that starts once the current one has saved
                                           and stopped (build_now), else null
    log    {"text"}                        a line from the build worth showing
    error  {"code", "message", "field"}    a refused command
    queued {"name", "position"}            a queued build is waiting (position 1 is next)

phase: idle, starting, training, stopping, finished, stopped or failed.
Error codes: invalid_name, missing_audio, invalid_numbers, busy, bad_request.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("REDPANDA_DATA_ROOT") or CODE_ROOT).resolve()
RUNS = Path(os.environ.get("CAUSALVC_RUNS") or (ROOT / "runs" / "models")).resolve()
SETTINGS = (ROOT / "assets" / "causal_gui.json") if os.environ.get("REDPANDA_DATA_ROOT") else (ROOT / "runs" / "gui.json")
PROGRESS = re.compile(r"^step (\d+)/(\d+) \((\d+)%\)\s+elapsed (\S+)\s+remaining (\S+)\s+([\d.]+) it/s")
#: Library warnings that say nothing about the build.
NOISE = ("Warning", "warnings.warn", "WeightNorm.apply", "_canonical_mask")
NAME = re.compile(r"[\w.-]+")
DEFAULT_FORM = {
    "audio": "",
    "name": "",
    "batch": 8,
    "steps": 10000,
    "save_every": 10000,
    "base": "default",
    "device": "cuda:0",
    "fixes": True,
    "noise": False,
    "colour": True,
    "app_models": str(ROOT / "realtime" / "models"),
}


def custom_bases() -> list[str]:
    from causalvc.build import bases

    return [name for name in bases() if name != "default"]


def gpus() -> list[str]:
    """Device choices for the Tk window, named when nvidia-smi is there to ask."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"], capture_output=True, text=True, timeout=5, check=False).stdout
        found = [f"cuda:{a.strip()}  ({b.strip()})" for a, b in (line.split(",", 1) for line in out.splitlines() if "," in line)]
    except (OSError, subprocess.SubprocessError):
        found = []
    return found or ["cuda:0"]


def torch_gpus() -> list[dict]:
    """Every GPU PyTorch can train on, NVIDIA (CUDA) or AMD (ROCm, which uses the same device names)."""
    # Query in a child process: on Windows, importing torch/numpy on a background thread
    # deadlocks while another thread is blocked reading sys.stdin.
    probe = (
        "import json, torch; "
        "print(json.dumps("
        "[{'id': f'cuda:{i}', 'label': torch.cuda.get_device_name(i)} for i in range(torch.cuda.device_count())] "
        "if torch.cuda.is_available() else [{'id': 'cpu', 'label': None}]"
        "))"
    )
    try:
        res = subprocess.run(
            [sys.executable, "-c", probe],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if lines:
            gpus = json.loads(lines[-1])
            if isinstance(gpus, list) and gpus:
                return gpus
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return [{"id": "cpu", "label": None}]


class Job:
    """One build process: start it, read its lines, ask it to stop."""

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.run_dir: Path | None = None
        self.stop_file: Path | None = None
        self.stopping = False
        self.stopped = False

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, name: str, audio: str, batch: int, steps: int, save_every: int, base: str, device: str,
              restart: bool = False, app_models: str = "", fixes: bool = True, noise: bool = False,
              colour: bool = True) -> None:
        self.run_dir, self.stopping = RUNS / f"voice-{name}", False
        self.stop_file = self.run_dir / f".stop-{uuid.uuid4().hex}"
        self.stopped = False
        cmd = [sys.executable, "-u", "-m", "causalvc.build", "--name", name, "--audio", audio, "--batch", str(batch),
               "--steps", str(steps), "--save-every", str(save_every), "--base", base, "--device", device]
        cmd += ["--restart"] if restart else []
        cmd += ["--bf16"]  # about 1.5x faster at the same quality; GPUs without native bfloat16 train in full precision
        # Natural breath noise, steadier tone, and less of the source voice leaking through; about a quarter slower.
        cmd += ["--perturb", "0.75", "--energy-loss", "20", "--decorr", "10"] if fixes else []
        # Random microphone colour on half of the inputs: the voice depends less on the microphone, keeps bright or
        # breathy voices clearer and sounds a little more natural for everyone (P89: MOS +0.08 over 40 sentences).
        cmd += ["--colour-aug", "0.5"] if colour else []
        # Save an average of the recent weights rather than the last step's, which varies from run to run
        # (P89: MOS +0.08 over the same run's last weights, clearer on a bright voice); costs no time.
        cmd += ["--ema", "0.999"]
        # Trained to stay quiet through clicks, snaps, claps and background noise; costs a little on whispers and laughs.
        cmd += ["--noise-aug", "0.6"] if noise else []
        cmd += ["--app-models", app_models] if app_models else []
        env = dict(os.environ, PYTHONPATH=str(CODE_ROOT), CAUSALVC_PROGRESS_SECONDS="1", PYTHONIOENCODING="utf-8",
                   CAUSALVC_STOP_FILE=str(self.stop_file))
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                     creationflags=flags)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            if line.startswith(("stopped before training", "stopped at step ")):
                self.stopped = True
            self.lines.put(line.rstrip("\n"))
        self.proc.wait()
        try:
            self.stop_file.unlink(missing_ok=True)
        finally:
            self.lines.put(None)  # finished

    def stop(self) -> None:
        """First call: ask for a checkpoint and a clean exit. Second call: end the process now."""
        if not self.running():
            return
        if self.stopping:
            self.proc.kill()
            self.stopped = True
            return
        self.stopping = True
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.touch()
        if not self.running():
            self.stop_file.unlink(missing_ok=True)


class ConfigError(ValueError):
    def __init__(self, code: str, message: str, field: str | None = None):
        super().__init__(message)
        self.code, self.field = code, field


def load_form() -> dict:
    """The form as last used, with defaults for anything missing or unreadable."""
    form = dict(DEFAULT_FORM)
    try:
        saved = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        saved = {}
    if isinstance(saved, dict):
        for key, default in DEFAULT_FORM.items():
            value = saved.get(key)
            if isinstance(default, bool):
                usable = isinstance(value, bool)
            elif isinstance(default, int):
                usable = isinstance(value, int) and not isinstance(value, bool)
            else:
                usable = isinstance(value, str)
            if usable:
                form[key] = value
    return form


def save_form(config: dict) -> None:
    keep = {key: config[key] for key in DEFAULT_FORM if key in config}
    SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.write_text(json.dumps(keep, indent=1), encoding="utf-8")


def check_config(args: dict, bases: list[str]) -> dict:
    """A build config from a window's form, checked; raises ConfigError naming the field."""
    if not isinstance(args, dict):
        raise ConfigError("bad_request", "config must be an object")
    name = str(args.get("name") or "").strip()
    # A pasted path may carry the quotes of Windows' "Copy as path".
    audio = str(args.get("audio") or "").strip().strip('"').strip()
    if not NAME.fullmatch(name):
        raise ConfigError("invalid_name", "Give the voice a name using letters, digits, '-', '_' or '.'", "name")
    if not audio or not Path(audio).exists():
        raise ConfigError("missing_audio", "The dataset path does not exist", "audio")
    numbers = {}
    for key in ("batch", "steps", "save_every"):
        value = args.get(key)
        if isinstance(value, str) and value.strip().isdigit():
            value = int(value.strip())
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigError("invalid_numbers", "Batch size, steps and save interval must be whole numbers of at least 1", key)
        numbers[key] = value
    base = str(args.get("base") or "default")
    if base != "default" and base not in bases:
        base = "default"
    return {
        "name": name,
        "audio": audio,
        **numbers,
        "base": base,
        "device": str(args.get("device") or "cuda:0"),
        "restart": bool(args.get("restart")),
        "app_models": str(args.get("app_models") or "").strip(),
        "fixes": bool(args.get("fixes", True)),
        "noise": bool(args.get("noise", False)),
        "colour": bool(args.get("colour", True)),
    }


class Service:
    """The queue and the current build, driven by commands; reports through ``emit``."""

    def __init__(self, emit, bases: list[str] | None = None, job: Job | None = None, updater=None):
        self.emit = emit
        self.updater = updater  # causalvc.update.Updater; None in tests
        self.bases = custom_bases() if bases is None else bases
        self.job = job or Job()
        self.pending: list[dict] = []
        self.current: dict | None = None
        self.phase = "idle"
        self.progress: dict | None = None
        self.exit_code: int | None = None
        self.next: dict | None = None  # build_now: starts once the current build has stopped

    def hello(self) -> None:
        self.emit("hello", {"form": load_form(), "bases": ["default", *self.bases], "gpus": [],
                            "update": self.updater.snapshot() if self.updater else None})
        self.emit_state()

    def emit_state(self) -> None:
        brief = lambda config: {"name": config["name"], "steps": config["steps"]}
        self.emit("state", {
            "phase": self.phase,
            "current": brief(self.current) if self.current else None,
            "queue": [brief(config) for config in self.pending],
            "progress": self.progress,
            "exit_code": self.exit_code,
            "next": brief(self.next) if self.next else None,
        })

    UPDATE_COMMANDS = ("check_update", "download_update", "install_update", "skip_update", "later_update",
                       "set_update_check")

    def update_command(self, cmd: str, args: dict) -> None:
        from causalvc.update import UpdateError

        u = self.updater
        if u is None:
            raise ConfigError("unavailable", "Updates are not available here")
        try:
            if cmd == "check_update":
                u.check(bool(args.get("manual", True)))
            elif cmd == "download_update":
                u.download()
            elif cmd == "install_update":
                if self.job.running() or self.pending:
                    raise ConfigError("busy", "Stop the running build (and clear the queue) before updating")
                pid = args.get("pid")
                u.install(pid if isinstance(pid, int) and not isinstance(pid, bool) else None)
            elif cmd == "skip_update":
                if not isinstance(args.get("version"), str):
                    raise ConfigError("bad_request", "'version' is required")
                u.skip(args["version"])
            elif cmd == "later_update":
                u.later()
            elif cmd == "set_update_check":
                if not isinstance(args.get("on"), bool):
                    raise ConfigError("bad_request", "'on' is required")
                u.set_check_at_launch(args["on"])
        except UpdateError as error:
            raise ConfigError(error.code, str(error)) from error

    def command(self, cmd: str, args: dict) -> None:
        try:
            if cmd in self.UPDATE_COMMANDS:
                self.update_command(cmd, args if isinstance(args, dict) else {})
                return
            if cmd in ("start", "queue"):
                config = check_config(args, self.bases)
                if cmd == "start" and self.job.running():
                    raise ConfigError("busy", "A build is already running; queue this one instead")
                save_form(config)
                if cmd == "start":
                    self.launch(config)
                else:
                    self.pending.append(config)
                    self.start_next()
                    if self.pending and self.pending[-1] is config:  # not started at once: it waits
                        self.emit("queued", {"name": config["name"], "position": len(self.pending)})
            elif cmd == "remove":
                index = args.get("index") if isinstance(args, dict) else None
                if not isinstance(index, int) or not 0 <= index < len(self.pending):
                    raise ConfigError("bad_request", "no queued build at that position")
                self.pending.pop(index)
            elif cmd == "move":
                moves = args if isinstance(args, dict) else {}
                source, target = moves.get("from"), moves.get("to")
                if not all(isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(self.pending)
                           for i in (source, target)):
                    raise ConfigError("bad_request", "no queued build at that position")
                self.pending.insert(target, self.pending.pop(source))
            elif cmd == "build_now":
                index = args.get("index") if isinstance(args, dict) else None
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self.pending):
                    raise ConfigError("bad_request", "no queued build at that position")
                chosen = self.pending.pop(index)
                if not self.job.running():
                    self.launch(chosen)
                else:
                    if self.next is not None:  # chosen again before the last switch finished
                        self.pending.insert(0, self.next)
                    elif self.current is not None:
                        # Back to the front of the queue; it resumes from the checkpoint it saves now.
                        self.pending.insert(0, {**self.current, "restart": False})
                    self.next = chosen
                    if not self.job.stopping:
                        self.phase = "stopping"
                        self.job.stop()
            elif cmd == "stop":
                if self.next is not None:
                    # Stop during a switch cancels the switch (the build is already saving and stopping;
                    # a second Stop ends it at once): the chosen build waits at the front of the queue.
                    self.pending.insert(0, self.next)
                    self.next = None
                elif self.job.running():
                    self.phase = "stopping"
                    self.job.stop()
            else:
                raise ConfigError("bad_request", f"unknown command: {cmd}")
        except ConfigError as error:
            self.emit("error", {"code": error.code, "message": str(error), "field": error.field})
            return
        self.emit_state()

    def launch(self, config: dict) -> None:
        self.job.start(**config)
        self.current, self.phase, self.progress, self.exit_code = config, "starting", None, None

    def start_next(self) -> None:
        if self.pending and not self.job.running() and not self.job.stopping:
            self.launch(self.pending.pop(0))

    def pump(self) -> bool:
        """Handle the build's waiting output; returns whether anything changed."""
        changed = False
        while True:
            try:
                line = self.job.lines.get_nowait()
            except queue.Empty:
                return changed
            changed = True
            if line is None:
                code = self.job.proc.returncode
                self.exit_code = code
                self.phase = "stopped" if self.job.stopped else "finished" if code == 0 else "failed"
                if self.next is not None:
                    chosen, self.next = self.next, None
                    self.launch(chosen)
                elif self.phase == "finished" and self.pending:
                    self.start_next()
                self.emit_state()
                continue
            match = PROGRESS.match(line)
            if match:
                step, total, percent, elapsed, remaining, rate = match.groups()
                self.progress = {"step": int(step), "total": int(total), "percent": int(percent),
                                 "elapsed": elapsed, "remaining": remaining, "rate": float(rate)}
                if self.phase == "starting":
                    self.phase = "training"
                self.emit_state()
            elif line.strip() and not any(noise in line for noise in NOISE):
                self.emit("log", {"text": line})

    def close(self, timeout: float = 600) -> None:
        """The window closed: a running build saves a checkpoint and stops (resumable), as with Stop."""
        if not self.job.running():
            return
        if not self.job.stopping:
            self.job.stop()
        try:
            self.job.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.job.proc.kill()


def main() -> None:
    out = sys.stdout
    lock = threading.Lock()

    def emit(event: str, data) -> None:
        with lock:
            out.write(json.dumps({"event": event, "data": data}, ensure_ascii=False) + "\n")
            out.flush()

    commands: queue.Queue = queue.Queue()

    def read_commands() -> None:
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict) and isinstance(message.get("cmd"), str):
                commands.put((message["cmd"], message.get("args") or {}))
        commands.put(None)  # the window closed

    try:
        from causalvc.update import Updater

        updater = Updater("training", emit)
    except Exception as error:  # never let the updater keep the trainer from starting
        print(f"updates unavailable: {error}", file=sys.stderr)
        updater = None
    service = Service(emit, updater=updater)
    service.hello()
    if updater is not None:
        updater.launch()
    threading.Thread(target=lambda: emit("gpus", {"gpus": torch_gpus()}), daemon=True).start()
    threading.Thread(target=read_commands, daemon=True).start()
    while True:
        try:
            item = commands.get(timeout=0.1)
        except queue.Empty:
            item = ()
        if item is None:
            break
        if item:
            service.command(*item)
        service.pump()
    service.close()


if __name__ == "__main__":
    # UTF-8 both ways whatever the system code page (paths can be in any language).
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    main()
