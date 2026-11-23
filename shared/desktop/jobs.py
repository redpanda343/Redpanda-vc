"""Desktop commands for the existing batch conversion and training entry points."""

import gc
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import psutil

from shared.desktop.paths import (
    BATCH_AUDIO_SUFFIXES,
    CODE_ROOT,
    LOGS,
    MODELS,
    ROOT,
    absolute,
    ensure_zluda,
    model_name,
    require_path,
    write_settings,
)
from shared.tools.causal_voices import is_causal_voice

VALID_SLICING = frozenset({"Automatic", "Simple", "Skip", "Silence"})
VALID_NORMALIZATION = frozenset({"chunk", "none", "pre", "post"})
VALID_DATASET_FORMATS = frozenset({"WAV 32-bit float", "WAV 16-bit", "FLAC"})
GPU_PATTERN = re.compile(r"^(?:-|[0-9]+(?:-[0-9]+)*)$")


class Stopped(Exception):
    pass


def result_message(result):
    if "failed" in result.lower():
        raise RuntimeError(result)
    print(result, flush=True)
    return result


def _release_cached_converter():
    core = sys.modules.get("core")
    if core is None:
        return
    converter = getattr(core, "import_voice_converter", None)
    if converter is not None and converter.cache_info().currsize:
        converter.cache_clear()
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _terminate_tree(proc, timeout=3):
    if proc is None or proc.poll() is not None:
        return
    try:
        root = psutil.Process(proc.pid)
        processes = list(reversed(root.children(recursive=True))) + [root]
    except psutil.Error:
        try:
            proc.terminate()
        except OSError:
            pass
        return
    for process in processes:
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(processes, timeout=timeout)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    if alive:
        psutil.wait_procs(alive, timeout=timeout)


def _format_duration(seconds):
    if seconds is None or seconds < 0:
        return None
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _read_progress(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _format_rvc_progress(progress, status_text=None):
    step = int(progress.get("step", 0))
    total = max(1, int(progress.get("total", 1)))
    pct = min(100.0, (step * 100.0) / total)
    pct_str = f"{pct:.0f}%" if pct >= 10 or pct == int(pct) else f"{pct:.1f}%"
    parts = [f"{pct_str} of 100%", f"{step:,} / {total:,} steps"]
    if progress.get("epoch"):
        parts.append(f"Epoch {progress['epoch']}")
    rate = progress.get("rate")
    if rate:
        parts.append(f"{rate:.1f} step/s")
    eta = _format_duration(progress.get("eta"))
    if eta:
        parts.append(f"ETA {eta}")
    losses = progress.get("losses")
    if isinstance(losses, dict):
        parts.append(f"g {losses.get('g', 0):.2f} · d {losses.get('d', 0):.2f} · mel {losses.get('mel', 0):.2f}")
    prefix = f"{status_text} · " if status_text else ""
    return prefix + " · ".join(parts)


class Jobs:
    def __init__(self, emit, assets):
        self.emit = emit
        self.assets = assets
        self.lock = threading.RLock()
        self.active = None
        self.stop_requested = threading.Event()
        self.train_name = None
        self.causal_job = None
        self.stage_proc = None

    def status(self, state, text, **values):
        self.emit("job", {"kind": self.active, "state": state, "message": text, **values})

    def start(self, kind, arguments):
        operation = {"infer": self.infer, "rvc": self.train_rvc, "causal": self.train_causal}[kind]
        with self.lock:
            if self.active is not None:
                raise RuntimeError("Wait for the current job to finish.")
            self.active = kind
            self.stop_requested.clear()
        self.status("working", "Preparing…", stoppable=kind != "infer")
        threading.Thread(target=self._run, args=(operation, arguments), daemon=True).start()

    def _run(self, operation, arguments):
        final = {"state": "finished", "message": "Finished"}
        try:
            final.update(operation(arguments))
        except Stopped:
            final.update(state="stopped", message="Stopped")
        except (Exception, SystemExit) as error:
            final.update(state="error", message=str(error) or type(error).__name__)
        finally:
            with self.lock:
                kind = self.active
                self.active = self.train_name = self.causal_job = self.stage_proc = None
            self.emit("job", {"kind": kind, **final})

    def check_stop(self):
        if self.stop_requested.is_set():
            raise Stopped()

    def _run_stage(self, command, failure_message, success_message):
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with self.lock:
            self.check_stop()
            proc = subprocess.Popen(command, cwd=str(ROOT), creationflags=flags)
            self.stage_proc = proc
        try:
            code = proc.wait()
        finally:
            with self.lock:
                if self.stage_proc is proc:
                    self.stage_proc = None
        if self.stop_requested.is_set():
            raise Stopped()
        if code != 0:
            raise RuntimeError(failure_message)
        print(success_message, flush=True)

    def control(self, command):
        from rvc.train.process import training_control

        with self.lock:
            kind = self.active
            if kind is None or kind == "infer":
                return
            if command == "stop":
                self.stop_requested.set()
            if kind == "causal" and self.causal_job:
                if command == "stop":
                    self.causal_job.stop()
                    self.status("stopping", "Stopping…", stoppable=True)
            elif self.train_name:
                control = {
                    "stop": training_control.stop_training,
                    "pause": training_control.pause_training,
                    "resume": training_control.resume_training,
                }[command]
                message = control(str(LOGS), self.train_name)
                print(message, flush=True)
            elif command == "stop" and self.stage_proc is not None:
                proc = self.stage_proc
                self.status("stopping", "Stopping…", stoppable=True)
                threading.Thread(target=_terminate_tree, args=(proc,), daemon=True).start()
            elif command == "stop":
                self.status("stopping", "Stopping…")

    def infer(self, args):
        ensure_zluda()
        from core import run_batch_infer_script, run_infer_script

        batch = bool(args.get("batch", False))
        source = require_path(args.get("input"), "an input folder" if batch else "an audio file", directory=batch)
        model = require_path(args.get("model_path"), "a voice model")
        extension = args.get("format", "wav").lower()
        if extension not in {"wav", "flac", "mp3", "ogg"}:
            raise ValueError("Unsupported export format.")
        if args.get("output"):
            target = absolute(args["output"])
        elif batch:
            target = source.parent / f"{source.name}_converted"
        else:
            target = source.with_name(f"{source.stem}_converted.{extension}")
        if batch:
            if source == target:
                raise ValueError("Select a separate output folder.")
            if not any(p.is_file() and p.suffix.lower() in BATCH_AUDIO_SUFFIXES for p in source.iterdir()):
                raise ValueError("No audio files were found in the input folder.")
            target.mkdir(parents=True, exist_ok=True)
        else:
            target = target.with_suffix(f".{extension}")
            if source == target:
                raise ValueError("Select a separate output file.")
            target.parent.mkdir(parents=True, exist_ok=True)

        self.status("working", "Converting audio…")
        if is_causal_voice(model):
            _release_cached_converter()
            from shared.desktop.convert import convert_causal

            convert_causal(model, source, target, args)
        else:
            raw_index = str(args.get("index_path") or "").strip().strip('"')
            index_path = str(require_path(raw_index, "a feature index")) if raw_index else ""
            normalization_db = float(args.get("normalization_db", -12))
            self.assets.ensure(
                rmvpe=args.get("f0_method", "rmvpe") == "rmvpe",
                vad=normalization_db > -12.0,
            )
            values = dict(
                pth_path=str(model), index_path=index_path,
                pitch=int(round(float(args.get("pitch", 0)))), index_rate=float(args.get("index_rate", 0)),
                normalization_db=normalization_db,
                protect=float(args.get("protect", 0.5)), f0_method=args.get("f0_method", "rmvpe"),
                split_audio=bool(args.get("split_audio")),
                export_format=extension.upper(), embedder_model=args.get("embedder_model", "contentvec"),
                sid=int(args.get("sid", 0)), seed=int(args.get("seed", 0)),
            )
            if batch:
                message = run_batch_infer_script(**values, input_folder=str(source), output_folder=str(target))
                result_message(message)
            else:
                with TemporaryDirectory(prefix=".redpanda-", dir=target.parent) as folder:
                    wav = target.parent / folder / "output.wav"
                    result_message(run_infer_script(**values, input_path=str(source), output_path=str(wav))[0])
                    wav.with_suffix(target.suffix).replace(target)
        if not target.exists():
            raise RuntimeError("Conversion did not produce the requested output.")
        return {"message": f"Conversion complete: {target.name}", "output": str(target), "directory": batch}

    def train_rvc(self, args):
        _release_cached_converter()
        from core import start_train_script
        from rvc.train.process.training_control import ACTIVE_STATES, get_training_state

        name = model_name(args.get("model_name"))
        exp_path = LOGS / name
        exp_dir = str(exp_path)
        raw_dataset = str(args.get("dataset_path") or "").strip().strip('"')
        selected = args.get("stage", "all")
        if selected == "all":
            if raw_dataset:
                stages = ("preprocess", "extract", "train", "index")
            elif (exp_path / "filelist.txt").is_file() and (exp_path / "config.json").is_file():
                stages = ("train", "index")
            else:
                raise ValueError("Select a dataset folder, or enter an existing prepared model name to resume.")
        else:
            stages = (selected,)
        stage_list = list(stages)

        rate, version = int(args.get("sample_rate", 32000)), args.get("version", "v2")
        cpu = int(args.get("cpu_cores") or max(1, min(32, (os.cpu_count() or 2) // 2)))
        gpu = str(args.get("gpu") or "0").strip()
        if not GPU_PATTERN.fullmatch(gpu):
            raise ValueError("GPU IDs must be digits separated by '-' (for example 0 or 0-1), or '-' for CPU.")
        vocoder = args.get("vocoder", "HiFi-GAN")
        embedder = args.get("embedder_model", "contentvec")
        algorithm = args.get("index_algorithm", "Auto")
        if version not in {"v1", "v2"}:
            raise ValueError(f"Unsupported RVC version: {version}")
        if version == "v1":
            rate, vocoder, embedder = 40000, "HiFi-GAN", "contentvec"

        for stage in stages:
            self.check_stop()
            if stage == "preprocess":
                dataset = require_path(raw_dataset, "a dataset folder", directory=True)
                slicing = args.get("cut_preprocess", "Automatic")
                if slicing not in VALID_SLICING:
                    raise ValueError(f"Unsupported slicing mode: {slicing}")
                normalization_mode = args.get("normalization_mode", "chunk")
                if normalization_mode not in VALID_NORMALIZATION:
                    raise ValueError(f"Unsupported normalization mode: {normalization_mode}")
                dataset_format = args.get("dataset_format", "WAV 32-bit float")
                if dataset_format not in VALID_DATASET_FORMATS:
                    raise ValueError(f"Unsupported dataset format: {dataset_format}")
                chunk_len = float(args.get("chunk_len", 3.0))
                overlap_len = float(args.get("overlap_len", 0.3))
                if chunk_len <= 0 or overlap_len < 0 or overlap_len >= chunk_len:
                    raise ValueError("Overlap length must be non-negative and smaller than chunk length.")
                self.status("working", "Preprocessing the dataset…", stage="preprocess", stages=stage_list, stoppable=True)
                if normalization_mode == "post":
                    self.assets.ensure(vad=True)
                self.check_stop()
                command = [
                    sys.executable,
                    str(CODE_ROOT / "shared" / "preprocess" / "preprocess.py"),
                    *map(
                        str,
                        [
                            exp_dir,
                            str(dataset),
                            rate,
                            cpu,
                            slicing,
                            chunk_len,
                            overlap_len,
                            normalization_mode,
                            dataset_format,
                            False,
                            -45.0,
                            0.3,
                            0.3,
                            "truncate",
                            50.0,
                            version,
                        ],
                    ),
                ]
                self._run_stage(
                    command,
                    f"Preprocessing failed for model {name}. Check Logs for details.",
                    f"Model {name} preprocessed successfully.",
                )
            elif stage == "extract":
                self.status("working", "Extracting features and pitch…", stage="extract", stages=stage_list, stoppable=True)
                f0_method = args.get("f0_method", "rmvpe")
                self.assets.ensure(rmvpe=f0_method == "rmvpe")
                self.check_stop()
                command = [
                    sys.executable,
                    str(CODE_ROOT / "shared" / "extract" / "extract.py"),
                    *map(
                        str,
                        [
                            exp_dir,
                            f0_method,
                            cpu,
                            gpu,
                            rate,
                            embedder,
                            int(args.get("include_mutes", 2)),
                            version,
                        ],
                    ),
                ]
                self._run_stage(
                    command,
                    f"Feature extraction failed for model {name}. Check Logs for details.",
                    f"Model {name} extracted successfully.",
                )
            elif stage == "index":
                self.status("working", "Building the retrieval index…", stage="index", stages=stage_list, stoppable=True)
                command = [
                    sys.executable,
                    str(CODE_ROOT / "rvc" / "train" / "process" / "extract_index.py"),
                    exp_dir,
                    algorithm,
                ]
                self._run_stage(
                    command,
                    f"Index generation failed for model {name}. Check Logs for details.",
                    f"Index file for {name} generated successfully.",
                )
            elif stage == "train":
                settings_update = {}
                if "precision" in args:
                    settings_update["precision"] = args["precision"]
                if "model_author" in args:
                    settings_update["model_author"] = args["model_author"]
                if settings_update:
                    write_settings(settings_update)
                pretrained = bool(args.get("pretrained", True))
                custom = pretrained and bool(args.get("g_pretrained_path") or args.get("d_pretrained_path"))
                generator = str(require_path(args.get("g_pretrained_path"), "pretrained generator")) if custom else None
                discriminator = str(require_path(args.get("d_pretrained_path"), "pretrained discriminator")) if custom else None
                if pretrained and not custom and vocoder == "HiFi-GAN":
                    self.assets.ensure(pretrain=True)
                total_steps = int(args["total_steps"])
                progress_path = exp_path / ".progress.json"
                try:
                    progress_path.unlink(missing_ok=True)
                except OSError:
                    pass
                train_env = dict(os.environ, REDPANDA_PROGRESS_FILE=str(progress_path))
                with self.lock:
                    self.check_stop()
                    message = start_train_script(
                        model_name=name, save_only_latest=bool(args.get("save_only_latest", True)),
                        save_every_weights=True,
                        total_steps=total_steps, sample_rate=rate, batch_size=int(args["batch_size"]),
                        gpu=gpu, pretrained=pretrained, cleanup=False, index_algorithm=algorithm,
                        vocoder=vocoder, version=version, save_every_steps=int(args["save_every_steps"]),
                        custom_pretrained=custom, g_pretrained_path=generator, d_pretrained_path=discriminator,
                        cache_data_in_gpu=bool(args.get("cache_data_in_gpu")),
                        checkpointing=bool(args.get("checkpointing")), warmup_steps=int(args.get("warmup_steps", 0)),
                        lr_decay_steps=int(args.get("lr_decay_steps", 0)), ema_decay=float(args.get("ema_decay", 0.999)),
                        lr_schedule=args.get("lr_schedule", "exponential"),
                        lr_final_ratio=float(args.get("lr_final_ratio", 0.1)),
                        max_batch_seconds=float(args.get("max_batch_seconds", 0)),
                        reset_optimizer=bool(args.get("reset_optimizer")),
                        env=train_env,
                    )
                    if not message.startswith("Training started"):
                        raise RuntimeError(message)
                    self.train_name = name
                previous_key = None
                self.status(
                    "running",
                    "Starting training…",
                    stage="train",
                    stages=stage_list,
                    step=0,
                    total=total_steps,
                    stoppable=True,
                    pausable=True,
                )
                try:
                    while True:
                        state = get_training_state(str(LOGS), name)
                        status = state.get("status")
                        if status not in ACTIVE_STATES:
                            break
                        progress = _read_progress(progress_path)
                        if progress:
                            step = int(progress.get("step", 0))
                            total = max(1, int(progress.get("total", total_steps)))
                            key = (status, step, total, progress.get("epoch"), progress.get("rate"), progress.get("losses"))
                            if key != previous_key:
                                prefix = "Paused" if status == "paused" else "Stopping" if status == "stopping" else None
                                self.status(
                                    status,
                                    _format_rvc_progress(progress, prefix),
                                    stage="train",
                                    stages=stage_list,
                                    step=step,
                                    total=total,
                                    epoch=progress.get("epoch"),
                                    rate=progress.get("rate"),
                                    eta=_format_duration(progress.get("eta")),
                                    losses=progress.get("losses"),
                                    stoppable=True,
                                    pausable=status in {"running", "paused"},
                                )
                                previous_key = key
                        elif status != previous_key:
                            self.status(
                                status,
                                state["message"],
                                stage="train",
                                stages=stage_list,
                                step=0,
                                total=total_steps,
                                stoppable=True,
                                pausable=status in {"running", "paused"},
                            )
                            previous_key = status
                        time.sleep(0.25)
                finally:
                    with self.lock:
                        self.train_name = None
                    try:
                        progress_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                if status == "stopped":
                    raise Stopped()
                if status != "finished":
                    raise RuntimeError(state["message"])
            else:
                raise ValueError("Unknown RVC training stage.")
        self.check_stop()
        final_steps = int(args["total_steps"]) if "train" in stages and args.get("total_steps") else None
        result = {
            "message": f"RVC workflow complete ({name})",
            "output": exp_dir,
            "directory": True,
            "stages": stage_list,
            "stage": "done",
        }
        if final_steps is not None:
            result["step"] = final_steps
            result["total"] = final_steps
        return result

    def train_causal(self, args):
        _release_cached_converter()
        from causalvc.trainer import Job, PROGRESS, check_config

        name = model_name(args.get("name"))
        base = str(args.get("base") or "default").strip().strip('"')
        if base != "default":
            base_path = Path(base).expanduser()
            if not base_path.exists():
                for parent in (MODELS / "custom-base", ROOT / "custom-base"):
                    candidate = parent / base_path
                    if candidate.exists():
                        base_path = candidate
                        break
            base_folder = require_path(base_path, "a base checkpoint folder", directory=True)
            if not all((base_folder / filename).is_file() for filename in ("args.json", "last.pt")):
                raise ValueError("The base folder must contain args.json and last.pt.")
            base = str(base_folder)
        audio = absolute(args.get("audio")) if args.get("audio") else ""
        settings = check_config(
            dict(args, name=name, audio=str(audio), base=base, restart=bool(args.get("restart", False))),
            [base],
        )
        total_target = int(settings["steps"])
        with self.lock:
            self.check_stop()
            job = Job()
            job.start(**settings)
            self.causal_job = job
        self.status(
            "running",
            "Preparing Casual Voice Changer…",
            stage="train",
            stages=["train"],
            step=0,
            total=total_target,
            stoppable=True,
        )
        while True:
            try:
                line = job.lines.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                break
            print(line, flush=True)
            if progress := PROGRESS.match(line):
                step_s, total_s, pct_s, elapsed, remaining, rate_s = progress.groups()
                step, total = int(step_s), max(1, int(total_s))
                pct = min(100.0, (step * 100.0) / total)
                pct_str = f"{pct:.0f}%" if pct >= 10 or pct == int(pct) else f"{pct:.1f}%"
                phase = "stopping" if job.stopping else "running"
                prefix = "Stopping · " if job.stopping else ""
                label = f"{prefix}{pct_str} of 100% · {step:,} / {total:,} steps · {float(rate_s):.1f} it/s · ETA {remaining}"
                self.status(
                    phase,
                    label,
                    stage="train",
                    stages=["train"],
                    step=step,
                    total=total,
                    rate=float(rate_s),
                    eta=remaining,
                    stoppable=True,
                )
        if job.stopped:
            raise Stopped()
        if job.proc.returncode:
            raise RuntimeError(f"Casual Voice Changer training failed (exit {job.proc.returncode}).")
        return {
            "message": f"Voice training complete ({name})",
            "output": str(job.run_dir / "voices"),
            "directory": True,
            "stage": "done",
            "stages": ["train"],
            "step": total_target,
            "total": total_target,
        }

    def close(self):
        if self.active is None:
            return
        self.control("stop")
        proc = self.stage_proc
        if proc is not None and proc.poll() is None:
            _terminate_tree(proc)
        job = self.causal_job
        if job and job.running():
            try:
                job.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                job.stop()
