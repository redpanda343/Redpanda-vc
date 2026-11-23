import json
import os
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone

import psutil


STATE_FILE_NAME = "training_state.json"
PROCESS_ACTIVE_STATES = {"running", "paused", "stopping"}
ACTIVE_STATES = PROCESS_ACTIVE_STATES | {"finalizing"}

_state_lock = threading.RLock()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _state_path(logs_root, model_name):
    return os.path.join(logs_root, model_name, STATE_FILE_NAME)


def _read_state(path):
    try:
        with open(path, "r", encoding="utf-8") as state_file:
            state = json.load(state_file)
        return state if isinstance(state, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{STATE_FILE_NAME}.",
        suffix=".tmp",
        dir=os.path.dirname(path),
        text=True,
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=4)
        os.replace(temporary_path, path)
    except Exception:
        try:
            os.remove(temporary_path)
        except OSError:
            pass
        raise


def _matching_process(process_data):
    try:
        process = psutil.Process(int(process_data["pid"]))
        expected_create_time = float(process_data["create_time"])
        if abs(process.create_time() - expected_create_time) > 0.01:
            return None
        return process if process.is_running() else None
    except (KeyError, TypeError, ValueError, psutil.Error):
        return None


def _root_process(state):
    return _matching_process(
        {
            "pid": state.get("pid"),
            "create_time": state.get("create_time"),
        }
    )


def _process_data(process):
    return {"pid": process.pid, "create_time": process.create_time()}


def _process_tree(root):
    descendants = root.children(recursive=True)
    return list(reversed(descendants)) + [root]


def _finish_training(logs_root, model_name, run_id, return_code):
    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = _read_state(path)
        if state.get("run_id") != run_id:
            return

        if state.get("status") in {"stopping", "stopped"}:
            status = "stopped"
            message = "Training stopped. Start training again to resume from the latest saved checkpoint."
        elif return_code == 0:
            status = "finished"
            message = "Training finished successfully."
        else:
            status = "failed"
            message = (
                f"Training exited with code {return_code}. Check the console logs for details."
            )

        state.update(
            {
                "status": status,
                "message": message,
                "return_code": return_code,
                "finished_at": _utc_now(),
                "suspended_processes": [],
            }
        )
        _write_state(path, state)


def _set_finished_message(logs_root, model_name, run_id, message):
    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = _read_state(path)
        if state.get("run_id") != run_id or state.get("status") not in {
            "finished",
            "finalizing",
        }:
            return
        state["status"] = "finished"
        state["message"] = message
        _write_state(path, state)


def _start_finalizing(logs_root, model_name, run_id):
    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = _read_state(path)
        if state.get("run_id") != run_id or state.get("status") != "finished":
            return
        state["status"] = "finalizing"
        state["message"] = "Training finished. Completing final steps."
        _write_state(path, state)


def _watch_training(process, logs_root, model_name, run_id, on_success):
    return_code = process.wait()
    _finish_training(logs_root, model_name, run_id, return_code)
    if return_code == 0 and on_success is not None:
        _start_finalizing(logs_root, model_name, run_id)
        try:
            message = on_success()
        except Exception as error:
            message = f"Training finished, but the completion action failed: {error}"
        _set_finished_message(logs_root, model_name, run_id, message)


def start_training(command, logs_root, model_name, cwd=None, on_success=None, env=None):
    if not model_name:
        return "Select a model before starting training."

    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = get_training_state(logs_root, model_name)
        if state.get("status") in ACTIVE_STATES:
            return state.get("message", "Training is already active for this model.")

        process = subprocess.Popen(command, cwd=cwd, env=env)
        process_info = psutil.Process(process.pid)
        run_id = uuid.uuid4().hex
        state = {
            "run_id": run_id,
            "model_name": model_name,
            "pid": process.pid,
            "create_time": process_info.create_time(),
            "status": "running",
            "message": "Training is running.",
            "started_at": _utc_now(),
            "suspended_processes": [],
        }
        _write_state(path, state)

        watcher = threading.Thread(
            target=_watch_training,
            args=(process, logs_root, model_name, run_id, on_success),
            name=f"training-watcher-{model_name}",
            daemon=True,
        )
        watcher.start()

    return f"Training started for model {model_name}."


def get_training_state(logs_root, model_name):
    if not model_name:
        return {"status": "idle", "message": "Select a model to view its training status."}

    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = _read_state(path)
        if not state:
            return {"status": "idle", "message": "Training is idle."}

        if (
            state.get("status") in PROCESS_ACTIVE_STATES
            and _root_process(state) is None
        ):
            previous_status = state.get("status")
            state.update(
                {
                    "status": "stopped" if previous_status == "stopping" else "failed",
                    "message": (
                        "Training stopped. Start training again to resume from the latest saved checkpoint."
                        if previous_status == "stopping"
                        else "The training process is no longer running. Check the console logs for details."
                    ),
                    "finished_at": _utc_now(),
                    "suspended_processes": [],
                }
            )
            _write_state(path, state)

        return state


def pause_training(logs_root, model_name):
    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = get_training_state(logs_root, model_name)
        if state.get("status") == "paused":
            return "Training is already paused."
        if state.get("status") != "running":
            return "There is no running training process to pause."

        root = _root_process(state)
        if root is None:
            return state.get("message", "The training process is no longer running.")

        suspended = []
        try:
            for process in _process_tree(root):
                process_data = _process_data(process)
                process.suspend()
                suspended.append(process_data)
        except psutil.Error as error:
            for process_data in reversed(suspended):
                process = _matching_process(process_data)
                if process is not None:
                    try:
                        process.resume()
                    except psutil.Error:
                        pass
            return f"Could not pause training safely: {error}"

        state.update(
            {
                "status": "paused",
                "message": "Training is paused. GPU memory remains allocated.",
                "paused_at": _utc_now(),
                "suspended_processes": suspended,
            }
        )
        _write_state(path, state)
        return state["message"]


def resume_training(logs_root, model_name):
    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = get_training_state(logs_root, model_name)
        if state.get("status") == "running":
            return "Training is already running."
        if state.get("status") != "paused":
            return "There is no paused training process to resume."

        root = _root_process(state)
        if root is None:
            return state.get("message", "The training process is no longer running.")

        errors = []
        for process_data in state.get("suspended_processes", []):
            process = _matching_process(process_data)
            if process is None:
                continue
            try:
                process.resume()
            except psutil.Error as error:
                errors.append(str(error))

        if errors:
            return "Training could not be fully resumed. Check the console logs and stop the run if it remains unresponsive."

        state.update(
            {
                "status": "running",
                "message": "Training resumed and is running.",
                "resumed_at": _utc_now(),
                "suspended_processes": [],
            }
        )
        _write_state(path, state)
        return state["message"]


def stop_training(logs_root, model_name, timeout=3):
    path = _state_path(logs_root, model_name)
    with _state_lock:
        state = get_training_state(logs_root, model_name)
        if state.get("status") == "finalizing":
            return "Training has finished and final steps are still running."
        if state.get("status") not in ACTIVE_STATES:
            return "There is no active training process to stop."

        root = _root_process(state)
        if root is None:
            return state.get("message", "The training process is no longer running.")

        try:
            processes = _process_tree(root)
        except psutil.Error as error:
            return f"Could not inspect the complete training process tree: {error}"

        state.update(
            {
                "status": "stopping",
                "message": "Stopping training...",
                "stopping_at": _utc_now(),
            }
        )
        _write_state(path, state)

        if state.get("suspended_processes"):
            for process_data in state["suspended_processes"]:
                process = _matching_process(process_data)
                if process is not None:
                    try:
                        process.resume()
                    except psutil.Error:
                        pass

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

        latest_state = _read_state(path)
        if latest_state.get("run_id") == state.get("run_id"):
            latest_state.update(
                {
                    "status": "stopped",
                    "message": "Training stopped. Start training again to resume from the latest saved checkpoint.",
                    "finished_at": _utc_now(),
                    "suspended_processes": [],
                }
            )
            _write_state(path, latest_state)

        return "Training stopped. Start training again to resume from the latest saved checkpoint."
