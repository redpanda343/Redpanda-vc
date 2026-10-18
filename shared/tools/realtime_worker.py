import os
import pickle
import queue
import signal
import struct
import subprocess
import sys
import threading
import traceback
from pathlib import Path


MAX_MESSAGE_BYTES = 64 << 20
MAX_ERROR_BYTES = 16384
START_TIMEOUT = 180
PROCESS_TIMEOUT = 30


def realtime_devices():
    import torch

    devices = [
        {"value": f"cuda:{index}", "label": f"GPU {index}: {torch.cuda.get_device_name(index)}"}
        for index in range(torch.cuda.device_count())
    ]
    devices.append({"value": "cpu", "label": "CPU"})
    return devices


def _worker_environment(device):
    environment = os.environ.copy()
    if device == "cpu":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    elif device.startswith("cuda:"):
        index = int(device.split(":")[1])
        visible = environment.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            tokens = [token.strip() for token in visible.split(",") if token.strip()]
            if not 0 <= index < len(tokens):
                raise ValueError(f"GPU {index} is outside CUDA_VISIBLE_DEVICES.")
            environment["CUDA_VISIBLE_DEVICES"] = tokens[index]
        else:
            environment["CUDA_VISIBLE_DEVICES"] = str(index)
    else:
        raise ValueError(f"Unsupported inference device: {device}")
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    return environment


def _memory_events():
    try:
        return {key: int(value) for key, value in (
            line.split() for line in Path("/sys/fs/cgroup/memory.events").read_text().splitlines()
        )}
    except (OSError, ValueError):
        return {}


def _read_exact(stream, length):
    data = bytearray()
    while len(data) < length:
        part = stream.read(length - len(data))
        if not part:
            raise EOFError("The inference worker stopped unexpectedly. Check the server console.")
        data.extend(part)
    return data


def _receive(stream):
    length = struct.unpack("<I", _read_exact(stream, 4))[0]
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("Inference worker message is too large.")
    return pickle.loads(_read_exact(stream, length))


def _send(stream, value):
    data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    if len(data) > MAX_MESSAGE_BYTES:
        raise ValueError("Inference worker message is too large.")
    packet = memoryview(struct.pack("<I", len(data)) + data)
    while packet:
        written = stream.write(packet)
        if not written:
            raise BrokenPipeError("The inference worker pipe was closed.")
        packet = packet[written:]
    stream.flush()


class RealtimeWorker:
    def __init__(self, settings, device):
        self.device = device
        self.lock = threading.Lock()
        self.closed = threading.Event()
        self.failed = threading.Event()
        self.responses = queue.Queue()
        self.stage = "starting the inference worker"
        self.error_lock = threading.Lock()
        self.error_output = bytearray()
        self.error_done = threading.Event()
        self.memory_events = _memory_events()
        self.process_handle = subprocess.Popen(
            [sys.executable, "-u", "-m", "shared.tools.realtime_worker"],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=_worker_environment(device),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        self.reader = threading.Thread(target=self._read_responses, name="realtime-worker-output", daemon=True)
        self.error_reader = threading.Thread(target=self._read_errors, name="realtime-worker-errors", daemon=True)
        self.error_reader.start()
        self.reader.start()
        try:
            self._call("load", settings, timeout=START_TIMEOUT)
        except Exception:
            self.close()
            raise

    def _read_responses(self):
        try:
            while True:
                response = _receive(self.process_handle.stdout)
                if isinstance(response, dict) and isinstance(response.get("stage"), str):
                    self.stage = response["stage"]
                else:
                    self.responses.put(response)
        except Exception as error:
            self.failed.set()
            self.responses.put({"ok": False, "error": str(error), "worker_exit": True})

    def _read_errors(self):
        try:
            while True:
                data = self.process_handle.stderr.read(4096)
                if not data:
                    return
                with self.error_lock:
                    self.error_output.extend(data)
                    del self.error_output[:-MAX_ERROR_BYTES]
                try:
                    sys.stderr.write(data.decode("utf-8", errors="replace"))
                    sys.stderr.flush()
                except (OSError, ValueError, UnicodeError):
                    pass
        except (OSError, ValueError):
            pass
        finally:
            self.error_done.set()

    def _failure_message(self, reason, killed=False):
        process = self.process_handle
        try:
            code = process.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            code = None
        if code is not None:
            self.error_done.wait(timeout=0.5)
        message = f"Inference worker failed while {self.stage} on {self.device}. {reason}"
        if code is not None:
            if code < 0 and os.name != "nt":
                try:
                    status = signal.Signals(-code).name
                except ValueError:
                    status = f"signal {-code}"
                message += f" Exit: {status} ({code})."
            else:
                message += f" Exit code: {code}."
            if not killed and code == -getattr(signal, "SIGKILL", 9) and os.name != "nt":
                events = _memory_events()
                if events.get("oom_kill", 0) > self.memory_events.get("oom_kill", 0):
                    message += " The container reported an out-of-memory kill. Check the instance's system RAM limit."
                else:
                    message += " The process was killed externally. Check system RAM limits and host logs."
        with self.error_lock:
            details = self.error_output.decode("utf-8", errors="replace").strip()
        if details:
            message += f"\nWorker output:\n{details}"
        else:
            message += "\nNo worker error output was captured."
        return message

    def _call(self, command, payload=None, timeout=PROCESS_TIMEOUT):
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("The inference worker was stopped.")
            if self.failed.is_set():
                raise RuntimeError(self._failure_message("The worker closed its response pipe."))
            self.stage = {"load": "loading the model", "start": "warming up the model", "process": "converting audio", "update": "updating settings"}.get(command, command)
            if command in {"load", "start"}:
                self.memory_events = _memory_events()
            try:
                _send(self.process_handle.stdin, (command, payload))
                response = self.responses.get(timeout=timeout)
            except queue.Empty:
                self.process_handle.kill()
                self.failed.set()
                raise RuntimeError(self._failure_message(f"Timed out after {timeout} seconds during {command}.", killed=True)) from None
            except (BrokenPipeError, OSError) as error:
                self.failed.set()
                raise RuntimeError(self._failure_message(str(error))) from error
            if not response["ok"]:
                if response.get("worker_exit"):
                    raise RuntimeError(self._failure_message(response["error"]))
                raise RuntimeError(response["error"])
            return response.get("result")

    def start(self, sample_rate):
        info = self._call("start", int(sample_rate), timeout=START_TIMEOUT)
        return {**info, "device": self.device}

    def process(self, block):
        return self._call("process", block)

    def update(self, values):
        self._call("update", values)

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        process = self.process_handle
        acquired = self.lock.acquire(timeout=0.1)
        try:
            if acquired and process.poll() is None:
                try:
                    _send(process.stdin, ("close", None))
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if acquired:
                self.lock.release()
            self.reader.join(timeout=1)
            self.error_reader.join(timeout=1)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()


def _run_worker():
    incoming = sys.stdin.buffer
    outgoing = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    processor = None
    beatrice_path = None
    try:
        while True:
            command, payload = _receive(incoming)
            if command == "close":
                break
            try:
                result = None
                if command == "load":
                    from shared.cuda_sync import use_blocking_cuda_sync

                    use_blocking_cuda_sync()
                    _send(outgoing, {"stage": "importing PyTorch"})
                    import torch
                    _send(outgoing, {"stage": "initializing CUDA"})
                    import shared.zluda
                    _send(outgoing, {"stage": "importing inference modules"})
                    from rvc.infer.realtime_engine import RealtimeProcessor

                    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
                    processor = RealtimeProcessor()
                    _send(outgoing, {"stage": "loading the voice model and embedder"})
                    _, beatrice_path = processor.load_model(payload)
                elif command == "start":
                    processor.sample_rate = payload
                    processor.channels = 1
                    with torch.inference_mode():
                        processor.prepare(beatrice_path)
                    lookahead = 0 if processor.rvc is None else processor.rvc.pitch_lookahead_frames(processor.settings["f0_method"])
                    delay = (processor.crossfade_frame + processor.sola_search_frame) / payload + lookahead / 100
                    result = {"block": processor.block_frame, "sampleRate": payload, "delay": delay}
                elif command == "process":
                    with torch.inference_mode():
                        output = processor.process(payload[:, None])
                    result = (output, {"blockMs": processor.last_block_ms, "modelMs": processor.last_infer_ms})
                elif command == "update":
                    processor.settings.update(payload)
                    if "pitch" in payload:
                        processor.update_pitch(payload["pitch"])
                    if "index_rate" in payload:
                        processor.update_index_rate(payload["index_rate"])
                else:
                    raise ValueError(f"Unknown inference worker command: {command}")
                _send(outgoing, {"ok": True, "result": result})
            except (Exception, SystemExit) as error:
                traceback.print_exc(file=sys.stderr)
                _send(outgoing, {"ok": False, "error": f"{type(error).__name__}: {error}"})
    except (EOFError, BrokenPipeError):
        pass
    finally:
        if processor is not None:
            processor._release_models()
        outgoing.close()


if __name__ == "__main__":
    _run_worker()
