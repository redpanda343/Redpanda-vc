"""Local and remote browser WebUI server reusing desktop/ui over HTTP + SSE."""

from __future__ import annotations

import base64
from collections import deque
from datetime import datetime
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import signal
import string
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, quote, urlsplit
import webbrowser

from shared.desktop.paths import CODE_ROOT, ROOT

UI_DIR = CODE_ROOT / "desktop" / "ui"
UPLOADS_DIR = ROOT / "datasets" / "uploads" if os.environ.get("REDPANDA_DATA_ROOT") else ROOT / "uploads"
FAVICON_PATH = (
    ROOT / "assets" / "red_panda_favicon.ico"
    if (ROOT / "assets" / "red_panda_favicon.ico").is_file()
    else CODE_ROOT / "assets" / "red_panda_favicon.ico"
)
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".aiff"}
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
}
MIME_OVERRIDES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
}


def _is_loopback(host: str) -> bool:
    return host.strip().lower() in LOOPBACK_HOSTS


def _safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name.strip()
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "_", base).strip(" .")
    return cleaned or "upload.bin"


def _safe_subfolder(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._-")
    return cleaned


def _windows_drives() -> list[dict]:
    drives = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            drives.append({"name": root, "path": root, "is_dir": True, "size": None, "audio": False})
    return drives


class BridgeWorker:
    """Manages one shared.desktop.bridge subprocess and fans out SSE events."""

    def __init__(self):
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.next_id = 1
        self.pending: dict[int, tuple[threading.Event, dict]] = {}
        self.seq = 0
        self.history: deque[tuple[int, dict]] = deque(maxlen=512)
        self.logs: deque[dict] = deque(maxlen=250)
        self.ready_data: dict | None = None
        self.ready_event = threading.Event()
        self.runtime_error: str | None = None
        self.last_job: dict | None = None
        self.last_realtime: dict | None = None
        self.subscribers: set[queue.Queue] = set()
        self.tb_port: int | None = None
        self.closed = False

    def start(self):
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                return
            self.closed = False
            self.ready_data = None
            self.runtime_error = None
            self.ready_event.clear()
            self.tb_port = None
            env = dict(
                os.environ,
                PYTHONUTF8="1",
                PYTHONUNBUFFERED="1",
                PYTHONDONTWRITEBYTECODE="1",
            )
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "shared.desktop.bridge"],
                cwd=str(ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                creationflags=flags,
            )
            proc = self.proc
        threading.Thread(target=self._read_stdout, args=(proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True).start()

    def wait_ready(self, timeout: float = 30.0) -> dict:
        self.start()
        if not self.ready_event.wait(timeout=timeout):
            raise RuntimeError("Timed out waiting for the Python engine to start.")
        if self.runtime_error:
            raise RuntimeError(self.runtime_error)
        if self.ready_data is None:
            raise RuntimeError("The Python engine closed before becoming ready.")
        return self.ready_data

    def _broadcast(self, payload: dict):
        with self.lock:
            self.seq += 1
            seq = self.seq
            event = payload.get("event")
            if event == "ready":
                self.ready_data = payload.get("data")
                self.ready_event.set()
            elif event == "runtime_error":
                self.runtime_error = str(payload.get("data") or "Runtime error")
                self.ready_event.set()
            elif event == "job":
                self.last_job = payload.get("data")
            elif event == "realtime":
                data = payload.get("data") or {}
                self.last_realtime = None if data.get("state") == "idle" else data
            elif event == "log":
                if payload.get("replace") and self.logs and self.logs[-1].get("replace"):
                    self.logs[-1] = payload
                else:
                    self.logs.append(payload)
            elif event == "closed":
                self.ready_data = None
                self.last_job = None
                self.last_realtime = None
                self.tb_port = None
                self.ready_event.set()
            self.history.append((seq, payload))
            targets = list(self.subscribers)
        for subscriber in targets:
            try:
                subscriber.put_nowait((seq, payload))
            except queue.Full:
                try:
                    subscriber.get_nowait()
                    subscriber.put_nowait((seq, payload))
                except queue.Empty:
                    pass

    def _read_stdout(self, proc: subprocess.Popen):
        assert proc.stdout is not None
        try:
            for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except ValueError:
                    self._broadcast({"event": "log", "data": line, "replace": False})
                    continue
                ident = message.get("id")
                if ident is not None:
                    with self.lock:
                        waiter = self.pending.pop(ident, None)
                    if waiter is not None:
                        ev, box = waiter
                        box.update(message)
                        ev.set()
                else:
                    self._broadcast(message)
        finally:
            with self.lock:
                still_current = self.proc is proc
                pending = list(self.pending.values())
                if still_current:
                    self.pending.clear()
                    self.proc = None
            for ev, box in pending:
                box["error"] = "The Python worker closed."
                ev.set()
            if still_current and not self.closed:
                self._broadcast({"event": "closed", "data": None})

    def _read_stderr(self, proc: subprocess.Popen):
        assert proc.stderr is not None
        buf = bytearray()
        try:
            while True:
                chunk = os.read(proc.stderr.fileno(), 4096)
                if not chunk:
                    break
                buf.extend(chunk)
                while True:
                    nl = buf.find(b"\n")
                    cr = buf.find(b"\r")
                    if nl == -1 and cr == -1:
                        break
                    if cr != -1 and (nl == -1 or cr < nl):
                        if cr + 1 == len(buf):
                            # Could be split CRLF; flush as CR only if more bytes won't pair with LF
                            line_bytes = bytes(buf[:cr])
                            del buf[: cr + 1]
                            is_cr = True
                        elif buf[cr + 1] == 10:
                            line_bytes = bytes(buf[:cr])
                            del buf[: cr + 2]
                            is_cr = False
                        else:
                            line_bytes = bytes(buf[:cr])
                            del buf[: cr + 1]
                            is_cr = True
                    else:
                        line_bytes = bytes(buf[:nl])
                        del buf[: nl + 1]
                        is_cr = False
                    text = line_bytes.decode("utf-8", errors="replace").strip()
                    if text:
                        self._broadcast({"event": "log", "data": text, "replace": is_cr})
            if buf:
                text = buf.decode("utf-8", errors="replace").strip()
                if text:
                    self._broadcast({"event": "log", "data": text, "replace": False})
        except OSError:
            pass

    def call(self, action: str, args: dict | None = None, timeout: float | None = None):
        if timeout is None:
            timeout = 900.0 if action == "audio_start" else 60.0
        with self.lock:
            proc = self.proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                raise RuntimeError("The Python worker is not running.")
            ident = self.next_id
            self.next_id += 1
            ev = threading.Event()
            box: dict = {}
            self.pending[ident] = (ev, box)
        payload = json.dumps({"id": ident, "action": action, "args": args or {}}, ensure_ascii=False) + "\n"
        try:
            with self.write_lock:
                proc.stdin.write(payload.encode("utf-8"))
                proc.stdin.flush()
        except OSError as error:
            with self.lock:
                self.pending.pop(ident, None)
            raise RuntimeError(f"Failed to send command to Python worker: {error}") from error
        if not ev.wait(timeout=timeout):
            with self.lock:
                self.pending.pop(ident, None)
            raise RuntimeError("The Python worker did not respond.")
        if box.get("error"):
            raise RuntimeError(box["error"])
        result = box.get("result")
        if action == "tensorboard_start" and isinstance(result, dict) and result.get("url"):
            parsed = urlsplit(result["url"])
            if parsed.port:
                with self.lock:
                    self.tb_port = parsed.port
            query = f"?{parsed.query}" if parsed.query else ""
            path = parsed.path or "/tensorboard/"
            result = {**result, "url": f"{path}{query}"}
        return result

    def ensure_tensorboard_port(self) -> int:
        with self.lock:
            port = self.tb_port
        if port:
            return port
        self.call("tensorboard_start", {})
        with self.lock:
            port = self.tb_port
        if not port:
            raise RuntimeError("TensorBoard port unavailable.")
        return port

    def subscribe(self, last_event_id: int | None = None) -> tuple[queue.Queue, list[tuple[int, dict]]]:
        q: queue.Queue = queue.Queue(maxsize=512)
        with self.lock:
            self.subscribers.add(q)
            replay: list[tuple[int, dict]] = []
            if last_event_id is not None and self.history and self.history[0][0] <= last_event_id + 1:
                replay = [(seq, item) for seq, item in self.history if seq > last_event_id]
            else:
                seq = self.seq
                if self.ready_data is not None:
                    replay.append((seq, {"event": "ready", "data": self.ready_data}))
                for log_item in self.logs:
                    replay.append((seq, log_item))
                if self.last_job is not None:
                    replay.append((seq, {"event": "job", "data": self.last_job}))
                if self.last_realtime is not None:
                    replay.append((seq, {"event": "realtime", "data": self.last_realtime}))
        return q, replay

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            self.subscribers.discard(q)

    def close(self):
        with self.lock:
            self.closed = True
            proc = self.proc
            self.proc = None
            subscribers = list(self.subscribers)
        for q in subscribers:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


class RedpandaWebServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address,
        worker: BridgeWorker,
        *,
        restrict_root: bool,
        auth_credentials: tuple[str, str] | None = None,
    ):
        self.worker = worker
        self.restrict_root = restrict_root
        self.auth_credentials = auth_credentials
        self.root = ROOT.resolve()
        super().__init__(server_address, RedpandaRequestHandler)

    def resolve_allowed_path(self, raw_path: str | None, *, default_to_root: bool = True) -> Path:
        text = (raw_path or "").strip().strip('"')
        if not text:
            if default_to_root:
                return self.root
            raise ValueError("Path is required.")
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        if self.restrict_root:
            try:
                resolved.relative_to(self.root)
            except ValueError as error:
                raise PermissionError(
                    "Access outside the Redpanda workspace is disabled when serving remotely."
                ) from error
        return resolved


class RedpandaRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: RedpandaWebServer

    def log_message(self, format, *args):
        return

    def _check_auth(self) -> bool:
        creds = self.server.auth_credentials
        if not creds:
            return True
        header = self.headers.get("Authorization", "")
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:].strip()).decode("utf-8")
                user, _, password = decoded.partition(":")
                if hmac.compare_digest(user, creds[0]) and hmac.compare_digest(password, creds[1]):
                    return True
            except Exception:
                pass
        body = b"Authentication required."
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Redpanda VC"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return False

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes, content_type: str, extra_headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_HEAD(self):
        self._dispatch()

    def do_OPTIONS(self):
        self._dispatch()

    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def _dispatch(self):
        if not self._check_auth():
            return
        parsed = urlsplit(self.path)
        path = parsed.path
        query = parse_qs(parsed.query, keep_blank_values=True)

        if path == "/tensorboard" or path.startswith("/tensorboard/") or path.startswith("/font-roboto/"):
            self._proxy_tensorboard(parsed)
            return

        if self.command == "GET":
            if path == "/api/events":
                self._handle_sse()
                return
            if path == "/api/workspace":
                self._send_json(200, {"root": str(self.server.root)})
                return
            if path == "/api/fs":
                self._handle_fs(query)
                return
            if path == "/api/download":
                self._handle_download(query)
                return
            self._handle_static(path)
            return

        if self.command == "HEAD":
            if path == "/api/download":
                self._handle_download(query, head_only=True)
                return
            self._handle_static(path)
            return

        if self.command == "POST":
            if path == "/api/connect":
                self._handle_connect()
                return
            if path == "/api/call":
                self._handle_call()
                return
            if path == "/api/upload":
                self._handle_upload(query)
                return

        self._send_json(404, {"error": "Not found"})

    def _handle_static(self, path: str):
        if path == "/favicon.ico" and FAVICON_PATH.is_file():
            self._send_bytes(200, FAVICON_PATH.read_bytes(), "image/x-icon")
            return
        rel = "index.html" if path in {"", "/"} else path.lstrip("/")
        if "/" in rel or "\\" in rel or rel.startswith("."):
            self._send_json(404, {"error": "Not found"})
            return
        target = (UI_DIR / rel).resolve()
        try:
            target.relative_to(UI_DIR.resolve())
        except ValueError:
            self._send_json(404, {"error": "Not found"})
            return
        if not target.is_file():
            self._send_json(404, {"error": "Not found"})
            return
        mime = MIME_OVERRIDES.get(target.suffix.lower()) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send_bytes(200, target.read_bytes(), mime, {"Cache-Control": "no-cache"})

    def _handle_connect(self):
        try:
            ready = self.server.worker.wait_ready(timeout=30.0)
            self._send_json(
                200,
                {
                    "root": ready.get("root", str(self.server.root)),
                    "python": ready.get("python", sys.executable),
                    "connected": True,
                },
            )
        except Exception as error:
            self._send_json(500, {"error": str(error)})

    def _handle_call(self):
        try:
            body = self._read_json_body()
            action = str(body.get("action") or "")
            args = body.get("args") or {}
            if not action:
                self._send_json(400, {"error": "Missing action."})
                return
            result = self.server.worker.call(action, args)
            self._send_json(200, {"result": result})
        except Exception as error:
            self._send_json(200, {"error": str(error) or type(error).__name__})

    def _handle_sse(self):
        last_id = None
        raw_last = self.headers.get("Last-Event-ID")
        if raw_last and raw_last.isdigit():
            last_id = int(raw_last)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        sub_queue, replay = self.server.worker.subscribe(last_id)
        try:
            for seq, payload in replay:
                self._write_sse(seq, payload)
            while True:
                try:
                    item = sub_queue.get(timeout=15.0)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                    continue
                if item is None:
                    break
                seq, payload = item
                self._write_sse(seq, payload)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            self.server.worker.unsubscribe(sub_queue)

    def _write_sse(self, seq: int, payload: dict):
        data = json.dumps(payload, ensure_ascii=False)
        frame = f"id: {seq}\ndata: {data}\n\n".encode("utf-8")
        self.wfile.write(frame)
        self.wfile.flush()

    def _handle_fs(self, query: dict[str, list[str]]):
        raw_path = (query.get("path") or [""])[0]
        ext_raw = (query.get("ext") or [""])[0]
        allowed_exts = {
            f".{item.strip().lower().lstrip('.')}"
            for item in ext_raw.split(",")
            if item.strip()
        }
        if raw_path == "__drives__" and os.name == "nt" and not self.server.restrict_root:
            self._send_json(
                200,
                {
                    "path": "__drives__",
                    "display_path": "This PC",
                    "parent": None,
                    "is_file": False,
                    "restricted": False,
                    "entries": _windows_drives(),
                },
            )
            return
        try:
            target = self.server.resolve_allowed_path(raw_path, default_to_root=True)
        except PermissionError as error:
            self._send_json(403, {"error": str(error)})
            return
        except Exception as error:
            self._send_json(400, {"error": str(error)})
            return

        if not target.exists():
            # Fall back to nearest existing parent directory (useful for suggested save paths)
            fallback = target.parent
            while not fallback.exists() and fallback != fallback.parent:
                fallback = fallback.parent
            if self.server.restrict_root:
                try:
                    fallback.relative_to(self.server.root)
                except ValueError:
                    fallback = self.server.root
            target = fallback if fallback.exists() else self.server.root

        if target.is_file():
            self._send_json(
                200,
                {
                    "path": str(target),
                    "name": target.name,
                    "parent": str(target.parent),
                    "is_file": True,
                    "size": target.stat().st_size,
                    "audio": target.suffix.lower() in AUDIO_EXTENSIONS,
                    "restricted": self.server.restrict_root,
                    "entries": [],
                },
            )
            return

        entries = []
        try:
            children = list(target.iterdir())
        except OSError as error:
            self._send_json(403, {"error": f"Cannot read folder: {error}"})
            return

        for child in children:
            try:
                is_dir = child.is_dir()
                if not is_dir and allowed_exts and child.suffix.lower() not in allowed_exts:
                    continue
                size = None if is_dir else child.stat().st_size
                entries.append(
                    {
                        "name": child.name,
                        "path": str(child),
                        "is_dir": is_dir,
                        "size": size,
                        "audio": (not is_dir) and (child.suffix.lower() in AUDIO_EXTENSIONS),
                    }
                )
            except OSError:
                continue
        entries.sort(key=lambda item: (not item["is_dir"], item["name"].lower()))

        parent = None
        if self.server.restrict_root:
            if target != self.server.root:
                parent = str(target.parent)
        else:
            if target.parent != target:
                parent = str(target.parent)
            elif os.name == "nt":
                parent = "__drives__"

        self._send_json(
            200,
            {
                "path": str(target),
                "display_path": str(target),
                "parent": parent,
                "is_file": False,
                "restricted": self.server.restrict_root,
                "entries": entries,
            },
        )

    def _handle_upload(self, query: dict[str, list[str]]):
        raw_name = (query.get("filename") or ["upload.bin"])[0]
        raw_folder = (query.get("folder") or [""])[0]
        filename = _safe_filename(raw_name)
        subfolder = _safe_subfolder(raw_folder) or datetime.now().strftime("%Y%m%d-%H%M%S")
        dest_dir = (UPLOADS_DIR / subfolder).resolve()
        try:
            dest_dir.relative_to(UPLOADS_DIR.resolve())
        except ValueError:
            self._send_json(400, {"error": "Invalid upload folder."})
            return
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / filename
        if dest_file.exists():
            stem, suffix = dest_file.stem, dest_file.suffix
            counter = 1
            while (dest_dir / f"{stem}_{counter}{suffix}").exists():
                counter += 1
            dest_file = dest_dir / f"{stem}_{counter}{suffix}"

        length_header = self.headers.get("Content-Length")
        if length_header is None:
            self._send_json(411, {"error": "Content-Length header is required for upload."})
            return
        remaining = int(length_header)
        if remaining < 0:
            self._send_json(400, {"error": "Invalid Content-Length."})
            return

        try:
            with dest_file.open("wb") as handle:
                while remaining > 0:
                    chunk = self.rfile.read(min(65536, remaining))
                    if not chunk:
                        break
                    handle.write(chunk)
                    remaining -= len(chunk)
        except Exception as error:
            dest_file.unlink(missing_ok=True)
            self._send_json(500, {"error": f"Upload failed: {error}"})
            return

        self._send_json(200, {"path": str(dest_file), "folder": str(dest_dir), "name": dest_file.name})

    def _handle_download(self, query: dict[str, list[str]], *, head_only: bool = False):
        raw_path = (query.get("path") or [""])[0]
        inline = (query.get("inline") or ["0"])[0] in {"1", "true", "yes"}
        try:
            target = self.server.resolve_allowed_path(raw_path, default_to_root=False)
        except PermissionError as error:
            self._send_json(403, {"error": str(error)})
            return
        except Exception as error:
            self._send_json(400, {"error": str(error)})
            return
        if not target.is_file():
            self._send_json(404, {"error": "File not found."})
            return

        file_size = target.stat().st_size
        mime = MIME_OVERRIDES.get(target.suffix.lower()) or mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        disposition = "inline" if inline else f"attachment; filename*=UTF-8''{quote(target.name)}"
        range_header = self.headers.get("Range")
        start, end = 0, file_size - 1
        status = 200
        if range_header and range_header.startswith("bytes=") and file_size > 0:
            spec = range_header[6:].split(",")[0].strip()
            lo, _, hi = spec.partition("-")
            try:
                if lo:
                    start = int(lo)
                    end = int(hi) if hi else file_size - 1
                else:
                    suffix = int(hi)
                    start = max(0, file_size - suffix)
                if start < 0 or start > end or start >= file_size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                end = min(end, file_size - 1)
                status = 206
            except ValueError:
                start, end, status = 0, file_size - 1, 200

        content_length = max(0, end - start + 1) if file_size > 0 else 0
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Disposition", disposition)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.send_header("Content-Length", str(content_length))
        self.end_headers()
        if head_only or content_length == 0:
            return
        with target.open("rb") as handle:
            handle.seek(start)
            remaining = content_length
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _proxy_tensorboard(self, parsed):
        if parsed.path == "/tensorboard":
            location = "/tensorboard/" + (f"?{parsed.query}" if parsed.query else "")
            self.send_response(302)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        try:
            port = self.server.worker.ensure_tensorboard_port()
        except Exception as error:
            self._send_bytes(502, f"TensorBoard unavailable: {error}".encode("utf-8"), "text/plain; charset=utf-8")
            return

        upstream_path = f"/tensorboard{parsed.path}" if parsed.path.startswith("/font-roboto/") else parsed.path
        target_path = upstream_path + (f"?{parsed.query}" if parsed.query else "")
        body = None
        length = int(self.headers.get("Content-Length") or 0)
        if length > 0:
            body = self.rfile.read(length)
        forward_headers = {}
        for key, value in self.headers.items():
            lower = key.lower()
            if lower in HOP_BY_HOP_HEADERS or lower in {"host", "authorization"}:
                continue
            forward_headers[key] = value
        forward_headers["Host"] = f"127.0.0.1:{port}"

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        try:
            conn.request(self.command, target_path, body=body, headers=forward_headers)
            upstream = conn.getresponse()
            data = upstream.read()
            self.send_response(upstream.status, upstream.reason)
            for key, value in upstream.getheaders():
                lower = key.lower()
                if lower in HOP_BY_HOP_HEADERS or lower == "content-length":
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        except Exception as error:
            self._send_bytes(502, f"TensorBoard proxy error: {error}".encode("utf-8"), "text/plain; charset=utf-8")
        finally:
            conn.close()


def serve_web(
    host: str = "127.0.0.1",
    port: int = 7860,
    *,
    open_browser: bool = True,
    auth: str | None = None,
    allow_external_paths: bool = False,
):
    if not (UI_DIR / "index.html").is_file():
        raise FileNotFoundError(f"Desktop UI assets not found in {UI_DIR}")

    auth_credentials = None
    if auth:
        user, sep, password = auth.partition(":")
        if not sep or not user or not password:
            raise ValueError("Auth must be formatted as username:password.")
        auth_credentials = (user, password)

    local = _is_loopback(host)
    restrict_root = (not local) and (not allow_external_paths)
    if not local and not auth_credentials:
        print(
            "WARNING: Serving on a non-loopback address without --auth. "
            "File browsing is restricted to the Redpanda workspace.",
            flush=True,
        )

    worker = BridgeWorker()
    print("Starting Redpanda Python engine…", flush=True)
    worker.wait_ready(timeout=30.0)

    server = RedpandaWebServer(
        (host, int(port)),
        worker,
        restrict_root=restrict_root,
        auth_credentials=auth_credentials,
    )
    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{actual_port}/"
    print(f"Redpanda WebUI running at {url}", flush=True)
    if open_browser:
        threading.Timer(0.25, lambda: webbrowser.open(url)).start()

    def _handle_signal(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except (ValueError, OSError):
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        worker.close()


def main(argv: list[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Launch the Redpanda browser WebUI (same interface as the desktop app).")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860).")
    parser.add_argument("--open", dest="open_browser", action="store_true", default=True, help="Open the WebUI in your browser automatically.")
    parser.add_argument("--no-open", dest="open_browser", action="store_false", help="Do not open a browser window on startup.")
    parser.add_argument("--auth", default=None, help="Optional HTTP Basic auth credentials formatted as user:pass.")
    parser.add_argument("--allow-external-paths", action="store_true", default=False, help="Allow browsing paths outside the workspace when bound to a non-loopback host.")
    args = parser.parse_args(argv)
    serve_web(
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        auth=args.auth,
        allow_external_paths=args.allow_external_paths,
    )


if __name__ == "__main__":
    main()
