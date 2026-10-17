import asyncio
import base64
import json
import math
import queue
import secrets
import struct
import threading
import time
from urllib.parse import urlsplit

import numpy as np
from fastapi import Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from beatrice.inference import find_paraphernalia
from shared.tools.realtime_worker import RealtimeWorker, realtime_devices
from shared.tools.web_auth import current_user

MAX_PENDING_CHUNKS = 64
MAX_QUEUED_SECONDS = 0.5
MAX_CHUNK_SAMPLES = 1 << 16
MAX_AHEAD_SECONDS = 4.0
GAP_SECONDS = 0.5
STATS_SECONDS = 0.5
IDLE_SECONDS = 30.0
UNUSED_SECONDS = 120.0
HANDSHAKE_SECONDS = 10.0
NO_STORE = {"Cache-Control": "no-store"}
LIVE_SETTINGS = (
    "pitch", "index_rate", "rms_mix_rate", "threshold", "input_gain_db", "output_gain_db",
    "input_noise_reduce", "output_noise_reduce", "f0_method",
)

_sessions = {}
_sessions_lock = threading.Lock()
_create_lock = threading.Lock()


def _valid_rate(sample_rate):
    return isinstance(sample_rate, int) and not isinstance(sample_rate, bool) and sample_rate % 100 == 0 and 8000 <= sample_rate <= 192000


class RealtimeSession:
    def __init__(self, user, settings, device):
        self.token = secrets.token_urlsafe(32)
        self.user = user
        self.settings = dict(settings)
        self.device = device
        self.created = time.monotonic()
        self.last_audio = self.created
        self.streaming = False
        self.closed = threading.Event()
        self.failed = threading.Event()
        self.lock = threading.Lock()
        self.preparing = threading.Lock()
        self.releasing = threading.Lock()
        self.released = False
        self.blocks = queue.Queue()
        self.pending = {}
        self.next_offset = 0
        self.waiting_since = None
        self.ahead = 0
        self.next_block = 0
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lost = 0
        self.skipped = 0
        self.worker = None
        self.loop = None
        self.outputs = None
        self.output_limit = 1
        self.sample_rate = 0
        self.block_frame = 0
        self.last_stats = {}
        self.backend = RealtimeWorker(self.settings, device)

    def expired(self):
        if self.closed.is_set() or self.failed.is_set():
            return True
        now = time.monotonic()
        if self.streaming:
            return self.worker is not None and now - self.last_audio > IDLE_SECONDS
        return now - self.created > UNUSED_SECONDS

    def reserve_stream(self):
        with self.lock:
            if self.streaming or self.closed.is_set() or self.failed.is_set():
                raise RuntimeError("This real-time session has already started or ended.")
            self.streaming = True
            self.last_audio = time.monotonic()

    def start_stream(self, sample_rate, loop, outputs, reserved=False):
        if not _valid_rate(sample_rate):
            raise ValueError(f"The browser runs audio at {sample_rate} Hz, which real-time conversion cannot use.")
        if not reserved:
            self.reserve_stream()
        with self.preparing:
            if self.closed.is_set():
                raise RuntimeError("The real-time session was stopped.")
            info = self.backend.start(sample_rate)
            if self.closed.is_set():
                raise RuntimeError("The real-time session was stopped.")
            self.sample_rate = int(info["sampleRate"])
            self.block_frame = int(info["block"])
            self.ahead = int(MAX_AHEAD_SECONDS * self.sample_rate)
            capacity = max(1, math.ceil(MAX_QUEUED_SECONDS * self.sample_rate / self.block_frame))
            self.blocks = queue.Queue(maxsize=capacity)
            self.output_limit = capacity
            self.loop = loop
            self.outputs = outputs
            self.last_audio = time.monotonic()
            self.worker = threading.Thread(target=self._work, name=f"realtime-{self.device}", daemon=True)
            self.worker.start()
        return {**info, "device": self.device}

    def feed(self, offset, samples):
        now = time.monotonic()
        with self.lock:
            if self.closed.is_set() or self.failed.is_set() or self.worker is None:
                return False
            self.last_audio = now
            if offset + samples.shape[0] <= self.next_offset:
                self.lost += 1
                return True
            if offset > self.next_offset + self.ahead or len(self.pending) >= MAX_PENDING_CHUNKS:
                return False
            if offset < self.next_offset:
                samples = samples[self.next_offset - offset:]
                offset = self.next_offset
            self.pending[offset] = samples
            if self.next_offset not in self.pending:
                if self.waiting_since is None:
                    self.waiting_since = now
                elif now - self.waiting_since > GAP_SECONDS:
                    first = min(self.pending)
                    self.lost += 1
                    self._append(np.zeros(first - self.next_offset, dtype=np.float32))
                    self.next_offset = first
            advanced = False
            while self.next_offset in self.pending:
                chunk = self.pending.pop(self.next_offset)
                self._append(chunk)
                self.next_offset += chunk.shape[0]
                advanced = True
            for stale in [key for key in self.pending if key < self.next_offset]:
                del self.pending[stale]
                self.lost += 1
            if not self.pending:
                self.waiting_since = None
            elif advanced:
                self.waiting_since = now
            while self.buffer.shape[0] >= self.block_frame:
                item = (self.next_block, self.buffer[:self.block_frame])
                self.buffer = self.buffer[self.block_frame:]
                self.next_block += 1
                while True:
                    try:
                        self.blocks.put_nowait(item)
                        break
                    except queue.Full:
                        try:
                            self.blocks.get_nowait()
                            self.skipped += 1
                        except queue.Empty:
                            pass
        return True

    def _append(self, samples):
        self.buffer = np.concatenate((self.buffer, samples))

    def _work(self):
        while not self.closed.is_set():
            try:
                index, block = self.blocks.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.closed.is_set():
                return
            try:
                output, stats = self.backend.process(block)
                self.last_stats = stats
                pcm = (np.clip(output, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
                self._send(("audio", index, pcm))
            except Exception as error:
                self.failed.set()
                self._send(("failure", str(error)))
                return

    def _send(self, item):
        loop = self.loop
        if loop is None or loop.is_closed() or self.closed.is_set():
            return
        try:
            loop.call_soon_threadsafe(self._put, item)
        except RuntimeError:
            pass

    def _put(self, item):
        outputs = self.outputs
        if outputs is None or self.closed.is_set():
            return
        while outputs.qsize() >= self.output_limit:
            outputs.get_nowait()
            self.skipped += 1
        outputs.put_nowait(item)

    def stats(self):
        return {**self.last_stats, "lost": self.lost, "skipped": self.skipped, "device": self.device}

    def update(self, **values):
        changes = {key: value for key, value in values.items() if key in LIVE_SETTINGS and value is not None}
        if changes and not self.closed.is_set():
            self.backend.update(changes)
            self.settings.update(changes)

    def release(self):
        self.closed.set()
        with self.releasing:
            if self.released:
                return
            self.backend.close()
            with self.preparing:
                worker = self.worker
                if worker is not None and worker is not threading.current_thread():
                    worker.join()
                self.released = True


def _select_device(settings, busy):
    requested = str(settings.get("inference_device", "auto") or "auto")
    devices = [entry["value"] for entry in realtime_devices()]
    beatrice = find_paraphernalia(settings["model_path"]) is not None
    if beatrice and requested not in {"auto", "cpu"}:
        raise ValueError("Beatrice real-time inference uses the CPU. Select Automatic or CPU.")
    if requested == "auto":
        candidates = [device for device in devices if device.startswith("cuda:")] if not beatrice else []
        candidates = candidates or ["cpu"]
    else:
        if requested not in devices:
            raise ValueError("The selected inference device is not available.")
        candidates = [requested]
    for device in candidates:
        if device not in busy:
            return device
    raise RuntimeError("The selected inference devices are busy. Stop an existing real-time session or choose another device.")


def create_session(user, settings, previous_token=""):
    with _create_lock:
        with _sessions_lock:
            stale = [session for token, session in _sessions.items() if session.expired() or (token == previous_token and session.user == user)]
            for session in stale:
                del _sessions[session.token]
        for session in stale:
            session.release()
        with _sessions_lock:
            busy = {session.device for session in _sessions.values()}
        device = _select_device(settings, busy)
        session = RealtimeSession(user, settings, device)
        with _sessions_lock:
            _sessions[session.token] = session
        timer = threading.Timer(UNUSED_SECONDS, _close_unused, args=(session.token,))
        timer.daemon = True
        timer.start()
        return session.token


def _close_unused(token):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is not None and session.expired():
        close_session(token, session.user)


def get_session(token, user):
    with _sessions_lock:
        session = _sessions.get(token or "")
    if session is None or user is None or session.user != user or session.closed.is_set():
        return None
    return session


def close_session(token, user):
    with _sessions_lock:
        session = _sessions.get(token or "")
        if session is None or session.user != user:
            return False
        session.closed.set()
    try:
        session.release()
    finally:
        with _sessions_lock:
            if _sessions.get(token) is session:
                del _sessions[token]
    return True


def update_session(token, user, **values):
    session = get_session(token, user)
    if session is not None:
        session.update(**values)


def _event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def _same_origin(connection):
    origin = connection.headers.get("origin", "")
    host = connection.headers.get("host", "")
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return bool(host) and parsed.scheme in {"http", "https"} and parsed.netloc.casefold() == host.casefold()


def _close_in_background(session):
    threading.Thread(target=close_session, args=(session.token, session.user), daemon=True).start()


async def _next_output(session, outputs):
    if session.closed.is_set():
        return None
    try:
        return await asyncio.wait_for(outputs.get(), timeout=STATS_SECONDS)
    except asyncio.TimeoutError:
        return None


def register_realtime_routes(app):
    first_new_route = len(app.router.routes)

    @app.api_route("/realtime/ping", methods=["GET", "HEAD"])
    async def realtime_ping(request: Request):
        if await current_user(app, request) is None:
            return Response(status_code=401, headers=NO_STORE)
        return Response(status_code=204, headers=NO_STORE)

    @app.websocket("/realtime/ws")
    async def realtime_websocket(websocket: WebSocket):
        user = await current_user(app, websocket)
        if user is None or not _same_origin(websocket):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = None
        receiver = None
        sender = None
        warmup = None
        startup_message = None
        try:
            opening = await asyncio.wait_for(websocket.receive(), timeout=HANDSHAKE_SECONDS)
            text = opening.get("text", "")
            if not text or len(text) > 4096:
                await websocket.close(code=1008)
                return
            config = json.loads(text)
            if not isinstance(config, dict) or not _valid_rate(config.get("rate")):
                await websocket.close(code=1008)
                return
            token = config.get("token")
            if not isinstance(token, str) or len(token) > 256:
                await websocket.close(code=1008)
                return
            candidate = get_session(token, user)
            if candidate is None:
                await websocket.close(code=1008)
                return
            candidate.reserve_stream()
            session = candidate
            await websocket.send_json({"type": "accepted"})
            await websocket.send_json({"type": "status", "message": "Warming up the model..."})
            outputs = asyncio.Queue()
            warmup = asyncio.create_task(asyncio.to_thread(session.start_stream, config["rate"], asyncio.get_running_loop(), outputs, True))
            startup_message = asyncio.create_task(websocket.receive())
            completed, pending = await asyncio.wait({warmup, startup_message}, return_when=asyncio.FIRST_COMPLETED)
            if startup_message in completed:
                if startup_message.result()["type"] == "websocket.disconnect":
                    return
                raise ValueError("Wait for the model to finish warming up before sending audio.")
            startup_message.cancel()
            await asyncio.gather(startup_message, return_exceptions=True)
            info = warmup.result()
            await websocket.send_json({"type": "ready", **info})
            send_lock = asyncio.Lock()

            async def receive_audio():
                while not session.closed.is_set():
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    text = message.get("text")
                    if text is not None:
                        if len(text) > 128:
                            raise ValueError("Invalid connection probe.")
                        probe = json.loads(text)
                        if not isinstance(probe, dict) or probe.get("type") != "ping" or type(probe.get("id")) is not int or not 0 <= probe["id"] < 1 << 32:
                            raise ValueError("Invalid connection probe.")
                        async with send_lock:
                            await websocket.send_json({"type": "pong", "id": probe["id"]})
                        continue
                    data = message.get("bytes")
                    if data is None or not 10 <= len(data) <= 8 + 2 * MAX_CHUNK_SAMPLES or len(data) % 2:
                        raise ValueError("Invalid microphone audio chunk.")
                    offset = struct.unpack_from("<Q", data)[0]
                    samples = np.frombuffer(data, dtype="<i2", offset=8).astype(np.float32) / 32768.0
                    if not session.feed(offset, samples):
                        raise ValueError("Microphone audio arrived too far ahead or the session has stopped.")

            async def send_audio():
                last_stats = 0.0
                while not session.closed.is_set():
                    item = await _next_output(session, outputs)
                    if item is not None and item[0] == "failure":
                        async with send_lock:
                            await websocket.send_json({"type": "failure", "message": item[1]})
                        return
                    if item is not None:
                        async with send_lock:
                            await websocket.send_bytes(struct.pack("<Q", item[1]) + item[2])
                    now = time.monotonic()
                    if now - last_stats >= STATS_SECONDS:
                        last_stats = now
                        async with send_lock:
                            await websocket.send_json({"type": "stats", **session.stats()})
                    if not session.failed.is_set() and session.expired():
                        async with send_lock:
                            await websocket.send_json({"type": "failure", "message": "No microphone audio arrived for 30 seconds."})
                        return

            receiver = asyncio.create_task(receive_audio())
            sender = asyncio.create_task(send_audio())
            done, pending = await asyncio.wait({receiver, sender}, return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        except (WebSocketDisconnect, asyncio.TimeoutError, asyncio.CancelledError):
            pass
        except Exception as error:
            if session is not None:
                try:
                    await websocket.send_json({"type": "failure", "message": str(error)})
                except (WebSocketDisconnect, RuntimeError):
                    pass
        finally:
            if session is not None:
                _close_in_background(session)
            tasks = (receiver, sender, warmup, startup_message)
            for task in tasks:
                if task is not None and not task.done():
                    task.cancel()
            try:
                await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
                await websocket.close()
            except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
                pass

    @app.get("/realtime/stream")
    async def realtime_stream(request: Request, token: str = "", rate: int = 0):
        session = get_session(token, await current_user(app, request))
        if session is None:
            return Response("Real-time session not found", status_code=404, headers=NO_STORE)
        if not _valid_rate(rate):
            return Response("Unsupported microphone sample rate", status_code=400, headers=NO_STORE)
        if session.streaming:
            return Response("Real-time session already started", status_code=409, headers=NO_STORE)
        loop = asyncio.get_running_loop()
        outputs = asyncio.Queue()

        async def events():
            claimed = False
            try:
                session.reserve_stream()
                claimed = True
                yield _event("status", {"message": "Warming up the model..."})
                info = await asyncio.to_thread(session.start_stream, rate, loop, outputs, True)
                yield _event("ready", info)
                last_stats = 0.0
                while not session.closed.is_set():
                    item = await _next_output(session, outputs)
                    if item is not None and item[0] == "failure":
                        yield _event("failure", {"message": item[1]})
                        break
                    if item is not None:
                        yield f"event: audio\ndata: {item[1]} {base64.b64encode(item[2]).decode('ascii')}\n\n"
                    now = time.monotonic()
                    if now - last_stats >= STATS_SECONDS:
                        last_stats = now
                        yield _event("stats", session.stats())
                    if not session.failed.is_set() and session.expired():
                        yield _event("failure", {"message": "No microphone audio arrived for 30 seconds."})
                        break
            except Exception as error:
                yield _event("failure", {"message": str(error)})
            finally:
                if claimed:
                    _close_in_background(session)

        return StreamingResponse(events(), media_type="text/event-stream", headers={**NO_STORE, "X-Accel-Buffering": "no"})

    @app.post("/realtime/stop")
    async def realtime_stop(request: Request):
        if not _same_origin(request):
            return Response("Cross-origin request denied", status_code=403, headers=NO_STORE)
        user = await current_user(app, request)
        token = request.headers.get("x-realtime-token")
        if user is None or not await asyncio.to_thread(close_session, token, user):
            return Response("Real-time session not found", status_code=404, headers=NO_STORE)
        return Response(status_code=204, headers=NO_STORE)

    @app.post("/realtime/audio")
    async def realtime_audio(request: Request):
        session = get_session(request.headers.get("x-realtime-token"), await current_user(app, request))
        if session is None:
            return Response("Real-time session not found", status_code=404, headers=NO_STORE)
        try:
            offset = int(request.headers.get("x-realtime-offset", ""))
        except ValueError:
            return Response("Missing audio offset", status_code=400, headers=NO_STORE)
        limit = 2 * MAX_CHUNK_SAMPLES
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            return Response("Audio chunk is too large", status_code=413, headers=NO_STORE)
        body = bytearray()
        async for part in request.stream():
            body.extend(part)
            if len(body) > limit:
                return Response("Audio chunk is too large", status_code=413, headers=NO_STORE)
        if not body or len(body) % 2 or offset < 0 or offset >= 1 << 64:
            return Response("Invalid audio chunk", status_code=400, headers=NO_STORE)
        if not session.feed(offset, np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0):
            return Response("Audio chunk is too far ahead or the session has stopped", status_code=400, headers=NO_STORE)
        return Response(status_code=204, headers=NO_STORE)

    new_routes = app.router.routes[first_new_route:]
    del app.router.routes[first_new_route:]
    app.router.routes[0:0] = new_routes
