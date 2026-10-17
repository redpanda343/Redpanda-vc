import asyncio
import base64
import json
import queue
import secrets
import threading
import time

import numpy as np
import torch
from fastapi import Request, Response
from fastapi.responses import StreamingResponse

from rvc.infer.realtime_engine import RealtimeProcessor
from shared.tools.web_auth import current_user

MAX_PENDING_CHUNKS = 12
MAX_QUEUED_BLOCKS = 4
MAX_QUEUED_OUTPUTS = 40
MAX_CHUNK_SAMPLES = 1 << 15
STATS_SECONDS = 0.5
IDLE_SECONDS = 30.0
UNUSED_SECONDS = 120.0
NO_STORE = {"Cache-Control": "no-store"}
LIVE_SETTINGS = (
    "rms_mix_rate", "threshold", "input_gain_db", "output_gain_db", "input_noise_reduce",
    "output_noise_reduce", "f0_method",
)

_sessions = {}
_sessions_lock = threading.Lock()
_create_lock = threading.Lock()


class RealtimeSession:
    def __init__(self, user, settings):
        self.token = secrets.token_urlsafe(32)
        self.user = user
        self.settings = settings
        self.processor = RealtimeProcessor()
        self.benchmark = torch.backends.cudnn.benchmark
        self.created = time.monotonic()
        self.last_audio = self.created
        self.streaming = False
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.preparing = threading.Lock()
        self.blocks = queue.Queue()
        self.pending = {}
        self.next_seq = 0
        self.next_block = 0
        self.chunk = 0
        self.buffer = np.zeros(0, dtype=np.float32)
        self.lost = 0
        self.skipped = 0
        self.worker = None
        self.loop = None
        self.outputs = None
        try:
            self.model_rate, self.beatrice_path = self.processor.load_model(settings)
        except Exception:
            self.release()
            raise

    def expired(self):
        if self.closed.is_set():
            return True
        now = time.monotonic()
        if self.streaming:
            return now - self.last_audio > IDLE_SECONDS
        return now - self.created > UNUSED_SECONDS

    def start_stream(self, sample_rate, chunk, loop, outputs):
        if sample_rate % 100 or not 8000 <= sample_rate <= 192000:
            raise ValueError(f"The browser runs audio at {sample_rate} Hz, which real-time conversion cannot use.")
        if not 0 < chunk <= MAX_CHUNK_SAMPLES:
            raise ValueError("The browser sent an invalid audio chunk size.")
        with self.lock:
            if self.streaming or self.closed.is_set():
                raise RuntimeError("This real-time session has already started or ended.")
            self.streaming = True
        with self.preparing:
            if self.closed.is_set():
                raise RuntimeError("The real-time session was stopped.")
            processor = self.processor
            processor.sample_rate = int(sample_rate)
            processor.channels = 1
            processor.prepare(self.beatrice_path)
        self.chunk = int(chunk)
        self.loop = loop
        self.outputs = outputs
        self.last_audio = time.monotonic()
        self.worker = threading.Thread(target=self._work, name="realtime-web", daemon=True)
        self.worker.start()
        lookahead = 0 if processor.rvc is None else processor.rvc.lookahead_frames(self.settings["f0_method"])
        delay = (processor.crossfade_frame + processor.sola_search_frame) / processor.sample_rate + lookahead / 100
        return {"block": processor.block_frame, "sampleRate": processor.sample_rate, "delay": delay}

    def feed(self, seq, samples):
        if self.closed.is_set() or self.worker is None:
            return
        self.last_audio = time.monotonic()
        ready = []
        with self.lock:
            if seq < self.next_seq:
                self.lost += 1
                return
            self.pending[seq] = samples
            if len(self.pending) > MAX_PENDING_CHUNKS and self.next_seq not in self.pending:
                first = min(self.pending)
                self.lost += first - self.next_seq
                self._append(np.zeros((first - self.next_seq) * self.chunk, dtype=np.float32))
                self.next_seq = first
            while self.next_seq in self.pending:
                self._append(self.pending.pop(self.next_seq))
                self.next_seq += 1
            block = self.processor.block_frame
            while self.buffer.shape[0] >= block:
                ready.append((self.next_block, self.buffer[:block]))
                self.buffer = self.buffer[block:]
                self.next_block += 1
        for item in ready:
            while self.blocks.qsize() >= MAX_QUEUED_BLOCKS:
                try:
                    self.blocks.get_nowait()
                    self.skipped += 1
                except queue.Empty:
                    break
            self.blocks.put(item)

    def _append(self, samples):
        self.buffer = np.concatenate((self.buffer, samples))

    def _work(self):
        processor = self.processor
        while not self.closed.is_set():
            try:
                index, block = self.blocks.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                output = processor.process(block[:, None])
            except Exception as error:
                self._send(("failure", str(error)))
                self.closed.set()
                return
            pcm = (np.clip(output, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
            self._send(("audio", index, pcm))

    def _send(self, item):
        loop = self.loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._put, item)
        except RuntimeError:
            pass

    def _put(self, item):
        outputs = self.outputs
        while outputs.qsize() >= MAX_QUEUED_OUTPUTS:
            outputs.get_nowait()
            self.skipped += 1
        outputs.put_nowait(item)

    def stats(self):
        processor = self.processor
        return {
            "blockMs": processor.last_block_ms,
            "modelMs": processor.last_infer_ms,
            "lost": self.lost,
            "skipped": self.skipped,
        }

    def update(self, pitch=None, index_rate=None, **values):
        processor = self.processor
        for key, value in values.items():
            if key in LIVE_SETTINGS and value is not None:
                self.settings[key] = value
        if pitch is not None:
            self.settings["pitch"] = pitch
            processor.update_pitch(pitch)
        if index_rate is not None:
            self.settings["index_rate"] = index_rate
            processor.update_index_rate(index_rate)

    def release(self):
        self.closed.set()
        worker = self.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=5)
        with self.preparing:
            self.processor._release_models()
        torch.backends.cudnn.benchmark = self.benchmark


def create_session(user, settings):
    with _create_lock:
        with _sessions_lock:
            for session in _sessions.values():
                if session.user != user and not session.expired():
                    raise RuntimeError(
                        "Real-time conversion is already running for another user of this WebUI."
                    )
            stale = list(_sessions.values())
            _sessions.clear()
        for session in stale:
            session.release()
        session = RealtimeSession(user, settings)
        with _sessions_lock:
            _sessions[session.token] = session
        timer = threading.Timer(UNUSED_SECONDS, _close_unused, args=(session.token,))
        timer.daemon = True
        timer.start()
        return session.token


def _close_unused(token):
    with _sessions_lock:
        session = _sessions.get(token)
    if session is not None and not session.streaming:
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
        del _sessions[token]
    session.release()
    return True


def update_session(token, user, **values):
    session = get_session(token, user)
    if session is not None:
        session.update(**values)


def _event(name, payload):
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


def register_realtime_routes(app):
    first_new_route = len(app.router.routes)

    @app.get("/realtime/stream")
    async def realtime_stream(request: Request, token: str = "", rate: int = 0, chunk: int = 0):
        session = get_session(token, await current_user(app, request))
        if session is None:
            return Response("Real-time session not found", status_code=404, headers=NO_STORE)
        loop = asyncio.get_running_loop()
        outputs = asyncio.Queue()

        async def events():
            try:
                yield _event("status", {"message": "Warming up the model..."})
                info = await asyncio.to_thread(session.start_stream, rate, chunk, loop, outputs)
                yield _event("ready", info)
                last_stats = 0.0
                while not session.closed.is_set():
                    try:
                        item = await asyncio.wait_for(outputs.get(), timeout=STATS_SECONDS)
                    except asyncio.TimeoutError:
                        item = None
                    if item is not None and item[0] == "failure":
                        yield _event("failure", {"message": item[1]})
                        break
                    if item is not None:
                        yield f"event: audio\ndata: {item[1]} {base64.b64encode(item[2]).decode('ascii')}\n\n"
                    now = time.monotonic()
                    if now - last_stats >= STATS_SECONDS:
                        last_stats = now
                        yield _event("stats", session.stats())
                    if session.expired():
                        yield _event("failure", {"message": "No microphone audio arrived for 30 seconds."})
                        break
            except Exception as error:
                yield _event("failure", {"message": str(error)})
            finally:
                threading.Thread(target=close_session, args=(session.token, session.user), daemon=True).start()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={**NO_STORE, "X-Accel-Buffering": "no"},
        )

    @app.post("/realtime/audio")
    async def realtime_audio(request: Request):
        session = get_session(request.headers.get("x-realtime-token"), await current_user(app, request))
        if session is None:
            return Response("Real-time session not found", status_code=404, headers=NO_STORE)
        try:
            seq = int(request.headers.get("x-realtime-seq", ""))
        except ValueError:
            return Response("Missing chunk number", status_code=400, headers=NO_STORE)
        body = await request.body()
        if len(body) % 2 or len(body) > 2 * MAX_CHUNK_SAMPLES or seq < 0:
            return Response("Invalid audio chunk", status_code=400, headers=NO_STORE)
        session.feed(seq, np.frombuffer(body, dtype="<i2").astype(np.float32) / 32768.0)
        return Response(status_code=204, headers=NO_STORE)

    new_routes = app.router.routes[first_new_route:]
    del app.router.routes[first_new_route:]
    app.router.routes[0:0] = new_routes
