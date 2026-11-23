"""Prepare local audio devices and control the realtime processor on one thread."""

import gc
import queue
import sys
import threading

from shared.desktop.paths import ensure_zluda, require_path
from shared.tools.causal_voices import PITCH_MAP_DEFAULTS, VOICE_DEFAULTS, is_causal_voice


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


class Realtime:
    def __init__(self, emit, reply, assets):
        self.emit, self.reply = emit, reply
        self.assets = assets
        self.errors = queue.Queue()
        self.requests = queue.Queue()
        self.engine = None
        self.inputs, self.outputs = {}, {}
        self.reported_shared = False
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def request(self, ident, action, args):
        self.requests.put((ident, action, args))

    def _work(self):
        actions = {"audio_devices": self.devices, "audio_start": self.start,
                   "audio_stop": self.stop, "audio_update": self.update}
        while True:
            try:
                request = self.requests.get(timeout=0.15 if self.engine and self.engine.running else None)
            except queue.Empty:
                self.publish()
                continue
            if request is None:
                self.stop()
                return
            ident, action, args = request
            try:
                self.reply(ident, result=actions[action](args))
            except (Exception, SystemExit) as error:
                self.reply(ident, error=str(error) or type(error).__name__)

    def devices(self, args):
        from shared.desktop.devices import audio_routes

        if self.engine and self.engine.running and args.get("rescan"):
            raise RuntimeError("Stop realtime conversion before rescanning audio devices.")
        data, self.inputs, self.outputs = audio_routes(bool(args.get("all_drivers")), bool(args.get("rescan")))
        return data

    def start(self, args):
        ensure_zluda()
        _release_cached_converter()
        from shared.desktop.audio import AudioEngine

        if self.engine and self.engine.running:
            raise RuntimeError("Realtime conversion is already running.")
        model = require_path(args.get("model_path"), "a voice model")
        causal = is_causal_voice(model)
        raw_index = str(args.get("index_path") or "").strip().strip('"')
        index_path = str(require_path(raw_index, "a feature index")) if raw_index and not causal else ""
        settings = {
            "model_path": str(model), "index_path": index_path, "passthrough": bool(args.get("passthrough")),
            "speaker_id": int(args.get("speaker_id", 0)), "embedder_model": args.get("embedder_model", "contentvec"),
            "f0_method": args.get("f0_method", "rmvpe"), "use_fp32": bool(args.get("use_fp32")),
            "rms_mix_rate": float(args.get("rms_mix_rate", 0.0)),
            "block_time": float(args.get("block_time", 0.04 if causal else 0.3)),
            "crossfade_time": float(args.get("crossfade_time", 0.01)),
            "extra_time": float(args.get("extra_time", 3.0)),
            "input_noise_reduce": bool(args.get("input_noise_reduce")),
            "exclusive_input": bool(args.get("exclusive_input")),
            "exclusive_output": bool(args.get("exclusive_output")),
        }
        if not causal:
            settings.update(VOICE_DEFAULTS)
        settings.update({key: float(args[key]) for key in VOICE_DEFAULTS if key in args})
        if not (0.01 <= settings["block_time"] <= 1.5):
            raise ValueError("Block time must be between 0.01 and 1.5 seconds.")
        for role, choices in (("input", self.inputs), ("output", self.outputs), ("monitor", self.outputs)):
            label = args.get(f"{role}_label")
            if role == "monitor" and not label:
                settings["monitor_device"], settings["monitor_selectors"] = None, None
                continue
            if label not in choices:
                raise ValueError(f"Select an audio {role} device.")
            settings[f"{role}_device"], settings[f"{role}_selectors"] = choices[label]
        while True:
            try:
                self.errors.get_nowait()
            except queue.Empty:
                break
        self.reported_shared = False
        self.emit("realtime", {"state": "starting"})
        self.engine = AudioEngine(self.errors)
        try:
            self.assets.ensure(rmvpe=not causal and settings["f0_method"] == "rmvpe")
            self.engine.start(settings)
        except (Exception, SystemExit):
            self.stop()
            raise
        self.publish()

    def update(self, args):
        if not self.engine or not self.engine.running:
            return
        engine = self.engine
        updates = {key: float(args[key]) for key in VOICE_DEFAULTS if key in args}
        for key in ("passthrough", "input_noise_reduce"):
            if key in args:
                updates[key] = bool(args[key])
        if "rms_mix_rate" in args:
            updates["rms_mix_rate"] = float(args["rms_mix_rate"])
        if "f0_method" in args:
            self.assets.ensure(rmvpe=engine.causal is None and args["f0_method"] == "rmvpe")
            updates["f0_method"] = args["f0_method"]
        if updates.get("input_noise_reduce"):
            engine.ensure_rnnoise()
        engine.settings.update(updates)
        if "pitch" in updates:
            engine.update_pitch(updates["pitch"] if engine.causal is not None else int(updates["pitch"]))
        if "index_rate" in updates:
            engine.update_index_rate(updates["index_rate"])
        if updates.keys() & PITCH_MAP_DEFAULTS.keys():
            engine.update_pitch_map()

    def publish(self):
        engine = self.engine
        if engine is None:
            return
        messages = []
        while True:
            try:
                message = self.errors.get_nowait()
            except queue.Empty:
                break
            print(message, flush=True)
            messages.append(message)
        if not engine.running:
            error_text = None
            if messages:
                lines = [line.strip() for line in messages[-1].splitlines() if line.strip()]
                error_text = lines[-1] if lines else messages[-1]
            self.stop(error=error_text or "Realtime conversion stopped unexpectedly. Check Logs for details.")
            return
        notice = None
        if not self.reported_shared and engine.shared_fallbacks:
            self.reported_shared = True
            notice = "WASAPI exclusive unavailable for " + ", ".join(engine.shared_fallbacks) + "; using shared mode."
        elif messages:
            lines = [line.strip() for line in messages[-1].splitlines() if line.strip()]
            notice = lines[-1] if lines else None
        payload = {
            "state": "running", "sampleRate": engine.sample_rate, "latency": engine.algorithm_latency_ms,
            "dropouts": engine.dropouts, "input": engine.input_level, "output": engine.output_level,
            "shared": engine.shared_fallbacks,
        }
        if notice:
            payload["notice"] = notice
        self.emit("realtime", payload)

    def stop(self, _=None, *, error=None):
        if self.engine:
            self.engine.stop()
            self.engine = None
        payload = {"state": "idle"}
        if error:
            payload["error"] = error
        self.emit("realtime", payload)

    def close(self):
        self.requests.put(None)
        self.worker.join(timeout=12)
