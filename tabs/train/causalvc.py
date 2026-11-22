from collections import deque
from pathlib import Path
from queue import Empty
from threading import Lock

import gradio as gr

from causalvc.trainer import ROOT, ConfigError, DEFAULT_FORM, Job, check_config
from shared.tools.realtime_worker import realtime_devices


_lock = Lock()
_job = Job()
_log = deque(maxlen=12)
_phase = "Ready"


def _poll_job():
    global _phase
    while True:
        try:
            line = _job.lines.get_nowait()
        except Empty:
            break
        if line is None:
            code = _job.proc.returncode
            _phase = "Stopped" if _job.stopping else "Finished" if code == 0 else f"Failed (exit {code})"
        else:
            _log.append(line)
    active = _phase in {"Training", "Stopping"}
    return (
        "\n".join((_phase, *_log)),
        gr.update(interactive=not active),
        gr.update(value="Force stop" if _phase == "Stopping" else "Stop", interactive=active),
        gr.Timer(active=active),
    )


def training_status():
    with _lock:
        return _poll_job()


def start_training(name, audio, steps, batch, save_every, base, device):
    global _job, _phase
    with _lock:
        if _job.running():
            raise gr.Error("Wait for the current voice to finish or stop it first.")
        base = str(base or "default").strip().strip('"')
        if base != "default":
            base_path = Path(base).expanduser()
            if not base_path.exists():
                base_path = ROOT / "custom-base" / base_path
            if not all((base_path / filename).is_file() for filename in ("last.pt", "args.json")):
                raise gr.Error("Select a base folder containing last.pt and args.json.")
            base = str(base_path.resolve())
        try:
            args = check_config(
                dict(name=name, audio=audio, steps=steps, batch=batch, save_every=save_every, base=base, device=device),
                [base],
            )
            args["audio"] = str(Path(args["audio"]).resolve())
            job = Job()
            job.start(**args)
        except (ConfigError, OSError) as error:
            raise gr.Error(str(error)) from error
        _job = job
        _log.clear()
        _phase = "Training"
        return _poll_job()


def stop_training():
    global _phase
    with _lock:
        if _job.running():
            _job.stop()
            _phase = "Stopping"
        return _poll_job()


def causalvc_train_tab():
    gr.Markdown("Fine-tune a Causal Voice Changer voice from an audio file or folder. The default pretrained base is for noncommercial use.")
    with gr.Row():
        name = gr.Textbox(label="Voice name")
        audio = gr.Textbox(label="Audio file or folder", placeholder="Path on the machine running Redpanda")
    with gr.Row():
        steps = gr.Number(label="Training steps", value=DEFAULT_FORM["steps"], minimum=1, precision=0)
        save_every = gr.Number(label="Save every (steps)", value=DEFAULT_FORM["save_every"], minimum=1, precision=0)
    with gr.Accordion("Advanced", open=False):
        devices = [(d["label"], d["value"]) for d in realtime_devices()]
        with gr.Row():
            batch = gr.Number(label="Batch size", value=DEFAULT_FORM["batch"], minimum=1, precision=0)
            device = gr.Dropdown(label="Device", choices=devices, value=devices[0][1])
            base = gr.Textbox(label="Base model", value="default", info="Default or a custom base folder.")
    with gr.Row():
        start = gr.Button("Start / resume", variant="primary")
        stop = gr.Button("Stop", interactive=False)
    status = gr.Textbox(label="Status", value="Ready", lines=4, interactive=False)
    timer = gr.Timer(1)
    outputs = [status, start, stop, timer]
    start.click(start_training, [name, audio, steps, batch, save_every, base, device], outputs, queue=False)
    stop.click(stop_training, outputs=outputs, queue=False)
    timer.tick(training_status, outputs=outputs, queue=False, show_progress="hidden")
