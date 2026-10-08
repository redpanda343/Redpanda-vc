import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil

from tabs.settings.sections.precision import get_precision
from tabs.train.rectified import experiment_path, positive_integer

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_process = None
_log_handle = None
_log_path = None
_experiment = None
_description = 'No Beatrice job has been started.'


def latest_export():
    if _experiment is None or not (_experiment / 'beatrice').is_dir():
        return None
    exports = [path for path in (_experiment / 'beatrice').glob('paraphernalia_*') if path.is_dir()]
    return max(exports, key=lambda path: path.stat().st_mtime, default=None)


def _status():
    global _log_handle
    code = _process.poll() if _process is not None else None
    active = _process is not None and code is None
    if not active and _log_handle is not None:
        _log_handle.close()
        _log_handle = None
    status = _description
    if _process is not None:
        status += ' Running.' if active else f' Finished with exit code {code}.'
    export = latest_export()
    if export is not None:
        status += f' Latest VST model folder: {export}'
    log = ''
    if _log_path is not None and _log_path.exists():
        with _log_path.open('rb') as handle:
            handle.seek(max(0, _log_path.stat().st_size - 16000))
            log = handle.read().decode('utf-8', errors='replace')
    return status, log, gr.update(interactive=not active), gr.update(interactive=active)


def status():
    with _lock:
        return _status()


def print_job_completion(process, description):
    print(f'{description} Finished with exit code {process.wait()}.', flush=True)


def launch(name, arguments):
    global _process, _log_handle, _log_path, _experiment, _description
    directory = experiment_path(name)
    with _lock:
        if _process is not None and _process.poll() is None:
            raise gr.Error('A Beatrice job is already running. Wait for it to finish or stop it first.')
        if _log_handle is not None:
            _log_handle.close()
        directory.mkdir(parents=True, exist_ok=True)
        _log_path = directory / 'beatrice_webui.log'
        _log_handle = _log_path.open('wb')
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        try:
            _process = subprocess.Popen(
                [sys.executable, '-u', '-m', 'rvc.beatrice.train', *arguments], cwd=ROOT,
                stdout=_log_handle, stderr=subprocess.STDOUT, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
        except Exception:
            _log_handle.close()
            _log_handle = None
            raise
        _experiment = directory
        _description = f'Training Beatrice: {name}.'
        threading.Thread(target=print_job_completion, args=(_process, _description), daemon=True).start()
        return _status()


def start(name, dataset, steps, batch, save_interval, workers, device):
    experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    device = str(device).strip().lower()
    if device not in {'auto', 'cpu'} and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise gr.Error('Device must be auto, cpu or cuda:N.')
    precision = get_precision() or 'fp32'
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    arguments = ['--model-name', str(name).strip(), '--dataset', dataset, '--precision', precision, '--device', device]
    for flag, value, label in (('--steps', steps, 'Training steps'), ('--batch-size', batch, 'Batch size'),
                               ('--save-interval', save_interval, 'Save interval'),
                               ('--workers', workers, 'CPU workers')):
        arguments.extend([flag, positive_integer(value, label)])
    return launch(name, arguments)


def stop():
    global _description
    with _lock:
        if _process is not None and _process.poll() is None:
            try:
                parent = psutil.Process(_process.pid)
                processes = parent.children(recursive=True) + [parent]
                for process in processes:
                    try:
                        process.terminate()
                    except psutil.NoSuchProcess:
                        pass
                _, alive = psutil.wait_procs(processes, timeout=3)
                for process in alive:
                    try:
                        process.kill()
                    except psutil.NoSuchProcess:
                        pass
                _process.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
            _description = 'Stopped. Resume starts from the last saved checkpoint; unsaved steps are lost.'
        return _status()


def beatrice_train_tab():
    gr.Markdown('### Beatrice')
    gr.Markdown('Fine-tunes a [Beatrice 2](https://prj-beatrice.com) low-latency voice conversion model with '
                '[Beatrice Trainer](https://huggingface.co/fierce-cats/beatrice-trainer) 2.0.0-rc.0 (MIT). Each save '
                'writes a `paraphernalia_*` folder under `logs/<model>/beatrice` that loads in the realtime GUI '
                '(select its `.toml` file), the Beatrice VST, VCClient or beatrice-client. The app downloads the trainer with its pretrained models, noise and '
                'impulse-response sets (about 440 MB) at startup, and training retries an interrupted download.')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-beatrice')
        device = gr.Textbox(label='Device', value='auto', info='auto, cpu or cuda:N. Beatrice trains on one GPU.')
    dataset = gr.Textbox(label='Dataset folder',
                         info='Audio files directly in this folder train one voice named after the model. A folder '
                              'holding only subfolders trains one voice per subfolder. WAV, FLAC, MP3, OGG, Opus and '
                              'AIFF are read as they are, so no preprocessing step is needed.')
    with gr.Row():
        steps = gr.Number(label='Training steps', value=10000, minimum=1, precision=0,
                          info='Upstream default 10000, about 40 minutes on an RTX 4090. Raise it to continue a '
                               'finished model.')
        batch = gr.Number(label='Batch size', value=8, minimum=1, precision=0,
                          info='4-second clips per step. Lower it to reduce GPU memory use.')
        save_interval = gr.Number(label='Save interval (steps)', value=2000, minimum=1, precision=0,
                                  info='Saves a resumable checkpoint, a VST model folder and TensorBoard previews.')
        workers = gr.Number(label='CPU workers', value=min(8, os.cpu_count() or 1), minimum=1, precision=0,
                            info='Data loader processes that decode and augment audio.')
    gr.Markdown('Training precision follows **Settings → Training → Precision**: fp32 trains in full precision, '
                "fp16 or bf16 turn on the trainer's FP16 mixed precision.")
    with gr.Row():
        train_button = gr.Button('Start / resume Beatrice training', variant='primary')
        stop_button = gr.Button('Stop current Beatrice job', interactive=False)
    state = gr.Textbox(label='Beatrice job status', interactive=False)
    log = gr.Textbox(label='Beatrice job log', lines=12, max_lines=20, interactive=False)
    outputs = [state, log, train_button, stop_button]
    train_button.click(start, [name, dataset, steps, batch, save_interval, workers, device], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    gr.Timer(2).tick(status, [], outputs, queue=False)
