import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil
import torch

from tabs.settings.sections.precision import get_precision
from rvc.rectified.distributed import parse_devices
from rvc.rectified.resources import default_vocoder

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
VOCODER_CHOICES = ['Default NSF-HiFiGAN', 'Custom NSF-HiFiGAN', 'Mel previews only']
_process = None
_log_handle = None
_log_path = None
_description = 'No rectified-flow job has been started.'


def experiment_path(name):
    name = str(name).strip()
    if not name or Path(name).name != name or name in {'.', '..'} or any(c in name for c in '/\\:'):
        raise gr.Error('Enter a model name without folders or path separators.')
    return ROOT / 'logs' / name


def positive_integer(value, label):
    if float(value) != int(value) or int(value) < 1:
        raise gr.Error(f'{label} must be a positive whole number.')
    return str(int(value))


def _status():
    global _log_handle
    code = _process.poll() if _process is not None else None
    active = _process is not None and code is None
    if not active and _log_handle is not None:
        _log_handle.close()
        _log_handle = None
    status = _description
    if _process is not None:
        if code == 0 and _description.startswith('Preprocessing:'):
            status += ' Dataset preparation completed. Ready to extract content and F0.'
        else:
            status += ' Running.' if active else f' Finished with exit code {code}.'
    log = ''
    if _log_path is not None and _log_path.exists():
        with _log_path.open('rb') as handle:
            handle.seek(max(0, _log_path.stat().st_size - 16000))
            log = handle.read().decode('utf-8', errors='replace')
    return status, log, gr.update(interactive=not active), gr.update(interactive=not active), gr.update(interactive=not active), gr.update(interactive=active)


def status():
    with _lock:
        return _status()


def print_job_completion(process, description):
    code = process.wait()
    if code == 0 and description.startswith('Preprocessing:'):
        message = 'Dataset preparation completed. Ready to extract content and F0.'
    else:
        message = f'Finished with exit code {code}.'
    print(f'{description} {message}', flush=True)


def launch(name, module, arguments, description):
    global _process, _log_handle, _log_path, _description
    directory = experiment_path(name)
    with _lock:
        if _process is not None and _process.poll() is None:
            raise gr.Error('A rectified-flow job is already running. Wait for it to finish or stop it first.')
        if _log_handle is not None:
            _log_handle.close()
        directory.mkdir(parents=True, exist_ok=True)
        _log_path = directory / 'rectified_webui.log'
        _log_handle = _log_path.open('wb')
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
        try:
            _process = subprocess.Popen(
                [sys.executable, '-u', '-m', module, *arguments], cwd=ROOT,
                stdout=_log_handle, stderr=subprocess.STDOUT, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
            )
        except Exception:
            _log_handle.close()
            _log_handle = None
            raise
        _description = f'{description}: {name}.'
        threading.Thread(target=print_job_completion, args=(_process, _description), daemon=True).start()
        return _status()


def preprocess(name, dataset, workers, slicing=True):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    return launch(name, 'rvc.train.preprocess.preprocess',
                  [str(directory), dataset, '44100', positive_integer(workers, 'CPU workers'),
                   'Automatic' if slicing else 'Skip', 'False', 'False', '0.0', '10.0', '0.3', 'none', 'WAV'], 'Preprocessing')


def extract(name, method, workers, device, embedder):
    directory = experiment_path(name)
    if not (directory / 'sliced_audios').is_dir():
        raise gr.Error('Preprocess this experiment first.')
    gpu = device_id(device)
    return launch(name, 'rvc.train.extract.extract',
                  [str(directory), method, positive_integer(workers, 'CPU workers'), gpu,
                   '44100', embedder, '0', 'v2', '--rectified'], 'Extracting content and F0')


def device_id(device):
    try:
        devices = parse_devices(device)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return '-' if devices == ['cpu'] else '-'.join(item[5:] for item in devices)


def toggle_pretrained(enabled):
    from rvc.rectified.config import resolve_config

    filename = '44100_finetune.json' if enabled else '44100_standard.json'
    config = resolve_config(json.loads((ROOT / 'rvc' / 'configs' / 'rectified' / filename).read_text(encoding='utf-8')))
    return gr.update(value=config['flow']['max_batch_size'])


def resolve_vocoder(mode, path):
    if mode == 'Mel previews only':
        return ''
    if mode == 'Custom NSF-HiFiGAN':
        path = str(path or '').strip().strip('"')
        if not path or not Path(path).is_file():
            raise gr.Error('Choose an existing OpenVPI NSF-HiFiGAN checkpoint or converted vocoder export.')
        return path
    if mode != 'Default NSF-HiFiGAN':
        raise gr.Error('Choose a supported vocoder option.')
    try:
        return default_vocoder()
    except Exception as error:
        raise gr.Error(f'Could not download NSF-HiFiGAN vocoder: {error}') from error


def resolve_pretrained(directory, enabled):
    if not enabled or (directory / 'flow' / 'checkpoint.pth').is_file():
        return ''
    path = ROOT / 'rvc' / 'models' / 'pretraineds' / 'rectified' / 'pretrained.pth'
    if not path.is_file():
        raise gr.Error(f'Pretrained model not found. Place your Rectified Flow checkpoint at {path}.')
    return str(path)


def start(name, vocoder, batch, max_frames, epochs, save_every, device, compile_backbone=False,
          use_pretrained=False, vocoder_mode='Default NSF-HiFiGAN'):
    directory = experiment_path(name)
    device_id(device)
    if not (directory / 'filelist.txt').is_file():
        raise gr.Error('Extract features for this experiment first.')
    precision = get_precision() or 'fp32'
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    config_path = directory / 'rectified_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.is_file() else None
    if config:
        from rvc.rectified.config import resolve_config
        from rvc.rectified.flow_model import validate_model_config

        try:
            config = resolve_config(config)
            validate_model_config(config['flow']['model'])
        except ValueError as error:
            raise gr.Error(str(error)) from error
    pretrained = resolve_pretrained(directory, use_pretrained)
    vocoder = resolve_vocoder(vocoder_mode, vocoder)
    arguments = ['--model-name', str(name).strip(), '--vocoder', vocoder,
                 '--batch-size', positive_integer(batch, 'Max clips per batch'),
                 '--max-batch-frames', positive_integer(max_frames, 'Max frames per batch'),
                 '--epochs', positive_integer(epochs, 'Total epochs'),
                 '--save-every', positive_integer(save_every, 'Save interval'),
                 '--device', str(device).strip().lower(), '--precision', precision]
    if compile_backbone:
        arguments.append('--compile')
    if pretrained:
        arguments.extend(['--pretrained-flow', pretrained])
    return launch(name, 'rvc.rectified.train_flow', arguments, 'Training rectified flow')


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
            _description = 'Stopped. Resume starts from the last saved epoch; unsaved steps are lost.'
        return _status()


def rectified_train_tab():
    gr.Markdown('### Rectified Flow')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-flow')
        device = gr.Textbox(label='Device', value=','.join(f'cuda:{index}' for index in range(torch.cuda.device_count())) or 'cpu',
                            info='Detected GPUs are selected automatically. One GPU: cuda:0. Multiple GPUs: cuda:0,cuda:1. CPU: cpu.')
    with gr.Accordion('1. Prepare dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder')
        workers = gr.Number(label='CPU workers', value=4, minimum=1, precision=0)
        slicing = gr.Checkbox(label='Slice dataset', value=True,
                              info='Disable to keep full clips. Audio is always resampled to 44.1 kHz.')
        preprocess_button = gr.Button('Preprocess dataset')
        with gr.Row():
            method = gr.Dropdown(label='Pitch extractor', choices=['rmvpe', 'swift', 'pm'], value='rmvpe')
            embedder = gr.Dropdown(label='Content embedder', choices=['contentvec', 'spin-v2'], value='contentvec')
        extract_button = gr.Button('Extract content and F0')
    with gr.Accordion('2. Train rectified flow', open=True):
        vocoder_mode = gr.Dropdown(label='Audio preview vocoder', choices=VOCODER_CHOICES, value='Default NSF-HiFiGAN', visible=False,
                                   info='The default NSF-HiFiGAN downloads automatically on start. Choose mel previews only to skip audio rendering.')
        vocoder = gr.Textbox(label='Custom OpenVPI NSF-HiFiGAN checkpoint path', visible=False,
                            info='Use a compatible .ckpt or converted .pth export: 44.1 kHz, 128 mel bins, hop 512. Keep config.json beside raw checkpoints when provided.')
        use_pretrained = gr.Checkbox(label='Pretrained', value=False)
        with gr.Row():
            batch = gr.Number(label='Max clips per batch (per GPU)', value=64, minimum=1, precision=0,
                              info='Clips are trained whole, never cropped. A batch is closed at this many clips or at the frame limit.')
            max_frames = gr.Number(label='Max frames per batch (per GPU)', value=50000, minimum=1, precision=0,
                                   info='Padded frames (clips x longest clip, 1 frame = 11.6 ms). Lower it if you run out of GPU memory.')
            epochs = gr.Number(label='Total epochs', value=100, minimum=1, precision=0)
            save_every = gr.Number(label='Save every N epochs', value=10, minimum=1, precision=0)
        compile_backbone = gr.Checkbox(label='Compile flow backbone', value=False, visible=False,
                                       info='Requires Linux, CUDA and Triton 3.6.0. NVIDIA GPUs need compute capability 8.0 or newer. The first step takes longer to compile. Unsupported setups train uncompiled.')
        with gr.Row():
            train_button = gr.Button('Start / resume rectified training', variant='primary')
            stop_button = gr.Button('Stop current rectified job', interactive=False)
    state = gr.Textbox(label='Rectified job status', interactive=False)
    log = gr.Textbox(label='Rectified job log', lines=12, max_lines=20, interactive=False)
    outputs = [state, log, preprocess_button, extract_button, train_button, stop_button]
    preprocess_button.click(preprocess, [name, dataset, workers, slicing], outputs, queue=False)
    extract_button.click(extract, [name, method, workers, device, embedder], outputs, queue=False)
    use_pretrained.change(toggle_pretrained, [use_pretrained], [batch], queue=False)
    vocoder_mode.change(lambda mode: gr.update(visible=mode == 'Custom NSF-HiFiGAN'),
                        [vocoder_mode], [vocoder], queue=False)
    train_button.click(start, [name, vocoder, batch, max_frames, epochs, save_every, device, compile_backbone,
                               use_pretrained, vocoder_mode], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    gr.Timer(2).tick(status, [], outputs, queue=False)
