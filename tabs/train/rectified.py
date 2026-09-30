import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil
import torch
from tqdm import tqdm

from tabs.settings.sections.precision import get_precision
from rvc.rectified.distributed import parse_devices
from rvc.lib.tools.prerequisites_download import _sha256, download_file

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
_pretrain_lock = threading.Lock()
FLOW_PRETRAIN_URL = ('https://huggingface.co/shiromiya/ShiroRVC-Resources/resolve/'
                     'f84d2dfb2f6cc0a3205e437c7676455971dc6bf5/Rectified_pretrains/pretrain_flow_contentvec.pth')
FLOW_PRETRAIN_SHA256 = '79fdb4e13ff3d68d755f23879985e86f0052cda10360c7d6bd8003aa08074568'
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
        return _status()


def preprocess(name, dataset, workers):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    return launch(name, 'rvc.train.preprocess.preprocess',
                  [str(directory), dataset, '44100', positive_integer(workers, 'CPU workers'),
                   'Automatic', 'False', 'False', '0.0', '10.0', '0.3', 'none', 'WAV'], 'Preprocessing')


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


def toggle_pretrained(enabled, custom):
    return gr.update(visible=enabled), gr.update(visible=enabled and custom)


def resolve_pretrained(directory, enabled, custom, path):
    if not enabled or (directory / 'flow' / 'checkpoint.pth').is_file():
        return ''
    if custom:
        path = str(path or '').strip().strip('"')
        if not path or not Path(path).is_file():
            raise gr.Error('Choose an existing custom rectified-flow pretrained checkpoint.')
        return path
    info_path = directory / 'model_info.json'
    info = json.loads(info_path.read_text(encoding='utf-8')) if info_path.is_file() else {}
    if info.get('embedder_model', 'contentvec') != 'contentvec':
        raise gr.Error('The default Shiro pretrained uses ContentVec. Extract with ContentVec or select a compatible custom pretrained.')
    destination = ROOT / 'rvc' / 'models' / 'pretraineds' / 'rectified' / 'pretrain_flow_contentvec.pth'
    with _pretrain_lock:
        if not destination.is_file() or _sha256(destination) != FLOW_PRETRAIN_SHA256:
            gr.Info('Downloading the Shiro ContentVec flow pretrained. It will be reused for future runs.')
            try:
                with tqdm(total=319370011, unit='B', unit_scale=True, desc='Shiro flow pretrained') as progress:
                    download_file(FLOW_PRETRAIN_URL, str(destination), progress, expected_sha256=FLOW_PRETRAIN_SHA256)
            except Exception as error:
                raise gr.Error(f'Could not download the Shiro flow pretrained: {error}') from error
    return str(destination)


def start(name, vocoder, pretrained, batch, epochs, save_every, device, compile_backbone=False,
          use_pretrained=True, custom_pretrained=False):
    directory = experiment_path(name)
    device_id(device)
    if not (directory / 'filelist.txt').is_file():
        raise gr.Error('Extract features for this experiment first.')
    vocoder = str(vocoder).strip().strip('"')
    if vocoder and not Path(vocoder).is_file():
        raise gr.Error('Choose an existing OpenVPI NSF-HiFiGAN checkpoint.')
    precision = get_precision() or 'fp32'
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    arguments = ['--model-name', str(name).strip(), '--vocoder', vocoder,
                 '--batch-size', positive_integer(batch, 'Batch size'),
                 '--epochs', positive_integer(epochs, 'Total epochs'),
                 '--save-every', positive_integer(save_every, 'Save interval'),
                 '--device', str(device).strip().lower(), '--precision', precision]
    if compile_backbone:
        arguments.append('--compile')
    pretrained = resolve_pretrained(directory, use_pretrained, custom_pretrained, pretrained)
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
    gr.Markdown('### Rectified Flow\nTrain Shiro rectified flow at **44.1 kHz** using the saved Settings > Precision selection. '
                'An optional frozen OpenVPI NSF-HiFiGAN vocoder renders audio previews. Use a separate experiment from RVC training.')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-flow')
        device = gr.Textbox(label='Device', value=','.join(f'cuda:{index}' for index in range(torch.cuda.device_count())) or 'cpu',
                            info='Detected GPUs are selected automatically. One GPU: cuda:0. Multiple GPUs: cuda:0,cuda:1. CPU: cpu.')
    with gr.Accordion('1. Prepare dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder')
        workers = gr.Number(label='CPU workers', value=4, minimum=1, precision=0)
        gr.Markdown('Automatic slicing, 44.1 kHz WAV, up to 10 seconds per slice. No normalization or noise reduction.')
        preprocess_button = gr.Button('Preprocess dataset')
    with gr.Accordion('2. Extract features', open=True):
        with gr.Row():
            method = gr.Dropdown(label='Pitch extractor', choices=['rmvpe', 'swift', 'pm'], value='rmvpe')
            embedder = gr.Dropdown(label='Content embedder', choices=['contentvec', 'spin-v2'], value='contentvec')
        extract_button = gr.Button('Extract content and F0')
    with gr.Accordion('3. Train rectified flow', open=True):
        vocoder = gr.Textbox(label='OpenVPI NSF-HiFiGAN checkpoint path (optional)',
                            info='Leave empty for mel previews only. 44.1 kHz, 128 mel bins, hop 512. Keep its config.json beside it. RVC generator checkpoints are incompatible.')
        use_pretrained = gr.Checkbox(label='Pretrained', value=True,
                                     info='Automatically download and use the Shiro ContentVec pretrained. Disable to train a new experiment from scratch.')
        custom_pretrained = gr.Checkbox(label='Custom pretrained', value=False,
                                       info='Use your own compatible rectified-flow checkpoint instead of the default Shiro pretrained.')
        with gr.Column(visible=False) as custom_pretrained_settings:
            pretrained_upload = gr.File(label='Upload custom flow pretrained', file_types=['.pth'], type='filepath')
            pretrained = gr.Textbox(label='Custom pretrained flow path',
                                    info='Choose a flow checkpoint compatible with the experiment configuration and content embedder.')
        with gr.Row():
            batch = gr.Number(label='Batch size per GPU', value=4, minimum=1, precision=0)
            epochs = gr.Number(label='Total epochs', value=100, minimum=1, precision=0)
            save_every = gr.Number(label='Save every N epochs', value=10, minimum=1, precision=0)
        compile_backbone = gr.Checkbox(label='Compile flow backbone', value=False,
                                       info='Requires CUDA and Triton. Training falls back to eager mode when unavailable.')
        gr.Markdown('Precision follows **Settings > Precision** when you start or resume. Click Update precision there to save it. '
                    'Existing runs resume from the last saved epoch. Stopping discards unsaved steps. Checkpoints and TensorBoard previews are saved in logs/<model name>/flow.')
        with gr.Row():
            train_button = gr.Button('Start / resume rectified training', variant='primary')
            stop_button = gr.Button('Stop current rectified job', interactive=False)
    state = gr.Textbox(label='Rectified job status', interactive=False)
    log = gr.Textbox(label='Rectified job log', lines=12, max_lines=20, interactive=False)
    outputs = [state, log, preprocess_button, extract_button, train_button, stop_button]
    preprocess_button.click(preprocess, [name, dataset, workers], outputs, queue=False)
    extract_button.click(extract, [name, method, workers, device, embedder], outputs, queue=False)
    use_pretrained.change(toggle_pretrained, [use_pretrained, custom_pretrained],
                          [custom_pretrained, custom_pretrained_settings], queue=False)
    custom_pretrained.change(toggle_pretrained, [use_pretrained, custom_pretrained],
                             [custom_pretrained, custom_pretrained_settings], queue=False)
    pretrained_upload.upload(lambda path: path or '', [pretrained_upload], [pretrained], queue=False)
    train_button.click(start, [name, vocoder, pretrained, batch, epochs, save_every, device, compile_backbone,
                               use_pretrained, custom_pretrained], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    gr.Timer(2).tick(status, [], outputs, queue=False)
