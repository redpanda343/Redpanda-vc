import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil

from tabs.settings.sections.precision import get_precision
from rvc.rectified.distributed import parse_devices
from rvc.rectified.resources import default_vocoder

ROOT = Path(__file__).resolve().parents[2]
_lock = threading.Lock()
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
    if value is None or float(value) != int(value) or int(value) < 1:
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


def preprocess(name, dataset, workers, cutting='Automatic', chunk_len=3.0, overlap_len=0.3, truncate_silence=False,
               silence_action='truncate', silence_threshold=-45.0, silence_minimum=0.3, silence_to=0.3,
               silence_compress=50.0):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    if cutting not in ('Skip', 'Simple', 'Automatic'):
        raise gr.Error('Choose Skip, Simple or Automatic audio cutting.')
    return launch(name, 'rvc.train.preprocess.preprocess',
                  [str(directory), dataset, '44100', positive_integer(workers, 'CPU workers'), cutting, 'False', 'False',
                   '0.0', str(float(chunk_len)), str(float(overlap_len)), 'none', 'WAV', str(bool(truncate_silence)),
                   str(float(silence_threshold)), str(float(silence_to)), str(float(silence_minimum)), silence_action,
                   str(float(silence_compress))], 'Preprocessing')


def cutting_visibility(cutting, truncate_silence, silence_action):
    simple = cutting == 'Simple'
    silence = simple and truncate_silence
    return (gr.update(visible=simple), gr.update(visible=simple), gr.update(visible=simple),
            gr.update(visible=silence), gr.update(visible=silence), gr.update(visible=silence),
            gr.update(visible=silence and silence_action == 'truncate'),
            gr.update(visible=silence and silence_action == 'compress'))


def extract(name, workers, device, embedder, pitch_extractor='parselmouth'):
    directory = experiment_path(name)
    if not (directory / 'sliced_audios').is_dir():
        raise gr.Error('Preprocess this experiment first.')
    gpu = device_id(device)
    method = 'pm' if pitch_extractor == 'parselmouth' else pitch_extractor
    return launch(name, 'rvc.train.extract.extract',
                  [str(directory), method, positive_integer(workers, 'CPU workers'), gpu,
                   '44100', embedder, '0', 'v2', '--rectified'], 'Extracting content and F0')


def device_id(device):
    try:
        devices = parse_devices(device)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return '-' if devices == ['cpu'] else '-'.join(item[5:] for item in devices)


def resolve_vocoder():
    try:
        return default_vocoder()
    except Exception as error:
        raise gr.Error(f'Could not download NSF-HiFiGAN vocoder: {error}') from error


def resolve_pretrained(directory, enabled, preferred=''):
    from rvc.rectified.lightning_train import latest_checkpoint

    if not enabled or latest_checkpoint(directory / 'flow') or (directory / 'flow' / 'checkpoint.pth').is_file():
        return ''
    path = Path(str(preferred).strip().strip('"')) if str(preferred).strip() else ROOT / 'rvc' / 'models' / 'pretraineds' / 'rectified' / 'pretrained.pth'
    if not path.is_file():
        raise gr.Error(f'Pretrained model not found. Place your Rectified Flow checkpoint at {path}.')
    return str(path)


def start(name, batch, max_frames, max_updates, checkpoint_interval, device, use_fused_kernels=False,
          use_pretrained=False, pretrained_path='', realtime=False, shortcut=False, variance_embeds=True):
    directory = experiment_path(name)
    device_id(device)
    if not (directory / 'filelist.txt').is_file():
        raise gr.Error('Extract features for this experiment first.')
    pretrained = resolve_pretrained(directory, use_pretrained, pretrained_path)
    preset = 'realtime' if realtime else 'standard'
    from rvc.rectified.train_flow import experiment_pitch_extractor, load_training_config

    pitch_extractor = experiment_pitch_extractor(directory)

    try:
        selected = load_training_config(directory, pretrained or None, use_fused_kernels, preset, pitch_extractor,
                                        bool(shortcut), bool(variance_embeds))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    precision = get_precision() or selected['flow'].get('precision', 'fp32')
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    vocoder = resolve_vocoder() if selected['flow'].get('val_with_vocoder', True) else ''
    arguments = ['--model-name', str(name).strip(), '--vocoder', vocoder,
                 '--precision', precision, '--preset', preset, '--pitch-extractor', pitch_extractor]
    for flag, value, label in (('--batch-size', batch, 'Max clips per batch'),
                               ('--max-batch-frames', max_frames, 'Max frames per batch'),
                               ('--max-updates', max_updates, 'Max training updates'),
                               ('--checkpoint-interval', checkpoint_interval, 'Checkpoint interval')):
        if value is not None:
            arguments.extend([flag, positive_integer(value, label)])
    if str(device).strip().lower() != 'auto':
        arguments.extend(['--device', str(device).strip().lower()])
    if use_fused_kernels:
        arguments.append('--use-fused-kernels')
    if shortcut:
        arguments.append('--shortcut')
    arguments.append('--variance-embeds' if variance_embeds else '--no-variance-embeds')
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
            _description = 'Stopped. Resume starts from the last saved checkpoint; unsaved steps are lost.'
        return _status()


def rectified_train_tab():
    gr.Markdown('### Rectified Flow')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-flow')
        device = gr.Textbox(label='Device', value='auto',
                            info='Auto uses the training config. Extraction selects detected GPUs. Examples: cuda:0, cuda:0,cuda:1, cpu.')
    with gr.Accordion('1. Prepare dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder')
        workers = gr.Number(label='CPU workers', value=4, minimum=1, precision=0)
        cutting = gr.Radio(label='Audio cutting', choices=['Skip', 'Simple', 'Automatic'], value='Automatic',
                           info="'Skip' keeps full clips, 'Simple' cuts fixed-length slices, 'Automatic' slices at "
                                'silences. Audio is always resampled to 44.1 kHz.')
        with gr.Row():
            chunk_len = gr.Slider(0.5, 10.0, 3.0, step=0.1, label='Chunk length (sec)',
                                  info="Length of the audio slice for 'Simple' method.", visible=False)
            overlap_len = gr.Slider(0.0, 0.4, 0.3, step=0.1, label='Overlap length (sec)',
                                    info="Length of the overlap between slices for 'Simple' method.", visible=False)
        truncate_silence = gr.Checkbox(label='Truncate silence', value=False, visible=False,
                                       info='For Simple slicing only. Shortens qualifying silent regions using the '
                                            'settings below.')
        silence_action = gr.Radio(label='Silence action', value='truncate', visible=False,
                                  choices=[('Truncate Detected Silence', 'truncate'),
                                           ('Compress Excess Silence', 'compress')])
        silence_threshold = gr.Slider(-80, -20, -45, step=1, label='Silence threshold (dB)', visible=False,
                                      info='For Simple slicing only. Audio below this level is treated as silence '
                                           'when truncation is enabled.')
        silence_minimum = gr.Slider(0.001, 5.0, 0.3, step=0.001, label='Minimum silence (sec)', visible=False,
                                    info='For Simple slicing only. A silent region must be at least this long '
                                         'before it can be truncated.')
        silence_to = gr.Slider(0.0, 0.5, 0.3, step=0.001, label='Truncate to (sec)', visible=False,
                               info='For Simple slicing only. Sets how much of each qualifying silent region remains '
                                    'after truncation.')
        silence_compress = gr.Slider(0.0, 99.9, 50.0, step=0.1, label='Compress excess silence to (%)', visible=False,
                                     info='Keeps the minimum silence plus this percentage of the silence beyond '
                                          'that minimum.')
        preprocess_button = gr.Button('Preprocess dataset')
        with gr.Row():
            embedder = gr.Dropdown(label='Content embedder', choices=['contentvec', 'spin-v2'], value='contentvec')
            pitch_extractor = gr.Dropdown(label='Pitch extractor', choices=['parselmouth', 'rmvpe'], value='parselmouth',
                                          info='As in DiffSinger. Parselmouth: 65-1100 Hz autocorrelation. RMVPE: neural '
                                               'tracker (230917 model), more robust on breathy or fry voices. Training '
                                               'extracts F0 with it, interpolates unvoiced frames and skips clips without '
                                               'voiced frames. Fixed once training starts.')
        extract_button = gr.Button('Extract content and F0')
    with gr.Accordion('2. Train rectified flow', open=True):
        use_pretrained = gr.Checkbox(label='Pretrained', value=False)
        pretrained_path = gr.Textbox(label='Voice checkpoint to fine-tune', value='',
                                    info='Optional exported flow .pth path, used when Pretrained is checked.')
        gr.Markdown('Training precision follows **Settings → Training → Precision**.')
        with gr.Row():
            batch = gr.Number(label='Max clips per batch (per GPU)', value=None, minimum=1, precision=0,
                              info='Blank uses config, default 64. Whole clips are trained without cropping.')
            max_frames = gr.Number(label='Max frames per batch (per GPU)', value=None, minimum=1, precision=0,
                                   info='Blank uses config, default 50000 padded frames. Lower to reduce GPU memory use.')
            max_updates = gr.Number(label='Max training updates', value=None, minimum=1, precision=0,
                               info='Blank uses config, default 100000 Lightning training steps.')
            checkpoint_interval = gr.Number(label='Checkpoint interval (updates)', value=None, minimum=1, precision=0,
                                   info='Blank uses config, default 4000 updates.')
        realtime = gr.Checkbox(label='Realtime', value=False,
                               info='Smaller, deeper model for new experiments: 256 hidden / 6 encoder layers, '
                                    '512-channel backbone with 12 layers, 384-channel aux decoder with 8 layers. '
                                    'Keep it set the same when resuming.')
        variance_embeds = gr.Checkbox(label='Breathiness / voicing conditioning', value=True,
                                      info="Optional. Conditions the flow on DiffSinger's breathiness and voicing curves, "
                                           'extracted from the source with the VR harmonic-noise separator, for more '
                                           'natural breaths and noise. Adds the separator to binarization, conversion '
                                           'and realtime (about 60 ms per realtime block on a GTX 1660 Ti). Off matches '
                                           "DiffSinger's default. Set it when the experiment starts and keep it when resuming.")
        shortcut = gr.Checkbox(label='Shortcut (few-step) flow', value=False,
                               info='Optional. Trains the flow to also take larger steps (Frans et al., 2024), so it can '
                                    'sample in 1, 2, 4, 8... steps. Costs about 15% more per update. Off keeps the '
                                    'DiffSinger flow unchanged. Set it when the experiment starts and keep it when resuming.')
        use_fused_kernels = gr.Checkbox(label='Fused Linear + SoftSignGLU kernels', value=False,
                                       info='Overrides the configured activation with SoftSignGLU, including on resume. Requires Triton and CUDA FP16 or BF16 for acceleration.')
        with gr.Row():
            train_button = gr.Button('Start / resume rectified training', variant='primary')
            stop_button = gr.Button('Stop current rectified job', interactive=False)
    state = gr.Textbox(label='Rectified job status', interactive=False)
    log = gr.Textbox(label='Rectified job log', lines=12, max_lines=20, interactive=False)
    outputs = [state, log, preprocess_button, extract_button, train_button, stop_button]
    cutting_inputs = [cutting, truncate_silence, silence_action]
    cutting_outputs = [chunk_len, overlap_len, truncate_silence, silence_action, silence_threshold, silence_minimum,
                       silence_to, silence_compress]
    for control in cutting_inputs:
        control.change(cutting_visibility, cutting_inputs, cutting_outputs, queue=False)
    preprocess_button.click(preprocess, [name, dataset, workers, cutting, chunk_len, overlap_len, truncate_silence,
                                         silence_action, silence_threshold, silence_minimum, silence_to,
                                         silence_compress], outputs, queue=False)
    extract_button.click(extract, [name, workers, device, embedder, pitch_extractor], outputs, queue=False)
    train_button.click(start, [name, batch, max_frames, max_updates, checkpoint_interval, device, use_fused_kernels,
                               use_pretrained, pretrained_path, realtime, shortcut, variance_embeds], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    gr.Timer(2).tick(status, [], outputs, queue=False)
