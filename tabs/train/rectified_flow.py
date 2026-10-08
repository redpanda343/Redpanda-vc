import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil

from tabs.settings.sections.precision import get_precision
from rectified_flow.distributed import parse_devices
from rectified_flow.resources import DEFAULT_VOCODER, VOCODERS, vocoder_path

ROOT = Path(__file__).resolve().parents[2]
QUIET_WARNINGS = ('ignore:pkg_resources is deprecated,ignore:Checkpoint directory,'
                  'ignore:`isinstance(treespec,ignore::FutureWarning')
_lock = threading.Lock()
_process = None
_runner = None
_cancelled = False


def experiment_path(name):
    name = str(name).strip()
    if not name or Path(name).name != name or name in {'.', '..'} or any(c in name for c in '/\\:'):
        raise gr.Error('Enter a model name without folders or path separators.')
    return ROOT / 'logs' / name


def positive_integer(value, label):
    if value is None or float(value) != int(value) or int(value) < 1:
        raise gr.Error(f'{label} must be a positive whole number.')
    return str(int(value))


def _active():
    return _runner is not None and _runner.is_alive()


def _status():
    active = _active()
    return (gr.update(interactive=not active), gr.update(interactive=not active), gr.update(interactive=active),
            gr.Timer(active=active))


def status():
    with _lock:
        return _status()


def run_steps(label, steps, finished):
    global _process
    warnings = ','.join(filter(None, (os.environ.get('PYTHONWARNINGS'), QUIET_WARNINGS)))
    environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONWARNINGS=warnings)
    for module, arguments in steps:
        with _lock:
            if _cancelled:
                return
            _process = subprocess.Popen([sys.executable, '-u', '-m', module, *arguments], cwd=ROOT, env=environment)
        code = _process.wait()
        if _cancelled:
            return
        if code != 0:
            print(f'{label} stopped with exit code {code}.', flush=True)
            return
    print(f'{label} {finished}', flush=True)


def launch(name, steps, description, finished='done.'):
    global _runner, _cancelled
    directory = experiment_path(name)
    with _lock:
        if _active():
            raise gr.Error('A rectified-flow job is already running. Wait for it to finish or stop it first.')
        directory.mkdir(parents=True, exist_ok=True)
        _cancelled = False
        label = f'{description} {name}:'
        print(f'{label} started.', flush=True)
        _runner = threading.Thread(target=run_steps, args=(label, steps, finished), daemon=True)
        _runner.start()
        gr.Info(f'{description} started. Progress is shown in the console.')
        return _status()


def preprocess(name, dataset, workers, device, cutting='Automatic', chunk_len=3.0, overlap_len=0.3,
               truncate_silence=False, silence_action='truncate', silence_threshold=-45.0, silence_minimum=0.3,
               silence_to=0.3, silence_compress=50.0):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    if cutting not in ('Skip', 'Simple', 'Automatic'):
        raise gr.Error('Choose Skip, Simple or Automatic audio cutting.')
    workers = positive_integer(workers, 'CPU workers')
    gpu = device_id(device)
    steps = [
        ('shared.preprocess.preprocess',
         [str(directory), dataset, '44100', workers, cutting, 'False', 'False', '0.0', str(float(chunk_len)),
          str(float(overlap_len)), 'none', 'WAV', str(bool(truncate_silence)), str(float(silence_threshold)),
          str(float(silence_to)), str(float(silence_minimum)), silence_action, str(float(silence_compress))]),
        ('shared.extract.extract',
         [str(directory), 'rmvpe', workers, gpu, '44100', 'contentvec', '0', 'v2', '--rectified']),
    ]
    return launch(name, steps, 'Preprocessing', 'done. Ready to train.')


def cutting_visibility(cutting, truncate_silence, silence_action):
    simple = cutting == 'Simple'
    silence = simple and truncate_silence
    return (gr.update(visible=simple), gr.update(visible=simple), gr.update(visible=simple),
            gr.update(visible=silence), gr.update(visible=silence), gr.update(visible=silence),
            gr.update(visible=silence and silence_action == 'truncate'),
            gr.update(visible=silence and silence_action == 'compress'))


def device_id(device):
    try:
        devices = parse_devices(device)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    return '-' if devices == ['cpu'] else '-'.join(item[5:] for item in devices)


def resolve_vocoder(name=DEFAULT_VOCODER):
    try:
        return vocoder_path(name)
    except Exception as error:
        raise gr.Error(f'Could not download the {name} vocoder: {error}') from error


def resolve_pretrained(directory, enabled, preferred=''):
    from rectified_flow.lightning_train import latest_checkpoint

    if not enabled or latest_checkpoint(directory / 'flow') or (directory / 'flow' / 'checkpoint.pth').is_file():
        return ''
    path = Path(str(preferred).strip().strip('"')) if str(preferred).strip() else ROOT / 'models' / 'pretraineds' / 'rectified' / 'pretrained.pth'
    if not path.is_file():
        raise gr.Error(f'Pretrained model not found. Place your Rectified Flow checkpoint at {path}.')
    return str(path)


def start(name, batch, max_frames, max_updates, checkpoint_interval, device, use_fused_kernels=False,
          use_pretrained=False, pretrained_path='', realtime=False, vocoder=DEFAULT_VOCODER):
    directory = experiment_path(name)
    device_id(device)
    if not (directory / 'filelist.txt').is_file():
        raise gr.Error('Extract features for this experiment first.')
    variance_embeds = None if (directory / 'rectified_config.json').is_file() else True
    pretrained = resolve_pretrained(directory, use_pretrained, pretrained_path)
    preset = 'realtime' if realtime else 'standard'
    from rectified_flow.train_flow import experiment_pitch_extractor, load_training_config

    pitch_extractor = experiment_pitch_extractor(directory)

    try:
        selected = load_training_config(directory, pretrained or None, use_fused_kernels, preset, pitch_extractor,
                                        variance_embeds)
    except ValueError as error:
        raise gr.Error(str(error)) from error
    precision = get_precision() or selected['flow'].get('precision', 'fp32')
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    if vocoder not in VOCODERS:
        raise gr.Error(f'Choose one of these vocoders: {", ".join(VOCODERS)}.')
    if selected['flow'].get('val_with_vocoder', True):
        resolve_vocoder(vocoder)
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
    if variance_embeds:
        arguments.append('--variance-embeds')
    if pretrained:
        arguments.extend(['--pretrained-flow', pretrained])
    return launch(name, [('rectified_flow.train_flow', arguments)], 'Training rectified flow')


def stop():
    global _cancelled
    with _lock:
        _cancelled = True
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
            print('Stopped. Resume starts from the last saved checkpoint; unsaved steps are lost.', flush=True)
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
    with gr.Accordion('2. Train rectified flow', open=True):
        use_pretrained = gr.Checkbox(label='Pretrained', value=False)
        pretrained_path = gr.Textbox(label='Voice checkpoint to fine-tune', value='',
                                    info='Optional exported flow .pth path, used when Pretrained is checked.')
        vocoder = gr.Dropdown(label='Vocoder', choices=list(VOCODERS), value=DEFAULT_VOCODER)
        gr.Markdown('Training precision follows **Settings → Training → Precision**.')
        with gr.Row():
            batch = gr.Number(label='Max clips per batch (per GPU)', value=None, minimum=1, precision=0,
                              info='Blank uses config, default 64.')
            max_frames = gr.Number(label='Max frames per batch (per GPU)', value=None, minimum=1, precision=0,
                                   info='Blank uses config, default 50000 padded frames. Lower to reduce GPU memory use.')
            max_updates = gr.Number(label='Max training updates', value=None, minimum=1, precision=0,
                               info='Blank uses config, default 100000 Lightning training steps.')
            checkpoint_interval = gr.Number(label='Checkpoint interval (updates)', value=None, minimum=1, precision=0,
                                   info='Blank uses config, default 4000 updates.')
        realtime = gr.Checkbox(label='Smaller model', value=False,
                               info='Lower vram usage and faster inference speeds, may decrease the quality of the model')
        use_fused_kernels = gr.Checkbox(label='Fused Linear + SoftSignGLU kernels', value=False,
                                       info='Set it when starting a new experiment: the experiment then trains with SoftSignGLU so the '
                                            'kernels can run. Experiments and pretrained checkpoints that use ATanGLU cannot enable it. '
                                            'Requires Triton and CUDA FP16 or BF16 for acceleration.')
        with gr.Row():
            train_button = gr.Button('Start / resume rectified training', variant='primary')
            stop_button = gr.Button('Stop current rectified job', interactive=False)
    timer = gr.Timer(2)
    outputs = [preprocess_button, train_button, stop_button, timer]
    cutting_inputs = [cutting, truncate_silence, silence_action]
    cutting_outputs = [chunk_len, overlap_len, truncate_silence, silence_action, silence_threshold, silence_minimum,
                       silence_to, silence_compress]
    for control in cutting_inputs:
        control.change(cutting_visibility, cutting_inputs, cutting_outputs, queue=False)
    preprocess_button.click(preprocess, [name, dataset, workers, device, cutting, chunk_len, overlap_len,
                                         truncate_silence, silence_action, silence_threshold, silence_minimum,
                                         silence_to, silence_compress], outputs, queue=False)
    train_button.click(start, [name, batch, max_frames, max_updates, checkpoint_interval, device, use_fused_kernels,
                               use_pretrained, pretrained_path, realtime, vocoder], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
