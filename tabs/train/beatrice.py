import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil

from tabs.settings.sections.precision import get_precision
from tabs.train.jobs import QUIET_WARNINGS, experiment_path, positive_integer
from tabs.train.slicing import SILENCE_ACTIONS, check_cutting, slicing_controls

ROOT = Path(__file__).resolve().parents[2]
BEATRICE_WARNINGS = 'ignore:wav_length % 160 != 0,ignore:Some clusters have no assigned data points'
SLICERS = ('Skip', 'Simple')
MINIMUM_CHUNK = 5.0
_lock = threading.Lock()
_process = None
_experiment = None


def latest_export():
    from beatrice.train import model_folders

    if _experiment is None or not _experiment.is_dir():
        return None
    exports = model_folders(_experiment)
    return exports[-1] if exports else None


def _status():
    active = _process is not None and _process.poll() is None
    return gr.update(interactive=not active), gr.update(interactive=active), gr.Timer(active=active)


def status():
    with _lock:
        return _status()


def print_job_completion(process, label, experiment):
    from beatrice.train import remove_work

    code = process.wait()
    try:
        remove_work(experiment)
    except OSError as error:
        print(f'Could not remove the temporary training folder: {error}', flush=True)
    if code != 0:
        print(f'{label} stopped with exit code {code}.', flush=True)
        return
    export = latest_export()
    print(f'{label} done.' + (f' Model folder: {export}' if export is not None else ''), flush=True)


def launch(name, arguments):
    global _process, _experiment
    directory = experiment_path(name)
    with _lock:
        if _process is not None and _process.poll() is None:
            raise gr.Error('A Beatrice job is already running. Wait for it to finish or stop it first.')
        directory.mkdir(parents=True, exist_ok=True)
        warnings = ','.join(filter(None, (os.environ.get('PYTHONWARNINGS'), QUIET_WARNINGS, BEATRICE_WARNINGS)))
        environment = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONWARNINGS=warnings)
        _process = subprocess.Popen([sys.executable, '-u', '-m', 'beatrice.train', *arguments], cwd=ROOT,
                                    env=environment)
        _experiment = directory
        label = f'Training Beatrice {name}:'
        print(f'{label} started.', flush=True)
        threading.Thread(target=print_job_completion, args=(_process, label, directory), daemon=True).start()
        gr.Info('Beatrice training started. Progress is shown in the console.')
        return _status()


def dataset_arguments(name, dataset, workers, device, cutting='Skip', chunk_len=MINIMUM_CHUNK, overlap_len=0.3,
                      truncate_silence=False, silence_action='truncate', silence_threshold=-45.0, silence_minimum=0.3,
                      silence_to=0.3, silence_compress=50.0):
    experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    device = str(device).strip().lower()
    if device not in {'auto', 'cpu'} and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise gr.Error('Device must be auto, cpu or cuda:N.')
    check_cutting(cutting, chunk_len, SLICERS, MINIMUM_CHUNK)
    if silence_action not in SILENCE_ACTIONS:
        raise gr.Error('Choose a silence action.')
    arguments = ['--model-name', str(name).strip(), '--dataset', dataset, '--device', device,
                 '--workers', positive_integer(workers, 'CPU workers'), '--cutting', cutting,
                 '--chunk-len', str(float(chunk_len)), '--overlap-len', str(float(overlap_len)),
                 '--silence-action', silence_action, '--silence-threshold', str(float(silence_threshold)),
                 '--silence-minimum', str(float(silence_minimum)), '--silence-to', str(float(silence_to)),
                 '--silence-compress', str(float(silence_compress))]
    if truncate_silence:
        arguments.append('--truncate-silence')
    return arguments


def start(name, dataset, workers, device, steps, batch, save_interval, *slicing):
    arguments = dataset_arguments(name, dataset, workers, device, *slicing)
    precision = get_precision() or 'fp32'
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    arguments.extend(['--precision', precision])
    for flag, value, label in (('--steps', steps, 'Training steps'), ('--batch-size', batch, 'Batch size'),
                               ('--save-interval', save_interval, 'Save interval')):
        arguments.extend([flag, positive_integer(value, label)])
    return launch(name, arguments)


def stop():
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
            print('Stopped. Resume starts from the last saved checkpoint; unsaved steps are lost.', flush=True)
        return _status()


def beatrice_train_tab():
    gr.Markdown('### Beatrice')
    gr.Markdown('Fine-tunes a [Beatrice 2](https://prj-beatrice.com) low-latency voice conversion model with '
                '[Beatrice Trainer](https://huggingface.co/fierce-cats/beatrice-trainer) 2.0.0-rc.0 (MIT). '
                '`logs/<model>` keeps only the latest `paraphernalia_*` model folder, which loads in the realtime GUI, '
                'the Beatrice VST, VCClient or beatrice-client, and `checkpoint_latest.pt.gz` to resume training. The '
                'app downloads the trainer with its pretrained models, noise and impulse-response sets (about 440 MB) '
                'at startup, and training retries an interrupted download.')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-beatrice')
        device = gr.Textbox(label='Device', value='auto', info='auto, cpu or cuda:N. Beatrice trains on one GPU.')
    with gr.Accordion('1. Prepare dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder',
                             info='Audio files directly in this folder train one voice named after the model. A '
                                  'folder holding only subfolders trains one voice per subfolder. With Skip, WAV, FLAC, '
                                  'MP3, OGG, Opus and AIFF are read as they are.')
        workers = gr.Number(label='CPU workers', value=min(8, os.cpu_count() or 1), minimum=1, precision=0,
                            info='Processes that slice the dataset and data loader processes that decode and '
                                 'augment audio.')
        slicing = slicing_controls(choices=SLICERS, value='Skip', chunk_minimum=MINIMUM_CHUNK,
                                   chunk_value=MINIMUM_CHUNK,
                                   note="Skip trains on the files as they are. Simple cuts the WAV, FLAC, MP3 and OGG "
                                        "files into 5-10 second chunks at the Beatrice model's 24 kHz sample rate, in "
                                        'a temporary folder that is deleted when training ends.')
    with gr.Accordion('2. Train Beatrice', open=True):
        with gr.Row():
            steps = gr.Number(label='Training steps', value=10000, minimum=1, precision=0,
                              info='Upstream default 10000, about 40 minutes on an RTX 4090. Raise it to continue a '
                                   'finished model.')
            batch = gr.Number(label='Batch size', value=8, minimum=1, precision=0,
                              info='4-second clips per step. Lower it to reduce GPU memory use.')
            save_interval = gr.Number(label='Save interval (steps)', value=2000, minimum=1, precision=0,
                                      info='Replaces checkpoint_latest.pt.gz and the model folder with the '
                                           'current step.')
        gr.Markdown('Training precision follows **Settings → Training → Precision**: fp32 trains in full precision, '
                    "fp16 or bf16 turn on the trainer's FP16 mixed precision.")
        with gr.Row():
            train_button = gr.Button('Start / resume Beatrice training', variant='primary')
            stop_button = gr.Button('Stop current Beatrice job', interactive=False)
    timer = gr.Timer(2)
    outputs = [train_button, stop_button, timer]
    train_button.click(start, [name, dataset, workers, device, steps, batch, save_interval, *slicing.inputs], outputs,
                       queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
