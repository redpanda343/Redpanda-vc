import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr
import psutil

from tabs.settings.sections.precision import get_precision
from tabs.train.jobs import QUIET_WARNINGS, experiment_path, positive_integer
from tabs.train.slicing import check_cutting, slicing_controls

ROOT = Path(__file__).resolve().parents[2]
BEATRICE_WARNINGS = 'ignore:wav_length % 160 != 0,ignore:Some clusters have no assigned data points'
SLICERS = ('Skip', 'Silence', 'Simple')
MINIMUM_CHUNK = 5.0
TRAINING_VQ_MODES = ('none', 'self', 'random')
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


def dataset_arguments(name, dataset, workers, device, pause_threshold=-42.0, pause_minimum=1.0, cutting='Skip',
                      chunk_len=MINIMUM_CHUNK, overlap_len=0.3, truncate_silence=False,
                      silence_threshold=-45.0, silence_minimum=0.3, silence_to=0.3):
    experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    device = str(device).strip().lower()
    if device not in {'auto', 'cpu'} and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise gr.Error('Device must be auto, cpu or cuda:N.')
    check_cutting(cutting, chunk_len, SLICERS, MINIMUM_CHUNK)
    if not 0.1 <= float(pause_minimum) <= 10.0:
        raise gr.Error('Minimum pause must be between 0.1 and 10 seconds.')
    arguments = ['--model-name', str(name).strip(), '--dataset', dataset, '--device', device,
                 '--workers', positive_integer(workers, 'CPU workers'), '--cutting', cutting,
                 '--chunk-len', str(float(chunk_len)), '--overlap-len', str(float(overlap_len)),
                 '--silence-threshold', str(float(silence_threshold)),
                 '--silence-minimum', str(float(silence_minimum)), '--silence-to', str(float(silence_to)),
                 '--pause-threshold', str(float(pause_threshold)),
                 '--pause-minimum', str(float(pause_minimum))]
    if truncate_silence:
        arguments.append('--truncate-silence')
    return arguments


def start(name, dataset, workers, device, steps, warmup, batch, save_interval, training_vq, pause_threshold,
          pause_minimum, *slicing):
    arguments = dataset_arguments(name, dataset, workers, device, pause_threshold, pause_minimum, *slicing)
    precision = get_precision() or 'fp32'
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    if warmup is None or float(warmup) != int(warmup) or int(warmup) < 0:
        raise gr.Error('Warmup steps must be a whole number of 0 or more.')
    if training_vq not in TRAINING_VQ_MODES:
        raise gr.Error('Choose a training-time VQ mode.')
    arguments.extend(['--precision', precision, '--warmup-steps', str(int(warmup)), '--training-vq', training_vq])
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
                                   note="Skip trains on the files as they are. Silence splits the WAV, FLAC, MP3 and "
                                        'OGG files only where a pause lasts at least the minimum pause, with no length '
                                        'limit. Simple cuts them into 5-10 second chunks. Both slice at the Beatrice '
                                        "model's 24 kHz sample rate, in a temporary folder that is deleted when "
                                        'training ends.')
        with gr.Row(visible=False) as pause_row:
            pause_threshold = gr.Slider(-80, -20, -42, step=1, label='Pause threshold (dB)',
                                        info='Audio quieter than this counts as a pause.')
            pause_minimum = gr.Slider(0.1, 10.0, 1.0, step=0.1, label='Minimum pause (sec)',
                                      info='Splits only at pauses at least this long. Shorter pauses stay inside the '
                                           'slice, and up to 0.3 s of each pause is kept at the slice edges.')
        slicing.cutting.change(lambda cutting: gr.update(visible=cutting == 'Silence'), slicing.cutting, pause_row,
                               queue=False)
    with gr.Accordion('2. Train Beatrice', open=True):
        with gr.Row():
            steps = gr.Number(label='Training steps', value=10000, minimum=1, precision=0,
                              info='Upstream default 10000, about 40 minutes on an RTX 4090. Raise it to continue a '
                                   'finished model.')
            warmup = gr.Number(label='Warmup steps', value=5000, minimum=0, precision=0,
                               info='Steps over which the learning rate rises from 0 to full. Upstream default '
                                    '5000. 0 starts at full rate.')
            batch = gr.Number(label='Batch size', value=8, minimum=1, precision=0,
                              info='4-second clips per step. Lower it to reduce GPU memory use.')
            save_interval = gr.Number(label='Save interval (steps)', value=2000, minimum=1, precision=0,
                                      info='Replaces checkpoint_latest.pt.gz and the model folder with the '
                                           'current step.')
        training_vq = gr.Radio(label='Training-time VQ', value='none',
                               choices=[('Off (upstream default)', 'none'), ('Own voice codebook', 'self'),
                                        ('Random voice codebook', 'random')],
                               info="Replaces the phone features with the average of their 4 nearest codebook entries "
                                    'during training, as VQ Neighbor Count does at inference, so the generator learns '
                                    'from the same kind of features. Set VQ Neighbor Count to 4 when using a model '
                                    "trained this way. Random uses a random voice's codebook and matches Own voice "
                                    'when training one voice.')
        gr.Markdown('Training precision follows **Settings → Training → Precision**: fp32 trains in full precision, '
                    "fp16 or bf16 turn on the trainer's FP16 mixed precision.")
        with gr.Row():
            train_button = gr.Button('Start / resume Beatrice training', variant='primary')
            stop_button = gr.Button('Stop current Beatrice job', interactive=False)
    timer = gr.Timer(2)
    outputs = [train_button, stop_button, timer]
    train_button.click(start, [name, dataset, workers, device, steps, warmup, batch, save_interval, training_vq,
                               pause_threshold, pause_minimum, *slicing.inputs], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
