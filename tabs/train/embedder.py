import math
from pathlib import Path

import gradio as gr

from beatrice.embedder_config import LOOKAHEAD_CHOICES, default_config
from tabs.settings.sections.precision import get_precision
from tabs.train.jobs import JobRunner, experiment_path, positive_integer

runner = JobRunner('embedder')
IDLE_CONTROLS = 3


def arguments(name, dataset, device, workers, clip_seconds, lookahead_ms, extraction_batch_size):
    experiment_path(name)
    dataset = str(dataset or '').strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    device = str(device).strip().lower()
    if device not in ('auto', 'cpu') and not (device.startswith('cuda:') and device[5:].isdigit()):
        raise gr.Error('Device must be auto, cpu or cuda:N.')
    if workers is None or not float(workers).is_integer() or workers < 0:
        raise gr.Error('CPU workers must be a nonnegative whole number.')
    if clip_seconds is None or not math.isfinite(float(clip_seconds)) or not 1 <= float(clip_seconds) <= 4:
        raise gr.Error('Clip length must be between one and four seconds.')
    if lookahead_ms not in LOOKAHEAD_CHOICES:
        raise gr.Error('Choose 0, 10, 20 or 25 milliseconds of extra lookahead.')
    return ['--model-name', str(name).strip(), '--dataset', dataset, '--device', device,
            '--workers', str(int(workers)), '--clip-seconds', str(float(clip_seconds)),
            '--lookahead-ms', str(int(lookahead_ms)),
            '--extraction-batch-size', positive_integer(extraction_batch_size, 'Extraction batch size')]


def prepare(name, dataset, device, workers, clip_seconds, lookahead_ms, extraction_batch_size):
    values = arguments(name, dataset, device, workers, clip_seconds, lookahead_ms, extraction_batch_size)
    return runner.launch(name, [('beatrice.embedder_train', [*values, '--prepare-only'])],
                         'Preparing embedder features', IDLE_CONTROLS)


def start(name, dataset, device, workers, clip_seconds, lookahead_ms, extraction_batch_size, steps, batch,
          learning_rate, save_interval):
    values = arguments(name, dataset, device, workers, clip_seconds, lookahead_ms, extraction_batch_size)
    if learning_rate is None or not math.isfinite(float(learning_rate)) or learning_rate <= 0:
        raise gr.Error('Learning rate must be positive and finite.')
    precision = get_precision() or default_config()['precision']
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise gr.Error(f'Unsupported training precision: {precision}')
    values.extend(['--precision', precision, '--learning-rate', str(float(learning_rate))])
    for flag, value, label in (('--steps', steps, 'Training steps'), ('--batch-size', batch, 'Batch size'),
                               ('--save-interval', save_interval, 'Save interval')):
        values.extend([flag, positive_integer(value, label)])
    return runner.launch(name, [('beatrice.embedder_train', values)], 'Training embedder', IDLE_CONTROLS)


def export(name):
    directory = experiment_path(name) / 'embedder'
    if not (directory / 'checkpoint_latest.pt').is_file():
        raise gr.Error('Train this embedder first. There is no saved checkpoint to export.')
    return runner.launch(name, [('beatrice.embedder_train', ['--model-name', str(name).strip(), '--export-only'])],
                         'Exporting embedder', IDLE_CONTROLS)


def status():
    return runner.controls(IDLE_CONTROLS)


def stop():
    return runner.stop(IDLE_CONTROLS)


def embedder_train_tab():
    defaults = default_config()
    gr.Markdown('### Embedder')
    gr.Markdown('Experimental training for clearer English pronunciation. A ContentVec teacher trains the small '
                'Beatrice embedder; only the small model is used during voice conversion. Train on varied English '
                'speakers and accents, then select the export under **Beatrice → Embedder** and train your voice model.')
    with gr.Row():
        name = gr.Textbox(label='Embedder name', value='my-english-embedder')
        device = gr.Textbox(label='Device', value='auto', info='auto, cpu or cuda:N. Training uses one GPU.')
    with gr.Accordion('1. Prepare teacher features', open=True):
        dataset = gr.Textbox(label='English dataset folder',
                             info='Reads audio from this folder and its subfolders. A validation subfolder is held '
                                  'out; otherwise speakers or whole files are reserved for validation. Use at least '
                                  'two files, each at least one second long.')
        with gr.Row():
            workers = gr.Number(label='CPU workers', value=defaults['workers'], minimum=0, precision=0,
                                info='Workers for feature caching and loading training clips.')
            extraction_batch_size = gr.Number(label='Extraction batch size', value=defaults['extraction_batch_size'],
                                              minimum=1, precision=0,
                                              info='Clips per extraction batch. Lower this if GPU memory runs out.')
            clip_seconds = gr.Slider(label='Clip length (seconds)', minimum=1, maximum=4, step=0.5,
                                     value=defaults['clip_seconds'])
            lookahead_ms = gr.Radio(label='Extra lookahead (ms)', choices=list(LOOKAHEAD_CHOICES),
                                   value=defaults['lookahead_ms'],
                                   info='Original phone lookahead is about 2.5 ms. Adding 25 ms gives about 27.5 ms '
                                        'and requires RedPanda realtime inference.')
        gr.Markdown('Teacher features are extracted in batches, then cached and reused. Keep the dataset, '
                    'clip length and lookahead '
                    'the same when resuming an experiment.')
        prepare_button = gr.Button('Prepare teacher features')
    with gr.Accordion('2. Train embedder', open=True):
        with gr.Row():
            steps = gr.Number(label='Training steps', value=defaults['steps'], minimum=1, precision=0,
                              info='Raise this to continue a finished experiment.')
            batch = gr.Number(label='Batch size', value=defaults['batch_size'], minimum=1, precision=0)
            learning_rate = gr.Number(label='Learning rate', value=defaults['learning_rate'], minimum=0,
                                      info='Peak learning rate after warmup. It then gradually decreases.')
            save_interval = gr.Number(label='Save interval (steps)', value=defaults['save_interval'], minimum=1,
                                      precision=0)
        gr.Markdown('New experiments start from Beatrice’s pretrained embedder and use **Settings → Training → '
                    f'Precision**. The first {defaults["warmup_steps"]} steps warm up the learning rate, then it '
                    f'decays to {defaults["minimum_learning_rate_ratio"]:.0%} of its peak. Resume keeps the saved '
                    'precision and training recipe. Validation feature losses '
                    'help compare experiments; pronunciation improvements '
                    'still need listening checks on unseen English speech.')
        with gr.Row():
            train_button = gr.Button('Start / resume embedder training', variant='primary')
            export_button = gr.Button('Export latest embedder')
            stop_button = gr.Button('Stop current embedder job', interactive=False)
    timer = gr.Timer(2)
    outputs = [prepare_button, train_button, export_button, stop_button, timer]
    common = [name, dataset, device, workers, clip_seconds, lookahead_ms, extraction_batch_size]
    prepare_button.click(prepare, common, outputs, queue=False)
    train_button.click(start, [*common, steps, batch, learning_rate, save_interval], outputs, queue=False)
    export_button.click(export, [name], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
