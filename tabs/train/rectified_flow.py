from pathlib import Path

import gradio as gr

from tabs.settings.sections.precision import get_precision
from tabs.train.jobs import ROOT, JobRunner, device_id, experiment_path, positive_integer
from tabs.train.slicing import preprocess_step, slicing_controls
from rectified_flow.resources import DEFAULT_VOCODER, VOCODERS, vocoder_path

IDLE_CONTROLS = 2
runner = JobRunner('rectified-flow')


def status():
    return runner.controls(IDLE_CONTROLS)


def launch(name, steps, description, finished='done.'):
    return runner.launch(name, steps, description, IDLE_CONTROLS, finished)


def preprocess(name, dataset, workers, device, cutting='Automatic', chunk_len=3.0, overlap_len=0.3,
               truncate_silence=False, silence_action='truncate', silence_threshold=-45.0, silence_minimum=0.3,
               silence_to=0.3, silence_compress=50.0):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    workers = positive_integer(workers, 'CPU workers')
    gpu = device_id(device)
    steps = [
        preprocess_step(directory, dataset, workers, cutting, chunk_len, overlap_len, truncate_silence,
                        silence_action, silence_threshold, silence_minimum, silence_to, silence_compress),
        ('shared.extract.extract',
         [str(directory), 'rmvpe', workers, gpu, '44100', 'contentvec', '0', 'v2', '--rectified']),
    ]
    return launch(name, steps, 'Preprocessing', 'done. Ready to train.')


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
        if value is not None and float(value) != 0:
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
    return runner.stop(IDLE_CONTROLS)


def rectified_train_tab():
    gr.Markdown('### Rectified Flow')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-flow')
        device = gr.Textbox(label='Device', value='auto',
                            info='Auto uses the training config. Extraction selects detected GPUs. Examples: cuda:0, cuda:0,cuda:1, cpu.')
    with gr.Accordion('1. Prepare dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder')
        workers = gr.Number(label='CPU workers', value=4, minimum=1, precision=0)
        slicing = slicing_controls()
        preprocess_button = gr.Button('Preprocess dataset')
    with gr.Accordion('2. Train rectified flow', open=True):
        use_pretrained = gr.Checkbox(label='Pretrained', value=False)
        pretrained_path = gr.Textbox(label='Voice checkpoint to fine-tune', value='',
                                    info='Optional exported flow .pth path, used when Pretrained is checked.')
        vocoder = gr.Dropdown(label='Vocoder', choices=list(VOCODERS), value=DEFAULT_VOCODER)
        gr.Markdown('Training precision follows **Settings → Training → Precision**.')
        with gr.Row():
            batch = gr.Number(label='Max clips per batch (per GPU)', value=0, minimum=0, precision=0,
                              info='0 uses config, default 64.')
            max_frames = gr.Number(label='Max frames per batch (per GPU)', value=0, minimum=0, precision=0,
                                   info='0 uses config, default 50000 padded frames. Lower to reduce GPU memory use.')
            max_updates = gr.Number(label='Max training updates', value=0, minimum=0, precision=0,
                               info='0 uses config, default 100000 Lightning training steps.')
            checkpoint_interval = gr.Number(label='Checkpoint interval (updates)', value=0, minimum=0, precision=0,
                                   info='0 uses config, default 4000 updates.')
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
    preprocess_button.click(preprocess, [name, dataset, workers, device, *slicing.inputs], outputs, queue=False)
    train_button.click(start, [name, batch, max_frames, max_updates, checkpoint_interval, device, use_fused_kernels,
                               use_pretrained, pretrained_path, realtime, vocoder], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
