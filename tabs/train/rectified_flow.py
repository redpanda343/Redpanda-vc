import json
from pathlib import Path

import gradio as gr

from tabs.settings.sections.precision import get_precision
from tabs.train.jobs import ROOT, JobRunner, device_id, experiment_path, positive_integer
from tabs.train.slicing import preprocess_step, slicing_controls
from rectified_flow.config import default_config, mel_preset, preset_path
from rectified_flow.resources import (CUSTOM_VOCODER_ROOT, DEFAULT_VOCODER, VOCODERS, custom_vocoder_file,
                                      custom_vocoders, vocoder_path, vocoder_title)

IDLE_CONTROLS = 2
SLICERS = ('Skip', 'Simple')
MINIMUM_CHUNK = 5.0
PRETRAINED_ROOT = ROOT / 'models' / 'pretraineds' / 'rectified'
TRAINING_PRESETS = ('standard', 'meanflow')
runner = JobRunner('rectified-flow')


def status():
    return runner.controls(IDLE_CONTROLS)


def preset_value(key):
    try:
        values = [default_config(preset=preset)['flow'][key] for preset in TRAINING_PRESETS]
    except (KeyError, ValueError):
        return 'see the preset files'
    if all(value == values[0] for value in values):
        return str(values[0])
    return ', '.join(f'{preset} {value}' for preset, value in zip(TRAINING_PRESETS, values))


def preset_info(key, unit=''):
    return f'0 uses the experiment config, which new experiments copy from the preset ({preset_value(key)}{unit}).'


def launch(name, steps, description, finished='done.'):
    return runner.launch(name, steps, description, IDLE_CONTROLS, finished)


def registry_mel(vocoder):
    mel = VOCODERS.get(vocoder, {}).get('mel')
    return mel_preset(mel) if mel else None


def experiment_sample_rate(directory, vocoder):
    config_path = directory / 'rectified_config.json'
    if config_path.is_file():
        return int(json.loads(config_path.read_text(encoding='utf-8'))['data']['sample_rate'])
    return int((registry_mel(vocoder) or default_config()['data'])['sample_rate'])


def preprocess(name, dataset, workers, device, cutting='Simple', chunk_len=MINIMUM_CHUNK, overlap_len=0.3,
               truncate_silence=False, silence_threshold=-45.0, silence_minimum=0.3, silence_to=0.3,
               vocoder=DEFAULT_VOCODER):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    workers = positive_integer(workers, 'CPU workers')
    gpu = device_id(device)
    rate = experiment_sample_rate(directory, vocoder)
    steps = [
        preprocess_step(directory, dataset, workers, cutting, chunk_len, overlap_len, truncate_silence,
                        silence_threshold, silence_minimum, silence_to,
                        choices=SLICERS, chunk_minimum=MINIMUM_CHUNK, sample_rate=rate),
        ('shared.extract.extract',
         [str(directory), 'rmvpe', workers, gpu, str(rate), 'contentvec', '0', 'v2', '--rectified']),
    ]
    return launch(name, steps, f'Preprocessing at {rate / 1000:g} kHz', 'done. Ready to train.')


def resolve_vocoder(name=DEFAULT_VOCODER):
    try:
        return vocoder_path(name)
    except Exception as error:
        raise gr.Error(f'Could not download the {name} vocoder: {error}') from error


def vocoder_choices():
    return ([(spec['title'], name) for name, spec in VOCODERS.items()] +
            [(f'Custom: {vocoder_title(path)}', path) for path in custom_vocoders()])


def refresh_vocoders(current):
    choices = vocoder_choices()
    values = {value for _, value in choices}
    return gr.update(choices=choices, value=current if current in values else DEFAULT_VOCODER)


def check_registry_vocoder(vocoder, data):
    from rectified_flow.vocoder import mel_mismatch

    mel = {key: value for key, value in registry_mel(vocoder).items() if key not in ('mel_mean', 'mel_std')}
    mismatch = mel_mismatch(data, mel)
    if mismatch:
        raise gr.Error(f'{vocoder_title(vocoder)} renders another mel than this experiment: {mismatch}. Choose the '
                       'vocoder the experiment was started with, or start a new experiment.')


def check_custom_vocoder(vocoder, data):
    from rectified_flow.vocoder import load_vocoder

    try:
        load_vocoder(str(custom_vocoder_file(vocoder)), data)
    except Exception as error:
        raise gr.Error(f'The custom vocoder {vocoder_title(vocoder)} cannot be used with this experiment: {error}') from error


def pretrained_choices():
    if not PRETRAINED_ROOT.is_dir():
        return []
    return sorted(path.relative_to(ROOT).as_posix() for path in PRETRAINED_ROOT.rglob('*.pth'))


def refresh_pretrained(current):
    choices = pretrained_choices()
    return gr.update(choices=choices, value=current or (choices[0] if choices else None))


def resolve_pretrained(directory, enabled, preferred=''):
    from rectified_flow.lightning_train import latest_checkpoint

    if not enabled or latest_checkpoint(directory / 'flow') or (directory / 'flow' / 'checkpoint.pth').is_file():
        return ''
    preferred = str(preferred or '').strip().strip('"')
    if not preferred:
        raise gr.Error(f'Choose a pretrained voice checkpoint. Put exported flow .pth files in {PRETRAINED_ROOT} '
                       'and press Refresh.')
    path = Path(preferred)
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise gr.Error(f'Pretrained model not found: {path}')
    return str(path)


def start(name, batch, max_frames, max_updates, checkpoint_interval, device, use_fused_kernels=False,
          use_pretrained=False, pretrained_path='', realtime=False, vocoder=DEFAULT_VOCODER):
    directory = experiment_path(name)
    device_id(device)
    if not (directory / 'filelist.txt').is_file():
        raise gr.Error('Extract features for this experiment first.')
    variance_embeds = None if (directory / 'rectified_config.json').is_file() else True
    pretrained = resolve_pretrained(directory, use_pretrained, pretrained_path)
    preset = 'meanflow' if realtime else 'standard'
    from rectified_flow.train_flow import experiment_pitch_extractor, load_training_config

    pitch_extractor = experiment_pitch_extractor(directory)

    try:
        selected = load_training_config(directory, pretrained or None, use_fused_kernels, preset, pitch_extractor,
                                        variance_embeds, VOCODERS.get(vocoder, {}).get('mel'))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    precision = get_precision() or selected['flow']['precision']
    if precision not in {'fp32', 'fp16', 'bf16'}:
        raise gr.Error(f'Unsupported training precision: {precision}')
    if vocoder in VOCODERS:
        check_registry_vocoder(vocoder, selected['data'])
        if selected['flow']['val_with_vocoder']:
            resolve_vocoder(vocoder)
    elif custom_vocoder_file(vocoder) is not None:
        check_custom_vocoder(vocoder, selected['data'])
    else:
        raise gr.Error(f'Choose a listed vocoder, or put a custom vocoder in '
                       f'{CUSTOM_VOCODER_ROOT.relative_to(ROOT).as_posix()} and press Refresh.')
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


def exclusive_option(enabled):
    return gr.update(value=False, interactive=False) if enabled else gr.update(interactive=True)


def rectified_train_tab():
    gr.Markdown('### Rectified Flow')
    gr.Markdown("Trains a voice conversion model with the same architecture as "
                "[DiffSinger](https://github.com/openvpi/DiffSinger)'s acoustic model, using ContentVec features in "
                'place of phonemes. It generates mel spectrograms from the source voice and its pitch with rectified '
                'flow, and the selected vocoder turns them into audio. The vocoder sets the mel of new experiments '
                'from `rectified_flow/presets/mel`: 44.1 kHz with 128 bins for the NSF-HiFiGAN vocoders (F0), '
                '44.1 kHz with 80 bins up to 8 kHz for Vocos 44.1 kHz, 24 kHz for Vocos 24 kHz, 44.1 kHz with 128 '
                'bins up to 22.05 kHz for Pupu-Vocoder and 24 kHz with 100 Slaney bins for BigVGAN (no F0), and 32 kHz '
                'with 100 bins for NSF-BigVGAN (F0), so choose it before preprocessing. '
                'Training from scratch needs a dataset of **1 hour or more** for good results; with less audio, check '
                '**Pretrained** to fine-tune instead.')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-flow')
        device = gr.Textbox(label='Device', value='auto',
                            info='Auto uses the training config. Extraction selects detected GPUs. Examples: cuda:0, cuda:0,cuda:1, cpu.')
    with gr.Accordion('1. Prepare dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder')
        workers = gr.Number(label='CPU workers', value=4, minimum=1, precision=0)
        slicing = slicing_controls(choices=SLICERS, value='Simple', chunk_minimum=MINIMUM_CHUNK,
                                   chunk_value=MINIMUM_CHUNK,
                                   note='Audio is resampled to the sample rate of the selected vocoder.')
        preprocess_button = gr.Button('Preprocess dataset')
    with gr.Accordion('2. Train rectified flow', open=True):
        use_pretrained = gr.Checkbox(label='Pretrained', value=False)
        with gr.Row(visible=False) as pretrained_row:
            choices = pretrained_choices()
            pretrained_path = gr.Dropdown(label='Voice checkpoint to fine-tune', choices=choices,
                                          value=choices[0] if choices else None, allow_custom_value=True, scale=4,
                                          info='Exported flow .pth files found in '
                                               f'{PRETRAINED_ROOT.relative_to(ROOT).as_posix()}, or a .pth path.')
            refresh_pretrained_button = gr.Button('Refresh', scale=1)
        use_pretrained.change(lambda enabled: gr.update(visible=enabled), use_pretrained, pretrained_row,
                              queue=False)
        refresh_pretrained_button.click(refresh_pretrained, pretrained_path, pretrained_path, queue=False)
        with gr.Row():
            vocoder = gr.Dropdown(label='Vocoder', choices=vocoder_choices(), value=DEFAULT_VOCODER, scale=4,
                                  info='Sets the mel and sample rate of new experiments, so choose it before '
                                       'preprocessing. Renders the training previews and becomes the default '
                                       'vocoder of exported models. Custom NSF-HiFiGAN (F0), Vocos, Pupu-Vocoder or '
                                       'BigVGAN (no F0) vocoders (.ckpt, .pth, .onnx or .safetensors, with any '
                                       'config.json or vocoder.yaml beside them) are listed from '
                                       f'{CUSTOM_VOCODER_ROOT.relative_to(ROOT).as_posix()}.')
            refresh_vocoder_button = gr.Button('Refresh', scale=1)
        refresh_vocoder_button.click(refresh_vocoders, vocoder, vocoder, queue=False)
        gr.Markdown('Training precision follows **Settings → Training → Precision**. New experiments copy their '
                    f'training settings from `{preset_path("standard").relative_to(ROOT).as_posix()}`, or '
                    f'`{preset_path("meanflow").relative_to(ROOT).as_posix()}` with **EXPERIMENTAL: Realtime model**; edit those '
                    'files to train with custom settings. A started experiment keeps its own copy in '
                    '`logs/<model>/rectified_config.json`.')
        with gr.Row():
            batch = gr.Number(label='Max clips per batch (per GPU)', value=0, minimum=0, precision=0,
                              info=preset_info('max_batch_size'))
            max_frames = gr.Number(label='Max frames per batch (per GPU)', value=0, minimum=0, precision=0,
                                   info=preset_info('max_batch_frames', ' padded frames') +
                                        ' Lower to reduce GPU memory use.')
            max_updates = gr.Number(label='Max training updates', value=0, minimum=0, precision=0,
                               info=preset_info('max_updates', ' Lightning training steps'))
            checkpoint_interval = gr.Number(label='Checkpoint interval (updates)', value=0, minimum=0, precision=0,
                                   info=preset_info('checkpoint_interval', ' updates'))
        realtime = gr.Checkbox(label='EXPERIMENTAL: Realtime model', value=False,
                               info='Compact MeanFlow model trained for one or two inference steps.')
        use_fused_kernels = gr.Checkbox(label='Fused Linear + SoftSignGLU kernels', value=False,
                                       info='Set it when starting a new experiment: the experiment then trains with SoftSignGLU so the '
                                            'kernels can run. Experiments and pretrained checkpoints that use ATanGLU cannot enable it. '
                                            'Requires Triton and CUDA FP16 or BF16 for acceleration.')
        realtime.change(exclusive_option, realtime, use_fused_kernels, queue=False)
        with gr.Row():
            train_button = gr.Button('Start / resume rectified training', variant='primary')
            stop_button = gr.Button('Stop current rectified job', interactive=False)
    timer = gr.Timer(2)
    outputs = [preprocess_button, train_button, stop_button, timer]
    preprocess_button.click(preprocess, [name, dataset, workers, device, *slicing.inputs, vocoder], outputs,
                            queue=False)
    train_button.click(start, [name, batch, max_frames, max_updates, checkpoint_interval, device, use_fused_kernels,
                               use_pretrained, pretrained_path, realtime, vocoder], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
