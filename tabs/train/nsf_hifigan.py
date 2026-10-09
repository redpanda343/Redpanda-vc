from pathlib import Path

import gradio as gr

from nsf_hifigan.checkpoints import finetune_sources, latest_checkpoint
from nsf_hifigan.config import KINDS, PITCH_EXTRACTORS, PRECISIONS, TRAINING, experiment_paths, read_json
from rectified_flow.resources import VOCODERS
from tabs.train.jobs import ROOT, JobRunner, device_id, experiment_path, positive_integer
from tabs.train.slicing import preprocess_step, slicing_controls

SLICERS = ('Simple',)
SCRATCH = 'scratch'
DEFAULT_SOURCES = {'pc': 'pc_nsf_hifigan', 'nsf': SCRATCH}
IDLE_CONTROLS = 3
runner = JobRunner('vocoder')


def status():
    return runner.controls(IDLE_CONTROLS)


def source_choices():
    labels = {name: spec['title'] for name, spec in VOCODERS.items()}
    return [('Train from scratch', SCRATCH), *[(labels.get(source, source), source) for source in finetune_sources()]]


def refresh_sources(kind, current):
    choices = source_choices()
    value = current if current else DEFAULT_SOURCES[kind]
    return gr.update(choices=choices, value=value)


def default_source(kind):
    return gr.update(value=DEFAULT_SOURCES[kind])


def check_kind(kind):
    if kind not in KINDS:
        raise gr.Error(f'Choose one of these vocoder types: {", ".join(KINDS.values())}.')
    return kind


def preprocess(name, dataset, workers, pitch_extractor, device, valid_clips, *slicing):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    if pitch_extractor not in PITCH_EXTRACTORS:
        raise gr.Error(f'Choose one of these pitch extractors: {", ".join(PITCH_EXTRACTORS)}.')
    if valid_clips is None or float(valid_clips) != int(valid_clips) or int(valid_clips) < 0:
        raise gr.Error('Validation clips must be zero or a positive whole number.')
    workers = positive_integer(workers, 'CPU workers')
    steps = [
        preprocess_step(directory, dataset, workers, *slicing, choices=SLICERS),
        ('nsf_hifigan.binarize', [str(directory), '--pitch-extractor', pitch_extractor, '--device', device_option(device),
                                  '--workers', workers, '--valid-clips', str(int(valid_clips))]),
    ]
    return runner.launch(name, steps, 'Preprocessing vocoder dataset', IDLE_CONTROLS, 'done. Ready to train.')


def device_option(device):
    device = str(device).strip().lower() or 'auto'
    device_id(device)
    return device


def resolve_source(kind, source, resuming):
    source = str(source or '').strip().strip('"')
    if resuming or source in {'', SCRATCH}:
        return ''
    if source in VOCODERS:
        if kind != 'pc':
            raise gr.Error(f'{VOCODERS[source]["title"]} is a PC-NSF-HiFiGAN vocoder. Choose the PC-NSF-HiFiGAN '
                           'type, or train NSF-HiFiGAN from scratch or from a classic NSF-HiFiGAN checkpoint path.')
        return source
    if not Path(source).is_file() and not (ROOT / source).is_file():
        raise gr.Error(f'Pretrained vocoder not found: {source}')
    return source


def optional_number(value, label, integer=True):
    if value is None or float(value) == 0:
        return None
    if float(value) < 0 or (integer and float(value) != int(value)):
        raise gr.Error(f'{label} must be 0 or a positive {"whole number" if integer else "number"}.')
    return str(int(value) if integer else float(value))


def start(name, kind, source, discriminator_warmup, precision, batch, crop_frames, max_updates, checkpoint_interval,
          learning_rate, key_aug, workers, device):
    directory = experiment_path(name)
    check_kind(kind)
    paths = experiment_paths(directory)
    if not paths['index'].is_file():
        raise gr.Error('Preprocess the dataset for this vocoder first.')
    resuming = latest_checkpoint(paths['output']) is not None
    saved = read_json(paths['config']) if resuming else None
    if saved and saved['kind'] != kind:
        raise gr.Error(f"This experiment trains a {KINDS[saved['kind']]} vocoder. Choose that type to resume, "
                       'or use another model name.')
    if precision not in PRECISIONS:
        raise gr.Error(f'Choose one of these precisions: {", ".join(PRECISIONS)}.')
    source = resolve_source(kind, source, resuming)
    arguments = ['--model-name', str(name).strip(), '--kind', kind, '--precision', precision,
                 '--device', device_option(device), '--workers', positive_integer(workers, 'CPU workers'),
                 '--key-aug' if key_aug else '--no-key-aug']
    if source:
        arguments.extend(['--pretrained', source, '--discriminator-warmup',
                          str(int(discriminator_warmup or 0))])
    for flag, value, label in (('--batch-size', batch, 'Batch size'),
                               ('--crop-mel-frames', crop_frames, 'Crop length'),
                               ('--checkpoint-interval', checkpoint_interval, 'Checkpoint interval')):
        arguments.extend([flag, positive_integer(value, label)])
    for flag, value in (('--max-updates', optional_number(max_updates, 'Max training steps')),
                        ('--learning-rate', optional_number(learning_rate, 'Learning rate', integer=False))):
        if value is not None:
            arguments.extend([flag, value])
    description = 'Fine-tuning vocoder' if source else 'Training vocoder'
    return runner.launch(name, [('nsf_hifigan.train', arguments)], description, IDLE_CONTROLS)


def export(name):
    directory = experiment_path(name)
    if latest_checkpoint(experiment_paths(directory)['output']) is None:
        raise gr.Error('This experiment has no vocoder checkpoint yet. Train it first.')
    return runner.launch(name, [('nsf_hifigan.export', ['--model-name', str(name).strip()])], 'Exporting vocoder',
                         IDLE_CONTROLS)


def stop():
    return runner.stop(IDLE_CONTROLS)


def nsf_hifigan_train_tab():
    gr.Markdown('### NSF-HiFiGAN vocoder')
    gr.Markdown('Trains or fine-tunes an OpenVPI NSF-HiFiGAN vocoder (44.1 kHz, hop 512, 128 mel bins) with the '
                '[SingingVocoders](https://github.com/openvpi/SingingVocoders) recipe. Checkpoints and TensorBoard '
                'logs go to `logs/<model>/vocoder`. **Export** writes `model.ckpt` and `config.json` to '
                '`models/pretraineds/rectified/trained/<model>_<steps>s`, where Rectified Flow inference lists it as a '
                'vocoder. A finished run exports automatically.')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-vocoder')
        device = gr.Textbox(label='Device', value='auto',
                            info='auto, cpu, cuda:0 or cuda:0,cuda:1. RMVPE extraction uses the first GPU.')
    kind = gr.Radio(label='Vocoder type', value='pc', choices=[(title, key) for key, title in KINDS.items()],
                    info='PC-NSF-HiFiGAN trains with pitch-cycle augmentation so pitch-shifted output stays clean; '
                         'it is the type Rectified Flow ships with. NSF-HiFiGAN is the classic model.')
    workers = gr.Number(label='CPU workers', value=4, minimum=1, precision=0,
                        info='Processes for slicing, Parselmouth F0 and the training data loader.')
    with gr.Accordion('1. Prepare dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder',
                             info='A validation subfolder, if present, holds the clips used for validation.')
        slicing = slicing_controls(choices=SLICERS, value='Simple')
        with gr.Row():
            pitch_extractor = gr.Radio(label='Pitch extractor', choices=list(PITCH_EXTRACTORS), value='rmvpe',
                                       info='RMVPE matches the F0 Rectified Flow extracts at inference. Parselmouth '
                                            'is the original SingingVocoders recipe.')
            valid_clips = gr.Number(label='Validation clips', value=5, minimum=0, precision=0,
                                    info='Held out when there is no validation subfolder, at most a tenth of the '
                                         'clips.')
        preprocess_button = gr.Button('Preprocess dataset')
    with gr.Accordion('2. Train vocoder', open=True):
        with gr.Row():
            source = gr.Dropdown(label='Fine-tune from', choices=source_choices(), value=DEFAULT_SOURCES['pc'],
                                 allow_custom_value=True, scale=4,
                                 info='The Rectified Flow vocoders (downloaded on first use), your exported '
                                      'vocoders, or a .ckpt/.onnx path. Only used when the experiment has no '
                                      'checkpoint yet.')
            refresh = gr.Button('Refresh', scale=1)
        with gr.Row():
            discriminator_warmup = gr.Number(label='Discriminator warm-up (steps)', value=1000, minimum=0, precision=0,
                                             info='Released vocoders ship without discriminator weights, so only the '
                                                  'discriminator trains for these first steps.')
            precision = gr.Radio(label='Precision', choices=list(PRECISIONS), value='fp32',
                                 info='SingingVocoders trains in fp32 and advises against bf16.')
        with gr.Row():
            batch = gr.Number(label='Batch size', value=TRAINING['batch_size'], minimum=1, precision=0,
                              info='Clips per step and GPU. Lower it to reduce GPU memory use.')
            crop_frames = gr.Number(label='Crop length (mel frames)', value=TRAINING['crop_mel_frames'], minimum=1,
                                    precision=0, info='32 frames is about 0.37 s. Higher gives better results but uses '
                                                      'more GPU memory; lower it if training runs out of memory.')
            checkpoint_interval = gr.Number(label='Checkpoint interval (steps)', value=TRAINING['checkpoint_interval'],
                                            minimum=1, precision=0, info='Validation runs at the same interval.')
            max_updates = gr.Number(label='Max training steps', value=0, minimum=0, precision=0,
                                    info=f"0 uses {TRAINING['finetune_max_updates']} to fine-tune, "
                                         f"{TRAINING['max_updates']} from scratch, or the experiment's saved value.")
            learning_rate = gr.Number(label='Learning rate', value=0, minimum=0,
                                      info=f"0 uses {TRAINING['finetune_learning_rate']:g} to fine-tune, "
                                           f"{TRAINING['learning_rate']:g} from scratch, or the experiment's saved "
                                           'value.')
        key_aug = gr.Checkbox(label='Speed augmentation', value=False,
                              info='Resamples random clips 0.9-1.4x faster to cover more pitches. It may lower quality.')
        with gr.Row():
            train_button = gr.Button('Start / resume vocoder training', variant='primary')
            export_button = gr.Button('Export latest checkpoint')
            stop_button = gr.Button('Stop current vocoder job', interactive=False)
    timer = gr.Timer(2)
    outputs = [preprocess_button, train_button, export_button, stop_button, timer]
    kind.change(default_source, [kind], [source], queue=False)
    refresh.click(refresh_sources, [kind, source], [source], queue=False)
    preprocess_button.click(preprocess, [name, dataset, workers, pitch_extractor, device, valid_clips,
                                         *slicing.inputs], outputs, queue=False)
    train_button.click(start, [name, kind, source, discriminator_warmup, precision, batch, crop_frames, max_updates,
                               checkpoint_interval, learning_rate, key_aug, workers, device], outputs, queue=False)
    export_button.click(export, [name], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
