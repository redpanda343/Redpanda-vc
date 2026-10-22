from pathlib import Path

import gradio as gr

from nsf_hifigan.checkpoints import finetune_sources, latest_checkpoint
from nsf_hifigan.config import (
    FINETUNE, KINDS, PITCH_EXTRACTORS, PRECISIONS, TRAINING, VOCOS_TRAINING, experiment_paths, read_json,
)
from nsf_hifigan.resources import FINETUNE_SOURCE, FINETUNE_TITLE, VOCOS_SOURCE, VOCOS_TITLE
from tabs.train.jobs import JobRunner, device_id, experiment_path, positive_integer
from tabs.train.slicing import preprocess_step, slicing_controls

SLICERS = ('Simple',)
SCRATCH = 'scratch'
DEFAULT_SOURCES = {'pc': FINETUNE_SOURCE, 'nsf': SCRATCH, 'vocos': VOCOS_SOURCE}
IDLE_CONTROLS = 4
runner = JobRunner('vocoder')


def status():
    return runner.controls(IDLE_CONTROLS)


def source_choices(kind='pc'):
    sources = {'pc': [(FINETUNE_TITLE, source) for source in finetune_sources()],
               'vocos': [(VOCOS_TITLE, VOCOS_SOURCE)]}
    return [('Train from scratch', SCRATCH), *sources.get(kind, [])]


def default_source(kind):
    return gr.update(choices=source_choices(kind), value=DEFAULT_SOURCES[kind])


def check_kind(kind):
    if kind not in KINDS:
        raise gr.Error(f'Choose one of these vocoder types: {", ".join(KINDS.values())}.')
    return kind


def slice_dataset(name, dataset, workers, *slicing):
    directory = experiment_path(name)
    dataset = str(dataset).strip().strip('"')
    if not Path(dataset).is_dir():
        raise gr.Error('The dataset folder does not exist.')
    workers = positive_integer(workers, 'CPU workers')
    steps = [preprocess_step(directory, dataset, workers, *slicing, choices=SLICERS)]
    return runner.launch(name, steps, 'Slicing vocoder dataset', IDLE_CONTROLS, 'done. Extract the features next.')


def extract_features(name, workers, pitch_extractor, device, valid_clips):
    directory = experiment_path(name)
    if pitch_extractor not in PITCH_EXTRACTORS:
        raise gr.Error(f'Choose one of these pitch extractors: {", ".join(PITCH_EXTRACTORS)}.')
    if valid_clips is None or float(valid_clips) != int(valid_clips) or int(valid_clips) < 0:
        raise gr.Error('Validation clips must be zero or a positive whole number.')
    sliced = directory / 'sliced_audios'
    if not sliced.is_dir() or not any(sliced.iterdir()):
        raise gr.Error('This experiment has no sliced audio. Slice the dataset first.')
    steps = [('nsf_hifigan.binarize', [str(directory), '--pitch-extractor', pitch_extractor,
                                       '--device', device_option(device),
                                       '--workers', positive_integer(workers, 'CPU workers'),
                                       '--valid-clips', str(int(valid_clips))])]
    return runner.launch(name, steps, 'Extracting vocoder features', IDLE_CONTROLS, 'done. Ready to train.')


def device_option(device):
    device = str(device).strip().lower() or 'auto'
    device_id(device)
    return device


def resolve_source(kind, source, resuming):
    source = str(source or '').strip().strip('"')
    if resuming or source in {'', SCRATCH}:
        return ''
    if kind == 'vocos':
        if source != VOCOS_SOURCE:
            raise gr.Error('Choose the Vocos 24 kHz backbone or train from scratch.')
        return source
    if source != FINETUNE_SOURCE:
        raise gr.Error('Choose the PC-NSF-HiFiGAN 2025.02 checkpoint or train from scratch.')
    if kind != 'pc':
        raise gr.Error('This checkpoint fine-tunes PC-NSF-HiFiGAN. Choose that vocoder type to fine-tune.')
    return source


def optional_number(value, label, integer=True):
    if value is None or float(value) == 0:
        return None
    if float(value) < 0 or (integer and float(value) != int(value)):
        raise gr.Error(f'{label} must be 0 or a positive {"whole number" if integer else "number"}.')
    return str(int(value) if integer else float(value))


def start(name, kind, source, precision, batch, crop_frames, max_updates, checkpoint_interval,
          learning_rate, key_aug, workers, device):
    directory = experiment_path(name)
    check_kind(kind)
    paths = experiment_paths(directory)
    if not paths['index'].is_file():
        raise gr.Error('Extract the features for this vocoder first.')
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
        arguments.extend(['--pretrained', source])
    for flag, value, label in (('--batch-size', batch, 'Batch size'),
                               ('--crop-mel-frames', crop_frames, 'Crop length')):
        arguments.extend([flag, positive_integer(value, label)])
    for flag, value in (('--checkpoint-interval', optional_number(checkpoint_interval, 'Checkpoint interval')),
                        ('--max-updates', optional_number(max_updates, 'Max training steps')),
                        ('--learning-rate', optional_number(learning_rate, 'Learning rate', integer=False))):
        if value is not None:
            arguments.extend([flag, value])
    description = 'Fine-tuning vocoder' if source and kind != 'vocos' else 'Training vocoder'
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
    gr.Markdown('### Vocoder')
    gr.Markdown('Trains or fine-tunes a Rectified Flow vocoder on the flow mel (44.1 kHz, hop 512, 128 mel bins). '
                'NSF-HiFiGAN and PC-NSF-HiFiGAN take the mel and its F0 and follow the '
                '[SingingVocoders](https://github.com/openvpi/SingingVocoders) recipe. '
                '[Vocos](https://github.com/gemelo-ai/vocos) takes only the mel and runs much faster, which suits '
                'real-time conversion. Checkpoints and TensorBoard logs go to `logs/<model>/vocoder`. **Export** '
                'writes `model.ckpt` and `config.json` to `models/pretraineds/rectified/trained/<model>_<steps>s`, '
                'where Rectified Flow inference lists it as a vocoder. A finished run exports automatically.')
    with gr.Row():
        name = gr.Textbox(label='Model name', value='my-vocoder')
        device = gr.Textbox(label='Device', value='auto',
                            info='auto, cpu, cuda:0 or cuda:0,cuda:1. RMVPE extraction uses the first GPU.')
    kind = gr.Radio(label='Vocoder type', value='pc', choices=[(title, key) for key, title in KINDS.items()],
                    info='F0 types follow the flow F0; Vocos (no F0) renders the mel alone. PC-NSF-HiFiGAN '
                         'fine-tunes from the full SingingVocoders checkpoint, Vocos starts from the 24 kHz Vocos '
                         'backbone, and classic NSF-HiFiGAN trains from scratch.')
    workers = gr.Number(label='CPU workers', value=4, minimum=1, precision=0,
                        info='Processes for slicing, feature extraction and the training data loader.')
    with gr.Accordion('1. Slice dataset', open=True):
        dataset = gr.Textbox(label='Dataset folder',
                             info='A validation subfolder, if present, holds the clips used for validation.')
        slicing = slicing_controls(choices=SLICERS, value='Simple')
        slice_button = gr.Button('Slice dataset')
    with gr.Accordion('2. Extract features', open=True):
        gr.Markdown('Saves the audio, mel and F0 of every sliced clip to `logs/<model>/vocoder/data` for training. '
                    'It needs about 1.25 times the space of the sliced audio as 32-bit WAV, and can be deleted '
                    'after training.')
        with gr.Row():
            pitch_extractor = gr.Radio(label='Pitch extractor', choices=['rmvpe'], value='rmvpe',
                                       info='RMVPE matches the F0 Rectified Flow extracts at inference. Vocos '
                                            'trains without F0.')
            valid_clips = gr.Number(label='Validation clips', value=5, minimum=0, precision=0,
                                    info='Held out when there is no validation subfolder, at most a tenth of the '
                                         'clips.')
        extract_button = gr.Button('Extract features')
    with gr.Accordion('3. Train vocoder', open=True):
        source = gr.Dropdown(label='Start from', choices=source_choices(), value=DEFAULT_SOURCES['pc'],
                             allow_custom_value=False,
                             info='Downloads automatically. The PC-NSF-HiFiGAN checkpoint loads generator and '
                                  'discriminator weights; the Vocos backbone initializes all but the Vocos input and '
                                  'output layers, and trains with the from-scratch schedule. Only used for new '
                                  'experiments; existing checkpoints resume normally.')
        precision = gr.Radio(label='Precision', choices=list(PRECISIONS), value='fp32',
                             info='SingingVocoders trains in fp32 and advises against bf16.')
        with gr.Row():
            batch = gr.Number(label='Batch size', value=TRAINING['batch_size'], minimum=1, precision=0,
                              info='Clips per step and GPU. Lower it to reduce GPU memory use.')
            crop_frames = gr.Number(label='Crop length (mel frames)', value=TRAINING['crop_mel_frames'], minimum=1,
                                    precision=0, info='32 frames is about 0.37 s. Higher gives better results but uses '
                                                      'more GPU memory; lower it if training runs out of memory.')
            checkpoint_interval = gr.Number(label='Checkpoint interval (steps)', value=0, minimum=0, precision=0,
                                            info=f"0 uses {FINETUNE['checkpoint_interval']} to fine-tune, "
                                                 f"{TRAINING['checkpoint_interval']} from scratch, or the experiment's "
                                                 'saved value. Validation runs at the same interval.')
            max_updates = gr.Number(label='Max training steps', value=0, minimum=0, precision=0,
                                    info=f"0 uses {FINETUNE['max_updates']} to fine-tune, "
                                         f"{TRAINING['max_updates']} from scratch, or the experiment's saved value.")
            learning_rate = gr.Number(label='Learning rate', value=0, minimum=0,
                                      info=f"0 uses {TRAINING['finetune_learning_rate']:g} to fine-tune, "
                                           f"{TRAINING['learning_rate']:g} from scratch, "
                                           f"{VOCOS_TRAINING['learning_rate']:g} for Vocos, or the experiment's "
                                           'saved value.')
        key_aug = gr.Checkbox(label='Speed augmentation', value=False,
                              info='Resamples random clips 0.9-1.4x faster to cover more pitches. It may lower quality. '
                                   'Fine-tuning always trains without it.')
        with gr.Row():
            train_button = gr.Button('Start / resume vocoder training', variant='primary')
            export_button = gr.Button('Export latest checkpoint')
            stop_button = gr.Button('Stop current vocoder job', interactive=False)
    timer = gr.Timer(2)
    outputs = [slice_button, extract_button, train_button, export_button, stop_button, timer]
    kind.change(default_source, [kind], [source], queue=False)
    slice_button.click(slice_dataset, [name, dataset, workers, *slicing.inputs], outputs, queue=False)
    extract_button.click(extract_features, [name, workers, pitch_extractor, device, valid_clips], outputs,
                         queue=False)
    train_button.click(start, [name, kind, source, precision, batch, crop_frames, max_updates,
                               checkpoint_interval, learning_rate, key_aug, workers, device], outputs, queue=False)
    export_button.click(export, [name], outputs, queue=False)
    stop_button.click(stop, [], outputs, queue=False)
    timer.tick(status, [], outputs, queue=False, show_progress='hidden')
