from dataclasses import dataclass

import gradio as gr

CUTTING_INFO = {
    'Skip': "'Skip' keeps full clips",
    'Simple': "'Simple' cuts fixed-length slices",
    'Automatic': "'Automatic' slices at silences",
}
SILENCE_ACTIONS = ('truncate', 'compress')


@dataclass
class SlicingControls:
    cutting: gr.Radio
    chunk_len: gr.Slider
    overlap_len: gr.Slider
    truncate_silence: gr.Checkbox
    silence_action: gr.Radio
    silence_threshold: gr.Slider
    silence_minimum: gr.Slider
    silence_to: gr.Slider
    silence_compress: gr.Slider

    @property
    def inputs(self):
        return [self.cutting, self.chunk_len, self.overlap_len, self.truncate_silence, self.silence_action,
                self.silence_threshold, self.silence_minimum, self.silence_to, self.silence_compress]


def cutting_visibility(cutting, truncate_silence, silence_action):
    simple = cutting == 'Simple'
    silence = simple and truncate_silence
    return (gr.update(visible=simple), gr.update(visible=simple), gr.update(visible=simple),
            gr.update(visible=silence), gr.update(visible=silence), gr.update(visible=silence),
            gr.update(visible=silence and silence_action == 'truncate'),
            gr.update(visible=silence and silence_action == 'compress'))


def slicing_controls(choices=('Skip', 'Simple', 'Automatic'), value='Automatic',
                     note='Audio is always resampled to 44.1 kHz.', chunk_minimum=0.5, chunk_value=3.0):
    cutting = gr.Radio(label='Audio cutting', choices=list(choices), value=value,
                       info=', '.join(CUTTING_INFO[choice] for choice in choices) + '. ' + note)
    with gr.Row():
        chunk_len = gr.Slider(chunk_minimum, 10.0, chunk_value, step=0.1, label='Chunk length (sec)',
                              info="Length of the audio slice for 'Simple' method.", visible=value == 'Simple')
        overlap_len = gr.Slider(0.0, 0.4, 0.3, step=0.1, label='Overlap length (sec)',
                                info="Length of the overlap between slices for 'Simple' method.",
                                visible=value == 'Simple')
    truncate_silence = gr.Checkbox(label='Truncate silence', value=False, visible=value == 'Simple',
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
    controls = SlicingControls(cutting, chunk_len, overlap_len, truncate_silence, silence_action, silence_threshold,
                               silence_minimum, silence_to, silence_compress)
    visibility_inputs = [cutting, truncate_silence, silence_action]
    visibility_outputs = controls.inputs[1:]
    for control in visibility_inputs:
        control.change(cutting_visibility, visibility_inputs, visibility_outputs, queue=False)
    return controls


def check_cutting(cutting, chunk_len, choices, chunk_minimum):
    if cutting not in choices:
        raise gr.Error(f'Choose {", ".join(choices[:-1])} or {choices[-1]} audio cutting.')
    if cutting == 'Simple' and float(chunk_len) < chunk_minimum:
        raise gr.Error(f'Chunk length must be at least {chunk_minimum:g} seconds.')


def preprocess_step(directory, dataset, workers, cutting, chunk_len, overlap_len, truncate_silence, silence_action,
                    silence_threshold, silence_minimum, silence_to, silence_compress,
                    choices=('Skip', 'Simple', 'Automatic'), chunk_minimum=0.5):
    check_cutting(cutting, chunk_len, choices, chunk_minimum)
    if silence_action not in SILENCE_ACTIONS:
        raise gr.Error('Choose a silence action.')
    return ('shared.preprocess.preprocess',
            [str(directory), dataset, '44100', workers, cutting, 'False', 'False', '0.0', str(float(chunk_len)),
             str(float(overlap_len)), 'none', 'WAV', str(bool(truncate_silence)), str(float(silence_threshold)),
             str(float(silence_to)), str(float(silence_minimum)), silence_action, str(float(silence_compress))])
