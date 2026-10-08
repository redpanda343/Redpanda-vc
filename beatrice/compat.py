import soundfile
import torch
import torchaudio


def load(uri, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, format=None, buffer_size=4096,
         backend=None):
    data, sample_rate = soundfile.read(uri, start=frame_offset, frames=num_frames, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(data)
    return (waveform.T.contiguous() if channels_first else waveform), sample_rate


def patch_torchaudio():
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ['soundfile']
        torchaudio.load = load
