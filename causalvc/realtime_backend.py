"""A CausalVC voice for the realtime engine.

It converts each block as it arrives and its output is already continuous
across blocks.  A voice is a folder in ``models/`` holding its description and
its decoder, as written when a voice is built:

    MyVoice-10000s/
      MyVoice-10000s.causalvc.json     {"decoder": "decoder.pt", "speaker": "MyVoice"}
      decoder.pt

The folder is self-contained (see ``causalvc.voice``); the content encoder and
pitch tracker every voice shares come with the project's base.  ``"runtime":
"classic"`` in the description selects the original runtime, which recomputes
its whole window every block.
"""

import torch

#: A voice file (.cvc), or an older voice folder's description (the second spelling is the
#: project's earlier name and still loads).
SUFFIXES = (".cvc", ".causalvc.json", ".nextvc.json")
#: Blocks are multiples of 10 ms (480 samples) at 100 Hz, or 20 ms (960 samples) at 50 Hz.
SAMPLE_RATE = 48000
BLOCK_MULTIPLE = 480
BLOCK_MULTIPLE_50HZ = 960


def is_causalvc(model_path):
    return str(model_path).endswith(SUFFIXES)


class CausalVC:
    is_causalvc = True
    tgt_sr = SAMPLE_RATE

    def __init__(
        self,
        key,
        model_path,
        device,
        pitch_offset=0.0,
        pitch_ceiling=0.0,
        pitch_ceiling_keep=1.0,
        pitch_floor=0.0,
        pitch_floor_keep=1.0,
        cuda_graphs=False,
        index_rate=0.0,
    ):
        from causalvc.stream import DELAY
        from causalvc.voice import load

        # Paths in the description are looked for beside it first, so a voice folder works wherever it is copied.
        spec = load(model_path)
        self.speaker = spec["speaker"]
        if spec["runtime"] == "classic":  # the original runtime, which recomputes its whole window every block
            from causalvc.stream import StreamingVC

            self.vc = StreamingVC(spec["decoder"], spec["pitch"], str(device), spec["encoder"])
        else:  # the cached runtime: same model, computes only what is new each block
            from causalvc.fast import streaming

            # cuda_graphs: replay each block's recorded kernels instead of launching them one by one.
            self.vc = streaming(spec["decoder"], spec["pitch"], str(device), spec["encoder"], graphs=cuda_graphs)
        if self.speaker not in self.vc.speakers:
            raise ValueError(f"{self.speaker} is not a voice of {spec['decoder']}")
        self.block_multiple = BLOCK_MULTIPLE if getattr(self.vc, "hop100", True) else BLOCK_MULTIPLE_50HZ
        self.delay_seconds = getattr(self.vc, "delay", DELAY) / 100.0
        # The voice's index (content frames of its own audio), when its file carries one.
        self.has_index = bool(spec.get("index"))
        if self.has_index:
            self.vc.set_index(torch.load(spec["index"], map_location="cpu")["frames"])
        self.key = float(key)
        self.change_pitch_map(pitch_offset, pitch_ceiling, pitch_ceiling_keep, pitch_floor, pitch_floor_keep)
        self.change_index_rate(index_rate)

    def change_key(self, key):
        """Pitch shift in semitones; the stream carries on."""
        self.key = float(key)
        self._set_voice()

    def change_pitch_map(self, offset, ceiling, keep, floor=0.0, floor_keep=1.0):
        """The advanced pitch settings; the stream carries on."""
        self.pitch_offset, self.pitch_ceiling, self.pitch_ceiling_keep = float(offset), float(ceiling), float(keep)
        self.pitch_floor, self.pitch_floor_keep = float(floor), float(floor_keep)
        self._set_voice()

    def change_index_rate(self, rate):
        """How much of what is said is drawn from the voice's own audio (0-1); the stream carries on.
        Does nothing for a voice without an index."""
        self.index_rate = float(rate)
        self.vc.set_index_rate(self.index_rate)

    def _set_voice(self):
        self.vc.set_voice(
            self.speaker,
            self.key,
            self.pitch_offset,
            self.pitch_ceiling,
            self.pitch_ceiling_keep,
            self.pitch_floor,
            self.pitch_floor_keep,
        )

    def reset_stream_state(self):
        self.vc.reset()

    def process(self, block):
        """One block of 48 kHz mono samples (a torch tensor) -> the same length."""
        return self.vc.process(block)

    @property
    def uses_cuda_graph(self):
        return getattr(self.vc, "graph", None) is not None

    def warm(self, block_frame, record=True):
        """Run a few silent blocks of this length (and, with cuda_graphs, record one), then start afresh.
        With ``record`` False the recording is left to ``record()`` (a voice warmed while another
        voice is converting: recording needs the GPU to itself)."""
        for _ in range(4):
            self.vc.process(torch.zeros(block_frame, device=self.vc.device))
        self.vc.reset()
        if record:
            self.record(block_frame)

    def record(self, block_frame):
        """Get ready for blocks of this length; with cuda_graphs, record one as a CUDA graph."""
        if hasattr(self.vc, "prepare"):
            self.vc.prepare(block_frame)
