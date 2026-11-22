"""A voice's index: content frames of the voice's own audio, for pulling what a speaker says
toward how the voice itself says it (like an RVC index).

The live voice changer looks up each content frame's nearest frames in the index and blends
them in at the strength set on its Model page ("Index", off by default). A higher strength
sounds more like the voice; too high blurs words.

The index travels inside the voice's .cvc as index.pt: {"frames": [N, C] half floats}, made
with the voice's own content encoder. A few thousand representative frames (k-means over
every frame of the audio) keep it small.

    python -m causalvc.index VOICE.cvc --audio FOLDER   add (or replace) the index of a built voice
"""

from __future__ import annotations

import argparse
import random
import zipfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

#: Representative frames kept: about 12 MB at 768 channels, half precision.
SIZE = 8192
#: Frames looked at when choosing them (about 40 minutes of speech).
MAX_FRAMES = 120_000
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".mp3")


def audio_files(path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)


@torch.no_grad()
def content_frames(enc, files, device, max_frames: int = MAX_FRAMES, seconds: float = 10.0) -> torch.Tensor:
    """[M, C] content frames of the speech in `files` (silence left out), at most `max_frames`,
    taken evenly from all of the audio."""
    feats = []
    step = int(16000 * seconds)
    for path in files:
        wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
        w16 = torchaudio.functional.resample(torch.from_numpy(wav.mean(1)), sr, 16000).to(device)
        for i in range(0, len(w16), step):
            seg = w16[i : i + step]
            if len(seg) < 8000:
                continue
            f = enc.encode(seg[None]).float()[0]
            # each content frame covers 20 ms (320 samples): keep the ones within 40 dB of the loudest
            n = min(len(f), len(seg) // 320)
            db = 10 * seg[: n * 320].view(n, 320).pow(2).mean(1).add(1e-10).log10()
            feats.append(f[:n][db > db.max() - 40].half())
    if not feats:
        raise ValueError("no speech found in the audio")
    frames = torch.cat(feats)
    if len(frames) > max_frames:
        frames = frames[torch.linspace(0, len(frames) - 1, max_frames, device=frames.device).long()]
    return frames


@torch.no_grad()
def representative(frames: torch.Tensor, size: int = SIZE, iterations: int = 20, seed: int = 0) -> torch.Tensor:
    """`size` frames standing for all of `frames`: spherical k-means (frames are compared by
    angle at lookup), each centre the mean of the frames nearest it."""
    x = frames.float()
    if len(x) <= size:
        return x.half()
    unit = torch.nn.functional.normalize(x, dim=-1)
    g = torch.Generator(device="cpu").manual_seed(seed)
    centres = x[torch.randperm(len(x), generator=g)[:size].to(x.device)].clone()
    for _ in range(iterations):
        c_unit = torch.nn.functional.normalize(centres, dim=-1)
        nearest = torch.cat([(unit[i : i + 8192] @ c_unit.T).argmax(1) for i in range(0, len(x), 8192)])
        sums = torch.zeros_like(centres).index_add_(0, nearest, x)
        counts = torch.bincount(nearest, minlength=size).float()
        empty = counts == 0
        centres = torch.where(empty[:, None], centres, sums / counts.clamp_min(1)[:, None])
        if empty.any():  # restart an unused centre on a random frame
            pick = torch.randint(0, len(x), (int(empty.sum()),), generator=g).to(x.device)
            centres[empty] = x[pick]
    return centres.half()


def build(enc, audio, device, size: int = SIZE) -> bytes:
    """The index.pt bytes for a voice made from `audio` (a file or folder)."""
    import io

    files = audio_files(audio)
    random.Random(0).shuffle(files)  # a capped bank then spans the whole set, not its first files
    buffer = io.BytesIO()
    torch.save({"frames": representative(content_frames(enc, files, device), size).cpu()}, buffer)
    return buffer.getvalue()


def lookup(content: torch.Tensor, frames: torch.Tensor, frames_unit: torch.Tensor, k: int = 8) -> torch.Tensor:
    """[T, C] -> the similarity-weighted mean of each frame's k nearest index frames."""
    sim = torch.nn.functional.normalize(content.float(), dim=-1) @ frames_unit.T
    top, idx = sim.topk(k, dim=-1)
    w = torch.softmax(top / 0.05, dim=-1)
    return (frames[idx].float() * w[..., None]).sum(1).to(content.dtype)


def add_to_cvc(paths, audio, device: str = "cuda:0") -> None:
    """Build an index from `audio` and put it in each .cvc of `paths` (checkpoints of one voice;
    replacing an index there). Built once per content encoder the voices use."""
    from causalvc import voice
    from causalvc.infer import load_encoder

    made = {}
    for path in map(Path, paths):
        encoder = str(Path(voice.load(path)["encoder"] or voice.encoder_path(None)).resolve())
        if encoder not in made:
            made[encoder] = build(load_encoder(encoder, device), audio, device)
        voice.replace_member(path, voice.CVC_INDEX, made[encoder])
        print(f"{path.name}: index added", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("voice", nargs="+", help=".cvc files to give an index")
    ap.add_argument("--audio", required=True, help="the voice's own audio: a file or folder")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    for path in a.voice:
        if not zipfile.is_zipfile(path):
            raise SystemExit(f"{path} is not a .cvc voice")
    add_to_cvc(a.voice, a.audio, a.device)


if __name__ == "__main__":
    main()
