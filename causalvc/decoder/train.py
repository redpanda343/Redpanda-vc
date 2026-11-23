"""Train the causal decoder as a multi-speaker base.

    uv run python -m causalvc.decoder.train --name dec-s0 --encoder runs/models/enc-la2-460/encoder.pt

Speakers: VCTK and EARS at 48 kHz, plus the harness's target voices (their
training clips only), so a checkpoint can be scored with `vcbench` directly.
Content, pitch and loudness are extracted on the GPU from each crop; the decoder
reconstructs the last part of the crop from them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import signal
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

from causalvc.decoder.disc import Discriminators, EnergyLoss, MelLoss, SpecLoss, d_loss, g_loss
from causalvc.decoder.model import FRAME, SR, CausalMel, Decoder
from causalvc.augment import add_noise, add_silent_transients, colour
from causalvc.decoder.model import tone as tone_of
from causalvc.infer import CausalVC, extract_all, load_encoder
from causalvc.pitch import Rmvpe
from causalvc.progress import Progress, stop_before_training


def speaker_files(args) -> dict[str, list[str]]:
    spk: dict[str, list[str]] = {}
    for d in sorted(Path(args.vctk).glob("wav48_silence_trimmed/*")):
        files = sorted(str(p) for p in d.glob("*_mic1.flac"))
        if files:
            spk[f"vctk_{d.name}"] = files
    for d in sorted(Path(args.ears).glob("p*")):
        files = sorted(str(p) for p in d.glob("*.wav"))
        if files:
            spk[f"ears_{d.name}"] = files
    for name, t in json.loads(Path(args.manifest).read_text())["targets"].items():
        spk[name] = t["train"]
    spk.update(voice_files(args))
    return spk


def voice_files(args) -> dict[str, list[str]]:
    """New voices given as NAME=folder or NAME=file."""
    found: dict[str, list[str]] = {}
    for item in args.voice or []:
        name, src = item.split("=", 1)
        src = Path(src)
        files = sorted(str(p) for p in src.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".ogg", ".mp3")) if src.is_dir() else [str(src)]
        assert files, f"{name}: no audio in {src}"
        found[name] = files
    return found


EXPRESSIVE_TAGS = (
    "nonverbal_", "emo_amazement", "emo_extasy", "emo_fear", "emo_pain", "emo_anger",
    "emo_sadness", "emo_relief", "interjection_", "highpitch", "lowpitch", "whisper", "loud",
)


class Crops(torch.utils.data.Dataset):
    """A random speaker, a random file, a random crop, resampled to 48 kHz."""

    def __init__(self, spk: dict[str, list[str]], seconds: float, steps: int, only: list[str] | None = None,
                 silence_share: float = 0.0):
        self.names, self.spk, self.seconds, self.steps = list(spk), spk, seconds, steps
        self.silence_share = silence_share
        self.pick = [self.names.index(n) for n in only] if only else list(range(len(self.names)))
        # During base training, oversample expressive / high-pitch / whisper / laugh EARS clips so 35-1000 Hz
        # registers and non-verbal vocals are densely covered on real human recordings.
        self.expressive = {
            name: [p for p in files if any(tag in Path(p).stem for tag in EXPRESSIVE_TAGS)] if not only else []
            for name, files in spk.items()
        }
        self.file_meta: dict[str, tuple[int, int]] = {}
        self.cum_weights: dict[str, list[float]] = {}
        # Weight files by duration during voice fine-tuning so mixing long and short recordings
        # samples every second of speech evenly without slicing files on disk.
        if only:
            for s in self.pick:
                name = self.names[s]
                files = self.spk[name]
                weights: list[float] = []
                for p in files:
                    try:
                        info = sf.info(p)
                        self.file_meta[p] = (info.frames, info.samplerate)
                        weights.append(max(info.frames / max(1, info.samplerate), 0.05))
                    except Exception:
                        weights.append(1.0)
                if len(weights) > 1:
                    self.cum_weights[name] = list(itertools.accumulate(weights))

    def _pick_file(self, name: str, pool: list[str]) -> tuple[str, int, int]:
        cum = self.cum_weights.get(name) if pool is self.spk[name] else None
        path = random.choices(pool, cum_weights=cum, k=1)[0] if cum and len(cum) == len(pool) else random.choice(pool)
        meta = self.file_meta.get(path)
        if meta is None:
            info = sf.info(path)
            meta = (info.frames, info.samplerate)
            self.file_meta[path] = meta
        return path, meta[0], meta[1]

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, _: int):
        n = int(self.seconds * SR)
        if self.silence_share > 0 and random.random() < self.silence_share:
            s = random.choice(self.pick)
            if random.random() < 0.5:
                return torch.zeros(n), s
            # Release-into-silence crop: speech that fades cleanly to exact 0 halfway through the clip.
            name = self.names[s]
            path, frames, sr_in = self._pick_file(name, self.spk[name])
            want = int(self.seconds * sr_in)
            wav, sr = sf.read(path, start=random.randint(0, max(0, frames - want)), frames=want, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav[:, 0])
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            wav = torch.nn.functional.pad(wav, (max(0, n - len(wav)), 0))[:n]
            peak = wav.abs().max().clamp_min(1e-4)
            wav = wav * min(10 ** (random.uniform(-14.0, 0.0) / 20), 0.95 / peak.item())
            cut = int(n * random.uniform(0.55, 0.80))
            fade = min(int(0.020 * SR), cut)
            wav[cut - fade : cut] *= 0.5 * (1.0 + torch.cos(math.pi * torch.arange(fade) / max(1, fade)))
            wav[cut:] = 0.0
            return wav, s
        for _try in range(16):
            s = random.choice(self.pick)
            name = self.names[s]
            exp = self.expressive.get(name)
            pool = exp if exp and random.random() < 0.40 else self.spk[name]
            path, frames, sr_in = self._pick_file(name, pool)
            want = int(self.seconds * sr_in)
            start = random.randint(0, max(0, frames - want))
            wav, sr = sf.read(path, start=start, frames=want, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav[:, 0])
            if sr != SR:
                wav = torchaudio.functional.resample(wav, sr, SR)
            wav = torch.nn.functional.pad(wav, (max(0, n - len(wav)), 0))[:n]  # short clips start in silence
            if wav[-n // 3 :].pow(2).mean().sqrt() > 2e-3:  # the reconstructed part must contain sound
                break
        peak = wav.abs().max().clamp_min(1e-4)
        gain = min(10 ** (random.uniform(-24.0, 3.0) / 20), 0.99 / peak.item())
        return wav * gain, s


def detect_fmax(files: list[str]) -> float | None:
    """Effective upper frequency (Hz) of a target voice's recordings, or None when full-band 48 kHz.
    Prevents 44.1 kHz or 16 kHz-cutoff livestream recordings from distorting spectral losses above their cutoff."""
    if not files:
        return None
    sampled = random.Random(0).sample(files, min(len(files), 16))
    nyq = 24000.0
    powers = []
    win = torch.hann_window(2048)
    for path in sampled:
        try:
            info = sf.info(path)
            nyq = min(nyq, 0.475 * info.samplerate)
            want = min(info.frames, 2 * info.samplerate)
            n_cuts = 8 if len(sampled) <= 2 and info.frames > 16 * info.samplerate else 1
            for idx_c in range(n_cuts):
                start = int((info.frames - want) * (idx_c + 0.5) / n_cuts)
                w, sr = sf.read(path, start=start, frames=want, dtype="float32", always_2d=True)
                x = torch.from_numpy(w[:, 0])
                if sr != SR:
                    x = torchaudio.functional.resample(x, sr, SR)
                if len(x) >= 2048 and x.pow(2).mean().sqrt() > 2e-3:
                    powers.append(torch.stft(x, 2048, 512, window=win, return_complex=True).abs().pow(2).mean(-1))
        except Exception:
            continue
    if powers:
        S = torch.stack(powers).mean(0)
        fr = torch.arange(len(S)) * SR / 2048.0
        ref = S[(fr >= 500) & (fr < 6000)].mean().clamp_min(1e-12)
        # Detect hard codec/livestream brickwalls (e.g. MP3/AAC 16-20 kHz cutoff where power plunges <-68 dB re midband
        # and drops >18 dB below the band right beneath it), without clipping natural 48 kHz speech air.
        for hi in (16500, 18500, 20500):
            above = 10.0 * torch.log10(S[(fr >= hi) & (fr < min(23500, hi + 3000))].mean() / ref + 1e-12)
            below = 10.0 * torch.log10(S[(fr >= hi - 5500) & (fr < hi - 1500)].mean() / ref + 1e-12)
            if above < -65.0 and (below - above) > 22.0:
                nyq = min(nyq, float(hi - 500))
                break
    return round(max(15500.0, nyq), 1) if nyq < 22000.0 else None


def split_outputs(outputs, n: int):
    """The discriminators' outputs for a batch of 2n made of two halves -> the outputs of each half."""
    def part(sl):
        return [(score[sl], [f[sl] for f in fmap]) for score, fmap in outputs]

    return part(slice(0, n)), part(slice(n, None))


@torch.no_grad()
def build_bank(spk: dict[str, list[str]], enc, frames: int, dev: str, cache: Path, only: list[str] | None = None) -> torch.Tensor:
    """[speakers, frames, C] half-precision content frames of each speaker's own audio."""
    if cache.exists():
        loaded = torch.load(cache, map_location=dev)
        if loaded.shape[0] == len(spk):
            return loaded
    want = set(only) if only else None
    feat_dim = enc.head.out_features if hasattr(enc, "head") else 768
    bank = []
    for i, (name, files) in enumerate(spk.items()):
        if (want is not None and name not in want) or not files:
            bank.append(torch.zeros(frames, feat_dim, dtype=torch.float16, device=dev))
            continue
        feats, n = [], 0
        for path in random.Random(0).sample(files, len(files)):
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            w16 = torchaudio.functional.resample(torch.from_numpy(wav[: 60 * sr, 0]), sr, 16000).to(dev)
            if len(w16) < 8000:
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
                feats.append(enc.encode(w16[None]).float()[0])
            n += len(feats[-1])
            if n >= frames:
                break
        if not feats:
            # Short single-file voice (<0.5s): still encode whatever audio is there.
            wav, sr = sf.read(files[0], dtype="float32", always_2d=True)
            w16 = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]), sr, 16000).to(dev)
            w16 = torch.nn.functional.pad(w16, (0, max(0, 3200 - len(w16))))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
                feats.append(enc.encode(w16[None]).float()[0])
        f = torch.cat(feats)
        bank.append(f.repeat(frames // len(f) + 1, 1)[:frames].half().to(dev))
        if want is None and i % 20 == 0:
            print(f"bank {i}/{len(spk)}", flush=True)
    bank = torch.stack(bank)
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, cache)
    return bank


WARP_STEPS = (14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26)
_WARP_RESAMPLERS: dict[tuple[str, int, int], torchaudio.transforms.Resample] = {}


def _warp_resampler(device: torch.device, orig: int, new: int) -> torchaudio.transforms.Resample:
    key = (str(device), orig, new)
    res = _WARP_RESAMPLERS.get(key)
    if res is None:
        res = torchaudio.transforms.Resample(orig, new).to(device)
        _WARP_RESAMPLERS[key] = res
    return res


@torch.no_grad()
def warped_content(enc, wav48: torch.Tensor, frames: int, k: int | None = None) -> torch.Tensor:
    """Content of a copy whose spectrum (formants and pitch) is stretched by a random factor
    of 0.77-1.43, put back on the original frame clock. Training on it stops the decoder
    reading the voice's tone off speaker traces left in the content features, even across
    large (+10..+14 st) pitch shifts."""
    k = random.choice(WARP_STEPS) if k is None else k
    wav16 = _warp_resampler(wav48.device, 3 * k, 20)(wav48)  # to 16 kHz and warped by 20/k in one step
    wav16 = torch.nn.functional.pad(wav16, (0, (-wav16.shape[-1]) % 320))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav48.is_cuda):
        c = enc.encode(wav16).float()
    return torch.nn.functional.interpolate(c.transpose(1, 2), size=frames, mode="linear", align_corners=False).transpose(1, 2)


@torch.no_grad()
def training_tone(wav48: torch.Tensor, spread: float = 0.0, warp_k: int | None = None) -> torch.Tensor:
    """Per-frame tone [B, T, 2] computed from uncoloured `wav48` (and spectrally warped by `warp_k` whenever
    `warped_content` runs), so `self.tone` never acts as a shortcut around `colour()` or `perturb`."""
    frames = wav48.shape[-1] // FRAME
    if warp_k is None:
        tn = tone_of(wav48)
    else:
        warped48 = _warp_resampler(wav48.device, warp_k, 20)(wav48)
        warped48 = torch.nn.functional.pad(warped48, (0, max(0, 2 * FRAME - warped48.shape[-1])))
        tn_w = tone_of(warped48)
        tn = torch.nn.functional.interpolate(tn_w.transpose(1, 2), size=frames, mode="linear", align_corners=False).transpose(1, 2)
    if spread > 0:
        tn = tn + spread * torch.randn(wav48.shape[0], 1, 2, device=wav48.device)
    return tn


def looked_up(content: torch.Tensor, bank: torch.Tensor, s: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Blend each content frame with its nearest frames from the same speaker's bank,
    at a random strength per item, so the decoder trains on what conversion feeds it."""
    b = bank[s].float()  # [B, N, C]
    sim = torch.nn.functional.normalize(content, dim=-1) @ torch.nn.functional.normalize(b, dim=-1).transpose(1, 2)
    sim = sim.masked_fill(sim > 0.995, -1.0)  # not the very same frame, when the crop is in the bank
    top, idx = sim.topk(k, dim=-1)  # [B, T, k]
    w = torch.softmax(top / 0.05, dim=-1)
    near = (torch.gather(b[:, None].expand(-1, content.shape[1], -1, -1), 2, idx[..., None].expand(-1, -1, -1, b.shape[-1])) * w[..., None]).sum(2)
    r = torch.rand(len(s), 1, 1, device=content.device)
    r = torch.where(r < 0.25, torch.zeros_like(r), 0.4 + 0.6 * (r - 0.25) / 0.75)  # a quarter untouched, the rest 0.4-1.0
    return (1 - r) * content + r * near


@torch.no_grad()
def write_samples(path: Path, out: Path, pairs: list[tuple[str, str, int]], device: str) -> None:
    vc = CausalVC(str(path), device)
    out.mkdir(parents=True, exist_ok=True)
    for wav_path, target, semitones in pairs:
        wav, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        x = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]), sr, SR)
        y = vc.convert(x, target, semitones)
        sf.write(str(out / f"{Path(wav_path).stem}__{target}.wav"), y.cpu().numpy(), SR, subtype="PCM_16")


def _quiet_worker(_: int) -> None:
    torch.set_num_threads(1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C is handled by the main process, which saves first


def main(on_save=None, *, stop_file=None, default_base_reg=0.0) -> None:
    """on_save(step, path) is called after each checkpoint of a voice build with that checkpoint's own decoder file."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--vctk", default="data/raw/vctk")
    ap.add_argument("--ears", default="data/raw/ears")
    ap.add_argument("--manifest", default="data/eval/manifest.json")
    ap.add_argument("--out", default="runs/models")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seconds", type=float, default=2.0, help="crop fed to the encoder")
    ap.add_argument("--segment", type=int, default=48, help="100 Hz frames reconstructed per crop")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--channels", type=int, default=384)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--rates", type=int, nargs="+", default=[10, 6, 4, 2], help="upsampling factors multiplying to 480 (e.g. 20 4 3 2)")
    ap.add_argument("--antialias", action="store_true", help="causal low-pass anti-imaging on upsamplers and 2x oversampled activations")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", help="another run's last.pt to start from (fine-tuning); the step count restarts")
    ap.add_argument("--disc", type=int, default=1, help="1: magnitude spectrogram discriminators; 2: banded complex ones; 3: +longer 48 kHz periods")
    ap.add_argument("--d-warmup", type=int, default=0, help="steps that train only the discriminators (new ones on a trained generator)")
    ap.add_argument("--bands", type=int, default=0, help="4: add band-summed harmonics up to Nyquist to the excitation")
    ap.add_argument("--glide", action="store_true", help="excitation pitch slides between frames instead of stepping")
    ap.add_argument("--glide-next", action="store_true", help="excitation pitch slides from each frame's value to the next one's (correctly timed)")
    ap.add_argument("--lookahead", type=int, default=0, help="100 Hz frames of future input the decoder sees")
    ap.add_argument("--mel-cond", action="store_true", help="diagnostic: also feed the target's own mel (copy-synthesis ceiling)")
    ap.add_argument("--index-bank", type=int, default=0, help="content frames kept per speaker; >0 trains on looked-up content as conversion does")
    ap.add_argument("--stft-loss", type=float, default=0.0, help="weight of a linear-frequency log-STFT loss (resolves high harmonics)")
    ap.add_argument("--exc", default="full", choices=["full", "bandlimit", "fundamental", "multirate"], help="which excitation partials each decoder stage is fed")
    ap.add_argument("--smooth-in", default="", help="letters: c interpolate content, p blend pitch buckets, h average hidden frames")
    ap.add_argument("--only", nargs="+", help="train on these speakers only; the speaker table is kept")
    ap.add_argument("--perturb", type=float, default=0.0, help="share of steps whose content comes from a spectrally warped copy of the crop")
    ap.add_argument("--energy-loss", type=float, default=0.0, help="weight of a loss on each band's segment-averaged power (long-term tone)")
    ap.add_argument("--decorr", type=float, default=0.0, help="weight of a loss that makes unvoiced output depend on the random noise input (breaths, whispers, sibilants)")
    ap.add_argument("--noise-aug", type=float, default=0.0, help="experimental: share of examples whose input gets background noise, clicks, snaps or thumps "
                    "while the target stays clean (some are noise alone with a silent target), so non-voice sounds come out quiet")
    ap.add_argument("--colour-aug", type=float, default=0.0, help="share of examples whose input gets a random microphone colour "
                    "(tone curve, treble shelf, breath, low-pass, room) while the target stays clean, so the voice does not depend on the microphone")
    ap.add_argument("--bf16", action="store_true", help="run the decoder and discriminators in bfloat16, about 1.5x faster at the same "
                    "quality; the excitation, spectrograms and spectral losses stay in full precision. GPUs without native "
                    "bfloat16 train in full precision")
    ap.add_argument("--ema", type=float, default=0.0, help="keep a moving average of the decoder's weights with this decay per step "
                    "(the training window uses 0.999) and save that: it is smoother than the weights of any one step. "
                    "The average starts short and lengthens to 1/(1-decay) steps, so early checkpoints do not lag")
    ap.add_argument("--tone", type=float, default=0.0, help="experimental: also tell the decoder each frame's brightness and vowel openness; "
                    "the value is the spread (dB) of a random per-crop offset that hides the source's own overall tone")
    ap.add_argument("--expressive", action="store_true", help="continuous voicing confidence (0..1), 32-1400 Hz log-F0 register conditioning, and silence/click floor training")
    ap.add_argument("--silence-share", type=float, default=-1.0, help="share of crops that are digital silence or fade-to-silence (default: 0.06 with --expressive, else 0)")
    ap.add_argument("--base-reg", type=float, help="L2-SP weight anchoring shared layers toward --init during single-voice builds so laughs, whispers and extreme registers do not drift")
    ap.add_argument("--voice", nargs="+", help="NAME=folder-or-file: new voices added to the base's speaker table (use with --init and --only)")
    args = ap.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dev = args.device
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    stop_file = stop_file if stop_file is not None else out / "stop"
    if stop_before_training(stop_file):
        return
    args_path = out / "args.json"
    last = out / "last.pt"
    if args.resume and args.only and last.is_file():
        try:
            previous = json.loads(args_path.read_text())
            if args.init is None:
                args.init = previous["init"]
            if args.base_reg is None:
                args.base_reg = previous["base_reg"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            ap.error(f"cannot resume without the saved training settings in {args_path}: {error}")
    if args.base_reg is None:
        args.base_reg = default_base_reg
    if args.base_reg > 0 and args.only and not (args.init and Path(args.init).is_file()):
        ap.error("base regularization requires the original base checkpoint (--init)")
    args_partial = args_path.with_suffix(".json.part")
    args_partial.write_text(json.dumps(vars(args), indent=1))
    args_partial.replace(args_path)

    if args.voice and (args.init or args.resume):
        # Building a voice needs only its own audio: the base's speaker list comes from the checkpoint,
        # so the datasets the base was trained on do not have to be on this machine.
        source = last if args.resume and last.exists() else args.init
        spk = {name: [] for name in torch.load(source, map_location="cpu")["speakers"]}
        spk.update(voice_files(args))
    else:
        spk = speaker_files(args)
    print(f"{len(spk)} speakers, {sum(map(len, spk.values()))} files", flush=True)
    sample_pairs = []  # a few conversions to listen to while a base trains
    if not args.voice and Path(args.manifest).is_file():
        manifest = json.loads(Path(args.manifest).read_text())
        by_id = {s["id"]: s["wav"] for s in manifest["sources"]}
        for target in list(manifest["targets"])[-1:] + list(manifest["targets"])[:1]:
            for p in [p for p in manifest["pairs"] if p["target"] == target][::23][:2]:
                sample_pairs.append((by_id[p["source_id"]], target, p["semitones"]))

    enc, rmvpe = load_encoder(args.encoder, dev), Rmvpe(dev)
    k = enc.lookahead if hasattr(enc, "lookahead") else 2
    content_hz = 100 if getattr(enc, "hop", 320) == 160 else 50
    f_max = None
    if args.only:
        target_audio = [p for n in args.only for p in spk.get(n, [])]
        f_max = detect_fmax(target_audio)
        if f_max is not None:
            print(f"target audio bandwidth ~{f_max / 1000:.1f} kHz: capping spectral losses there", flush=True)
    cfg = {
        "channels": args.channels,
        "dim": args.dim,
        "bands": args.bands,
        "glide": "next" if args.glide_next else args.glide,
        "lookahead": args.lookahead,
        "mel_cond": args.mel_cond,
        "exc": args.exc,
        "smooth_in": args.smooth_in,
        **(({"rates": list(args.rates)}) if list(args.rates) != [10, 6, 4, 2] else {}),
        **(({"antialias": True}) if args.antialias else {}),
        **(({"content_hz": 100}) if content_hz == 100 else {}),
        **(({"tone": 2}) if args.tone else {}),
        **(({"expressive": True}) if args.expressive else {}),
    }
    gen, disc, mel_loss = Decoder(len(spk), **cfg).to(dev), Discriminators(args.disc).to(dev), MelLoss(f_max=f_max).to(dev)
    print(f"generator {sum(p.numel() for p in gen.parameters()) / 1e6:.1f}M params", flush=True)
    opt_g = torch.optim.AdamW(gen.parameters(), args.lr, betas=(0.8, 0.99), weight_decay=0.0 if args.base_reg else 0.01)
    opt_d = torch.optim.AdamW(disc.parameters(), args.lr, betas=(0.8, 0.99))
    step = 0
    saved_ema = None
    if args.resume and last.exists():
        ck = torch.load(last, map_location=dev)
        gen.load_state_dict(ck["generator"]), disc.load_state_dict(ck["discriminator"])
        opt_g.load_state_dict(ck["opt_g"]), opt_d.load_state_dict(ck["opt_d"])
        step, saved_ema = ck["step"], ck.get("ema")
        print(f"resumed at step {step}", flush=True)
    elif args.init:
        ck = torch.load(args.init, map_location=dev)
        own = gen.state_dict()  # excitation convs may have gained input channels: new ones start at zero
        loaded_g = 0
        for a, b in ck["generator"].items():
            if a not in own:
                continue
            if own[a].shape != b.shape:  # new inputs (speakers, future taps) start at zero; changed upsamplers/src keep fresh init
                if a.startswith(("ups.", "src.")):
                    continue
                own[a].zero_()
                part = tuple(slice(0, min(m, n)) for m, n in zip(own[a].shape, b.shape))  # the base may also have more (speakers)
                own[a][part] = b[part]
                loaded_g += 1
            else:
                own[a].copy_(b)
                loaded_g += 1
        gen.load_state_dict(own)
        print(f"generator: {loaded_g} of {len(own)} tensors loaded", flush=True)
        own = disc.state_dict()  # keep the discriminators whose shape is unchanged
        keep = {a: b for a, b in ck["discriminator"].items() if a in own and own[a].shape == b.shape}
        disc.load_state_dict(keep, strict=False)
        print(f"discriminator: {len(keep)} of {len(own)} tensors loaded", flush=True)
        base_n = len(ck["speakers"])
        shared = min(base_n, len(spk))
        assert ck["speakers"][:shared] == list(spk)[:shared], "speaker table differs from the base run"
        if base_n > len(spk):  # trailing speakers of the base that this run does not have are left out
            print(f"dropped {base_n - len(spk)} speaker(s) of the base: {ck['speakers'][len(spk):]}", flush=True)
        if base_n < len(spk):  # new voices start from the average of the voices the base knows
            w = gen.speaker.weight.data
            w[base_n:] = w[:base_n].mean(0, keepdim=True)
            print(f"added {len(spk) - base_n} new voice(s): {list(spk)[base_n:]}", flush=True)
        print(f"initialised from {args.init} (step {ck['step']})", flush=True)
    # The average of the generator's weights: the state_dict's tensors share the parameters' storage, so
    # `live` follows every optimiser step. last.pt keeps the raw weights (to resume or fine-tune from) and
    # the average under "ema"; the decoder files hold the average.
    ema = live = None
    if args.ema:
        own = gen.state_dict()
        names = [a for a, b in own.items() if b.is_floating_point()]
        live = [own[a] for a in names]
        ema = {a: (saved_ema or {}).get(a, own[a]).detach().clone().float() for a in names}
        if saved_ema is None and step:
            print("the checkpoint has no weight average: it starts from the current weights", flush=True)
    decay = lambda s: 0.999996**s  # noqa: E731  (halves every ~170k steps)
    sch_g = torch.optim.lr_scheduler.LambdaLR(opt_g, decay, last_epoch=-1)
    sch_d = torch.optim.lr_scheduler.LambdaLR(opt_d, decay, last_epoch=-1)
    for _ in range(step):
        sch_g.step(), sch_d.step()

    base_anchor: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    anchor_numel = 0
    if args.base_reg > 0 and args.init:
        ref_gen = torch.load(args.init, map_location=dev)["generator"]
        for pname, param in gen.named_parameters():
            if pname.startswith(("speaker.", "pre_cond.", "post.")) or ".cond." in pname:
                continue
            if pname in ref_gen and ref_gen[pname].shape == param.shape:
                base_anchor.append((param, ref_gen[pname].detach().float().clone()))
                anchor_numel += param.numel()
        del ref_gen
        if anchor_numel:
            print(f"L2-SP base anchor active on {len(base_anchor)} shared tensors ({anchor_numel / 1e6:.1f}M params, weight={args.base_reg})", flush=True)

    silence_share = (0.06 if args.expressive else 0.0) if args.silence_share < 0 else args.silence_share
    loader = torch.utils.data.DataLoader(
        Crops(spk, args.seconds, args.steps * args.batch, args.only, silence_share=silence_share), batch_size=args.batch,
        num_workers=args.workers, drop_last=True, persistent_workers=True, worker_init_fn=_quiet_worker,
    )
    # Reconstruct frames that end before the encoder's zero-padded lookahead region.
    t_total = int(args.seconds * 100)
    step_frames = 1 if content_hz == 100 else 2
    hi = t_total - step_frames * (k + 2)
    lo = hi - args.segment
    cmel = CausalMel().to(dev)
    spec_loss = SpecLoss(f_max=f_max).to(dev)
    energy_loss = EnergyLoss(f_max=f_max).to(dev)
    bank = None
    if args.index_bank:
        bank_cache = (
            out / f"bank_{args.index_bank}.pt"
            if args.only
            else Path("runs/cache") / f"bank_{Path(args.encoder).parent.name}_{args.index_bank}.pt"
        )
        bank = build_bank(spk, enc, args.index_bank, dev, bank_cache, only=args.only)
        print(f"bank {tuple(bank.shape)}", flush=True)
    log, shown = {}, ""

    # bfloat16 only where the GPU does it natively (NVIDIA RTX 30 series and newer, recent AMD); elsewhere it
    # would be emulated, slower than full precision.
    use_bf16 = False
    if args.bf16 and "cuda" in str(dev):
        with torch.cuda.device(dev):
            use_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
        print("training in bfloat16" if use_bf16 else "this GPU has no native bfloat16: training in full precision", flush=True)

    def amp():
        """bfloat16 for the networks when --bf16; the spectrogram front ends switch it off for themselves."""
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16)

    prog = Progress(args.steps, step)
    stop = {"now": False}  # set by Ctrl-C or a terminate signal: save a checkpoint, then leave

    def _stop(*_):
        stop["now"] = True

    signal.signal(signal.SIGINT, _stop), signal.signal(signal.SIGTERM, _stop)
    if stop_before_training(stop_file):
        return
    starting_step = step
    for wav, s in loader:
        if step >= args.steps:
            break
        if step == starting_step and stop_before_training(stop_file):
            return
        wav, s = wav.to(dev, non_blocking=True), s.to(dev)
        src = wav
        if args.noise_aug:  # what the microphone hears (src) differs from what should come out (wav)
            src, wav = add_noise(wav, args.noise_aug)
        elif silence_share > 0:
            src = add_silent_transients(src, wav)
        if args.colour_aug:
            src = colour(src, args.colour_aug)
        content, f0, voicing, loud = extract_all(enc, rmvpe, src)
        warp_k = None
        if args.perturb and random.random() < args.perturb:
            warp_k = random.choice(WARP_STEPS)
            content = warped_content(enc, src, content.shape[1], k=warp_k)
        elif bank is not None:
            content = looked_up(content, bank, s)
        tn = training_tone(wav, args.tone, warp_k=warp_k) if args.tone else None
        v_in = voicing if args.expressive else None
        with amp():
            h, g = gen.features(content, f0, loud, s, cmel(wav) if args.mel_cond else None, tn, voicing=v_in)
        th = h.shape[-1]
        exc = gen.source(f0[:, :th], voicing=None if v_in is None else v_in[:, :th])
        reg = gen.register_vec(f0[:, lo:hi], None if v_in is None else v_in[:, lo:hi]) if gen.reg_cond is not None else None
        with amp():
            fake = gen.generate(h[..., lo:hi], g, exc[..., lo * FRAME : hi * FRAME], reg=reg).float()
        real = wav[:, None, lo * FRAME : hi * FRAME]

        # Real and generated audio in one pass: the discriminators treat every example on its own, so
        # this is the same as two passes, with half the kernel launches.
        with amp():
            loss_d = d_loss(*split_outputs(disc(torch.cat([real, fake.detach()])), real.shape[0]))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()

        if step < args.d_warmup:
            step += 1
            sch_g.step(), sch_d.step()
            prog.update(step, f"warmup d {loss_d.item():.3f}")
            continue
        loss_mel = mel_loss(fake, real)
        with torch.no_grad(), amp():
            real_out = disc(real)
        # The generator's loss reaches it through the discriminators, but their own weight gradients
        # from it were only thrown away (cleared before their next step): skip computing them.
        disc.requires_grad_(False)
        with amp():
            adv, fm = g_loss(real_out, disc(fake))
        loss_g = adv + 2.0 * fm + 45.0 * loss_mel
        if args.stft_loss:
            loss_stft = spec_loss(fake, real)
            loss_g = loss_g + args.stft_loss * loss_stft
            log["stft"] = log.get("stft", 0.0) + loss_stft.item() / 100
        if args.decorr:
            # The same frames again with fresh noise in the excitation. Real breath noise is different every
            # time; without this the decoder computes one fixed noise-like pattern from the features and
            # ignores the noise it is given. Penalise the per-frame correlation of the two outputs where unvoiced
            # (or half-voiced in laughs/sighs), but never on silent target frames where the output should be 0.
            exc2 = gen.source(f0[:, :th], voicing=None if v_in is None else v_in[:, :th])[..., lo * FRAME : hi * FRAME]
            with amp():
                again = gen.generate(h[..., lo:hi], g, exc2, reg=reg).float()
            a, b = (y[:, 0].unflatten(1, (-1, FRAME)) for y in (fake, again))
            a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
            corr = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-5)
            audible = (real[:, 0].unflatten(1, (-1, FRAME)).pow(2).mean(-1).sqrt() > 3e-4).float()
            quiet = ((1.0 - v_in[:, lo:hi]).pow(2) if v_in is not None else (f0[:, lo:hi] == 0).float()) * audible
            loss_dc = (corr.pow(2) * quiet).sum() / quiet.sum().clamp_min(1.0)
            loss_g = loss_g + args.decorr * loss_dc
            log["decorr"] = log.get("decorr", 0.0) + loss_dc.item() / 100
        if silence_share > 0:
            real_f = real[:, 0].unflatten(1, (-1, FRAME))
            silent_mask = (real_f.pow(2).mean(-1).sqrt() < 1e-4).float()
            if silent_mask.any():
                fake_f = fake[:, 0].unflatten(1, (-1, FRAME))
                loss_sil = (fake_f.abs().mean(-1) * silent_mask).sum() / silent_mask.sum().clamp_min(1.0)
                loss_g = loss_g + 40.0 * loss_sil
                log["sil"] = log.get("sil", 0.0) + loss_sil.item() / 100
        if base_anchor:
            loss_reg = sum((p.float() - p0).pow(2).sum() for p, p0 in base_anchor) / anchor_numel
            loss_g = loss_g + args.base_reg * loss_reg
            log["reg"] = log.get("reg", 0.0) + loss_reg.item() / 100
        if args.energy_loss:
            loss_en = energy_loss(fake, real)
            loss_g = loss_g + args.energy_loss * loss_en
            log["energy"] = log.get("energy", 0.0) + loss_en.item() / 100
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        disc.requires_grad_(True)
        opt_g.step()
        if ema is not None:
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.values()), live, 1.0 - min(args.ema, (1 + step) / (10 + step)))
        sch_g.step(), sch_d.step()
        step += 1

        for key, v in (("mel", loss_mel), ("d", loss_d), ("adv", adv), ("fm", fm)):
            log[key] = log.get(key, 0.0) + v.item() / 100
        if step % 100 == 0:
            shown, log = " ".join(f"{a} {b:.3f}" for a, b in log.items()), {}
        prog.update(step, shown)
        if stop_file.is_file():
            stop["now"] = True
        if step % args.save_every == 0 or step == args.steps or stop["now"]:
            prog.pause()
            ck = {
                "generator": gen.state_dict(), "discriminator": disc.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step,
                "cfg": cfg, "speakers": list(spk), "encoder": args.encoder,  # a hint only: causalvc.voice.encoder_path falls back to the shared encoder
            }
            if ema is not None:
                ck["ema"] = ema
            last_partial = last.with_suffix(".pt.part")
            torch.save(ck, last_partial)
            last_partial.replace(last)
            slim = {a: b for a, b in ck.items() if a not in ("discriminator", "opt_g", "opt_d", "ema")}
            if ema is not None:
                slim["generator"] = {**ck["generator"], **ema}
            torch.save(slim, out / "decoder.pt")
            if args.voice:  # every checkpoint of a voice is kept, so different amounts of training can be compared
                torch.save(slim, out / f"decoder-{step}s.pt")
                if on_save:
                    on_save(step, out / f"decoder-{step}s.pt")
            elif step % (4 * args.save_every) == 0:
                torch.save(slim, out / f"decoder_{step // 1000}k.pt")
            gen.eval()
            if not args.mel_cond and sample_pairs and not stop["now"]:
                write_samples(out / "decoder.pt", out / "samples" / f"step_{step // 1000:04d}k", sample_pairs, dev)
            gen.train()
            prog.resume()
            prog.note(f"saved {out / 'decoder.pt'} at step {step}")
            if stop["now"]:
                stop_file.unlink(missing_ok=True)
                prog.note(f"stopped at step {step} of {args.steps}; run the same command again to resume")
                break


if __name__ == "__main__":
    main()
