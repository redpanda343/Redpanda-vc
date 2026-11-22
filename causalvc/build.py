"""Build a voice from a folder (or one file) of the target's speech.

    PYTHONPATH=$PWD .venv/bin/python -m causalvc.build --name Kira --audio /path/to/audio

Fine-tunes a base decoder on that audio alone and writes
a voice at every checkpoint, as one self-contained file that can be copied to
another machine: runs/models/voice-NAME/voices/NAME-STEPs.cvc (also installed in
--app-models/NAME/NAME-STEPs.cvc when given), so NAME-1000s, NAME-2000s and so on can be
compared. The audio needs no preparation: any sample rate, any length, pauses left in.

Ctrl-C (or a terminate signal) saves a checkpoint before stopping, and running
the same command again resumes from it. --restart starts over instead.
--base takes a run folder, or the name of a folder under custom-base/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if sys.platform != "win32":  # PyTorch supports this only on Linux
    # Let the CUDA allocator grow its blocks instead of keeping fixed ones: a build at batch 16 on a
    # 16 GB card otherwise loses a few hundred MB to fragmentation and runs out with the desktop's share
    # taken. Set before torch is imported; a value the user set themselves is kept.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASES = ("runs/models/dec-s8-100hz-v4", "runs/models/dec-s7-100hz", "runs/models/dec-s6-100hz", "runs/models/dec-s6")
CUSTOM = ROOT / "custom-base"


def bases() -> dict[str, Path]:
    """Bases a voice can start from: the default, then every folder in custom-base/ holding last.pt and args.json."""
    from causalvc.weights import base_folder

    local = next((ROOT / b for b in DEFAULT_BASES if (ROOT / b / "last.pt").is_file()), None)
    found = {"default": local if local is not None else base_folder()}
    if CUSTOM.is_dir():
        for d in sorted(CUSTOM.iterdir()):
            if (d / "last.pt").is_file() and (d / "args.json").is_file():
                found[d.name] = d
    return found


def resolve_base(value: str) -> Path:
    known = bases()
    path = known.get(value) or Path(value)
    if value == "default" and not (path / "last.pt").is_file():
        from causalvc.weights import ensure_base

        path = ensure_base()
    if not (path / "last.pt").is_file() or not (path / "args.json").is_file():
        raise SystemExit(f"base '{value}' not found: expected last.pt and args.json in {path} (known bases: {', '.join(known)})")
    return path


def checkpoint_step(run_dir: Path) -> int | None:
    """Step of the run's last checkpoint, or None when there is none to resume."""
    import torch

    last = run_dir / "last.pt"
    if not last.is_file():
        return None
    try:
        return int(torch.load(last, map_location="cpu", weights_only=False)["step"])
    except Exception:
        return None


def make_index(run_dir: Path, encoder: str, audio: str, device: str) -> None:
    """The voice's index (causalvc.index) in run_dir/index.pt, made once; every checkpoint's .cvc
    carries it. A voice without one still works (its Index slider just does nothing)."""
    target = run_dir / "index.pt"
    if target.is_file():
        return
    try:
        import torch

        from causalvc.index import build
        from causalvc.infer import load_encoder

        print("making the voice's index", flush=True)
        enc = load_encoder(str(ROOT / encoder) if not Path(encoder).is_absolute() else encoder, device)
        data = build(enc, audio, device)
        del enc
        torch.cuda.empty_cache()
        run_dir.mkdir(parents=True, exist_ok=True)
        target.with_suffix(".part").write_bytes(data)
        target.with_suffix(".part").replace(target)
    except Exception as error:  # noqa: BLE001 - the voice itself matters more than its index
        print(f"no index for this voice ({error})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--audio", required=True, help="folder or file of the target voice")
    ap.add_argument("--base", default="default", help="'default', a folder name under custom-base/, or a path to a base run")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--save-every", type=int, default=0, help="steps between checkpoints (default: a quarter of --steps)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--restart", action="store_true", help="ignore an existing checkpoint for this name and start over")
    ap.add_argument("--perturb", type=float, default=0.0, help="experimental: see causalvc.decoder.train")
    ap.add_argument("--exc", help="override base excitation mode (default: bandlimit when base is full, else base's)")
    ap.add_argument("--smooth-in", default=None, help="override base input smoothing (default: cp when base has none)")
    ap.add_argument("--index-bank", type=int, default=0, help="content frames kept for index-aware training (default 0 on voice builds)")
    ap.add_argument("--energy-loss", type=float, default=0.0, help="experimental: see causalvc.decoder.train")
    ap.add_argument("--decorr", type=float, default=0.0, help="experimental: see causalvc.decoder.train")
    ap.add_argument("--noise-aug", type=float, default=0.0, help="experimental: see causalvc.decoder.train")
    ap.add_argument("--colour-aug", type=float, default=0.0, help="share of inputs given a random microphone colour (the training window uses 0.5); see causalvc.decoder.train")
    ap.add_argument("--base-reg", type=float, default=None, help="L2-SP anchor weight toward base shared layers (default: 25 on expressive bases, else 0)")
    ap.add_argument("--ema", type=float, default=0.0, help="save a moving average of the decoder's weights (the training window uses 0.999); see causalvc.decoder.train")
    ap.add_argument("--bf16", action="store_true", help="train in bfloat16, about 1.5x faster at the same quality (the training window always passes it); see causalvc.decoder.train")
    ap.add_argument("--run", help="run folder name under runs/models (default voice-NAME)")
    ap.add_argument("--app-models", help="a realtime app's models folder to register the voice in (the bundled app's is realtime/models)")
    a = ap.parse_args()

    if not Path(a.audio).exists():
        raise SystemExit(f"audio not found: {a.audio}")
    from causalvc.progress import stop_before_training

    run = a.run or f"voice-{a.name}"
    run_dir = ROOT / "runs/models" / run
    stop_file = Path(os.environ.get("CAUSALVC_STOP_FILE", run_dir / "stop"))
    if stop_before_training(stop_file):
        return
    base_dir = resolve_base(a.base)
    if stop_before_training(stop_file):
        return
    base = json.loads((base_dir / "args.json").read_text())

    def beside(name: str, fallback: str) -> str:
        """A file shipped inside the base folder, if it is there; otherwise the path the base was trained with."""
        own = base_dir / name
        if not own.is_file():
            return fallback
        own = own.resolve()
        return str(own.relative_to(ROOT)) if own.is_relative_to(ROOT) else str(own)

    encoder = beside("encoder.pt", base["encoder"])
    pitch_own = beside("pitch.pt", "")

    def register(step: int, decoder: Path) -> None:
        """Turn one checkpoint into a voice of its own, NAME-STEPs.cvc: one file that needs nothing else from this machine."""
        from causalvc.voice import export_cvc, install, own_parts, voice_folder

        label = f"{a.name}-{step}s"
        index = run_dir / "index.pt"
        parts = own_parts({"encoder": encoder, **({"pitch": pitch_own} if pitch_own else {})}, [base_dir, ROOT], decoder)
        made = export_cvc(decoder, a.name, run_dir / "voices" / f"{label}.cvc", label,
                          index.read_bytes() if index.is_file() else None, parts=parts)
        decoder.unlink(missing_ok=True)  # the voice's own copy (this voice only) replaces the checkpoint's
        if a.app_models:
            install(made, voice_folder(Path(a.app_models), a.name) / made.name)  # the same file, in the voice changer's folder for this voice
        print(f"voice {label} ready: {made}" + (f" (installed in {voice_folder(Path(a.app_models), a.name)})" if a.app_models else ""), flush=True)

    done = None if a.restart else checkpoint_step(run_dir)
    if done is not None and done >= a.steps:
        print(f"{run} is already trained to step {done}; raise --steps to continue or pass --restart", flush=True)
        return
    if done is not None:
        print(f"resuming {run} from step {done}", flush=True)
    index_bank = max(0, a.index_bank)
    exc = a.exc or ("bandlimit" if base.get("exc", "full") == "full" else base["exc"])
    smooth_in = (base.get("smooth_in") or "cp") if a.smooth_in is None else a.smooth_in
    argv = ["--name", run, "--encoder", encoder,
            "--disc", str(base["disc"]), "--bands", str(base["bands"]), "--lookahead", str(base["lookahead"]),
            "--channels", str(base["channels"]), "--dim", str(base["dim"]), "--exc", exc,
            "--batch", str(a.batch), "--lr", "1e-4", "--stft-loss", "20", "--steps", str(a.steps),
            "--save-every", str(a.save_every or max(1, a.steps // 4)), "--device", a.device,
            "--voice", f"{a.name}={a.audio}", "--only", a.name, "--index-bank", str(index_bank),
            "--perturb", str(a.perturb), "--energy-loss", str(a.energy_loss), "--decorr", str(a.decorr), "--noise-aug", str(a.noise_aug)]
    argv += ["--resume"] if done is not None else ["--init", str((base_dir / "last.pt").resolve())]
    argv += ["--rates", *(str(r) for r in base["rates"])] if base.get("rates") else []
    argv += ["--antialias"] if base.get("antialias") else []
    argv += ["--glide-next"] if base.get("glide_next") else ["--glide"] if base.get("glide") else []
    argv += ["--smooth-in", smooth_in] if smooth_in else []
    argv += ["--tone", str(base["tone"])] if base.get("tone") else []
    argv += ["--expressive"] if base.get("expressive") else []
    argv += ["--base-reg", str(max(0.0, a.base_reg))] if a.base_reg is not None else []
    argv += ["--bf16"] if a.bf16 else []
    argv += ["--colour-aug", str(a.colour_aug)] if a.colour_aug else []
    argv += ["--ema", str(a.ema)] if a.ema else []
    if stop_before_training(stop_file):
        return
    make_index(run_dir, encoder, a.audio, a.device)
    if stop_before_training(stop_file):
        return
    from causalvc.decoder import train

    sys.argv = [sys.argv[0], *argv]
    train.main(on_save=register, stop_file=stop_file, default_base_reg=25.0 if base.get("expressive") else 0.0)


if __name__ == "__main__":
    main()
