"""Distil ContentVec into the causal streaming encoder.

    uv run python -m causalvc.encoder.train --name enc-la2 --lookahead 2 --device cuda:0

The teacher sees the clean crop; the student sees the same crop (with a random
gain) and must reproduce the teacher's features causally.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from causalvc.augment import degrade
from causalvc.encoder.model import HOP, SR, StreamingEncoder

ROOT = Path(__file__).resolve().parents[2]
TEACHER_CANDIDATES = (
    ROOT / "weights" / "contentvec",
    Path("/mnt/ssd/Applio-3.7.1/rvc/models/embedders/contentvec"),
)
TEACHER = str(next((p for p in TEACHER_CANDIDATES if p.is_dir()), TEACHER_CANDIDATES[-1]))
EXPRESSIVE_TAGS = (
    "nonverbal_", "interjection_", "emo_amazement", "emo_extasy", "emo_fear",
    "emo_pain", "emo_anger", "emo_sadness", "emo_relief", "highpitch", "lowpitch", "whisper",
)


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


class Crops(torch.utils.data.Dataset):
    """Random fixed-length crops resampled to 16 kHz across mixed-rate datasets (LibriTTS, VCTK, EARS)."""

    def __init__(self, files: list[str], seconds: float, pitch_aug: bool = False):
        self.files, self.seconds, self.n, self.pitch_aug = files, seconds, int(seconds * SR), pitch_aug
        self.ears = [p for p in files if "/ears/" in p.replace("\\", "/")]
        self.vctk = [p for p in files if "/vctk/" in p.replace("\\", "/")]
        self.other = [p for p in files if "/ears/" not in p.replace("\\", "/") and "/vctk/" not in p.replace("\\", "/")]
        self.expressive = [p for p in files if any(t in Path(p).stem for t in EXPRESSIVE_TAGS)]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        r = random.random()
        pool = None
        if self.expressive and r < 0.22:
            pool = self.expressive
        elif self.ears and self.other and r < 0.50:
            pool = self.ears
        elif self.vctk and self.other and r < 0.70:
            pool = self.vctk
        alpha = 1.0
        if self.pitch_aug and random.random() < 0.25:
            st = random.choice([-7, -5, -3, 2, 4, 6, 8, 10, 12])
            alpha = 2.0 ** (st / 12.0)
        for _ in range(20):
            path = random.choice(pool) if pool else self.files[i]
            info = sf.info(path)
            eff_sr = max(6400, int(round(info.samplerate * alpha / 100.0)) * 100)
            want = int(self.seconds * eff_sr)
            if info.frames >= want // 2:
                start = random.randint(0, max(0, info.frames - want))
                wav, _ = sf.read(path, start=start, frames=want, dtype="float32", always_2d=True)
                x = torch.from_numpy(wav[:, 0])
                if eff_sr != SR:
                    x = torchaudio.functional.resample(x, eff_sr, SR)
                if len(x) < self.n:
                    x = F.pad(x, (0, self.n - len(x)))
                return x[: self.n]
            i = random.randrange(len(self.files))
        raise RuntimeError("no file long enough")


def load_teacher(device: str, teacher_path: str | None = None):
    from transformers import HubertModel

    src = teacher_path or str(next((p for p in TEACHER_CANDIDATES if p.is_dir()), TEACHER))
    return HubertModel.from_pretrained(src).to(device).eval().requires_grad_(False)


@torch.no_grad()
def teacher_features(teacher, wav16: torch.Tensor, hop: int = 320) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav16.is_cuda):
        if hop == 320:
            return teacher(wav16).last_hidden_state.float()
        # 100 Hz (10 ms) targets in ContentVec's exact space: pad by one 10 ms hop on each side so frame t is
        # centred on [160*t - 160 .. 160*t + 240] (~5 ms), matching StreamingEncoder's 10 ms frame clock.
        w_pad = F.pad(wav16, (160, 160))
        even = teacher(w_pad).last_hidden_state
        odd = teacher(w_pad[:, 160:]).last_hidden_state
        m = min(even.shape[1], odd.shape[1])
        return torch.stack([even[:, :m], odd[:, :m]], dim=2).flatten(1, 2)[:, : wav16.shape[1] // 160].float()


def distill_loss(pred: torch.Tensor, target: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """pred step t is trained on target frame t - k."""
    n = min(pred.shape[1] - k, target.shape[1])
    p, y = pred[:, k : k + n].float(), target[:, :n].float()
    cos = F.cosine_similarity(p, y, dim=-1).mean()
    return F.l1_loss(p, y) + (1 - cos), cos


@torch.no_grad()
def evaluate(model, teacher, files: list[str], device: str, degraded: bool = False) -> float:
    model.eval()
    cos = []
    hop = getattr(model, "hop", 320)
    torch.manual_seed(1)  # the same degradations at every evaluation
    for path in files:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
        x = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]).to(device)[None], sr, SR)
        if x.shape[-1] < 1600:
            continue
        y = teacher_features(teacher, x, hop=hop)
        p = model.encode(degrade(x) if degraded else x)
        n = min(p.shape[1], y.shape[1])
        cos.append(F.cosine_similarity(p[:, :n], y[:, :n], dim=-1).mean().item())
    model.train()
    return float(np.mean(cos))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", default=["data/raw/libritts_r/LibriTTS_R/train-clean-100"])
    ap.add_argument("--valid", default="data/raw/libritts_r/LibriTTS_R/test-clean")
    ap.add_argument("--out", default="runs/models")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hop", type=int, default=320, choices=[160, 320], help="320: 50 Hz (20 ms); 160: 100 Hz (10 ms)")
    ap.add_argument("--window", type=int, default=150, help="frames of left context attention may use")
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--augment", action="store_true", help="degrade the student's input like a real microphone")
    ap.add_argument("--pitch-aug", action="store_true", help="pitch-shift 25% of crops (-7 to +12 semitones) from native rate")
    ap.add_argument("--teacher", help="path to ContentVec teacher folder")
    ap.add_argument("--init", help="encoder.pt to start from (weights only)")
    ap.add_argument("--resume", action="store_true", help="continue from the last saved weights (optimizer state is rebuilt)")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    files = sorted(
        str(p)
        for d in args.train
        if Path(d).exists()
        for p in Path(d).rglob("*")
        if p.suffix == ".wav" or p.name.endswith("_mic1.flac")
    )
    valid = sorted(str(p) for p in Path(args.valid).rglob("*.wav"))
    random.Random(0).shuffle(valid)
    valid = valid[:200]
    loader = torch.utils.data.DataLoader(
        Crops(files, args.seconds, pitch_aug=args.pitch_aug), batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, persistent_workers=True, worker_init_fn=_worker_init,
    )
    dev = args.device
    teacher = load_teacher(dev, args.teacher)
    cfg = {
        "dim": args.dim,
        "layers": args.layers,
        "lookahead": args.lookahead,
        **({"hop": args.hop} if args.hop != 320 else {}),
        **({"window": args.window} if args.window != 150 else {}),
    }
    model = StreamingEncoder(**cfg).to(dev)
    print(f"{len(files)} files, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params (hop={args.hop})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    warm = min(2000, max(200, args.steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, 0.5 * (1 + np.cos(np.pi * min(s / args.steps, 1.0))))
    )
    step, t0, log = 0, time.time(), []
    if args.resume and (out / "encoder.pt").exists():
        ck = torch.load(out / "encoder.pt", map_location=dev)
        model.load_state_dict(ck["model"])
        step, log = ck["step"], json.loads((out / "log.json").read_text())
        for _ in range(step):
            sched.step()
        print(f"resumed at step {step}", flush=True)
    elif args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
        print(f"initialised from {args.init}", flush=True)
    start = step
    while step < args.steps:
        for wav in loader:
            wav16 = wav.to(dev, non_blocking=True)
            wav16 = wav16[:, : HOP * (wav16.shape[1] // HOP)]
            target = teacher_features(teacher, wav16, hop=args.hop)
            gain = torch.empty(wav16.shape[0], 1, device=dev).uniform_(-12, 3)
            inp16 = wav16
            if args.augment:
                deg = degrade(wav16)
                keep_clean = (torch.rand(wav16.shape[0], 1, device=dev) < 0.35).float()
                inp16 = keep_clean * wav16 + (1.0 - keep_clean) * deg
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(inp16 * 10 ** (gain / 20))
            loss, cos = distill_loss(pred, target, args.lookahead)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 250 == 0:
                print(f"step {step} loss {loss.item():.4f} cos {cos.item():.4f} "
                      f"{(step - start) / (time.time() - t0):.1f} it/s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                v = evaluate(model, teacher, valid, dev)
                vd = evaluate(model, teacher, valid, dev, degraded=True)
                log.append({"step": step, "valid_cos": v, "valid_cos_degraded": vd})
                print(f"== step {step} valid cos {v:.4f} degraded {vd:.4f}", flush=True)
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, out / "encoder.pt")
                (out / "log.json").write_text(json.dumps(log, indent=1))
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
