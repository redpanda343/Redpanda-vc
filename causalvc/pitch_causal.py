"""Causal pitch tracker, distilled from RMVPE.

Same output convention as `causalvc.pitch.Rmvpe` (100 frames per second, frame j
centred on sample 160 * j, 0 Hz where unvoiced), but each frame is final after
a fixed delay: half the 64 ms analysis window (32 ms) plus `lookahead` frames
of 10 ms. With the default of 2 that is 52 ms, inside the content encoder's
own delay, so it adds nothing to the total.

    uv run python -m causalvc.pitch_causal --name pitch-la2 --device cuda:1
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
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

HOP = 160


class Block(nn.Module):
    def __init__(self, dim: int, k: int, dilation: int):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.dw = nn.Conv1d(dim, dim, k, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.fc1, self.fc2 = nn.Linear(dim, 3 * dim), nn.Linear(3 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        y = self.dw(F.pad(x, (self.pad, 0))).transpose(1, 2)
        return x + self.fc2(F.gelu(self.fc1(self.norm(y)))).transpose(1, 2)


class CausalPitch(nn.Module):
    def __init__(self, dim: int = 256, blocks: int = 8, lookahead: int = 2, n_mels: int = 128, threshold: float = 0.03):
        super().__init__()
        self.lookahead, self.thr = lookahead, threshold
        self.mel = torchaudio.transforms.MelSpectrogram(
            16000, n_fft=1024, hop_length=HOP, n_mels=n_mels, f_min=30, f_max=8000,
            center=True, pad_mode="constant", power=1.0,
        )
        self.inp = nn.Conv1d(n_mels, dim, 3)
        self.blocks = nn.ModuleList(Block(dim, 7, 2 ** (i % 3)) for i in range(blocks))
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.head = nn.Linear(2 * dim, 360)
        cents = 20 * torch.arange(360, dtype=torch.float32) + 1997.3794084376191
        self.register_buffer("cents", F.pad(cents, (4, 4)), persistent=False)

    def forward(self, wav16: torch.Tensor) -> torch.Tensor:
        """[B, N] -> salience logits [B, N // 160 + 1, 360]; step t is frame t - lookahead."""
        with torch.autocast(wav16.device.type, enabled=False):
            m = torch.log(self.mel(wav16.float()).clamp_min(1e-5))
        x = self.inp(F.pad(m, (2, 0)))
        for b in self.blocks:
            x = b(x)
        x = x.transpose(1, 2)
        return self.head(torch.cat([x, self.gru(x)[0]], dim=-1))

    def logits(self, wav16: torch.Tensor) -> torch.Tensor:
        """Whole-signal logits aligned with RMVPE's frames (what streaming would produce)."""
        k = self.lookahead
        return self.forward(F.pad(wav16, (0, k * HOP)))[:, k:]

    def decode_with_voicing(self, sal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from causalvc.pitch import voicing_from_peak

        peak, centre = sal.max(dim=-1)
        idx = centre[..., None] + torch.arange(9, device=sal.device)
        local = F.pad(sal, (4, 4)).gather(-1, idx)
        cents = (local * self.cents[idx]).sum(-1) / local.sum(-1).clamp_min(1e-9)
        f0 = 10.0 * 2.0 ** (cents / 1200.0)
        return torch.where(peak > self.thr, f0, torch.zeros_like(f0)), voicing_from_peak(peak, self.thr)

    def decode(self, sal: torch.Tensor) -> torch.Tensor:
        return self.decode_with_voicing(sal)[0]

    @torch.no_grad()
    def pitch_and_voicing(self, wav16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decode_with_voicing(torch.sigmoid(self.logits(wav16).float()))

    @torch.no_grad()
    def __call__(self, wav16: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.forward(wav16)
        return self.decode(torch.sigmoid(self.logits(wav16).float()))


def load_pitch(path: str, device: str) -> CausalPitch:
    ck = torch.load(path, map_location="cpu")
    m = CausalPitch(**ck["cfg"])
    m.load_state_dict(ck["model"])
    return m.to(device).eval().requires_grad_(False)


def agreement(f_s: torch.Tensor, f_t: torch.Tensor) -> dict[str, float]:
    """Student against teacher: pitch accuracy where both voiced, and voicing errors."""
    vs, vt = f_s > 0, f_t > 0
    both = vs & vt
    cents = 1200 * torch.log2(f_s[both] / f_t[both]).abs()
    return {
        "rpa50": (cents < 50).float().mean().item(), "rpa20": (cents < 20).float().mean().item(),
        "octave": (cents > 550).float().mean().item(), "vuv_err": (vs != vt).float().mean().item(),
    }


def _worker_init(_: int) -> None:
    torch.set_num_threads(1)


class Crops(torch.utils.data.Dataset):
    def __init__(self, files: list[str], seconds: float, pitch_aug: bool = False):
        self.files, self.seconds, self.pitch_aug = files, seconds, pitch_aug

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> torch.Tensor:
        n = int(self.seconds * 16000)
        path = self.files[i]
        info = sf.info(path)
        # Pitch-shift upward/downward by resampling directly from native rate (24-48 kHz) so sub-50 Hz vocal fry
        # (down to ~32 Hz) and high pitches (up to ~1400 Hz) are both densely covered without Nyquist aliasing.
        alpha = 1.0
        if self.pitch_aug and random.random() < 0.50:
            st = random.choice([-9, -7, -5, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
            alpha = 2.0 ** (st / 12.0)
        eff_sr = max(6400, int(round(info.samplerate * alpha / 100.0)) * 100)
        want = int(self.seconds * eff_sr)
        start = random.randint(0, max(0, info.frames - want))
        wav, _ = sf.read(path, start=start, frames=want, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav[:, 0])
        if eff_sr != 16000:
            wav = torchaudio.functional.resample(wav, eff_sr, 16000)
        return F.pad(wav, (0, max(0, n - len(wav))))[:n]


def main() -> None:
    from causalvc.pitch import Rmvpe

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--train", nargs="+", default=[
        "data/raw/libritts_r/LibriTTS_R/train-clean-100", "data/raw/libritts_r/LibriTTS_R/train-clean-360",
        "data/raw/vctk/wav48_silence_trimmed", "data/raw/ears",
    ])
    ap.add_argument("--valid", default="data/raw/libritts_r/LibriTTS_R/test-clean")
    ap.add_argument("--out", default="runs/models")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--lookahead", type=int, default=2)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=2500)
    ap.add_argument("--init", help="pitch.pt checkpoint to fine-tune from")
    ap.add_argument("--pitch-aug", action="store_true", help="augment crops up to +14 semitones so >500 Hz pitches track cleanly")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)
    out = Path(args.out) / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    files = sorted(str(p) for d in args.train if Path(d).exists() for p in Path(d).rglob("*") if p.suffix == ".wav" or p.name.endswith("_mic1.flac"))
    valid = sorted(str(p) for p in Path(args.valid).rglob("*.wav"))
    random.Random(0).shuffle(valid)
    valid = valid[:200]
    dev = args.device
    teacher = Rmvpe(dev)
    cfg = {"dim": args.dim, "blocks": args.blocks, "lookahead": args.lookahead}
    model = CausalPitch(**cfg).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev)["model"])
        print(f"initialised from {args.init}", flush=True)
    print(f"{len(files)} files, {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 1000, 0.5 * (1 + np.cos(np.pi * min(s / args.steps, 1.0))))
    )
    loader = torch.utils.data.DataLoader(
        Crops(files, args.seconds, pitch_aug=args.pitch_aug), batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, persistent_workers=True, worker_init_fn=_worker_init,
    )
    k, step, t0, log = args.lookahead, 0, time.time(), []
    while step < args.steps:
        for wav in loader:
            wav = wav.to(dev, non_blocking=True)
            wav = wav * 10 ** (torch.empty(wav.shape[0], 1, device=dev).uniform_(-20, 3) / 20)
            target = teacher.salience(wav)
            pred = model.forward(wav)
            n = pred.shape[1] - k
            # Skip the first frames (no left context yet) and the last ones (teacher sees padding).
            p, y = pred[:, k + 20 : k + n - 8].float(), target[:, 20 : n - 8]
            loss = F.binary_cross_entropy_with_logits(p, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 250 == 0:
                print(f"step {step} loss {loss.item():.5f} {step / (time.time() - t0):.1f} it/s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                model.eval()
                fs, ft, fs_hi, ft_hi = [], [], [], []
                with torch.no_grad():
                    for idx_v, path in enumerate(valid):
                        w, sr = sf.read(path, dtype="float32", always_2d=True)
                        wt = torch.from_numpy(w[:, 0]).to(dev)[None]
                        x = torchaudio.functional.resample(wt, sr, 16000)
                        a, b = model(x)[0], teacher(x)[0]
                        m = min(len(a), len(b))
                        fs.append(a[:m]), ft.append(b[:m])
                        if idx_v < 50:  # +10 st high-pitch tracking check (~300-700 Hz)
                            sr_hi = int(round(sr * (2.0 ** (10.0 / 12.0)) / 100.0)) * 100
                            xh = torchaudio.functional.resample(wt, sr_hi, 16000)
                            if xh.shape[-1] > 3200:
                                ah, bh = model(xh)[0], teacher(xh)[0]
                                mh = min(len(ah), len(bh))
                                fs_hi.append(ah[:mh]), ft_hi.append(bh[:mh])
                model.train()
                hi_ag = agreement(torch.cat(fs_hi), torch.cat(ft_hi)) if fs_hi else {}
                r = {"step": step, **agreement(torch.cat(fs), torch.cat(ft)), **({f"hi_{k}": v for k, v in hi_ag.items() if k in ("rpa50", "vuv_err")})}
                log.append(r)
                print("== " + " ".join(f"{a} {b:.4f}" if a != "step" else f"step {b}" for a, b in r.items()), flush=True)
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, out / "pitch.pt")
                (out / "log.json").write_text(json.dumps(log, indent=1))
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
