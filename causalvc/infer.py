"""Whole-file conversion with the causal encoder and decoder.

Both networks are causal, so this is what a streaming runtime would output; the
pitch track is the exception while it still comes from RMVPE.
"""

from __future__ import annotations

import torch
import torchaudio

from causalvc.decoder.model import FRAME, SR, Decoder, loudness, pitch_sync_loudness, tone
from causalvc.encoder.model import StreamingEncoder
from causalvc.pitch import Rmvpe


def load_encoder(path: str, device: str) -> StreamingEncoder:
    ckpt = torch.load(path, map_location="cpu")
    enc = StreamingEncoder(**ckpt["cfg"])
    enc.load_state_dict(ckpt["model"])
    return enc.to(device).eval().requires_grad_(False)


_RES16: dict[str, torchaudio.transforms.Resample] = {}


@torch.no_grad()
def extract_all(enc: StreamingEncoder, rmvpe: Rmvpe, wav48: torch.Tensor):
    """[B, N] at 48 kHz -> content [B, Tc, C], f0 [B, T], voicing [B, T], loudness [B, T]."""
    dev_key = str(wav48.device)
    res = _RES16.get(dev_key)
    if res is None:
        res = torchaudio.transforms.Resample(SR, 16000).to(wav48.device)
        _RES16[dev_key] = res
    wav16 = res(wav48)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=wav48.is_cuda and isinstance(enc, StreamingEncoder)):
        content = enc.encode(wav16).float()
    if hasattr(rmvpe, "pitch_and_voicing"):
        f0, voicing = rmvpe.pitch_and_voicing(wav16)
    else:
        f0 = rmvpe(wav16)
        voicing = (f0 > 0).float()
    return content, f0, voicing, loudness(wav48)


@torch.no_grad()
def extract(enc: StreamingEncoder, rmvpe: Rmvpe, wav48: torch.Tensor):
    """[B, N] at 48 kHz -> content [B, T/2, C], f0 [B, T], loudness [B, T]."""
    content, f0, _, loud = extract_all(enc, rmvpe, wav48)
    return content, f0, loud


class CausalVC:
    def __init__(self, decoder: str, device: str = "cuda:0", encoder: str | None = None, pitch: str | None = None,
                 index: float = 0.0, index_k: int = 8, manifest: str = "data/eval/manifest.json", hsmooth: int = 0):
        self.hsmooth = hsmooth  # diagnostic: moving average (2k+1 frames) over the decoder's 100 Hz features
        self.index, self.index_k, self.manifest, self.banks = index, index_k, manifest, {}
        ckpt = torch.load(decoder, map_location="cpu")
        self.device, self.speakers = device, ckpt["speakers"]
        if encoder == "teacher":  # diagnostic: the non-causal teacher's features instead of the student's
            from causalvc.encoder.train import load_teacher

            class _Teacher:
                def __init__(self):
                    self.m = load_teacher(device)

                def encode(self, wav16):
                    return self.m(wav16.float()).last_hidden_state

            self.enc = _Teacher()
        else:
            from causalvc.voice import encoder_path

            self.enc = load_encoder(encoder_path(encoder or ckpt.get("encoder")), device)
        self.dec = Decoder(len(self.speakers), **ckpt["cfg"])
        self.dec.load_state_dict(ckpt["generator"])
        self.dec.to(device).eval()
        if pitch:  # causal tracker instead of whole-file RMVPE
            from causalvc.pitch_causal import load_pitch

            self.rmvpe = load_pitch(pitch, device)
        else:
            self.rmvpe = Rmvpe(device)

    @torch.no_grad()
    def convert(
        self,
        wav48: torch.Tensor,
        speaker: str,
        semitones: float = 0.0,
        offset_hz: float = 0.0,
        ceiling_hz: float = 0.0,
        ceiling_keep: float = 1.0,
        floor_hz: float = 0.0,
        floor_keep: float = 1.0,
    ) -> torch.Tensor:
        """Mono [N] at 48 kHz in, same length out."""
        n = wav48.shape[-1]
        unit = FRAME if getattr(self.enc, "hop", 320) == 160 else 2 * FRAME
        x = torch.nn.functional.pad(wav48.to(self.device)[None], (0, (-n) % unit))
        content, f0, voicing, loud = extract_all(self.enc, self.rmvpe, x)
        if self.index > 0:  # pull each content frame toward the target's own nearest frames
            near = self.lookup(content[0], self.bank(speaker))
            content = (1 - self.index) * content + self.index * near[None]
        spk = torch.tensor([self.speakers.index(speaker)], device=self.device)
        from causalvc.stream import map_f0, normalize_tone, smooth_f0

        n_fr = min(f0.shape[1], loud.shape[1])
        f0, voicing = f0[:, :n_fr], voicing[:, :n_fr]
        loud = pitch_sync_loudness(x[:, : n_fr * FRAME], f0)
        tn = normalize_tone(tone(x)[0, :n_fr], loud[0], f0[0])[0][None] if self.dec.tone is not None else None
        f0 = map_f0(
            smooth_f0(f0[0])[0],
            2.0 ** (semitones / 12.0),
            offset_hz,
            ceiling_hz,
            ceiling_keep,
            floor_hz,
            floor_keep,
        )[None]
        enc_hz = 100 if getattr(self.enc, "hop", 320) == 160 else 50
        v_in = voicing if self.dec.expressive else None
        h, g = self.dec.features(content, f0, loud, spk, tone=tn, content_hz=enc_hz, voicing=v_in)
        if self.hsmooth:
            k = self.hsmooth
            h = torch.nn.functional.avg_pool1d(torch.nn.functional.pad(h, (k, k), mode="replicate"), 2 * k + 1, 1)
        th = h.shape[-1]
        reg = self.dec.register_vec(f0[:, :th], None if v_in is None else v_in[:, :th]) if self.dec.reg_cond is not None else None
        exc = self.dec.source(f0[:, :th], voicing=None if v_in is None else v_in[:, :th])
        y = self.dec.generate(h, g, exc, reg=reg)[0, 0]
        return torch.nn.functional.pad(y, (0, max(0, n - len(y))))[:n]

    @torch.no_grad()
    def bank(self, speaker: str, max_frames: int = 120000) -> torch.Tensor:
        """Content frames of the target's own training audio, [M, C]."""
        if speaker not in self.banks:
            import json

            import soundfile as sf

            feats, n = [], 0
            for path in json.load(open(self.manifest))["targets"][speaker]["train"]:
                wav, sr = sf.read(path, dtype="float32", always_2d=True)
                w16 = torchaudio.functional.resample(torch.from_numpy(wav[:, 0]), sr, 16000).to(self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=isinstance(self.enc, StreamingEncoder)):
                    feats.append(self.enc.encode(w16[None]).float()[0])
                n += len(feats[-1])
                if n >= max_frames:
                    break
            self.banks[speaker] = torch.cat(feats)[:max_frames]
        return self.banks[speaker]

    def lookup(self, x: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """[T, C] -> the similarity-weighted mean of each frame's k nearest bank frames."""
        sim = torch.nn.functional.normalize(x, dim=-1) @ torch.nn.functional.normalize(bank, dim=-1).T
        top, idx = sim.topk(self.index_k, dim=-1)
        w = torch.softmax(top / 0.05, dim=-1)
        return (bank[idx] * w[..., None]).sum(1)
