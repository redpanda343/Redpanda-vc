"""Block-by-block conversion with the causal encoder, pitch tracker and decoder.

Each block, the encoder and pitch tracker run over the last few seconds of
input and only their newly final frames are kept; the decoder then generates
just the new audio from a short history of those frames. Output trails input by
`DELAY` frames (50 ms): the encoder's 40 ms lookahead and the pitch tracker's
52 ms, rounded up to the 10 ms frame clock. Blocks must be multiples of 20 ms.
"""

from __future__ import annotations

import math

import torch
import torchaudio

from causalvc.decoder.model import FRAME, PS_LOOKBACK, SR, Decoder, loudness, pitch_sync_loudness, tone
from causalvc.infer import load_encoder
from causalvc.voice import encoder_path
from causalvc.pitch_causal import load_pitch

DELAY = 5  # 100 Hz frames between the newest input and the newest output
HIST = 128  # frames of features kept for the decoder
GEN = 32  # frames of context the waveform generator is given
F0_RANGE = (32.0, 1400.0)  # Hz: sub-50 Hz vocal fry up to high soprano/whistle
REF_TONE = (-14.5, -6.5)  # dB (brightness, openness) of clean studio modal speech
_REF_TONE_BUF: dict[str, torch.Tensor] = {}


def _ref_tone(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    key = f"{device}:{dtype}"
    buf = _REF_TONE_BUF.get(key)
    if buf is None:
        buf = torch.tensor(REF_TONE, device=device, dtype=dtype)
        _REF_TONE_BUF[key] = buf
    return buf


def smooth_f0(f0: torch.Tensor, state: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal zero-latency micro-jitter smoother on low-pitched F0 [m] (Hz, 0 unvoiced) -> (smoothed [m], state [1]).

    Only damps <25-cent 10 ms glottal-window jitter when 0 < F0 < 145 Hz; real intonation glides (>=32 cents / 10 ms),
    unvoiced-to-voiced onsets, and higher-pitched voices pass through with zero lag. CUDA-graph safe."""
    cur = f0.new_zeros(1) if state is None else state.clone()
    out = []
    for i in range(f0.shape[0]):
        fi = f0[i : i + 1]
        both = (fi > 0) & (cur > 0)
        low_w = ((145.0 - fi) / 35.0).clamp(0.0, 1.0)
        cents = 1200.0 * torch.abs(torch.log2(fi.clamp_min(1.0) / cur.clamp_min(1.0)))
        t = ((cents - 10.0) / 22.0).clamp(0.0, 1.0)
        alpha = 1.0 - low_w * (1.0 - (0.28 + 0.72 * (t * t)))
        blended = cur * (fi.clamp_min(1.0) / cur.clamp_min(1.0)) ** alpha
        cur = torch.where(fi <= 0, torch.zeros_like(fi), torch.where(both, blended, fi))
        out.append(cur[0])
    return (torch.stack(out) if out else f0), cur


def normalize_tone(
    tn: torch.Tensor,
    loud: torch.Tensor,
    raw_f0: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift dark/chest-heavy microphone baselines toward studio vocal presence while keeping sibilant & laugh dynamics."""
    ref = _ref_tone(tn.device, tn.dtype)
    avg = ref.clone() if state is None else state.clone()
    out = []
    for i in range(tn.shape[0]):
        ti = tn[i]
        voiced_act = ((raw_f0[i] > 0) & (loud[i] > -55.0)).float()
        avg = avg + (0.04 * voiced_act) * (ti - avg)
        lift = (ref - avg).clamp(-2.0, 8.5)
        conf = ((loud[i] + 58.0) / 14.0).clamp(0.0, 1.0)
        out.append(ref + conf * ((ti + lift) - ref))
    return (torch.stack(out) if out else tn), avg


def map_f0(
    f0: torch.Tensor,
    ratio: float,
    offset: float = 0.0,
    ceiling: float = 0.0,
    keep: float = 1.0,
    floor: float = 0.0,
    floor_keep: float = 1.0,
) -> torch.Tensor:
    """The pitch the voice is given for the tracked pitch ``f0`` (Hz, 0 unvoiced).

    ``f0 * ratio + offset``, then an optional ``floor`` (Hz, 0 = none) below which only
    ``floor_keep`` (0-1) of the drop is kept in semitones, and an optional ``ceiling`` (Hz,
    0 = none) above which only ``keep`` (0-1) of the rise is kept in semitones. Applies to
    voiced frames only (unvoiced ones stay 0) and stays within ``F0_RANGE``."""
    if offset == 0.0 and ceiling <= 0.0 and floor <= 0.0:
        return f0 * ratio
    y = (f0 * ratio + offset).clamp(*F0_RANGE)
    if floor > 0.0:
        fl = min(floor, ceiling) if ceiling > 0.0 else floor
        y = torch.where(y < fl, fl * (y / fl) ** max(0.0, min(1.0, floor_keep)), y)
    if ceiling > 0.0:
        y = torch.where(y > ceiling, ceiling * (y / ceiling) ** max(0.0, min(1.0, keep)), y)
    return torch.where(f0 > 0, y.clamp(*F0_RANGE), torch.zeros_like(f0))


class StreamingVC:
    def __init__(self, decoder: str, pitch: str, device: str = "cuda:0", encoder: str | None = None, window: float = 4.0,
                 level: float | None = None):
        self.level_target = level  # None (default) leaves the input as is; a dB RMS value turns on a slow automatic gain toward it
        ckpt = torch.load(decoder, map_location="cpu")
        self.device, self.speakers = device, ckpt["speakers"]
        self.enc = load_encoder(encoder_path(encoder or ckpt.get("encoder")), device)
        self.pitch = load_pitch(pitch, device)
        self.dec = Decoder(len(self.speakers), **ckpt["cfg"])
        self.dec.load_state_dict(ckpt["generator"])
        self.dec.to(device).eval()
        # glide="next" slides the excitation pitch to the following frame's value, so it runs one frame behind
        self.hop100 = getattr(self.enc, "hop", 320) == 160
        self.base_delay = (self.pitch.lookahead + 3) if self.hop100 else DELAY
        self.late = int(self.dec.source.glide == "next")
        self.delay = self.base_delay + self.dec.lookahead + self.late  # frames between the newest input and the newest output
        if self.hop100:
            assert self.enc.lookahead <= self.base_delay
        else:
            assert self.enc.lookahead * 2 <= DELAY - 1 and self.pitch.lookahead + 4 <= DELAY + 1
        self.n = int(window * SR)
        assert self.n >= max(self.base_delay * FRAME + PS_LOOKBACK, (self.base_delay + 1) * FRAME) + FRAME
        self.ratio, self.offset, self.ceiling, self.keep, self.floor, self.floor_keep, self.spk = 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, None
        self.index, self.rate = None, 0.0  # the voice's index (causalvc.index) and its strength
        self.reset()

    def set_index(self, frames: torch.Tensor | None) -> None:
        """The voice's index frames [N, C] (None: no index)."""
        self.index = None if frames is None else frames.to(self.device, torch.float32)
        self.index_unit = None if frames is None else torch.nn.functional.normalize(self.index, dim=-1)

    def set_index_rate(self, rate: float) -> None:
        """How much of each content frame comes from the index (0 = none, 1 = all)."""
        self.rate = max(0.0, min(1.0, float(rate)))

    def set_voice(
        self,
        speaker: str,
        semitones: float = 0.0,
        offset_hz: float = 0.0,
        ceiling_hz: float = 0.0,
        ceiling_keep: float = 1.0,
        floor_hz: float = 0.0,
        floor_keep: float = 1.0,
    ) -> None:
        """The voice to convert to and its pitch mapping; see ``map_f0``."""
        self.spk = torch.tensor([self.speakers.index(speaker)], device=self.device)
        self.ratio, self.offset = 2.0 ** (semitones / 12.0), float(offset_hz)
        self.ceiling, self.keep = float(ceiling_hz), float(ceiling_keep)
        self.floor, self.floor_keep = float(floor_hz), float(floor_keep)

    def reset(self) -> None:
        d = self.device
        self.buf = torch.zeros(self.n, device=d)
        self.content = torch.zeros(HIST, 768, device=d)
        self.f0 = torch.zeros(HIST, device=d)
        self.voicing = torch.zeros(HIST, device=d)
        self.loud = torch.full((HIST,), -100.0, device=d)
        self.tone = torch.zeros(HIST, 2, device=d)
        self.exc = torch.zeros(self.dec.source.channels, HIST * FRAME, device=d)
        self.phase = torch.zeros(1, dtype=torch.float64, device=d)
        self.prev_f0 = torch.zeros(1, device=d)
        self.smooth_state = torch.zeros(1, device=d)
        self.tone_state = torch.tensor(REF_TONE, device=d)
        self.pending_f0 = torch.zeros(1, device=d)  # newest pitch frame, whose excitation waits for the next one
        self.pending_v = torch.zeros(1, device=d)
        self.level, self.level_pow, self.level_n, self.gain = -23.0, 0.0, 0, 1.0  # running speech level (dB, power), frames seen, gain

    LEVEL_BAND = 5.0  # dB above the target that still counts as an ordinary level and is left alone

    def normalise(self, block: torch.Tensor, m: int) -> torch.Tensor:
        """Slow automatic gain for microphones far from ordinary speech levels. The decoder
        is conditioned on loudness and was trained on speech at ordinary levels, so a very
        quiet microphone would otherwise be rendered as soft, dull speech.

        The speech level is the mean power of frames within 20 dB of the running level, so
        pauses and word tails do not drag it down. Nothing changes until 0.3 s of speech has
        been heard, input already between the target and LEVEL_BAND above it is left untouched,
        and the gain moves at most 10 dB per second."""
        db = loudness(block[None])[0]
        floor = max(self.level - 20.0, -60.0) if self.level_n else -50.0
        for v in db[db > floor].tolist():
            self.level_n += 1
            p = self.level_pow + (10 ** (v / 10) - self.level_pow) * max(1.0 / self.level_n, 1.0 / 300.0)
            self.level_pow, self.level = p, 10 * math.log10(p)
        want = 0.0
        if self.level_n >= 30:
            lo, hi = self.level_target, self.level_target + self.LEVEL_BAND
            want = min(max(lo - self.level, 0.0), 30.0) if self.level < lo else max(hi - self.level, -12.0) if self.level > hi else 0.0
        step = 0.1 * m  # dB this block may move
        now = 20 * math.log10(self.gain)
        gain = 10 ** ((now + min(max(want - now, -step), step)) / 20)
        ramp = torch.linspace(self.gain, gain, block.shape[0] + 1, device=block.device)[1:]
        self.gain = gain
        return (block * ramp).clamp(-1.0, 1.0)

    @torch.no_grad()
    def process(self, block: torch.Tensor) -> torch.Tensor:
        """[n] at 48 kHz (n a multiple of 960 at 50 Hz, or 480 at 100 Hz) -> [n], delayed by base_delay frames."""
        m = block.shape[0] // FRAME  # new 100 Hz frames
        assert block.shape[0] == m * FRAME and m >= 1 and (self.hop100 or m % 2 == 0)
        D = self.base_delay
        block = block.to(self.device)
        if self.level_target is not None:
            block = self.normalise(block, m)
        self.buf = torch.cat([self.buf[block.shape[0] :], block])
        total = self.n // FRAME
        frames = torch.arange(total - D - m, total - D, device=self.device)  # frames made final now

        wav16 = torchaudio.functional.resample(self.buf[None], SR, 16000)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.buf.is_cuda):
            steps = self.enc.forward(wav16).float()[0]  # step t is content frame t - lookahead
        if self.hop100:
            content = steps[frames + self.enc.lookahead]
            if self.index is not None and self.rate > 0:
                from causalvc.index import lookup

                content = torch.lerp(content, lookup(content, self.index, self.index_unit), self.rate)
        else:
            idx_curr = frames // 2 + self.enc.lookahead
            if "c" in self.dec.smooth_in:
                idx_prev = (frames // 2 - 1).clamp_min(0) + self.enc.lookahead
                c_curr, c_prev = steps[idx_curr], steps[idx_prev]
                if self.index is not None and self.rate > 0:
                    from causalvc.index import lookup

                    c_curr = torch.lerp(c_curr, lookup(c_curr, self.index, self.index_unit), self.rate)
                    c_prev = torch.lerp(c_prev, lookup(c_prev, self.index, self.index_unit), self.rate)
                even = (frames % 2 == 0)[:, None]
                content = torch.where(even, 0.5 * (c_curr + c_prev), c_curr)
            else:
                content = steps[idx_curr]
                if self.index is not None and self.rate > 0:
                    from causalvc.index import lookup

                    content = torch.lerp(content, lookup(content, self.index, self.index_unit), self.rate)
        sal = torch.sigmoid(self.pitch.forward(wav16).float())  # step t is pitch frame t - lookahead
        raw_f0_all, raw_v = self.pitch.decode_with_voicing(sal)
        raw_f0 = raw_f0_all[0, frames + self.pitch.lookahead]
        v = raw_v[0, frames + self.pitch.lookahead]
        f0_s, self.smooth_state = smooth_f0(raw_f0, self.smooth_state)
        f0 = map_f0(f0_s, self.ratio, self.offset, self.ceiling, self.keep, self.floor, self.floor_keep)
        start_s = int(frames[0].item()) * FRAME - PS_LOOKBACK
        assert start_s >= 0, "StreamingVC window is too short for pitch_sync_loudness lookback"
        end_s = int(frames[-1].item() + 1) * FRAME
        loud = pitch_sync_loudness(self.buf[None, start_s:end_s], raw_f0[None], prepadded=True)[0]

        use_v = self.dec.expressive
        if self.late:  # excitation for the frame before each new one, now that its following pitch is known
            f0_e = torch.cat([self.pending_f0, f0[:-1]])
            v_e = torch.cat([self.pending_v, v[:-1]]) if use_v else None
            exc = self.dec.source(f0_e[None], self.phase, nxt=f0[-1:], voicing=None if v_e is None else v_e[None], nxt_v=v[-1:] if use_v else None)[0]
            self.pending_f0, self.pending_v = f0[-1:], v[-1:]
        else:
            exc = self.dec.source(f0[None], self.phase, self.prev_f0, voicing=v[None] if use_v else None)[0]
        self.phase, self.prev_f0 = self.dec.source.last_phase, f0[-1:]
        self.content = torch.cat([self.content[m:], content])
        self.f0 = torch.cat([self.f0[m:], f0])
        self.voicing = torch.cat([self.voicing[m:], v])
        self.loud = torch.cat([self.loud[m:], loud])
        if self.dec.tone is not None:
            tn_new, self.tone_state = normalize_tone(tone(self.buf[None])[0, frames], loud, raw_f0, self.tone_state)
            self.tone = torch.cat([self.tone[m:], tn_new])
        self.exc = torch.cat([self.exc[:, m * FRAME :], exc], dim=1)

        dec = self.dec
        g = dec.speaker(self.spk)
        x = dec.content(self.content[None]) + dec.pitch_embed(self.f0[None])
        x = x + dec.loud((self.loud[None, :, None] + 40.0) / 20.0)
        if dec.f0_proj is not None:
            x = x + dec.f0_proj(dec.register_vec(self.f0[None], self.voicing[None]).to(dec.f0_proj.weight.dtype))
        if dec.tone is not None:
            tn = dec.tone((self.tone[None] / 10.0).to(dec.tone.weight.dtype))
            if dec.expressive:
                tn = tn * dec.tone_gate(self.loud[None])[..., None].to(tn.dtype)
            x = x + tn
        x = x.transpose(1, 2)
        for b in dec.blocks:
            x = b(x, g)
        if "h" in dec.smooth_in:  # as Decoder.features does
            x = (x + torch.nn.functional.pad(x, (1, 0), mode="replicate")[..., :-1]) / 2
        k, a = m + GEN, dec.lookahead + self.late  # the last `a` frames still lack their future input or excitation
        end = x.shape[-1] - a
        e = self.late  # the excitation buffer runs this many frames behind the features
        reg = dec.register_vec(self.f0[None, end - k : end], self.voicing[None, end - k : end]) if dec.reg_cond is not None else None
        y = dec.generate(x[..., end - k : end], g, self.exc[None, :, (end - k + e) * FRAME : (end + e) * FRAME], reg=reg)
        return y[0, 0, -m * FRAME :]
