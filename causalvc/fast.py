"""Block-by-block conversion that computes only what is new.

`causalvc.stream.StreamingVC` re-runs the encoder and pitch tracker over the last
four seconds for every block and regenerates a third of a second of waveform to
emit 20 ms. Every network here only looks backward, so all of that older work is
already final. This runtime keeps, for each layer, the little history the layer
needs (a few frames for a convolution, the attention window for the encoder) and
runs each layer on the new frames alone. Same weights, same output, a fraction
of the work.

Same interface as StreamingVC. Models with decoder lookahead or the diagnostic
conditioning options are not handled here; `streaming()` falls back for those.

With ``graphs``, `prepare()` records one block's work as a CUDA graph and every
later block replays it: a block is a few hundred small kernels, and launching
them one by one from Python costs more than running them. The stream then
starts from the state four seconds of silence leave (instead of from nothing),
which only changes its first three seconds.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torchaudio

from causalvc.decoder.model import FRAME, PS_LOOKBACK, SR, AntiAliasAct, CausalLPF, Decoder, _ps_idx, _tone_buffers, pitch_sync_loudness, tone
from causalvc.index import lookup
from causalvc.infer import load_encoder
from causalvc.voice import encoder_path
from causalvc.pitch_causal import load_pitch
from causalvc.stream import DELAY, F0_RANGE, REF_TONE, _ref_tone, map_f0, normalize_tone, smooth_f0

TAIL = 3840  # 48 kHz samples of input kept between blocks (80 ms): covers every front end's history


def check_tail_capacity(base_delay: int) -> None:
    """Ensure both pitch_sync_loudness and tone slices stay non-negative inside the 80 ms TAIL buffer."""
    assert base_delay * FRAME + PS_LOOKBACK <= TAIL and (base_delay + 1) * FRAME <= TAIL, (
        f"base_delay={base_delay} exceeds TAIL={TAIL // FRAME} frames"
    )


def supported(dec: Decoder) -> bool:
    return dec.lookahead == 0 and set(dec.smooth_in) <= {"c", "p", "h"} and dec.mel is None and dec.exc in ("full", "bandlimit", "multirate")


def map_f0_tensor(f0: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """`map_f0` with its settings in a tensor, p = [ratio, offset, ceiling, keep, floor, floor_keep, bounded],
    so a recorded graph follows changes to them. bounded is 1 when offset, ceiling, or floor is in use."""
    y = (f0 * p[0] + p[1]).clamp(*F0_RANGE)
    c = p[2]
    fl = torch.where(c > 0, torch.minimum(p[4], c), p[4])
    y = torch.where((fl > 0) & (y < fl), fl * (y / fl.clamp_min(1e-6)) ** p[5], y)
    y = torch.where((c > 0) & (y > c), c * (y / c.clamp_min(1e-6)) ** p[3], y)
    y = y.clamp(*F0_RANGE)
    return torch.where(p[6] > 0, torch.where(f0 > 0, y, torch.zeros_like(f0)), f0 * p[0])


class FastStreamingVC:
    def __init__(self, decoder: str, pitch: str, device: str = "cuda:0", encoder: str | None = None, level: float | None = None,
                 graphs: bool = False):
        assert level is None, "the automatic input gain lives in StreamingVC only"
        ckpt = torch.load(decoder, map_location="cpu")
        self.device, self.speakers = device, ckpt["speakers"]
        self.enc = load_encoder(encoder_path(encoder or ckpt.get("encoder")), device)
        self.pitch = load_pitch(pitch, device)
        self.dec = Decoder(len(self.speakers), **ckpt["cfg"])
        self.dec.load_state_dict(ckpt["generator"])
        self.dec.to(device).eval().requires_grad_(False)
        assert supported(self.dec), "this decoder needs causalvc.stream.StreamingVC"
        # Weight normalisation recomputes every kernel on every call; the weights are fixed now, so store them plain.
        for mod in self.dec.modules():
            if torch.nn.utils.parametrize.is_parametrized(mod, "weight"):
                torch.nn.utils.parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
        # Rotary position factors for every length the attention cache can have, computed once.
        e = self.enc
        d = e.blocks[0].qkv.out_features // 3 // e.blocks[0].heads // 2
        freq = torch.exp(-torch.log(torch.tensor(10000.0)) * torch.arange(d, device=device) / d)
        ang = torch.arange(e.window + 64, device=device)[:, None].float() * freq[None]
        self.rot = (ang.cos(), ang.sin())
        # The same filter as torchaudio.functional.resample, built once (and so recordable in a graph).
        self.resample = torchaudio.transforms.Resample(SR, 16000).to(device)
        self.hop100 = getattr(self.enc, "hop", 320) == 160
        self.base_delay = (self.pitch.lookahead + 3) if self.hop100 else DELAY
        self.late = int(self.dec.source.glide == "next")
        self.delay = self.base_delay + self.late
        if self.hop100:
            assert self.enc.lookahead <= self.base_delay
        else:
            assert self.enc.lookahead * 2 <= DELAY - 1 and self.pitch.lookahead + 4 <= DELAY + 1
        check_tail_capacity(self.base_delay)
        self.ratio, self.offset, self.ceiling, self.keep, self.floor, self.floor_keep = 1.0, 0.0, 0.0, 1.0, 0.0, 1.0
        # Kept in place (a recorded graph reads them where they are): the speaker and the pitch settings.
        self.spk = torch.zeros(1, dtype=torch.long, device=device)
        self.pmap = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0], device=device)
        # The voice's index (causalvc.index), if it has one, and how strongly it is blended in (0-1).
        self.index = self.index_unit = None
        self.index_rate, self.rate = torch.zeros((), device=device), 0.0
        _ps_idx(torch.device(device))
        if self.dec.tone is not None:
            _tone_buffers(torch.device(device))
            _ref_tone(torch.device(device))
        self.graphs = bool(graphs) and str(device).startswith("cuda")
        self.graph = None
        self.reset()

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
        self.spk.fill_(self.speakers.index(speaker))
        self.ratio, self.offset = 2.0 ** (semitones / 12.0), float(offset_hz)
        self.ceiling, self.keep = float(ceiling_hz), float(ceiling_keep)
        self.floor, self.floor_keep = float(floor_hz), float(floor_keep)
        bounded = float(self.offset != 0.0 or self.ceiling > 0.0 or self.floor > 0.0)
        vals = torch.tensor(
            [
                self.ratio,
                self.offset,
                self.ceiling,
                max(0.0, min(1.0, self.keep)),
                self.floor,
                max(0.0, min(1.0, self.floor_keep)),
                bounded,
            ],
            device=self.pmap.device,
            dtype=self.pmap.dtype,
        )
        self.pmap.copy_(vals)

    def set_index(self, frames: torch.Tensor | None) -> None:
        """The voice's index frames [N, C] (None: no index). Set before ``prepare``: a recorded graph
        either looks up in the index or does not."""
        self.graph = None
        if frames is None:
            self.index = self.index_unit = None
        else:
            self.index = frames.to(self.device, torch.float32)
            self.index_unit = torch.nn.functional.normalize(self.index, dim=-1)

    def set_index_rate(self, rate: float) -> None:
        """How much of each content frame comes from the index (0 = none, 1 = all)."""
        self.rate = max(0.0, min(1.0, float(rate)))
        self.index_rate.fill_(self.rate)

    @torch.inference_mode()  # the recorded state tensors were made in inference mode
    def reset(self) -> None:
        if self.graph is not None:  # back to the recorded starting state, in place
            for buf, start in zip(self.static, self.snapshot):
                buf.copy_(start)
            self.frames = self.snapshot_frames
            return
        d = self.device
        self.tail = torch.zeros(TAIL, device=d)  # newest input, 48 kHz
        self.st: dict = {}  # per-layer history, created on first use
        self.steps: list[torch.Tensor] = []  # newest encoder outputs [768]; the last is the newest step
        self.phase = torch.zeros(1, dtype=torch.float64, device=d)
        self.prev_f0 = torch.zeros(1, device=d)
        self.smooth_state = torch.zeros(1, device=d)
        self.tone_state = torch.tensor(REF_TONE, device=d)
        self.pending_f0 = torch.zeros(1, device=d)
        self.pending_v = torch.zeros(1, device=d)
        self.pending_h = None  # features of the newest frame, waiting for its excitation (glide="next")
        self.frames = 0  # 100 Hz frames of input received so far

    # ---- history helpers -------------------------------------------------

    def _hist(self, key, x: torch.Tensor, n: int) -> torch.Tensor:
        """x [..., T] preceded by the last n samples seen under this key (zeros at the start); remembers the new tail."""
        old = self.st.get(key)
        if old is None:
            old = x.new_zeros(*x.shape[:-1], n)
        z = torch.cat([old, x], dim=-1)
        self.st[key] = z[..., z.shape[-1] - n :]
        return z

    def _cconv(self, m, x: torch.Tensor, key) -> torch.Tensor:
        """A causal convolution (decoder.model.CConv) on new frames only."""
        return m.conv(self._hist(key, x, m.pad)) if m.pad else m.conv(x)

    def _lpf(self, lpf: CausalLPF, x: torch.Tensor, key) -> torch.Tensor:
        return F.conv1d(self._hist(key, x, lpf.pad), lpf.weight, groups=lpf.channels)

    def _aa_act(self, act: AntiAliasAct, x: torch.Tensor, key) -> torch.Tensor:
        b, c, t = x.shape
        u = F.conv1d(self._hist((key, "u"), x, act.pad_up), act.w_up, groups=c).view(b, c, 2, t).transpose(2, 3).reshape(b, c, 2 * t)
        u = F.leaky_relu(u, 0.1)
        return F.conv1d(self._hist((key, "d"), u, act.pad_dn), act.w_dn, stride=2, groups=c)

    # ---- content encoder -------------------------------------------------

    def _encode(self, wav16: torch.Tensor, n: int) -> torch.Tensor:
        """The last 240 + hop * n samples at 16 kHz -> n new encoder steps [n, 768]."""
        e = self.enc
        m = torch.log(e.mel(wav16[None]).clamp_min(1e-5))  # [1, mels, 2n at 50 Hz or n at 100 Hz]
        h = F.gelu(e.conv1(self._hist("e.mel", m, 2)))
        h2_pad = 3 if getattr(e, "hop", 320) == 160 else 2
        x = F.gelu(e.conv2(self._hist("e.h", h, h2_pad))).transpose(1, 2)  # [1, n, dim]
        for i, blk in enumerate(e.blocks):
            b, t, c = x.shape
            q, k, v = blk.qkv(blk.n1(x)).view(b, t, 3, blk.heads, c // blk.heads).permute(2, 0, 3, 1, 4)
            if ("e.k", i) in self.st:
                k, v = torch.cat([self.st["e.k", i], k], dim=2), torch.cat([self.st["e.v", i], v], dim=2)
            total = k.shape[2]
            # keep what the next step may attend to: itself plus `window` frames back means `window` cached frames
            self.st["e.k", i], self.st["e.v", i] = k[:, :, -e.window :], v[:, :, -e.window :]
            if t == 1:  # one new step sees every cached key: the cache is exactly its window
                a = F.scaled_dot_product_attention(self._rot(q, total - 1, total), self._rot(k, 0, total), v)
            else:
                pos = torch.arange(total, device=x.device)
                rel = pos[-t:, None] - pos[None, :]
                a = F.scaled_dot_product_attention(self._rot(q, total - t, total), self._rot(k, 0, total), v, attn_mask=(rel >= 0) & (rel <= e.window))
            x = x + blk.proj(a.transpose(1, 2).reshape(b, t, c))
            hh = self._hist(("e.c", i), blk.n2(x).transpose(1, 2), blk.kernel - 1)
            x = x + blk.conv_out(F.gelu(blk.conv(hh).transpose(1, 2)))
            x = x + blk.ff(blk.n3(x))
        return e.head(e.norm(x))[0]

    def _rot(self, x: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """encoder.model.rotary for positions a..b-1, from the stored factors."""
        cos, sin = self.rot[0][a:b], self.rot[1][a:b]
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    # ---- pitch tracker ---------------------------------------------------

    def _pitch_with_voicing(self, wav16: torch.Tensor, m: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The last 1120 + 160 m samples at 16 kHz -> (F0 [m], voicing [m]) of m new frames."""
        p = self.pitch
        mel = torch.log(p.mel(wav16[None]).clamp_min(1e-5))[..., 4 : 4 + m]  # frames with a full window on both sides
        x = p.inp(self._hist("p.mel", mel, 2))
        for i, b in enumerate(p.blocks):
            y = b.dw(self._hist(("p.b", i), x, b.pad)).transpose(1, 2)
            x = x + b.fc2(F.gelu(b.fc1(b.norm(y)))).transpose(1, 2)
        x = x.transpose(1, 2)
        g, self.st["p.gru"] = p.gru(x, self.st.get("p.gru"))
        f0, v = p.decode_with_voicing(torch.sigmoid(p.head(torch.cat([x, g], dim=-1))))
        return f0[0], v[0]

    def _pitch(self, wav16: torch.Tensor, m: int) -> torch.Tensor:
        return self._pitch_with_voicing(wav16, m)[0]

    # ---- decoder ---------------------------------------------------------

    def _features(
        self,
        content: torch.Tensor,
        f0: torch.Tensor,
        loud: torch.Tensor,
        g: torch.Tensor,
        tn: torch.Tensor | None = None,
        voicing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """content [m, 768], f0 [m], loud [m] for the new frames -> hidden [1, D, m]."""
        d = self.dec
        x = d.content(content[None]) + d.pitch_embed(f0[None]) + d.loud((loud[None, :, None] + 40.0) / 20.0)
        if d.f0_proj is not None:
            rv = d.register_vec(f0[None], None if voicing is None else voicing[None]).to(d.f0_proj.weight.dtype)
            x = x + d.f0_proj(rv)
        if tn is not None:
            t_emb = d.tone((tn[None] / 10.0).to(d.tone.weight.dtype))
            if d.expressive:
                t_emb = t_emb * d.tone_gate(loud[None])[..., None].to(t_emb.dtype)
            x = x + t_emb
        x = x.transpose(1, 2)
        for i, b in enumerate(d.blocks):
            y = self._cconv(b.dw, x, ("d.b", i)).transpose(1, 2) + b.cond(g)[:, None]
            x = x + b.fc2(F.gelu(b.fc1(b.norm(y)))).transpose(1, 2)
        if "h" in d.smooth_in:
            z = self._hist("d.h", x, 1)
            x = (z[..., 1:] + z[..., :-1]) / 2
        return x

    def _generate(self, h: torch.Tensor, g: torch.Tensor, exc: torch.Tensor, reg: torch.Tensor | None = None) -> torch.Tensor:
        """hidden [1, D, m] and excitation [1, C, m * FRAME] for the same frames -> waveform [m * FRAME]."""
        d = self.dec
        x = self._cconv(d.pre, h, "g.pre") + d.pre_cond(g)[..., None]
        if d.reg_cond is not None and reg is not None:
            x = x + d.reg_cond(reg.to(d.reg_cond.weight.dtype)).transpose(1, 2)
        for i, (up, src, res) in enumerate(zip(d.ups, d.src, d.res)):
            x = F.leaky_relu(x, 0.1)
            z = self._hist(("g.up", i), x, 1)  # a transposed convolution's output hop depends on this frame and the one before
            u = up.conv(z)[..., up.stride : z.shape[-1] * up.stride]
            if up.lpf is not None:
                u = self._lpf(up.lpf, u, ("g.uplpf", i))
            if d.exc == "multirate":
                e_i = d.source.stage_excitation(exc, d.stage_rates[i], d.stage_rem[i]).to(x.dtype)
                s_i = self._cconv(src, e_i, ("g.src", i))
            elif d.exc == "bandlimit":
                s_i = src(d.limit(exc[:, :-1], exc[:, -1:], d.stage_rates[i]).to(x.dtype))
            else:
                s_i = src(exc)
            x = u + s_i
            out = 0
            for j, r in enumerate(res):
                y = x
                if r.a1 is None:
                    for k, (c1, c2) in enumerate(zip(r.c1, r.c2)):
                        t = self._cconv(c1, F.leaky_relu(y, 0.1), ("g.r", i, j, k, 1))
                        y = y + self._cconv(c2, F.leaky_relu(t, 0.1), ("g.r", i, j, k, 2))
                else:
                    for k, (c1, c2, a1, a2) in enumerate(zip(r.c1, r.c2, r.a1, r.a2)):
                        t = self._cconv(c1, self._aa_act(a1, y, ("g.a", i, j, k, 1)), ("g.r", i, j, k, 1))
                        y = y + self._cconv(c2, self._aa_act(a2, t, ("g.a", i, j, k, 2)), ("g.r", i, j, k, 2))
                out = out + y
            x = out / len(res)
        return torch.tanh(self._cconv(d.post, F.leaky_relu(x, 0.1), "g.post"))[0, 0]

    # ---- one block -------------------------------------------------------

    # ---- recorded graph --------------------------------------------------

    def _slots(self) -> list[torch.Tensor]:
        """Every tensor the stream carries from one block to the next."""
        late = [] if self.pending_h is None else [self.pending_h]
        return [
            self.tail, self.phase, self.prev_f0, self.smooth_state, self.tone_state,
            self.pending_f0, self.pending_v, *late, *self.steps, *(self.st[k] for k in self.keys),
        ]

    def _put(self, slots: list[torch.Tensor]) -> None:
        (
            self.tail, self.phase, self.prev_f0, self.smooth_state,
            self.tone_state, self.pending_f0, self.pending_v,
        ) = slots[:7]
        i = 7
        if self.pending_h is not None:
            self.pending_h, i = slots[7], 8
        n = len(self.steps)
        self.steps = list(slots[i : i + n])
        for k, t in zip(self.keys, slots[i + n :]):
            self.st[k] = t

    def _step_in_place(self) -> torch.Tensor:
        """One block from `self.inp`, leaving the new state in the same tensors as the old."""
        out = self._process(self.inp)
        for old, new in zip(self.static, self._slots()):
            if new is not old:  # in slot order: a slot that moved (the encoder steps) is read before it is overwritten
                old.copy_(new)
        self._put(self.static)
        return out

    @torch.inference_mode()
    def prepare(self, n: int) -> None:
        """Get ready for blocks of n samples; with `graphs`, record one block as a CUDA graph."""
        self.graph = None
        self.reset()
        if not self.graphs:
            return
        try:
            with torch.cuda.device(self.device):  # record on the voice's own GPU, not the current one
                self._record(n)
        except Exception as error:  # a GPU or driver that cannot record: carry on without
            print(f"CUDA graph not used: {error}", flush=True)
            self.graph, self.graphs = None, False
        self.reset()

    def _record(self, n: int) -> None:
        """Four seconds of silence, then one block of n samples recorded with every state tensor updated in place."""
        silence = torch.zeros(32 * FRAME, device=self.device)  # blocks short enough for the rotary table
        for _ in range(13):  # four seconds: every history (the encoder's three-second window above all) is full
            self._process(silence)
        for _ in range(3):  # and the encoder steps kept are those of this block length
            self._process(torch.zeros(n, device=self.device))
        self.keys = sorted(self.st, key=repr)
        self.static = [t.clone() for t in self._slots()]
        self._put(self.static)
        self.snapshot, self.snapshot_frames = [t.clone() for t in self.static], self.frames
        self.inp = torch.zeros(n, device=self.device)
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self._step_in_place()
        torch.cuda.current_stream(self.device).wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.out = self._step_in_place()
        self.graph, self.graph_n = graph, n

    @torch.inference_mode()
    def process(self, block: torch.Tensor) -> torch.Tensor:
        """[n] at 48 kHz (n a multiple of 960) -> [n], delayed by `delay` frames."""
        if self.graph is not None and block.shape[0] == self.graph_n:
            self.inp.copy_(block)
            with torch.cuda.device(self.device):
                self.graph.replay()
            self.frames += self.graph_n // FRAME
            return self.out.clone()
        if self.graph is not None:  # a different block length: back to running block by block
            self.graph = None
            self.reset()
        return self._process(block)

    def _process(self, block: torch.Tensor) -> torch.Tensor:
        m = block.shape[0] // FRAME
        assert block.shape[0] == m * FRAME and m >= 1 and (self.hop100 or m % 2 == 0)
        D = self.base_delay
        buf = torch.cat([self.tail, block.to(self.device)])
        self.tail = buf[-TAIL:]
        self.frames += m

        wav16 = self.resample(buf[None])[0]
        if self.hop100:
            for step in self._encode(wav16[-(240 + 160 * m) :], m):
                self.steps.append(step)
            self.steps = self.steps[-(m + D + 1) :]
            newest = self.frames - 1
            first = self.frames - D - m
            zero = self.steps[0].new_zeros(self.steps[0].shape)
            content = []
            for f in range(first, first + m):
                back = newest - (f + self.enc.lookahead)
                content.append(self.steps[-1 - back] if f >= 0 and 0 <= back < len(self.steps) else zero)
            content = torch.stack(content)
            if self.index is not None and (self.graphs or self.rate > 0):
                content = torch.lerp(content, lookup(content, self.index, self.index_unit), self.index_rate)
        else:
            for step in self._encode(wav16[-(240 + 160 * m) :], m // 2):
                self.steps.append(step)
            self.steps = self.steps[-(m // 2 + 3) :]
            # Frame f is final now (f = frames - D - m .. frames - D - 1); it uses content frame f // 2,
            # which is encoder step f // 2 + lookahead. The newest step is frames // 2 - 1.
            newest = self.frames // 2 - 1
            first = self.frames - D - m
            zero = self.steps[0].new_zeros(self.steps[0].shape)
            smooth_c = "c" in self.dec.smooth_in
            s_lo = first // 2 - int(smooth_c)
            s_hi = (first + m - 1) // 2
            c50 = []
            for s_idx in range(s_lo, s_hi + 1):
                back = newest - (s_idx + self.enc.lookahead)  # how many steps before the newest
                c50.append(self.steps[-1 - back] if s_idx >= 0 and 0 <= back < len(self.steps) else zero)
            c50_t = torch.stack(c50)
            if self.index is not None and (self.graphs or self.rate > 0):
                c50_t = torch.lerp(c50_t, lookup(c50_t, self.index, self.index_unit), self.index_rate)
            content = []
            for f in range(first, first + m):
                if f < 0:
                    content.append(zero)
                elif smooth_c and f % 2 == 0 and f > 0:
                    content.append(0.5 * (c50_t[f // 2 - s_lo] + c50_t[f // 2 - 1 - s_lo]))
                else:
                    content.append(c50_t[f // 2 - s_lo])
            content = torch.stack(content)

        raw_f0, v = self._pitch_with_voicing(wav16[-(1120 + 160 * m) :], m)
        if first < 0:  # before any real frame is final the old runtime's history is silence too
            valid = torch.arange(first, first + m, device=raw_f0.device) >= 0
            raw_f0 = torch.where(valid, raw_f0, torch.zeros_like(raw_f0))
            v = torch.where(valid, v, torch.zeros_like(v))
        f0_s, self.smooth_state = smooth_f0(raw_f0, self.smooth_state)
        f0 = (
            map_f0_tensor(f0_s, self.pmap)
            if self.graphs
            else map_f0(f0_s, self.ratio, self.offset, self.ceiling, self.keep, self.floor, self.floor_keep)
        )
        loud = pitch_sync_loudness(
            buf[None, buf.shape[0] - (D + m) * FRAME - PS_LOOKBACK : buf.shape[0] - D * FRAME],
            raw_f0[None],
            prepadded=True,
        )[0]

        g = self.dec.speaker(self.spk)
        tn = None
        if self.dec.tone is not None:  # each frame's tone is measured over it and the frame before
            raw_tn = tone(buf[None, buf.shape[0] - (D + m + 1) * FRAME : buf.shape[0] - D * FRAME])[0, 1:]
            tn, self.tone_state = normalize_tone(raw_tn, loud, raw_f0, self.tone_state)
        use_v = self.dec.expressive
        h = self._features(content, f0, loud, g, tn, v if use_v else None)
        if self.late:  # the excitation of a frame needs the next frame's pitch, so the waveform runs one frame behind
            f0_e = torch.cat([self.pending_f0, f0[:-1]])
            v_e = torch.cat([self.pending_v, v[:-1]]) if use_v else None
            exc = self.dec.source(f0_e[None], self.phase, nxt=f0[-1:], voicing=None if v_e is None else v_e[None], nxt_v=v[-1:] if use_v else None)
            self.pending_f0, self.pending_v = f0[-1:], v[-1:]
            held = h.new_zeros(1, h.shape[1], 1) if self.pending_h is None else self.pending_h
            h, self.pending_h = torch.cat([held, h[..., :-1]], dim=-1), h[..., -1:]
            reg = self.dec.register_vec(f0_e[None], None if v_e is None else v_e[None]) if self.dec.reg_cond is not None else None
        else:
            exc = self.dec.source(f0[None], self.phase, self.prev_f0, voicing=v[None] if use_v else None)
            reg = self.dec.register_vec(f0[None], v[None] if use_v else None) if self.dec.reg_cond is not None else None
        self.phase, self.prev_f0 = self.dec.source.last_phase, f0[-1:]
        return self._generate(h, g, exc, reg=reg)


def streaming(decoder: str, pitch: str, device: str = "cuda:0", encoder: str | None = None, level: float | None = None,
              graphs: bool = False):
    """The fastest runtime that handles this decoder."""
    cfg = torch.load(decoder, map_location="cpu")["cfg"]
    plain = (
        not cfg.get("lookahead")
        and set(cfg.get("smooth_in", "")) <= {"c", "p", "h"}
        and not cfg.get("mel_cond")
        and cfg.get("exc", "full") in ("full", "bandlimit", "multirate")
    )
    if plain and level is None:
        return FastStreamingVC(decoder, pitch, device, encoder, graphs=graphs)
    from causalvc.stream import StreamingVC

    return StreamingVC(decoder, pitch, device, encoder, level=level)
