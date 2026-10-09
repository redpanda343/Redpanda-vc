import math

import numpy as np

from beatrice.inference import IN_SAMPLE_RATE, LATENCY_SECONDS, OUT_SAMPLE_RATE, BeatriceModel


class StreamResampler:
    def __init__(self, in_rate, out_rate, zeros=16, beta=8.0):
        in_rate, out_rate = int(in_rate), int(out_rate)
        divisor = math.gcd(in_rate, out_rate)
        self.up = out_rate // divisor
        self.down = in_rate // divisor
        self.delay_seconds = 0.0
        if self.up == self.down:
            return
        lower = min(in_rate, out_rate)
        delay = zeros * in_rate / lower
        cutoff = 0.92 * lower / in_rate
        self.taps = int(math.ceil(2.0 * delay)) + 1
        self.delay_seconds = delay / in_rate
        fraction = (np.arange(self.up) * self.down % self.up) / self.up
        distance = fraction[:, None] + np.arange(self.taps)[None] - delay
        window = np.i0(beta * np.sqrt(np.clip(1.0 - (distance / delay) ** 2, 0.0, None))) / np.i0(beta)
        table = cutoff * np.sinc(cutoff * distance) * window * (np.abs(distance) <= delay)
        self.table = (table / table.sum(1, keepdims=True)).astype(np.float32)
        self.reset()

    def reset(self):
        if self.up == self.down:
            return
        self.buffer = np.zeros(self.taps, dtype=np.float32)
        self.start = -self.taps
        self.next = 0

    def __call__(self, audio):
        if self.up == self.down:
            return audio
        self.buffer = np.concatenate([self.buffer, audio])
        last = self.start + self.buffer.shape[0] - 1
        final = ((last + 1) * self.up - 1) // self.down
        if final < self.next:
            return np.zeros(0, dtype=np.float32)
        outputs = np.arange(self.next, final + 1)
        newest = outputs * self.down // self.up
        indices = newest[:, None] - np.arange(self.taps)[None] - self.start
        result = (self.buffer[indices] * self.table[outputs % self.up]).sum(1, dtype=np.float32)
        self.next = final + 1
        keep = self.next * self.down // self.up - self.taps + 1 - self.start
        self.buffer = self.buffer[max(0, keep):]
        self.start += max(0, keep)
        return result


class BeatriceRealtime:
    def __init__(self, model_path, stream_rate, device='cpu', speaker=0, pitch=0.0, formant=0.0, vq_neighbors=0):
        self.model = BeatriceModel(model_path, device)
        self.model.set_speaker(speaker)
        self.model.set_pitch_shift(pitch)
        self.model.set_formant_shift(formant)
        self.model.set_vq_neighbors(vq_neighbors)
        self.stream_rate = int(stream_rate)
        self.input_resampler = StreamResampler(self.stream_rate, IN_SAMPLE_RATE)
        self.output_resampler = StreamResampler(OUT_SAMPLE_RATE, self.stream_rate)
        self.hop = math.ceil(self.stream_rate / 100)
        self.margin = 0 if self.stream_rate % 100 == 0 else self.hop
        self.reset()

    @property
    def latency_seconds(self):
        return (LATENCY_SECONDS + self.input_resampler.delay_seconds + self.output_resampler.delay_seconds
                + self.margin / self.stream_rate)

    def reset(self):
        self.model.reset()
        self.input_resampler.reset()
        self.output_resampler.reset()
        self.pending = np.zeros(0, dtype=np.float32)
        self.started = False

    def process(self, audio):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        converted = self.model.process(self.input_resampler(audio))
        converted = self.output_resampler(converted.cpu().numpy())
        self.pending = np.concatenate([self.pending, converted])
        count = audio.shape[0]
        if not self.started:
            if self.pending.shape[0] < count + self.margin:
                return np.zeros(count, dtype=np.float32)
            self.started = True
            self.pending = self.pending[-(count + self.margin):]
        if self.pending.shape[0] < count:
            self.pending = np.concatenate([np.zeros(count - self.pending.shape[0], dtype=np.float32), self.pending])
        output, self.pending = self.pending[:count], self.pending[count:]
        if self.pending.shape[0] > self.margin + self.hop:
            self.pending = self.pending[-self.margin:] if self.margin else self.pending[:0]
        return output
