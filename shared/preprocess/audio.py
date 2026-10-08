import io
import math
import os
import shutil
import subprocess

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal

from shared.preprocess.constants import (
    AUTOMATIC_DECODE_BLOCK_SECONDS,
    RESAMPLE_LOWPASS_FILTER_WIDTH,
    RESAMPLE_STREAM_CONTEXT_SECONDS,
    SIMPLE_STREAM_BLOCK_SECONDS,
)


now_directory = os.getcwd()
_RESAMPLER_CACHE = {}


def _ffmpeg_path():
    bundled_ffmpeg = os.path.join(now_directory, "ffmpeg.exe")
    if os.name == "nt" and os.path.isfile(bundled_ffmpeg):
        return bundled_ffmpeg
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _clean_audio_path(file: str) -> str:
    if os.name == "nt":
        file = file.replace("/", "\\")
    return file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")


def _get_audio_sample_rate(file: str) -> int:
    file = _clean_audio_path(file)
    try:
        return int(sf.info(file).samplerate)
    except (RuntimeError, TypeError, ValueError, OSError):
        pass
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-v",
        "error",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        file,
        "-frames:a",
        "0",
        "-f",
        "wav",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return int(sf.info(io.BytesIO(result.stdout)).samplerate)


def _get_resampler(source_sample_rate: int, target_sample_rate: int):
    key = (int(source_sample_rate), int(target_sample_rate))
    resampler = _RESAMPLER_CACHE.get(key)
    if resampler is None:
        resampler = torchaudio.transforms.Resample(
            orig_freq=source_sample_rate,
            new_freq=target_sample_rate,
            lowpass_filter_width=RESAMPLE_LOWPASS_FILTER_WIDTH,
        )
        _RESAMPLER_CACHE[key] = resampler
    return resampler


def _resample_audio(
    audio: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
    resampler=None,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if not audio.flags.c_contiguous or not audio.flags.writeable:
        audio = np.array(audio, dtype=np.float32, copy=True, order="C")
    if source_sample_rate == target_sample_rate:
        return audio
    if resampler is None:
        resampler = _get_resampler(source_sample_rate, target_sample_rate)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    with torch.inference_mode():
        output = resampler(waveform).squeeze(0)
    return output.contiguous().numpy()


def load_audio_ffmpeg(file: str, sample_rate: int) -> np.ndarray:
    file = _clean_audio_path(file)
    source_sample_rate = _get_audio_sample_rate(file)
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        file,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "pipe:1",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    return _resample_audio(audio, source_sample_rate, sample_rate)


def load_audio_ffmpeg_segment(
    file: str, sample_rate: int, start_s: float, duration_s: float
) -> np.ndarray:
    start_s = max(0.0, start_s)
    end_s = start_s + max(0.0, duration_s)
    with FFmpegAudioStreamReader(file, sample_rate) as reader:
        return reader.read_segment(start_s, end_s)


def iter_audio_ffmpeg(file: str, sample_rate: int, block_seconds: float):
    file = _clean_audio_path(file)
    block_samples = max(1, int(round(sample_rate * block_seconds)))
    block_bytes = block_samples * np.dtype(np.float32).itemsize
    command = [
        _ffmpeg_path(),
        "-nostdin",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-i",
        file,
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=block_bytes,
    )
    try:
        pending = bytearray()
        while True:
            while len(pending) < block_bytes:
                data = process.stdout.read(block_bytes - len(pending))
                if not data:
                    break
                pending.extend(data)
            if not pending:
                break
            usable = len(pending) - (len(pending) % np.dtype(np.float32).itemsize)
            if usable:
                yield np.frombuffer(pending[:usable], dtype=np.float32).copy()
            pending.clear()
            if usable < block_bytes:
                break
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.poll() is None:
            process.kill()
            process.wait()


class FFmpegAudioStreamReader:
    def __init__(self, file: str, sample_rate: int):
        self.source_sample_rate = _get_audio_sample_rate(file)
        self.target_sample_rate = sample_rate
        self.resampler = None
        if self.source_sample_rate != self.target_sample_rate:
            self.resampler = _get_resampler(
                self.source_sample_rate, self.target_sample_rate
            )
        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start = 0
        command = [
            _ffmpeg_path(),
            "-nostdin",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
            _clean_audio_path(file),
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "pipe:1",
        ]
        block_bytes = (
            int(self.source_sample_rate * AUTOMATIC_DECODE_BLOCK_SECONDS) * 4
        )
        self.command = command
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=block_bytes,
        )

    def _read_samples(self, count: int) -> np.ndarray:
        if count <= 0:
            return np.empty(0, dtype=np.float32)
        target_bytes = count * np.dtype(np.float32).itemsize
        data = bytearray()
        while len(data) < target_bytes:
            chunk = self.process.stdout.read(target_bytes - len(data))
            if not chunk:
                break
            data.extend(chunk)
        usable = len(data) - (len(data) % np.dtype(np.float32).itemsize)
        if usable < target_bytes:
            return_code = self.process.wait()
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, self.command)
        if usable == 0:
            return np.empty(0, dtype=np.float32)
        return np.frombuffer(data[:usable], dtype=np.float32).copy()

    def _discard_to(self, target: int):
        buffer_end = self.buffer_start + len(self.buffer)
        if target <= buffer_end:
            offset = max(0, target - self.buffer_start)
            self.buffer = self.buffer[offset:]
            self.buffer_start += offset
            return
        self.buffer = np.empty(0, dtype=np.float32)
        self.buffer_start = buffer_end
        while self.buffer_start < target:
            discarded = self._read_samples(target - self.buffer_start)
            if discarded.size == 0:
                break
            self.buffer_start += len(discarded)

    def read_segment(self, start_s: float, end_s: float) -> np.ndarray:
        target_start = max(0, int(round(start_s * self.target_sample_rate)))
        target_end = max(
            target_start, int(round(end_s * self.target_sample_rate))
        )
        phase_period = self.source_sample_rate // math.gcd(
            self.source_sample_rate, self.target_sample_rate
        )
        context = int(
            round(self.source_sample_rate * RESAMPLE_STREAM_CONTEXT_SECONDS)
        )
        source_start = max(
            0, int(math.floor(start_s * self.source_sample_rate)) - context
        )
        source_start -= source_start % phase_period
        source_end = int(math.ceil(end_s * self.source_sample_rate)) + context
        source_end = (
            (source_end + phase_period - 1) // phase_period * phase_period
        )
        if source_start < self.buffer_start:
            raise ValueError("FFmpeg stream segments must be read in start-time order")
        self._discard_to(source_start)
        if self.buffer_start < source_start:
            return np.empty(0, dtype=np.float32)
        required = source_end - self.buffer_start
        while len(self.buffer) < required:
            current = self._read_samples(required - len(self.buffer))
            if current.size == 0:
                break
            self.buffer = np.concatenate((self.buffer, current))
        audio = self.buffer[: source_end - source_start].copy()
        resampled = _resample_audio(
            audio,
            self.source_sample_rate,
            self.target_sample_rate,
            self.resampler,
        )
        global_target_start = (
            source_start * self.target_sample_rate // self.source_sample_rate
        )
        local_start = target_start - global_target_start
        local_end = target_end - global_target_start
        return resampled[local_start:local_end].copy()

    def close(self):
        if self.process.stdout is not None:
            self.process.stdout.close()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def iter_resampled_audio_ffmpeg(
    file: str, sample_rate: int, block_seconds: float = SIMPLE_STREAM_BLOCK_SECONDS
):
    block_frames = max(1, int(round(sample_rate * block_seconds)))
    start_frame = 0
    with FFmpegAudioStreamReader(file, sample_rate) as reader:
        while True:
            block = reader.read_segment(
                start_frame / sample_rate,
                (start_frame + block_frames) / sample_rate,
            )
            if block.size == 0:
                break
            yield block
            start_frame += len(block)
            if len(block) < block_frames:
                break


class SequentialAudioReader:
    def __init__(self, blocks):
        self.blocks = iter(blocks)
        self.block = np.empty(0, dtype=np.float32)
        self.block_offset = 0
        self.position = 0
        self.finished = False

    def _advance_block(self):
        if self.finished:
            return False
        try:
            self.block = np.asarray(next(self.blocks), dtype=np.float32)
        except StopIteration:
            self.block = np.empty(0, dtype=np.float32)
            self.finished = True
            return False
        self.block_offset = 0
        return self.block.size > 0

    def discard_to(self, target: int):
        if target < self.position:
            raise ValueError("Audio stream cannot move backwards")
        remaining = target - self.position
        while remaining > 0:
            if self.block_offset >= len(self.block) and not self._advance_block():
                break
            available = len(self.block) - self.block_offset
            take = min(remaining, available)
            self.block_offset += take
            self.position += take
            remaining -= take

    def iter_read(self, count: int):
        remaining = max(0, int(count))
        while remaining > 0:
            if self.block_offset >= len(self.block) and not self._advance_block():
                break
            available = len(self.block) - self.block_offset
            take = min(remaining, available)
            start = self.block_offset
            self.block_offset += take
            self.position += take
            remaining -= take
            yield self.block[start : start + take]

    def read(self, count: int) -> np.ndarray:
        parts = list(self.iter_read(count))
        if not parts:
            return np.empty(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0].copy()
        return np.concatenate(parts)


def iter_high_pass_audio(blocks, b_high: np.ndarray, a_high: np.ndarray):
    state = np.zeros(max(len(a_high), len(b_high)) - 1, dtype=np.float64)
    for block in blocks:
        filtered, state = signal.lfilter(b_high, a_high, block, zi=state)
        yield filtered
