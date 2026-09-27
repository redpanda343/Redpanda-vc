import concurrent.futures
import io
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import sys
import time

def strtobool(val):
    """Convert a string representation of truth to a bool."""
    return val.lower() in ("yes", "true", "t", "y", "1")


import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

now_directory = os.getcwd()
sys.path.append(now_directory)

import logging

from rvc.train.preprocess.slicer import (
    Slicer,
    fireredvad_cuda_available,
    shutdown_fireredvad_gpu,
)
from rvc.train.preprocess.rms_slicer import Slicer as AutomaticSlicer

logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

OVERLAP = 0.3
PERCENTAGE = 3.0
MAX_AMPLITUDE = 0.9
ALPHA = 0.75
POST_NORMALIZATION_MAX_GAIN = 4.0
HIGH_PASS_CUTOFF = 20
SAMPLE_RATE_16K = 16000
MINIMUM_AUTOMATIC_SOURCE_AUDIO_SECONDS = 3.0
MINIMUM_OUTPUT_AUDIO_SECONDS = 1.0
AUTOMATIC_DECODE_BLOCK_SECONDS = 60.0
SUPPORTED_DATASET_FORMATS = {"wav", "wav_float32", "flac"}
VALIDATION_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")
AUDIO_WRITE_MAX_WORKERS = 8
AUDIO_WRITE_PENDING_MULTIPLIER = 2
GPU_PREPROCESS_MAX_WORKERS = 4
PROCESS_PENDING_MULTIPLIER = 2
FLAC_COMPRESSION_LEVEL = 0.0
RESAMPLE_LOWPASS_FILTER_WIDTH = 128
RESAMPLE_STREAM_CONTEXT_SECONDS = 1.0
SIMPLE_SILENCE_THRESHOLD_DB = -45.0
SIMPLE_MIN_SILENCE_SECONDS = 0.3
SIMPLE_TRUNCATE_TO_SECONDS = 0.3
SIMPLE_SILENCE_COMPRESS_PERCENT = 50.0
SIMPLE_BLEND_FRAMES = 100
SIMPLE_STREAM_BLOCK_SECONDS = 60.0
SIMPLE_STREAM_THRESHOLD_BYTES = 256 * 1024 * 1024
_RESAMPLER_CACHE = {}


def normalize_dataset_format(dataset_format: str) -> str:
    normalized_format = str(dataset_format).strip().lower()
    if normalized_format == "wav 16-bit":
        normalized_format = "wav"
    if normalized_format == "wav 32-bit float":
        normalized_format = "wav_float32"
    if normalized_format not in SUPPORTED_DATASET_FORMATS:
        raise ValueError(
            f"Unsupported dataset format '{dataset_format}'. Expected WAV 16-bit, WAV 32-bit float, or FLAC."
        )
    return normalized_format


def stage_validation_audio(input_root: str, exp_dir: str) -> int:
    validation_sources = [
        os.path.join(input_root, name)
        for name in os.listdir(input_root)
        if name.lower() == "validation"
        and os.path.isdir(os.path.join(input_root, name))
    ]
    if len(validation_sources) > 1:
        raise RuntimeError("The dataset contains multiple validation folders")

    validation_target = os.path.join(exp_dir, "validation")
    if not validation_sources:
        if os.path.isdir(validation_target):
            shutil.rmtree(validation_target)
        return 0

    validation_source = validation_sources[0]
    staging_dir = os.path.join(exp_dir, f"validation.{os.getpid()}.tmp")
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    audio_target = os.path.join(staging_dir, "audio")
    os.makedirs(audio_target, exist_ok=True)
    copied = 0
    try:
        for root, directories, filenames in os.walk(validation_source):
            directories.sort()
            relative_root = os.path.relpath(root, validation_source)
            destination_root = (
                audio_target
                if relative_root == "."
                else os.path.join(audio_target, relative_root)
            )
            os.makedirs(destination_root, exist_ok=True)
            for filename in sorted(filenames):
                if not filename.lower().endswith(VALIDATION_AUDIO_EXTENSIONS):
                    continue
                shutil.copy2(
                    os.path.join(root, filename),
                    os.path.join(destination_root, filename),
                )
                copied += 1
        if copied == 0:
            raise RuntimeError("The validation folder contains no supported audio files")
        if os.path.isdir(validation_target):
            shutil.rmtree(validation_target)
        os.replace(staging_dir, validation_target)
    except Exception:
        if os.path.isdir(staging_dir):
            shutil.rmtree(staging_dir)
        raise
    return copied


def write_training_audio(
    directory: str,
    stem: str,
    sample_rate: int,
    audio: np.ndarray,
    dataset_format: str,
):
    """Write a processed training slice without changing the existing WAV path."""
    audio = np.asarray(audio, dtype=np.float32)
    if not np.all(np.isfinite(audio)):
        raise ValueError(
            f"Cannot write non-finite audio samples to {stem}.{dataset_format}"
        )
    if dataset_format == "wav":
        wavfile.write(
            os.path.join(directory, f"{stem}.wav"),
            sample_rate,
            (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16),
        )
        return
    if dataset_format == "wav_float32":
        sf.write(
            os.path.join(directory, f"{stem}.wav"),
            np.clip(audio, -1.0, 1.0),
            sample_rate,
            format="WAV",
            subtype="FLOAT",
        )
        return

    sf.write(
        os.path.join(directory, f"{stem}.flac"),
        np.clip(audio, -1.0, 1.0),
        sample_rate,
        format="FLAC",
        subtype="PCM_24",
        compression_level=FLAC_COMPRESSION_LEVEL,
    )


class BoundedAudioWriter:
    def __init__(self, max_workers: int):
        self.max_workers = max(1, int(max_workers))
        self.max_pending = self.max_workers * AUDIO_WRITE_PENDING_MULTIPLIER
        self.executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
            if self.max_workers > 1
            else None
        )
        self.pending = set()
        self.skipped_short = 0

    def submit(
        self,
        directory: str,
        stem: str,
        sample_rate: int,
        audio: np.ndarray,
        dataset_format: str,
    ):
        if len(audio) < round(sample_rate * MINIMUM_OUTPUT_AUDIO_SECONDS):
            self.skipped_short += 1
            return 1
        args = (directory, stem, sample_rate, audio, dataset_format)
        if self.executor is None:
            write_training_audio(*args)
            return 0
        self.pending.add(self.executor.submit(write_training_audio, *args))
        if len(self.pending) >= self.max_pending:
            done, self.pending = concurrent.futures.wait(
                self.pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                future.result()
        return 0

    def close(self):
        if self.executor is None:
            return
        try:
            for future in concurrent.futures.as_completed(self.pending):
                future.result()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.pending.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        elif self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.pending.clear()
        return False


def clear_flac_preprocess_artifacts(exp_dir: str):
    """Remove only FLAC-derived caches so a FLAC reprocess cannot reuse stale data."""
    patterns_by_directory = {
        "sliced_audios": (".flac", ".spec.pt"),
        "sliced_audios_16k": (".flac",),
        "f0": (".flac.npy",),
        "f0_voiced": (".flac.npy",),
        "extracted": (".flac.npy",),
    }
    for directory_name, suffixes in patterns_by_directory.items():
        directory = os.path.join(exp_dir, directory_name)
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if filename.lower().endswith(suffixes):
                os.remove(os.path.join(directory, filename))

    filelist_path = os.path.join(exp_dir, "filelist.txt")
    if os.path.isfile(filelist_path):
        os.remove(filelist_path)


def clear_simple_preprocess_artifacts(exp_dir: str):
    patterns_by_directory = {
        "sliced_audios": (".wav", ".flac", ".spec.pt"),
        "sliced_audios_16k": (".wav", ".flac"),
        "f0": (".wav.npy", ".flac.npy"),
        "f0_voiced": (".wav.npy", ".flac.npy"),
        "extracted": (".npy",),
    }
    for directory_name, suffixes in patterns_by_directory.items():
        directory = os.path.join(exp_dir, directory_name)
        if not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if filename.lower().endswith(suffixes):
                os.remove(os.path.join(directory, filename))

    filelist_path = os.path.join(exp_dir, "filelist.txt")
    if os.path.isfile(filelist_path):
        os.remove(filelist_path)


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


def truncate_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = SIMPLE_SILENCE_THRESHOLD_DB,
    minimum_silence: float = SIMPLE_MIN_SILENCE_SECONDS,
    truncate_to: float = SIMPLE_TRUNCATE_TO_SECONDS,
    blend_frames: int = SIMPLE_BLEND_FRAMES,
    action: str = "truncate",
    compress_percent: float = SIMPLE_SILENCE_COMPRESS_PERCENT,
) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio

    action = normalize_silence_action(action)
    compress_percent = validate_silence_compress_percent(compress_percent)
    threshold = 10.0 ** (threshold_db / 20.0)
    silent = np.abs(audio) < threshold
    boundaries = np.flatnonzero(
        np.diff(np.pad(silent.astype(np.int8), (1, 1)))
    )
    if boundaries.size == 0:
        return audio

    minimum_frames = max(1, int(round(minimum_silence * sample_rate)))
    truncate_frames = max(0, int(round(truncate_to * sample_rate)))
    cuts = []
    if action == "truncate":
        for start, end in boundaries.reshape(-1, 2):
            silence_frames = int(end - start)
            if silence_frames < minimum_frames:
                continue
            output_frames = min(truncate_frames, silence_frames)
            cut_frames = silence_frames - output_frames
            if cut_frames <= 0:
                continue
            cut_start = int(start + output_frames // 2)
            cuts.append((cut_start, cut_start + cut_frames))
    else:
        for start, end in boundaries.reshape(-1, 2):
            silence_frames = int(end - start)
            if silence_frames < minimum_frames:
                continue
            excess_frames = silence_frames - minimum_frames
            output_frames = minimum_frames + int(
                round(excess_frames * compress_percent / 100.0)
            )
            cut_frames = silence_frames - output_frames
            if cut_frames <= 0:
                continue
            cut_start = int(start + output_frames // 2)
            cuts.append((cut_start, cut_start + cut_frames))

    if not cuts:
        return audio

    parts = []
    cursor = 0
    for cut_start, cut_end in cuts:
        splice_frames = min(
            blend_frames,
            cut_start * 2,
            (len(audio) - cut_end) * 2,
        )
        if splice_frames > 0:
            half_blend = splice_frames // 2
            blend_start = cut_start - half_blend
            right_start = cut_end - half_blend
            left = audio[blend_start : blend_start + splice_frames]
            right = audio[right_start : right_start + splice_frames]
            weights = np.arange(splice_frames, dtype=np.float32) / splice_frames
            blended = left * (1.0 - weights) + right * weights
            parts.append(audio[cursor:blend_start])
            parts.append(blended)
            cursor = right_start + splice_frames
        else:
            parts.append(audio[cursor:cut_start])
            cursor = cut_end

    parts.append(audio[cursor:])
    return np.concatenate(parts)


def normalize_silence_action(action: str) -> str:
    normalized = str(action).strip().lower()
    if normalized not in {"truncate", "compress"}:
        raise ValueError(f"Unsupported silence action: {action}")
    return normalized


def validate_silence_compress_percent(compress_percent: float) -> float:
    compress_percent = float(compress_percent)
    if not 0.0 <= compress_percent <= 99.9:
        raise ValueError("Silence compression must be between 0.0 and 99.9 percent")
    return compress_percent


def silence_cut(
    start: int,
    end: int,
    minimum_frames: int,
    truncate_frames: int,
    action: str,
    compress_percent: float,
):
    silence_frames = end - start
    if silence_frames < minimum_frames:
        return None
    if action == "compress":
        excess_frames = silence_frames - minimum_frames
        output_frames = minimum_frames + int(
            round(excess_frames * compress_percent / 100.0)
        )
    else:
        output_frames = min(truncate_frames, silence_frames)
    output_frames = min(output_frames, silence_frames)
    cut_frames = silence_frames - output_frames
    if cut_frames <= 0:
        return None
    cut_start = start + output_frames // 2
    return cut_start, cut_start + cut_frames


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


def find_streaming_silence_cuts(
    blocks,
    sample_rate: int,
    threshold_db: float,
    minimum_silence: float,
    truncate_to: float,
    action: str,
    compress_percent: float,
):
    action = normalize_silence_action(action)
    compress_percent = validate_silence_compress_percent(compress_percent)
    threshold = 10.0 ** (threshold_db / 20.0)
    minimum_frames = max(1, int(round(minimum_silence * sample_rate)))
    truncate_frames = max(0, int(round(truncate_to * sample_rate)))
    previous_silent = False
    silence_start = None
    total_frames = 0
    cuts = []
    for block in blocks:
        block = np.asarray(block, dtype=np.float32)
        if block.size == 0:
            continue
        silent = np.abs(block) < threshold
        transitions = np.flatnonzero(
            np.diff(silent.astype(np.int8), prepend=np.int8(previous_silent))
        )
        for index in transitions:
            position = total_frames + int(index)
            if silent[index]:
                silence_start = position
            elif silence_start is not None:
                if position - silence_start >= minimum_frames:
                    cut = silence_cut(
                        silence_start,
                        position,
                        minimum_frames,
                        truncate_frames,
                        action,
                        compress_percent,
                    )
                    if cut is not None:
                        cuts.append(cut)
                silence_start = None
        previous_silent = bool(silent[-1])
        total_frames += len(block)
    if previous_silent and silence_start is not None:
        if total_frames - silence_start >= minimum_frames:
            cut = silence_cut(
                silence_start,
                total_frames,
                minimum_frames,
                truncate_frames,
                action,
                compress_percent,
            )
            if cut is not None:
                cuts.append(cut)
    return cuts, total_frames


def iter_audio_with_silence_cuts(
    blocks, cuts, total_frames: int, blend_frames: int = SIMPLE_BLEND_FRAMES
):
    reader = SequentialAudioReader(blocks)
    for cut_start, cut_end in cuts:
        splice_frames = min(
            blend_frames,
            cut_start * 2,
            (total_frames - cut_end) * 2,
        )
        half_blend = splice_frames // 2
        blend_start = cut_start - half_blend
        yield from reader.iter_read(blend_start - reader.position)
        if splice_frames > 0:
            left = reader.read(splice_frames)
            right_start = cut_end - half_blend
            if right_start < reader.position:
                overlap_start = right_start - blend_start
                right_prefix = left[overlap_start:]
                right_suffix = reader.read(splice_frames - len(right_prefix))
                right = np.concatenate((right_prefix, right_suffix))
            else:
                reader.discard_to(right_start)
                right = reader.read(splice_frames)
            weights = (
                np.arange(splice_frames, dtype=np.float32) / splice_frames
            )
            yield left * (1.0 - weights) + right * weights
        else:
            reader.discard_to(cut_end)
    yield from reader.iter_read(total_frames - reader.position)


def iter_high_pass_audio(blocks, b_high: np.ndarray, a_high: np.ndarray):
    state = np.zeros(max(len(a_high), len(b_high)) - 1, dtype=np.float64)
    for block in blocks:
        filtered, state = signal.lfilter(b_high, a_high, block, zi=state)
        yield filtered


class PreProcess:
    def __init__(
        self,
        sr: int,
        exp_dir: str,
        dataset_format: str = "wav",
        use_fireredvad_gpu: bool = False,
    ):
        self.post_normalization_slicer = Slicer(
            sr=sr,
            use_gpu=use_fireredvad_gpu,
        )
        self.automatic_slicer = AutomaticSlicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sr = sr
        self.b_high, self.a_high = signal.butter(
            N=5, Wn=HIGH_PASS_CUTOFF, btype="high", fs=self.sr
        )
        self.exp_dir = exp_dir
        self.dataset_format = normalize_dataset_format(dataset_format)
        self.audio_write_workers = 1
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)

    def _normalize_audio(self, audio: np.ndarray):
        tmp_max = np.abs(audio).max()
        if tmp_max > 2.5:
            return None
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    @staticmethod
    def _post_normalization_gain(voice_peak: float):
        if not np.isfinite(voice_peak) or voice_peak <= 0:
            return 1.0
        return min(MAX_AMPLITUDE / voice_peak, POST_NORMALIZATION_MAX_GAIN)

    def _detect_post_normalization_gain(self, audio: np.ndarray):
        voiced_chunks = self.post_normalization_slicer.slice(audio)
        if not voiced_chunks:
            return 1.0
        voice_peak = max(float(np.max(np.abs(chunk))) for chunk in voiced_chunks)
        return self._post_normalization_gain(voice_peak)

    def _peak_normalize_audio(self, audio: np.ndarray, source_gain: float):
        if audio.size == 0:
            return audio
        peak = np.abs(audio).max()
        if not np.isfinite(peak) or peak > 2.5:
            return None
        if peak == 0:
            return audio
        gain = min(source_gain, MAX_AMPLITUDE / peak)
        return audio * gain

    def process_audio_segment(
        self,
        normalized_audio: np.ndarray,
        sid: int,
        idx0: int,
        idx1: int,
        normalization_mode: str,
        normalization_gain: float = 1.0,
        writer: BoundedAudioWriter | None = None,
    ):
        if normalized_audio is None:
            print(f"{sid}-{idx0}-{idx1}-filtered")
            return
        if normalization_mode == "post":
            normalized_audio = self._peak_normalize_audio(
                normalized_audio, normalization_gain
            )
        args = (
            self.gt_wavs_dir,
            f"{sid}_{idx0}_{idx1}",
            self.sr,
            normalized_audio,
            self.dataset_format,
        )
        if writer is None:
            if len(normalized_audio) < round(
                self.sr * MINIMUM_OUTPUT_AUDIO_SECONDS
            ):
                return 1
            write_training_audio(*args)
            return 0
        return writer.submit(*args)

    def simple_cut(
        self,
        audio: np.ndarray,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        normalization_gain: float = 1.0,
    ):
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        with BoundedAudioWriter(self.audio_write_workers) as writer:
            i = 0
            while i < len(audio):
                chunk = audio[i : i + chunk_length]
                if normalization_mode == "post":
                    chunk = self._peak_normalize_audio(chunk, normalization_gain)
                writer.submit(
                    self.gt_wavs_dir,
                    f"{sid}_{idx0}_{i // (chunk_length - overlap_length)}",
                    self.sr,
                    chunk,
                    self.dataset_format,
                )
                i += chunk_length - overlap_length
        return writer.skipped_short

    def simple_cut_stream(
        self,
        blocks,
        sid: int,
        idx0: int,
        chunk_len: float,
        overlap_len: float,
    ):
        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        step_length = chunk_length - overlap_length
        buffer = np.empty(0, dtype=np.float32)
        buffer_start = 0
        next_start = 0
        with BoundedAudioWriter(self.audio_write_workers) as writer:
            for block in blocks:
                block = np.asarray(block)
                if block.size == 0:
                    continue
                buffer = (
                    np.concatenate((buffer, block))
                    if buffer.size
                    else block.copy()
                )
                buffer_end = buffer_start + len(buffer)
                while next_start + chunk_length <= buffer_end:
                    local_start = next_start - buffer_start
                    chunk = buffer[local_start : local_start + chunk_length].copy()
                    writer.submit(
                        self.gt_wavs_dir,
                        f"{sid}_{idx0}_{next_start // step_length}",
                        self.sr,
                        chunk,
                        self.dataset_format,
                    )
                    next_start += step_length
                    drop_frames = next_start - buffer_start
                    buffer = buffer[drop_frames:]
                    buffer_start = next_start
                    buffer_end = buffer_start + len(buffer)
            if next_start < buffer_start + len(buffer):
                local_start = next_start - buffer_start
                writer.submit(
                    self.gt_wavs_dir,
                    f"{sid}_{idx0}_{next_start // step_length}",
                    self.sr,
                    buffer[local_start:].copy(),
                    self.dataset_format,
                )
        return writer.skipped_short

    def should_stream_simple_audio(
        self,
        path: str,
        noise_reduction: bool,
        normalization_mode: str,
    ) -> bool:
        if noise_reduction or normalization_mode != "none":
            return False
        try:
            duration = float(sf.info(_clean_audio_path(path)).duration)
        except (RuntimeError, TypeError, ValueError, OSError):
            return False
        return duration * self.sr * np.dtype(np.float32).itemsize >= (
            SIMPLE_STREAM_THRESHOLD_BYTES
        )

    def process_simple_audio_streaming(
        self,
        path: str,
        idx0: int,
        sid: int,
        process_effects: bool,
        chunk_len: float,
        overlap_len: float,
        truncate_silence_threshold_db: float,
        truncate_silence_to_seconds: float,
        truncate_silence_minimum_seconds: float,
        truncate_silence_action: str,
        truncate_silence_compress_percent: float,
    ):
        cuts, total_frames = find_streaming_silence_cuts(
            iter_resampled_audio_ffmpeg(path, self.sr),
            self.sr,
            truncate_silence_threshold_db,
            truncate_silence_minimum_seconds,
            truncate_silence_to_seconds,
            truncate_silence_action,
            truncate_silence_compress_percent,
        )
        blocks = iter_audio_with_silence_cuts(
            iter_resampled_audio_ffmpeg(path, self.sr),
            cuts,
            total_frames,
        )
        if process_effects:
            blocks = iter_high_pass_audio(blocks, self.b_high, self.a_high)
        skipped_short = self.simple_cut_stream(
            blocks,
            sid,
            idx0,
            chunk_len,
            overlap_len,
        )
        return total_frames / self.sr, skipped_short

    def process_simple_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
        truncate_silence_enabled: bool = False,
        truncate_silence_threshold_db: float = SIMPLE_SILENCE_THRESHOLD_DB,
        truncate_silence_to_seconds: float = SIMPLE_TRUNCATE_TO_SECONDS,
        truncate_silence_minimum_seconds: float = SIMPLE_MIN_SILENCE_SECONDS,
        truncate_silence_action: str = "truncate",
        truncate_silence_compress_percent: float = SIMPLE_SILENCE_COMPRESS_PERCENT,
    ):
        if truncate_silence_enabled and self.should_stream_simple_audio(
            path,
            noise_reduction,
            normalization_mode,
        ):
            return self.process_simple_audio_streaming(
                path,
                idx0,
                sid,
                process_effects,
                chunk_len,
                overlap_len,
                truncate_silence_threshold_db,
                truncate_silence_to_seconds,
                truncate_silence_minimum_seconds,
                truncate_silence_action,
                truncate_silence_compress_percent,
            )
        audio = load_audio_ffmpeg(path, self.sr)
        audio_length = len(audio) / self.sr
        if truncate_silence_enabled:
            audio = truncate_silence(
                audio,
                self.sr,
                threshold_db=truncate_silence_threshold_db,
                minimum_silence=truncate_silence_minimum_seconds,
                truncate_to=truncate_silence_to_seconds,
                action=truncate_silence_action,
                compress_percent=truncate_silence_compress_percent,
            )
        audio = self._prepare_audio(
            audio,
            process_effects,
            noise_reduction,
            reduction_strength,
            normalization_mode,
        )
        normalization_gain = 1.0
        if normalization_mode == "post" and audio is not None:
            normalization_gain = self._detect_post_normalization_gain(audio)
        skipped_short = self.simple_cut(
            audio,
            sid,
            idx0,
            chunk_len,
            overlap_len,
            normalization_mode,
            normalization_gain,
        )
        return audio_length, skipped_short

    def _prepare_audio(
        self,
        audio: np.ndarray,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        normalization_mode: str,
    ):
        if process_effects:
            audio = signal.lfilter(self.b_high, self.a_high, audio)
        if normalization_mode == "pre":
            audio = self._normalize_audio(audio)
        if noise_reduction and audio is not None and audio.size:
            audio = nr.reduce_noise(
                y=audio, sr=self.sr, prop_decrease=reduction_strength
            )
        return audio

    def _process_automatic(
        self,
        path: str,
        idx0: int,
        sid: int,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        normalization_mode: str,
    ):
        audio = load_audio_ffmpeg(path, self.sr)
        duration_s = librosa.get_duration(y=audio, sr=self.sr)
        if duration_s < MINIMUM_AUTOMATIC_SOURCE_AUDIO_SECONDS:
            return 0.0, 0
        audio = self._prepare_audio(
            audio,
            process_effects,
            noise_reduction,
            reduction_strength,
            normalization_mode,
        )
        if audio is None:
            return duration_s, 0
        normalization_gain = 1.0
        if normalization_mode == "post":
            normalization_gain = self._detect_post_normalization_gain(audio)

        segments = self.automatic_slicer.slice(audio)
        idx1 = 0
        step_samples = int(self.sr * (PERCENTAGE - OVERLAP))
        chunk_samples = int(self.sr * PERCENTAGE)
        long_tail_samples = int(self.sr * (PERCENTAGE + OVERLAP))
        with BoundedAudioWriter(self.audio_write_workers) as writer:
            for segment in segments:
                start = 0
                while start < len(segment):
                    if len(segment) - start > long_tail_samples:
                        chunk = segment[start : start + chunk_samples]
                    else:
                        chunk = segment[start:]
                    self.process_audio_segment(
                        chunk,
                        sid,
                        idx0,
                        idx1,
                        normalization_mode,
                        normalization_gain,
                        writer,
                    )
                    idx1 += 1
                    if len(segment) - start <= long_tail_samples:
                        break
                    start += step_samples
        return duration_s, writer.skipped_short

    def process_audio(
        self,
        path: str,
        idx0: int,
        sid: int,
        cut_preprocess: str,
        process_effects: bool,
        noise_reduction: bool,
        reduction_strength: float,
        chunk_len: float,
        overlap_len: float,
        normalization_mode: str,
    ):
        audio_length = 0
        skipped_short = 0
        try:
            if cut_preprocess == "Automatic":
                return self._process_automatic(
                    path,
                    idx0,
                    sid,
                    process_effects,
                    noise_reduction,
                    reduction_strength,
                    normalization_mode,
                )

            audio = load_audio_ffmpeg(path, self.sr)
            audio_length = librosa.get_duration(y=audio, sr=self.sr)
            audio = self._prepare_audio(
                audio,
                process_effects,
                noise_reduction,
                reduction_strength,
                normalization_mode,
            )
            normalization_gain = 1.0
            if normalization_mode == "post" and audio is not None:
                normalization_gain = self._detect_post_normalization_gain(audio)
            if cut_preprocess == "Skip":
                # no cutting
                skipped_short = self.process_audio_segment(
                    audio,
                    sid,
                    idx0,
                    0,
                    normalization_mode,
                    normalization_gain,
                )
            elif cut_preprocess == "Simple":
                # simple
                skipped_short = self.simple_cut(
                    audio,
                    sid,
                    idx0,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                    normalization_gain,
                )
        except Exception as error:
            print(f"Error processing audio: {error}")
            if cut_preprocess == "Automatic" or self.dataset_format in {
                "flac",
                "wav_float32",
            }:
                raise
        return audio_length, skipped_short


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def save_dataset_duration(file_path, dataset_duration, dataset_format="wav"):
    normalized_format = normalize_dataset_format(dataset_format)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    formatted_duration = format_duration(dataset_duration)
    new_data = {
        "total_dataset_duration": formatted_duration,
        "total_seconds": dataset_duration,
        "dataset_format": "flac" if normalized_format == "flac" else "wav",
        "dataset_subtype": {
            "wav": "PCM_16",
            "wav_float32": "FLOAT",
            "flac": "PCM_24",
        }[normalized_format],
    }
    data.update(new_data)

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


_PROCESS_PREPROCESSOR = None


def initialize_preprocess_worker(
    sr, exp_dir, dataset_format, audio_write_workers, torch_threads
):
    global _PROCESS_PREPROCESSOR
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _PROCESS_PREPROCESSOR = PreProcess(sr, exp_dir, dataset_format, False)
    _PROCESS_PREPROCESSOR.audio_write_workers = max(1, int(audio_write_workers))


def process_audio_wrapper(args):
    (
        pp,
        file,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    ) = args
    if pp is None:
        pp = _PROCESS_PREPROCESSOR
        if pp is None:
            raise RuntimeError("Preprocess worker is not initialized")
    file_path, idx0, sid = file
    return pp.process_audio(
        file_path,
        idx0,
        sid,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
    )


def process_simple_audio_wrapper(args):
    (
        pp,
        path,
        idx0,
        sid,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        truncate_silence_enabled,
        truncate_silence_threshold_db,
        truncate_silence_to_seconds,
        truncate_silence_minimum_seconds,
        truncate_silence_action,
        truncate_silence_compress_percent,
    ) = args
    if pp is None:
        pp = _PROCESS_PREPROCESSOR
        if pp is None:
            raise RuntimeError("Preprocess worker is not initialized")
    return pp.process_simple_audio(
        path,
        idx0,
        sid,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        truncate_silence_enabled,
        truncate_silence_threshold_db,
        truncate_silence_to_seconds,
        truncate_silence_minimum_seconds,
        truncate_silence_action,
        truncate_silence_compress_percent,
    )


def preprocess_training_set(
    input_root: str,
    sr: int,
    num_processes: int,
    exp_dir: str,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    reduction_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str,
    dataset_format: str = "wav",
    truncate_silence_enabled: bool = False,
    truncate_silence_threshold_db: float = SIMPLE_SILENCE_THRESHOLD_DB,
    truncate_silence_to_seconds: float = SIMPLE_TRUNCATE_TO_SECONDS,
    truncate_silence_minimum_seconds: float = SIMPLE_MIN_SILENCE_SECONDS,
    truncate_silence_action: str = "truncate",
    truncate_silence_compress_percent: float = SIMPLE_SILENCE_COMPRESS_PERCENT,
):
    if not os.path.exists(input_root):
        print(f"The dataset path does not exist: '{input_root}'.")
        sys.exit(1)

    if not os.path.isdir(input_root):
        print(f"The dataset path is not a directory: '{input_root}'.")
        sys.exit(1)
    start_time = time.time()
    dataset_format = normalize_dataset_format(dataset_format)
    validation_count = stage_validation_audio(input_root, exp_dir)
    if validation_count:
        print(
            f"Copied {validation_count} external validation audio file(s) to "
            f"{os.path.join(exp_dir, 'validation', 'audio')}."
        )

    files = []
    idx = 0

    for root, directories, filenames in os.walk(input_root):
        if root == input_root:
            directories[:] = [
                directory
                for directory in directories
                if directory.lower() != "validation"
            ]
        directories.sort()
        try:
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in sorted(filenames):
                if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            print(
                f'Speaker ID folder is expected to be integer, got "{os.path.basename(root)}" instead.'
            )

    # print(f"Number of files: {len(files)}")
    if len(files) == 0:
        print(
            f"No audio files found in the dataset path: '{input_root}'. Please check that the path is correct and contains valid audio files."
        )
        sys.exit(1)

    if cut_preprocess == "Simple":
        clear_simple_preprocess_artifacts(exp_dir)
    elif dataset_format == "flac":
        clear_flac_preprocess_artifacts(exp_dir)
    uses_fireredvad = normalization_mode == "post"
    use_fireredvad_gpu = uses_fireredvad and fireredvad_cuda_available()
    if use_fireredvad_gpu:
        print("FireRedVAD inference: CUDA")
    elif uses_fireredvad:
        print("FireRedVAD inference: CPU")
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = multiprocessing.cpu_count()
    available_cpus = max(1, int(available_cpus))
    active_workers = max(1, min(num_processes, len(files), available_cpus))
    if use_fireredvad_gpu:
        active_workers = min(active_workers, GPU_PREPROCESS_MAX_WORKERS)
    print(f"Starting preprocess with {active_workers} workers...")
    pp = PreProcess(sr, exp_dir, dataset_format, use_fireredvad_gpu)
    pp.audio_write_workers = (
        min(AUDIO_WRITE_MAX_WORKERS, available_cpus)
        if not use_fireredvad_gpu and active_workers == 1
        else 1
    )
    print(f"Audio output pipeline: {pp.audio_write_workers} workers per source")
    work_pp = pp if use_fireredvad_gpu else None

    if cut_preprocess == "Simple":
        work_items = [
            (
                work_pp,
                file_path,
                idx0,
                sid,
                process_effects,
                noise_reduction,
                reduction_strength,
                chunk_len,
                overlap_len,
                normalization_mode,
                truncate_silence_enabled,
                truncate_silence_threshold_db,
                truncate_silence_to_seconds,
                truncate_silence_minimum_seconds,
                truncate_silence_action,
                truncate_silence_compress_percent,
            )
            for file_path, idx0, sid in files
        ]
        worker = process_simple_audio_wrapper
    else:
        work_items = [
            (
                work_pp,
                file,
                cut_preprocess,
                process_effects,
                noise_reduction,
                reduction_strength,
                chunk_len,
                overlap_len,
                normalization_mode,
            )
            for file in files
        ]
        worker = process_audio_wrapper

    executor_class = (
        concurrent.futures.ThreadPoolExecutor
        if use_fireredvad_gpu
        else concurrent.futures.ProcessPoolExecutor
    )
    executor_kwargs = {}
    if not use_fireredvad_gpu:
        executor_kwargs = {
            "initializer": initialize_preprocess_worker,
            "initargs": (
                sr,
                exp_dir,
                dataset_format,
                pp.audio_write_workers,
                max(1, available_cpus // active_workers),
            ),
        }
    max_pending = max(active_workers, active_workers * PROCESS_PENDING_MULTIPLIER)
    audio_length = 0.0
    skipped_short_outputs = 0
    try:
        with tqdm(total=len(work_items)) as pbar:
            with executor_class(
                max_workers=active_workers, **executor_kwargs
            ) as executor:
                work_iterator = iter(work_items)
                pending = set()
                for _ in range(min(max_pending, len(work_items))):
                    pending.add(executor.submit(worker, next(work_iterator)))
                while pending:
                    done, pending = concurrent.futures.wait(
                        pending,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        result = future.result()
                        audio_length += result[0]
                        skipped_short_outputs += result[1]
                        pbar.update(1)
                        try:
                            work_item = next(work_iterator)
                        except StopIteration:
                            continue
                        pending.add(executor.submit(worker, work_item))
    finally:
        if use_fireredvad_gpu:
            shutdown_fireredvad_gpu()

    save_dataset_duration(
        os.path.join(exp_dir, "model_info.json"),
        dataset_duration=audio_length,
        dataset_format=dataset_format,
    )
    elapsed_time = time.time() - start_time
    automatic_filter = (
        f"Automatic sources under {MINIMUM_AUTOMATIC_SOURCE_AUDIO_SECONDS:.1f}s skipped; "
        if cut_preprocess == "Automatic"
        else ""
    )
    print(
        f"Preprocess completed in {elapsed_time:.2f} seconds on "
        f"{format_duration(audio_length)} seconds of audio. Short-audio filter: "
        f"{automatic_filter}{skipped_short_outputs} output slice(s) under "
        f"{MINIMUM_OUTPUT_AUDIO_SECONDS:.1f}s skipped before writing."
    )


if __name__ == "__main__":
    experiment_directory = str(sys.argv[1])
    input_root = str(sys.argv[2])
    sample_rate = int(sys.argv[3])
    num_processes = sys.argv[4]
    if num_processes.lower() == "none":
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = int(num_processes)
    cut_preprocess = str(sys.argv[5])
    process_effects = strtobool(sys.argv[6])
    noise_reduction = strtobool(sys.argv[7])
    reduction_strength = float(sys.argv[8])
    chunk_len = float(sys.argv[9])
    overlap_len = float(sys.argv[10])
    normalization_mode = str(sys.argv[11])
    dataset_format = str(sys.argv[12]) if len(sys.argv) > 12 else "WAV"
    truncate_silence_enabled = strtobool(sys.argv[13]) if len(sys.argv) > 13 else False
    truncate_silence_threshold_db = (
        float(sys.argv[14]) if len(sys.argv) > 14 else SIMPLE_SILENCE_THRESHOLD_DB
    )
    truncate_silence_to_seconds = (
        float(sys.argv[15]) if len(sys.argv) > 15 else SIMPLE_TRUNCATE_TO_SECONDS
    )
    truncate_silence_minimum_seconds = (
        float(sys.argv[16]) if len(sys.argv) > 16 else SIMPLE_MIN_SILENCE_SECONDS
    )
    truncate_silence_action = (
        str(sys.argv[17]) if len(sys.argv) > 17 else "truncate"
    )
    truncate_silence_compress_percent = (
        float(sys.argv[18])
        if len(sys.argv) > 18
        else SIMPLE_SILENCE_COMPRESS_PERCENT
    )
    preprocess_training_set(
        input_root,
        sample_rate,
        num_processes,
        experiment_directory,
        cut_preprocess,
        process_effects,
        noise_reduction,
        reduction_strength,
        chunk_len,
        overlap_len,
        normalization_mode,
        dataset_format,
        truncate_silence_enabled,
        truncate_silence_threshold_db,
        truncate_silence_to_seconds,
        truncate_silence_minimum_seconds,
        truncate_silence_action,
        truncate_silence_compress_percent,
    )
