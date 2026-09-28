import concurrent.futures
import json
import os
import shutil

import numpy as np
import soundfile as sf
from scipy.io import wavfile

from rvc.train.preprocess.constants import (
    AUDIO_WRITE_PENDING_MULTIPLIER,
    FLAC_COMPRESSION_LEVEL,
    MINIMUM_OUTPUT_AUDIO_SECONDS,
    SUPPORTED_DATASET_FORMATS,
    VALIDATION_AUDIO_EXTENSIONS,
)


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
