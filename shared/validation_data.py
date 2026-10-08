import hashlib
import json
import os
import random
from contextlib import contextmanager

import numpy as np
import soundfile as sf
import torch


VALIDATION_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg")
VALIDATION_MANIFEST_VERSION = 1
MINIMUM_VALIDATION_SECONDS = 2
PREFERRED_VALIDATION_SECONDS = 10
MAXIMUM_VALIDATION_SECONDS = 20
VALIDATION_INFERENCE_BATCH_SECONDS = 40


def should_run_external_validation(reference, timbre_validator):
    return reference is not None and timbre_validator is not None


@contextmanager
def deterministic_validation_scope(seed, cuda_devices=None):
    seed = int(seed) % (2**63 - 1)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    previous_settings = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
        torch.backends.cuda.matmul.fp32_precision,
        torch.backends.cudnn.fp32_precision,
    )
    devices = list(cuda_devices or [])
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.fp32_precision = "ieee"
            try:
                yield
            finally:
                (
                    deterministic_algorithms,
                    deterministic_warn_only,
                    cudnn_benchmark,
                    cudnn_deterministic,
                    matmul_precision,
                    cudnn_precision,
                ) = previous_settings
                torch.use_deterministic_algorithms(
                    deterministic_algorithms,
                    warn_only=deterministic_warn_only,
                )
                torch.backends.cudnn.benchmark = cudnn_benchmark
                torch.backends.cudnn.deterministic = cudnn_deterministic
                torch.backends.cuda.matmul.fp32_precision = matmul_precision
                torch.backends.cudnn.fp32_precision = cudnn_precision
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def infer_validation_audio(model, inference_inputs, sample_rate, hop_length):
    sample_count = int(inference_inputs[0].size(0))
    if sample_count < 1:
        raise ValueError("No external validation inputs are available")
    maximum_frames = max(1, int(inference_inputs[1].max().item()))
    maximum_seconds = maximum_frames * int(hop_length) / int(sample_rate)
    batch_size = max(
        1,
        min(
            sample_count,
            int(VALIDATION_INFERENCE_BATCH_SECONDS // maximum_seconds),
        ),
    )
    device = inference_inputs[0].device
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    while True:
        generated = []
        try:
            for start in range(0, sample_count, batch_size):
                batch = tuple(
                    value[start : start + batch_size] for value in inference_inputs
                )
                output, *_ = model.infer(*batch)
                generated.append(output.detach().cpu())
                del output, batch
            return torch.cat(generated, dim=0)
        except RuntimeError as error:
            if (
                device.type != "cuda"
                or batch_size == 1
                or "out of memory" not in str(error).lower()
            ):
                raise
            generated.clear()
            torch.set_rng_state(cpu_rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, device)
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            print(
                "External validation generation ran out of VRAM; retrying with "
                f"batch {batch_size}."
            )


def _sort_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def build_validation_extraction_files(experiment_dir):
    validation_dir = os.path.join(experiment_dir, "validation")
    audio_dir = os.path.join(validation_dir, "audio")
    if not os.path.isdir(audio_dir):
        return [], []

    cache_dirs = {
        "f0": os.path.join(validation_dir, "f0"),
        "f0_voiced": os.path.join(validation_dir, "f0_voiced"),
        "extracted": os.path.join(validation_dir, "extracted"),
    }
    for directory in cache_dirs.values():
        os.makedirs(directory, exist_ok=True)

    audio_paths = []
    for root, directories, filenames in os.walk(audio_dir):
        directories.sort()
        for filename in sorted(filenames):
            if filename.lower().endswith(VALIDATION_AUDIO_EXTENSIONS):
                audio_paths.append(os.path.join(root, filename))

    files = []
    entries = []
    for audio_path in audio_paths:
        if "mute" in os.path.basename(audio_path).lower():
            print(f"Skipping mute external validation file: '{audio_path}'.")
            continue
        try:
            audio, audio_sample_rate = sf.read(
                audio_path, dtype="float32", always_2d=False
            )
        except (OSError, RuntimeError) as error:
            print(f"Skipping unreadable external validation file '{audio_path}': {error}")
            continue
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1, dtype=np.float32)
        duration_seconds = len(audio) / audio_sample_rate
        if duration_seconds < 2.0:
            print(f"Skipping external validation file under 2 seconds: '{audio_path}'.")
            continue
        if not np.isfinite(audio).all() or np.mean(np.abs(audio)) <= 1e-6:
            print(f"Skipping silent external validation file: '{audio_path}'.")
            continue
        relative_audio = os.path.relpath(audio_path, experiment_dir).replace("\\", "/")
        cache_key = hashlib.sha256(relative_audio.encode("utf-8")).hexdigest()
        f0_path = os.path.join(cache_dirs["f0"], f"{cache_key}.npy")
        f0_voiced_path = os.path.join(
            cache_dirs["f0_voiced"], f"{cache_key}.npy"
        )
        feature_path = os.path.join(cache_dirs["extracted"], f"{cache_key}.npy")
        files.append([audio_path, f0_path, f0_voiced_path, feature_path])
        entries.append(
            {
                "audio_path": relative_audio,
                "duration_seconds": duration_seconds,
                "f0_path": os.path.relpath(f0_path, experiment_dir).replace(
                    "\\", "/"
                ),
                "f0_voiced_path": os.path.relpath(
                    f0_voiced_path, experiment_dir
                ).replace("\\", "/"),
                "feature_path": os.path.relpath(
                    feature_path, experiment_dir
                ).replace("\\", "/"),
            }
        )
    return files, entries


def write_validation_manifest(experiment_dir, entries):
    validation_dir = os.path.join(experiment_dir, "validation")
    manifest_path = os.path.join(validation_dir, "manifest.json")
    if not entries:
        if os.path.isfile(manifest_path):
            os.remove(manifest_path)
        return
    temporary_path = f"{manifest_path}.{os.getpid()}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(
            {"version": VALIDATION_MANIFEST_VERSION, "entries": entries},
            file,
            ensure_ascii=False,
            indent=2,
        )
    os.replace(temporary_path, manifest_path)


def _load_entry(
    experiment_dir,
    entry,
    frames_per_second,
    minimum_frames,
    maximum_frames,
    feature_dim,
):
    try:
        audio_path = str(entry["audio_path"])
        duration_seconds = float(entry["duration_seconds"])
        feature = np.load(
            os.path.join(experiment_dir, entry["feature_path"]),
            allow_pickle=False,
        )
        pitch = np.load(
            os.path.join(experiment_dir, entry["f0_path"]),
            allow_pickle=False,
        ).reshape(-1)
        pitchf = np.load(
            os.path.join(experiment_dir, entry["f0_voiced_path"]),
            allow_pickle=False,
        ).reshape(-1)
    except (KeyError, OSError, ValueError) as error:
        print(f"Skipping external validation entry: {error}")
        return None

    if feature.ndim != 2 or feature.shape[1] != feature_dim:
        print(
            f"Skipping external validation clip '{audio_path}': feature shape "
            f"{feature.shape} does not match (*, {feature_dim})."
        )
        return None
    if (
        not np.isfinite(feature).all()
        or not np.isfinite(pitch).all()
        or not np.isfinite(pitchf).all()
    ):
        print(f"Skipping non-finite external validation clip '{audio_path}'.")
        return None

    source_length = int(round(duration_seconds * frames_per_second))
    if source_length < minimum_frames:
        print(f"Skipping external validation clip under 2 seconds: '{audio_path}'.")
        return None
    phone = np.repeat(feature.astype(np.float32, copy=False), 2, axis=0)
    sequence_lengths = (len(phone), len(pitch), len(pitchf))
    if min(sequence_lengths) < source_length - 4:
        print(
            f"Skipping external validation clip '{audio_path}': cached sequence "
            f"lengths {sequence_lengths} do not match {source_length}."
        )
        return None
    if len(phone) < source_length:
        phone = np.pad(phone, ((0, source_length - len(phone)), (0, 0)), mode="edge")
    if len(pitch) < source_length:
        pitch = np.pad(pitch, (0, source_length - len(pitch)), mode="edge")
    if len(pitchf) < source_length:
        pitchf = np.pad(pitchf, (0, source_length - len(pitchf)), mode="edge")
    phone = phone[:source_length]
    pitch = pitch[:source_length]
    pitchf = pitchf[:source_length]
    if source_length > maximum_frames:
        start = (source_length - maximum_frames) // 2
        end = start + maximum_frames
        phone = phone[start:end]
        pitch = pitch[start:end]
        pitchf = pitchf[start:end]
        length = maximum_frames
    else:
        length = source_length

    return {
        "audio_path": audio_path,
        "phone": torch.from_numpy(np.ascontiguousarray(phone)).float(),
        "pitch": torch.from_numpy(np.ascontiguousarray(pitch)).long(),
        "pitchf": torch.from_numpy(np.ascontiguousarray(pitchf)).float(),
        "length": length,
    }


def prepare_validation_reference(
    experiment_dir,
    device,
    seed,
    speaker_count,
    sample_rate,
    hop_length,
    feature_dim,
    max_samples=16,
    max_per_speaker=4,
):
    feature_dim = int(feature_dim)
    manifest_path = os.path.join(experiment_dir, "validation", "manifest.json")
    if not os.path.isfile(manifest_path):
        print(
            "External validation is disabled because logs/<model>/validation/manifest.json "
            "was not found. Add a validation folder to the dataset and run preprocessing "
            "and extraction again."
        )
        return None

    with open(manifest_path, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("version") != VALIDATION_MANIFEST_VERSION:
        raise RuntimeError("External validation manifest version is not supported")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("External validation manifest contains no audio entries")

    frames_per_second = sample_rate / hop_length
    minimum_frames = int(round(MINIMUM_VALIDATION_SECONDS * frames_per_second))
    preferred_frames = int(round(PREFERRED_VALIDATION_SECONDS * frames_per_second))
    maximum_frames = int(round(MAXIMUM_VALIDATION_SECONDS * frames_per_second))
    sources = []
    for entry in entries:
        source = _load_entry(
            experiment_dir,
            entry,
            frames_per_second,
            minimum_frames,
            maximum_frames,
            feature_dim,
        )
        if source is not None:
            sources.append(source)
    preferred = sorted(
        (source for source in sources if source["length"] >= preferred_frames),
        key=lambda source: _sort_key(seed, source["audio_path"]),
    )
    fallback = sorted(
        (source for source in sources if source["length"] < preferred_frames),
        key=lambda source: _sort_key(seed, source["audio_path"]),
    )
    ordered_sources = (preferred + fallback)[:max_per_speaker]
    if not ordered_sources:
        raise RuntimeError("No external validation clips of at least 2 seconds are available")

    speaker_count = int(speaker_count)
    if speaker_count < 1:
        raise RuntimeError("External validation requires at least one target speaker")
    speaker_order = sorted(
        range(speaker_count),
        key=lambda speaker_id: _sort_key(seed, f"speaker:{speaker_id}"),
    )
    selected = []
    for source in ordered_sources:
        for speaker_id in speaker_order:
            selected.append((source, speaker_id))
            if len(selected) == max_samples:
                break
        if len(selected) == max_samples:
            break

    maximum_length = max(source["length"] for source, _ in selected)
    phone = torch.zeros(len(selected), maximum_length, feature_dim)
    phone_lengths = torch.empty(len(selected), dtype=torch.long)
    pitch = torch.zeros(len(selected), maximum_length, dtype=torch.long)
    pitchf = torch.zeros(len(selected), maximum_length)
    speaker_ids = torch.empty(len(selected), dtype=torch.long)
    for index, (source, speaker_id) in enumerate(selected):
        length = source["length"]
        phone[index, :length] = source["phone"]
        phone_lengths[index] = length
        pitch[index, :length] = source["pitch"]
        pitchf[index, :length] = source["pitchf"]
        speaker_ids[index] = speaker_id

    inference_inputs = (
        phone.to(device),
        phone_lengths.to(device),
        pitch.to(device),
        pitchf.to(device),
        speaker_ids.to(device),
    )
    source_count = len({source["audio_path"] for source, _ in selected})
    target_count = len(set(speaker_ids.tolist()))
    print(
        f"External validation uses {len(selected)} probes from {source_count} source "
        f"clips across {target_count} target speakers. Ten-to-twenty-second clips are "
        "preferred and clips down to two seconds remain eligible."
    )
    return inference_inputs, None, None, speaker_ids
