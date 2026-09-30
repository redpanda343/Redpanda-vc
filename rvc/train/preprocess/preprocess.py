import concurrent.futures
import logging
import multiprocessing
import os
import sys
import time

import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from rvc.train.preprocess.slicer import (
    fireredvad_cuda_available,
    shutdown_fireredvad_gpu,
)
from rvc.train.preprocess.constants import (
    AUDIO_WRITE_MAX_WORKERS,
    GPU_PREPROCESS_MAX_WORKERS,
    MINIMUM_AUTOMATIC_SOURCE_AUDIO_SECONDS,
    MINIMUM_OUTPUT_AUDIO_SECONDS,
    PROCESS_PENDING_MULTIPLIER,
    SIMPLE_MIN_SILENCE_SECONDS,
    SIMPLE_SILENCE_COMPRESS_PERCENT,
    SIMPLE_SILENCE_THRESHOLD_DB,
    SIMPLE_TRUNCATE_TO_SECONDS,
)
from rvc.train.preprocess.dataset import (
    clear_flac_preprocess_artifacts,
    clear_simple_preprocess_artifacts,
    format_duration,
    normalize_dataset_format,
    save_dataset_duration,
    stage_validation_audio,
)
from rvc.train.preprocess.processor import PreProcess


logging.getLogger("numba.core.byteflow").setLevel(logging.WARNING)
logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)
logging.getLogger("numba.core.interpreter").setLevel(logging.WARNING)

_PROCESS_PREPROCESSOR = None


def strtobool(val):
    """Convert a string representation of truth to a bool."""
    return val.lower() in ("yes", "true", "t", "y", "1")


def initialize_preprocess_worker(
    sr, exp_dir, dataset_format, audio_write_workers, torch_threads, version
):
    global _PROCESS_PREPROCESSOR
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _PROCESS_PREPROCESSOR = PreProcess(
        sr, exp_dir, dataset_format, False, version
    )
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
    version: str = "v2",
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
    pp = PreProcess(sr, exp_dir, dataset_format, use_fireredvad_gpu, version)
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
                version,
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
                print("\nSlicing completed. Finalizing preprocessing...", flush=True)
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
        f"{MINIMUM_OUTPUT_AUDIO_SECONDS:.1f}s skipped before writing.",
        flush=True,
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
    version = str(sys.argv[19]) if len(sys.argv) > 19 else "v2"
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
        version,
    )
