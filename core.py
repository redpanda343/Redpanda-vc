import json
import os
import secrets
import subprocess
import sys

import click

from functools import lru_cache
from datetime import datetime, timedelta

now_dir = os.getcwd()
sys.path.append(now_dir)

from shared.platform import migrate_models_folder

migrate_models_folder(now_dir)

current_script_directory = os.path.dirname(os.path.realpath(__file__))
logs_path = os.path.join(current_script_directory, "logs")

from shared.tools.analyzer import analyze_audio
from shared.tools.launch_tensorboard import launch_tensorboard_pipeline
from shared.tools.model_download import model_download_pipeline
from shared.tools.prerequisites_download import prequisites_download_pipeline
from rvc.train.process.checkpoint_exporter import export_generator_checkpoint
from rvc.train.process.model_blender import model_blender as blend_models
from rvc.train.process.model_information import model_information
from rvc.train.process.training_control import start_training

python = sys.executable


def resolve_inference_seed(seed: int) -> int:
    """Return a random seed for 0; keep positive seeds deterministic."""
    seed = int(seed)
    if seed == 0:
        return secrets.randbelow(2_147_483_647) + 1
    return seed


@lru_cache(maxsize=None)
def import_voice_converter():
    from rvc.infer.infer import VoiceConverter

    return VoiceConverter()


@lru_cache(maxsize=1)
def get_config():
    from rvc.configs.config import Config

    return Config()


# Infer
def run_infer_script(
    pitch: int,
    index_rate: float,
    normalization_db: float,
    protect: float,
    f0_method: str,
    input_path: str,
    output_path: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    embedder_model: str,
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.5,
    reverb_dry_gain: float = 0.5,
    reverb_width: float = 0.5,
    reverb_freeze_mode: float = 0.5,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -6,
    limiter_release_time: float = 0.01,
    gain_db: float = 0.0,
    distortion_gain: float = 25,
    chorus_rate: float = 1.0,
    chorus_depth: float = 0.25,
    chorus_center_delay: float = 7,
    chorus_feedback: float = 0.0,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 8,
    clipping_threshold: float = -6,
    compressor_threshold: float = 0,
    compressor_ratio: float = 1,
    compressor_attack: float = 1.0,
    compressor_release: float = 100,
    delay_seconds: float = 0.5,
    delay_feedback: float = 0.0,
    delay_mix: float = 0.5,
    sid: int = 0,
    seed: int = 0,
    rectified_vocoder_path: str = "",
):
    seed = resolve_inference_seed(seed)
    kwargs = {
        "audio_input_path": input_path,
        "audio_output_path": output_path,
        "model_path": pth_path,
        "index_path": index_path,
        "normalization_db": normalization_db,
        "pitch": pitch,
        "index_rate": index_rate,
        "protect": protect,
        "f0_method": f0_method,
        "split_audio": split_audio,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "export_format": export_format,
        "embedder_model": embedder_model,
        "post_process": post_process,
        "reverb": reverb,
        "pitch_shift": pitch_shift,
        "limiter": limiter,
        "gain": gain,
        "distortion": distortion,
        "chorus": chorus,
        "bitcrush": bitcrush,
        "clipping": clipping,
        "compressor": compressor,
        "delay": delay,
        "reverb_room_size": reverb_room_size,
        "reverb_damping": reverb_damping,
        "reverb_wet_level": reverb_wet_gain,
        "reverb_dry_level": reverb_dry_gain,
        "reverb_width": reverb_width,
        "reverb_freeze_mode": reverb_freeze_mode,
        "pitch_shift_semitones": pitch_shift_semitones,
        "limiter_threshold": limiter_threshold,
        "limiter_release": limiter_release_time,
        "gain_db": gain_db,
        "distortion_gain": distortion_gain,
        "chorus_rate": chorus_rate,
        "chorus_depth": chorus_depth,
        "chorus_delay": chorus_center_delay,
        "chorus_feedback": chorus_feedback,
        "chorus_mix": chorus_mix,
        "bitcrush_bit_depth": bitcrush_bit_depth,
        "clipping_threshold": clipping_threshold,
        "compressor_threshold": compressor_threshold,
        "compressor_ratio": compressor_ratio,
        "compressor_attack": compressor_attack,
        "compressor_release": compressor_release,
        "delay_seconds": delay_seconds,
        "delay_feedback": delay_feedback,
        "delay_mix": delay_mix,
        "sid": sid,
        "seed": seed,
        "rectified_vocoder_path": rectified_vocoder_path,
    }
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio(**kwargs)
    return f"File {input_path} inferred successfully.", output_path.replace(
        ".wav", f".{export_format.lower()}"
    )


# Batch infer
def run_batch_infer_script(
    pitch: int,
    index_rate: float,
    normalization_db: float,
    protect: float,
    f0_method: str,
    input_folder: str,
    output_folder: str,
    pth_path: str,
    index_path: str,
    split_audio: bool,
    clean_audio: bool,
    clean_strength: float,
    export_format: str,
    embedder_model: str,
    post_process: bool = False,
    reverb: bool = False,
    pitch_shift: bool = False,
    limiter: bool = False,
    gain: bool = False,
    distortion: bool = False,
    chorus: bool = False,
    bitcrush: bool = False,
    clipping: bool = False,
    compressor: bool = False,
    delay: bool = False,
    reverb_room_size: float = 0.5,
    reverb_damping: float = 0.5,
    reverb_wet_gain: float = 0.5,
    reverb_dry_gain: float = 0.5,
    reverb_width: float = 0.5,
    reverb_freeze_mode: float = 0.5,
    pitch_shift_semitones: float = 0.0,
    limiter_threshold: float = -6,
    limiter_release_time: float = 0.01,
    gain_db: float = 0.0,
    distortion_gain: float = 25,
    chorus_rate: float = 1.0,
    chorus_depth: float = 0.25,
    chorus_center_delay: float = 7,
    chorus_feedback: float = 0.0,
    chorus_mix: float = 0.5,
    bitcrush_bit_depth: int = 8,
    clipping_threshold: float = -6,
    compressor_threshold: float = 0,
    compressor_ratio: float = 1,
    compressor_attack: float = 1.0,
    compressor_release: float = 100,
    delay_seconds: float = 0.5,
    delay_feedback: float = 0.0,
    delay_mix: float = 0.5,
    sid: int = 0,
    seed: int = 0,
    rectified_vocoder_path: str = "",
):
    seed = resolve_inference_seed(seed)
    kwargs = {
        "audio_input_paths": input_folder,
        "audio_output_path": output_folder,
        "model_path": pth_path,
        "index_path": index_path,
        "pitch": pitch,
        "index_rate": index_rate,
        "normalization_db": normalization_db,
        "protect": protect,
        "f0_method": f0_method,
        "split_audio": split_audio,
        "clean_audio": clean_audio,
        "clean_strength": clean_strength,
        "export_format": export_format,
        "embedder_model": embedder_model,
        "post_process": post_process,
        "reverb": reverb,
        "pitch_shift": pitch_shift,
        "limiter": limiter,
        "gain": gain,
        "distortion": distortion,
        "chorus": chorus,
        "bitcrush": bitcrush,
        "clipping": clipping,
        "compressor": compressor,
        "delay": delay,
        "reverb_room_size": reverb_room_size,
        "reverb_damping": reverb_damping,
        "reverb_wet_level": reverb_wet_gain,
        "reverb_dry_level": reverb_dry_gain,
        "reverb_width": reverb_width,
        "reverb_freeze_mode": reverb_freeze_mode,
        "pitch_shift_semitones": pitch_shift_semitones,
        "limiter_threshold": limiter_threshold,
        "limiter_release": limiter_release_time,
        "gain_db": gain_db,
        "distortion_gain": distortion_gain,
        "chorus_rate": chorus_rate,
        "chorus_depth": chorus_depth,
        "chorus_delay": chorus_center_delay,
        "chorus_feedback": chorus_feedback,
        "chorus_mix": chorus_mix,
        "bitcrush_bit_depth": bitcrush_bit_depth,
        "clipping_threshold": clipping_threshold,
        "compressor_threshold": compressor_threshold,
        "compressor_ratio": compressor_ratio,
        "compressor_attack": compressor_attack,
        "compressor_release": compressor_release,
        "delay_seconds": delay_seconds,
        "delay_feedback": delay_feedback,
        "delay_mix": delay_mix,
        "sid": sid,
        "seed": seed,
        "rectified_vocoder_path": rectified_vocoder_path,
    }
    infer_pipeline = import_voice_converter()
    infer_pipeline.convert_audio_batch(**kwargs)
    return f"Files from {input_folder} inferred successfully."


def run_multi_model_infer_script(
    pitch: int,
    index_rate: float,
    normalization_db: float,
    protect: float,
    f0_method: str,
    input_path: str,
    output_folder: str,
    pth_paths,
    index_paths,
    split_audio: bool,
    export_format: str,
    embedder_model: str,
    sid: int = 0,
    seed: int = 0,
):
    if not input_path or not os.path.isfile(input_path):
        raise ValueError("Select a valid input audio file.")

    if isinstance(pth_paths, str):
        pth_paths = [pth_paths]
    if isinstance(index_paths, str):
        index_paths = [index_paths]
    model_paths = [path for path in (pth_paths or []) if path]
    if not model_paths:
        raise ValueError("Select at least one voice model.")

    index_paths = list(index_paths or [])
    if len(index_paths) < len(model_paths):
        index_paths.extend([""] * (len(model_paths) - len(index_paths)))

    output_folder = output_folder or os.path.dirname(input_path)
    os.makedirs(output_folder, exist_ok=True)
    seed = resolve_inference_seed(seed)
    infer_pipeline = import_voice_converter()
    audio, chunks, intervals = infer_pipeline.prepare_audio(input_path, split_audio)
    input_name = os.path.splitext(os.path.basename(input_path))[0][:80]
    audio_results = []
    failures = []
    used_names = {}

    for model_path, index_path in zip(model_paths, index_paths):
        model_label = os.path.splitext(os.path.basename(model_path))[0]
        model_name = model_label[:80]
        output_name = f"{input_name}_{model_name}_output"
        name_key = output_name.lower()
        used_names[name_key] = used_names.get(name_key, 0) + 1
        if used_names[name_key] > 1:
            output_name = f"{output_name}_{used_names[name_key]}"
        wav_path = os.path.join(output_folder, f"{output_name}.wav")
        exported_path = os.path.splitext(wav_path)[0] + f".{export_format.lower()}"

        try:
            infer_pipeline.convert_audio(
                audio_input_path=input_path,
                audio_output_path=wav_path,
                model_path=model_path,
                index_path=index_path or "",
                pitch=pitch,
                index_rate=index_rate,
                normalization_db=normalization_db,
                protect=protect,
                f0_method=f0_method,
                split_audio=split_audio,
                export_format=export_format,
                embedder_model=embedder_model,
                sid=int(sid),
                seed=seed,
                _prepared_audio=audio,
                _prepared_chunks=chunks,
                _prepared_intervals=intervals,
            )
            audio_results.append((model_label, exported_path))
        except Exception as error:
            failures.append(f"{os.path.basename(model_path)}: {error}")

    summary = f"Converted {len(audio_results)} of {len(model_paths)} models."
    if failures:
        summary += " Failed: " + "; ".join(failures)
    return summary, audio_results


def run_multi_model_batch_infer_script(
    pitch: int,
    index_rate: float,
    normalization_db: float,
    protect: float,
    f0_method: str,
    input_folder: str,
    output_folder: str,
    pth_paths,
    index_paths,
    split_audio: bool,
    export_format: str,
    embedder_model: str,
    sid: int = 0,
    seed: int = 0,
):
    if not input_folder or not os.path.isdir(input_folder):
        raise ValueError("Select a valid input folder.")

    if isinstance(pth_paths, str):
        pth_paths = [pth_paths]
    if isinstance(index_paths, str):
        index_paths = [index_paths]
    model_paths = [path for path in (pth_paths or []) if path]
    if not model_paths:
        raise ValueError("Select at least one voice model.")

    audio_extensions = (
        ".wav",
        ".mp3",
        ".flac",
        ".ogg",
        ".opus",
        ".m4a",
        ".mp4",
        ".aac",
        ".alac",
        ".wma",
        ".aiff",
        ".webm",
        ".ac3",
    )
    audio_count = sum(
        1
        for name in os.listdir(input_folder)
        if name.lower().endswith(audio_extensions)
        and os.path.isfile(os.path.join(input_folder, name))
    )
    if audio_count == 0:
        raise ValueError("The input folder contains no supported audio files.")

    index_paths = list(index_paths or [])
    if len(index_paths) < len(model_paths):
        index_paths.extend([""] * (len(model_paths) - len(index_paths)))

    output_folder = output_folder or input_folder
    os.makedirs(output_folder, exist_ok=True)
    seed = resolve_inference_seed(seed)
    infer_pipeline = import_voice_converter()
    batch_results = []
    failures = []
    used_names = {}

    for model_path, index_path in zip(model_paths, index_paths):
        model_label = os.path.splitext(os.path.basename(model_path))[0]
        folder_name = model_label[:80]
        name_key = folder_name.lower()
        used_names[name_key] = used_names.get(name_key, 0) + 1
        if used_names[name_key] > 1:
            folder_name = f"{folder_name}_{used_names[name_key]}"
        model_output_folder = os.path.join(output_folder, folder_name)
        os.makedirs(model_output_folder, exist_ok=True)

        try:
            infer_pipeline.convert_audio_batch(
                audio_input_paths=input_folder,
                audio_output_path=model_output_folder,
                model_path=model_path,
                index_path=index_path or "",
                pitch=pitch,
                index_rate=index_rate,
                normalization_db=normalization_db,
                protect=protect,
                f0_method=f0_method,
                split_audio=split_audio,
                export_format=export_format,
                embedder_model=embedder_model,
                sid=int(sid),
                seed=seed,
            )
            batch_results.append((model_label, model_output_folder))
        except Exception as error:
            failures.append(f"{os.path.basename(model_path)}: {error}")

    summary = (
        f"Processed {audio_count} audio files with "
        f"{len(batch_results)} of {len(model_paths)} models."
    )
    if batch_results:
        summary += "\n" + "\n".join(
            f"{model_name}: {model_folder}"
            for model_name, model_folder in batch_results
        )
    if failures:
        summary += "\nFailed: " + "; ".join(failures)
    return summary


# Preprocess
def run_preprocess_script(
    model_name: str,
    dataset_path: str,
    sample_rate: int,
    cpu_cores: int,
    cut_preprocess: str,
    process_effects: bool,
    noise_reduction: bool,
    clean_strength: float,
    chunk_len: float,
    overlap_len: float,
    normalization_mode: str = "none",
    dataset_format: str = "WAV",
    truncate_silence_enabled: bool = False,
    truncate_silence_threshold_db: float = -45.0,
    truncate_silence_to_seconds: float = 0.3,
    truncate_silence_minimum_seconds: float = 0.3,
    truncate_silence_action: str = "truncate",
    truncate_silence_compress_percent: float = 50.0,
    version: str = "v2",
):
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC version: {version}")
    if version == "v1" and int(sample_rate) != 40000:
        raise ValueError("RVC v1 preprocessing requires a 40000 Hz model.")
    preprocess_script_path = os.path.join("shared", "preprocess", "preprocess.py")
    command = [
        python,
        preprocess_script_path,
        *map(
            str,
            [
                os.path.join(logs_path, model_name),
                dataset_path,
                sample_rate,
                cpu_cores,
                cut_preprocess,
                process_effects,
                noise_reduction,
                clean_strength,
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
            ],
        ),
    ]
    result = subprocess.run(command)
    if result.returncode != 0:
        return f"Preprocessing failed for model {model_name}. Please check the console logs for more details."

    return f"Model {model_name} preprocessed successfully."


# Extract
def run_extract_script(
    model_name: str,
    f0_method: str,
    cpu_cores: int,
    gpu: int,
    sample_rate: int,
    embedder_model: str,
    include_mutes: int = 2,
    version: str = "v2",
):
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC version: {version}")
    if version == "v1" and (
        int(sample_rate) != 40000 or embedder_model != "contentvec"
    ):
        raise ValueError("RVC v1 extraction requires ContentVec at 40000 Hz.")
    model_path = os.path.join(logs_path, model_name)
    extract = os.path.join("shared", "extract", "extract.py")

    command_1 = [
        python,
        extract,
        *map(
            str,
            [
                model_path,
                f0_method,
                cpu_cores,
                gpu,
                sample_rate,
                embedder_model,
                include_mutes,
                version,
            ],
        ),
    ]

    result = subprocess.run(command_1)
    if result.returncode != 0:
        return f"Feature extraction failed for model {model_name}. Please check the console logs for more details."

    return f"Model {model_name} extracted successfully."


def shutdown_after_training():
    os_name = sys.platform
    shutdown_time = None

    # Windows
    if os_name == "win32":
        delay_seconds = 300
        shutdown_time = datetime.now() + timedelta(seconds=delay_seconds)
        os.system(f"shutdown /s /t {delay_seconds}")

    # MacOS
    elif os_name == "darwin":
        shutdown_time = datetime.now()
        os.system("osascript -e 'tell app \"System Events\" to shut down'")

    # Linux
    elif os_name.startswith("linux"):
        delay_minutes = 5
        shutdown_time = datetime.now() + timedelta(minutes=delay_minutes)
        os.system(f"shutdown -h +{delay_minutes}")

    # Unknown
    else:
        print("Unsupported OS")
        return os_name, None

    return os_name, shutdown_time


def append_data_shutdown_log(
    model_name, total_epoch, batch_size, sample_rate, gpu, shutdown_time, os_name
):
    log_file = "training_shutdown_log.txt"

    log_entry = (
        f"[{datetime.now()}] "
        f"Model: {model_name} | "
        f"Epochs: {total_epoch} | "
        f"Batch: {batch_size} | "
        f"SR: {sample_rate} | "
        f"GPU: {gpu} | "
        f"OS: {os_name} | "
        f"Shutdown at: {shutdown_time}\n"
    )

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)


def _build_train_command(
    model_name: str,
    save_every_epoch: int,
    save_only_latest: bool,
    save_every_weights: bool,
    total_epoch: int,
    sample_rate: int,
    batch_size: int,
    gpu: int,
    pretrained: bool,
    cleanup: bool,
    index_algorithm: str = "Auto",
    cache_data_in_gpu: bool = False,
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
    vocoder: str = "HiFi-GAN",
    checkpointing: bool = False,
    save_every_steps: int = 0,
    version: str = "v2",
):
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC version: {version}")
    if version == "v1" and (int(sample_rate) != 40000 or vocoder != "HiFi-GAN"):
        raise ValueError("RVC v1 training requires HiFi-GAN at 40000 Hz.")
    expected_feature_dim = 256 if version == "v1" else 768
    model_config_path = os.path.join("logs", model_name, "config.json")
    feature_dim = expected_feature_dim
    if os.path.isfile(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as config_file:
            feature_dim = int(
                json.load(config_file).get("model", {}).get(
                    "text_enc_hidden_dim", 768
                )
            )
    if feature_dim != expected_feature_dim:
        raise ValueError(
            f"RVC {version} training requires {expected_feature_dim}-channel features, "
            f"but this experiment uses {feature_dim}. Re-extract the experiment with "
            f"RVC {version}."
        )
    if pretrained == True:
        from shared.tools.pretrained_selector import pretrained_selector

        if custom_pretrained == False:
            pg, pd = pretrained_selector(str(vocoder), int(sample_rate), version)
        else:
            if g_pretrained_path is None or d_pretrained_path is None:
                raise ValueError(
                    "Please provide the path to the pretrained G and D models."
                )
            pg, pd = g_pretrained_path, d_pretrained_path
    else:
        pg, pd = "", ""

    train_script_path = os.path.join("rvc", "train", "train.py")
    command = [
        python,
        train_script_path,
        *map(
            str,
            [
                model_name,
                save_every_epoch,
                total_epoch,
                pg,
                pd,
                gpu,
                batch_size,
                sample_rate,
                save_only_latest,
                save_every_weights,
                cache_data_in_gpu,
                cleanup,
                vocoder,
                checkpointing,
                save_every_steps,
                version,
            ],
        ),
    ]
    return command


# Train
def run_train_script(
    model_name: str,
    save_every_epoch: int,
    save_only_latest: bool,
    save_every_weights: bool,
    total_epoch: int,
    sample_rate: int,
    batch_size: int,
    gpu: int,
    pretrained: bool,
    cleanup: bool,
    index_algorithm: str = "Auto",
    cache_data_in_gpu: bool = False,
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
    vocoder: str = "HiFi-GAN",
    checkpointing: bool = False,
    shutdown_check: bool = False,
    save_every_steps: int = 0,
    version: str = "v2",
):
    command = _build_train_command(
        model_name,
        save_every_epoch,
        save_only_latest,
        save_every_weights,
        total_epoch,
        sample_rate,
        batch_size,
        gpu,
        pretrained,
        cleanup,
        index_algorithm,
        cache_data_in_gpu,
        custom_pretrained,
        g_pretrained_path,
        d_pretrained_path,
        vocoder,
        checkpointing,
        save_every_steps,
        version,
    )
    result = subprocess.run(command)
    if result.returncode != 0:
        return f"Training failed for model {model_name}. Please check the console logs for more details."

    if shutdown_check:
        os_name, shutdown_datetime = shutdown_after_training()

        append_data_shutdown_log(
            model_name=model_name,
            total_epoch=total_epoch,
            batch_size=batch_size,
            sample_rate=sample_rate,
            gpu=gpu,
            shutdown_time=shutdown_datetime,
            os_name=os_name,
        )

        print(
            f"Model {model_name} trained successfully. Shutdown scheduled at {shutdown_datetime}"
        )
        return f"Model {model_name} trained successfully. Shutdown scheduled at {shutdown_datetime}"

    return f"Model {model_name} trained successfully."


def start_train_script(
    model_name: str,
    save_every_epoch: int,
    save_only_latest: bool,
    save_every_weights: bool,
    total_epoch: int,
    sample_rate: int,
    batch_size: int,
    gpu: int,
    pretrained: bool,
    cleanup: bool,
    index_algorithm: str = "Auto",
    cache_data_in_gpu: bool = False,
    custom_pretrained: bool = False,
    g_pretrained_path: str = None,
    d_pretrained_path: str = None,
    vocoder: str = "HiFi-GAN",
    checkpointing: bool = False,
    shutdown_check: bool = False,
    save_every_steps: int = 0,
    version: str = "v2",
    generate_index: bool = False,
):
    command = _build_train_command(
        model_name,
        save_every_epoch,
        save_only_latest,
        save_every_weights,
        total_epoch,
        sample_rate,
        batch_size,
        gpu,
        pretrained,
        cleanup,
        index_algorithm,
        cache_data_in_gpu,
        custom_pretrained,
        g_pretrained_path,
        d_pretrained_path,
        vocoder,
        checkpointing,
        save_every_steps,
        version,
    )

    on_success = None
    if generate_index or shutdown_check:

        def complete_training():
            messages = ["Training finished successfully."]
            if generate_index:
                messages.append(run_index_script(model_name, index_algorithm))
            if not shutdown_check:
                return " ".join(messages)
            os_name, shutdown_datetime = shutdown_after_training()
            append_data_shutdown_log(
                model_name=model_name,
                total_epoch=total_epoch,
                batch_size=batch_size,
                sample_rate=sample_rate,
                gpu=gpu,
                shutdown_time=shutdown_datetime,
                os_name=os_name,
            )
            messages.append(f"Shutdown scheduled at {shutdown_datetime}.")
            return " ".join(messages)

        on_success = complete_training

    return start_training(
        command,
        logs_path,
        model_name,
        cwd=now_dir,
        on_success=on_success,
    )


# Index
def run_index_script(model_name: str, index_algorithm: str):
    index_script_path = os.path.join("rvc", "train", "process", "extract_index.py")
    command = [
        python,
        index_script_path,
        os.path.join(logs_path, model_name),
        index_algorithm,
    ]

    result = subprocess.run(command)
    if result.returncode != 0:
        return f"Index generation failed for model {model_name}. Make sure you have enough GPU available to generate the Index file. Please check the console logs for more details."

    return f"Index file for {model_name} generated successfully."


# Model information
def run_model_information_script(pth_path: str):
    print(model_information(pth_path))
    return model_information(pth_path)


# Model blender
def run_model_blender_script(
    model_name: str, pth_path_1: str, pth_path_2: str, ratio: float
):
    message, model_blended = blend_models(model_name, pth_path_1, pth_path_2, ratio)
    return message, model_blended


# Checkpoint exporter
def run_checkpoint_export_script(
    checkpoint_path: str, precision: str, output_name: str = None
):
    return export_generator_checkpoint(checkpoint_path, precision, output_name)


# Tensorboard
def run_tensorboard_script():
    launch_tensorboard_pipeline()


# Download
def run_download_script(model_link: str):
    result = model_download_pipeline(model_link)
    if result == "Error" or result is None:
        return "An error occurred downloading the model. Please check the console logs for more details."
    return "Model downloaded successfully."


# Prerequisites
def run_prerequisites_script(
    pretraineds_hifigan: bool,
    models: bool,
    exe: bool,
    beatrice: bool = False,
):
    prequisites_download_pipeline(
        pretraineds_hifigan,
        models,
        exe,
    )
    if beatrice:
        from beatrice.resources import ensure_trainer

        try:
            ensure_trainer()
        except Exception as error:
            print(
                f"Beatrice Trainer download failed: {error}. "
                "It is retried when Beatrice training starts."
            )
    return "Prerequisites installed successfully."


# Audio analyzer
def run_audio_analyzer_script(
    input_path: str, save_plot_path: str = "logs/audio_analysis.png"
):
    audio_info, plot_path = analyze_audio(input_path, save_plot_path)
    print(
        f"Audio info of {input_path}: {audio_info}",
        f"Audio file {input_path} analyzed successfully. Plot saved at: {plot_path}",
    )
    return audio_info, plot_path


def _get_version():
    config_path = os.path.join(
        current_script_directory, "assets", "config_template.json"
    )
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("version", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        return "unknown"


VERSION = _get_version()


def _infer_opts(func):
    """Core inference options shared by infer and batch_infer."""
    opts = [
        click.option(
            "--pitch",
            type=click.IntRange(-24, 24),
            default=0,
            help="Set the pitch of the audio. Higher values result in a higher pitch.",
        ),
        click.option(
            "--index-rate",
            type=click.FloatRange(0, 1),
            default=0.3,
            help="Control the influence of the index file on the output.",
        ),
        click.option(
            "--normalization-db",
            type=click.FloatRange(-12, 0),
            default=-1.0,
            help="Set the voiced output peak in dBFS. Use -12 to disable normalization.",
        ),
        click.option(
            "--protect",
            type=click.FloatRange(0, 0.5),
            default=0.33,
            help="Protect consonants and breathing sounds from artifacts.",
        ),
        click.option(
            "--f0-method",
            type=click.Choice(
                [
                    "rmvpe",
                    "swift",
                    "pm",
                ]
            ),
            default="rmvpe",
            help="Choose the pitch extraction algorithm.",
        ),
        click.option(
            "--split-audio",
            is_flag=True,
            default=False,
            help="Split audio into smaller segments before inference.",
        ),
        click.option(
            "--seed",
            type=click.IntRange(0, 2147483647),
            default=0,
            help="Set the inference seed. Use 0 for a random seed; values above 0 are deterministic.",
        ),
        click.option(
            "--clean-audio",
            is_flag=True,
            default=False,
            help="Clean output audio using noise reduction.",
        ),
        click.option(
            "--clean-strength",
            type=click.FloatRange(0, 1),
            default=0.7,
            help="Intensity of the audio cleaning process.",
        ),
        click.option(
            "--export-format",
            type=click.Choice(["WAV", "MP3", "FLAC", "OGG", "M4A"]),
            default="WAV",
            help="Output audio format.",
        ),
        click.option(
            "--embedder-model",
            type=click.Choice(
                [
                    "contentvec",
                    "spin-v2",
                ]
            ),
            default="contentvec",
            help="Model used for generating speaker embeddings.",
        ),
        click.option(
            "--sid", type=int, default=0, help="Speaker ID for multi-speaker models."
        ),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


def _post_process_opts(func):
    """Post-processing options shared by infer and batch_infer."""
    opts = [
        click.option(
            "--post-process",
            is_flag=True,
            default=False,
            help="Apply post-processing effects.",
        ),
        click.option(
            "--reverb", is_flag=True, default=False, help="Apply reverb effect."
        ),
        click.option("--reverb-room-size", type=float, default=0.5),
        click.option("--reverb-damping", type=float, default=0.5),
        click.option("--reverb-wet-gain", type=float, default=0.5),
        click.option("--reverb-dry-gain", type=float, default=0.5),
        click.option("--reverb-width", type=float, default=0.5),
        click.option("--reverb-freeze-mode", type=float, default=0.5),
        click.option(
            "--pitch-shift",
            is_flag=True,
            default=False,
            help="Apply pitch shift effect.",
        ),
        click.option("--pitch-shift-semitones", type=float, default=0.0),
        click.option(
            "--limiter", is_flag=True, default=False, help="Apply limiter effect."
        ),
        click.option("--limiter-threshold", type=float, default=-6),
        click.option("--limiter-release-time", type=float, default=0.01),
        click.option("--gain", is_flag=True, default=False, help="Apply gain effect."),
        click.option("--gain-db", type=float, default=0.0),
        click.option(
            "--distortion", is_flag=True, default=False, help="Apply distortion effect."
        ),
        click.option("--distortion-gain", type=float, default=25),
        click.option(
            "--chorus", is_flag=True, default=False, help="Apply chorus effect."
        ),
        click.option("--chorus-rate", type=float, default=1.0),
        click.option("--chorus-depth", type=float, default=0.25),
        click.option("--chorus-center-delay", type=float, default=7),
        click.option("--chorus-feedback", type=float, default=0.0),
        click.option("--chorus-mix", type=float, default=0.5),
        click.option(
            "--bitcrush", is_flag=True, default=False, help="Apply bitcrush effect."
        ),
        click.option("--bitcrush-bit-depth", type=int, default=8),
        click.option(
            "--clipping", is_flag=True, default=False, help="Apply clipping effect."
        ),
        click.option("--clipping-threshold", type=float, default=-6),
        click.option(
            "--compressor", is_flag=True, default=False, help="Apply compressor effect."
        ),
        click.option("--compressor-threshold", type=float, default=0),
        click.option("--compressor-ratio", type=float, default=1),
        click.option("--compressor-attack", type=float, default=1.0),
        click.option("--compressor-release", type=float, default=100),
        click.option(
            "--delay", is_flag=True, default=False, help="Apply delay effect."
        ),
        click.option("--delay-seconds", type=float, default=0.5),
        click.option("--delay-feedback", type=float, default=0.0),
        click.option("--delay-mix", type=float, default=0.5),
    ]
    for opt in reversed(opts):
        func = opt(func)
    return func


@click.group(invoke_without_command=True)
@click.version_option(
    version=VERSION, prog_name="Applio", message="%(prog)s v%(version)s"
)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit()


@cli.command()
@click.option("--input-path", required=True, help="Full path to the input audio file.")
@click.option(
    "--output-path", required=True, help="Full path to the output audio file."
)
@click.option(
    "--pth-path", required=True, help="Full path to the RVC model file (.pth)."
)
@click.option(
    "--index-path", required=True, help="Full path to the index file (.index)."
)
@_infer_opts
@_post_process_opts
def infer(**kwargs):
    """Run voice conversion on a single audio file."""
    result = run_infer_script(**kwargs)
    click.echo(result[0])


@cli.command()
@click.option(
    "--input-folder", required=True, help="Folder containing input audio files."
)
@click.option(
    "--output-folder", required=True, help="Folder for saving output audio files."
)
@click.option(
    "--pth-path", required=True, help="Full path to the RVC model file (.pth)."
)
@click.option(
    "--index-path", required=True, help="Full path to the index file (.index)."
)
@_infer_opts
@_post_process_opts
def batch_infer(**kwargs):
    """Run voice conversion on multiple audio files in a folder."""
    result = run_batch_infer_script(**kwargs)
    click.echo(result)


@cli.command()
@click.option("--model-name", required=True, help="Name of the model to train.")
@click.option("--dataset-path", required=True, help="Path to the dataset directory.")
@click.option(
    "--sample-rate",
    required=True,
    type=click.Choice(["32000", "40000", "48000"]),
    help="Target sampling rate.",
)
@click.option(
    "--version",
    type=click.Choice(["v1", "v2"]),
    default="v2",
    help="RVC feature and model version.",
)
@click.option(
    "--cpu-cores",
    type=click.IntRange(1, 64),
    default=None,
    help="Number of CPU cores to use.",
)
@click.option(
    "--cut-preprocess",
    type=click.Choice(["Skip", "Simple", "Automatic"]),
    default="Automatic",
    help="Dataset cutting method.",
)
@click.option(
    "--process-effects",
    is_flag=True,
    default=False,
    help="Disable filters during preprocessing.",
)
@click.option(
    "--noise-reduction",
    is_flag=True,
    default=False,
    help="Enable noise reduction during preprocessing.",
)
@click.option(
    "--noise-reduction-strength",
    type=click.FloatRange(0, 1),
    default=0.7,
    help="Strength of the noise reduction filter.",
)
@click.option(
    "--chunk-len",
    type=click.Choice([str(i * 0.5) for i in range(1, 11)]),
    default="3.0",
    help="Chunk length in seconds.",
)
@click.option(
    "--overlap-len",
    type=click.Choice(["0.0", "0.1", "0.2", "0.3", "0.4"]),
    default="0.3",
    help="Overlap length.",
)
@click.option(
    "--normalization-mode",
    type=click.Choice(["none", "pre", "post"]),
    default="none",
    help="Normalization mode.",
)
@click.option(
    "--dataset-format",
    type=click.Choice(["WAV", "WAV 32-bit float", "FLAC"], case_sensitive=False),
    default="WAV",
    show_default=True,
    help="Format and encoding used for processed training slices.",
)
@click.option(
    "--truncate-silence/--no-truncate-silence",
    default=False,
    show_default=True,
    help=(
        "Shorten silence below -45 dB before Simple slicing. "
        "Ignored by other cutting methods."
    ),
    hidden=True,
)
@click.option(
    "--truncate-silence-threshold-db",
    type=click.FloatRange(-80, -20),
    default=-45.0,
    show_default=True,
    help="Silence threshold used by Simple slicing when truncation is enabled.",
    hidden=True,
)
@click.option(
    "--truncate-silence-to-seconds",
    type=click.FloatRange(0.0, 0.5),
    default=0.3,
    show_default=True,
    help="Length retained from qualifying silence during Simple slicing.",
    hidden=True,
)
@click.option(
    "--truncate-silence-minimum-seconds",
    type=click.FloatRange(0.001, 5.0),
    default=0.3,
    show_default=True,
    help="Minimum silence duration detected during Simple slicing.",
    hidden=True,
)
@click.option(
    "--truncate-silence-action",
    type=click.Choice(["truncate", "compress"]),
    default="truncate",
    show_default=True,
    help="Audacity-style action used for qualifying silence.",
    hidden=True,
)
@click.option(
    "--truncate-silence-compress-percent",
    type=click.FloatRange(0.0, 99.9),
    default=50.0,
    show_default=True,
    help="Percentage of excess silence retained in compress mode.",
    hidden=True,
)
def preprocess(**kwargs):
    """Preprocess a dataset for training."""
    kwargs["sample_rate"] = int(kwargs["sample_rate"])
    kwargs["noise_reduction_strength"] = float(kwargs["noise_reduction_strength"])
    kwargs["chunk_len"] = float(kwargs["chunk_len"])
    kwargs["overlap_len"] = float(kwargs["overlap_len"])
    kwargs["cpu_cores"] = kwargs.get("cpu_cores") or 1
    result = run_preprocess_script(
        model_name=kwargs["model_name"],
        dataset_path=kwargs["dataset_path"],
        sample_rate=kwargs["sample_rate"],
        cpu_cores=kwargs["cpu_cores"],
        cut_preprocess=kwargs["cut_preprocess"],
        process_effects=kwargs["process_effects"],
        noise_reduction=kwargs["noise_reduction"],
        clean_strength=kwargs["noise_reduction_strength"],
        chunk_len=kwargs["chunk_len"],
        overlap_len=kwargs["overlap_len"],
        normalization_mode=kwargs["normalization_mode"],
        dataset_format=kwargs["dataset_format"],
        truncate_silence_enabled=kwargs["truncate_silence"],
        truncate_silence_threshold_db=kwargs["truncate_silence_threshold_db"],
        truncate_silence_to_seconds=kwargs["truncate_silence_to_seconds"],
        truncate_silence_minimum_seconds=kwargs[
            "truncate_silence_minimum_seconds"
        ],
        truncate_silence_action=kwargs["truncate_silence_action"],
        truncate_silence_compress_percent=kwargs[
            "truncate_silence_compress_percent"
        ],
        version=kwargs["version"],
    )
    click.echo(result)


@cli.command()
@click.option("--model-name", required=True, help="Name of the model.")
@click.option(
    "--f0-method",
    type=click.Choice(["rmvpe", "swift"]),
    default="rmvpe",
    help="Pitch extraction method.",
)
@click.option(
    "--cpu-cores", type=click.IntRange(1, 64), default=None, help="Number of CPU cores."
)
@click.option("--gpu", type=str, default="-", help="GPU device to use (e.g. '0').")
@click.option(
    "--sample-rate",
    required=True,
    type=click.Choice(["32000", "40000", "48000"]),
    help="Target sampling rate.",
)
@click.option(
    "--embedder-model",
    type=click.Choice(
        [
            "contentvec",
            "spin-v2",
        ]
    ),
    default="contentvec",
    help="Model used for generating speaker embeddings.",
)
@click.option(
    "--include-mutes",
    type=click.IntRange(0, 10),
    default=2,
    help="Number of silent files to include.",
)
@click.option(
    "--version",
    type=click.Choice(["v1", "v2"]),
    default="v2",
    help="RVC feature and model version.",
)
def extract(**kwargs):
    """Extract features from a preprocessed dataset."""
    kwargs["sample_rate"] = int(kwargs["sample_rate"])
    kwargs["cpu_cores"] = kwargs.get("cpu_cores") or 1
    result = run_extract_script(
        model_name=kwargs["model_name"],
        f0_method=kwargs["f0_method"],
        cpu_cores=kwargs["cpu_cores"],
        gpu=kwargs["gpu"],
        sample_rate=kwargs["sample_rate"],
        embedder_model=kwargs["embedder_model"],
        include_mutes=kwargs["include_mutes"],
        version=kwargs["version"],
    )
    click.echo(result)


@cli.command()
@click.option("--model-name", required=True, help="Name of the model to train.")
@click.option(
    "--vocoder",
    type=click.Choice(["HiFi-GAN", "MRF HiFi-GAN", "RefineGAN"]),
    default="HiFi-GAN",
    help="Vocoder to use.",
)
@click.option(
    "--checkpointing",
    is_flag=True,
    default=False,
    help="Enable memory-efficient checkpointing.",
)
@click.option(
    "--save-every-epoch",
    required=True,
    type=click.IntRange(1, 100),
    help="Save checkpoint every N epochs.",
)
@click.option(
    "--save-only-latest",
    is_flag=True,
    default=False,
    help="Keep only the latest checkpoint.",
)
@click.option(
    "--save-every-weights",
    is_flag=True,
    default=True,
    help="Save model weights every epoch.",
)
@click.option(
    "--save-every-steps",
    type=click.IntRange(min=0),
    default=0,
    help="Save an inference-only model every N training steps (0 disables it).",
)
@click.option(
    "--total-epoch",
    type=click.IntRange(1, 10000),
    default=1000,
    help="Total number of training epochs.",
)
@click.option(
    "--sample-rate",
    required=True,
    type=click.Choice(["32000", "40000", "48000"]),
    help="Training sampling rate.",
)
@click.option(
    "--version",
    type=click.Choice(["v1", "v2"]),
    default="v2",
    help="RVC feature and model version.",
)
@click.option(
    "--batch-size", type=click.IntRange(1, 50), default=8, help="Training batch size."
)
@click.option("--gpu", type=str, default="0", help="GPU device to use.")
@click.option(
    "--pretrained/--no-pretrained",
    default=True,
    help="Use pretrained model for initialisation.",
)
@click.option(
    "--custom-pretrained",
    is_flag=True,
    default=False,
    help="Use custom pretrained model paths.",
)
@click.option(
    "--g-pretrained-path", type=str, default=None, help="Path to pretrained generator."
)
@click.option(
    "--d-pretrained-path",
    type=str,
    default=None,
    help="Path to pretrained discriminator.",
)
@click.option(
    "--cleanup", is_flag=True, default=False, help="Clean up previous training attempt."
)
@click.option(
    "--cache-data-in-gpu",
    is_flag=True,
    default=False,
    help="Cache training data in GPU memory.",
)
@click.option(
    "--index-algorithm",
    type=click.Choice(["Auto", "Faiss", "KMeans"]),
    default="Auto",
    help="Index file generation algorithm.",
)
def train(**kwargs):
    """Train an RVC model."""
    result = run_train_script(
        model_name=kwargs["model_name"],
        save_every_epoch=kwargs["save_every_epoch"],
        save_only_latest=kwargs["save_only_latest"],
        save_every_weights=kwargs["save_every_weights"],
        total_epoch=kwargs["total_epoch"],
        sample_rate=int(kwargs["sample_rate"]),
        batch_size=kwargs["batch_size"],
        gpu=kwargs["gpu"],
        pretrained=kwargs["pretrained"],
        cleanup=kwargs["cleanup"],
        index_algorithm=kwargs["index_algorithm"],
        cache_data_in_gpu=kwargs["cache_data_in_gpu"],
        custom_pretrained=kwargs["custom_pretrained"],
        g_pretrained_path=kwargs.get("g_pretrained_path"),
        d_pretrained_path=kwargs.get("d_pretrained_path"),
        vocoder=kwargs["vocoder"],
        checkpointing=kwargs["checkpointing"],
        save_every_steps=kwargs["save_every_steps"],
        version=kwargs["version"],
    )
    click.echo(result)


@cli.command()
@click.option("--model-name", required=True, help="Name of the model.")
@click.option(
    "--index-algorithm",
    type=click.Choice(["Auto", "Faiss", "KMeans"]),
    default="Auto",
    help="Index file generation algorithm.",
)
def index(**kwargs):
    """Generate an index file for an RVC model."""
    result = run_index_script(kwargs["model_name"], kwargs["index_algorithm"])
    click.echo(result)


@cli.command()
@click.option("--pth-path", required=True, help="Path to the .pth model file.")
def model_information(**kwargs):
    """Display information about a trained model."""
    run_model_information_script(kwargs["pth_path"])


@cli.command()
@click.option("--model-name", required=True, help="Name of the new fused model.")
@click.option("--pth-path-1", required=True, help="Path to the first .pth model.")
@click.option("--pth-path-2", required=True, help="Path to the second .pth model.")
@click.option(
    "--ratio",
    type=click.Choice([str(i / 10) for i in range(11)]),
    default="0.5",
    help="Blend weight: 0.0 is Model B, 0.5 is equal, and 1.0 is Model A.",
)
def model_blender(**kwargs):
    """Fuse two RVC models together."""
    kwargs["ratio"] = float(kwargs["ratio"])
    msg, path = run_model_blender_script(
        kwargs["model_name"],
        kwargs["pth_path_1"],
        kwargs["pth_path_2"],
        kwargs["ratio"],
    )
    click.echo(f"{msg} {path}")


@cli.command()
def tensorboard():
    """Launch TensorBoard for monitoring training progress."""
    run_tensorboard_script()


@cli.command()
@click.option("--model-link", required=True, help="Direct link to the model file.")
def download(**kwargs):
    """Download a model from a provided link."""
    result = run_download_script(kwargs["model_link"])
    click.echo(result)


@cli.command()
@click.option(
    "--pretraineds-hifigan/--no-pretraineds-hifigan",
    default=True,
    help="Download pretrained HiFi-GAN models.",
)
@click.option("--models/--no-models", default=True, help="Download additional models.")
@click.option("--exe/--no-exe", default=True, help="Download required executables.")
@click.option(
    "--beatrice/--no-beatrice",
    default=True,
    help="Download Beatrice Trainer with its pretrained models.",
)
def prerequisites(**kwargs):
    """Install prerequisites for RVC."""
    result = run_prerequisites_script(
        kwargs["pretraineds_hifigan"],
        kwargs["models"],
        kwargs["exe"],
        kwargs["beatrice"],
    )
    click.echo(result)


@cli.command()
@click.option("--input-path", required=True, help="Path to the input audio file.")
def audio_analyzer(**kwargs):
    """Analyze an audio file and display information."""
    run_audio_analyzer_script(kwargs["input_path"])


def main():
    cli()


if __name__ == "__main__":
    main()
