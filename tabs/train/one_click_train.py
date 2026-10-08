import json
import os
import tempfile
import threading
from multiprocessing import cpu_count

import gradio as gr

from assets.i18n.i18n import I18nAuto
from core import (
    run_extract_script,
    run_preprocess_script,
    start_train_script,
)
from rvc.configs.config import get_number_of_gpus
from tabs.settings.sections.restart import get_train_state
from tabs.train.rvc import (
    get_datasets_list,
    get_models_list,
    get_pretrained_list,
)

i18n = I18nAuto()
now_dir = os.getcwd()
custom_presets_path = os.path.join(now_dir, "logs", "one_click_train_presets.json")
custom_presets_lock = threading.RLock()


DEFAULT_TRAINING_PRESETS = {
    "32k hifigan,cvec": {
        "version": "v2",
        "sampling_rate": "32000",
        "vocoder": "HiFi-GAN",
        "batch_size": 8,
        "total_epoch": 200,
        "save_every_epoch": 10,
        "cut_preprocess": "Automatic",
        "normalization_mode": "none",
        "dataset_format": "WAV 16-bit",
        "process_effects": False,
        "f0_method": "rmvpe",
        "embedder_model": "contentvec",
        "include_mutes": 2,
        "pretrained_mode": "Default pretrained",
        "cache_dataset_in_gpu": False,
        "checkpointing": False,
        "index_algorithm": "Auto",
        "chunk_len": 3.0,
        "overlap_len": 0.3,
        "truncate_silence_enabled": True,
        "truncate_silence_threshold_db": -45,
        "truncate_silence_minimum_seconds": 0.3,
        "truncate_silence_to_seconds": 0.3,
        "truncate_silence_action": "truncate",
        "truncate_silence_compress_percent": 50.0,
        "g_pretrained_path": None,
        "d_pretrained_path": None,
        "cpu_cores": max(1, min(cpu_count() // 2, 32)),
        "gpu": str(get_number_of_gpus()),
        "cleanup": False,
    },
    "32k From scratch": {
        "version": "v2",
        "sampling_rate": "32000",
        "vocoder": "HiFi-GAN",
        "batch_size": 16,
        "total_epoch": 2000,
        "save_every_epoch": 10,
        "cut_preprocess": "Automatic",
        "normalization_mode": "none",
        "dataset_format": "WAV 16-bit",
        "process_effects": False,
        "f0_method": "rmvpe",
        "embedder_model": "contentvec",
        "include_mutes": 0,
        "pretrained_mode": "No pretrained",
        "cache_dataset_in_gpu": False,
        "checkpointing": False,
        "index_algorithm": "Auto",
        "chunk_len": 3.0,
        "overlap_len": 0.3,
        "truncate_silence_enabled": True,
        "truncate_silence_threshold_db": -45,
        "truncate_silence_minimum_seconds": 0.3,
        "truncate_silence_to_seconds": 0.3,
        "truncate_silence_action": "truncate",
        "truncate_silence_compress_percent": 50.0,
        "g_pretrained_path": None,
        "d_pretrained_path": None,
        "cpu_cores": max(1, min(cpu_count() // 2, 32)),
        "gpu": str(get_number_of_gpus()),
        "cleanup": False,
    },
}


def _load_custom_presets():
    try:
        with open(custom_presets_path, "r", encoding="utf-8") as presets_file:
            presets = json.load(presets_file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(presets, dict):
        return {}
    defaults = next(iter(DEFAULT_TRAINING_PRESETS.values()))
    required_fields = set(defaults) - {
        "version",
        "vocoder",
        "truncate_silence_action",
        "truncate_silence_compress_percent",
    }
    loaded_presets = {}
    for name, settings in presets.items():
        if (
            isinstance(name, str)
            and name.strip()
            and isinstance(settings, dict)
            and required_fields.issubset(settings)
            and name not in DEFAULT_TRAINING_PRESETS
        ):
            loaded_settings = defaults | settings
            if loaded_settings["f0_method"] == "pm":
                loaded_settings["f0_method"] = "swift"
            if loaded_settings["dataset_format"] == "WAV":
                loaded_settings["dataset_format"] = "WAV 16-bit"
            if loaded_settings["cut_preprocess"] == "Simple":
                loaded_settings["cut_preprocess"] = "Automatic"
            loaded_presets[name] = loaded_settings
    return loaded_presets


TRAINING_PRESETS = {
    name: settings.copy() for name, settings in DEFAULT_TRAINING_PRESETS.items()
}
TRAINING_PRESETS.update(_load_custom_presets())


def _failed(message):
    return isinstance(message, str) and "failed" in message.lower()


def _apply_preset(preset_name):
    preset = TRAINING_PRESETS[preset_name]
    version = preset.get("version", "v2")
    sample_rate_choices = ["40000"] if version == "v1" else (
        ["24000", "32000"]
        if preset["vocoder"] == "RefineGAN"
        else ["32000", "40000", "48000"]
    )
    sample_rate = "40000" if version == "v1" else preset["sampling_rate"]
    vocoder = "HiFi-GAN" if version == "v1" else preset["vocoder"]
    embedder_model = "contentvec" if version == "v1" else preset["embedder_model"]
    return (
        version,
        gr.update(choices=sample_rate_choices, value=sample_rate),
        vocoder,
        preset["batch_size"],
        preset["total_epoch"],
        preset["save_every_epoch"],
        preset["cut_preprocess"],
        preset["normalization_mode"],
        preset["dataset_format"],
        False,
        preset["f0_method"],
        embedder_model,
        preset["include_mutes"],
        preset["pretrained_mode"],
        preset["cache_dataset_in_gpu"],
        preset["checkpointing"],
        preset["index_algorithm"],
        preset["chunk_len"],
        preset["overlap_len"],
        preset["truncate_silence_enabled"],
        preset["truncate_silence_threshold_db"],
        preset["truncate_silence_minimum_seconds"],
        gr.update(
            value=preset["truncate_silence_to_seconds"],
            visible=preset["truncate_silence_action"] == "truncate",
        ),
        preset["truncate_silence_action"],
        gr.update(
            value=preset["truncate_silence_compress_percent"],
            visible=preset["truncate_silence_action"] == "compress",
        ),
        preset["g_pretrained_path"],
        preset["d_pretrained_path"],
        preset["cpu_cores"],
        preset["gpu"],
        preset["cleanup"],
        gr.update(visible=preset["cut_preprocess"] == "Simple"),
        gr.update(
            visible=preset["cut_preprocess"] == "Simple"
            and preset["truncate_silence_enabled"]
        ),
        gr.update(visible=preset["pretrained_mode"] == "Custom pretrained"),
    )


def _save_custom_preset(
    preset_name,
    model_version,
    sampling_rate,
    vocoder,
    batch_size,
    total_epoch,
    save_every_epoch,
    cut_preprocess,
    normalization_mode,
    dataset_format,
    process_effects,
    f0_method,
    embedder_model,
    include_mutes,
    pretrained_mode,
    cache_dataset_in_gpu,
    checkpointing,
    index_algorithm,
    chunk_len,
    overlap_len,
    truncate_silence_enabled,
    truncate_silence_threshold_db,
    truncate_silence_minimum_seconds,
    truncate_silence_to_seconds,
    truncate_silence_action,
    truncate_silence_compress_percent,
    g_pretrained_path,
    d_pretrained_path,
    cpu_cores,
    gpu,
    cleanup,
):
    preset_name = str(preset_name or "").strip()
    if not preset_name:
        message = "Enter a name for the custom preset."
        gr.Warning(message)
        return gr.update(), preset_name
    if preset_name in DEFAULT_TRAINING_PRESETS:
        message = "Choose a different name. Default presets cannot be replaced."
        gr.Warning(message)
        return gr.update(), preset_name

    settings = {
        "version": str(model_version),
        "sampling_rate": str(sampling_rate),
        "vocoder": vocoder,
        "batch_size": int(batch_size),
        "total_epoch": int(total_epoch),
        "save_every_epoch": int(save_every_epoch),
        "cut_preprocess": cut_preprocess,
        "normalization_mode": normalization_mode,
        "dataset_format": dataset_format,
        "process_effects": bool(process_effects),
        "f0_method": f0_method,
        "embedder_model": embedder_model,
        "include_mutes": int(include_mutes),
        "pretrained_mode": pretrained_mode,
        "cache_dataset_in_gpu": bool(cache_dataset_in_gpu),
        "checkpointing": bool(checkpointing),
        "index_algorithm": index_algorithm,
        "chunk_len": float(chunk_len),
        "overlap_len": float(overlap_len),
        "truncate_silence_enabled": bool(truncate_silence_enabled),
        "truncate_silence_threshold_db": float(truncate_silence_threshold_db),
        "truncate_silence_minimum_seconds": float(
            truncate_silence_minimum_seconds
        ),
        "truncate_silence_to_seconds": float(truncate_silence_to_seconds),
        "truncate_silence_action": str(truncate_silence_action),
        "truncate_silence_compress_percent": float(
            truncate_silence_compress_percent
        ),
        "g_pretrained_path": g_pretrained_path or None,
        "d_pretrained_path": d_pretrained_path or None,
        "cpu_cores": int(cpu_cores),
        "gpu": str(gpu),
        "cleanup": bool(cleanup),
    }

    with custom_presets_lock:
        custom_presets = {
            name: preset
            for name, preset in TRAINING_PRESETS.items()
            if name not in DEFAULT_TRAINING_PRESETS
        }
        custom_presets[preset_name] = settings
        os.makedirs(os.path.dirname(custom_presets_path), exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix=".one_click_train_presets.",
            suffix=".tmp",
            dir=os.path.dirname(custom_presets_path),
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as presets_file:
                json.dump(custom_presets, presets_file, indent=4)
            os.replace(temporary_path, custom_presets_path)
        except Exception:
            try:
                os.remove(temporary_path)
            except OSError:
                pass
            raise
        TRAINING_PRESETS[preset_name] = settings

    gr.Info(f"Preset '{preset_name}' saved.")
    return gr.update(choices=list(TRAINING_PRESETS), value=preset_name), ""


def _simple_settings_visibility(cut_preprocess, truncate_silence_enabled):
    is_simple = cut_preprocess == "Simple"
    return (
        gr.update(visible=is_simple),
        gr.update(visible=is_simple and truncate_silence_enabled),
    )


def _truncate_settings_visibility(truncate_silence_enabled, cut_preprocess):
    return gr.update(
        visible=truncate_silence_enabled and cut_preprocess == "Simple"
    )


def _truncate_action_visibility(action):
    return (
        gr.update(visible=action == "truncate"),
        gr.update(visible=action == "compress"),
    )


def _custom_pretrained_visibility(pretrained_mode):
    return gr.update(visible=pretrained_mode == "Custom pretrained")


def _sampling_rate_for_vocoder(model_version, vocoder, sampling_rate):
    if model_version == "v1":
        choices = ["40000"]
        value = "40000"
    else:
        choices = (
            ["24000", "32000"]
            if vocoder == "RefineGAN"
            else ["32000", "40000", "48000"]
        )
        value = sampling_rate if sampling_rate in choices else "32000"
    return gr.update(choices=choices, value=value)


def _controls_for_version(model_version, vocoder, sampling_rate, embedder_model):
    if model_version == "v1":
        return (
            gr.update(choices=["40000"], value="40000"),
            gr.update(
                choices=["HiFi-GAN"], value="HiFi-GAN", interactive=False
            ),
            gr.update(
                choices=["contentvec"], value="contentvec", interactive=False
            ),
        )
    sample_rate_choices = (
        ["24000", "32000"]
        if vocoder == "RefineGAN"
        else ["32000", "40000", "48000"]
    )
    sample_rate_value = (
        sampling_rate if sampling_rate in sample_rate_choices else "32000"
    )
    return (
        gr.update(choices=sample_rate_choices, value=sample_rate_value),
        gr.update(
            choices=["HiFi-GAN", "RefineGAN"],
            value=vocoder if vocoder in {"HiFi-GAN", "RefineGAN"} else "HiFi-GAN",
            interactive=True,
        ),
        gr.update(
            choices=["contentvec", "spin-v2"],
            value=embedder_model
            if embedder_model in {"contentvec", "spin-v2"}
            else "contentvec",
            interactive=True,
        ),
    )


def _training_ui_state(model_name):
    state = get_train_state(model_name)
    status = state.get("status", "idle")
    message = state.get("message", "Ready to start one-click training.")
    is_active = status in {"running", "paused", "stopping", "finalizing"}
    if status == "idle":
        message = "Ready to start one-click training."
    elif status == "running":
        message = (
            "Training is running. The index will be generated automatically "
            "after training finishes."
        )
    return gr.update(interactive=not is_active), message


def _run_one_click_training(
    model_name,
    dataset_path,
    model_version,
    sampling_rate,
    vocoder,
    cut_preprocess,
    normalization_mode,
    dataset_format,
    process_effects,
    chunk_len,
    overlap_len,
    truncate_silence_enabled,
    truncate_silence_threshold_db,
    truncate_silence_minimum_seconds,
    truncate_silence_to_seconds,
    truncate_silence_action,
    truncate_silence_compress_percent,
    cpu_cores,
    gpu,
    f0_method,
    embedder_model,
    include_mutes,
    batch_size,
    total_epoch,
    save_every_epoch,
    pretrained_mode,
    g_pretrained_path,
    d_pretrained_path,
    cleanup,
    cache_dataset_in_gpu,
    checkpointing,
    index_algorithm,
    progress=gr.Progress(),
):
    model_name = str(model_name or "").strip()
    dataset_path = str(dataset_path or "").strip()
    if not model_name:
        message = "Enter a model name before starting."
        gr.Warning(message)
        return gr.update(interactive=True), message
    if not dataset_path or not os.path.isdir(dataset_path):
        message = "Select a valid dataset folder before starting."
        gr.Warning(message)
        return gr.update(interactive=True), message
    state = get_train_state(model_name)
    if state.get("status") in {"running", "paused", "stopping", "finalizing"}:
        message = state.get("message", "Training is already active for this model.")
        gr.Warning(message)
        return gr.update(interactive=False), message
    if cut_preprocess == "Simple" and float(overlap_len) >= float(chunk_len):
        message = "Overlap length must be shorter than chunk length."
        gr.Warning(message)
        return gr.update(interactive=True), message
    if model_version == "v1" and (
        str(sampling_rate) != "40000"
        or vocoder != "HiFi-GAN"
        or embedder_model != "contentvec"
    ):
        message = "RVC v1 requires ContentVec, HiFi-GAN, and 40000 Hz."
        gr.Warning(message)
        return gr.update(interactive=True), message
    valid_sample_rates = {"40000"} if model_version == "v1" else (
        {"24000", "32000"}
        if vocoder == "RefineGAN"
        else {"32000", "40000", "48000"}
    )
    if str(sampling_rate) not in valid_sample_rates:
        message = f"{sampling_rate} Hz is not available for {vocoder}."
        gr.Warning(message)
        return gr.update(interactive=True), message

    pretrained = pretrained_mode != "No pretrained"
    custom_pretrained = pretrained_mode == "Custom pretrained"
    if custom_pretrained:
        if not g_pretrained_path or not os.path.isfile(g_pretrained_path):
            message = "Select a valid custom pretrained G file."
            gr.Warning(message)
            return gr.update(interactive=True), message
        if not d_pretrained_path or not os.path.isfile(d_pretrained_path):
            message = "Select a valid custom pretrained D file."
            gr.Warning(message)
            return gr.update(interactive=True), message

    progress(0.05, desc="Preprocessing dataset")
    preprocess_message = run_preprocess_script(
        model_name=model_name,
        dataset_path=dataset_path,
        sample_rate=sampling_rate,
        cpu_cores=cpu_cores,
        cut_preprocess=cut_preprocess,
        process_effects=process_effects,
        noise_reduction=False,
        clean_strength=0.5,
        chunk_len=chunk_len,
        overlap_len=overlap_len,
        normalization_mode=normalization_mode,
        dataset_format=dataset_format,
        truncate_silence_enabled=truncate_silence_enabled,
        truncate_silence_threshold_db=truncate_silence_threshold_db,
        truncate_silence_to_seconds=truncate_silence_to_seconds,
        truncate_silence_minimum_seconds=truncate_silence_minimum_seconds,
        truncate_silence_action=truncate_silence_action,
        truncate_silence_compress_percent=truncate_silence_compress_percent,
        version=model_version,
    )
    if _failed(preprocess_message):
        gr.Warning(preprocess_message)
        return gr.update(interactive=True), preprocess_message

    progress(0.4, desc="Extracting pitch and features")
    extract_message = run_extract_script(
        model_name=model_name,
        f0_method=f0_method,
        cpu_cores=cpu_cores,
        gpu=gpu,
        sample_rate=sampling_rate,
        embedder_model=embedder_model,
        include_mutes=include_mutes,
        version=model_version,
    )
    if _failed(extract_message):
        gr.Warning(extract_message)
        return gr.update(interactive=True), extract_message

    progress(0.8, desc="Starting training")
    training_message = start_train_script(
        model_name=model_name,
        save_every_epoch=save_every_epoch,
        save_only_latest=True,
        save_every_weights=True,
        total_epoch=total_epoch,
        sample_rate=sampling_rate,
        batch_size=batch_size,
        gpu=gpu,
        pretrained=pretrained,
        cleanup=cleanup,
        index_algorithm=index_algorithm,
        cache_data_in_gpu=cache_dataset_in_gpu,
        custom_pretrained=custom_pretrained,
        g_pretrained_path=g_pretrained_path,
        d_pretrained_path=d_pretrained_path,
        vocoder=vocoder,
        checkpointing=checkpointing,
        shutdown_check=False,
        save_every_steps=0,
        version=model_version,
        generate_index=True,
    )
    progress(1.0, desc="Training started")
    if "started" not in training_message.lower():
        gr.Warning(training_message)
        return _training_ui_state(model_name)
    status_message = (
        f"{training_message} The index will be generated automatically after "
        "training finishes."
    )
    gr.Info(status_message)
    return gr.update(interactive=False), status_message


def one_click_train_tab():
    gr.Markdown(
        i18n(
            "Choose a preset, adjust any settings you want, then preprocess, "
            "extract, train, and generate the index with one click."
        )
    )

    with gr.Row():
        preset = gr.Dropdown(
            choices=list(TRAINING_PRESETS),
            value="32k hifigan,cvec",
            label=i18n("Training Preset"),
            info=i18n("Loads editable defaults for a common training goal."),
            interactive=True,
        )
        model_name = gr.Dropdown(
            choices=get_models_list(),
            value="my-project",
            label=i18n("Model Name"),
            info=i18n("Name of the new model."),
            allow_custom_value=True,
            interactive=True,
        )
        dataset_path = gr.Dropdown(
            choices=get_datasets_list(),
            label=i18n("Dataset Path"),
            info=i18n("Path to the dataset folder."),
            allow_custom_value=True,
            interactive=True,
        )
    with gr.Row():
        custom_preset_name = gr.Textbox(
            label=i18n("New Preset Name"),
            info=i18n("Adjust the settings below, then save them as a new preset."),
            placeholder=i18n("Enter a preset name"),
            interactive=True,
        )
        save_custom_preset_button = gr.Button(i18n("Save Custom Preset"))

    with gr.Accordion(i18n("Dataset Settings"), open=True):
        with gr.Row():
            model_version = gr.Radio(
                choices=["v1", "v2"],
                value="v2",
                label=i18n("RVC Version"),
                info=i18n(
                    "v1 uses ContentVec 256 features with HiFi-GAN at 40 kHz."
                ),
                interactive=True,
            )
            sampling_rate = gr.Radio(
                choices=["32000", "40000", "48000"],
                value="32000",
                label=i18n("Sampling Rate"),
                interactive=True,
            )
            vocoder = gr.Radio(
                choices=["HiFi-GAN", "RefineGAN"],
                value="HiFi-GAN",
                label=i18n("Vocoder"),
                info=i18n(
                    "HiFi-GAN supports 32, 40, and 48 kHz. RefineGAN supports "
                    "24 and 32 kHz."
                ),
                interactive=True,
            )
            cut_preprocess = gr.Radio(
                choices=["Skip", "Automatic"],
                value="Automatic",
                label=i18n("Audio cutting"),
                interactive=True,
            )
            normalization_mode = gr.Radio(
                choices=["none", "pre", "post"],
                value="none",
                label=i18n("Normalization mode"),
                interactive=True,
                visible=True,
            )
            dataset_format = gr.Radio(
                choices=["WAV 16-bit", "WAV 32-bit float", "FLAC"],
                value="WAV 16-bit",
                label=i18n("Dataset format"),
                interactive=True,
            )
        process_effects = gr.Checkbox(
            value=False,
            label=i18n("DC-offset removal"),
            interactive=True,
            visible=False,
        )
        with gr.Column(visible=False) as simple_settings:
            with gr.Row():
                chunk_len = gr.Slider(
                    0.5,
                    5.0,
                    value=3.0,
                    step=0.1,
                    label=i18n("Chunk length (sec)"),
                    interactive=True,
                )
                overlap_len = gr.Slider(
                    0.0,
                    0.4,
                    value=0.3,
                    step=0.1,
                    label=i18n("Overlap length (sec)"),
                    interactive=True,
                )
                truncate_silence_enabled = gr.Checkbox(
                    value=True,
                    label=i18n("Truncate silence"),
                    interactive=True,
                )
            with gr.Column(visible=False) as truncate_settings:
                truncate_silence_action = gr.Radio(
                    choices=[
                        (i18n("Truncate Detected Silence"), "truncate"),
                        (i18n("Compress Excess Silence"), "compress"),
                    ],
                    value="truncate",
                    label=i18n("Silence action"),
                    interactive=True,
                )
                with gr.Row():
                    truncate_silence_threshold_db = gr.Slider(
                        -80,
                        -20,
                        value=-45,
                        step=1,
                        label=i18n("Silence threshold (dB)"),
                        interactive=True,
                    )
                    truncate_silence_minimum_seconds = gr.Slider(
                        0.001,
                        5.0,
                        value=0.3,
                        step=0.001,
                        label=i18n("Minimum silence (sec)"),
                        interactive=True,
                    )
                    truncate_silence_to_seconds = gr.Slider(
                        0.0,
                        0.5,
                        value=0.3,
                        step=0.001,
                        label=i18n("Truncate to (sec)"),
                        interactive=True,
                    )
                    truncate_silence_compress_percent = gr.Slider(
                        0.0,
                        99.9,
                        value=50.0,
                        step=0.1,
                        label=i18n("Compress excess silence to (%)"),
                        interactive=True,
                        visible=False,
                    )

    with gr.Accordion(i18n("Feature Extraction"), open=False):
        with gr.Row():
            f0_method = gr.Radio(
                choices=["swift", "rmvpe"],
                value="rmvpe",
                label=i18n("Pitch extraction algorithm"),
                interactive=True,
            )
            embedder_model = gr.Radio(
                choices=["contentvec", "spin-v2"],
                value="contentvec",
                label=i18n("Embedder Model"),
                interactive=True,
            )
            include_mutes = gr.Slider(
                0,
                10,
                value=2,
                step=1,
                label=i18n("Silent training files"),
                interactive=True,
            )
        with gr.Row():
            cpu_cores = gr.Slider(
                1,
                min(cpu_count(), 32),
                value=max(1, min(cpu_count() // 2, 32)),
                step=1,
                label=i18n("CPU Cores"),
                interactive=True,
            )
            gpu = gr.Textbox(
                value=str(get_number_of_gpus()),
                label=i18n("GPU Number"),
                info=i18n("Use hyphens between multiple GPU numbers."),
                interactive=True,
            )

    with gr.Accordion(i18n("Training Settings"), open=True):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                64,
                value=8,
                step=1,
                label=i18n("Batch Size"),
                interactive=True,
            )
            total_epoch = gr.Slider(
                1,
                10000,
                value=200,
                step=1,
                label=i18n("Total Epoch"),
                interactive=True,
            )
            save_every_epoch = gr.Slider(
                1,
                100,
                value=10,
                step=1,
                label=i18n("Save Every Epoch"),
                interactive=True,
            )
        pretrained_mode = gr.Radio(
            choices=["Default pretrained", "Custom pretrained", "No pretrained"],
            value="Default pretrained",
            label=i18n("Pretrained Model"),
            interactive=True,
        )
        with gr.Column(visible=False) as custom_pretrained_settings:
            with gr.Row():
                g_pretrained_path = gr.Dropdown(
                    choices=sorted(get_pretrained_list("G")),
                    label=i18n("Custom Pretrained G"),
                    allow_custom_value=True,
                    interactive=True,
                )
                d_pretrained_path = gr.Dropdown(
                    choices=sorted(get_pretrained_list("D")),
                    label=i18n("Custom Pretrained D"),
                    allow_custom_value=True,
                    interactive=True,
                )
        with gr.Row():
            cleanup = gr.Checkbox(
                value=False,
                label=i18n("Fresh Training"),
                interactive=True,
            )
            cache_dataset_in_gpu = gr.Checkbox(
                value=False,
                label=i18n("Cache Dataset in GPU"),
                interactive=True,
            )
            checkpointing = gr.Checkbox(
                value=False,
                label=i18n("Checkpointing"),
                interactive=True,
            )
            index_algorithm = gr.Radio(
                choices=["Auto", "Faiss", "KMeans"],
                value="Auto",
                label=i18n("Index Algorithm"),
                interactive=True,
            )

    training_status = gr.Textbox(
        value=i18n("Ready to start one-click training."),
        label=i18n("Status"),
        max_lines=3,
        interactive=False,
    )
    one_click_button = gr.Button(i18n("One-click Training"), variant="primary")

    preset_outputs = [
        model_version,
        sampling_rate,
        vocoder,
        batch_size,
        total_epoch,
        save_every_epoch,
        cut_preprocess,
        normalization_mode,
        dataset_format,
        process_effects,
        f0_method,
        embedder_model,
        include_mutes,
        pretrained_mode,
        cache_dataset_in_gpu,
        checkpointing,
        index_algorithm,
        chunk_len,
        overlap_len,
        truncate_silence_enabled,
        truncate_silence_threshold_db,
        truncate_silence_minimum_seconds,
        truncate_silence_to_seconds,
        truncate_silence_action,
        truncate_silence_compress_percent,
        g_pretrained_path,
        d_pretrained_path,
        cpu_cores,
        gpu,
        cleanup,
        simple_settings,
        truncate_settings,
        custom_pretrained_settings,
    ]
    preset.input(
        fn=_apply_preset,
        inputs=[preset],
        outputs=preset_outputs,
        queue=False,
    )
    save_custom_preset_button.click(
        fn=_save_custom_preset,
        inputs=[
            custom_preset_name,
            model_version,
            sampling_rate,
            vocoder,
            batch_size,
            total_epoch,
            save_every_epoch,
            cut_preprocess,
            normalization_mode,
            dataset_format,
            process_effects,
            f0_method,
            embedder_model,
            include_mutes,
            pretrained_mode,
            cache_dataset_in_gpu,
            checkpointing,
            index_algorithm,
            chunk_len,
            overlap_len,
            truncate_silence_enabled,
            truncate_silence_threshold_db,
            truncate_silence_minimum_seconds,
            truncate_silence_to_seconds,
            truncate_silence_action,
            truncate_silence_compress_percent,
            g_pretrained_path,
            d_pretrained_path,
            cpu_cores,
            gpu,
            cleanup,
        ],
        outputs=[preset, custom_preset_name],
        queue=False,
    )
    cut_preprocess.input(
        fn=_simple_settings_visibility,
        inputs=[cut_preprocess, truncate_silence_enabled],
        outputs=[simple_settings, truncate_settings],
        queue=False,
    )
    truncate_silence_enabled.input(
        fn=_truncate_settings_visibility,
        inputs=[truncate_silence_enabled, cut_preprocess],
        outputs=[truncate_settings],
        queue=False,
    )
    truncate_silence_action.input(
        fn=_truncate_action_visibility,
        inputs=[truncate_silence_action],
        outputs=[
            truncate_silence_to_seconds,
            truncate_silence_compress_percent,
        ],
        queue=False,
    )
    pretrained_mode.input(
        fn=_custom_pretrained_visibility,
        inputs=[pretrained_mode],
        outputs=[custom_pretrained_settings],
        queue=False,
    )
    vocoder.input(
        fn=_sampling_rate_for_vocoder,
        inputs=[model_version, vocoder, sampling_rate],
        outputs=[sampling_rate],
        queue=False,
    )
    model_version.input(
        fn=_controls_for_version,
        inputs=[model_version, vocoder, sampling_rate, embedder_model],
        outputs=[sampling_rate, vocoder, embedder_model],
        queue=False,
    )

    one_click_button.click(
        fn=_run_one_click_training,
        inputs=[
            model_name,
            dataset_path,
            model_version,
            sampling_rate,
            vocoder,
            cut_preprocess,
            normalization_mode,
            dataset_format,
            process_effects,
            chunk_len,
            overlap_len,
            truncate_silence_enabled,
            truncate_silence_threshold_db,
            truncate_silence_minimum_seconds,
            truncate_silence_to_seconds,
            truncate_silence_action,
            truncate_silence_compress_percent,
            cpu_cores,
            gpu,
            f0_method,
            embedder_model,
            include_mutes,
            batch_size,
            total_epoch,
            save_every_epoch,
            pretrained_mode,
            g_pretrained_path,
            d_pretrained_path,
            cleanup,
            cache_dataset_in_gpu,
            checkpointing,
            index_algorithm,
        ],
        outputs=[one_click_button, training_status],
        concurrency_limit=1,
        concurrency_id="one-click-training",
        trigger_mode="once",
    )

    training_status_timer = gr.Timer(value=2.0, active=True)
    training_status_timer.tick(
        fn=_training_ui_state,
        inputs=[model_name],
        outputs=[one_click_button, training_status],
        queue=False,
    )
