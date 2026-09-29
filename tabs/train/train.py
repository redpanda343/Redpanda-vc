import os
import shutil
import sys
import zipfile
from multiprocessing import cpu_count

import gradio as gr

from assets.i18n.i18n import I18nAuto
from core import (
    run_extract_script,
    run_index_script,
    run_preprocess_script,
    start_train_script,
)
from rvc.configs.config import get_gpu_info, get_number_of_gpus, max_vram_gpu
from rvc.lib.utils import format_title
from tabs.settings.sections.restart import (
    get_train_state,
    pause_train,
    resume_train,
    stop_train,
)

i18n = I18nAuto()
now_dir = os.getcwd()
sys.path.append(now_dir)


sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "ogg",
    "opus",
    "m4a",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}

validation_audioext = {"wav", "mp3", "flac", "ogg"}

# Custom Pretraineds
pretraineds_custom_path = os.path.join(
    now_dir, "rvc", "models", "pretraineds", "custom"
)

pretraineds_custom_path_relative = os.path.relpath(pretraineds_custom_path, now_dir)

os.makedirs(pretraineds_custom_path_relative, exist_ok=True)


def get_pretrained_list(suffix):
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(pretraineds_custom_path_relative)
        for filename in filenames
        if filename.endswith(".pth") and suffix in filename
    ]


pretraineds_list_d = get_pretrained_list("D")
pretraineds_list_g = get_pretrained_list("G")


def refresh_custom_pretraineds():
    return (
        {"choices": sorted(get_pretrained_list("G")), "__type__": "update"},
        {"choices": sorted(get_pretrained_list("D")), "__type__": "update"},
    )


# Dataset Creator
datasets_path = os.path.join(now_dir, "assets", "datasets")

if not os.path.exists(datasets_path):
    os.makedirs(datasets_path)

datasets_path_relative = os.path.relpath(datasets_path, now_dir)


def get_datasets_list():
    return [
        dirpath
        for dirpath, _, filenames in os.walk(datasets_path_relative)
        if any(filename.endswith(tuple(sup_audioext)) for filename in filenames)
    ]


def refresh_datasets():
    return {"choices": sorted(get_datasets_list()), "__type__": "update"}


# Model Names
models_path = os.path.join(now_dir, "logs")
os.makedirs(models_path, exist_ok=True)


def get_models_list():
    return [
        os.path.basename(dirpath)
        for dirpath in os.listdir(models_path)
        if os.path.isdir(os.path.join(models_path, dirpath))
        and all(excluded not in dirpath for excluded in ["zips", "mute", "reference"])
    ]


def refresh_models():
    return {"choices": sorted(get_models_list()), "__type__": "update"}


# Refresh Models and Datasets
def refresh_models_and_datasets():
    return (
        {"choices": sorted(get_models_list()), "__type__": "update"},
        {"choices": sorted(get_datasets_list()), "__type__": "update"},
    )


# Drop Model
def save_drop_model(dropbox):
    if ".pth" not in dropbox:
        gr.Info(
            i18n(
                "The file you dropped is not a valid pretrained file. Please try again."
            )
        )
    else:
        file_name = os.path.basename(dropbox)
        pretrained_path = os.path.join(pretraineds_custom_path_relative, file_name)
        if os.path.exists(pretrained_path):
            os.remove(pretrained_path)
        shutil.copy(dropbox, pretrained_path)
        gr.Info(
            i18n(
                "Click the refresh button to see the pretrained file in the dropdown menu."
            )
        )
    return None


# Drop Dataset
def save_drop_dataset_audio(dropbox, dataset_name):
    if not dataset_name:
        gr.Info("Please enter a valid dataset name. Please try again.")
        return None, None
    else:
        file_extension = os.path.splitext(dropbox)[1][1:].lower()
        if file_extension not in sup_audioext:
            gr.Info("The file you dropped is not a valid audio file. Please try again.")
        else:
            dataset_name = format_title(dataset_name)
            audio_file = format_title(os.path.basename(dropbox))
            dataset_path = os.path.join(now_dir, "assets", "datasets", dataset_name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)
            destination_path = os.path.join(dataset_path, audio_file)
            if os.path.exists(destination_path):
                os.remove(destination_path)
            shutil.copy(dropbox, destination_path)
            gr.Info(
                i18n(
                    "The audio file has been successfully added to the dataset. Please click the preprocess button."
                )
            )
            dataset_path = os.path.dirname(destination_path)
            relative_dataset_path = os.path.relpath(dataset_path, now_dir)

            return None, relative_dataset_path


def save_drop_validation_audio(dropbox, dataset_path):
    if not dropbox or not dataset_path:
        gr.Info(i18n("Please select a dataset before adding validation audio."))
        return None

    file_extension = os.path.splitext(dropbox)[1][1:].lower()
    if file_extension not in validation_audioext:
        gr.Info(i18n("The file you dropped is not a supported validation audio file."))
        return None

    dataset_path = str(dataset_path).strip()
    if os.path.isabs(dataset_path):
        resolved_dataset_path = os.path.abspath(dataset_path)
    else:
        resolved_dataset_path = os.path.abspath(os.path.join(now_dir, dataset_path))
    if not os.path.isdir(resolved_dataset_path):
        gr.Info(i18n("The selected dataset folder does not exist."))
        return None

    validation_path = os.path.join(resolved_dataset_path, "validation")
    os.makedirs(validation_path, exist_ok=True)
    audio_file = format_title(os.path.basename(dropbox))
    source_path = os.path.abspath(dropbox)
    destination_path = os.path.abspath(os.path.join(validation_path, audio_file))
    if os.path.normcase(source_path) != os.path.normcase(destination_path):
        shutil.copy2(source_path, destination_path)
    gr.Info(
        i18n(
            "The validation audio was added successfully. Please preprocess and extract again."
        )
    )
    return None


# Export
def get_pth_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".pth")
    ]


def get_index_list():
    return [
        os.path.relpath(os.path.join(dirpath, filename), now_dir)
        for dirpath, _, filenames in os.walk(models_path)
        for filename in filenames
        if filename.endswith(".index") and not filename.startswith("trained_")
    ]


def refresh_pth_and_index_list():
    return (
        {"choices": sorted(get_pth_list()), "__type__": "update"},
        {"choices": sorted(get_index_list()), "__type__": "update"},
    )


# Export Pth and Index Files
def export_pth(pth_path):
    allowed_paths = get_pth_list()
    normalized_allowed_paths = [
        os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths
    ]
    normalized_pth_path = os.path.abspath(os.path.join(now_dir, pth_path))

    if normalized_pth_path in normalized_allowed_paths:
        return pth_path
    else:
        print(f"Attempted to export invalid pth path: {pth_path}")
        return None


def export_index(index_path):
    allowed_paths = get_index_list()
    normalized_allowed_paths = [
        os.path.abspath(os.path.join(now_dir, p)) for p in allowed_paths
    ]
    normalized_index_path = os.path.abspath(os.path.join(now_dir, index_path))

    if normalized_index_path in normalized_allowed_paths:
        return index_path
    else:
        print(f"Attempted to export invalid index path: {index_path}")
        return None


# Upload to Google Drive
def upload_to_google_drive(pth_path, index_path, model_name):
    if not os.path.exists(os.path.join(models_path, model_name)):
        return gr.Info("Model folder not found.")
    if not pth_path or not os.path.exists(pth_path):
        return gr.Info(".pth not found.")
    try:
        zip_path = f"{model_name}.zip"
        gr.Info("Uploading...")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            zipf.write(pth_path, os.path.basename(pth_path))
            if index_path and os.path.exists(index_path):
                zipf.write(index_path, os.path.basename(index_path))
        drive_folder = "/content/drive/MyDrive/ApplioExported"
        os.makedirs(drive_folder, exist_ok=True)
        dest = os.path.join(drive_folder, zip_path)
        if os.path.exists(dest):
            os.remove(dest)
        shutil.move(zip_path, dest)
        gr.Info("Uploaded.")
    except Exception as error:
        print(f"An error occurred uploading to Google Drive: {error}")
        gr.Info("Upload failed.")


def auto_enable_checkpointing():
    try:
        return max_vram_gpu(0) < 6
    except:
        return False


# Train Tab
def train_tab():
    def _preprocess_with_toast(
        model_name,
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
    ):
        gr.Info(i18n("Preprocessing dataset..."))
        result = run_preprocess_script(
            model_name=model_name,
            dataset_path=dataset_path,
            sample_rate=sample_rate,
            cpu_cores=cpu_cores,
            cut_preprocess=cut_preprocess,
            process_effects=process_effects,
            noise_reduction=noise_reduction,
            clean_strength=clean_strength,
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
        )
        if isinstance(result, str):
            if "error" in result.lower() or "failed" in result.lower():
                gr.Warning(result)
            else:
                gr.Info(result)
        return result

    def _extract_with_toast(*args):
        gr.Info(i18n("Extracting features..."))
        result = run_extract_script(*args)
        if isinstance(result, str):
            if "error" in result.lower() or "failed" in result.lower():
                gr.Warning(result)
            else:
                gr.Info(result)
        return result

    def _index_with_toast(*args):
        gr.Info(i18n("Generating index..."))
        result = run_index_script(*args)
        if isinstance(result, str):
            if "error" in result.lower() or "failed" in result.lower():
                gr.Warning(result)
            else:
                gr.Info(result)

    # Model settings section
    with gr.Accordion(i18n("Model Settings")):
        with gr.Row():
            with gr.Column():
                model_name = gr.Dropdown(
                    label=i18n("Model Name"),
                    info=i18n("Name of the new model."),
                    choices=get_models_list(),
                    value="my-project",
                    interactive=True,
                    allow_custom_value=True,
                )
                model_version = gr.Radio(
                    label=i18n("RVC Version"),
                    choices=["v1", "v2"],
                    value="v2",
                    interactive=True,
                )
            with gr.Column():
                sampling_rate = gr.Radio(
                    label=i18n("Sampling Rate"),
                    info=i18n("The sampling rate of the audio files."),
                    choices=["32000", "40000", "48000"],
                    value="32000",
                    interactive=True,
                )
                vocoder = gr.Radio(
                    label=i18n("Vocoder"),
                    info=i18n(
                        "Choose the vocoder for audio synthesis. RVC v1 uses HiFi-GAN at 40 kHz."
                    ),
                    choices=["HiFi-GAN", "RefineGAN"],
                    value="HiFi-GAN",
                    interactive=True,
                    visible=True,
                )
        with gr.Accordion(
            i18n("Advanced Settings"),
            open=False,
        ):
            with gr.Row():
                with gr.Column():
                    cpu_cores = gr.Slider(
                        1,
                        min(cpu_count(), 32),  # max 32 parallel processes
                        max(1, min(cpu_count() // 2, 32)),
                        step=1,
                        label=i18n("CPU Cores"),
                        info=i18n(
                            "The number of CPU cores to use in the extraction process."
                        ),
                        interactive=True,
                    )

                with gr.Column():
                    gpu = gr.Textbox(
                        label=i18n("GPU Number"),
                        info=i18n(
                            "Specify the number of GPUs you wish to utilize for extracting by entering them separated by hyphens (-)."
                        ),
                        placeholder=i18n("0 to ∞ separated by -"),
                        value=str(get_number_of_gpus()),
                        interactive=True,
                    )
                    gr.Textbox(
                        label=i18n("GPU Information"),
                        info=i18n("The GPU information will be displayed here."),
                        value=get_gpu_info(),
                        interactive=False,
                    )
    # Preprocess section
    with gr.Accordion(i18n("Preprocess")):
        dataset_path = gr.Dropdown(
            label=i18n("Dataset Path"),
            info=i18n("Path to the dataset folder."),
            # placeholder=i18n("Enter dataset path"),
            choices=get_datasets_list(),
            allow_custom_value=True,
            interactive=True,
        )
        with gr.Row():
            dataset_creator = gr.Checkbox(
                label=i18n("Dataset Creator"),
                value=False,
                interactive=True,
                visible=True,
            )
            validation_audio_enabled = gr.Checkbox(
                label=i18n("Validation Audio"),
                value=False,
                interactive=True,
                visible=True,
            )
        with gr.Column(visible=False) as dataset_creator_settings:
            with gr.Accordion(i18n("Dataset Creator")):
                dataset_name = gr.Textbox(
                    label=i18n("Dataset Name"),
                    info=i18n("Name of the new dataset."),
                    placeholder=i18n("Enter dataset name"),
                    interactive=True,
                )
                upload_audio_dataset = gr.File(
                    label=i18n("Upload Audio Dataset"),
                    type="filepath",
                    interactive=True,
                )
        with gr.Column(visible=False) as validation_audio_settings:
            with gr.Accordion(i18n("Validation Audio")):
                gr.Markdown(
                    i18n(
                        "For reliable, efficient validation, use 1 to 3 clean clips "
                        "between 2 and 20 seconds. Additional or longer clips are unnecessary."
                    )
                )
                upload_validation_audio = gr.File(
                    label=i18n("Upload Validation Audio"),
                    file_types=[".wav", ".mp3", ".flac", ".ogg"],
                    type="filepath",
                    interactive=True,
                )
        refresh = gr.Button(i18n("Refresh"))

        with gr.Accordion(i18n("Advanced Settings"), open=False):
            dataset_format = gr.Radio(
                label=i18n("Dataset format"),
                info=i18n(
                    "Format used for processed training slices."
                ),
                choices=["WAV 16-bit", "WAV 32-bit float", "FLAC"],
                value="WAV 16-bit",
                interactive=True,
            )
            cut_preprocess = gr.Radio(
                label=i18n("Audio cutting"),
                info=i18n(
                    "Audio file slicing method: Select 'Skip' if the files are "
                    "already pre-sliced, 'Simple' for fixed-length slicing when "
                    "excessive silence has already been removed, or 'Automatic' "
                    "for RMS-based silence detection and slicing."
                ),
                choices=["Skip", "Simple", "Automatic"],
                value="Automatic",
                interactive=True,
            )
            truncate_silence_enabled = gr.Checkbox(
                label=i18n("Truncate silence"),
                info=i18n(
                    "For Simple slicing only. Shortens qualifying silent regions "
                    "using the settings below."
                ),
                value=False,
                interactive=True,
                visible=False,
            )
            truncate_silence_action = gr.Radio(
                choices=[
                    (i18n("Truncate Detected Silence"), "truncate"),
                    (i18n("Compress Excess Silence"), "compress"),
                ],
                value="truncate",
                label=i18n("Silence action"),
                interactive=True,
                visible=False,
            )
            truncate_silence_threshold_db = gr.Slider(
                minimum=-80,
                maximum=-20,
                value=-45,
                step=1,
                label=i18n("Silence threshold (dB)"),
                info=i18n(
                    "For Simple slicing only. Audio below this level is treated "
                    "as silence when truncation is enabled."
                ),
                interactive=True,
                visible=False,
            )
            truncate_silence_minimum_seconds = gr.Slider(
                minimum=0.001,
                maximum=5.0,
                value=0.3,
                step=0.001,
                label=i18n("Minimum silence (sec)"),
                info=i18n(
                    "For Simple slicing only. A silent region must be at least "
                    "this long before it can be truncated."
                ),
                interactive=True,
                visible=False,
            )
            truncate_silence_to_seconds = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                value=0.3,
                step=0.001,
                label=i18n("Truncate to (sec)"),
                info=i18n(
                    "For Simple slicing only. Sets how much of each qualifying "
                    "silent region remains after truncation."
                ),
                interactive=True,
                visible=False,
            )
            truncate_silence_compress_percent = gr.Slider(
                minimum=0.0,
                maximum=99.9,
                value=50.0,
                step=0.1,
                label=i18n("Compress excess silence to (%)"),
                info=i18n(
                    "Keeps the minimum silence plus this percentage of the "
                    "silence beyond that minimum."
                ),
                interactive=True,
                visible=False,
            )
            with gr.Row():
                chunk_len = gr.Slider(
                    0.5,
                    5.0,
                    3.0,
                    step=0.1,
                    label=i18n("Chunk length (sec)"),
                    info=i18n("Length of the audio slice for 'Simple' method."),
                    interactive=True,
                )
                overlap_len = gr.Slider(
                    0.0,
                    0.4,
                    0.3,
                    step=0.1,
                    label=i18n("Overlap length (sec)"),
                    info=i18n(
                        "Length of the overlap between slices for 'Simple' method."
                    ),
                    interactive=True,
                )

            with gr.Row():
                process_effects = gr.Checkbox(
                    label=i18n("DC-offset removal"),
                    info=i18n(
                        "Removes DC-offset from the dataset. Disable this if your dataset has no DC-offset or has already been passed through a high-pass filter."
                    ),
                    value=True,
                    interactive=True,
                    visible=False,
                )

                normalization_mode = gr.Radio(
                    label=i18n("Normalization mode"),
                    info=i18n(
                        "Audio normalization: Select 'none' if the files are already normalized, 'pre' to normalize the entire input file at once, or 'post' to normalize each slice individually."
                    ),
                    choices=["none", "pre", "post"],
                    value="none",
                    interactive=True,
                    visible=True,
                )

                noise_reduction = gr.Checkbox(
                    label=i18n("Noise Reduction"),
                    info=i18n(
                        "It's recommended keep deactivate this option if your dataset has already been processed."
                    ),
                    value=False,
                    interactive=True,
                    visible=False,
                )
            clean_strength = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Noise Reduction Strength"),
                info=i18n(
                    "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                ),
                visible=False,
                value=0.5,
                interactive=True,
            )
        preprocess_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )

        with gr.Row():
            preprocess_button = gr.Button(i18n("Preprocess Dataset"))
            preprocess_button.click(
                fn=_preprocess_with_toast,
                inputs=[
                    model_name,
                    dataset_path,
                    sampling_rate,
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
                    model_version,
                ],
                outputs=[preprocess_output_info],
            )

    # Extract section
    with gr.Accordion(i18n("Extract")):
        with gr.Row():
            f0_method = gr.Radio(
                label=i18n("Pitch extraction algorithm"),
                info=i18n(
                    "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                ),
                choices=[
                    "swift",
                    "rmvpe",
                    # "fcpe"
                ],
                value="rmvpe",
                interactive=True,
            )

            embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for learning speaker embedding."),
                choices=[
                    "contentvec",
                    "spin-v2",
                ],
                value="contentvec",
                interactive=True,
            )
        include_mutes = gr.Slider(
            0,
            10,
            2,
            step=1,
            label=i18n("Silent training files"),
            info=i18n(
                "Adding several silent files to the training set enables the model to handle pure silence in inferred audio files. 2 is fine for 5-10 minute datasets"
            ),
            value=True,
            interactive=True,
        )
        extract_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        extract_button = gr.Button(i18n("Extract Features"))
        extract_button.click(
            fn=_extract_with_toast,
            inputs=[
                model_name,
                f0_method,
                cpu_cores,
                gpu,
                sampling_rate,
                embedder_model,
                include_mutes,
                model_version,
            ],
            outputs=[extract_output_info],
        )

    # Training section
    with gr.Accordion(i18n("Training")):
        with gr.Row():
            batch_size = gr.Slider(
                1,
                64,
                4,
                step=1,
                label=i18n("Batch Size"),
                info=i18n(
                    "It's advisable to use the highest your GPU can handle."
                ),
                interactive=True,
            )
            save_every_epoch = gr.Slider(
                1,
                100,
                10,
                step=1,
                label=i18n("Save Every Epoch"),
                info=i18n("Determine at how many epochs the model will saved at."),
                interactive=True,
            )
            total_epoch = gr.Slider(
                1,
                10000,
                200,
                step=1,
                label=i18n("Total Epoch"),
                info=i18n(
                    "Specifies the overall quantity of epochs for the model training process."
                ),
                interactive=True,
            )
        with gr.Accordion(i18n("Advanced Settings"), open=False):
            with gr.Row():
                with gr.Column():
                    save_only_latest = gr.Checkbox(
                        label=i18n("Save Only Latest"),
                        info=i18n(
                            "Enabling this setting will result in the G and D files saving only their most recent versions, effectively conserving storage space."
                        ),
                        value=True,
                        interactive=True,
                    )
                    save_every_weights = gr.Checkbox(
                        label=i18n("Save Every Weights"),
                        info=i18n(
                            "Saves an inference model at each epoch checkpoint interval when step-based inference saving is disabled."
                        ),
                        value=True,
                        interactive=True,
                    )
                    save_every_steps = gr.Slider(
                        0,
                        100000,
                        0,
                        step=1,
                        label=i18n("Save Inference Model Every X Steps"),
                        info=i18n(
                            "Saves an inference-only model every X training steps. Values above 0 disable epoch-based inference model saves."
                        ),
                        interactive=True,
                    )
                    pretrained = gr.Checkbox(
                        label=i18n("Pretrained"),
                        info=i18n(
                            "Utilize pretrained models when training your own. This approach reduces training duration and enhances overall quality."
                        ),
                        value=True,
                        interactive=True,
                    )
                with gr.Column():
                    cleanup = gr.Checkbox(
                        label=i18n("Fresh Training"),
                        info=i18n(
                            "Deletes all previously generated G_ and D_ checkpoints, spec.pt files, and tensorboard logs."
                        ),
                        value=False,
                        interactive=True,
                    )
                    cache_dataset_in_gpu = gr.Checkbox(
                        label=i18n("Cache Dataset in GPU"),
                        info=i18n(
                            "Cache the dataset in GPU memory to speed up the training process."
                        ),
                        value=False,
                        interactive=True,
                    )
                    checkpointing = gr.Checkbox(
                        label=i18n("Checkpointing"),
                        info=i18n(
                            "Enables memory-efficient training. This reduces VRAM usage at the cost of slower training speed. It is useful for GPUs with limited memory (e.g., <6GB VRAM) or when training with a batch size larger than what your GPU can normally accommodate."
                        ),
                        value=auto_enable_checkpointing(),
                        interactive=True,
                    )
                    shutdown_check = gr.Checkbox(
                        label=i18n("Shutdown after finishing"),
                        info=i18n(
                            "Automatically shut down computer when training is finished."
                        ),
                        value=False,
                        interactive=True,
                    )
            with gr.Row():
                custom_pretrained = gr.Checkbox(
                    label=i18n("Custom Pretrained"),
                    info=i18n(
                        "Utilizing custom pretrained models can lead to superior results, as selecting the most suitable pretrained models tailored to the specific use case can significantly enhance performance."
                    ),
                    value=False,
                    interactive=True,
                )
            with gr.Row():
                with gr.Column(visible=False) as pretrained_custom_settings:
                    with gr.Accordion(i18n("Pretrained Custom Settings")):
                        upload_pretrained = gr.File(
                            label=i18n("Upload Pretrained Model"),
                            type="filepath",
                            interactive=True,
                        )
                        refresh_custom_pretaineds_button = gr.Button(
                            i18n("Refresh Custom Pretraineds")
                        )
                        g_pretrained_path = gr.Dropdown(
                            label=i18n("Custom Pretrained G"),
                            info=i18n(
                                "Select the custom pretrained model for the generator."
                            ),
                            choices=sorted(pretraineds_list_g),
                            interactive=True,
                            allow_custom_value=True,
                        )
                        d_pretrained_path = gr.Dropdown(
                            label=i18n("Custom Pretrained D"),
                            info=i18n(
                                "Select the custom pretrained model for the discriminator."
                            ),
                            choices=sorted(pretraineds_list_d),
                            interactive=True,
                            allow_custom_value=True,
                        )

            index_algorithm = gr.Radio(
                label=i18n("Index Algorithm"),
                info=i18n(
                    "KMeans is a clustering algorithm that divides the dataset into K clusters. This setting is particularly useful for large datasets."
                ),
                choices=["Auto", "Faiss", "KMeans"],
                value="Auto",
                interactive=True,
            )

        def start_training(*args):
            message = start_train_script(*args)
            gr.Info(message)
            return training_ui_state(args[0])

        def training_ui_state(selected_model):
            state = get_train_state(selected_model)
            status = state.get("status", "idle")
            message = state.get("message", "Training is idle.")
            is_running = status == "running"
            is_paused = status == "paused"
            is_active = status in {"running", "paused", "stopping", "finalizing"}
            can_stop = status in {"running", "paused", "stopping"}

            if status == "idle":
                display_message = "Training is idle"
            elif status == "running":
                display_message = "Training is running"
            elif status == "paused":
                display_message = "Training paused"
            elif status == "stopped":
                display_message = "Training stopped"
            else:
                display_message = message

            return (
                gr.update(visible=not is_active),
                gr.update(visible=is_running),
                gr.update(visible=is_paused),
                gr.update(visible=can_stop),
                display_message,
            )

        def control_training(action, selected_model):
            message = action(selected_model)
            gr.Info(message)
            return training_ui_state(selected_model)

        training_status_info = gr.Textbox(
            label=i18n("Training Status"),
            value=i18n("Training is idle"),
            max_lines=2,
            interactive=False,
        )

        with gr.Row():
            train_button = gr.Button(i18n("Start Training"), variant="primary")

            pause_train_button = gr.Button(i18n("Pause Training"), visible=False)
            resume_train_button = gr.Button(
                i18n("Resume Training"), visible=False, variant="primary"
            )
            stop_train_button = gr.Button(
                i18n("Stop Training"), visible=False, variant="stop"
            )

            index_button = gr.Button(i18n("Generate Index"))
            index_button.click(
                fn=_index_with_toast,
                inputs=[model_name, index_algorithm],
                outputs=[],
            )

    # Export Model section
    with gr.Accordion(i18n("Export Model"), open=False):
        if os.getenv("COLAB_RELEASE_TAG"):
            gr.Markdown(
                i18n(
                    "The 'Upload' button packages the model into a .zip file and saves it to the ApplioExported folder in your Google Drive."
                )
            )
        with gr.Row():
            with gr.Column():
                pth_file_export = gr.File(
                    label=i18n("Exported Pth file"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                pth_dropdown_export = gr.Dropdown(
                    label=i18n("Pth file"),
                    info=i18n("Select the pth file to be exported"),
                    choices=get_pth_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
            with gr.Column():
                index_file_export = gr.File(
                    label=i18n("Exported Index File"),
                    type="filepath",
                    value=None,
                    interactive=False,
                )
                index_dropdown_export = gr.Dropdown(
                    label=i18n("Index File"),
                    info=i18n("Select the index file to be exported"),
                    choices=get_index_list(),
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
        with gr.Row():
            with gr.Column():
                refresh_export = gr.Button(i18n("Refresh"))
                if os.getenv("COLAB_RELEASE_TAG"):
                    upload_exported = gr.Button(i18n("Upload"))
                    upload_exported.click(
                        fn=upload_to_google_drive,
                        inputs=[pth_dropdown_export, index_dropdown_export, model_name],
                        outputs=[],
                    )

            def toggle_visible(checkbox):
                return gr.update(visible=checkbox)

            def toggle_pretrained(pretrained, custom_pretrained):
                if not custom_pretrained:
                    return gr.update(visible=pretrained), gr.update(visible=False)
                else:
                    return gr.update(visible=pretrained), gr.update(visible=pretrained)

            def toggle_version(version):
                if version == "v1":
                    return (
                        gr.update(
                            choices=["40000"], value="40000", visible=False
                        ),
                        gr.update(
                            choices=["HiFi-GAN"],
                            value="HiFi-GAN",
                            interactive=False,
                            visible=False,
                        ),
                        gr.update(
                            choices=["contentvec"],
                            value="contentvec",
                            interactive=False,
                        ),
                    )
                return (
                    gr.update(
                        choices=["32000", "40000", "48000"],
                        value="40000",
                        visible=True,
                    ),
                    gr.update(
                        choices=["HiFi-GAN", "RefineGAN"],
                        value="HiFi-GAN",
                        interactive=True,
                        visible=True,
                    ),
                    gr.update(
                        choices=["contentvec", "spin-v2"],
                        value="contentvec",
                        interactive=True,
                    ),
                )

            def toggle_vocoder(vocoder):
                if vocoder == "HiFi-GAN":
                    return {
                        "choices": ["32000", "40000", "48000"],
                        "__type__": "update",
                        "value": "40000",
                    }
                else:
                    return {
                        "choices": ["24000", "32000"],
                        "__type__": "update",
                        "value": "32000",
                    }

            def update_slider_visibility(noise_reduction):
                return gr.update(visible=noise_reduction)

            def update_truncate_silence_visibility(
                cut_method, truncate_enabled, action
            ):
                simple_selected = cut_method == "Simple"
                settings_visible = simple_selected and truncate_enabled
                return (
                    gr.update(visible=simple_selected),
                    gr.update(visible=settings_visible),
                    gr.update(visible=settings_visible),
                    gr.update(visible=settings_visible),
                    gr.update(visible=settings_visible and action == "truncate"),
                    gr.update(visible=settings_visible and action == "compress"),
                )

            def update_truncate_controls_visibility(
                truncate_enabled, cut_method, action
            ):
                visible = truncate_enabled and cut_method == "Simple"
                return (
                    gr.update(visible=visible),
                    gr.update(visible=visible),
                    gr.update(visible=visible),
                    gr.update(visible=visible and action == "truncate"),
                    gr.update(visible=visible and action == "compress"),
                )

            def update_truncate_action_visibility(
                action, truncate_enabled, cut_method
            ):
                visible = truncate_enabled and cut_method == "Simple"
                return (
                    gr.update(visible=visible and action == "truncate"),
                    gr.update(visible=visible and action == "compress"),
                )

            noise_reduction.change(
                fn=update_slider_visibility,
                inputs=noise_reduction,
                outputs=clean_strength,
            )
            cut_preprocess.change(
                fn=update_truncate_silence_visibility,
                inputs=[
                    cut_preprocess,
                    truncate_silence_enabled,
                    truncate_silence_action,
                ],
                outputs=[
                    truncate_silence_enabled,
                    truncate_silence_action,
                    truncate_silence_threshold_db,
                    truncate_silence_minimum_seconds,
                    truncate_silence_to_seconds,
                    truncate_silence_compress_percent,
                ],
            )
            truncate_silence_enabled.change(
                fn=update_truncate_controls_visibility,
                inputs=[
                    truncate_silence_enabled,
                    cut_preprocess,
                    truncate_silence_action,
                ],
                outputs=[
                    truncate_silence_action,
                    truncate_silence_threshold_db,
                    truncate_silence_minimum_seconds,
                    truncate_silence_to_seconds,
                    truncate_silence_compress_percent,
                ],
            )
            truncate_silence_action.change(
                fn=update_truncate_action_visibility,
                inputs=[
                    truncate_silence_action,
                    truncate_silence_enabled,
                    cut_preprocess,
                ],
                outputs=[
                    truncate_silence_to_seconds,
                    truncate_silence_compress_percent,
                ],
            )
            model_version.change(
                fn=toggle_version,
                inputs=[model_version],
                outputs=[sampling_rate, vocoder, embedder_model],
            )
            vocoder.change(
                fn=toggle_vocoder,
                inputs=[vocoder],
                outputs=[sampling_rate],
            )
            refresh.click(
                fn=refresh_models_and_datasets,
                inputs=[],
                outputs=[model_name, dataset_path],
            )
            dataset_creator.change(
                fn=toggle_visible,
                inputs=[dataset_creator],
                outputs=[dataset_creator_settings],
            )
            validation_audio_enabled.change(
                fn=toggle_visible,
                inputs=[validation_audio_enabled],
                outputs=[validation_audio_settings],
            )
            upload_audio_dataset.upload(
                fn=save_drop_dataset_audio,
                inputs=[upload_audio_dataset, dataset_name],
                outputs=[upload_audio_dataset, dataset_path],
            )
            upload_validation_audio.upload(
                fn=save_drop_validation_audio,
                inputs=[upload_validation_audio, dataset_path],
                outputs=[upload_validation_audio],
            )
            pretrained.change(
                fn=toggle_pretrained,
                inputs=[pretrained, custom_pretrained],
                outputs=[custom_pretrained, pretrained_custom_settings],
            )
            custom_pretrained.change(
                fn=toggle_visible,
                inputs=[custom_pretrained],
                outputs=[pretrained_custom_settings],
            )
            refresh_custom_pretaineds_button.click(
                fn=refresh_custom_pretraineds,
                inputs=[],
                outputs=[g_pretrained_path, d_pretrained_path],
            )
            upload_pretrained.upload(
                fn=save_drop_model,
                inputs=[upload_pretrained],
                outputs=[upload_pretrained],
            )
            training_outputs = [
                train_button,
                pause_train_button,
                resume_train_button,
                stop_train_button,
                training_status_info,
            ]

            train_button.click(
                fn=start_training,
                inputs=[
                    model_name,
                    save_every_epoch,
                    save_only_latest,
                    save_every_weights,
                    total_epoch,
                    sampling_rate,
                    batch_size,
                    gpu,
                    pretrained,
                    cleanup,
                    index_algorithm,
                    cache_dataset_in_gpu,
                    custom_pretrained,
                    g_pretrained_path,
                    d_pretrained_path,
                    vocoder,
                    checkpointing,
                    shutdown_check,
                    save_every_steps,
                    model_version,
                ],
                outputs=training_outputs,
            )

            pause_train_button.click(
                fn=lambda selected_model: control_training(pause_train, selected_model),
                inputs=[model_name],
                outputs=training_outputs,
            )
            resume_train_button.click(
                fn=lambda selected_model: control_training(resume_train, selected_model),
                inputs=[model_name],
                outputs=training_outputs,
            )
            stop_train_button.click(
                fn=lambda selected_model: control_training(stop_train, selected_model),
                inputs=[model_name],
                outputs=training_outputs,
            )

            training_status_timer = gr.Timer(value=2.0, active=True)
            training_status_timer.tick(
                fn=training_ui_state,
                inputs=[model_name],
                outputs=[
                    train_button,
                    pause_train_button,
                    resume_train_button,
                    stop_train_button,
                    training_status_info,
                ],
            )
            pth_dropdown_export.change(
                fn=export_pth,
                inputs=[pth_dropdown_export],
                outputs=[pth_file_export],
            )
            index_dropdown_export.change(
                fn=export_index,
                inputs=[index_dropdown_export],
                outputs=[index_file_export],
            )
            refresh_export.click(
                fn=refresh_pth_and_index_list,
                inputs=[],
                outputs=[pth_dropdown_export, index_dropdown_export],
            )
