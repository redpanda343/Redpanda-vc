import datetime
from functools import lru_cache
import os
import shutil
import sys
import traceback

import gradio as gr
import regex as re
import torch

from assets.i18n.i18n import I18nAuto
from core import (
    run_batch_infer_script,
    run_infer_script,
    run_multi_model_batch_infer_script,
    run_multi_model_infer_script,
)
from rvc.lib.utils import format_title
from tabs.settings.sections.filter import get_filter_trigger, load_config_filter
from tabs.settings.sections.restart import stop_infer

i18n = I18nAuto()

now_dir = os.getcwd()
sys.path.append(now_dir)

model_root = os.path.join(now_dir, "logs")
audio_root = os.path.join(now_dir, "assets", "audios")

os.makedirs(audio_root, exist_ok=True)

model_root_relative = os.path.relpath(model_root, now_dir)
audio_root_relative = os.path.relpath(audio_root, now_dir)

sup_audioext = {
    "wav",
    "mp3",
    "flac",
    "opus",
    "mp4",
    "aac",
    "alac",
    "wma",
    "aiff",
    "webm",
    "ac3",
}


def normalize_path(p):
    return os.path.normpath(p).replace("\\", "/").lower()


# BASE model/index folder names for many latin languages (legacy: zips = models)
MODEL_FOLDER = re.compile(r"^(?:model.{0,4}|mdl(?:s)?|weight.{0,4}|zip(?:s)?)$")
INDEX_FOLDER = re.compile(r"^(?:ind.{0,4}|idx(?:s)?)$")


def is_mdl_alias(name: str) -> bool:
    return bool(MODEL_FOLDER.match(name))


def is_idx_alias(name: str) -> bool:
    return bool(INDEX_FOLDER.match(name))


def alias_score(path: str, want_model: bool) -> int:
    """
    Handles duplicate files, compare file type to path and assign a score:
    2 = Path contains correct alias  (e.g., model file in 'modelos/' folder)
    1 = Path contains opposite alias (e.g., model file in 'index/' folder)
    0 = Path contains no recognized aliases
    """
    parts = normalize_path(os.path.dirname(path)).split("/")
    has_mdl = any(is_mdl_alias(p) for p in parts)
    has_idx = any(is_idx_alias(p) for p in parts)
    if want_model:
        return 2 if has_mdl else (1 if has_idx else 0)
    else:
        return 2 if has_idx else (1 if has_mdl else 0)


def get_files(type="model"):
    assert type in ("model", "index"), "Invalid type for get_files (models or index)"
    is_model = type == "model"
    exts = (".pth", ".onnx") if is_model else (".index",)
    exclude_prefixes = ("G_", "D_") if is_model else ("trained_",)

    best = {}
    order = 0

    for root, _, files in os.walk(model_root_relative, followlinks=True):
        for file in files:
            if not file.endswith(exts):
                continue
            if file.startswith(exclude_prefixes):
                continue

            full = os.path.join(root, file)
            real = os.path.realpath(full)
            score = alias_score(full, is_model)

            prev = best.get(real)
            if (
                prev is None
            ):  # Prefer higher score; if equal score, use first encountered
                best[real] = (score, order, full)
            else:
                prev_score, prev_order, _ = prev
                if score > prev_score:
                    best[real] = (score, prev_order, full)
            order += 1

    return [t[2] for t in sorted(best.values(), key=lambda x: x[1])]


default_weight = next(iter(get_files("model")), None)

audio_paths = [
    os.path.join(root, name)
    for root, _, files in os.walk(audio_root_relative, topdown=False)
    for name in files
    if name.endswith(tuple(sup_audioext))
    and root == audio_root_relative
    and "_output" not in name
]

def output_path_fn(input_audio_path):
    original_name_without_extension = os.path.basename(input_audio_path).rsplit(".", 1)[
        0
    ]
    new_name = original_name_without_extension + "_output.wav"
    output_path = os.path.join(os.path.dirname(input_audio_path), new_name)
    return output_path


def change_choices(model):
    if model:
        speakers = get_speakers_id(model)
    else:
        speakers = [0]

    models_list = sorted(get_files("model"))
    indexes_list = sorted(get_files("index"))

    audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    ]

    return (
        {"choices": models_list, "__type__": "update"},
        {"choices": indexes_list, "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
        {
            "choices": (
                sorted(speakers)
                if speakers is not None and isinstance(speakers, (list, tuple))
                else [0]
            ),
            "__type__": "update",
        },
        {"choices": models_list, "__type__": "update"},
        {"choices": sorted(audio_paths), "__type__": "update"},
    )


def change_multi_choices():
    current_audio_paths = [
        os.path.join(root, name)
        for root, _, files in os.walk(audio_root_relative, topdown=False)
        for name in files
        if name.endswith(tuple(sup_audioext))
        and root == audio_root_relative
        and "_output" not in name
    ]
    return (
        gr.update(
            choices=sorted(get_files("model"), key=extract_model_and_epoch)
        ),
        gr.update(choices=sorted(current_audio_paths)),
    )


def extract_model_and_epoch(path):
    base_name = os.path.basename(path)
    match = re.match(r"(.+?)_(\d+)e_", base_name)
    if match:
        model, epoch = match.groups()
        return model, int(epoch)
    return "", 0


def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file = record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".wav"
        target_path = os.path.join(audio_root_relative, os.path.basename(new_name))

        shutil.move(path_to_file, target_path)
        return target_path, output_path_fn(target_path)


def save_to_wav2(upload_audio):
    file_path = upload_audio
    formated_name = format_title(os.path.basename(file_path))
    target_path = os.path.join(audio_root_relative, formated_name)

    if os.path.exists(target_path):
        os.remove(target_path)

    shutil.copy(file_path, target_path)
    return target_path, output_path_fn(target_path)


def save_multi_audio(upload_audio):
    result = save_to_wav2(upload_audio)
    return result[0] if result else None


def delete_outputs():
    gr.Info(f"Outputs cleared!")
    for root, _, files in os.walk(audio_root_relative, topdown=False):
        for name in files:
            if name.endswith(tuple(sup_audioext)) and name.__contains__("_output"):
                os.remove(os.path.join(root, name))


def folders_same(
    a: str, b: str
) -> bool:  # Used to "pair" index and model folders based on path names
    """
    True if:
      1) The two normalized paths are totally identical..OR
      2) One lives under a MODEL_FOLDER and the other lives
         under an INDEX_FOLDER, at the same relative subpath
         i.e.  logs/models/miku  and  logs/index/miku  =  "SAME FOLDER"
    """
    a = normalize_path(a)
    b = normalize_path(b)
    if a == b:
        return True

    def split_after_alias(p):
        parts = p.split("/")
        for i, part in enumerate(parts):
            if is_mdl_alias(part) or is_idx_alias(part):
                base = part
                rel = "/".join(parts[i + 1 :])
                return base, rel
        return None, None

    base_a, rel_a = split_after_alias(a)
    base_b, rel_b = split_after_alias(b)

    if rel_a is None or rel_b is None:
        return False

    if rel_a == rel_b and (
        (is_mdl_alias(base_a) and is_idx_alias(base_b))
        or (is_idx_alias(base_a) and is_mdl_alias(base_b))
    ):
        return True
    return False


def match_index(model_file_value, available_indexes=None):
    if not model_file_value:
        return ""

    # Derive the information about the model's name and path for index matching
    model_folder = normalize_path(os.path.dirname(model_file_value))
    model_name = os.path.basename(model_file_value)
    base_name = os.path.splitext(model_name)[0]
    common = re.sub(r"[_\-\.\+](?:e|s|v|V)\d.*$", "", base_name)
    prefix_match = re.match(r"^(.*?)[_\-\.\+]", base_name)
    prefix = prefix_match.group(1) if prefix_match else None

    same_count = 0
    last_same = None
    same_substr = None
    same_prefixed = None
    external_exact = None
    external_substr = None
    external_pref = None

    indexes = available_indexes if available_indexes is not None else get_files("index")
    for idx in indexes:
        idx_folder = os.path.dirname(idx)
        idx_folder_n = normalize_path(idx_folder)
        idx_name = os.path.basename(idx)
        idx_base = os.path.splitext(idx_name)[0]

        in_same = folders_same(model_folder, idx_folder_n)
        if in_same:
            same_count += 1
            last_same = idx

            # 1) EXACT match to loaded model name and folders_same = True
            if idx_base == base_name:
                return idx

            # 2) Substring match to model name and folders_same
            if common in idx_base and same_substr is None:
                same_substr = idx

            # 3) Prefix match to model name and folders_same
            if prefix and idx_base.startswith(prefix) and same_prefixed is None:
                same_prefixed = idx

        # If it's NOT in a paired folder (folders_same = False) we look elseware:
        else:
            # 4) EXACT match to model name in external directory
            if idx_base == base_name and external_exact is None:
                external_exact = idx

            # 5) Substring match to model name in ED
            if common in idx_base and external_substr is None:
                external_substr = idx

            # 6) Prefix match to model name in ED
            if prefix and idx_base.startswith(prefix) and external_pref is None:
                external_pref = idx

    # Fallback: If there is exactly one index file in the same (or paired) folder,
    # we should assume that's the intended index file even if the name doesnt match
    if same_count == 1:
        return last_same

    # Then by remaining priority queue:
    if same_substr:
        return same_substr
    if same_prefixed:
        return same_prefixed
    if external_exact:
        return external_exact
    if external_substr:
        return external_substr
    if external_pref:
        return external_pref

    return ""


@lru_cache(maxsize=32)
def read_model_info(path, modified, size):
    model_data = torch.load(path, map_location="cpu", weights_only=True)
    rectified = model_data.get("kind") == "rectified_flow"
    speakers = model_data.get("speaker_count", model_data.get("speakers_id")) or 1
    return rectified, speakers


def get_model_info(model):
    if model:
        try:
            path = os.path.abspath(os.path.join(now_dir, model))
            stat = os.stat(path)
            return read_model_info(path, stat.st_mtime_ns, stat.st_size)
        except Exception:
            pass
    return False, 1


def get_speakers_id(model):
    return list(range(get_model_info(model)[1]))


def get_vocoders():
    root = os.path.join(now_dir, "rvc", "models", "pretraineds", "rectified")
    return sorted(
        os.path.relpath(os.path.join(folder, name), now_dir)
        for folder, _, files in os.walk(root)
        for name in files
        if name.lower().endswith((".pth", ".ckpt"))
    )


def update_vocoder_visibility(model):
    return gr.update(visible=get_model_info(model)[0])


def filter_dropdowns(filter_text):
    ft = (filter_text or "").lower()
    all_models = sorted(get_files("model"), key=extract_model_and_epoch)
    all_indexes = sorted(get_files("index"))
    filtered_models = [m for m in all_models if ft in m.lower()]
    filtered_indexes = [i for i in all_indexes if ft in i.lower()]
    return (gr.update(choices=filtered_models), gr.update(choices=filtered_indexes))


def update_filter_visibility(_):
    en = load_config_filter()
    if not en:
        box = gr.update(visible=False, value="")
        m_upd, i_upd = filter_dropdowns("")
        return box, m_upd, i_upd
    return gr.update(visible=True), gr.skip(), gr.skip()


# Inference tab
def inference_tab():
    trigger = get_filter_trigger()
    with gr.Column() as model_controls:
        with gr.Row():
            model_file = gr.Dropdown(
                label=i18n("Voice Model"),
                info=i18n("Select the voice model to use for the conversion."),
                choices=sorted(get_files("model"), key=extract_model_and_epoch),
                value=default_weight,
                interactive=True,
                allow_custom_value=True,
            )
            filter_box_inf = gr.Textbox(
                label=i18n("Filter"),
                info=i18n("Path must contain:"),
                placeholder=i18n("Type to filter..."),
                interactive=True,
                scale=0.1,
                visible=load_config_filter(),
            )
            index_file = gr.Dropdown(
                label=i18n("Index File"),
                info=i18n("Select the index file to use for the conversion."),
                choices=sorted(get_files("index")),
                value=match_index(default_weight),
                interactive=True,
                allow_custom_value=True,
            )
        rectified_vocoder_path = gr.Dropdown(
            label=i18n("Rectified Flow Vocoder"),
            info=i18n(
                "Optional: select a compatible NSF-HiFiGAN vocoder or enter its .pth/.ckpt path. Leave empty to use the model's recorded vocoder or the default. Keep any config.json beside the vocoder."
            ),
            choices=get_vocoders(),
            value=None,
            interactive=True,
            allow_custom_value=True,
            visible=get_model_info(default_weight)[0],
        )
        model_file.change(
            fn=update_vocoder_visibility,
            inputs=[model_file],
            outputs=[rectified_vocoder_path],
            show_progress=False,
        )
        filter_box_inf.blur(
            fn=filter_dropdowns,
            inputs=[filter_box_inf],
            outputs=[model_file, index_file],
        )
        trigger.change(
            fn=update_filter_visibility,
            inputs=[trigger],
            outputs=[filter_box_inf, model_file, index_file],
            show_progress=False,
        )
        with gr.Row():
            unload_button = gr.Button(i18n("Unload Voice"))
            refresh_button = gr.Button(i18n("Refresh"))
            refresh_button.click(
                fn=lambda: gr.update(choices=get_vocoders()),
                inputs=[],
                outputs=[rectified_vocoder_path],
                show_progress=False,
            )

            unload_button.click(
                fn=lambda: (
                    {"value": "", "__type__": "update"},
                    {"value": "", "__type__": "update"},
                ),
                inputs=[],
                outputs=[model_file, index_file],
            )
            model_file.select(
                fn=lambda model_file_value: match_index(model_file_value),
                inputs=[model_file],
                outputs=[index_file],
            )

    # Single inference tab
    with gr.Tab(i18n("Single")) as single_tab:
        with gr.Column():
            upload_audio = gr.Audio(
                label=i18n("Upload Audio"), type="filepath", editable=False
            )
            with gr.Row():
                audio = gr.Dropdown(
                    label=i18n("Select Audio"),
                    info=i18n("Select the audio to convert."),
                    choices=sorted(audio_paths),
                    value=audio_paths[0] if audio_paths else "",
                    interactive=True,
                    allow_custom_value=True,
                )

        with gr.Accordion(i18n("Advanced Settings"), open=False):
            with gr.Column():
                clear_outputs_infer = gr.Button(
                    i18n("Clear Outputs (Deletes all audios in assets/audios)")
                )
                output_path = gr.Textbox(
                    label=i18n("Output Path"),
                    placeholder=i18n("Enter output path"),
                    info=i18n(
                        "The path where the output audio will be saved, by default in assets/audios/output.wav"
                    ),
                    value=(
                        output_path_fn(audio_paths[0])
                        if audio_paths
                        else os.path.join(now_dir, "assets", "audios", "output.wav")
                    ),
                    interactive=True,
                )
                export_format = gr.Radio(
                    label=i18n("Export Format"),
                    info=i18n("Select the format to export the audio."),
                    choices=["WAV", "MP3", "FLAC"],
                    value="WAV",
                    interactive=True,
                )
                sid = gr.Dropdown(
                    label=i18n("Speaker ID"),
                    info=i18n("Select the speaker ID to use for the conversion."),
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio = gr.Checkbox(
                    label=i18n("Split Audio"),
                    info=i18n(
                        "Split the audio into chunks for inference to obtain better results in some cases."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                seed = gr.Textbox(
                    label=i18n("Seed"),
                    info=i18n(
                        "Keep this at 0 if you want a random seed."
                    ),
                    value="0",
                    placeholder=i18n("Enter an integer seed"),
                    max_lines=1,
                    interactive=True,
                )
                clean_audio = gr.Checkbox(
                    label=i18n("Clean Audio"),
                    info=i18n(
                        "Clean your audio output using noise detection algorithms, recommended for speaking audios."
                    ),
                    visible=False,
                    value=False,
                    interactive=True,
                )
                clean_strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Clean Strength"),
                    info=i18n(
                        "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                    ),
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                post_process = gr.Checkbox(
                    label=i18n("Post-Process"),
                    info=i18n("Post-process the audio to apply effects to the output."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb = gr.Checkbox(
                    label=i18n("Reverb"),
                    info=i18n("Apply reverb to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb_room_size = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Room Size"),
                    info=i18n("Set the room size of the reverb."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                reverb_damping = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Damping"),
                    info=i18n("Set the damping of the reverb."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                reverb_wet_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Wet Gain"),
                    info=i18n("Set the wet gain of the reverb."),
                    value=0.33,
                    interactive=True,
                    visible=False,
                )
                reverb_dry_gain = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Dry Gain"),
                    info=i18n("Set the dry gain of the reverb."),
                    value=0.4,
                    interactive=True,
                    visible=False,
                )
                reverb_width = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Width"),
                    info=i18n("Set the width of the reverb."),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                reverb_freeze_mode = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Freeze Mode"),
                    info=i18n("Set the freeze mode of the reverb."),
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                pitch_shift = gr.Checkbox(
                    label=i18n("Pitch Shift"),
                    info=i18n("Apply pitch shift to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_semitones = gr.Slider(
                    minimum=-12,
                    maximum=12,
                    label=i18n("Pitch Shift Semitones"),
                    info=i18n("Set the pitch shift semitones."),
                    value=0,
                    interactive=True,
                    visible=False,
                )
                limiter = gr.Checkbox(
                    label=i18n("Limiter"),
                    info=i18n("Apply limiter to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                limiter_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label=i18n("Limiter Threshold dB"),
                    info=i18n("Set the limiter threshold dB."),
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                limiter_release_time = gr.Slider(
                    minimum=0.01,
                    maximum=1,
                    label=i18n("Limiter Release Time"),
                    info=i18n("Set the limiter release time."),
                    value=0.05,
                    interactive=True,
                    visible=False,
                )
                gain = gr.Checkbox(
                    label=i18n("Gain"),
                    info=i18n("Apply gain to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                gain_db = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label=i18n("Gain dB"),
                    info=i18n("Set the gain dB."),
                    value=0,
                    interactive=True,
                    visible=False,
                )
                distortion = gr.Checkbox(
                    label=i18n("Distortion"),
                    info=i18n("Apply distortion to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                distortion_gain = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label=i18n("Distortion Gain"),
                    info=i18n("Set the distortion gain."),
                    value=25,
                    interactive=True,
                    visible=False,
                )
                chorus = gr.Checkbox(
                    label=i18n("Chorus"),
                    info=i18n("Apply chorus to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                chorus_rate = gr.Slider(
                    minimum=0,
                    maximum=100,
                    label=i18n("Chorus Rate Hz"),
                    info=i18n("Set the chorus rate Hz."),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                chorus_depth = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Chorus Depth"),
                    info=i18n("Set the chorus depth."),
                    value=0.25,
                    interactive=True,
                    visible=False,
                )
                chorus_center_delay = gr.Slider(
                    minimum=7,
                    maximum=8,
                    label=i18n("Chorus Center Delay ms"),
                    info=i18n("Set the chorus center delay ms."),
                    value=7,
                    interactive=True,
                    visible=False,
                )
                chorus_feedback = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Chorus Feedback"),
                    info=i18n("Set the chorus feedback."),
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                chorus_mix = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Chorus Mix"),
                    info=i18n("Set the chorus mix."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                bitcrush = gr.Checkbox(
                    label=i18n("Bitcrush"),
                    info=i18n("Apply bitcrush to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                bitcrush_bit_depth = gr.Slider(
                    minimum=1,
                    maximum=32,
                    label=i18n("Bitcrush Bit Depth"),
                    info=i18n("Set the bitcrush bit depth."),
                    value=8,
                    interactive=True,
                    visible=False,
                )
                clipping = gr.Checkbox(
                    label=i18n("Clipping"),
                    info=i18n("Apply clipping to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                clipping_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label=i18n("Clipping Threshold"),
                    info=i18n("Set the clipping threshold."),
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                compressor = gr.Checkbox(
                    label=i18n("Compressor"),
                    info=i18n("Apply compressor to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                compressor_threshold = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label=i18n("Compressor Threshold dB"),
                    info=i18n("Set the compressor threshold dB."),
                    value=0,
                    interactive=True,
                    visible=False,
                )
                compressor_ratio = gr.Slider(
                    minimum=1,
                    maximum=20,
                    label=i18n("Compressor Ratio"),
                    info=i18n("Set the compressor ratio."),
                    value=1,
                    interactive=True,
                    visible=False,
                )
                compressor_attack = gr.Slider(
                    minimum=0.0,
                    maximum=100,
                    label=i18n("Compressor Attack ms"),
                    info=i18n("Set the compressor attack ms."),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                compressor_release = gr.Slider(
                    minimum=0.01,
                    maximum=100,
                    label=i18n("Compressor Release ms"),
                    info=i18n("Set the compressor release ms."),
                    value=100,
                    interactive=True,
                    visible=False,
                )
                delay = gr.Checkbox(
                    label=i18n("Delay"),
                    info=i18n("Apply delay to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                delay_seconds = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    label=i18n("Delay Seconds"),
                    info=i18n("Set the delay seconds."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                delay_feedback = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label=i18n("Delay Feedback"),
                    info=i18n("Set the delay feedback."),
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                delay_mix = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label=i18n("Delay Mix"),
                    info=i18n("Set the delay mix."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=i18n("Pitch"),
                    info=i18n(
                        "Set the pitch of the audio, the higher the value, the higher the pitch."
                    ),
                    value=0,
                    interactive=True,
                )
                index_rate = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Search Feature Ratio"),
                    info=i18n(
                        "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                    ),
                    value=0.75,
                    interactive=True,
                )
                normalization_db = gr.Slider(
                    minimum=-12,
                    maximum=-0.5,
                    step=0.5,
                    label=i18n("Audio Normalization"),
                    info=i18n(
                        "Applies peak normalization to the output."
                    ),
                    value=-1,
                    interactive=True,
                    elem_id="normalization-db",
                )
                protect = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect Voiceless Consonants"),
                    info=i18n(
                        "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                    ),
                    value=0.5,
                    interactive=True,
                )
                f0_method = gr.Radio(
                    label=i18n("Pitch extraction algorithm"),
                    info=i18n(
                        "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                    ),
                    choices=[
                        "rmvpe",
                        "fcpe",
                        "swift",
                        ("parselmouth", "pm"),
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
        def convert_audio(*args):
            try:
                gr.Info(i18n("Converting audio..."))
                result = run_infer_script(*args)
                gr.Info(result[0])
                return result
            except Exception:
                traceback.print_exc()
                gr.Warning(
                    i18n(
                        "An error occurred during audio conversion. Please check the console logs for more details."
                    )
                )
                return (
                    "An error occurred during audio conversion. Please check the console logs for more details.",
                    None,
                )

        def convert_audio_batch(*args):
            try:
                return run_batch_infer_script(*args)
            except Exception:
                traceback.print_exc()
                return "An error occurred during audio batch conversion. Please check the console logs for more details."

        convert_button1 = gr.Button(i18n("Convert"))

        with gr.Row():
            vc_output1 = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
            )
            vc_output2 = gr.Audio(label=i18n("Export Audio"))

    # Batch inference tab
    with gr.Tab(i18n("Batch")) as batch_tab:
        with gr.Row():
            with gr.Column():
                input_folder_batch = gr.Textbox(
                    label=i18n("Input Folder"),
                    info=i18n("Select the folder containing the audios to convert."),
                    placeholder=i18n("Enter input path"),
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
                output_folder_batch = gr.Textbox(
                    label=i18n("Output Folder"),
                    info=i18n(
                        "Select the folder where the output audios will be saved."
                    ),
                    placeholder=i18n("Enter output path"),
                    value=os.path.join(now_dir, "assets", "audios"),
                    interactive=True,
                )
        with gr.Accordion(i18n("Advanced Settings"), open=False):
            with gr.Column():
                clear_outputs_batch = gr.Button(
                    i18n("Clear Outputs (Deletes all audios in assets/audios)")
                )
                export_format_batch = gr.Radio(
                    label=i18n("Export Format"),
                    info=i18n("Select the format to export the audio."),
                    choices=["WAV", "MP3", "FLAC"],
                    value="WAV",
                    interactive=True,
                )
                sid_batch = gr.Dropdown(
                    label=i18n("Speaker ID"),
                    info=i18n("Select the speaker ID to use for the conversion."),
                    choices=get_speakers_id(model_file.value),
                    value=0,
                    interactive=True,
                )
                split_audio_batch = gr.Checkbox(
                    label=i18n("Split Audio"),
                    info=i18n(
                        "Split the audio into chunks for inference to obtain better results in some cases."
                    ),
                    visible=True,
                    value=False,
                    interactive=True,
                )
                seed_batch = gr.Textbox(
                    label=i18n("Seed"),
                    info=i18n(
                        "Keep this at 0 if you want a random seed. Any value above 0 is deterministic."
                    ),
                    value="0",
                    placeholder=i18n("Enter an integer seed"),
                    max_lines=1,
                    interactive=True,
                )
                clean_audio_batch = gr.Checkbox(
                    label=i18n("Clean Audio"),
                    info=i18n(
                        "Clean your audio output using noise detection algorithms, recommended for speaking audios."
                    ),
                    visible=False,
                    value=False,
                    interactive=True,
                )
                clean_strength_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Clean Strength"),
                    info=i18n(
                        "Set the clean-up level to the audio you want, the more you increase it the more it will clean up, but it is possible that the audio will be more compressed."
                    ),
                    visible=False,
                    value=0.5,
                    interactive=True,
                )
                post_process_batch = gr.Checkbox(
                    label=i18n("Post-Process"),
                    info=i18n("Post-process the audio to apply effects to the output."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb_batch = gr.Checkbox(
                    label=i18n("Reverb"),
                    info=i18n("Apply reverb to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                reverb_room_size_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Room Size"),
                    info=i18n("Set the room size of the reverb."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                reverb_damping_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Damping"),
                    info=i18n("Set the damping of the reverb."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                reverb_wet_gain_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Wet Gain"),
                    info=i18n("Set the wet gain of the reverb."),
                    value=0.33,
                    interactive=True,
                    visible=False,
                )
                reverb_dry_gain_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Dry Gain"),
                    info=i18n("Set the dry gain of the reverb."),
                    value=0.4,
                    interactive=True,
                    visible=False,
                )
                reverb_width_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Width"),
                    info=i18n("Set the width of the reverb."),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                reverb_freeze_mode_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Reverb Freeze Mode"),
                    info=i18n("Set the freeze mode of the reverb."),
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_batch = gr.Checkbox(
                    label=i18n("Pitch Shift"),
                    info=i18n("Apply pitch shift to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                pitch_shift_semitones_batch = gr.Slider(
                    minimum=-12,
                    maximum=12,
                    label=i18n("Pitch Shift Semitones"),
                    info=i18n("Set the pitch shift semitones."),
                    value=0,
                    interactive=True,
                    visible=False,
                )
                limiter_batch = gr.Checkbox(
                    label=i18n("Limiter"),
                    info=i18n("Apply limiter to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                limiter_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label=i18n("Limiter Threshold dB"),
                    info=i18n("Set the limiter threshold dB."),
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                limiter_release_time_batch = gr.Slider(
                    minimum=0.01,
                    maximum=1,
                    label=i18n("Limiter Release Time"),
                    info=i18n("Set the limiter release time."),
                    value=0.05,
                    interactive=True,
                    visible=False,
                )
                gain_batch = gr.Checkbox(
                    label=i18n("Gain"),
                    info=i18n("Apply gain to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                gain_db_batch = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label=i18n("Gain dB"),
                    info=i18n("Set the gain dB."),
                    value=0,
                    interactive=True,
                    visible=False,
                )
                distortion_batch = gr.Checkbox(
                    label=i18n("Distortion"),
                    info=i18n("Apply distortion to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                distortion_gain_batch = gr.Slider(
                    minimum=-60,
                    maximum=60,
                    label=i18n("Distortion Gain"),
                    info=i18n("Set the distortion gain."),
                    value=25,
                    interactive=True,
                    visible=False,
                )
                chorus_batch = gr.Checkbox(
                    label=i18n("Chorus"),
                    info=i18n("Apply chorus to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                chorus_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=100,
                    label=i18n("Chorus Rate Hz"),
                    info=i18n("Set the chorus rate Hz."),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                chorus_depth_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Chorus Depth"),
                    info=i18n("Set the chorus depth."),
                    value=0.25,
                    interactive=True,
                    visible=False,
                )
                chorus_center_delay_batch = gr.Slider(
                    minimum=7,
                    maximum=8,
                    label=i18n("Chorus Center Delay ms"),
                    info=i18n("Set the chorus center delay ms."),
                    value=7,
                    interactive=True,
                    visible=False,
                )
                chorus_feedback_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Chorus Feedback"),
                    info=i18n("Set the chorus feedback."),
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                chorus_mix_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Chorus Mix"),
                    info=i18n("Set the chorus mix."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                bitcrush_batch = gr.Checkbox(
                    label=i18n("Bitcrush"),
                    info=i18n("Apply bitcrush to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                bitcrush_bit_depth_batch = gr.Slider(
                    minimum=1,
                    maximum=32,
                    label=i18n("Bitcrush Bit Depth"),
                    info=i18n("Set the bitcrush bit depth."),
                    value=8,
                    interactive=True,
                    visible=False,
                )
                clipping_batch = gr.Checkbox(
                    label=i18n("Clipping"),
                    info=i18n("Apply clipping to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                clipping_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label=i18n("Clipping Threshold"),
                    info=i18n("Set the clipping threshold."),
                    value=-6,
                    interactive=True,
                    visible=False,
                )
                compressor_batch = gr.Checkbox(
                    label=i18n("Compressor"),
                    info=i18n("Apply compressor to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                compressor_threshold_batch = gr.Slider(
                    minimum=-60,
                    maximum=0,
                    label=i18n("Compressor Threshold dB"),
                    info=i18n("Set the compressor threshold dB."),
                    value=0,
                    interactive=True,
                    visible=False,
                )
                compressor_ratio_batch = gr.Slider(
                    minimum=1,
                    maximum=20,
                    label=i18n("Compressor Ratio"),
                    info=i18n("Set the compressor ratio."),
                    value=1,
                    interactive=True,
                    visible=False,
                )
                compressor_attack_batch = gr.Slider(
                    minimum=0.0,
                    maximum=100,
                    label=i18n("Compressor Attack ms"),
                    info=i18n("Set the compressor attack ms."),
                    value=1.0,
                    interactive=True,
                    visible=False,
                )
                compressor_release_batch = gr.Slider(
                    minimum=0.01,
                    maximum=100,
                    label=i18n("Compressor Release ms"),
                    info=i18n("Set the compressor release ms."),
                    value=100,
                    interactive=True,
                    visible=False,
                )
                delay_batch = gr.Checkbox(
                    label=i18n("Delay"),
                    info=i18n("Apply delay to the audio."),
                    value=False,
                    interactive=True,
                    visible=False,
                )
                delay_seconds_batch = gr.Slider(
                    minimum=0.0,
                    maximum=5.0,
                    label=i18n("Delay Seconds"),
                    info=i18n("Set the delay seconds."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                delay_feedback_batch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label=i18n("Delay Feedback"),
                    info=i18n("Set the delay feedback."),
                    value=0.0,
                    interactive=True,
                    visible=False,
                )
                delay_mix_batch = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    label=i18n("Delay Mix"),
                    info=i18n("Set the delay mix."),
                    value=0.5,
                    interactive=True,
                    visible=False,
                )
                pitch_batch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    label=i18n("Pitch"),
                    info=i18n(
                        "Set the pitch of the audio, the higher the value, the higher the pitch."
                    ),
                    value=0,
                    interactive=True,
                )
                index_rate_batch = gr.Slider(
                    minimum=0,
                    maximum=1,
                    label=i18n("Search Feature Ratio"),
                    info=i18n(
                        "Influence exerted by the index file; a higher value corresponds to greater influence. However, opting for lower values can help mitigate artifacts present in the audio."
                    ),
                    value=0.75,
                    interactive=True,
                )
                normalization_db_batch = gr.Slider(
                    minimum=-12,
                    maximum=-0.5,
                    step=0.5,
                    label=i18n("Audio Normalization"),
                    info=i18n(
                        "Applies peak normalization to the output."
                    ),
                    value=-1,
                    interactive=True,
                    elem_id="normalization-db-batch",
                )
                protect_batch = gr.Slider(
                    minimum=0,
                    maximum=0.5,
                    label=i18n("Protect Voiceless Consonants"),
                    info=i18n(
                        "Safeguard distinct consonants and breathing sounds to prevent electro-acoustic tearing and other artifacts. Pulling the parameter to its maximum value of 0.5 offers comprehensive protection. However, reducing this value might decrease the extent of protection while potentially mitigating the indexing effect."
                    ),
                    value=0.5,
                    interactive=True,
                )
                f0_method_batch = gr.Radio(
                    label=i18n("Pitch extraction algorithm"),
                    info=i18n(
                        "Pitch extraction algorithm to use for the audio conversion. The default algorithm is rmvpe, which is recommended for most cases."
                    ),
                    choices=[
                        "rmvpe",
                        "fcpe",
                        "swift",
                        ("parselmouth", "pm"),
                    ],
                    value="rmvpe",
                    interactive=True,
                )
                embedder_model_batch = gr.Radio(
                    label=i18n("Embedder Model"),
                    info=i18n("Model used for learning speaker embedding."),
                    choices=[
                        "contentvec",
                        "spin-v2",
                    ],
                    value="contentvec",
                    interactive=True,
                )
        convert_button_batch = gr.Button(i18n("Convert"))
        stop_button = gr.Button(i18n("Stop convert"), visible=False)
        stop_button.click(fn=stop_infer, inputs=[], outputs=[])

        with gr.Row():
            vc_output3 = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
            )

    with gr.Tab(i18n("Multi-Model infer")) as multi_model_tab:
        multi_upload_audio = gr.Audio(
            label=i18n("Upload Audio"), type="filepath", editable=False
        )
        with gr.Row():
            multi_audio = gr.Dropdown(
                label=i18n("Select Audio"),
                info=i18n("Select one audio file to convert with every model."),
                choices=sorted(audio_paths),
                value=audio_paths[0] if audio_paths else "",
                interactive=True,
                allow_custom_value=True,
            )
            multi_model_files = gr.Dropdown(
                label=i18n("Voice Models"),
                info=i18n(
                    "Select the voice models to use. The matching index is selected automatically for each model."
                ),
                choices=sorted(get_files("model"), key=extract_model_and_epoch),
                value=[],
                multiselect=True,
                interactive=True,
            )
        multi_index_refresh = gr.State(0)

        @gr.render(inputs=[multi_model_files, multi_index_refresh])
        def render_multi_index_selectors(model_paths, _):
            available_indexes = sorted(get_files("index"))
            index_dropdowns = []
            if model_paths:
                gr.Markdown(i18n("### Index Files for Selected Models"))
            for model_path in model_paths or []:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                index_dropdowns.append(
                    gr.Dropdown(
                        label=f"{model_name}: {i18n('Index File')}",
                        info=i18n(
                            "Select the index file for this model, or leave it empty."
                        ),
                        choices=[""] + available_indexes,
                        value=match_index(model_path, available_indexes),
                        interactive=True,
                        allow_custom_value=True,
                        key=("multi-model-index", model_path),
                    )
                )

            convert_button_multi.click(
                fn=convert_audio_multi_model,
                inputs=[
                    multi_pitch,
                    multi_index_rate,
                    multi_normalization_db,
                    multi_protect,
                    multi_f0_method,
                    multi_audio,
                    multi_output_folder,
                    multi_model_files,
                    multi_split_audio,
                    multi_export_format,
                    multi_embedder_model,
                    multi_sid,
                    multi_seed,
                    *index_dropdowns,
                ],
                outputs=[vc_output_multi_info, multi_audio_results],
            )
        with gr.Row():
            multi_output_folder = gr.Textbox(
                label=i18n("Output Folder"),
                info=i18n("Each model creates a separate, uniquely named audio file."),
                value=os.path.join(now_dir, "assets", "audios"),
                interactive=True,
            )
            multi_refresh_button = gr.Button(i18n("Refresh"), scale=0)
        with gr.Accordion(i18n("Advanced Settings"), open=False):
            multi_export_format = gr.Radio(
                label=i18n("Export Format"),
                info=i18n("Select the format to export the audio."),
                choices=["WAV", "MP3", "FLAC"],
                value="WAV",
                interactive=True,
            )
            multi_sid = gr.Textbox(
                label=i18n("Speaker ID"),
                info=i18n("Speaker ID used by every selected model."),
                value="0",
                max_lines=1,
                interactive=True,
            )
            multi_split_audio = gr.Checkbox(
                label=i18n("Split Audio"),
                info=i18n(
                    "Split the audio into chunks for inference to obtain better results in some cases."
                ),
                value=False,
                interactive=True,
            )
            multi_seed = gr.Textbox(
                label=i18n("Seed"),
                info=i18n("Keep this at 0 if you want a random seed."),
                value="0",
                placeholder=i18n("Enter an integer seed"),
                max_lines=1,
                interactive=True,
            )
            multi_pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=i18n("Pitch"),
                info=i18n(
                    "Set the pitch of the audio, the higher the value, the higher the pitch."
                ),
                value=0,
                interactive=True,
            )
            multi_index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Search Feature Ratio"),
                info=i18n("Set the influence of each model's matched index file."),
                value=0.75,
                interactive=True,
            )
            multi_normalization_db = gr.Slider(
                minimum=-12,
                maximum=-0.5,
                step=0.5,
                label=i18n("Audio Normalization"),
                info=i18n("Applies peak normalization to each output."),
                value=-1,
                interactive=True,
                elem_id="normalization-db-multi-model",
            )
            multi_protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("Protect Voiceless Consonants"),
                info=i18n("Protect consonants and breathing sounds from artifacts."),
                value=0.5,
                interactive=True,
            )
            multi_f0_method = gr.Radio(
                label=i18n("Pitch extraction algorithm"),
                info=i18n("Pitch extraction algorithm used by every model."),
                choices=["rmvpe", "fcpe", "swift", ("parselmouth", "pm")],
                value="rmvpe",
                interactive=True,
            )
            multi_embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for generating content features."),
                choices=["contentvec", "spin-v2"],
                value="contentvec",
                interactive=True,
            )

        convert_button_multi = gr.Button(i18n("Convert"))
        multi_audio_results = gr.State([])
        with gr.Row():
            vc_output_multi_info = gr.Textbox(
                label=i18n("Output Information"),
                info=i18n("The output information will be displayed here."),
            )
            with gr.Column():
                @gr.render(inputs=multi_audio_results)
                def render_multi_audio_results(audio_results):
                    if not audio_results:
                        gr.Markdown(i18n("Converted model audios will appear here."))
                    for position, (model_name, audio_path) in enumerate(audio_results):
                        gr.Audio(
                            value=audio_path,
                            label=f"{model_name}: audio",
                            type="filepath",
                            editable=False,
                            buttons=["download"],
                            key=(position, audio_path),
                        )

    with gr.Tab(i18n("Batch Multi-Model infer")) as batch_multi_model_tab:
        with gr.Row():
            batch_multi_input_folder = gr.Textbox(
                label=i18n("Input Folder"),
                info=i18n("Select the folder containing the audios to convert."),
                placeholder=i18n("Enter input path"),
                value=os.path.join(now_dir, "assets", "audios"),
                interactive=True,
            )
            batch_multi_output_folder = gr.Textbox(
                label=i18n("Output Folder"),
                info=i18n(
                    "Each model saves its converted audios in a separate subfolder."
                ),
                placeholder=i18n("Enter output path"),
                value=os.path.join(now_dir, "assets", "audios"),
                interactive=True,
            )
        with gr.Row():
            batch_multi_model_files = gr.Dropdown(
                label=i18n("Voice Models"),
                info=i18n("Select the voice models to use for the batch."),
                choices=sorted(get_files("model"), key=extract_model_and_epoch),
                value=[],
                multiselect=True,
                interactive=True,
            )
            batch_multi_refresh_button = gr.Button(i18n("Refresh"), scale=0)

        batch_multi_index_refresh = gr.State(0)

        @gr.render(inputs=[batch_multi_model_files, batch_multi_index_refresh])
        def render_batch_multi_index_selectors(model_paths, _):
            available_indexes = sorted(get_files("index"))
            index_dropdowns = []
            if model_paths:
                gr.Markdown(i18n("### Index Files for Selected Models"))
            for model_path in model_paths or []:
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                index_dropdowns.append(
                    gr.Dropdown(
                        label=f"{model_name}: {i18n('Index File')}",
                        info=i18n(
                            "Select the index file for this model, or leave it empty."
                        ),
                        choices=[""] + available_indexes,
                        value=match_index(model_path, available_indexes),
                        interactive=True,
                        allow_custom_value=True,
                        key=("batch-multi-model-index", model_path),
                    )
                )

            convert_button_batch_multi.click(
                fn=enable_stop_convert_button,
                inputs=[],
                outputs=[convert_button_batch_multi, stop_button_batch_multi],
            ).then(
                fn=convert_audio_batch_multi_model,
                inputs=[
                    batch_multi_pitch,
                    batch_multi_index_rate,
                    batch_multi_normalization_db,
                    batch_multi_protect,
                    batch_multi_f0_method,
                    batch_multi_input_folder,
                    batch_multi_output_folder,
                    batch_multi_model_files,
                    batch_multi_split_audio,
                    batch_multi_export_format,
                    batch_multi_embedder_model,
                    batch_multi_sid,
                    batch_multi_seed,
                    *index_dropdowns,
                ],
                outputs=[batch_multi_output_info],
            ).then(
                fn=disable_stop_convert_button,
                inputs=[],
                outputs=[convert_button_batch_multi, stop_button_batch_multi],
            )

        with gr.Accordion(i18n("Advanced Settings"), open=False):
            batch_multi_export_format = gr.Radio(
                label=i18n("Export Format"),
                info=i18n("Select the format to export the audio."),
                choices=["WAV", "MP3", "FLAC"],
                value="WAV",
                interactive=True,
            )
            batch_multi_sid = gr.Textbox(
                label=i18n("Speaker ID"),
                info=i18n("Speaker ID used by every selected model."),
                value="0",
                max_lines=1,
                interactive=True,
            )
            batch_multi_split_audio = gr.Checkbox(
                label=i18n("Split Audio"),
                info=i18n(
                    "Split each audio into chunks for inference to obtain better results in some cases."
                ),
                value=False,
                interactive=True,
            )
            batch_multi_seed = gr.Textbox(
                label=i18n("Seed"),
                info=i18n("Keep this at 0 if you want a random seed."),
                value="0",
                placeholder=i18n("Enter an integer seed"),
                max_lines=1,
                interactive=True,
            )
            batch_multi_pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                label=i18n("Pitch"),
                info=i18n("Set the pitch used for every audio and model."),
                value=0,
                interactive=True,
            )
            batch_multi_index_rate = gr.Slider(
                minimum=0,
                maximum=1,
                label=i18n("Search Feature Ratio"),
                info=i18n("Set the influence of each model's selected index file."),
                value=0.75,
                interactive=True,
            )
            batch_multi_normalization_db = gr.Slider(
                minimum=-12,
                maximum=-0.5,
                step=0.5,
                label=i18n("Audio Normalization"),
                info=i18n("Applies peak normalization to each output."),
                value=-1,
                interactive=True,
                elem_id="normalization-db-batch-multi-model",
            )
            batch_multi_protect = gr.Slider(
                minimum=0,
                maximum=0.5,
                label=i18n("Protect Voiceless Consonants"),
                info=i18n("Protect consonants and breathing sounds from artifacts."),
                value=0.5,
                interactive=True,
            )
            batch_multi_f0_method = gr.Radio(
                label=i18n("Pitch extraction algorithm"),
                info=i18n("Pitch extraction algorithm used by every model."),
                choices=["rmvpe", "fcpe", "swift", ("parselmouth", "pm")],
                value="rmvpe",
                interactive=True,
            )
            batch_multi_embedder_model = gr.Radio(
                label=i18n("Embedder Model"),
                info=i18n("Model used for generating content features."),
                choices=["contentvec", "spin-v2"],
                value="contentvec",
                interactive=True,
            )

        convert_button_batch_multi = gr.Button(i18n("Convert"))
        stop_button_batch_multi = gr.Button(i18n("Stop convert"), visible=False)
        stop_button_batch_multi.click(fn=stop_infer, inputs=[], outputs=[])
        batch_multi_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("Model output folders will be displayed here."),
            lines=6,
        )

    def convert_audio_multi_model(
        pitch,
        index_rate,
        normalization_db,
        protect,
        f0_method,
        input_path,
        output_folder,
        model_paths,
        split_audio,
        export_format,
        embedder_model,
        sid,
        seed,
        *index_paths,
    ):
        try:
            gr.Info(i18n("Converting audio with the selected models..."))
            result = run_multi_model_infer_script(
                pitch,
                index_rate,
                normalization_db,
                protect,
                f0_method,
                input_path,
                output_folder,
                model_paths,
                index_paths,
                split_audio,
                export_format,
                embedder_model,
                sid,
                seed,
            )
            gr.Info(result[0])
            return result
        except Exception:
            traceback.print_exc()
            message = i18n(
                "An error occurred during multi-model inference. Please check the console logs for more details."
            )
            gr.Warning(message)
            return message, []

    def convert_audio_batch_multi_model(
        pitch,
        index_rate,
        normalization_db,
        protect,
        f0_method,
        input_folder,
        output_folder,
        model_paths,
        split_audio,
        export_format,
        embedder_model,
        sid,
        seed,
        *index_paths,
    ):
        try:
            gr.Info(i18n("Converting the batch with the selected models..."))
            result = run_multi_model_batch_infer_script(
                pitch,
                index_rate,
                normalization_db,
                protect,
                f0_method,
                input_folder,
                output_folder,
                model_paths,
                index_paths,
                split_audio,
                export_format,
                embedder_model,
                sid,
                seed,
            )
            gr.Info(result.splitlines()[0])
            return result
        except Exception:
            traceback.print_exc()
            message = i18n(
                "An error occurred during multi-model batch inference. Please check the console logs for more details."
            )
            gr.Warning(message)
            return message

    def toggle_visible(checkbox):
        return {"visible": checkbox, "__type__": "update"}

    def enable_stop_convert_button():
        return {"visible": False, "__type__": "update"}, {
            "visible": True,
            "__type__": "update",
        }

    def disable_stop_convert_button():
        return {"visible": True, "__type__": "update"}, {
            "visible": False,
            "__type__": "update",
        }

    def update_visibility(checkbox, count):
        return [gr.update(visible=checkbox) for _ in range(count)]

    def post_process_visible(checkbox):
        return update_visibility(checkbox, 10)

    def reverb_visible(checkbox):
        return update_visibility(checkbox, 6)

    def limiter_visible(checkbox):
        return update_visibility(checkbox, 2)

    def chorus_visible(checkbox):
        return update_visibility(checkbox, 6)

    def compress_visible(checkbox):
        return update_visibility(checkbox, 4)

    def delay_visible(checkbox):
        return update_visibility(checkbox, 3)

    single_tab.select(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[model_controls],
        show_progress=False,
    )
    batch_tab.select(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[model_controls],
        show_progress=False,
    )
    multi_model_tab.select(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[model_controls],
        show_progress=False,
    )
    batch_multi_model_tab.select(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[model_controls],
        show_progress=False,
    )
    clean_audio.change(
        fn=toggle_visible,
        inputs=[clean_audio],
        outputs=[clean_strength],
    )
    post_process.change(
        fn=post_process_visible,
        inputs=[post_process],
        outputs=[
            reverb,
            pitch_shift,
            limiter,
            gain,
            distortion,
            chorus,
            bitcrush,
            clipping,
            compressor,
            delay,
        ],
    )
    reverb.change(
        fn=reverb_visible,
        inputs=[reverb],
        outputs=[
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
            reverb_freeze_mode,
        ],
    )
    pitch_shift.change(
        fn=toggle_visible,
        inputs=[pitch_shift],
        outputs=[pitch_shift_semitones],
    )
    limiter.change(
        fn=limiter_visible,
        inputs=[limiter],
        outputs=[limiter_threshold, limiter_release_time],
    )
    gain.change(
        fn=toggle_visible,
        inputs=[gain],
        outputs=[gain_db],
    )
    distortion.change(
        fn=toggle_visible,
        inputs=[distortion],
        outputs=[distortion_gain],
    )
    chorus.change(
        fn=chorus_visible,
        inputs=[chorus],
        outputs=[
            chorus_rate,
            chorus_depth,
            chorus_center_delay,
            chorus_feedback,
            chorus_mix,
        ],
    )
    bitcrush.change(
        fn=toggle_visible,
        inputs=[bitcrush],
        outputs=[bitcrush_bit_depth],
    )
    clipping.change(
        fn=toggle_visible,
        inputs=[clipping],
        outputs=[clipping_threshold],
    )
    compressor.change(
        fn=compress_visible,
        inputs=[compressor],
        outputs=[
            compressor_threshold,
            compressor_ratio,
            compressor_attack,
            compressor_release,
        ],
    )
    delay.change(
        fn=delay_visible,
        inputs=[delay],
        outputs=[delay_seconds, delay_feedback, delay_mix],
    )
    post_process_batch.change(
        fn=post_process_visible,
        inputs=[post_process_batch],
        outputs=[
            reverb_batch,
            pitch_shift_batch,
            limiter_batch,
            gain_batch,
            distortion_batch,
            chorus_batch,
            bitcrush_batch,
            clipping_batch,
            compressor_batch,
            delay_batch,
        ],
    )
    reverb_batch.change(
        fn=reverb_visible,
        inputs=[reverb_batch],
        outputs=[
            reverb_room_size_batch,
            reverb_damping_batch,
            reverb_wet_gain_batch,
            reverb_dry_gain_batch,
            reverb_width_batch,
            reverb_freeze_mode_batch,
        ],
    )
    pitch_shift_batch.change(
        fn=toggle_visible,
        inputs=[pitch_shift_batch],
        outputs=[pitch_shift_semitones_batch],
    )
    limiter_batch.change(
        fn=limiter_visible,
        inputs=[limiter_batch],
        outputs=[limiter_threshold_batch, limiter_release_time_batch],
    )
    gain_batch.change(
        fn=toggle_visible,
        inputs=[gain_batch],
        outputs=[gain_db_batch],
    )
    distortion_batch.change(
        fn=toggle_visible,
        inputs=[distortion_batch],
        outputs=[distortion_gain_batch],
    )
    chorus_batch.change(
        fn=chorus_visible,
        inputs=[chorus_batch],
        outputs=[
            chorus_rate_batch,
            chorus_depth_batch,
            chorus_center_delay_batch,
            chorus_feedback_batch,
            chorus_mix_batch,
        ],
    )
    bitcrush_batch.change(
        fn=toggle_visible,
        inputs=[bitcrush_batch],
        outputs=[bitcrush_bit_depth_batch],
    )
    clipping_batch.change(
        fn=toggle_visible,
        inputs=[clipping_batch],
        outputs=[clipping_threshold_batch],
    )
    compressor_batch.change(
        fn=compress_visible,
        inputs=[compressor_batch],
        outputs=[
            compressor_threshold_batch,
            compressor_ratio_batch,
            compressor_attack_batch,
            compressor_release_batch,
        ],
    )
    delay_batch.change(
        fn=delay_visible,
        inputs=[delay_batch],
        outputs=[delay_seconds_batch, delay_feedback_batch, delay_mix_batch],
    )
    clean_audio_batch.change(
        fn=toggle_visible,
        inputs=[clean_audio_batch],
        outputs=[clean_strength_batch],
    )
    refresh_button.click(
        fn=change_choices,
        inputs=[model_file],
        outputs=[
            model_file,
            index_file,
            audio,
            sid,
            sid_batch,
            multi_model_files,
            multi_audio,
        ],
    ).then(
        fn=filter_dropdowns,
        inputs=[filter_box_inf],
        outputs=[model_file, index_file],
    )
    multi_refresh_button.click(
        fn=change_multi_choices,
        inputs=[],
        outputs=[multi_model_files, multi_audio],
        show_progress=False,
    ).then(
        fn=lambda refresh_count: refresh_count + 1,
        inputs=[multi_index_refresh],
        outputs=[multi_index_refresh],
        show_progress=False,
    )
    batch_multi_refresh_button.click(
        fn=lambda: gr.update(
            choices=sorted(get_files("model"), key=extract_model_and_epoch)
        ),
        inputs=[],
        outputs=[batch_multi_model_files],
        show_progress=False,
    ).then(
        fn=lambda refresh_count: refresh_count + 1,
        inputs=[batch_multi_index_refresh],
        outputs=[batch_multi_index_refresh],
        show_progress=False,
    )
    audio.change(
        fn=output_path_fn,
        inputs=[audio],
        outputs=[output_path],
    )
    upload_audio.upload(
        fn=save_to_wav2,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    upload_audio.stop_recording(
        fn=save_to_wav,
        inputs=[upload_audio],
        outputs=[audio, output_path],
    )
    multi_upload_audio.upload(
        fn=save_multi_audio,
        inputs=[multi_upload_audio],
        outputs=[multi_audio],
    )
    multi_upload_audio.stop_recording(
        fn=save_multi_audio,
        inputs=[multi_upload_audio],
        outputs=[multi_audio],
    )
    clear_outputs_infer.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    clear_outputs_batch.click(
        fn=delete_outputs,
        inputs=[],
        outputs=[],
    )
    convert_button1.click(
        fn=convert_audio,
        inputs=[
            pitch,
            index_rate,
            normalization_db,
            protect,
            f0_method,
            audio,
            output_path,
            model_file,
            index_file,
            split_audio,
            clean_audio,
            clean_strength,
            export_format,
            embedder_model,
            post_process,
            reverb,
            pitch_shift,
            limiter,
            gain,
            distortion,
            chorus,
            bitcrush,
            clipping,
            compressor,
            delay,
            reverb_room_size,
            reverb_damping,
            reverb_wet_gain,
            reverb_dry_gain,
            reverb_width,
            reverb_freeze_mode,
            pitch_shift_semitones,
            limiter_threshold,
            limiter_release_time,
            gain_db,
            distortion_gain,
            chorus_rate,
            chorus_depth,
            chorus_center_delay,
            chorus_feedback,
            chorus_mix,
            bitcrush_bit_depth,
            clipping_threshold,
            compressor_threshold,
            compressor_ratio,
            compressor_attack,
            compressor_release,
            delay_seconds,
            delay_feedback,
            delay_mix,
            sid,
            seed,
            rectified_vocoder_path,
        ],
        outputs=[vc_output1, vc_output2],
    )
    convert_button_batch.click(
        fn=enable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    ).then(
        fn=convert_audio_batch,
        inputs=[
            pitch_batch,
            index_rate_batch,
            normalization_db_batch,
            protect_batch,
            f0_method_batch,
            input_folder_batch,
            output_folder_batch,
            model_file,
            index_file,
            split_audio_batch,
            clean_audio_batch,
            clean_strength_batch,
            export_format_batch,
            embedder_model_batch,
            post_process_batch,
            reverb_batch,
            pitch_shift_batch,
            limiter_batch,
            gain_batch,
            distortion_batch,
            chorus_batch,
            bitcrush_batch,
            clipping_batch,
            compressor_batch,
            delay_batch,
            reverb_room_size_batch,
            reverb_damping_batch,
            reverb_wet_gain_batch,
            reverb_dry_gain_batch,
            reverb_width_batch,
            reverb_freeze_mode_batch,
            pitch_shift_semitones_batch,
            limiter_threshold_batch,
            limiter_release_time_batch,
            gain_db_batch,
            distortion_gain_batch,
            chorus_rate_batch,
            chorus_depth_batch,
            chorus_center_delay_batch,
            chorus_feedback_batch,
            chorus_mix_batch,
            bitcrush_bit_depth_batch,
            clipping_threshold_batch,
            compressor_threshold_batch,
            compressor_ratio_batch,
            compressor_attack_batch,
            compressor_release_batch,
            delay_seconds_batch,
            delay_feedback_batch,
            delay_mix_batch,
            sid_batch,
            seed_batch,
            rectified_vocoder_path,
        ],
        outputs=[vc_output3],
    ).then(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
    stop_button.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch, stop_button],
    )
    stop_button_batch_multi.click(
        fn=disable_stop_convert_button,
        inputs=[],
        outputs=[convert_button_batch_multi, stop_button_batch_multi],
    )
