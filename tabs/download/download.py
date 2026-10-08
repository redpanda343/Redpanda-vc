import os
import sys
import json
import shutil
import requests
import tempfile
import gradio as gr
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(now_dir)

from core import run_download_script
from shared.utils import format_title

from assets.i18n.i18n import I18nAuto

i18n = I18nAuto()

gradio_temp_dir = os.path.join(tempfile.gettempdir(), "gradio")

PRETRAINED_MODELS = {
    "Legacy core 1.5 NEW": [
        (
            "https://huggingface.co/lyery/test/resolve/main/G_2333333%20%286%29.pth",
            "G_legacy_core15_new.pth",
        ),
        (
            "https://huggingface.co/lyery/test/resolve/main/D_2333333%20%286%29.pth",
            "D_legacy_core15_new.pth",
        ),
    ],
    "Legacy core 1.6": [
        (
            "https://huggingface.co/lyery/legacy_core1.6/resolve/main/G_11.pth",
            "G_legacy_core16.pth",
        ),
        (
            "https://huggingface.co/lyery/legacy_core1.6/resolve/main/D_11.pth",
            "D_legacy_core16.pth",
        ),
    ],
    "Legacy core 1.5 OLD 32k": [
        (
            "https://huggingface.co/lyery/mode4/resolve/main/G_15.pth",
            "G_legacy_core15_old_32k.pth",
        ),
        (
            "https://huggingface.co/lyery/mode4/resolve/main/D_15.pth",
            "D_legacy_core15_old_32k.pth",
        ),
    ],
    "Legacy core 1.5 OLD 40k": [
        (
            "https://huggingface.co/lyery/mode4/resolve/main/G_40k.pth",
            "G_legacy_core15_old_40k.pth",
        ),
        (
            "https://huggingface.co/lyery/mode4/resolve/main/D_40k.pth",
            "D_legacy_core15_old_40k.pth",
        ),
    ],
    "Legacy core 1.5 OLD 48k": [
        (
            "https://huggingface.co/lyery/mode4/resolve/main/G_48k.pth",
            "G_legacy_core15_old_48k.pth",
        ),
        (
            "https://huggingface.co/lyery/mode4/resolve/main/D_48k.pth",
            "D_legacy_core15_old_48k.pth",
        ),
    ],
}

if os.path.exists(gradio_temp_dir):
    shutil.rmtree(gradio_temp_dir)


def save_drop_model(dropbox):
    if "pth" not in dropbox and "index" not in dropbox:
        raise gr.Error(
            message="The file you dropped is not a valid model file. Please try again."
        )

    file_name = format_title(os.path.basename(dropbox))
    model_name = file_name

    if ".pth" in model_name:
        model_name = model_name.split(".pth")[0]
    elif ".index" in model_name:
        replacements = ["nprobe_1_", "_v1", "_v2", "added_"]
        for rep in replacements:
            model_name = model_name.replace(rep, "")
        model_name = model_name.split(".index")[0]

    model_path = os.path.join(now_dir, "logs", model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if os.path.exists(os.path.join(model_path, file_name)):
        os.remove(os.path.join(model_path, file_name))
    shutil.move(dropbox, os.path.join(model_path, file_name))
    print(f"{file_name} saved in {model_path}")
    gr.Info(f"{file_name} saved in {model_path}")

    return None


json_url = "https://huggingface.co/IAHispano/Applio/raw/main/pretrains.json"


def fetch_pretrained_data():
    pretraineds_custom_path = os.path.join("models", "pretraineds", "custom")
    os.makedirs(pretraineds_custom_path, exist_ok=True)
    try:
        with open(
            os.path.join(pretraineds_custom_path, json_url.split("/")[-1]),
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
    except:
        try:
            response = requests.get(json_url)
            response.raise_for_status()
            data = response.json()
            with open(
                os.path.join(pretraineds_custom_path, json_url.split("/")[-1]),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    separators=(",", ": "),
                    ensure_ascii=False,
                )
        except:
            data = {
                "Titan": {
                    "32k": {"D": "null", "G": "null"},
                },
            }
    return data


def get_pretrained_list():
    data = fetch_pretrained_data()
    return list(data.keys())


def get_pretrained_sample_rates(model):
    data = fetch_pretrained_data()
    return list(data[model].keys())


def get_file_size(url):
    response = None
    try:
        response = requests.head(
            url, allow_redirects=True, timeout=(10, 30)
        )
        response.raise_for_status()
        return int(response.headers.get("content-length", 0))
    except requests.RequestException:
        return 0
    finally:
        if response is not None:
            response.close()


def download_file(url, destination_path, progress_bar):
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    temporary_path = f"{destination_path}.part"
    response = requests.get(
        url,
        headers={"Accept-Encoding": "identity"},
        stream=True,
        timeout=(10, 120),
    )
    try:
        response.raise_for_status()
        expected_size = int(response.headers.get("content-length", 0))
        downloaded_size = 0
        with open(temporary_path, "wb") as file:
            for data in response.iter_content(1024 * 1024):
                if data:
                    file.write(data)
                    downloaded_size += len(data)
                    progress_bar.update(len(data))
        if expected_size and downloaded_size != expected_size:
            raise IOError(
                f"Incomplete download for {os.path.basename(destination_path)}"
            )
        os.replace(temporary_path, destination_path)
    finally:
        response.close()
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def download_pretrained_presets(selected_models):
    if not selected_models:
        raise gr.Error(i18n("Select at least one pretrained model."))

    if isinstance(selected_models, str):
        selected_models = [selected_models]

    save_path = os.path.join(now_dir, "models", "pretraineds", "custom")
    tasks = [
        (url, os.path.join(save_path, filename))
        for model in selected_models
        for url, filename in PRETRAINED_MODELS[model]
    ]

    gr.Info(i18n("Downloading pretrains..."))
    with tqdm(
        total=sum(get_file_size(url) for url, _ in tasks),
        unit="iB",
        unit_scale=True,
        desc="Downloading pretrains",
    ) as progress_bar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(download_file, url, path, progress_bar)
                for url, path in tasks
            ]
            for future in futures:
                future.result()

    message = i18n("Pretrains downloaded successfully!")
    gr.Info(message)
    return message


def download_pretrained_model(model, sample_rate, url_g="", url_d=""):
    save_path = os.path.join("models", "pretraineds", "custom")
    os.makedirs(save_path, exist_ok=True)
    tasks = []

    if url_g or url_d:
        tasks = [
            (u, os.path.join(save_path, os.path.basename(u)))
            for u in [url_g, url_d]
            if u
        ]
        if not tasks:
            return gr.Warning(i18n("Please provide at least one URL."))
    else:
        data = fetch_pretrained_data()
        paths = data[model][sample_rate]
        tasks = [
            (
                f"https://huggingface.co/{p}",
                os.path.join(save_path, os.path.basename(p)),
            )
            for p in [paths["D"], paths["G"]]
        ]

    gr.Info(i18n("Downloading pretrained model..."))

    with tqdm(
        total=sum(get_file_size(u) for u, _ in tasks),
        unit="iB",
        unit_scale=True,
        desc="Downloading files",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(download_file, url, dst, pbar) for url, dst in tasks
            ]
            for f in futures:
                f.result()

    gr.Info(i18n("Pretrained model downloaded successfully!"))
    print("Pretrained model downloaded successfully!")


def update_sample_rate_dropdown(model):
    return {
        "choices": get_pretrained_sample_rates(model),
        "value": get_pretrained_sample_rates(model)[0],
        "__type__": "update",
    }


def download_handler(is_custom, model, sample_rate, url_g, url_d):
    if is_custom:
        download_pretrained_model(
            None,
            None,
            url_g.replace("?download=true", ""),
            url_d.replace("?download=true", ""),
        )
    else:
        download_pretrained_model(model, sample_rate, "", "")


def download_tab():
    def _download_with_toast(*args):
        gr.Info(i18n("Downloading model..."))
        result = run_download_script(*args)
        if isinstance(result, str):
            if "error" in result.lower() or "failed" in result.lower():
                gr.Warning(result)
            else:
                gr.Info(result)
        return result

    with gr.Column():
        gr.Markdown(value=i18n("## Download Model"))
        model_link = gr.Textbox(
            label=i18n("Model Link"),
            placeholder=i18n("Introduce the model link"),
            interactive=True,
        )
        model_download_output_info = gr.Textbox(
            label=i18n("Output Information"),
            info=i18n("The output information will be displayed here."),
            value="",
            max_lines=8,
            interactive=False,
        )
        model_download_button = gr.Button(i18n("Download Model"))
        model_download_button.click(
            fn=_download_with_toast,
            inputs=[model_link],
            outputs=[model_download_output_info],
        )
        gr.Markdown(value=i18n("## Download Pretrains"))
        pretrained_models = gr.Dropdown(
            choices=list(PRETRAINED_MODELS),
            label=i18n("Pretrained Models"),
            info=i18n("Select one or more pretrained model sets."),
            multiselect=True,
        )
        pretrained_download_output_info = gr.Textbox(
            label=i18n("Output Information"),
            value="",
            interactive=False,
        )
        pretrained_download_button = gr.Button(i18n("Download Pretrains"))
        pretrained_download_button.click(
            fn=download_pretrained_presets,
            inputs=[pretrained_models],
            outputs=[pretrained_download_output_info],
        )
        gr.Markdown(value=i18n("## Drop files"))
        dropbox = gr.File(
            label=i18n(
                "Drag your .pth file and .index file into this space. Drag one and then the other."
            ),
            type="filepath",
        )

        dropbox.upload(
            fn=save_drop_model,
            inputs=[dropbox],
            outputs=[dropbox],
        )
