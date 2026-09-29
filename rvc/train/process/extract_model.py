import datetime
import hashlib
import json
import os
import sys
import tempfile
from collections import OrderedDict

import torch

now_dir = os.getcwd()
sys.path.append(now_dir)


def replace_keys_in_dict(d, old_key_part, new_key_part):
    if isinstance(d, OrderedDict):
        updated_dict = OrderedDict()
    else:
        updated_dict = {}
    for key, value in d.items():
        new_key = key.replace(old_key_part, new_key_part)
        if isinstance(value, dict):
            value = replace_keys_in_dict(value, old_key_part, new_key_part)
        updated_dict[new_key] = value
    return updated_dict


def extract_model(
    ckpt,
    sr,
    name,
    model_path,
    epoch,
    step,
    hps,
    vocoder,
    pitch_guidance=True,
    version=None,
    export_dtype=torch.float16,
):
    temporary_model_path = None
    try:
        model_dir = os.path.dirname(model_path)
        os.makedirs(model_dir, exist_ok=True)

        dataset_length = None
        embedder_model = None
        speakers_id = 1
        feature_metadata = {}
        model_info_version = None
        if os.path.exists(os.path.join(model_dir, "model_info.json")):
            with open(
                os.path.join(model_dir, "model_info.json"), "r", encoding="utf-8"
            ) as f:
                data = json.load(f)
                dataset_length = data.get("total_dataset_duration", None)
                embedder_model = data.get("embedder_model", None)
                speakers_id = data.get("speakers_id", 1)
                model_info_version = data.get("version")
                feature_metadata = {
                    key: data[key]
                    for key in (
                        "feature_dim",
                        "feature_output",
                        "feature_fingerprint",
                    )
                    if key in data
                }

        if version is None:
            if model_info_version is not None:
                version = model_info_version
            else:
                configured_feature_dim = int(
                    getattr(hps.model, "text_enc_hidden_dim", 768)
                )
                version = (
                    "v3"
                    if vocoder == "RefineGAN"
                    else ("v1" if configured_feature_dim == 256 else "v2")
                )

        feature_dim = int(
            feature_metadata.get(
                "feature_dim", getattr(hps.model, "text_enc_hidden_dim", 768)
            )
        )
        feature_output = feature_metadata.get(
            "feature_output",
            "hidden_states[9]+final_proj" if version == "v1" else "last_hidden_state",
        )
        feature_fingerprint = feature_metadata.get("feature_fingerprint")

        with open(
            os.path.join(now_dir, "assets", "config.json"), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)
            model_author = data.get("model_author", None)

        if export_dtype not in (torch.float16, torch.float32):
            raise ValueError("Inference checkpoints must use float16 or float32.")

        opt = OrderedDict(
            weight={
                key: value.detach().to(
                    device="cpu",
                    dtype=export_dtype if value.is_floating_point() else value.dtype,
                )
                for key, value in ckpt.items()
                if "enc_q" not in key
            }
        )
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sample_rate,
        ]

        opt["epoch"] = epoch
        opt["step"] = step
        opt["sr"] = sr
        opt["f0"] = pitch_guidance
        opt["version"] = version
        opt["creation_date"] = datetime.datetime.now().isoformat()

        hash_input = (
            f"{name}-{epoch}-{step}-{sr}-{version}-{opt['config']}-"
            f"{feature_dim}-{feature_output}-{feature_fingerprint}"
        )
        opt["model_hash"] = hashlib.sha256(hash_input.encode()).hexdigest()
        opt["dataset_length"] = dataset_length
        opt["model_name"] = name
        opt["author"] = model_author
        opt["embedder_model"] = embedder_model
        opt["feature_dim"] = feature_dim
        opt["feature_output"] = feature_output
        opt["feature_fingerprint"] = feature_fingerprint
        opt["speakers_id"] = speakers_id
        opt["vocoder"] = vocoder

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=model_dir,
            prefix=f".{os.path.basename(model_path)}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_model:
            temporary_model_path = temporary_model.name

        torch.save(
            replace_keys_in_dict(
                replace_keys_in_dict(
                    opt, ".parametrizations.weight.original1", ".weight_v"
                ),
                ".parametrizations.weight.original0",
                ".weight_g",
            ),
            temporary_model_path,
        )
        os.replace(temporary_model_path, model_path)
        temporary_model_path = None

        print(f"Saved model '{model_path}' (epoch {epoch} and step {step})")

    except Exception as error:
        if temporary_model_path and os.path.exists(temporary_model_path):
            try:
                os.remove(temporary_model_path)
            except OSError:
                pass
        raise RuntimeError(f"Failed to export inference model '{model_path}'.") from error
