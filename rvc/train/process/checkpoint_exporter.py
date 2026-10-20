import json
import os
import re

import torch

from rvc.train.process.extract_model import extract_model
from rvc.train.utils import HParams


def _detect_vocoder(state_dict):
    keys = state_dict.keys()
    if any(
        key.startswith(("dec.upsample_blocks.", "dec.downsample_blocks."))
        for key in keys
    ):
        return "RefineGAN"
    if any(key.startswith(("dec.upsamples.", "dec.mrfs.")) for key in keys):
        return "MRF HiFi-GAN"
    return "HiFi-GAN"


def _checkpoint_step(checkpoint_path):
    match = re.fullmatch(r"G_(\d+)\.pth", os.path.basename(checkpoint_path))
    if not match:
        return 0
    step = int(match.group(1))
    return 0 if step == 2333333 else step


def _output_filename(output_name, automatic_name):
    if not output_name or not output_name.strip():
        return automatic_name

    filename = os.path.basename(output_name.strip())
    if filename.lower().endswith(".pth"):
        filename = filename[:-4]
    filename = re.sub(r'[^\w .-]', "_", filename, flags=re.UNICODE).strip(" .")
    if not filename:
        raise ValueError("Enter a valid output file name.")
    return f"{filename}.pth"


def export_generator_checkpoint(checkpoint_path, precision, output_name=None):
    if not checkpoint_path:
        raise ValueError("Select a generator checkpoint to export.")
    if precision not in ("fp16", "fp32"):
        raise ValueError("Precision must be fp16 or fp32.")

    checkpoint_path = os.path.abspath(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Generator checkpoint not found: {checkpoint_path}")
    if not os.path.basename(checkpoint_path).startswith("G_"):
        raise ValueError("Select a generator checkpoint whose name starts with G_.")

    model_dir = os.path.dirname(checkpoint_path)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"Training configuration not found beside the checkpoint: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as config_file:
        hps = HParams(**json.load(config_file))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("The selected file is not a resumable generator checkpoint.")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, dict):
        raise ValueError("The selected file is not a resumable generator checkpoint.")
    if "emb_g.weight" not in state_dict:
        raise ValueError("The selected generator has no speaker embedding weights.")

    hps.model.spk_embed_dim = state_dict["emb_g.weight"].shape[0]
    epoch = int(checkpoint.get("iteration", 0))
    step = _checkpoint_step(checkpoint_path)
    model_name = os.path.basename(model_dir)
    vocoder = _detect_vocoder(state_dict)
    feature_dim = int(getattr(hps.model, "text_enc_hidden_dim", 768))
    if feature_dim == 256:
        version = "v1"
    else:
        version = "v3" if vocoder == "RefineGAN" else "v2"
    export_dtype = torch.float16 if precision == "fp16" else torch.float32
    ema = checkpoint.get("ema")
    use_ema = isinstance(ema, dict) and isinstance(ema.get("weights"), dict)
    if use_ema:
        state_dict = ema["weights"]

    checkpoint_label = f"{epoch}e"
    if step:
        checkpoint_label += f"_{step}s"
    automatic_name = f"{model_name}_{checkpoint_label}_{precision}.pth"
    filename = _output_filename(output_name, automatic_name)
    output_path = os.path.join(model_dir, filename)
    exported_model_name = os.path.splitext(filename)[0]

    extract_model(
        ckpt=state_dict,
        sr=hps.data.sample_rate,
        name=exported_model_name,
        model_path=output_path,
        epoch=epoch,
        step=step,
        hps=hps,
        vocoder=vocoder,
        pitch_guidance=True,
        version=version,
        export_dtype=export_dtype,
    )

    message = (
        f"Exported epoch {epoch} to '{filename}' as {precision.upper()} "
        f"using {vocoder}{' and the generator EMA' if use_ema else ''}."
    )
    return message, output_path
