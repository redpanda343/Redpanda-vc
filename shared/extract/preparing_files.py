import json
import glob
import os
from random import shuffle

current_directory = os.getcwd()


def generate_config(sample_rate: int, model_path: str, version: str = "v2"):
    if version not in {"v1", "v2"}:
        raise ValueError(f"Unsupported RVC version: {version}")
    expected_feature_dim = 256 if version == "v1" else 768
    if version == "v1":
        if int(sample_rate) != 40000:
            raise ValueError("RVC v1 preprocessing requires a 40000 Hz model.")
        config_path = os.path.join("rvc", "configs", "v1", "40k.json")
    else:
        config_path = os.path.join("rvc", "configs", f"{sample_rate}.json")
    config_save_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_save_path):
        with open(config_save_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        model_config = config.setdefault("model", {})
        current_feature_dim = int(
            model_config.get("text_enc_hidden_dim", expected_feature_dim)
        )
        if current_feature_dim != expected_feature_dim:
            if glob.glob(os.path.join(model_path, "G_*.pth")):
                raise RuntimeError(
                    "The experiment already has generator checkpoints for a different "
                    "RVC version. Use a new model name before changing versions."
                )
            model_config["text_enc_hidden_dim"] = expected_feature_dim
            with open(config_save_path, "w", encoding="utf-8") as config_file:
                json.dump(config, config_file, indent=4)
        return
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    config.setdefault("model", {})["text_enc_hidden_dim"] = expected_feature_dim
    with open(config_save_path, "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=4)


def generate_filelist(model_path: str, sample_rate: int, include_mutes: int = 2):
    gt_wavs_dir = os.path.join(model_path, "sliced_audios")
    feature_dir = os.path.join(model_path, f"extracted")

    f0_dir, f0nsf_dir = None, None
    f0_dir = os.path.join(model_path, "f0")
    f0nsf_dir = os.path.join(model_path, "f0_voiced")

    try:
        model_info_path = os.path.join(model_path, "model_info.json")
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            embedder_name = model_info["embedder_model"]
            dataset_format = str(model_info.get("dataset_format", "wav")).lower()
    except:
        embedder_name = "contentvec"
        dataset_format = "wav"

    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"
    if embedder_name not in {"contentvec", "spin-v2"}:
        raise RuntimeError(f"Unsupported embedder model: {embedder_name}")

    if embedder_name == "spin-v2":
        mute_base_path = os.path.join(current_directory, "logs", "mute_spin-v2")
    else:
        mute_base_path = os.path.join(current_directory, "logs", "mute")

    options = []
    sids = []
    if dataset_format == "flac":
        audio_files = sorted(
            filename
            for filename in os.listdir(gt_wavs_dir)
            if filename.lower().endswith(".flac")
        )
        if not audio_files:
            raise RuntimeError(f"No FLAC training slices found in {gt_wavs_dir}")

        for audio_file in audio_files:
            name = os.path.splitext(audio_file)[0]
            feature_file = f"{audio_file}.npy"
            f0_file = f"{audio_file}.npy"
            required_paths = (
                os.path.join(feature_dir, feature_file),
                os.path.join(f0_dir, f0_file),
                os.path.join(f0nsf_dir, f0_file),
            )
            missing_paths = [path for path in required_paths if not os.path.isfile(path)]
            if missing_paths:
                raise RuntimeError(
                    "Missing FLAC extraction artifacts: " + ", ".join(missing_paths)
                )

            sid = name.split("_")[0]
            if sid not in sids:
                sids.append(sid)
            rel_audio = os.path.relpath(os.path.join(gt_wavs_dir, audio_file))
            rel_feat = os.path.relpath(required_paths[0])
            rel_f0 = os.path.relpath(required_paths[1])
            rel_f0nsf = os.path.relpath(required_paths[2])
            options.append(
                f"{rel_audio}|{rel_feat}|{rel_f0}|{rel_f0nsf}|{sid}".replace(
                    "\\", "/"
                )
            )
    else:
        gt_wavs_files = {
            name[: -len(".wav")]
            for name in os.listdir(gt_wavs_dir)
            if name.lower().endswith(".wav")
        }
        feature_files = {
            name[: -len(".npy")]
            for name in os.listdir(feature_dir)
            if name.lower().endswith(".npy")
            and not name.lower().endswith(".flac.npy")
        }
        f0_files = {
            name[: -len(".wav.npy")]
            for name in os.listdir(f0_dir)
            if name.lower().endswith(".wav.npy")
        }
        f0nsf_files = {
            name[: -len(".wav.npy")]
            for name in os.listdir(f0nsf_dir)
            if name.lower().endswith(".wav.npy")
        }
        names = gt_wavs_files & feature_files & f0_files & f0nsf_files

        for name in names:
            sid = name.split("_")[0]
            if sid not in sids:
                sids.append(sid)

            # Preserve the existing WAV paths and naming.
            rel_wav = os.path.relpath(f"{os.path.join(gt_wavs_dir, name)}.wav")
            rel_feat = os.path.relpath(f"{os.path.join(feature_dir, name)}.npy")
            rel_f0 = os.path.relpath(f"{os.path.join(f0_dir, name)}.wav.npy")
            rel_f0nsf = os.path.relpath(
                f"{os.path.join(f0nsf_dir, name)}.wav.npy"
            )
            options.append(
                f"{rel_wav}|{rel_feat}|{rel_f0}|{rel_f0nsf}|{sid}".replace(
                    "\\", "/"
                )
            )

    if include_mutes > 0:
        mute_audio_path = os.path.relpath(
            os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav")
        )
        mute_feature_path = os.path.relpath(
            os.path.join(mute_base_path, "extracted", "mute.npy")
        )
        mute_f0_path = os.path.relpath(
            os.path.join(mute_base_path, "f0", "mute.wav.npy")
        )
        mute_f0nsf_path = os.path.relpath(
            os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")
        )

        absolute_mute_feature_path = os.path.abspath(mute_feature_path)
        if not os.path.isfile(absolute_mute_feature_path):
            raise RuntimeError(f"Mute feature not found: {absolute_mute_feature_path}")

        # adding x files per sid
        for sid in sids * include_mutes:
            options.append(
                f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
            )

    file_path = os.path.join(model_path, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    data.update(
        {
            "speakers_id": len(sids),
        }
    )
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    shuffle(options)

    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))
