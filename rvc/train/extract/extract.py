import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import parselmouth
import torch
import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import rvc.lib.zluda
from rvc.configs.config import Config
from rvc.lib.predictors.f0 import RMVPE, Swift
from rvc.lib.predictors.swift_dependencies import ensure_swift_f0
from rvc.lib.utils import (
    extract_embedding_features,
    get_embedding_metadata,
    load_audio_16k,
    load_embedding,
)
from rvc.train.extract.preparing_files import generate_config, generate_filelist
from rvc.train.validation_data import (
    build_validation_extraction_files,
    write_validation_manifest,
)

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


class FeatureInput:
    def __init__(self, f0_method="rmvpe", device="cpu"):
        self.hop_size = 160  # default
        self.sample_rate = 16000  # default
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        if f0_method == "rmvpe":
            self.model = RMVPE(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "swift":
            self.model = Swift(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method != "pm":
            raise ValueError(f"Unsupported F0 method: {f0_method}")
        self.f0_method = f0_method

    def compute_f0(self, x, p_len=None):
        if self.f0_method == "pm":
            if p_len is None:
                p_len = x.shape[0] // self.hop_size
            f0 = self.get_pm(x, p_len)
        elif self.f0_method == "rmvpe":
            f0 = self.model.get_f0(x, filter_radius=0.03)
        elif self.f0_method == "swift":
            f0 = self.model.get_f0(
                x, p_len=p_len, f0_min=self.f0_min, f0_max=self.f0_max
            )
        return np.asarray(f0).copy()

    def get_pm(self, x, p_len):
        f0 = (
            parselmouth.Sound(x, self.sample_rate)
            .to_pitch_ac(
                time_step=self.hop_size / self.sample_rate,
                voicing_threshold=0.6,
                pitch_floor=self.f0_min,
                pitch_ceiling=self.f0_max,
            )
            .selected_array["frequency"]
        )
        pad_size = (p_len - len(f0) + 1) // 2
        if pad_size > 0 or p_len - len(f0) - pad_size > 0:
            f0 = np.pad(
                f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
            )
        return f0

    def coarse_f0(self, f0):
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (
            self.f0_bin - 2
        ) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        f0_coarse = np.rint(f0_mel).astype(int)
        assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
            f0_coarse.max(),
            f0_coarse.min(),
        )
        return f0_coarse

    def process_file(self, file_info, force=False):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if not force and os.path.exists(opt_path_coarse) and os.path.exists(opt_path_full):
            return

        try:
            np_arr = load_audio_16k(inp_path)
            feature_pit = self.compute_f0(np_arr)
            if feature_pit.ndim != 1 or not feature_pit.size:
                raise ValueError("Pitch extraction returned an empty or invalid contour.")
            if not np.isfinite(feature_pit).all() or np.any(feature_pit < 0):
                raise ValueError("Pitch extraction returned invalid frequencies.")
            coarse_pit = self.coarse_f0(feature_pit)
            np.save(opt_path_full, feature_pit, allow_pickle=False)
            np.save(opt_path_coarse, coarse_pit, allow_pickle=False)
        except Exception as error:
            raise RuntimeError(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            ) from error


def process_files(files, f0_method, device, force=False):
    fe = FeatureInput(f0_method=f0_method, device=device)
    with tqdm.tqdm(total=len(files), leave=True) as pbar:
        for file_info in files:
            fe.process_file(file_info, force=force)
            pbar.update(1)


def run_pitch_extraction(files, devices, f0_method, threads, force=False):
    if not files:
        return
    if not devices:
        raise ValueError("No pitch extraction devices were selected.")
    if f0_method == "swift" and (
        not torch.cuda.is_available()
        or any(torch.device(device).type != "cuda" for device in devices)
    ):
        raise RuntimeError("SwiftF0 training extraction requires a CUDA GPU. Select a GPU or use RMVPE.")
    if f0_method == "swift":
        ensure_swift_f0()
    threads = max(1, int(threads))
    if f0_method == "pm":
        worker_count = min(threads, len(files))
        worker_devices = ["cpu"] * worker_count
        print(
            f"Starting pitch extraction with {worker_count} CPU worker(s) using pm..."
        )
    else:
        worker_count = min(len(devices), len(files))
        worker_devices = devices[:worker_count]
        devices_str = ", ".join(worker_devices)
        print(f"Starting pitch extraction on {devices_str} using {f0_method}...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        tasks = [
            executor.submit(
                process_files,
                files[i::worker_count],
                f0_method,
                worker_devices[i],
                force,
            )
            for i in range(worker_count)
        ]
        for task in concurrent.futures.as_completed(tasks):
            task.result()

    print(f"Pitch extraction completed in {time.time() - start_time:.2f} seconds.")


def process_file_embedding(
    files, embedder_model, version, device_num, device, n_threads
):
    model = load_embedding(embedder_model, version).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)

    def worker(file_info):
        wav_file_path, _, _, out_file_path = file_info
        if os.path.exists(out_file_path):
            return
        feats = torch.from_numpy(load_audio_16k(wav_file_path)).float()
        if getattr(model, "audio_requires_normalization", False):
            feats = torch.nn.functional.layer_norm(feats, feats.shape)
        feats = feats.to(device)
        feats = feats.view(1, -1)
        with torch.no_grad():
            result = extract_embedding_features(model, feats, version)
        feats_out = result.squeeze(0).float().cpu().numpy()
        expected_dim = int(getattr(model, "feature_dim", feats_out.shape[-1]))
        if feats_out.ndim != 2 or feats_out.shape[1] != expected_dim:
            raise RuntimeError(
                f"{wav_file_path} produced shape {feats_out.shape}; expected "
                f"[frames, {expected_dim}]."
            )
        if not np.isfinite(feats_out).all():
            raise RuntimeError(f"{wav_file_path} produced non-finite values.")
        np.save(out_file_path, feats_out, allow_pickle=False)

    with tqdm.tqdm(total=len(files), leave=True, position=device_num) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, f) for f in files]
            for future in concurrent.futures.as_completed(futures):
                future.result()
                pbar.update(1)


def run_embedding_extraction(files, devices, embedder_model, version, threads):
    devices_str = ", ".join(devices)
    print(
        f"Starting embedding extraction with {threads} cores on {devices_str}..."
    )
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        tasks = [
            executor.submit(
                process_file_embedding,
                files[i :: len(devices)],
                embedder_model,
                version,
                i,
                devices[i],
                threads // len(devices),
            )
            for i in range(len(devices))
        ]
        for task in concurrent.futures.as_completed(tasks):
            task.result()

    print(f"Embedding extraction completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    exp_dir = sys.argv[1]
    f0_method = sys.argv[2]
    num_processes = int(sys.argv[3])
    gpus = sys.argv[4]
    sample_rate = sys.argv[5]
    embedder_model = sys.argv[6]
    include_mutes = int(sys.argv[7]) if len(sys.argv) > 7 else 2
    version = sys.argv[8] if len(sys.argv) > 8 else "v2"
    rectified = len(sys.argv) > 9 and sys.argv[9] == "--rectified"
    if rectified and (int(sample_rate) != 44100 or version != "v2" or f0_method not in ("pm", "rmvpe")):
        raise ValueError("Rectified flow extraction requires 44100 Hz, v2 features and Parselmouth or RMVPE F0.")

    wav_path = os.path.join(exp_dir, "sliced_audios")

    if not os.path.exists(wav_path):
        print(
            f"Folder for feature extraction not found at {wav_path}. Did you run the preprocessing step?"
        )
        sys.exit(1)

    os.makedirs(os.path.join(exp_dir, "f0"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "f0_voiced"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "extracted"), exist_ok=True)

    metadata = get_embedding_metadata(embedder_model, version)
    file_path = os.path.join(exp_dir, "model_info.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {}
    dataset_format = str(data.get("dataset_format", "wav")).strip().lower()
    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"
    previous_embedder = data.get("embedder_model")
    previous_version = data.get("version")
    previous_dim = data.get("feature_dim")
    previous_output = data.get("feature_output")
    previous_fingerprint = data.get("feature_fingerprint")
    feature_contract_changed = previous_embedder is not None and (
        previous_embedder != metadata["embedder_model"]
        or (
            previous_version is not None
            and previous_version != metadata["version"]
        )
        or (previous_dim is not None and int(previous_dim) != metadata["feature_dim"])
        or (
            previous_output is not None
            and previous_output != metadata["feature_output"]
        )
        or (
            previous_fingerprint is not None
            and previous_fingerprint != metadata["feature_fingerprint"]
        )
    )
    if not feature_contract_changed:
        for existing_feature_path in glob.glob(
            os.path.join(exp_dir, "extracted", "*.npy")
        ):
            try:
                existing_shape = np.load(
                    existing_feature_path, mmap_mode="r", allow_pickle=False
                ).shape
            except Exception:
                feature_contract_changed = True
                break
            if len(existing_shape) != 2 or existing_shape[1] != metadata["feature_dim"]:
                feature_contract_changed = True
                break
    if feature_contract_changed:
        generator_checkpoints = glob.glob(os.path.join(exp_dir, "G_*.pth"))
        if generator_checkpoints:
            raise RuntimeError(
                "This experiment already has resumable generator checkpoints for a "
                "different embedder. Use a new model name or remove those checkpoints "
                "before changing the embedder."
            )
        for stale_path in glob.glob(os.path.join(exp_dir, "extracted", "*.npy")):
            os.remove(stale_path)
        for stale_path in glob.glob(os.path.join(exp_dir, "*.index")):
            os.remove(stale_path)
        print("Embedder changed; removed stale features and indexes.")
    force_pitch_extraction = data.get("f0_method") != f0_method
    data.update(metadata)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    files = []
    audio_extension = f".{dataset_format}"
    for file in sorted(glob.glob(os.path.join(wav_path, f"*{audio_extension}"))):
        file_name = os.path.basename(file)
        feature_name = (
            file_name.replace("wav", "npy")
            if dataset_format == "wav"
            else f"{file_name}.npy"
        )
        file_info = [
            file,
            os.path.join(exp_dir, "f0", file_name + ".npy"),
            os.path.join(exp_dir, "f0_voiced", file_name + ".npy"),
            os.path.join(exp_dir, "extracted", feature_name),
        ]
        files.append(file_info)

    if not files:
        print(
            f"Sliced audios not found at {wav_path}. Did you run the preprocessing step?"
        )
        sys.exit(1)

    validation_files, validation_entries = build_validation_extraction_files(exp_dir)
    extraction_files = files + validation_files
    if validation_files:
        print(
            f"Caching pitch and features for {len(validation_files)} external "
            "validation source clip(s)."
        )

    devices = ["cpu"] if gpus == "-" else [f"cuda:{idx}" for idx in gpus.split("-")]

    run_pitch_extraction(
        extraction_files, devices, f0_method, num_processes, force=force_pitch_extraction
    )
    data["f0_method"] = f0_method
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    run_embedding_extraction(
        extraction_files, devices, embedder_model, version, num_processes
    )

    write_validation_manifest(exp_dir, validation_entries)

    if rectified:
        generate_filelist(exp_dir, sample_rate, 0)
    else:
        generate_config(sample_rate, exp_dir, version)
        generate_filelist(exp_dir, sample_rate, include_mutes)
