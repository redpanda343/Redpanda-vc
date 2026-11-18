import concurrent.futures
import glob
import json
import multiprocessing as mp
import os
import queue
import sys
import tempfile

import numpy as np
import parselmouth
import torch
import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

# Zluda hijack
import shared.zluda
from rvc.configs.config import Config
from beatrice.resources import ensure_pitch_estimator
from shared.predictors.f0 import RMVPE, RVCRMVPE, BeatricePitch, Swift
from shared.predictors.swift_dependencies import ensure_swift_f0
from shared.utils import (
    compute_coarse_f0,
    extract_embedding_features,
    get_embedding_metadata,
    load_audio_16k,
    load_embedding,
    readable_npy,
    save_npy,
)
from shared.extract.preparing_files import generate_config, generate_filelist

# Load config
config = Config()
mp.set_start_method("spawn", force=True)


def _configure_cpu_worker_threads(device, worker_count):
    if str(device) == "cpu":
        cpu_total = os.cpu_count() or 1
        torch.set_num_threads(max(1, cpu_total // max(1, int(worker_count))))


class FeatureInput:
    def __init__(self, f0_method="rmvpe", device="cpu", rectified=False):
        self.hop_size = 160  # default
        self.sample_rate = 16000  # default
        self.f0_bin = 256
        self.f0_max = 1100.0
        self.f0_min = 50.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.device = device
        if f0_method == "rmvpe":
            rmvpe_class = RMVPE if rectified else RVCRMVPE
            self.model = rmvpe_class(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "swift":
            self.model = Swift(
                device=self.device, sample_rate=self.sample_rate, hop_size=self.hop_size
            )
        elif f0_method == "beatrice":
            self.model = BeatricePitch(
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
        elif self.f0_method == "beatrice":
            f0 = self.model.get_f0(x, p_len)
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
        return compute_coarse_f0(
            f0,
            f0_bin=self.f0_bin,
            f0_min=self.f0_min,
            f0_max=self.f0_max,
        )

    def process_file(self, file_info, force=False):
        inp_path, opt_path_coarse, opt_path_full, _ = file_info
        if not force and readable_npy(opt_path_coarse) and readable_npy(opt_path_full):
            return

        try:
            np_arr = load_audio_16k(inp_path)
            with torch.inference_mode():
                feature_pit = self.compute_f0(np_arr)
            if feature_pit.ndim != 1 or not feature_pit.size:
                raise ValueError("Pitch extraction returned an empty or invalid contour.")
            if not np.isfinite(feature_pit).all() or np.any(feature_pit < 0):
                raise ValueError("Pitch extraction returned invalid frequencies.")
            coarse_pit = self.coarse_f0(feature_pit)
            save_npy(opt_path_full, feature_pit)
            save_npy(opt_path_coarse, coarse_pit)
        except Exception as error:
            raise RuntimeError(
                f"An error occurred extracting file {inp_path} on {self.device}: {error}"
            ) from error


def process_files(files, f0_method, device, force, updates, rectified, worker_count=1):
    _configure_cpu_worker_threads(device, worker_count)
    fe = FeatureInput(f0_method=f0_method, device=device, rectified=rectified)
    for file_info in files:
        fe.process_file(file_info, force=force)
        updates.put(1)


def follow_progress(tasks, updates, progress):
    pending = set(tasks)
    while pending:
        done, pending = concurrent.futures.wait(pending, timeout=0.2)
        for task in done:
            task.result()
        while True:
            try:
                progress.update(updates.get_nowait())
            except queue.Empty:
                break


def run_pitch_extraction(files, devices, f0_method, threads, progress, updates, force=False, rectified=False):
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
    if f0_method == "beatrice":
        ensure_pitch_estimator()
    threads = max(1, int(threads))
    if f0_method == "pm":
        worker_count = min(threads, len(files))
        worker_devices = ["cpu"] * worker_count
    elif len(devices) == 1 and str(devices[0]) == "cpu":
        # Scale neural F0 (RMVPE/Beatrice) across CPU worker processes instead of 1 worker.
        cpu_total = os.cpu_count() or threads
        worker_count = min(threads, cpu_total, len(files), 8)
        worker_devices = ["cpu"] * max(1, worker_count)
    else:
        worker_count = min(len(devices), len(files))
        worker_devices = devices[:worker_count]
    progress.set_description(f"Extracting F0 ({f0_method})")

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        tasks = [
            executor.submit(
                process_files,
                files[i::worker_count],
                f0_method,
                worker_devices[i],
                force,
                updates,
                rectified,
                worker_count,
            )
            for i in range(worker_count)
        ]
        follow_progress(tasks, updates, progress)


def process_file_embedding(
    files, embedder_model, version, device, n_threads, updates, worker_count=1
):
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()
    transformers_logging.set_verbosity_error()
    _configure_cpu_worker_threads(device, worker_count)
    model = load_embedding(embedder_model, version).to(device).float()
    model.eval()
    n_threads = max(1, n_threads)

    decode_workers = min(4, n_threads) if str(device) != "cpu" else 1
    write_workers = min(4, max(2, n_threads))

    def _infer_and_save(file_info, wav_16k, writer_pool, write_futures):
        wav_file_path, _, _, out_file_path = file_info
        feats = torch.from_numpy(wav_16k).float()
        if getattr(model, "audio_requires_normalization", False):
            feats = torch.nn.functional.layer_norm(feats, feats.shape)
        feats = feats.to(device, non_blocking=str(device).startswith("cuda")).view(1, -1)
        with torch.inference_mode():
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
        fut = writer_pool.submit(save_npy, out_file_path, feats_out)
        write_futures.append(fut)
        while len(write_futures) > 16:
            write_futures.pop(0).result()
        updates.put(1)

    with concurrent.futures.ThreadPoolExecutor(max_workers=decode_workers) as dec_pool, concurrent.futures.ThreadPoolExecutor(
        max_workers=write_workers
    ) as writer_pool:
        write_futures = []
        pending_files = [fi for fi in files if not readable_npy(fi[3])]
        skipped = len(files) - len(pending_files)
        for _ in range(skipped):
            updates.put(1)

        if decode_workers > 1 and pending_files:
            decoded = dec_pool.map(lambda fi: (fi, load_audio_16k(fi[0])), pending_files)
            for fi, wav_16k in decoded:
                _infer_and_save(fi, wav_16k, writer_pool, write_futures)
        else:
            for fi in pending_files:
                _infer_and_save(fi, load_audio_16k(fi[0]), writer_pool, write_futures)

        for fut in write_futures:
            fut.result()


def run_embedding_extraction(files, devices, embedder_model, version, threads, progress, updates):
    if not files:
        return
    progress.set_description(f"Extracting features ({embedder_model})")
    threads = max(1, int(threads))
    if len(devices) == 1 and str(devices[0]) == "cpu":
        # Shard CPU embedding extraction across processes instead of threading one shared CPU model.
        cpu_total = os.cpu_count() or threads
        worker_count = min(threads, cpu_total, len(files), 8)
        worker_devices = ["cpu"] * max(1, worker_count)
    else:
        worker_count = min(len(devices), len(files))
        worker_devices = devices[:worker_count]

    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        tasks = [
            executor.submit(
                process_file_embedding,
                files[i::worker_count],
                embedder_model,
                version,
                worker_devices[i],
                max(1, threads // worker_count),
                updates,
                worker_count,
            )
            for i in range(worker_count)
        ]
        follow_progress(tasks, updates, progress)


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
    if rectified and (version != "v2" or f0_method not in ("pm", "rmvpe")):
        raise ValueError("Rectified flow extraction requires v2 features and Parselmouth or RMVPE F0.")

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
    for stale_path in glob.glob(os.path.join(exp_dir, ".*.partial")) + glob.glob(
        os.path.join(exp_dir, "*", ".*.partial")
    ):
        os.remove(stale_path)
    if not feature_contract_changed:
        unreadable = 0
        for existing_feature_path in glob.glob(
            os.path.join(exp_dir, "extracted", "*.npy")
        ):
            try:
                existing_shape = np.load(
                    existing_feature_path, mmap_mode="r", allow_pickle=False
                ).shape
            except Exception:
                os.remove(existing_feature_path)
                unreadable += 1
                continue
            if len(existing_shape) != 2 or existing_shape[1] != metadata["feature_dim"]:
                feature_contract_changed = True
                break
        if unreadable:
            print(f"Removed {unreadable} unreadable feature file(s); they are extracted again.")
    if feature_contract_changed:
        generator_checkpoints = glob.glob(os.path.join(exp_dir, "G_*.pth")) + glob.glob(
            os.path.join(exp_dir, "G_*.npz")
        )
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
    f0_model = "rmvpe_rvc" if f0_method == "rmvpe" and not rectified else f0_method
    force_pitch_extraction = (
        data.get("f0_method") != f0_method
        or data.get("f0_model", data.get("f0_method")) != f0_model
    )
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

    devices = (
        ["cpu"]
        if (gpus == "-" or not torch.cuda.is_available())
        else [f"cuda:{idx}" for idx in gpus.split("-")]
    )

    extract_backend = os.environ.get("REDPANDA_EXTRACT_BACKEND", "auto").strip().lower()
    use_jax = False
    if extract_backend != "torch" and (devices == ["cpu"] or extract_backend == "jax"):
        from rvc.train.backend import has_jax_cuda_plugin, has_jax_tpu_runtime

        allow_jax_gpu = gpus != "-" and (not torch.cuda.is_available() or extract_backend == "jax")
        if extract_backend == "jax" or has_jax_tpu_runtime() or (allow_jax_gpu and has_jax_cuda_plugin()):
            from rvc.jax.extract import jax_extraction_available, run_jax_extraction

            use_jax = jax_extraction_available(allow_gpu=allow_jax_gpu)

    if use_jax and f0_method == "rmvpe":
        with tqdm.tqdm(
            total=2 * len(files),
            desc=f"Extracting F0 ({f0_method}) + Features ({embedder_model}) [JAX]",
            unit="step",
            dynamic_ncols=True,
        ) as progress:
            run_jax_extraction(
                files,
                extract_f0=True,
                f0_force=force_pitch_extraction,
                rectified=rectified,
                extract_features=True,
                embedder_model=embedder_model,
                version=version,
                accelerator="auto",
                progress=progress,
            )
        data["f0_method"] = f0_method
        data["f0_model"] = f0_model
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
    elif use_jax and f0_method != "swift":
        with mp.Manager() as manager, tqdm.tqdm(
            total=2 * len(files), desc=f"Extracting F0 ({f0_method})", unit="file",
            dynamic_ncols=True,
        ) as progress:
            updates = manager.Queue()
            run_pitch_extraction(
                files, devices, f0_method, num_processes, progress, updates,
                force=force_pitch_extraction,
                rectified=rectified,
            )
        data["f0_method"] = f0_method
        data["f0_model"] = f0_model
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        with tqdm.tqdm(
            total=2 * len(files),
            initial=len(files),
            desc=f"Extracting features ({embedder_model}) [JAX]",
            unit="file",
            dynamic_ncols=True,
        ) as progress:
            run_jax_extraction(
                files,
                extract_f0=False,
                extract_features=True,
                embedder_model=embedder_model,
                version=version,
                accelerator="auto",
                progress=progress,
            )
    else:
        with mp.Manager() as manager, tqdm.tqdm(
            total=2 * len(files), desc=f"Extracting F0 ({f0_method})", unit="file",
            dynamic_ncols=True,
        ) as progress:
            updates = manager.Queue()
            run_pitch_extraction(
                files, devices, f0_method, num_processes, progress, updates,
                force=force_pitch_extraction,
                rectified=rectified,
            )
            data["f0_method"] = f0_method
            data["f0_model"] = f0_model
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)

            run_embedding_extraction(
                files, devices, embedder_model, version, num_processes, progress, updates
            )

    if rectified:
        generate_filelist(exp_dir, sample_rate, 0)
    else:
        generate_config(sample_rate, exp_dir, version)
        generate_filelist(exp_dir, sample_rate, include_mutes)
