import os
import sys

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
import datetime
import glob
import hashlib
import json
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from random import randint, shuffle
from time import time as ttime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

from rvc.train.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.train.mel_processing import (
    MultiScaleMelSpectrogramLoss,
    mel_spectrogram_torch,
    spec_to_mel_torch,
)
from rvc.train.utils import (
    HParams,
    latest_checkpoint_path,
    load_checkpoint,
    load_wav_to_torch,
    plot_spectrogram_to_numpy,
    save_checkpoint,
    summarize,
)

# Zluda hijack
import rvc.lib.zluda
from rvc.lib.algorithm import commons
from rvc.train.mos_validation import (
    UTMOSv2Validator,
    deterministic_validation_scope,
)
from rvc.train.process.extract_model import extract_model
from rvc.train.timbre_validation import ECAPATimbreValidator
from rvc.train.validation_data import (
    infer_validation_audio,
    prepare_validation_reference,
    should_run_external_validation,
)

# Parse command line arguments
model_name = sys.argv[1]
save_every_epoch = int(sys.argv[2])
total_epoch = int(sys.argv[3])
pretrainG = sys.argv[4]
pretrainD = sys.argv[5]
gpus = sys.argv[6]
batch_size = int(sys.argv[7])
sample_rate = int(sys.argv[8])


def _speaker_id(item):
    try:
        return int(item[4])
    except (TypeError, ValueError):
        return 0


def _validation_sort_key(seed, value):
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()


def build_timbre_enrollment(
    dataset,
    collate_fn,
    speaker_ids,
    seed,
    max_per_speaker=3,
):
    target_speakers = sorted({int(speaker_id) for speaker_id in speaker_ids})
    groups = defaultdict(list)
    for index, item in enumerate(dataset.audiopaths_and_text):
        speaker_id = _speaker_id(item)
        if speaker_id not in target_speakers:
            continue
        audio_path = item[0]
        if "mute" in os.path.basename(audio_path).lower():
            continue
        groups[speaker_id].append((audio_path, index))

    selected = []
    for speaker_id in target_speakers:
        candidates = sorted(
            groups[speaker_id],
            key=lambda item: _validation_sort_key(
                seed, f"enrollment:{speaker_id}:{item[0]}"
            ),
        )
        speaker_samples = []
        for _, index in candidates:
            try:
                sample = dataset[index]
            except Exception:
                continue
            if sample[1].abs().mean().item() <= 1e-4:
                continue
            speaker_samples.append(sample)
            if len(speaker_samples) == max_per_speaker:
                break
        if not speaker_samples:
            raise RuntimeError(f"No enrollment audio is available for speaker {speaker_id}")
        selected.extend(speaker_samples)

    info = collate_fn(selected)
    return info[6], info[7], info[8]


def _strtobool(val):
    return val.lower() in ("yes", "true", "t", "y", "1")


save_only_latest = _strtobool(sys.argv[9])
save_every_weights = _strtobool(sys.argv[10])
cache_data_in_gpu = _strtobool(sys.argv[11])
cleanup = _strtobool(sys.argv[12])
vocoder = sys.argv[13]
checkpointing = _strtobool(sys.argv[14])
save_every_steps = max(0, int(sys.argv[15])) if len(sys.argv) > 15 else 0
# experimental settings
randomized = True
d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1
multiscale_mel_loss = False
bf16_adamw = False
disc_version = "v2"

if vocoder == "RefineGAN":
    disc_version = "v3"
    multiscale_mel_loss = True

current_dir = os.getcwd()

try:
    with open(
        os.path.join(current_dir, "assets", "config.json"),
        "r",
        encoding="utf-8",
    ) as f:
        config = json.load(f)
        precision = config["precision"]
        if (
            precision == "bf16"
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            train_dtype = torch.bfloat16
        elif precision == "fp16" and torch.cuda.is_available():
            train_dtype = torch.float16
        else:
            train_dtype = torch.float32
except (FileNotFoundError, json.JSONDecodeError, KeyError):
    train_dtype = torch.float32

inference_export_dtype = (
    torch.float32 if train_dtype == torch.float32 else torch.float16
)

experiment_dir = os.path.join(current_dir, "logs", model_name)
config_save_path = os.path.join(experiment_dir, "config.json")
dataset_path = os.path.join(experiment_dir, "sliced_audios")
model_info_path = os.path.join(experiment_dir, "model_info.json")

try:
    with open(config_save_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = HParams(**config)
except FileNotFoundError:
    print(
        f"Config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
    )
    sys.exit(1)

config.data.training_files = os.path.join(experiment_dir, "filelist.txt")

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = os.name == "nt"
try:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
except Exception as e:
    print(f'Torch tf32: {e}')

global_step = 0
last_loss_gen_all = 0
training_file_path = os.path.join(experiment_dir, "training_data.json")

import logging

logging.getLogger("torch").setLevel(logging.ERROR)


class AsyncInferenceExporter:
    def __init__(self, device):
        self.device = device
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="inference-export"
        )
        self.export_future = None
        self.staging_future = None
        self.copy_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )

    @staticmethod
    def _export_staged(snapshot, copy_event, staging_future, export_args):
        if copy_event is not None:
            try:
                copy_event.synchronize()
            except Exception as error:
                staging_future.set_exception(error)
                raise
            staging_future.set_result(None)
        return extract_model(ckpt=snapshot, **export_args)

    def _stage_cuda(self, state_dict, export_dtype):
        snapshot = {}
        current_stream = torch.cuda.current_stream(self.device)
        try:
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_stream(current_stream)
                for key, value in state_dict.items():
                    if "enc_q" in key:
                        continue
                    value = value.detach()
                    dtype = export_dtype if value.is_floating_point() else value.dtype
                    staged = torch.empty_like(
                        value, device="cpu", dtype=dtype, pin_memory=True
                    )
                    staged.copy_(value, non_blocking=True)
                    snapshot[key] = staged
                copy_event = torch.cuda.Event()
                copy_event.record(self.copy_stream)
        except Exception:
            self.copy_stream.synchronize()
            raise
        return snapshot, copy_event

    @staticmethod
    def _stage_synchronous(state_dict, export_dtype):
        return {
            key: value.detach().to(
                device="cpu",
                dtype=export_dtype if value.is_floating_point() else value.dtype,
                copy=True,
            )
            for key, value in state_dict.items()
            if "enc_q" not in key
        }

    def submit(self, model, export_dtype, **export_args):
        self.wait_for_completion()
        state_dict = model.state_dict()
        if self.device.type == "cuda":
            snapshot, copy_event = self._stage_cuda(state_dict, export_dtype)
            staging_future = Future()
        else:
            snapshot = self._stage_synchronous(state_dict, export_dtype)
            copy_event = None
            staging_future = Future()
            staging_future.set_result(None)
        self.staging_future = staging_future
        try:
            self.export_future = self.executor.submit(
                self._export_staged,
                snapshot,
                copy_event,
                staging_future,
                export_args,
            )
        except Exception:
            if copy_event is not None:
                copy_event.synchronize()
            self.staging_future = None
            raise

    def wait_for_staging(self):
        if self.staging_future is not None:
            self.staging_future.result()
        self.raise_if_failed()

    def raise_if_failed(self):
        if self.export_future is not None and self.export_future.done():
            self.wait_for_completion()

    def wait_for_completion(self):
        export_future = self.export_future
        if export_future is None:
            return
        try:
            export_future.result()
        finally:
            self.export_future = None
            self.staging_future = None

    def close(self):
        if self.executor is None:
            return
        try:
            self.wait_for_completion()
        finally:
            self.executor.shutdown(wait=True, cancel_futures=False)
            self.executor = None


class EpochRecorder:
    """
    Records the time elapsed per epoch.
    """

    def __init__(self):
        self.last_time = ttime()

    def record(self):
        """
        Records the elapsed time and returns a formatted string.
        """
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        elapsed_time = round(elapsed_time, 1)
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        return f"time={current_time} | training_speed={elapsed_time_str}"


def main():
    """
    Main function to start the training process.
    """
    global training_file_path, last_loss_gen_all, gpus

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    # Check sample rate
    dataset_format = "wav"
    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            dataset_format = str(json.load(f).get("dataset_format", "wav")).lower()
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"
    wavs = glob.glob(
        os.path.join(
            os.path.join(experiment_dir, "sliced_audios"),
            f"*.{dataset_format}",
        )
    )
    if wavs:
        _, sr = load_wav_to_torch(wavs[0])
        if sr != config.data.sample_rate:
            print(
                f"Error: Pretrained model sample rate ({config.data.sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz)."
            )
            os._exit(1)
    else:
        print(f"No {dataset_format} file found.")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpus = [int(item) for item in gpus.split("-")]
        n_gpus = len(gpus)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        gpus = [0]
        n_gpus = 1
    else:
        device = torch.device("cpu")
        gpus = [0]
        n_gpus = 1
        print("Training with CPU, this will take a long time.")

    def start():
        """
        Starts the training process with multi-GPU support or CPU.
        """
        children = []
        for rank, device_id in enumerate(gpus):
            subproc = mp.Process(
                target=run,
                args=(
                    rank,
                    n_gpus,
                    experiment_dir,
                    pretrainG,
                    pretrainD,
                    total_epoch,
                    save_every_weights,
                    config,
                    device,
                    device_id,
                ),
            )
            children.append(subproc)
            subproc.start()

        for i in range(n_gpus):
            children[i].join()

    if cleanup:
        print("Removing files from the prior training attempt...")

        # Clean up unnecessary files
        for root, dirs, files in os.walk(
            os.path.join(now_dir, "logs", model_name), topdown=False
        ):
            for name in files:
                file_path = os.path.join(root, name)
                file_name, file_extension = os.path.splitext(name)
                if (
                    file_extension == ".0"
                    or file_extension == ".pth"
                    or name.endswith(".spec.pt")
                    or (file_name.startswith("added") and file_extension == ".index")
                ):
                    os.remove(file_path)
            for name in dirs:
                if name == "eval":
                    folder_path = os.path.join(root, name)
                    for item in os.listdir(folder_path):
                        item_path = os.path.join(folder_path, item)
                        if os.path.isfile(item_path):
                            os.remove(item_path)
                    os.rmdir(folder_path)

        print("Cleanup done!")

    start()


def run(
    rank,
    n_gpus,
    experiment_dir,
    pretrainG,
    pretrainD,
    custom_total_epoch,
    custom_save_every_weights,
    config,
    device,
    device_id,
):
    """
    Runs the training loop on a specific GPU or CPU.

    Args:
        rank (int): The rank of the current process within the distributed training setup.
        n_gpus (int): The total number of GPUs available for training.
        experiment_dir (str): The directory where experiment logs and checkpoints will be saved.
        pretrainG (str): Path to the pre-trained generator model.
        pretrainD (str): Path to the pre-trained discriminator model.
        custom_total_epoch (int): The total number of epochs for training.
        custom_save_every_weights (int): The interval (in epochs) at which to save model weights.
        config (object): Configuration object containing training parameters.
        device (torch.device): The device to use for training (CPU or GPU).
    """
    global global_step
    writer_eval = None

    dist.init_process_group(
        backend="gloo" if sys.platform == "win32" or device.type != "cuda" else "nccl",
        init_method="env://",
        world_size=n_gpus if device.type == "cuda" else 1,
        rank=rank if device.type == "cuda" else 0,
    )

    torch.manual_seed(config.train.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)

    # Create datasets and dataloaders
    from data_utils import (
        DistributedBucketSampler,
        TextAudioCollateMultiNSFsid,
        TextAudioLoaderMultiNSFsid,
    )

    train_dataset = TextAudioLoaderMultiNSFsid(config.data)
    collate_fn = TextAudioCollateMultiNSFsid()
    timbre_reference = None
    audio_reference = None
    train_sampler = DistributedBucketSampler(
        train_dataset,
        batch_size,
        [
            50, 100, 200, 300, 400, 500, 600, 700, 800, 900,
            1000, 1100, 1200, 1300, 1400, 1500,
        ],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    # Validations
    if len(train_loader) < 3:
        print(
            "Not enough data present in the training set. Perhaps you forgot to slice the audio files in preprocess?"
        )
        os._exit(2333333)

    # defaults
    spk_dim = config.model.spk_embed_dim  # 109 default speakers

    try:
        with open(model_info_path, "r", encoding="utf-8") as f:
            model_info = json.load(f)
            spk_dim = model_info["speakers_id"]
    except Exception as e:
        print(f"Could not load model info file: {e}. Using defaults.")

    dataset_spk_dim = spk_dim
    last_g = latest_checkpoint_path(experiment_dir, "G_*.pth")
    fresh_speaker_embeddings = last_g is None

    try:
        chk_path = last_g
        if chk_path:
            ckpt = torch.load(chk_path, map_location="cpu", weights_only=True)
            spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
            del ckpt
    except Exception as e:
        print(f"Failed to load checkpoint: {e}. Using default number of speakers.")

    # update config before the model init
    print(f"Initializing the generator with {spk_dim} speakers.")
    config.model.spk_embed_dim = spk_dim

    if rank == 0:
        try:
            timbre_reference = prepare_validation_reference(
                experiment_dir,
                device,
                config.train.seed,
                dataset_spk_dim,
                config.data.sample_rate,
                config.data.hop_length,
                config.model.text_enc_hidden_dim,
            )
            if timbre_reference is not None:
                audio_reference = tuple(value[:1] for value in timbre_reference[0])
                print("TensorBoard audio validation uses one external validation clip.")
        except Exception as error:
            print(f"External validation disabled: {error}")
            timbre_reference = None

    # Initialize models and optimizers
    from rvc.lib.algorithm.discriminators import MultiPeriodDiscriminator
    from rvc.lib.algorithm.synthesizers import Synthesizer

    net_g = Synthesizer(
        config.data.filter_length // 2 + 1,
        config.train.segment_size // config.data.hop_length,
        **config.model,
        use_f0=True,
        sr=config.data.sample_rate,
        vocoder=vocoder,
        checkpointing=checkpointing,
        randomized=randomized,
    )

    net_d = MultiPeriodDiscriminator(
        config.model.use_spectral_norm,
        checkpointing=checkpointing,
        version=disc_version,
    )

    if torch.cuda.is_available():
        net_g = net_g.cuda(device_id)
        net_d = net_d.cuda(device_id)
    else:
        net_g = net_g.to(device)
        net_d = net_d.to(device)

    if bf16_adamw == True and train_dtype == torch.bfloat16:
        print("Using BFload16 AdamW optimizer")
        from rvc.train.anyprecision_optimizer import AnyPrecisionAdamW

        optimizer = AnyPrecisionAdamW
    else:
        print("Using AdamW optimizer")
        optimizer = torch.optim.AdamW

    optim_g = optimizer(
        net_g.parameters(),
        config.train.learning_rate * g_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    optim_d = optimizer(
        net_d.parameters(),
        config.train.learning_rate * d_lr_coeff,
        betas=config.train.betas,
        eps=config.train.eps,
    )
    if multiscale_mel_loss:
        fn_mel_loss = MultiScaleMelSpectrogramLoss(sample_rate=config.data.sample_rate)
        print("Using Multi-Scale Mel loss function")
    else:
        fn_mel_loss = torch.nn.L1Loss()
        print("Using Single-Scale Mel loss function")

    # Wrap models with DDP for multi-gpu processing
    if n_gpus > 1 and device.type == "cuda":
        net_g = DDP(net_g, device_ids=[device_id])
        net_d = DDP(net_d, device_ids=[device_id])

    if rank == 0 and train_dtype == torch.bfloat16:
        print("Using BFloat16 for training.")
    elif rank == 0 and train_dtype == torch.float16:
        print("Using Float16 for training.")

    # Load checkpoint if available
    scaler_dict = {}
    try:
        print("Starting training...")
        _, _, _, epoch_str, scaler_dict = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "D_*.pth"), net_d, optim_d
        )
        _, _, _, epoch_str, _ = load_checkpoint(
            latest_checkpoint_path(experiment_dir, "G_*.pth"), net_g, optim_g
        )
        epoch_str += 1
        global_step = (epoch_str - 1) * len(train_loader)

    except Exception as e:
        epoch_str = 1
        global_step = 0

        if pretrainG not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (G) '{pretrainG}'")
            try:
                ckpt = torch.load(pretrainG, map_location="cpu", weights_only=True)[
                    "model"
                ]
                target_net_g = net_g.module if hasattr(net_g, "module") else net_g
                if fresh_speaker_embeddings:
                    if "emb_g.weight" not in ckpt:
                        raise KeyError(
                            "The pretrained generator has no speaker embedding."
                        )
                    ckpt["emb_g.weight"] = (
                        target_net_g.emb_g.weight.detach().cpu().clone()
                    )
                target_net_g.load_state_dict(ckpt)
                del ckpt
            except Exception as e:
                print(
                    "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
                )
                print(e)
                sys.exit(1)

        if pretrainD not in ("", "None"):
            if rank == 0:
                print(f"Loaded pretrained (D) '{pretrainD}'")
            try:
                ckpt = torch.load(pretrainD, map_location="cpu", weights_only=True)[
                    "model"
                ]
                if hasattr(net_d, "module"):
                    net_d.module.load_state_dict(ckpt)
                else:
                    net_d.load_state_dict(ckpt)
                del ckpt
            except Exception as e:
                print(
                    "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
                )
                print(e)
                sys.exit(1)

    if rank == 0:
        purge_step = global_step + 1 if global_step else 0
        writer_eval = SummaryWriter(
            log_dir=os.path.join(experiment_dir, "eval"),
            purge_step=purge_step,
            max_queue=1000,
        )

    # Initialize schedulers
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=config.train.lr_decay, last_epoch=epoch_str - 2
    )

    use_scaler = device.type == "cuda" and train_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)
    if len(scaler_dict) > 0:
        scaler.load_state_dict(scaler_dict)

    cache = []

    timbre_validator = None
    if rank == 0 and timbre_reference is not None:
        try:
            timbre_model_path = os.path.join(
                "rvc", "models", "pretraineds", "ecapa_tdnn", "pretrain.model"
            )
            timbre_validator = ECAPATimbreValidator(timbre_model_path)
            enrollment_wave, enrollment_lengths, enrollment_speakers = (
                build_timbre_enrollment(
                    train_dataset,
                    collate_fn,
                    timbre_reference[3],
                    config.train.seed,
                )
            )
            timbre_validator.set_references(
                enrollment_wave,
                enrollment_lengths,
                enrollment_speakers,
                config.data.sample_rate,
            )
            print(
                f"ECAPA timbre validation enabled with {len(timbre_reference[3])} probes and {len(enrollment_speakers)} enrollment clips."
            )
        except Exception as error:
            print(f"ECAPA timbre validation disabled: {error}")
            timbre_validator = None

    mos_validator = None
    if rank == 0 and timbre_reference is not None:
        try:
            mos_model_paths = [
                os.path.join(
                    "rvc",
                    "models",
                    "pretraineds",
                    "utmosv2",
                    f"fold{fold}_s42_best_model.pth",
                )
                for fold in range(5)
            ]
            mos_device = (
                torch.device("cuda", device_id) if device.type == "cuda" else device
            )
            mos_validator = UTMOSv2Validator(
                mos_model_paths, config.train.seed, mos_device
            )
            print(
                "UTMOSv2 paper-style five-fold, five-frame batched validation enabled "
                f"with deterministic FP32 {mos_device.type.upper()} inference."
            )
        except Exception as error:
            print(f"UTMOSv2 validation disabled: {error}")
            mos_validator = None

    inference_exporter = None
    if rank == 0 and save_every_steps > 0:
        export_device = (
            torch.device("cuda", device_id) if device.type == "cuda" else device
        )
        inference_exporter = AsyncInferenceExporter(export_device)

    scalar_log_state = {
        "loss_sums": torch.zeros(6, device=device, dtype=torch.float32),
        "batch_count": 0,
    }
    try:
        for epoch in range(epoch_str, total_epoch + 1):
            train_and_evaluate(
                rank,
                epoch,
                config,
                [net_g, net_d],
                [optim_g, optim_d],
                [train_loader, None],
                [writer_eval],
                scalar_log_state,
                cache,
                custom_save_every_weights,
                custom_total_epoch,
                device,
                device_id,
                audio_reference,
                timbre_validator,
                mos_validator,
                timbre_reference,
                fn_mel_loss,
                scaler,
                inference_exporter,
            )

            scheduler_g.step()
            scheduler_d.step()
    finally:
        if inference_exporter is not None:
            inference_exporter.close()
        if writer_eval is not None:
            writer_eval.close()


def train_and_evaluate(
    rank,
    epoch,
    hps,
    nets,
    optims,
    loaders,
    writers,
    scalar_log_state,
    cache,
    custom_save_every_weights,
    custom_total_epoch,
    device,
    device_id,
    audio_reference,
    timbre_validator,
    mos_validator,
    timbre_reference,
    fn_mel_loss,
    scaler,
    inference_exporter,
):
    """
    Trains and evaluates the model for one epoch.

    Args:
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        hps (Namespace): Hyperparameters.
        nets (list): List of models [net_g, net_d].
        optims (list): List of optimizers [optim_g, optim_d].
        loaders (list): List of dataloaders [train_loader, eval_loader].
        writers (list): List of TensorBoard writers [writer_eval].
        cache (list): List to cache data in GPU memory.
        use_cpu (bool): Whether to use CPU for training.
    """
    global global_step, loss_disc

    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader = loaders[0] if loaders is not None else None
    if writers is not None:
        writer = writers[0]

    log_interval = int(hps.train.log_interval)
    if log_interval < 1:
        raise ValueError("train.log_interval must be at least 1")

    train_loader.batch_sampler.set_epoch(epoch)

    net_g.train()
    net_d.train()
    freeze_discriminator_for_generator = device.type == "cuda"

    use_amp = device.type == "cuda" and (
        train_dtype == torch.bfloat16 or train_dtype == torch.float16
    )

    # Data caching
    if device.type == "cuda" and cache_data_in_gpu:
        data_iterator = cache
        if cache == []:
            for batch_idx, info in enumerate(train_loader):
                # phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, wave_lengths, sid
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
                cache.append((batch_idx, info))
        else:
            shuffle(cache)
    else:
        data_iterator = enumerate(train_loader)

    epoch_recorder = EpochRecorder()
    with tqdm(total=len(train_loader), leave=False) as pbar:
        for batch_idx, info in data_iterator:
            if device.type == "cuda" and not cache_data_in_gpu:
                info = [tensor.cuda(device_id, non_blocking=True) for tensor in info]
            elif device.type != "cuda":
                info = [tensor.to(device) for tensor in info]
            # else iterator is going thru a cached list with a device already assigned

            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info

            log_grad_norms = rank == 0 and (global_step + 1) % log_interval == 0

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Forward pass
                model_output = net_g(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
                )
                y_hat, ids_slice, x_mask, z_mask, (z, z_p, m_p, logs_p, m_q, logs_q) = (
                    model_output
                )
                # slice of the original waveform to match a generate slice
                if randomized:
                    wave = commons.slice_segments(
                        wave,
                        ids_slice * config.data.hop_length,
                        config.train.segment_size,
                        dim=3,
                    )
            for discriminator_step in range(d_step_per_g_step):
                with torch.amp.autocast(
                    device_type="cuda", enabled=use_amp, dtype=train_dtype
                ):
                    y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
                loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
                # Discriminator backward and update
                optim_d.zero_grad()
                if train_dtype == torch.float16:
                    scaler.scale(loss_disc).backward()
                    if log_grad_norms and discriminator_step + 1 == d_step_per_g_step:
                        scaler.unscale_(optim_d)
                        grad_norm_d = commons.grad_norm(net_d.parameters())
                    scaler.step(optim_d)
                else:
                    loss_disc.backward()
                    if log_grad_norms and discriminator_step + 1 == d_step_per_g_step:
                        grad_norm_d = commons.grad_norm(net_d.parameters())
                    optim_d.step()

            if freeze_discriminator_for_generator:
                optim_d.zero_grad(set_to_none=True)
                net_d.requires_grad_(False)

            with torch.amp.autocast(
                device_type="cuda", enabled=use_amp, dtype=train_dtype
            ):
                # Generator backward and update
                _, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)

            if multiscale_mel_loss:
                loss_mel = fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
                loss_kl = (
                    kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
                )
                loss_fm = feature_loss(fmap_r, fmap_g)
            else:
                wave_mel = spec_to_mel_torch(
                    spec,
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                wave_mel = commons.slice_segments(
                    wave_mel,
                    ids_slice,
                    config.train.segment_size // config.data.hop_length,
                    dim=3,
                )
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    config.data.filter_length,
                    config.data.n_mel_channels,
                    config.data.sample_rate,
                    config.data.hop_length,
                    config.data.win_length,
                    config.data.mel_fmin,
                    config.data.mel_fmax,
                )
                loss_mel = fn_mel_loss(wave_mel, y_hat_mel) * config.train.c_mel
                loss_kl = (
                    kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
                )
                loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
            optim_g.zero_grad()
            if train_dtype == torch.float16:
                scaler.scale(loss_gen_all).backward()
                if log_grad_norms:
                    scaler.unscale_(optim_g)
                    grad_norm_g = commons.grad_norm(net_g.parameters())
                if inference_exporter is not None:
                    inference_exporter.wait_for_staging()
                scaler.step(optim_g)
                scaler.update()
            else:
                loss_gen_all.backward()
                if log_grad_norms:
                    grad_norm_g = commons.grad_norm(net_g.parameters())
                if inference_exporter is not None:
                    inference_exporter.wait_for_staging()
                optim_g.step()

            if freeze_discriminator_for_generator:
                net_d.requires_grad_(True)

            global_step += 1

            scalar_log_state["loss_sums"].add_(
                torch.stack(
                    (
                        loss_gen_all.detach(),
                        loss_disc.detach(),
                        loss_gen.detach(),
                        loss_fm.detach(),
                        loss_mel.detach(),
                        loss_kl.detach(),
                    )
                ).float()
            )
            scalar_log_state["batch_count"] += 1

            if global_step % log_interval == 0:
                scalar_log_totals = torch.cat(
                    (
                        scalar_log_state["loss_sums"],
                        scalar_log_state["loss_sums"].new_tensor(
                            [scalar_log_state["batch_count"]]
                        ),
                    )
                )
                if dist.get_world_size() > 1:
                    if dist.get_backend() == "gloo" and scalar_log_totals.is_cuda:
                        scalar_log_totals = scalar_log_totals.cpu()
                    dist.reduce(scalar_log_totals, dst=0, op=dist.ReduceOp.SUM)
                if rank == 0:
                    scalar_values = (
                        scalar_log_totals[:-1] / scalar_log_totals[-1]
                    ).cpu().tolist()
                    gradient_values = (
                        torch.stack((grad_norm_d, grad_norm_g)).float().cpu().tolist()
                    )
                    summarize(
                        writer=writer,
                        global_step=global_step,
                        scalars={
                            "loss/g/total": scalar_values[0],
                            "loss/d/adv": scalar_values[1],
                            "learning_rate": optim_g.param_groups[0]["lr"],
                            "loss/g/adv": scalar_values[2],
                            "loss/g/fm": scalar_values[3],
                            "loss/g/mel": scalar_values[4],
                            "loss/g/kl": scalar_values[5],
                            "grad/norm_d": gradient_values[0],
                            "grad/norm_g": gradient_values[1],
                        },
                    )
                scalar_log_state["loss_sums"].zero_()
                scalar_log_state["batch_count"] = 0
            if (
                inference_exporter is not None
                and save_every_steps > 0
                and global_step % save_every_steps == 0
            ):
                inference_model_path = os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
                if not os.path.exists(inference_model_path):
                    inference_model = (
                        net_g.module if hasattr(net_g, "module") else net_g
                    )
                    inference_exporter.submit(
                        model=inference_model,
                        export_dtype=inference_export_dtype,
                        sr=config.data.sample_rate,
                        name=model_name,
                        model_path=inference_model_path,
                        epoch=epoch,
                        step=global_step,
                        hps=hps,
                        vocoder=vocoder,
                    )

            pbar.update(1)
        # end of batch train
    # end of tqdm

    if rank == 0:
        if epoch % save_every_epoch == 0:
            validation_scalars = {}
            mel = spec_to_mel_torch(
                spec,
                config.data.filter_length,
                config.data.n_mel_channels,
                config.data.sample_rate,
                config.data.mel_fmin,
                config.data.mel_fmax,
            )
            if randomized:
                y_mel = commons.slice_segments(
                    mel,
                    ids_slice,
                    config.train.segment_size // config.data.hop_length,
                    dim=3,
                )
            else:
                y_mel = mel
            y_hat_mel = mel_spectrogram_torch(
                y_hat.float().squeeze(1),
                config.data.filter_length,
                config.data.n_mel_channels,
                config.data.sample_rate,
                config.data.hop_length,
                config.data.win_length,
                config.data.mel_fmin,
                config.data.mel_fmax,
            )
            image_dict = {
                "slice/mel_org": plot_spectrogram_to_numpy(
                    y_mel[0].data.cpu().numpy()
                ),
                "slice/mel_gen": plot_spectrogram_to_numpy(
                    y_hat_mel[0].data.cpu().numpy()
                ),
                "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            }
            audio_o = None
            timbre_o = None
            if should_run_external_validation(
                timbre_reference, timbre_validator, mos_validator
            ):
                inference_model = net_g.module if hasattr(net_g, "module") else net_g
                inference_model.eval()
                rng_devices = [device_id] if device.type == "cuda" else []
                try:
                    with deterministic_validation_scope(
                        config.train.seed, cuda_devices=rng_devices
                    ):
                        with torch.amp.autocast(
                            device_type="cuda", enabled=False
                        ):
                            with torch.inference_mode():
                                try:
                                    timbre_o = infer_validation_audio(
                                        inference_model,
                                        timbre_reference[0],
                                        config.data.sample_rate,
                                        config.data.hop_length,
                                    )
                                except Exception as error:
                                    print(
                                        f"External validation generation failed: {error}"
                                    )
                                if audio_reference is not None:
                                    if timbre_o is not None:
                                        audio_o = timbre_o[:1]
                                    else:
                                        audio_o, *_ = inference_model.infer(
                                            *audio_reference
                                        )
                finally:
                    inference_model.train()

            generated_lengths = None
            if timbre_o is not None:
                generated_lengths = (
                    timbre_reference[0][1].detach() * config.data.hop_length
                )

            if (
                timbre_validator is not None
                and timbre_o is not None
                and generated_lengths is not None
            ):
                try:
                    speaker_ids = timbre_reference[3]
                    timbre_scores = timbre_validator.score_batch_accelerated(
                        timbre_o.detach(),
                        generated_lengths,
                        speaker_ids,
                        config.data.sample_rate,
                        device,
                    )
                    if timbre_scores["multi_speaker"]:
                        validation_scalars.update(
                            {
                                "validation/voice_similarity": timbre_scores[
                                    "speaker_mean"
                                ],
                                "validation/ecapa_margin_mean": timbre_scores[
                                    "margin_mean"
                                ],
                                "validation/ecapa_top1_accuracy_percent": timbre_scores[
                                    "top1_accuracy_percent"
                                ],
                                "validation/ecapa_eer_percent": timbre_scores[
                                    "eer_percent"
                                ],
                                "validation/ecapa_min_dcf": timbre_scores["min_dcf"],
                            }
                        )
                    else:
                        validation_scalars["validation/voice_similarity"] = (
                            timbre_scores["speaker_mean"]
                        )
                except Exception as error:
                    print(f"ECAPA timbre validation failed: {error}")
            if (
                mos_validator is not None
                and timbre_o is not None
                and generated_lengths is not None
            ):
                try:
                    validation_scalars["validation/MOS_utmosv2"] = (
                        mos_validator.score_batch(
                            timbre_o.detach(),
                            generated_lengths,
                            timbre_reference[3],
                            config.data.sample_rate,
                        )
                    )
                except Exception as error:
                    print(f"UTMOSv2 validation failed: {error}")
            audio_dict = {}
            if audio_o is not None:
                audio_dict[f"gen/audio_{global_step:07d}"] = audio_o[0, :, :]
            summarize(
                writer=writer,
                global_step=global_step,
                images=image_dict,
                scalars=validation_scalars,
                audios=audio_dict,
                audio_sample_rate=config.data.sample_rate,
            )

    # Save checkpoint
    model_add = []
    model_del = []
    done = False

    if rank == 0:
        # Print training progress
        record = f"{model_name} | epoch={epoch} | step={global_step} | {epoch_recorder.record()}"
        print(record)

        # Save weights every N epochs
        if epoch % save_every_epoch == 0:
            checkpoint_suffix = f"{2333333 if save_only_latest else global_step}.pth"
            save_checkpoint(
                net_g,
                optim_g,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "G_" + checkpoint_suffix),
                scaler,
            )
            save_checkpoint(
                net_d,
                optim_d,
                config.train.learning_rate,
                epoch,
                os.path.join(experiment_dir, "D_" + checkpoint_suffix),
                scaler,
            )
            if custom_save_every_weights and save_every_steps == 0:
                model_add.append(
                    os.path.join(
                        experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                    )
                )

        # Check completion
        if epoch >= custom_total_epoch:
            print(
                f"Training has been successfully completed with {epoch} epoch, {global_step} steps and {round(loss_gen_all.item(), 3)} loss gen."
            )
            # Final model
            model_add.append(
                os.path.join(
                    experiment_dir, f"{model_name}_{epoch}e_{global_step}s.pth"
                )
            )
            done = True

        if done and inference_exporter is not None:
            inference_exporter.wait_for_completion()

        # Clean-up old best epochs
        for m in model_del:
            os.remove(m)

        if model_add:
            ckpt = (
                net_g.module.state_dict()
                if hasattr(net_g, "module")
                else net_g.state_dict()
            )
            for m in model_add:
                if not os.path.exists(m):
                    extract_model(
                        ckpt=ckpt,
                        sr=config.data.sample_rate,
                        name=model_name,
                        model_path=m,
                        epoch=epoch,
                        step=global_step,
                        hps=hps,
                        vocoder=vocoder,
                        export_dtype=inference_export_dtype,
                    )

        if done:
            if inference_exporter is not None:
                inference_exporter.close()
            writer.close()
            os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
