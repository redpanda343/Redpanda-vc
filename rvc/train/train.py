import os
import sys

os.environ["USE_LIBUV"] = "0" if sys.platform == "win32" else "1"
import argparse
import glob
import json
import logging
import math
from concurrent.futures import Future, ThreadPoolExecutor
from random import randint

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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
    load_checkpoint,
    load_wav_to_torch,
    plot_spectrogram_to_numpy,
    save_checkpoint,
    summarize,
)

# Zluda hijack
import shared.zluda
from rvc.algorithm import commons
from rvc.train.process.extract_model import extract_model

VOCODERS = ("HiFi-GAN", "MRF HiFi-GAN", "RefineGAN")
LR_SCHEDULES = ("exponential", "cosine")
BUCKET_BOUNDARIES = (100, 200, 300, 400, 500, 600, 700, 800, 900)
LATEST_CHECKPOINT_SUFFIX = 2333333

# experimental settings
randomized = True
d_lr_coeff = 1.0
g_lr_coeff = 1.0
d_step_per_g_step = 1
bf16_adamw = False

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = os.name == "nt"
try:
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"
except Exception as e:
    print(f'Torch tf32: {e}')

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

    def submit(self, state_dict, export_dtype, **export_args):
        self.wait_for_completion()
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


class GeneratorEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.updates = 0
        live = {
            key: value
            for key, value in model.state_dict().items()
            if "enc_q" not in key
        }
        self.weights = {key: value.clone() for key, value in live.items()}
        float_keys = [key for key, value in live.items() if value.is_floating_point()]
        self.live_floats = [live[key] for key in float_keys]
        self.ema_floats = [self.weights[key] for key in float_keys]
        self.copied = [
            (self.weights[key], value)
            for key, value in live.items()
            if not value.is_floating_point()
        ]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        beta = self.decay
        decay = beta * (1 - beta ** (self.updates - 1)) / (1 - beta**self.updates)
        torch._foreach_lerp_(self.ema_floats, self.live_floats, 1 - decay)
        for target, value in self.copied:
            target.copy_(value)

    def state_dict(self):
        return {"updates": self.updates, "weights": self.weights}

    @torch.no_grad()
    def load_state_dict(self, state):
        weights = state["weights"]
        if weights.keys() != self.weights.keys():
            raise ValueError("the saved EMA weights do not match the generator")
        for key, value in weights.items():
            self.weights[key].copy_(value)
        self.updates = int(state["updates"])


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Train an RVC voice model.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--save-every-epoch", type=int, required=True)
    parser.add_argument("--save-every-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", default="0", help="CUDA device indices joined by -, such as 0-1.")
    parser.add_argument("--pretrained-g", default="")
    parser.add_argument("--pretrained-d", default="")
    parser.add_argument("--vocoder", choices=VOCODERS, default="HiFi-GAN")
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--save-only-latest", action="store_true")
    parser.add_argument("--save-every-weights", action="store_true")
    parser.add_argument("--cache-data-in-gpu", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Ramp the learning rate up over this many steps.",
    )
    parser.add_argument(
        "--lr-decay-steps",
        type=int,
        default=0,
        help="Apply lr_decay every this many steps instead of once per epoch (0).",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=LR_SCHEDULES,
        default="exponential",
        help="exponential applies lr_decay every epoch or every --lr-decay-steps; cosine decays to --lr-final-ratio at --total-steps.",
    )
    parser.add_argument(
        "--lr-final-ratio",
        type=float,
        default=0.1,
        help="Fraction of the starting learning rate that the cosine schedule reaches at the last step.",
    )
    parser.add_argument(
        "--max-batch-seconds",
        type=float,
        default=0.0,
        help="Limit clips per batch times the longest clip to this much audio (0 disables it).",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="Export inference models from an EMA of the generator weights with this decay (0 disables it).",
    )
    args = parser.parse_args(argv)

    for name in ("total_steps", "save_every_epoch", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    for name in ("warmup_steps", "lr_decay_steps", "max_batch_seconds"):
        if not getattr(args, name) >= 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative.")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be at least 0 and below 1.")
    if not 0 < args.lr_final_ratio <= 1:
        parser.error("--lr-final-ratio must be above 0 and at most 1.")
    args.save_every_steps = max(0, args.save_every_steps)
    if args.version == "v1" and (args.sample_rate != 40000 or args.vocoder != "HiFi-GAN"):
        parser.error("RVC v1 training requires HiFi-GAN at 40000 Hz.")

    args.experiment_dir = os.path.join(now_dir, "logs", args.model_name)
    args.train_dtype = configured_train_dtype()
    args.export_dtype = (
        torch.float32 if args.train_dtype == torch.float32 else torch.float16
    )
    if args.train_dtype == torch.float16:
        unreliable = fp16_unreliable_gpus(args.gpus)
        if unreliable:
            print(
                f"FP16 training is unreliable on {', '.join(unreliable)}, so this run trains in FP32."
            )
            args.train_dtype = torch.float32
    args.disc_version = "v3" if args.vocoder == "RefineGAN" else args.version
    args.multiscale_mel_loss = args.vocoder == "RefineGAN"
    return args


def configured_train_dtype():
    try:
        with open(
            os.path.join(now_dir, "assets", "config.json"), "r", encoding="utf-8"
        ) as f:
            precision = json.load(f)["precision"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return torch.float32
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if precision == "fp16" and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def fp16_unreliable_gpus(gpus):
    names = [torch.cuda.get_device_name(int(index)) for index in gpus.split("-")]
    return [name for name in names if "GTX16" in name.upper().replace(" ", "")]


def load_experiment_config(args):
    config_save_path = os.path.join(args.experiment_dir, "config.json")
    try:
        with open(config_save_path, "r", encoding="utf-8") as f:
            config = HParams(**json.load(f))
    except FileNotFoundError:
        print(
            f"Config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?"
        )
        sys.exit(1)

    config.data.training_files = os.path.join(args.experiment_dir, "filelist.txt")
    expected_feature_dim = 256 if args.version == "v1" else 768
    configured_feature_dim = int(getattr(config.model, "text_enc_hidden_dim", 768))
    if configured_feature_dim != expected_feature_dim:
        raise ValueError(
            f"RVC {args.version} training requires {expected_feature_dim}-channel features, "
            f"but the experiment config uses {configured_feature_dim}."
        )
    return config


def read_model_info(args):
    try:
        with open(
            os.path.join(args.experiment_dir, "model_info.json"), "r", encoding="utf-8"
        ) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def check_dataset_sample_rate(args, config):
    dataset_format = str(read_model_info(args).get("dataset_format", "wav")).lower()
    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"
    wavs = glob.glob(
        os.path.join(args.experiment_dir, "sliced_audios", f"*.{dataset_format}")
    )
    if not wavs:
        print(f"No {dataset_format} file found.")
        return
    _, sr = load_wav_to_torch(wavs[0])
    if sr != config.data.sample_rate:
        print(
            f"Error: Pretrained model sample rate ({config.data.sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz)."
        )
        sys.exit(1)


def remove_previous_attempt(experiment_dir):
    for root, dirs, files in os.walk(experiment_dir, topdown=False):
        for name in files:
            file_name, file_extension = os.path.splitext(name)
            if (
                file_extension in (".0", ".pth")
                or name.endswith(".spec.pt")
                or (file_name.startswith("added") and file_extension == ".index")
            ):
                os.remove(os.path.join(root, name))
        for name in dirs:
            if name == "eval":
                folder_path = os.path.join(root, name)
                for item in os.listdir(folder_path):
                    item_path = os.path.join(folder_path, item)
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                os.rmdir(folder_path)


def latest_checkpoint_pair(experiment_dir):
    generators = sorted(
        glob.glob(os.path.join(experiment_dir, "G_*.pth")),
        key=os.path.getmtime,
        reverse=True,
    )
    for generator in generators:
        discriminator = os.path.join(
            experiment_dir, "D_" + os.path.basename(generator)[2:]
        )
        if os.path.isfile(discriminator):
            return generator, discriminator
    if generators:
        return generators[0], None
    discriminators = glob.glob(os.path.join(experiment_dir, "D_*.pth"))
    return None, discriminators[0] if discriminators else None


def main():
    args = parse_arguments()
    config = load_experiment_config(args)
    check_dataset_sample_rate(args, config)

    if torch.cuda.is_available():
        device_type = "cuda"
        device_ids = [int(item) for item in args.gpus.split("-")]
    elif torch.backends.mps.is_available():
        device_type = "mps"
        device_ids = [0]
    else:
        device_type = "cpu"
        device_ids = [0]

    if args.cleanup:
        remove_previous_attempt(args.experiment_dir)

    world_size = len(device_ids)
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(randint(20000, 55555))
        mp.spawn(
            run,
            args=(world_size, args, config, device_type, device_ids),
            nprocs=world_size,
            join=True,
        )
    else:
        run(0, 1, args, config, device_type, device_ids)


def run(rank, world_size, args, config, device_type, device_ids):
    device_id = device_ids[rank]
    if world_size > 1:
        dist.init_process_group(
            backend="gloo" if sys.platform == "win32" else "nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    try:
        torch.manual_seed(config.train.seed)
        if device_type == "cuda":
            torch.cuda.set_device(device_id)
            device = torch.device("cuda", device_id)
        else:
            device = torch.device(device_type)
        Trainer(rank, world_size, args, config, device).fit()
    finally:
        if world_size > 1:
            dist.destroy_process_group()


class Trainer:
    def __init__(self, rank, world_size, args, config, device):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.config = config
        self.device = device
        self.segment_frames = config.train.segment_size // config.data.hop_length
        self.log_interval = int(config.train.log_interval)
        if self.log_interval < 1:
            raise ValueError("train.log_interval must be at least 1")
        self.use_amp = device.type == "cuda" and args.train_dtype in (
            torch.bfloat16,
            torch.float16,
        )
        self.cache_in_gpu = device.type == "cuda" and args.cache_data_in_gpu
        self.global_step = 0
        self.loss_sums = torch.zeros(6, device=device, dtype=torch.float32)
        self.batch_count = 0
        self.cache = None
        self.loader = None
        self.last_batch = None
        self.writer = None
        self.exporter = None
        self.ema = None
        self.lr_schedule = None
        self.progress = None
        self.epoch = 0
        self.logged_losses = None

    def fit(self):
        self.build_data()
        last_g, last_d = latest_checkpoint_pair(self.args.experiment_dir)
        self.build_models(last_g)
        epoch_str, extra = self.restore(last_g, last_d)
        for optimizer in (self.optim_g, self.optim_d):
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])
        if self.args.lr_schedule == "cosine":
            self.build_lr_schedule(extra.get("lr_schedule"))
        if self.rank == 0 and self.args.ema_decay > 0:
            self.build_ema(last_g, extra.get("ema"))

        if self.rank == 0:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.args.experiment_dir, "eval"),
                purge_step=self.global_step + 1 if self.global_step else 0,
                max_queue=1000,
            )
            if self.args.save_every_steps > 0:
                self.exporter = AsyncInferenceExporter(self.device)

        if self.rank == 0:
            print(self.summary(epoch_str, last_g))
        total_steps = self.args.total_steps
        finished = self.global_step >= total_steps
        if finished and self.rank == 0:
            print(
                f"{self.args.model_name} already reached step {total_steps}. "
                "Raise the total steps to continue training."
            )
        self.progress = tqdm(
            total=max(total_steps, self.global_step),
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            disable=self.rank != 0 or finished,
        )
        completed = None
        epoch = epoch_str
        try:
            while self.global_step < total_steps:
                complete = self.train_epoch(epoch)
                if self.finish_epoch(epoch, complete):
                    completed = epoch
                    break
                epoch += 1
        finally:
            self.progress.close()
            if self.exporter is not None:
                self.exporter.close()
            if self.writer is not None:
                self.writer.close()
        if completed is not None:
            print(
                f"{self.args.model_name} | training completed at epoch {completed}, step {self.global_step}"
            )

    def summary(self, epoch_str, last_g):
        args, data = self.args, self.config.data
        precision = (
            {torch.float16: "FP16", torch.bfloat16: "BF16"}[args.train_dtype]
            if self.use_amp
            else "FP32"
        )
        device = f"GPU {args.gpus}" if self.device.type == "cuda" else self.device.type.upper()
        speakers = self.config.model.spk_embed_dim
        parts = [
            args.model_name,
            f"{args.vocoder} {data.sample_rate} Hz {args.version}",
            f"{speakers} speaker{'' if speakers == 1 else 's'}",
            f"{device} {precision}",
            f"batch {args.batch_size}, {len(self.sampler)} steps per epoch",
            f"{args.total_steps} total steps (~{math.ceil(args.total_steps / len(self.sampler))} epochs)",
        ]
        options = []
        if args.max_batch_seconds:
            sizes = self.sampler.bucket_batch_sizes
            options.append(
                f"max {args.max_batch_seconds:g} s audio per batch ({min(sizes)}-{max(sizes)} clips)"
            )
        if args.warmup_steps:
            options.append(f"{args.warmup_steps} warmup steps")
        if self.lr_schedule is not None:
            schedule = self.lr_schedule
            end = min(schedule["final_ratio"], schedule["start_factor"])
            options.append(
                f"cosine LR {schedule['start_factor']:.3g}x to {end:.3g}x from step {schedule['start_step']}"
            )
        elif args.lr_decay_steps:
            options.append(f"lr_decay every {args.lr_decay_steps} steps")
        if self.ema is not None:
            options.append(f"generator EMA {args.ema_decay:g}")
        if self.cache_in_gpu:
            options.append("dataset cached in GPU")
        if args.checkpointing:
            options.append("checkpointing")
        if options:
            parts.append(", ".join(options))
        pretrained = [
            os.path.basename(path)
            for path in (args.pretrained_g, args.pretrained_d)
            if path not in ("", "None")
        ]
        if last_g:
            parts.append(f"resuming at epoch {epoch_str}, step {self.global_step}")
        elif pretrained:
            parts.append(f"from {' and '.join(pretrained)}")
        else:
            parts.append("from scratch")
        return " | ".join(parts)

    def update_postfix(self):
        text = f"epoch {self.epoch}"
        if self.logged_losses is not None:
            total, disc, mel = self.logged_losses
            text += f", g {total:.2f}, d {disc:.2f}, mel {mel:.2f}"
        self.progress.set_postfix_str(text, refresh=False)

    def build_data(self):
        from rvc.train.data_utils import (
            DistributedBucketSampler,
            TextAudioCollateMultiNSFsid,
            TextAudioLoaderMultiNSFsid,
        )

        data = self.config.data
        max_frames = int(self.args.max_batch_seconds * data.sample_rate / data.hop_length)
        self.dataset = TextAudioLoaderMultiNSFsid(data)
        self.collate = TextAudioCollateMultiNSFsid()
        self.sampler = DistributedBucketSampler(
            self.dataset,
            self.args.batch_size,
            list(BUCKET_BOUNDARIES),
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            max_frames=max_frames or None,
        )
        if len(self.sampler) < 3:
            print(
                "Not enough data present in the training set. Perhaps you forgot to slice the audio files in preprocess?"
            )
            sys.exit(1)
        if not self.cache_in_gpu:
            self.loader = DataLoader(
                self.dataset,
                num_workers=4,
                shuffle=False,
                pin_memory=self.device.type == "cuda",
                collate_fn=self.collate,
                batch_sampler=self.sampler,
                persistent_workers=True,
                prefetch_factor=8,
            )

    def build_models(self, last_g):
        from rvc.algorithm.discriminators import MultiPeriodDiscriminator
        from rvc.algorithm.synthesizers import Synthesizer

        config, args = self.config, self.args
        spk_dim = config.model.spk_embed_dim
        try:
            spk_dim = read_model_info(args)["speakers_id"]
        except Exception as e:
            print(f"Could not load model info file: {e}. Using defaults.")
        if last_g:
            try:
                ckpt = torch.load(last_g, map_location="cpu", weights_only=True)
                spk_dim = ckpt["model"]["emb_g.weight"].shape[0]
                del ckpt
            except Exception as e:
                print(f"Failed to load checkpoint: {e}. Using default number of speakers.")
        config.model.spk_embed_dim = spk_dim

        net_g = Synthesizer(
            config.data.filter_length // 2 + 1,
            self.segment_frames,
            **config.model,
            use_f0=True,
            sr=config.data.sample_rate,
            vocoder=args.vocoder,
            checkpointing=args.checkpointing,
            randomized=randomized,
        ).to(self.device)
        net_d = MultiPeriodDiscriminator(
            config.model.use_spectral_norm,
            checkpointing=args.checkpointing,
            version=args.disc_version,
        ).to(self.device)

        if bf16_adamw and args.train_dtype == torch.bfloat16:
            from rvc.train.anyprecision_optimizer import AnyPrecisionAdamW

            optimizer = AnyPrecisionAdamW
            options = {}
        else:
            optimizer = torch.optim.AdamW
            options = {"fused": self.device.type == "cuda"}
        self.optim_g = optimizer(
            net_g.parameters(),
            config.train.learning_rate * g_lr_coeff,
            betas=config.train.betas,
            eps=config.train.eps,
            **options,
        )
        self.optim_d = optimizer(
            net_d.parameters(),
            config.train.learning_rate * d_lr_coeff,
            betas=config.train.betas,
            eps=config.train.eps,
            **options,
        )
        if args.multiscale_mel_loss:
            self.fn_mel_loss = MultiScaleMelSpectrogramLoss(
                sample_rate=config.data.sample_rate
            )
        else:
            self.fn_mel_loss = torch.nn.L1Loss()

        if self.world_size > 1 and self.device.type == "cuda":
            net_g = DDP(net_g, device_ids=[self.device.index])
            net_d = DDP(net_d, device_ids=[self.device.index])
        self.net_g, self.net_d = net_g, net_d
        self.freeze_discriminator = self.device.type == "cuda"
        self.net_d_for_generator = (
            net_d.module if self.freeze_discriminator and hasattr(net_d, "module") else net_d
        )

        self.scaler = torch.amp.GradScaler(
            enabled=self.device.type == "cuda" and args.train_dtype == torch.float16
        )

    def restore(self, last_g, last_d):
        if last_g or last_d:
            try:
                if not (last_g and last_d):
                    raise FileNotFoundError(
                        "resuming needs a G_*.pth and a D_*.pth checkpoint with the same suffix"
                    )
                _, _, _, epoch_str, scaler_dict, _ = load_checkpoint(
                    last_d, self.net_d, self.optim_d
                )
                _, _, _, epoch_str, _, extra = load_checkpoint(
                    last_g, self.net_g, self.optim_g
                )
            except Exception as e:
                print(f"Could not resume from the checkpoints in {self.args.experiment_dir}: {e}")
                print("Fix or remove them, or enable cleanup to start over from the pretrained models.")
                sys.exit(1)
            if scaler_dict:
                self.scaler.load_state_dict(scaler_dict)
            epoch_str += 1
            step = extra.get("step")
            self.global_step = (
                int(step) if step is not None else (epoch_str - 1) * len(self.sampler)
            )
            return epoch_str, extra

        self.load_pretrained(self.net_g, self.args.pretrained_g, "G", match_speakers=True)
        self.load_pretrained(self.net_d, self.args.pretrained_d, "D")
        return 1, {}

    def build_lr_schedule(self, saved):
        target = {
            "total_steps": self.args.total_steps,
            "final_ratio": self.args.lr_final_ratio,
        }
        if saved and all(saved.get(key) == value for key, value in target.items()):
            self.lr_schedule = saved
            return
        start_factor = 1.0
        if self.global_step:
            group = self.optim_g.param_groups[0]
            start_factor = (
                group["lr"] / group["initial_lr"] / self.warmup_ramp(self.global_step - 1)
            )
        self.lr_schedule = dict(
            target,
            start_step=max(self.global_step, self.args.warmup_steps),
            start_factor=start_factor,
        )

    def build_ema(self, last_g, ema_state):
        self.ema = GeneratorEMA(self.generator(), self.args.ema_decay)
        if ema_state is not None:
            try:
                self.ema.load_state_dict(ema_state)
                return
            except Exception as e:
                print(f"Could not restore the generator EMA from {last_g}: {e}")
        if last_g:
            print("Starting the generator EMA from the resumed weights.")

    def load_pretrained(self, net, path, label, match_speakers=False):
        if path in ("", "None"):
            return
        target = net.module if hasattr(net, "module") else net
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)["model"]
            if match_speakers:
                if "emb_g.weight" not in ckpt:
                    raise KeyError("The pretrained generator has no speaker embedding.")
                speakers = target.emb_g.weight.detach().cpu().clone()
                rows = min(speakers.shape[0], ckpt["emb_g.weight"].shape[0])
                speakers[:rows] = ckpt["emb_g.weight"][:rows].to(speakers.dtype)
                ckpt["emb_g.weight"] = speakers
            target.load_state_dict(ckpt)
            del ckpt
        except Exception as e:
            print(
                "The parameters of the pretrain model such as the sample rate or architecture do not match the selected model."
            )
            print(e)
            sys.exit(1)

    def autocast(self):
        return torch.amp.autocast(
            device_type="cuda", enabled=self.use_amp, dtype=self.args.train_dtype
        )

    def batches(self):
        if not self.cache_in_gpu:
            for batch in self.loader:
                if self.device.type == "cuda":
                    yield [tensor.to(self.device, non_blocking=True) for tensor in batch]
                else:
                    yield [tensor.to(self.device) for tensor in batch]
            return
        if self.cache is None:
            loader = DataLoader(
                self.dataset, batch_size=None, num_workers=4, pin_memory=True
            )
            self.cache = [
                tuple(tensor.to(self.device, non_blocking=True) for tensor in item)
                for item in loader
            ]
        for indices in self.sampler:
            batch = self.collate([self.cache[index] for index in indices])
            yield [tensor.to(self.device, non_blocking=True) for tensor in batch]

    def train_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.net_g.train()
        self.net_d.train()
        self.epoch = epoch
        self.update_postfix()
        steps = 0
        for batch in self.batches():
            self.set_learning_rate(epoch)
            self.train_step(batch)
            self.export_step_weights(epoch)
            self.progress.update(1)
            steps += 1
            if self.global_step >= self.args.total_steps:
                break
        return steps == len(self.sampler)

    def warmup_ramp(self, step):
        warmup = self.args.warmup_steps
        return (step + 1) / warmup if step < warmup else 1.0

    def exponential_factor(self, epoch, step):
        args, decay = self.args, self.config.train.lr_decay
        if args.lr_decay_steps:
            return decay ** (max(0, step - args.warmup_steps) / args.lr_decay_steps)
        return decay ** (epoch - 1)

    def cosine_factor(self, step):
        schedule = self.lr_schedule
        start, start_factor = schedule["start_step"], schedule["start_factor"]
        end = min(schedule["final_ratio"], start_factor)
        span = schedule["total_steps"] - 1 - start
        progress = min(1.0, max(0, step - start) / span) if span > 0 else 1.0
        return end + (start_factor - end) * 0.5 * (1 + math.cos(math.pi * progress))

    def set_learning_rate(self, epoch):
        step = self.global_step
        if self.lr_schedule is not None:
            factor = self.cosine_factor(step)
        else:
            factor = self.exponential_factor(epoch, step)
        factor *= self.warmup_ramp(step)
        for optimizer in (self.optim_g, self.optim_d):
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * factor

    def optimizer_step(self, optimizer, net, loss, log_grad_norm, update_scaler, before_step=None):
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        norm = None
        if log_grad_norm:
            self.scaler.unscale_(optimizer)
            norm = commons.grad_norm(net.parameters())
        if before_step is not None:
            before_step()
        self.scaler.step(optimizer)
        if update_scaler:
            self.scaler.update()
        return norm

    def mel_loss(self, wave, y_hat, spec, ids_slice):
        config = self.config
        if self.args.multiscale_mel_loss:
            return self.fn_mel_loss(wave, y_hat) * config.train.c_mel / 3.0
        wave_mel = spec_to_mel_torch(
            commons.slice_segments(spec, ids_slice, self.segment_frames, dim=3),
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
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
        return self.fn_mel_loss(wave_mel, y_hat_mel) * config.train.c_mel

    def train_step(self, batch):
        config = self.config
        self.last_batch = None
        phone, phone_lengths, pitch, pitchf, spec, spec_lengths, wave, _, sid = batch
        log_grad_norms = self.rank == 0 and (self.global_step + 1) % self.log_interval == 0

        with self.autocast():
            y_hat, ids_slice, _, z_mask, (_, z_p, m_p, logs_p, _, logs_q) = self.net_g(
                phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid
            )
            if randomized:
                wave = commons.slice_segments(
                    wave,
                    ids_slice * config.data.hop_length,
                    config.train.segment_size,
                    dim=3,
                )

        for discriminator_step in range(d_step_per_g_step):
            last_discriminator_step = discriminator_step + 1 == d_step_per_g_step
            with self.autocast():
                y_d_hat_r, y_d_hat_g, _, _ = self.net_d(wave, y_hat.detach())
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)
            grad_norm_d = self.optimizer_step(
                self.optim_d,
                self.net_d,
                loss_disc,
                log_grad_norms and last_discriminator_step,
                update_scaler=not last_discriminator_step,
            )

        if self.freeze_discriminator:
            self.optim_d.zero_grad(set_to_none=True)
            self.net_d.requires_grad_(False)

        with self.autocast():
            _, y_d_hat_g, fmap_r, fmap_g = self.net_d_for_generator(wave, y_hat)

        loss_mel = self.mel_loss(wave, y_hat, spec, ids_slice)
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * config.train.c_kl
        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, _ = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        grad_norm_g = self.optimizer_step(
            self.optim_g,
            self.net_g,
            loss_gen_all,
            log_grad_norms,
            update_scaler=True,
            before_step=self.exporter.wait_for_staging if self.exporter else None,
        )

        if self.freeze_discriminator:
            self.net_d.requires_grad_(True)
        if self.ema is not None:
            self.ema.update()

        self.global_step += 1
        self.record_losses(
            (loss_gen_all, loss_disc, loss_gen, loss_fm, loss_mel, loss_kl),
            (grad_norm_d, grad_norm_g),
        )
        self.last_batch = (spec, ids_slice, y_hat)

    def record_losses(self, losses, grad_norms):
        self.loss_sums.add_(torch.stack([loss.detach() for loss in losses]).float())
        self.batch_count += 1
        if self.global_step % self.log_interval != 0:
            return
        totals = torch.cat((self.loss_sums, self.loss_sums.new_tensor([self.batch_count])))
        if self.world_size > 1:
            if dist.get_backend() == "gloo" and totals.is_cuda:
                totals = totals.cpu()
            dist.reduce(totals, dst=0, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            loss_values = (totals[:-1] / totals[-1]).cpu().tolist()
            self.logged_losses = (loss_values[0], loss_values[1], loss_values[4])
            self.update_postfix()
            gradient_values = torch.stack(grad_norms).float().cpu().tolist()
            summarize(
                writer=self.writer,
                global_step=self.global_step,
                scalars={
                    "loss/g/total": loss_values[0],
                    "loss/d/adv": loss_values[1],
                    "learning_rate": self.optim_g.param_groups[0]["lr"],
                    "loss/g/adv": loss_values[2],
                    "loss/g/fm": loss_values[3],
                    "loss/g/mel": loss_values[4],
                    "loss/g/kl": loss_values[5],
                    "grad/norm_d": gradient_values[0],
                    "grad/norm_g": gradient_values[1],
                },
            )
        self.loss_sums.zero_()
        self.batch_count = 0

    def generator(self):
        return self.net_g.module if hasattr(self.net_g, "module") else self.net_g

    def export_state_dict(self):
        return self.ema.weights if self.ema is not None else self.generator().state_dict()

    def weights_path(self, epoch):
        return os.path.join(
            self.args.experiment_dir,
            f"{self.args.model_name}_{epoch}e_{self.global_step}s.pth",
        )

    def export_arguments(self, epoch):
        return dict(
            sr=self.config.data.sample_rate,
            name=self.args.model_name,
            epoch=epoch,
            step=self.global_step,
            hps=self.config,
            vocoder=self.args.vocoder,
            version=self.args.version,
            verbose=False,
        )

    def export_step_weights(self, epoch):
        if self.exporter is None or self.global_step % self.args.save_every_steps != 0:
            return
        path = self.weights_path(epoch)
        if not os.path.exists(path):
            self.exporter.submit(
                state_dict=self.export_state_dict(),
                export_dtype=self.args.export_dtype,
                model_path=path,
                **self.export_arguments(epoch),
            )

    def summarize_epoch(self, spec, ids_slice, y_hat):
        config = self.config
        mel = spec_to_mel_torch(
            spec,
            config.data.filter_length,
            config.data.n_mel_channels,
            config.data.sample_rate,
            config.data.mel_fmin,
            config.data.mel_fmax,
        )
        y_mel = (
            commons.slice_segments(mel, ids_slice, self.segment_frames, dim=3)
            if randomized
            else mel
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
        summarize(
            writer=self.writer,
            global_step=self.global_step,
            images={
                "slice/mel_org": plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
                "slice/mel_gen": plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
                "all/mel": plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            },
        )

    def finish_epoch(self, epoch, complete):
        if self.rank != 0:
            return False
        args = self.args
        spec, ids_slice, y_hat = self.last_batch
        save_epoch = complete and epoch % args.save_every_epoch == 0
        done = self.global_step >= args.total_steps
        if save_epoch:
            self.summarize_epoch(spec, ids_slice, y_hat)

        weights = []
        if save_epoch:
            suffix = LATEST_CHECKPOINT_SUFFIX if args.save_only_latest else self.global_step
            generator_extra = {"step": self.global_step}
            if self.ema is not None:
                generator_extra["ema"] = self.ema.state_dict()
            if self.lr_schedule is not None:
                generator_extra["lr_schedule"] = self.lr_schedule
            for prefix, net, optimizer, extra in (
                ("G", self.net_g, self.optim_g, generator_extra),
                ("D", self.net_d, self.optim_d, None),
            ):
                save_checkpoint(
                    net,
                    optimizer,
                    self.config.train.learning_rate,
                    epoch,
                    os.path.join(args.experiment_dir, f"{prefix}_{suffix}.pth"),
                    self.scaler,
                    extra,
                )
            if args.save_every_weights and args.save_every_steps == 0:
                weights.append(self.weights_path(epoch))

        if done:
            weights.append(self.weights_path(epoch))
            if self.exporter is not None:
                self.exporter.wait_for_completion()

        state_dict = self.export_state_dict() if weights else None
        for path in weights:
            if not os.path.exists(path):
                extract_model(
                    ckpt=state_dict,
                    model_path=path,
                    export_dtype=args.export_dtype,
                    **self.export_arguments(epoch),
                )
        return done


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
