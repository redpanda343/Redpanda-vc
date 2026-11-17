"""Native JAX/XLA trainer for RVC models (TPU, GPU, CPU) with PyTorch .pth inference export."""

from __future__ import annotations

import os
import sys

# Pre-parse accelerator and GPU selection before importing JAX so XLA sees CUDA_VISIBLE_DEVICES.
def _preparse_device_env(argv: list[str] | None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    acc = "auto"
    gpus = None
    for i, tok in enumerate(raw):
        if tok == "--accelerator" and i + 1 < len(raw):
            acc = raw[i + 1].lower()
        elif tok.startswith("--accelerator="):
            acc = tok.split("=", 1)[1].lower()
        elif tok == "--gpus" and i + 1 < len(raw):
            gpus = raw[i + 1]
        elif tok.startswith("--gpus="):
            gpus = tok.split("=", 1)[1]
    if acc == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif acc in ("auto", "gpu") and gpus and gpus != "-" and all(p.isdigit() for p in gpus.split("-")):
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpus.replace("-", ","))


_preparse_device_env(None)

import argparse
import glob
import json
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from tqdm import tqdm

now_dir = os.getcwd()
if now_dir not in sys.path:
    sys.path.append(now_dir)

from rvc.jax import JAX_BUCKET_BOUNDARIES, PrecisionConfig, resolve_jax_devices, supports_bfloat16
from rvc.jax.checkpoint import (
    LATEST_CHECKPOINT_SUFFIX,
    AsyncJaxInferenceExporter,
    latest_checkpoint_pair,
    load_jax_checkpoint,
    load_pretrained_torch_d,
    load_pretrained_torch_g,
    read_jax_checkpoint_meta,
    save_jax_checkpoint,
)
from rvc.jax.data import BucketBatchSampler, PrefetchBatchLoader, TextAudioDataset
from rvc.jax.discriminators import (
    MultiPeriodDiscriminatorConfig,
    init_mpd_params,
    mpd_forward,
    mpd_forward_single,
    resolve_mpd_params,
)
from rvc.jax.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from rvc.jax.mel import (
    MultiScaleMelLossConfig,
    mel_spectrogram_jax,
    multi_scale_mel_loss,
    spec_to_mel_jax,
    spectrogram_jax_padded,
)
from rvc.jax.ops import slice_segments
from rvc.jax.synthesizers import (
    init_synthesizer_params,
    synthesizer_config_from_hparams,
    synthesizer_forward,
)
from rvc.train.utils import HParams, load_wav_to_torch, plot_spectrogram_to_numpy, summarize

VOCODERS = ("HiFi-GAN", "MRF HiFi-GAN", "RefineGAN")
LR_SCHEDULES = ("exponential", "cosine")
BUCKET_BOUNDARIES = JAX_BUCKET_BOUNDARIES
LEGACY_BUCKET_BOUNDARIES = (100, 200, 300, 400, 500, 600, 700, 800, 900)

# Experimental settings matching main
randomized = True
d_lr_coeff = 0.2
g_lr_coeff = 1.0
d_step_per_g_step = 1


def _init_xla_cache() -> None:
    cache_dir = os.path.join(now_dir, "assets", "xla_cache")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        jax.config.update("jax_compilation_cache_dir", cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    except Exception:
        pass


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an RVC voice model with JAX/XLA.")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--sample-rate", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument(
        "--save-every-steps",
        type=int,
        required=True,
        help="Save the G and D checkpoints every this many steps. The last step always saves them.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", default="0", help="CUDA device indices joined by -, such as 0-1.")
    parser.add_argument("--accelerator", choices=("auto", "tpu", "gpu", "cpu"), default="auto")
    parser.add_argument("--tpu-cores", choices=("all", "1"), default="all")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp32"), default="auto")
    parser.add_argument("--pretrained-g", default="")
    parser.add_argument("--pretrained-d", default="")
    parser.add_argument("--vocoder", choices=VOCODERS, default="HiFi-GAN")
    parser.add_argument("--version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--save-only-latest", action="store_true")
    parser.add_argument(
        "--save-every-weights",
        action="store_true",
        help="Save an inference model with every G and D checkpoint. The last step always saves one.",
    )
    parser.add_argument("--cache-data-in-gpu", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming, start optimizers without their saved moments.",
    )
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--lr-decay-steps", type=int, default=0)
    parser.add_argument("--lr-schedule", choices=LR_SCHEDULES, default="exponential")
    parser.add_argument("--lr-final-ratio", type=float, default=0.1)
    parser.add_argument("--max-batch-seconds", type=float, default=0.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    args = parser.parse_args(argv)

    for name in ("total_steps", "save_every_steps", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    for name in ("warmup_steps", "lr_decay_steps", "max_batch_seconds"):
        if not getattr(args, name) >= 0:
            parser.error(f"--{name.replace('_', '-')} cannot be negative.")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be at least 0 and below 1.")
    if not 0 < args.lr_final_ratio <= 1:
        parser.error("--lr-final-ratio must be above 0 and at most 1.")
    if args.version == "v1" and (args.sample_rate != 40000 or args.vocoder != "HiFi-GAN"):
        parser.error("RVC v1 training requires HiFi-GAN at 40000 Hz.")
    if args.sample_rate == 22000 and (args.version != "v2" or args.vocoder != "HiFi-GAN"):
        parser.error("22000 Hz training only supports RVC v2 with HiFi-GAN.")

    args.experiment_dir = os.path.join(now_dir, "logs", args.model_name)
    args.disc_version = "v3" if args.vocoder == "RefineGAN" else args.version
    args.multiscale_mel_loss = args.vocoder == "RefineGAN"
    return args


def _configured_precision_setting() -> str:
    try:
        with open(os.path.join(now_dir, "assets", "config.json"), "r", encoding="utf-8") as f:
            return str(json.load(f).get("precision", "fp16")).lower()
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "fp16"


def resolve_precision(args: argparse.Namespace, devices: list[jax.Device]) -> tuple[PrecisionConfig, str, str]:
    """Choose BF16 MXU compute or pure FP32 compute, plus inference export dtype."""
    bf16_hw = supports_bfloat16(devices)
    cfg_prec = _configured_precision_setting()
    if args.dtype == "bf16":
        return PrecisionConfig.bfloat16(), "BF16", "float16"
    if args.dtype == "fp32":
        return PrecisionConfig.float32(), "FP32", "float32"
    if bf16_hw:
        return PrecisionConfig.bfloat16(), "BF16", "float32" if cfg_prec == "fp32" else "float16"
    export_str = "float32" if cfg_prec == "fp32" or devices[0].platform == "cpu" else "float16"
    return PrecisionConfig.float32(), "FP32", export_str


def load_experiment_config(args: argparse.Namespace) -> HParams:
    config_save_path = os.path.join(args.experiment_dir, "config.json")
    try:
        with open(config_save_path, "r", encoding="utf-8") as f:
            config = HParams(**json.load(f))
    except FileNotFoundError:
        print(f"Config file not found at {config_save_path}. Did you run preprocessing and feature extraction steps?")
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


def read_model_info(args: argparse.Namespace) -> dict:
    try:
        with open(os.path.join(args.experiment_dir, "model_info.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def check_dataset_sample_rate(args: argparse.Namespace, config: HParams) -> None:
    dataset_format = str(read_model_info(args).get("dataset_format", "wav")).lower()
    if dataset_format not in {"wav", "flac"}:
        dataset_format = "wav"
    wavs = glob.glob(os.path.join(args.experiment_dir, "sliced_audios", f"*.{dataset_format}"))
    if not wavs:
        print(f"No {dataset_format} file found.")
        return
    _, sr = load_wav_to_torch(wavs[0])
    if sr != config.data.sample_rate:
        print(f"Error: Pretrained model sample rate ({config.data.sample_rate} Hz) does not match dataset audio sample rate ({sr} Hz).")
        sys.exit(1)


def remove_previous_attempt(experiment_dir: str) -> None:
    if not os.path.isdir(experiment_dir):
        return
    for root, dirs, files in os.walk(experiment_dir, topdown=False):
        for name in files:
            file_name, file_extension = os.path.splitext(name)
            if (
                file_extension in (".0", ".pth", ".npz")
                or name.endswith((".spec.pt", ".pth.tmp", ".tmp.npz"))
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


def _create_summary_writer(log_dir: str, purge_step: int):
    for mod_name in ("torch.utils.tensorboard", "tensorboardX"):
        try:
            mod = __import__(mod_name, fromlist=["SummaryWriter"])
            return mod.SummaryWriter(log_dir=log_dir, purge_step=purge_step, max_queue=1000)
        except Exception:
            continue
    return None


class JaxTrainer:
    def __init__(self, args: argparse.Namespace, config: HParams) -> None:
        self.args = args
        self.config = config
        _init_xla_cache()

        self.devices = resolve_jax_devices(args.accelerator, args.tpu_cores)
        self.num_devices = len(self.devices)
        self.platform = self.devices[0].platform
        self.prec, self.precision_label, self.export_dtype_str = resolve_precision(args, self.devices)

        self.mesh = Mesh(np.array(self.devices), ("data",))
        self.replicated_sharding = NamedSharding(self.mesh, PartitionSpec())
        self.data_sharding = NamedSharding(self.mesh, PartitionSpec("data"))

        self.segment_frames = int(config.train.segment_size) // int(config.data.hop_length)
        self.workers = min(8, os.cpu_count() or 1)
        self.log_interval = 200
        self.global_step = 0
        self.epoch = 1
        self.epoch_step = 0
        self.ema_updates = 0
        self.lr_schedule: dict | None = None
        self.logged_losses: tuple[float, float, float] | None = None
        self.loss_sums = jax.device_put(jnp.zeros((6,), dtype=jnp.float32), self.replicated_sharding)
        self.batch_count = 0
        self.last_batch: tuple | None = None
        self.rng_key = jax.random.PRNGKey(int(getattr(config.train, "seed", 1234)))

        self.writer = None
        self.exporter: AsyncJaxInferenceExporter | None = None
        self.progress = None

    def build_data(self) -> None:
        data = self.config.data
        max_frames = int(self.args.max_batch_seconds * data.sample_rate / data.hop_length)
        self.dataset = TextAudioDataset(data, compute_spec=False)
        use_tpu_buckets = self.platform in ("tpu", "gpu")
        boundaries = list(BUCKET_BOUNDARIES if use_tpu_buckets else LEGACY_BUCKET_BOUNDARIES)
        # Ensure tiny test clips are also representable if user didn't filter or if dataset has short clips
        if self.dataset.lengths and min(self.dataset.lengths) <= boundaries[0] and max(self.dataset.lengths) <= boundaries[0]:
            boundaries = [max(1, self.segment_frames), boundaries[-1]]

        if self.args.batch_size % self.num_devices != 0:
            aligned_bs = ((self.args.batch_size + self.num_devices - 1) // self.num_devices) * self.num_devices
            print(f"Adjusting batch size from {self.args.batch_size} to {aligned_bs} for {self.num_devices} JAX devices.")
            self.args.batch_size = aligned_bs

        self.sampler = BucketBatchSampler(
            self.dataset.lengths,
            self.args.batch_size,
            boundaries,
            shuffle=True,
            max_frames=max_frames or None,
            batch_multiple=self.num_devices,
            static_bucket_padding=use_tpu_buckets,
        )
        if len(self.sampler) < 3 and not os.environ.get("REDPANDA_ALLOW_SMALL_DATASET"):
            print("Not enough data present in the training set. Perhaps you forgot to slice the audio files in preprocess?")
            sys.exit(1)

        self.loader = PrefetchBatchLoader(
            self.dataset,
            self.sampler,
            hop_length=int(data.hop_length),
            num_workers=self.workers,
            prefetch_batches=8,
            cache_in_ram=bool(self.args.cache_data_in_gpu),
        )

    def build_models(self, last_g: str | None) -> None:
        config, args = self.config, self.args
        spk_dim = int(config.model.spk_embed_dim)
        try:
            spk_dim = int(read_model_info(args).get("speakers_id", spk_dim))
        except Exception as exc:
            print(f"Could not load model info file: {exc}. Using defaults.")

        if last_g:
            try:
                if last_g.lower().endswith(".npz"):
                    meta = read_jax_checkpoint_meta(last_g)
                    if "spk_embed_dim" in meta:
                        spk_dim = int(meta["spk_embed_dim"])
                else:
                    import torch
                    ckpt = torch.load(last_g, map_location="cpu", weights_only=True)
                    spk_dim = int(ckpt["model"]["emb_g.weight"].shape[0])
            except Exception as exc:
                print(f"Failed to read speaker count from checkpoint: {exc}. Using default {spk_dim}.")

        config.model.spk_embed_dim = spk_dim
        self.syn_cfg = synthesizer_config_from_hparams(
            config,
            vocoder=args.vocoder,
            use_f0=True,
            checkpointing=args.checkpointing,
            randomized=randomized,
        )
        self.mpd_cfg = MultiPeriodDiscriminatorConfig(
            use_spectral_norm=bool(config.model.use_spectral_norm),
            checkpointing=args.checkpointing,
            version=args.disc_version,
        )

        seed = int(getattr(config.train, "seed", 1234))
        rng_np = np.random.default_rng(seed)
        self.host_params_g = init_synthesizer_params(rng_np, self.syn_cfg)
        self.host_params_d = init_mpd_params(rng_np, self.mpd_cfg)

        b1, b2 = (float(x) for x in config.train.betas)
        eps = float(config.train.eps)
        base_lr = float(config.train.learning_rate)
        self.opt_g = optax.inject_hyperparams(optax.adamw)(
            learning_rate=base_lr * g_lr_coeff, b1=b1, b2=b2, eps=eps, weight_decay=0.0
        )
        self.opt_d = optax.inject_hyperparams(optax.adamw)(
            learning_rate=base_lr * d_lr_coeff, b1=b1, b2=b2, eps=eps, weight_decay=0.0
        )
        self.opt_state_g = self.opt_g.init(self.host_params_g)
        self.opt_state_d = self.opt_d.init(self.host_params_d)
        self.ema_params_host = (
            {k: jax.tree.map(np.copy, v) for k, v in self.host_params_g.items() if k != "enc_q"}
            if args.ema_decay > 0
            else {}
        )

    def restore(self, last_g: str | None, last_d: str | None) -> tuple[int, int, dict]:
        if last_g or last_d:
            if not (last_g and last_d):
                raise FileNotFoundError("Resuming needs both G and D checkpoints with the same suffix.")
            if last_g.lower().endswith(".npz"):
                p_d, opt_d, _, _, d_extra, _ = load_jax_checkpoint(
                    last_d,
                    self.host_params_d,
                    self.opt_d,
                    self.opt_state_d,
                    reset_state=self.args.reset_optimizer,
                )
                p_g, opt_g, _, epoch, extra, ema_loaded = load_jax_checkpoint(
                    last_g,
                    self.host_params_g,
                    self.opt_g,
                    self.opt_state_g,
                    reset_state=self.args.reset_optimizer,
                )
                self.host_params_d, self.opt_state_d = p_d, opt_d
                self.host_params_g, self.opt_state_g = p_g, opt_g
                if self.args.ema_decay > 0 and ema_loaded is not None:
                    self.ema_params_host = ema_loaded
                    self.ema_updates = int(extra.get("ema_updates", 0))
                step = extra.get("step")
                if d_extra.get("step") not in (None, step):
                    print(f"{os.path.basename(last_d)} was saved at step {d_extra['step']}, {os.path.basename(last_g)} at step {step}.")
                start_batch = int(extra.get("epoch_step", 0))
                if start_batch >= len(self.sampler):
                    epoch, start_batch = epoch + 1, 0
                self.global_step = int(step) if step is not None else (epoch - 1) * len(self.sampler) + start_batch
                if "rng_key" in extra:
                    self.rng_key = jnp.asarray(extra["rng_key"], dtype=jnp.uint32)
                return epoch, start_batch, extra

            # Legacy PyTorch .pth checkpoint migration
            import torch
            d_ckpt = torch.load(last_d, map_location="cpu", weights_only=True)
            g_ckpt = torch.load(last_g, map_location="cpu", weights_only=True)
            self.host_params_d = load_pretrained_torch_d(last_d, self.mpd_cfg)
            self.host_params_g = load_pretrained_torch_g(last_g, self.syn_cfg, self.host_params_g, match_speakers=False)
            self.opt_state_g = self.opt_g.init(self.host_params_g)
            self.opt_state_d = self.opt_d.init(self.host_params_d)
            if self.args.ema_decay > 0:
                self.ema_params_host = {k: jax.tree.map(np.copy, v) for k, v in self.host_params_g.items() if k != "enc_q"}
            epoch = int(g_ckpt.get("iteration", 1))
            step = int(g_ckpt.get("step", (epoch - 1) * len(self.sampler)))
            start_batch = int(g_ckpt.get("epoch_step", 0))
            if start_batch >= len(self.sampler):
                epoch, start_batch = epoch + 1, 0
            self.global_step = step
            print(f"Imported PyTorch weights from {os.path.basename(last_g)} into native JAX with fresh Optax moments.")
            return epoch, start_batch, {k: v for k, v in g_ckpt.items() if k in ("lr_schedule",)}

        if self.args.pretrained_g not in ("", "None"):
            self.host_params_g = load_pretrained_torch_g(
                self.args.pretrained_g, self.syn_cfg, self.host_params_g, match_speakers=True
            )
            if self.args.ema_decay > 0:
                self.ema_params_host = {k: jax.tree.map(np.copy, v) for k, v in self.host_params_g.items() if k != "enc_q"}
        if self.args.pretrained_d not in ("", "None"):
            self.host_params_d = load_pretrained_torch_d(self.args.pretrained_d, self.mpd_cfg)
        return 1, 0, {}

    def _put_replicated(self) -> None:
        self.params_g = jax.device_put(jax.tree.map(jnp.asarray, self.host_params_g), self.replicated_sharding)
        self.opt_state_g = jax.device_put(self.opt_state_g, self.replicated_sharding)
        self.params_d = jax.device_put(jax.tree.map(jnp.asarray, self.host_params_d), self.replicated_sharding)
        self.opt_state_d = jax.device_put(self.opt_state_d, self.replicated_sharding)
        self.ema_params = jax.device_put(jax.tree.map(jnp.asarray, self.ema_params_host), self.replicated_sharding)

    def build_lr_schedule(self, saved: dict | None) -> None:
        target = {"total_steps": self.args.total_steps, "final_ratio": self.args.lr_final_ratio}
        if saved and all(saved.get(k) == v for k, v in target.items()):
            self.lr_schedule = saved
            return
        start_factor = 1.0
        if self.global_step:
            lr_cur = float(self.opt_state_g.hyperparams["learning_rate"])
            lr_init = float(self.config.train.learning_rate) * g_lr_coeff
            ramp = self.warmup_ramp(self.global_step - 1)
            start_factor = (lr_cur / lr_init / ramp) if (lr_init and ramp) else 1.0
        self.lr_schedule = dict(
            target,
            start_step=max(self.global_step, self.args.warmup_steps),
            start_factor=start_factor,
        )

    def warmup_ramp(self, step: int) -> float:
        warmup = self.args.warmup_steps
        return (step + 1) / warmup if step < warmup else 1.0

    def exponential_factor(self, epoch: int, step: int) -> float:
        decay = float(self.config.train.lr_decay)
        if self.args.lr_decay_steps:
            return decay ** (max(0, step - self.args.warmup_steps) / self.args.lr_decay_steps)
        return decay ** max(0, epoch - 1)

    def cosine_factor(self, step: int) -> float:
        schedule = self.lr_schedule
        start, start_factor = schedule["start_step"], schedule["start_factor"]
        end = min(schedule["final_ratio"], start_factor)
        span = schedule["total_steps"] - 1 - start
        progress = min(1.0, max(0, step - start) / span) if span > 0 else 1.0
        return end + (start_factor - end) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def current_lr_factor(self, epoch: int) -> float:
        step = self.global_step
        factor = self.cosine_factor(step) if self.lr_schedule is not None else self.exponential_factor(epoch, step)
        return factor * self.warmup_ramp(step)

    def _compile_train_step(self):
        syn_cfg = self.syn_cfg
        mpd_cfg = self.mpd_cfg
        prec = self.prec
        opt_g = self.opt_g
        opt_d = self.opt_d
        data_cfg = self.config.data
        c_mel = float(self.config.train.c_mel)
        c_kl = float(self.config.train.c_kl)
        segment_frames = self.segment_frames
        hop_length = int(data_cfg.hop_length)
        segment_samples = int(self.config.train.segment_size)
        multiscale_mel = self.args.multiscale_mel_loss
        ms_mel_cfg = MultiScaleMelLossConfig(sample_rate=int(data_cfg.sample_rate))
        use_ema = self.args.ema_decay > 0

        @partial(jax.jit, donate_argnums=(0, 1, 2, 3, 4, 5))
        def _step(
            params_g,
            opt_state_g,
            params_d,
            opt_state_d,
            ema_params,
            loss_sums,
            lr_g,
            lr_d,
            ema_lerp,
            compute_grad_norms,
            rng_key,
            batch,
        ):
            phone, phone_lengths, pitch, pitchf, spec_lengths, wave, wave_lengths, sid = batch
            spec = lax.stop_gradient(
                spectrogram_jax_padded(
                    wave,
                    wave_lengths,
                    int(data_cfg.filter_length),
                    hop_length,
                    int(data_cfg.win_length),
                    max_frames=phone.shape[1],
                )
            )
            rng_key, step_rng = jax.random.split(rng_key)

            def _gen_diff(p_g):
                o, ids_slice, x_m, z_m, (_, z_p, m_p, logs_p, _, logs_q) = synthesizer_forward(
                    phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid,
                    p_g, syn_cfg, step_rng, deterministic=False, prec=prec,
                )
                return (o, z_p, m_p, logs_p, logs_q), (ids_slice, x_m, z_m)

            diff_out, gen_vjp, aux_out = jax.vjp(_gen_diff, params_g, has_aux=True)
            y_hat, _, _, _, _ = diff_out
            ids_slice, _, z_mask = aux_out

            if syn_cfg.randomized:
                wave_slice = slice_segments(wave, ids_slice * hop_length, segment_samples)
                spec_slice = slice_segments(spec, ids_slice, segment_frames)
            else:
                wave_slice = wave
                spec_slice = spec

            wave_slice_f32 = lax.stop_gradient(jnp.asarray(wave_slice, jnp.float32))
            if not multiscale_mel:
                wave_mel_ref = lax.stop_gradient(
                    spec_to_mel_jax(
                        spec_slice,
                        int(data_cfg.filter_length),
                        int(data_cfg.n_mel_channels),
                        int(data_cfg.sample_rate),
                        float(data_cfg.mel_fmin),
                        data_cfg.mel_fmax,
                    )
                )
            else:
                wave_mel_ref = None

            y_hat_det = lax.stop_gradient(y_hat)

            def _loss_d(p_d):
                yr, yg, _, _ = mpd_forward(p_d, mpd_cfg, wave_slice_f32, y_hat_det, prec=prec, transpose_fmap=False)
                l_d, _, _ = discriminator_loss(yr, yg)
                return l_d

            loss_disc = jnp.float32(0.0)
            grads_d = None
            opt_state_d = opt_state_d._replace(hyperparams={**opt_state_d.hyperparams, "learning_rate": lr_d})
            for _ in range(d_step_per_g_step):
                loss_disc, grads_d = jax.value_and_grad(_loss_d)(params_d)
                upd_d, opt_state_d = opt_d.update(grads_d, opt_state_d, params_d)
                params_d = optax.apply_updates(params_d, upd_d)

            params_d_resolved = lax.stop_gradient(resolve_mpd_params(params_d))
            _, fmap_r = lax.stop_gradient(
                mpd_forward_single(params_d_resolved, mpd_cfg, wave_slice_f32, prec=prec, transpose_fmap=False)
            )
            z_mask_sg = lax.stop_gradient(z_mask)

            def _loss_from_gen(d_out):
                y_h, z_p_v, m_p_v, logs_p_v, logs_q_v = d_out
                y_d_g, fmap_g = mpd_forward_single(
                    params_d_resolved, mpd_cfg, y_h, prec=prec, transpose_fmap=False
                )
                if multiscale_mel:
                    l_mel = multi_scale_mel_loss(wave_slice_f32, y_h, ms_mel_cfg) * c_mel / 3.0
                else:
                    y_h_mel = mel_spectrogram_jax(
                        y_h,
                        int(data_cfg.filter_length),
                        int(data_cfg.n_mel_channels),
                        int(data_cfg.sample_rate),
                        int(data_cfg.hop_length),
                        int(data_cfg.win_length),
                        float(data_cfg.mel_fmin),
                        data_cfg.mel_fmax,
                        center=False,
                    )
                    l_mel = jnp.mean(jnp.abs(wave_mel_ref - y_h_mel)) * c_mel
                l_kl = kl_loss(z_p_v, logs_q_v, m_p_v, logs_p_v, z_mask_sg) * c_kl
                l_fm = feature_loss(fmap_r, fmap_g)
                l_gen, _ = generator_loss(y_d_g)
                l_total = l_gen + l_fm + l_mel + l_kl
                return l_total, (l_gen, l_fm, l_mel, l_kl)

            ((loss_gen_all, (loss_gen, loss_fm, loss_mel, loss_kl)), grad_diff_out) = jax.value_and_grad(
                _loss_from_gen, has_aux=True
            )(diff_out)
            (grads_g,) = gen_vjp(grad_diff_out)

            opt_state_g = opt_state_g._replace(hyperparams={**opt_state_g.hyperparams, "learning_rate": lr_g})
            upd_g, opt_state_g = opt_g.update(grads_g, opt_state_g, params_g)
            params_g = optax.apply_updates(params_g, upd_g)

            if use_ema:
                gen_view = {k: v for k, v in params_g.items() if k != "enc_q"}
                ema_params = jax.tree.map(lambda e, p: e + ema_lerp * (p - e), ema_params, gen_view)

            gn_d, gn_g = lax.cond(
                compute_grad_norms,
                lambda _: (optax.global_norm(grads_d), optax.global_norm(grads_g)),
                lambda _: (jnp.float32(0.0), jnp.float32(0.0)),
                None,
            )

            loss_sums = loss_sums + jnp.stack([loss_gen_all, loss_disc, loss_gen, loss_fm, loss_mel, loss_kl])
            return (
                params_g,
                opt_state_g,
                params_d,
                opt_state_d,
                ema_params,
                loss_sums,
                gn_d,
                gn_g,
                rng_key,
                y_hat[0],
                ids_slice[0],
                spec[0],
            )

        @jax.jit
        def _mel_summary_arrays(spec_0, id_0, y_hat_0):
            mel_all = spec_to_mel_jax(
                spec_0[None, ...],
                int(data_cfg.filter_length),
                int(data_cfg.n_mel_channels),
                int(data_cfg.sample_rate),
                float(data_cfg.mel_fmin),
                data_cfg.mel_fmax,
            )
            y_mel = slice_segments(mel_all, id_0[None], segment_frames) if syn_cfg.randomized else mel_all
            y_hat_mel = mel_spectrogram_jax(
                y_hat_0[None, ...],
                int(data_cfg.filter_length),
                int(data_cfg.n_mel_channels),
                int(data_cfg.sample_rate),
                int(data_cfg.hop_length),
                int(data_cfg.win_length),
                float(data_cfg.mel_fmin),
                data_cfg.mel_fmax,
                center=False,
            )
            # Transpose (TT, n_mels) -> (n_mels, TT) to match plot_spectrogram_to_numpy axes.
            return y_mel[0].T, y_hat_mel[0].T, mel_all[0].T

        self._step_fn = _step
        self._mel_summary_fn = _mel_summary_arrays

    def _shard_batch(self, np_batch):
        if self.num_devices == 1:
            return tuple(jax.device_put(x, self.devices[0]) for x in np_batch)
        return tuple(jax.device_put(x, self.data_sharding) for x in np_batch)

    def _next_ema_lerp(self) -> float:
        if self.args.ema_decay <= 0:
            return 0.0
        self.ema_updates += 1
        beta = float(self.args.ema_decay)
        decay = beta * (1.0 - beta ** (self.ema_updates - 1)) / (1.0 - beta ** self.ema_updates)
        return float(1.0 - decay)

    def summary(self, epoch: int, start_batch: int, last_g: str | None) -> str:
        args, data = self.args, self.config.data
        dev_str = f"{self.platform.upper()} x{self.num_devices}"
        speakers = int(self.config.model.spk_embed_dim)
        parts = [
            args.model_name,
            f"{args.vocoder} {data.sample_rate} Hz {args.version} [JAX]",
            f"{speakers} speaker{'' if speakers == 1 else 's'}",
            f"{dev_str} {self.precision_label}",
            f"batch {args.batch_size}, {len(self.sampler)} steps per epoch",
            f"{args.total_steps} total steps (~{math.ceil(args.total_steps / len(self.sampler))} epochs)",
            f"saving every {args.save_every_steps} steps",
        ]
        options = []
        if args.max_batch_seconds:
            sizes = self.sampler.bucket_batch_sizes
            options.append(f"max {args.max_batch_seconds:g} s audio per batch ({min(sizes)}-{max(sizes)} clips)")
        if args.warmup_steps:
            options.append(f"{args.warmup_steps} warmup steps")
        if self.lr_schedule is not None:
            sc = self.lr_schedule
            end = min(sc["final_ratio"], sc["start_factor"])
            options.append(f"cosine LR {sc['start_factor']:.3g}x to {end:.3g}x from step {sc['start_step']}")
        elif args.lr_decay_steps:
            options.append(f"lr_decay every {args.lr_decay_steps} steps")
        if args.ema_decay > 0:
            options.append(f"generator EMA {args.ema_decay:g}")
        if args.cache_data_in_gpu:
            options.append("dataset cached in RAM")
        if args.checkpointing:
            options.append("checkpointing")
        if options:
            parts.append(", ".join(options))

        pretrained = [os.path.basename(p) for p in (args.pretrained_g, args.pretrained_d) if p not in ("", "None")]
        if last_g:
            parts.append(
                f"resuming at step {self.global_step}, epoch {epoch} batch {start_batch + 1} of {len(self.sampler)}"
                + (" with a fresh optimizer state" if args.reset_optimizer else "")
            )
        elif pretrained:
            parts.append(f"from {' and '.join(pretrained)}")
        else:
            parts.append("from scratch")
        return " | ".join(parts)

    def update_postfix(self) -> None:
        if self.progress is None:
            return
        text = f"epoch {self.epoch}"
        if self.logged_losses is not None:
            total, disc, mel = self.logged_losses
            text += f", g {total:.2f}, d {disc:.2f}, mel {mel:.2f}"
        self.progress.set_postfix_str(text, refresh=False)

    def _record_and_maybe_log(self, gn_d, gn_g, current_lr_g: float) -> None:
        self.batch_count += 1
        if self.global_step % self.log_interval != 0:
            return
        sums_np = np.asarray(jax.device_get(self.loss_sums), dtype=np.float32) / float(max(1, self.batch_count))
        self.loss_sums = jax.device_put(jnp.zeros((6,), dtype=jnp.float32), self.replicated_sharding)
        self.batch_count = 0
        self.logged_losses = (float(sums_np[0]), float(sums_np[1]), float(sums_np[4]))
        self.update_postfix()
        if self.writer is not None:
            summarize(
                writer=self.writer,
                global_step=self.global_step,
                scalars={
                    "loss/g/total": float(sums_np[0]),
                    "loss/d/adv": float(sums_np[1]),
                    "learning_rate": float(current_lr_g),
                    "loss/g/adv": float(sums_np[2]),
                    "loss/g/fm": float(sums_np[3]),
                    "loss/g/mel": float(sums_np[4]),
                    "loss/g/kl": float(sums_np[5]),
                    "grad/norm_d": float(jax.device_get(gn_d)),
                    "grad/norm_g": float(jax.device_get(gn_g)),
                },
            )

    def save(self, epoch: int, final: bool) -> None:
        args = self.args
        if self.writer is not None and self.last_batch is not None:
            spec_0, id_0, y_hat_0 = self.last_batch
            y_m, yh_m, m_all = self._mel_summary_fn(spec_0, id_0, y_hat_0)
            y_m_np, yh_m_np, m_all_np = jax.device_get((y_m, yh_m, m_all))
            summarize(
                writer=self.writer,
                global_step=self.global_step,
                images={
                    "slice/mel_org": plot_spectrogram_to_numpy(y_m_np),
                    "slice/mel_gen": plot_spectrogram_to_numpy(yh_m_np),
                    "all/mel": plot_spectrogram_to_numpy(m_all_np),
                },
            )

        weights_path = os.path.join(args.experiment_dir, f"{args.model_name}_{epoch}e_{self.global_step}s.pth")
        export_src = self.ema_params if args.ema_decay > 0 else self.params_g
        if (args.save_every_weights or final) and not os.path.exists(weights_path):
            assert self.exporter is not None
            self.exporter.submit(
                export_src,
                self.syn_cfg,
                sr=int(self.config.data.sample_rate),
                name=args.model_name,
                model_path=weights_path,
                epoch=epoch,
                step=self.global_step,
                hps=self.config,
                vocoder=args.vocoder,
                pitch_guidance=True,
                version=args.version,
                export_dtype_str=self.export_dtype_str,
                verbose=False,
            )

        suffix = LATEST_CHECKPOINT_SUFFIX if args.save_only_latest else self.global_step
        rng_words = [int(w) for w in np.asarray(jax.device_get(self.rng_key), dtype=np.uint32).reshape(-1).tolist()]
        gen_extra = {
            "step": self.global_step,
            "epoch_step": self.epoch_step,
            "ema_updates": self.ema_updates,
            "rng_key": rng_words,
            "vocoder": args.vocoder,
        }
        if self.lr_schedule is not None:
            gen_extra["lr_schedule"] = self.lr_schedule

        d_path = os.path.join(args.experiment_dir, f"D_{suffix}.npz")
        g_path = os.path.join(args.experiment_dir, f"G_{suffix}.npz")
        save_jax_checkpoint(
            d_path,
            self.params_d,
            self.opt_state_d,
            self.config.train.learning_rate,
            epoch,
            extra={"step": self.global_step},
        )
        save_jax_checkpoint(
            g_path,
            self.params_g,
            self.opt_state_g,
            self.config.train.learning_rate,
            epoch,
            extra=gen_extra,
            ema_params=self.ema_params if args.ema_decay > 0 else None,
        )
        if final and self.exporter is not None:
            self.exporter.wait_for_completion()

    def train_epoch(self, epoch: int, start_batch: int) -> None:
        self.sampler.set_epoch(epoch, start_batch)
        self.epoch = epoch
        self.epoch_step = start_batch
        self.update_postfix()
        base_lr = float(self.config.train.learning_rate)

        batch_iter = iter(self.loader)
        try:
            cur_np = next(batch_iter, None)
            if cur_np is None:
                return
            cur_dev = self._shard_batch(cur_np)

            while cur_dev is not None:
                if self.exporter is not None:
                    self.exporter.raise_if_failed()
                factor = self.current_lr_factor(epoch)
                lr_g_val = base_lr * g_lr_coeff * factor
                lr_d_val = base_lr * d_lr_coeff * factor
                ema_lerp = self._next_ema_lerp()
                log_norms = (self.global_step + 1) % self.log_interval == 0

                (
                    self.params_g,
                    self.opt_state_g,
                    self.params_d,
                    self.opt_state_d,
                    self.ema_params,
                    self.loss_sums,
                    gn_d,
                    gn_g,
                    self.rng_key,
                    y_hat_0,
                    id_0,
                    spec_0,
                ) = self._step_fn(
                    self.params_g,
                    self.opt_state_g,
                    self.params_d,
                    self.opt_state_d,
                    self.ema_params,
                    self.loss_sums,
                    np.float32(lr_g_val),
                    np.float32(lr_d_val),
                    np.float32(ema_lerp),
                    np.bool_(log_norms),
                    self.rng_key,
                    cur_dev,
                )
                self.last_batch = (spec_0, id_0, y_hat_0)
                self.global_step += 1
                self.epoch_step += 1
                self._record_and_maybe_log(gn_d, gn_g, lr_g_val)
                self.progress.update(1)
                final = self.global_step >= self.args.total_steps
                if final or self.global_step % self.args.save_every_steps == 0:
                    self.save(epoch, final)
                if final:
                    break

                # Stage next batch on device while current _step_fn executes asynchronously on TPU.
                next_np = next(batch_iter, None)
                cur_dev = self._shard_batch(next_np) if next_np is not None else None
        finally:
            if hasattr(batch_iter, "close"):
                batch_iter.close()

    def fit(self) -> None:
        self.build_data()
        last_g, last_d = latest_checkpoint_pair(self.args.experiment_dir)
        self.build_models(last_g)
        epoch, start_batch, extra = self.restore(last_g, last_d)
        self._put_replicated()
        if self.args.lr_schedule == "cosine":
            self.build_lr_schedule(extra.get("lr_schedule"))
        self._compile_train_step()

        self.writer = _create_summary_writer(
            os.path.join(self.args.experiment_dir, "eval"),
            purge_step=self.global_step + 1 if self.global_step else 0,
        )
        self.exporter = AsyncJaxInferenceExporter()
        print(self.summary(epoch, start_batch, last_g))

        total_steps = self.args.total_steps
        finished = self.global_step >= total_steps
        if finished:
            print(f"{self.args.model_name} already reached step {total_steps}. Raise --total-steps to continue training.")

        self.progress = tqdm(
            total=max(total_steps, self.global_step),
            initial=self.global_step,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            disable=finished,
        )
        try:
            while self.global_step < total_steps:
                self.train_epoch(epoch, start_batch)
                epoch, start_batch = epoch + 1, 0
        finally:
            self.progress.close()
            self.loader.close()
            if self.exporter is not None:
                self.exporter.close()
            if self.writer is not None:
                self.writer.close()

        if not finished:
            print(f"{self.args.model_name} | training completed at epoch {self.epoch}, step {self.global_step}")


def main(argv: list[str] | None = None) -> None:
    args = parse_arguments(argv)
    config = load_experiment_config(args)
    check_dataset_sample_rate(args, config)
    if args.cleanup:
        remove_previous_attempt(args.experiment_dir)
    trainer = JaxTrainer(args, config)
    trainer.fit()


if __name__ == "__main__":
    main()
