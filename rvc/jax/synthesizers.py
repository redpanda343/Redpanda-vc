"""JAX NWC Synthesizer combining TextEncoder, PosteriorEncoder, Flow, and Vocoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rvc.jax import PrecisionConfig
from rvc.jax.encoders import (
    PosteriorEncoderConfig,
    TextEncoderConfig,
    init_posterior_encoder_params,
    init_text_encoder_params,
    posterior_encoder_forward,
    posterior_encoder_from_torch_state,
    posterior_encoder_to_torch_state,
    text_encoder_forward,
    text_encoder_from_torch_state,
    text_encoder_to_torch_state,
)
from rvc.jax.generators import (
    HiFiGANConfig,
    HiFiGANMRFConfig,
    HiFiGANNSFConfig,
    RefineGANConfig,
    hifigan_forward,
    hifigan_from_torch_state,
    hifigan_mrf_forward,
    hifigan_mrf_from_torch_state,
    hifigan_mrf_to_torch_state,
    hifigan_nsf_forward,
    hifigan_nsf_from_torch_state,
    hifigan_nsf_to_torch_state,
    hifigan_to_torch_state,
    init_hifigan_mrf_params,
    init_hifigan_nsf_params,
    init_hifigan_params,
    init_refinegan_params,
    refinegan_forward,
    refinegan_from_torch_state,
    refinegan_to_torch_state,
)
from rvc.jax.ops import (
    Array,
    embedding,
    init_embedding_params,
    rand_slice_segments,
    slice_segments,
)
from rvc.jax.residuals import (
    ResidualCouplingBlockConfig,
    flow_forward,
    flow_from_torch_state,
    flow_to_torch_state,
    init_flow_params,
)


@dataclass(frozen=True)
class SynthesizerConfig:
    spec_channels: int
    segment_size: int
    inter_channels: int
    hidden_channels: int
    filter_channels: int
    n_heads: int
    n_layers: int
    kernel_size: int
    p_dropout: float
    resblock: str
    resblock_kernel_sizes: tuple[int, ...]
    resblock_dilation_sizes: tuple[tuple[int, ...], ...]
    upsample_rates: tuple[int, ...]
    upsample_initial_channel: int
    upsample_kernel_sizes: tuple[int, ...]
    spk_embed_dim: int
    gin_channels: int
    sr: int
    use_f0: bool = True
    text_enc_hidden_dim: int = 768
    vocoder: str = "HiFi-GAN"
    randomized: bool = True
    checkpointing: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "resblock_kernel_sizes", tuple(int(x) for x in self.resblock_kernel_sizes))
        object.__setattr__(
            self,
            "resblock_dilation_sizes",
            tuple(tuple(int(d) for d in row) for row in self.resblock_dilation_sizes),
        )
        object.__setattr__(self, "upsample_rates", tuple(int(x) for x in self.upsample_rates))
        object.__setattr__(self, "upsample_kernel_sizes", tuple(int(x) for x in self.upsample_kernel_sizes))

    @property
    def text_encoder_config(self) -> TextEncoderConfig:
        return TextEncoderConfig(
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            filter_channels=self.filter_channels,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            kernel_size=self.kernel_size,
            p_dropout=self.p_dropout,
            embedding_dim=self.text_enc_hidden_dim,
            f0=self.use_f0,
            checkpointing=self.checkpointing,
        )

    @property
    def posterior_encoder_config(self) -> PosteriorEncoderConfig:
        return PosteriorEncoderConfig(
            in_channels=self.spec_channels,
            out_channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=16,
            gin_channels=self.gin_channels,
            checkpointing=self.checkpointing,
        )

    @property
    def flow_config(self) -> ResidualCouplingBlockConfig:
        return ResidualCouplingBlockConfig(
            channels=self.inter_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=5,
            dilation_rate=1,
            n_layers=3,
            n_flows=4,
            gin_channels=self.gin_channels,
            mean_only=True,
            checkpointing=self.checkpointing,
        )

    @property
    def vocoder_config(self) -> HiFiGANConfig | HiFiGANNSFConfig | HiFiGANMRFConfig | RefineGANConfig:
        if not self.use_f0:
            if self.vocoder in ("MRF HiFi-GAN", "RefineGAN"):
                raise ValueError(f"{self.vocoder} does not support training without pitch guidance.")
            return HiFiGANConfig(
                initial_channel=self.inter_channels,
                resblock_kernel_sizes=self.resblock_kernel_sizes,
                resblock_dilation_sizes=self.resblock_dilation_sizes,
                upsample_rates=self.upsample_rates,
                upsample_initial_channel=self.upsample_initial_channel,
                upsample_kernel_sizes=self.upsample_kernel_sizes,
                gin_channels=self.gin_channels,
                checkpointing=self.checkpointing,
            )
        if self.vocoder == "MRF HiFi-GAN":
            return HiFiGANMRFConfig(
                in_channel=self.inter_channels,
                upsample_initial_channel=self.upsample_initial_channel,
                upsample_rates=self.upsample_rates,
                upsample_kernel_sizes=self.upsample_kernel_sizes,
                resblock_kernel_sizes=self.resblock_kernel_sizes,
                resblock_dilations=self.resblock_dilation_sizes,
                gin_channels=self.gin_channels,
                sample_rate=self.sr,
                harmonic_num=8,
                checkpointing=self.checkpointing,
            )
        if self.vocoder == "RefineGAN":
            return RefineGANConfig(
                sample_rate=self.sr,
                upsample_rates=self.upsample_rates,
                num_mels=self.inter_channels,
                start_channels=16,
                gin_channels=self.gin_channels,
                checkpointing=self.checkpointing,
            )
        return HiFiGANNSFConfig(
            initial_channel=self.inter_channels,
            resblock_kernel_sizes=self.resblock_kernel_sizes,
            resblock_dilation_sizes=self.resblock_dilation_sizes,
            upsample_rates=self.upsample_rates,
            upsample_initial_channel=self.upsample_initial_channel,
            upsample_kernel_sizes=self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
            sr=self.sr,
            checkpointing=self.checkpointing,
        )


def init_synthesizer_params(rng: np.random.Generator, cfg: SynthesizerConfig) -> dict[str, Any]:
    """Initialize Synthesizer parameters in float32 PyTree matching PyTorch structure."""
    dec_cfg = cfg.vocoder_config
    if isinstance(dec_cfg, HiFiGANMRFConfig):
        dec_params = init_hifigan_mrf_params(rng, dec_cfg)
    elif isinstance(dec_cfg, RefineGANConfig):
        dec_params = init_refinegan_params(rng, dec_cfg)
    elif isinstance(dec_cfg, HiFiGANNSFConfig):
        dec_params = init_hifigan_nsf_params(rng, dec_cfg)
    else:
        dec_params = init_hifigan_params(rng, dec_cfg)

    return {
        "enc_p": init_text_encoder_params(rng, cfg.text_encoder_config),
        "enc_q": init_posterior_encoder_params(rng, cfg.posterior_encoder_config),
        "flow": init_flow_params(rng, cfg.flow_config),
        "dec": dec_params,
        "emb_g": init_embedding_params(rng, cfg.spk_embed_dim, cfg.gin_channels),
    }


def _decode_vocoder(
    z_in: Array,
    pitchf_in: Array | None,
    g: Array,
    dec_params: Mapping[str, Any],
    cfg: SynthesizerConfig,
    rng_key: Array | None,
    deterministic: bool,
    prec: PrecisionConfig,
) -> Array:
    dec_cfg = cfg.vocoder_config
    if isinstance(dec_cfg, HiFiGANMRFConfig):
        return hifigan_mrf_forward(z_in, pitchf_in, dec_params, dec_cfg, rng_key=rng_key, g=g, deterministic=deterministic, prec=prec)
    if isinstance(dec_cfg, RefineGANConfig):
        return refinegan_forward(z_in, pitchf_in, dec_params, dec_cfg, rng_key=rng_key, g=g, deterministic=deterministic, prec=prec)
    if isinstance(dec_cfg, HiFiGANNSFConfig):
        return hifigan_nsf_forward(z_in, pitchf_in, dec_params, dec_cfg, rng_key=rng_key, g=g, deterministic=deterministic, prec=prec)
    return hifigan_forward(z_in, dec_params, dec_cfg, g=g, prec=prec)


def synthesizer_forward(
    phone: Array,
    phone_lengths: Array,
    pitch: Array | None,
    pitchf: Array | None,
    y: Array,
    y_lengths: Array,
    ds: Array,
    params: Mapping[str, Any],
    cfg: SynthesizerConfig,
    rng_key: Array,
    deterministic: bool = False,
    prec: PrecisionConfig = PrecisionConfig(),
) -> tuple[Array, Array | None, Array, Array, tuple[Array, Array, Array, Array, Array, Array]]:
    """Training forward pass for RVC Synthesizer using NWC layout (B, T, C)."""
    k_p, k_q, k_slice, k_dec = jax.random.split(rng_key, 4)

    g = embedding(ds, params["emb_g"]["weight"])[:, None, :]  # (B, 1, gin_channels)
    m_p, logs_p, x_mask = text_encoder_forward(
        phone,
        pitch,
        phone_lengths,
        params["enc_p"],
        cfg.text_encoder_config,
        rng_key=k_p,
        deterministic=deterministic,
        prec=prec,
        expected_frames=int(y.shape[1]),
    )
    z, m_q, logs_q, y_mask = posterior_encoder_forward(
        y,
        y_lengths,
        params["enc_q"],
        cfg.posterior_encoder_config,
        rng_key=k_q,
        g=g,
        prec=prec,
    )
    z_p = flow_forward(z, y_mask, params["flow"], cfg.flow_config, g=g, reverse=False, prec=prec)

    if cfg.randomized:
        z_slice, ids_slice = rand_slice_segments(z, y_lengths, cfg.segment_size, k_slice)
        pitchf_in = slice_segments(pitchf, ids_slice, cfg.segment_size) if cfg.use_f0 and pitchf is not None else None
        o = _decode_vocoder(z_slice, pitchf_in, g, params["dec"], cfg, k_dec, deterministic, prec)
        return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    o = _decode_vocoder(z, pitchf if cfg.use_f0 else None, g, params["dec"], cfg, k_dec, deterministic, prec)
    return o, None, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)


def synthesizer_infer(
    phone: Array,
    phone_lengths: Array,
    pitch: Array | None,
    nsff0: Array | None,
    sid: Array,
    params: Mapping[str, Any],
    cfg: SynthesizerConfig,
    rng_key: Array,
    skip_head: int | None = None,
    return_length: int | None = None,
    prec: PrecisionConfig = PrecisionConfig(),
) -> tuple[Array, Array, tuple[Array, Array, Array, Array]]:
    """Deterministic/sampled inference pass matching rvc.algorithm.synthesizers.Synthesizer.infer."""
    k_noise, k_dec = jax.random.split(rng_key, 2)
    g = embedding(sid, params["emb_g"]["weight"])[:, None, :]
    flow_head = 0
    decoder_head = 0
    if skip_head is not None and return_length is not None:
        flow_head = max(skip_head - 24, 0)
        decoder_head = skip_head - flow_head

    m_p, logs_p, x_mask = text_encoder_forward(
        phone,
        pitch,
        phone_lengths,
        params["enc_p"],
        cfg.text_encoder_config,
        skip_head=flow_head if skip_head is not None else None,
        deterministic=True,
        prec=prec,
    )
    eps = jax.random.normal(k_noise, shape=m_p.shape, dtype=jnp.float32)
    z_p = (m_p + jnp.exp(logs_p) * eps * 0.66666) * x_mask
    z = flow_forward(z_p, x_mask, params["flow"], cfg.flow_config, g=g, reverse=True, prec=prec)

    if skip_head is not None and return_length is not None:
        z = z[:, decoder_head : decoder_head + return_length, :]
        x_mask = x_mask[:, decoder_head : decoder_head + return_length, :]
        if nsff0 is not None:
            nsff0 = nsff0[:, skip_head : skip_head + return_length]

    o = _decode_vocoder(z * x_mask, nsff0 if cfg.use_f0 else None, g, params["dec"], cfg, k_dec, True, prec)
    return o, x_mask, (z, z_p, m_p, logs_p)


def synthesizer_from_torch_state(state: Mapping[str, Any], cfg: SynthesizerConfig) -> dict[str, Any]:
    """Convert PyTorch Synthesizer state_dict into JAX Synthesizer param PyTree."""
    dec_cfg = cfg.vocoder_config
    if isinstance(dec_cfg, HiFiGANMRFConfig):
        dec_params = hifigan_mrf_from_torch_state(state, "dec", dec_cfg)
    elif isinstance(dec_cfg, RefineGANConfig):
        dec_params = refinegan_from_torch_state(state, "dec", dec_cfg)
    elif isinstance(dec_cfg, HiFiGANNSFConfig):
        dec_params = hifigan_nsf_from_torch_state(state, "dec", dec_cfg)
    else:
        dec_params = hifigan_from_torch_state(state, "dec", dec_cfg)

    return {
        "enc_p": text_encoder_from_torch_state(state, "enc_p", cfg.text_encoder_config),
        "enc_q": posterior_encoder_from_torch_state(state, "enc_q", cfg.posterior_encoder_config),
        "flow": flow_from_torch_state(state, "flow", cfg.flow_config),
        "dec": dec_params,
        "emb_g": {"weight": np.asarray(state["emb_g.weight"], dtype=np.float32).copy()},
    }


def synthesizer_to_torch_state(
    params: Mapping[str, Any],
    cfg: SynthesizerConfig,
    *,
    remove_weight_norm: bool = False,
    inference_only: bool = False,
) -> dict[str, np.ndarray]:
    """Convert JAX Synthesizer param PyTree into PyTorch state_dict (training or extract_model inference)."""
    state: dict[str, np.ndarray] = {}
    text_encoder_to_torch_state(state, "enc_p", params["enc_p"])
    if not inference_only:
        posterior_encoder_to_torch_state(state, "enc_q", params["enc_q"], remove_weight_norm=remove_weight_norm)
    flow_to_torch_state(state, "flow", params["flow"], remove_weight_norm=remove_weight_norm)

    dec_cfg = cfg.vocoder_config
    if isinstance(dec_cfg, HiFiGANMRFConfig):
        hifigan_mrf_to_torch_state(state, "dec", params["dec"], remove_weight_norm=remove_weight_norm)
    elif isinstance(dec_cfg, RefineGANConfig):
        refinegan_to_torch_state(state, "dec", params["dec"], remove_weight_norm=remove_weight_norm)
    elif isinstance(dec_cfg, HiFiGANNSFConfig):
        hifigan_nsf_to_torch_state(state, "dec", params["dec"], remove_weight_norm=remove_weight_norm)
    else:
        hifigan_to_torch_state(state, "dec", params["dec"], remove_weight_norm=remove_weight_norm)

    state["emb_g.weight"] = np.asarray(params["emb_g"]["weight"], dtype=np.float32).copy()
    return state


def synthesizer_config_from_hparams(
    hps: Any,
    *,
    vocoder: str = "HiFi-GAN",
    use_f0: bool = True,
    checkpointing: bool = False,
    randomized: bool = True,
) -> SynthesizerConfig:
    """Build SynthesizerConfig cleanly from HParams JSON loaded from experiment config.json."""
    model = hps.model
    data = hps.data
    segment_frames = int(hps.train.segment_size) // int(data.hop_length)
    return SynthesizerConfig(
        spec_channels=int(data.filter_length) // 2 + 1,
        segment_size=segment_frames,
        inter_channels=int(model.inter_channels),
        hidden_channels=int(model.hidden_channels),
        filter_channels=int(model.filter_channels),
        n_heads=int(model.n_heads),
        n_layers=int(model.n_layers),
        kernel_size=int(model.kernel_size),
        p_dropout=float(model.p_dropout),
        resblock=str(model.resblock),
        resblock_kernel_sizes=tuple(int(x) for x in model.resblock_kernel_sizes),
        resblock_dilation_sizes=tuple(tuple(int(d) for d in r) for r in model.resblock_dilation_sizes),
        upsample_rates=tuple(int(x) for x in model.upsample_rates),
        upsample_initial_channel=int(model.upsample_initial_channel),
        upsample_kernel_sizes=tuple(int(x) for x in model.upsample_kernel_sizes),
        spk_embed_dim=int(model.spk_embed_dim),
        gin_channels=int(model.gin_channels),
        sr=int(data.sample_rate),
        use_f0=bool(use_f0),
        text_enc_hidden_dim=int(getattr(model, "text_enc_hidden_dim", 768)),
        vocoder=str(vocoder),
        randomized=bool(randomized),
        checkpointing=bool(checkpointing),
    )

