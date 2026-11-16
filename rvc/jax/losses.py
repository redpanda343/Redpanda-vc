"""Strictly FP32 GAN, feature-matching, and KL losses matching rvc.train.losses."""

from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

from rvc.jax.ops import Array


def feature_loss(
    fmap_r: Sequence[Sequence[Array]],
    fmap_g: Sequence[Sequence[Array]],
) -> Array:
    """L1 feature-matching loss in strict FP32."""
    total = jnp.array(0.0, dtype=jnp.float32)
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            total = total + jnp.mean(jnp.abs(jnp.asarray(rl, jnp.float32) - jnp.asarray(gl, jnp.float32)))
    return 2.0 * total


def discriminator_loss(
    disc_real_outputs: Sequence[Array],
    disc_generated_outputs: Sequence[Array],
) -> tuple[Array, list[Array], list[Array]]:
    """LSGAN discriminator loss in strict FP32."""
    loss = jnp.array(0.0, dtype=jnp.float32)
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = jnp.mean(jnp.square(1.0 - jnp.asarray(dr, jnp.float32)))
        g_loss = jnp.mean(jnp.square(jnp.asarray(dg, jnp.float32)))
        loss = loss + r_loss + g_loss
    return loss, [], []


def generator_loss(disc_outputs: Sequence[Array]) -> tuple[Array, list[Array]]:
    """LSGAN generator adversarial loss in strict FP32."""
    loss = jnp.array(0.0, dtype=jnp.float32)
    for dg in disc_outputs:
        loss = loss + jnp.mean(jnp.square(1.0 - jnp.asarray(dg, jnp.float32)))
    return loss, []


def discriminator_loss_scaled(
    disc_real: Sequence[Array],
    disc_fake: Sequence[Array],
    scale: float = 1.0,
) -> tuple[Array, None, None]:
    """LSGAN discriminator loss scaling losses from midpoint onward by `scale`."""
    midpoint = len(disc_real) // 2
    loss = jnp.array(0.0, dtype=jnp.float32)
    for i, (d_real, d_fake) in enumerate(zip(disc_real, disc_fake)):
        r_loss = jnp.mean(jnp.square(1.0 - jnp.asarray(d_real, jnp.float32)))
        f_loss = jnp.mean(jnp.square(jnp.asarray(d_fake, jnp.float32)))
        cur = r_loss + f_loss
        if i >= midpoint:
            cur = cur * float(scale)
        loss = loss + cur
    return loss, None, None


def generator_loss_scaled(
    disc_outputs: Sequence[Array],
    scale: float = 1.0,
) -> tuple[Array, None, None]:
    """LSGAN generator loss scaling losses from midpoint onward by `scale`."""
    midpoint = len(disc_outputs) // 2
    loss = jnp.array(0.0, dtype=jnp.float32)
    for i, d_fake in enumerate(disc_outputs):
        cur = jnp.mean(jnp.square(1.0 - jnp.asarray(d_fake, jnp.float32)))
        if i >= midpoint:
            cur = cur * float(scale)
        loss = loss + cur
    return loss, None, None


def kl_loss(
    z_p: Array,
    logs_q: Array,
    m_p: Array,
    logs_p: Array,
    z_mask: Array,
) -> Array:
    """Strictly FP32 KL divergence loss matching rvc.train.losses.kl_loss."""
    z_p_f = jnp.asarray(z_p, jnp.float32)
    logs_q_f = jnp.asarray(logs_q, jnp.float32)
    m_p_f = jnp.asarray(m_p, jnp.float32)
    logs_p_f = jnp.asarray(logs_p, jnp.float32)
    mask_f = jnp.asarray(z_mask, jnp.float32)
    kl = logs_p_f - logs_q_f - 0.5 + 0.5 * jnp.square(z_p_f - m_p_f) * jnp.exp(-2.0 * logs_p_f)
    return jnp.sum(kl * mask_f) / jnp.maximum(jnp.sum(mask_f), 1.0)
