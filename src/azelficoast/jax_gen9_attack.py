"""JAX lowering of the bounded whole-attack transition."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from azelficoast.gen9_attack import (
    ATTACK_ACCURACY_COLUMN,
    ATTACKER_HP_COLUMN,
    ATTACKER_MAX_HP_COLUMN,
    DEFENDER_HP_COLUMN,
    MOVE_PP_COLUMN,
    PACK_FLAGS_BASE,
    PACK_HP_BASE,
    PACK_PP_BASE,
)
from azelficoast.jax_gen9_damage import FINAL_MOD, damage_batch


@jax.jit
def attack_transition_batch(
    params: jax.Array,
    accuracy_rolls: jax.Array,
    damage_rolls: jax.Array,
) -> jax.Array:
    hit = accuracy_rolls < params[:, ATTACK_ACCURACY_COLUMN]
    damage = damage_batch(params[:, :18], damage_rolls)

    defender_before = params[:, DEFENDER_HP_COLUMN]
    attacker_before = params[:, ATTACKER_HP_COLUMN]
    attacker_max = params[:, ATTACKER_MAX_HP_COLUMN]
    pp_after = jnp.maximum(jnp.int32(0), params[:, MOVE_PP_COLUMN] - 1)

    defender_after = jnp.where(
        hit,
        jnp.maximum(jnp.int32(0), defender_before - damage),
        defender_before,
    )

    recoil = jnp.maximum(jnp.int32(1), attacker_max // 10)
    life_orb = params[:, FINAL_MOD] == jnp.int32(5324)
    attacker_after = jnp.where(
        hit & life_orb,
        jnp.maximum(jnp.int32(0), attacker_before - recoil),
        attacker_before,
    )

    flags = hit.astype(jnp.int32)
    flags += (defender_after == 0).astype(jnp.int32) * 2
    flags += (attacker_after == 0).astype(jnp.int32) * 4

    return (
        defender_after
        + attacker_after * jnp.int32(PACK_HP_BASE)
        + pp_after * jnp.int32(PACK_PP_BASE)
        + flags * jnp.int32(PACK_FLAGS_BASE)
    )


@jax.jit
def attack_transition_score(
    params: jax.Array,
    accuracy_rolls: jax.Array,
    damage_rolls: jax.Array,
) -> jax.Array:
    return jnp.sum(
        attack_transition_batch(params, accuracy_rolls, damage_rolls),
        dtype=jnp.int32,
    )


@jax.jit
def weighted_attack_transition_score(
    params: jax.Array,
    accuracy_rolls: jax.Array,
    damage_rolls: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    return jnp.sum(
        attack_transition_batch(params, accuracy_rolls, damage_rolls) * weights,
        dtype=jnp.int32,
    )
