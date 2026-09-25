"""JAX lowering for the bounded ordered attack transition."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from azelficoast.research.mechanics.gen9_ordered_attack import (
    ATTACKER_PRIORITY_COLUMN,
    ATTACKER_SPEED_COLUMN,
    DEFENDER_SPD_STAGE_COLUMN,
    OPPONENT_PRIORITY_COLUMN,
    OPPONENT_SPEED_COLUMN,
    PACK_HP_BASE,
    PACK_ORDER_BASE,
    PACK_PP_BASE,
    PACK_STAGE_BASE,
    SECONDARY_CHANCE_COLUMN,
)
from azelficoast.research.mechanics.jax_gen9_attack import attack_transition_batch


@jax.jit
def ordered_attack_transition_batch(
    params: jax.Array,
    order_tie_rolls: jax.Array,
    accuracy_rolls: jax.Array,
    damage_rolls: jax.Array,
    secondary_rolls: jax.Array,
) -> jax.Array:
    packed_attack = attack_transition_batch(params[:, :23], accuracy_rolls, damage_rolls)

    attack_flags = packed_attack // jnp.int32(134217728)
    remainder = packed_attack - attack_flags * jnp.int32(134217728)
    move_pp = remainder // jnp.int32(1048576)
    remainder = remainder - move_pp * jnp.int32(1048576)
    attacker_hp = remainder // jnp.int32(1024)
    defender_hp = remainder - attacker_hp * jnp.int32(1024)
    hit = (attack_flags - (attack_flags // 2) * 2) != 0

    stage = params[:, DEFENDER_SPD_STAGE_COLUMN]
    secondary_applies = (
        hit
        & (defender_hp > 0)
        & (params[:, SECONDARY_CHANCE_COLUMN] > 0)
        & (secondary_rolls < params[:, SECONDARY_CHANCE_COLUMN])
        & (stage > -6)
    )
    stage_after = jnp.where(secondary_applies, stage - 1, stage)

    attacker_priority = params[:, ATTACKER_PRIORITY_COLUMN]
    opponent_priority = params[:, OPPONENT_PRIORITY_COLUMN]
    attacker_speed = params[:, ATTACKER_SPEED_COLUMN]
    opponent_speed = params[:, OPPONENT_SPEED_COLUMN]

    priority_greater = attacker_priority > opponent_priority
    priority_less = attacker_priority < opponent_priority
    speed_greater = attacker_speed > opponent_speed
    speed_less = attacker_speed < opponent_speed
    tie_first = order_tie_rolls == 0

    acted_first = jnp.where(
        priority_greater,
        True,
        jnp.where(
            priority_less,
            False,
            jnp.where(
                speed_greater,
                True,
                jnp.where(speed_less, False, tie_first),
            ),
        ),
    )

    return (
        defender_hp
        + attacker_hp * jnp.int32(PACK_HP_BASE)
        + move_pp * jnp.int32(PACK_PP_BASE)
        + (stage_after + 6) * jnp.int32(PACK_STAGE_BASE)
        + acted_first.astype(jnp.int32) * jnp.int32(PACK_ORDER_BASE)
    )


@jax.jit
def ordered_attack_score(
    params: jax.Array,
    order_tie_rolls: jax.Array,
    accuracy_rolls: jax.Array,
    damage_rolls: jax.Array,
    secondary_rolls: jax.Array,
) -> jax.Array:
    return jnp.sum(
        ordered_attack_transition_batch(
            params,
            order_tie_rolls,
            accuracy_rolls,
            damage_rolls,
            secondary_rolls,
        ),
        dtype=jnp.int32,
    )


@jax.jit
def weighted_ordered_attack_score(
    params: jax.Array,
    order_tie_rolls: jax.Array,
    accuracy_rolls: jax.Array,
    damage_rolls: jax.Array,
    secondary_rolls: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    return jnp.sum(
        ordered_attack_transition_batch(
            params,
            order_tie_rolls,
            accuracy_rolls,
            damage_rolls,
            secondary_rolls,
        )
        * weights,
        dtype=jnp.int32,
    )
