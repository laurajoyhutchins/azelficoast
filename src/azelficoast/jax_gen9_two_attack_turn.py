"""JAX lowering for the bounded two-action turn."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from azelficoast.gen9_two_attack_turn import (
    PACK_FLAGS_BASE,
    PACK_HP_BASE,
    PACK_P1_PP_BASE,
    PACK_P2_PP_BASE,
    PACK_STAGE_BASE,
    P1_ACTION_KIND_COLUMN,
    P1_ACTION_PROTECT,
    P1_PRIORITY_COLUMN,
    P1_SPA_DROP_CHANCE_COLUMN,
    P1_SPEED_COLUMN,
    P2_ATTACK_OFFSET,
    P2_PRIORITY_COLUMN,
    P2_SPA_STAGE_COLUMN,
    P2_SPEED_COLUMN,
)

ATTACKER_LEVEL = 0
DEFENDER_LEVEL = 1
BASE_POWER = 2
ATTACK_BASE = 3
ATTACK_IV = 4
ATTACK_EV = 5
ATTACK_NATURE = 6
DEFENSE_BASE = 7
DEFENSE_IV = 8
DEFENSE_EV = 9
DEFENSE_NATURE = 10
DEFENSE_MOD = 11
ATTACK_MOD = 12
STAB_MOD = 13
TYPE_MOD = 14
BURN_MOD = 15
FINAL_MOD = 16
ACCURACY = 18
ATTACKER_HP = 19
ATTACKER_MAX_HP = 20
DEFENDER_HP = 21
MOVE_PP = 22


def _modify(value: jax.Array, modifier: jax.Array) -> jax.Array:
    return (value * modifier + jnp.int32(2047)) // jnp.int32(4096)


def _ordinary_stat(
    base: jax.Array,
    iv: jax.Array,
    ev: jax.Array,
    level: jax.Array,
    nature: jax.Array,
) -> jax.Array:
    stat = ((2 * base + iv + ev // 4) * level) // 100 + 5
    return (stat * nature) // 100


def _stage_stat(stat: jax.Array, stage: jax.Array) -> jax.Array:
    positive = (stat * (2 + stage)) // 2
    negative = (stat * 2) // (2 - stage)
    return jnp.where(stage >= 0, positive, negative)


def _damage(
    params: jax.Array,
    offset: int,
    stage: jax.Array,
    rolls: jax.Array,
) -> jax.Array:
    attacker_level = params[:, offset + ATTACKER_LEVEL]
    defender_level = params[:, offset + DEFENDER_LEVEL]
    attack = _ordinary_stat(
        params[:, offset + ATTACK_BASE],
        params[:, offset + ATTACK_IV],
        params[:, offset + ATTACK_EV],
        attacker_level,
        params[:, offset + ATTACK_NATURE],
    )
    attack = _stage_stat(attack, stage)
    attack = _modify(attack, params[:, offset + ATTACK_MOD])
    defense = _ordinary_stat(
        params[:, offset + DEFENSE_BASE],
        params[:, offset + DEFENSE_IV],
        params[:, offset + DEFENSE_EV],
        defender_level,
        params[:, offset + DEFENSE_NATURE],
    )
    defense = _modify(defense, params[:, offset + DEFENSE_MOD])

    base = (
        ((2 * attacker_level) // 5 + 2)
        * params[:, offset + BASE_POWER]
        * attack
    ) // defense
    base = base // 50 + 2
    base = (base * (100 - rolls)) // 100
    base = _modify(base, params[:, offset + STAB_MOD])

    type_mod = params[:, offset + TYPE_MOD]
    positive = jnp.maximum(type_mod, 0)
    negative = jnp.maximum(-type_mod, 0)
    base = jnp.where(
        type_mod >= 0,
        base * jnp.left_shift(jnp.int32(1), positive),
        base // jnp.left_shift(jnp.int32(1), negative),
    )
    base = _modify(base, params[:, offset + BURN_MOD])
    base = _modify(base, params[:, offset + FINAL_MOD])
    return jnp.maximum(jnp.int32(1), base)


def _p1_first(params: jax.Array, order_tie_rolls: jax.Array) -> jax.Array:
    p1_priority = params[:, P1_PRIORITY_COLUMN]
    p2_priority = params[:, P2_PRIORITY_COLUMN]
    p1_speed = params[:, P1_SPEED_COLUMN]
    p2_speed = params[:, P2_SPEED_COLUMN]
    return jnp.where(
        p1_priority > p2_priority,
        True,
        jnp.where(
            p1_priority < p2_priority,
            False,
            jnp.where(
                p1_speed > p2_speed,
                True,
                jnp.where(p1_speed < p2_speed, False, order_tie_rolls == 0),
            ),
        ),
    )


@jax.jit
def two_attack_turn_batch(
    params: jax.Array,
    order_tie_rolls: jax.Array,
    p1_accuracy_rolls: jax.Array,
    p1_damage_rolls: jax.Array,
    p1_secondary_rolls: jax.Array,
    p2_accuracy_rolls: jax.Array,
    p2_damage_rolls: jax.Array,
) -> jax.Array:
    p1_first = _p1_first(params, order_tie_rolls)
    p1_protect = params[:, P1_ACTION_KIND_COLUMN] == P1_ACTION_PROTECT

    initial_p1_hp = params[:, ATTACKER_HP]
    initial_p2_hp = params[:, DEFENDER_HP]
    initial_p1_pp = params[:, MOVE_PP]
    initial_p2_pp = params[:, P2_ATTACK_OFFSET + MOVE_PP]
    initial_stage = params[:, P2_SPA_STAGE_COLUMN]

    p1_damage = _damage(
        params,
        0,
        jnp.zeros_like(initial_stage),
        p1_damage_rolls,
    )
    p2_damage_initial = _damage(
        params,
        P2_ATTACK_OFFSET,
        initial_stage,
        p2_damage_rolls,
    )

    p1_hit = p1_accuracy_rolls < params[:, ACCURACY]
    p2_hit = p2_accuracy_rolls < params[:, P2_ATTACK_OFFSET + ACCURACY]

    p2_after_p1 = jnp.maximum(
        jnp.int32(0),
        initial_p2_hp - jnp.where(p1_hit & ~p1_protect, p1_damage, 0),
    )
    secondary_applies_p1_first = (
        p1_hit
        & ~p1_protect
        & (p2_after_p1 > 0)
        & (p1_secondary_rolls < params[:, P1_SPA_DROP_CHANCE_COLUMN])
        & (initial_stage > -6)
    )
    stage_after_p1_first = jnp.where(
        secondary_applies_p1_first,
        initial_stage - 1,
        initial_stage,
    )
    p2_damage_after_secondary = _damage(
        params,
        P2_ATTACK_OFFSET,
        stage_after_p1_first,
        p2_damage_rolls,
    )
    p2_exec_after_p1 = p2_after_p1 > 0
    p1_after_p2_second = jnp.maximum(
        jnp.int32(0),
        initial_p1_hp
        - jnp.where(
            p2_exec_after_p1 & p2_hit & ~p1_protect,
            p2_damage_after_secondary,
            0,
        ),
    )

    p1_after_p2_first = jnp.maximum(
        jnp.int32(0),
        initial_p1_hp - jnp.where(p2_hit, p2_damage_initial, 0),
    )
    p1_exec_after_p2 = p1_after_p2_first > 0
    p2_after_p1_second = jnp.maximum(
        jnp.int32(0),
        initial_p2_hp
        - jnp.where(p1_exec_after_p2 & p1_hit & ~p1_protect, p1_damage, 0),
    )
    secondary_applies_p2_first = (
        p1_exec_after_p2
        & p1_hit
        & ~p1_protect
        & (p2_after_p1_second > 0)
        & (p1_secondary_rolls < params[:, P1_SPA_DROP_CHANCE_COLUMN])
        & (initial_stage > -6)
    )
    stage_after_p2_first = jnp.where(
        secondary_applies_p2_first,
        initial_stage - 1,
        initial_stage,
    )

    p1_hp = jnp.where(p1_first, p1_after_p2_second, p1_after_p2_first)
    p2_hp = jnp.where(p1_first, p2_after_p1, p2_after_p1_second)
    p1_acted = jnp.where(p1_first, True, p1_exec_after_p2)
    p2_acted = jnp.where(p1_first, p2_exec_after_p1, True)
    p1_pp = jnp.maximum(
        jnp.int32(0),
        initial_p1_pp - p1_acted.astype(jnp.int32),
    )
    p2_pp = jnp.maximum(
        jnp.int32(0),
        initial_p2_pp - p2_acted.astype(jnp.int32),
    )
    p2_stage = jnp.where(p1_first, stage_after_p1_first, stage_after_p2_first)
    flags = p1_acted.astype(jnp.int32) + p2_acted.astype(jnp.int32) * 2

    return (
        p1_hp
        + p2_hp * jnp.int32(PACK_HP_BASE)
        + p1_pp * jnp.int32(PACK_P1_PP_BASE)
        + p2_pp * jnp.int32(PACK_P2_PP_BASE)
        + (p2_stage + 6) * jnp.int32(PACK_STAGE_BASE)
        + flags * jnp.int32(PACK_FLAGS_BASE)
    )


@jax.jit
def two_attack_turn_score(
    params: jax.Array,
    order_tie_rolls: jax.Array,
    p1_accuracy_rolls: jax.Array,
    p1_damage_rolls: jax.Array,
    p1_secondary_rolls: jax.Array,
    p2_accuracy_rolls: jax.Array,
    p2_damage_rolls: jax.Array,
) -> jax.Array:
    return jnp.sum(
        two_attack_turn_batch(
            params,
            order_tie_rolls,
            p1_accuracy_rolls,
            p1_damage_rolls,
            p1_secondary_rolls,
            p2_accuracy_rolls,
            p2_damage_rolls,
        ),
        dtype=jnp.int32,
    )


@jax.jit
def weighted_two_attack_turn_score(
    params: jax.Array,
    order_tie_rolls: jax.Array,
    p1_accuracy_rolls: jax.Array,
    p1_damage_rolls: jax.Array,
    p1_secondary_rolls: jax.Array,
    p2_accuracy_rolls: jax.Array,
    p2_damage_rolls: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    return jnp.sum(
        two_attack_turn_batch(
            params,
            order_tie_rolls,
            p1_accuracy_rolls,
            p1_damage_rolls,
            p1_secondary_rolls,
            p2_accuracy_rolls,
            p2_damage_rolls,
        )
        * weights,
        dtype=jnp.int32,
    )
