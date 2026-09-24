"""JAX lowering of the narrow Gen 9 damage kernel."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from azelficoast.gen9_damage import DamageContext, compile_numeric_context

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
CATEGORY = 17
PARAM_WIDTH = 18


def contexts_to_array(contexts: Sequence[DamageContext]) -> jax.Array:
    return jnp.asarray(
        np.asarray([compile_numeric_context(context) for context in contexts], dtype=np.int32),
        dtype=jnp.int32,
    )


def contexts_to_numpy(contexts: Sequence[DamageContext]) -> np.ndarray:
    return np.asarray(
        [compile_numeric_context(context) for context in contexts],
        dtype=np.int32,
    )


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


@jax.jit
def damage_batch(params: jax.Array, rolls: jax.Array) -> jax.Array:
    attacker_level = params[:, ATTACKER_LEVEL]
    defender_level = params[:, DEFENDER_LEVEL]
    attack = _ordinary_stat(
        params[:, ATTACK_BASE],
        params[:, ATTACK_IV],
        params[:, ATTACK_EV],
        attacker_level,
        params[:, ATTACK_NATURE],
    )
    attack = _modify(attack, params[:, ATTACK_MOD])
    defense = _ordinary_stat(
        params[:, DEFENSE_BASE],
        params[:, DEFENSE_IV],
        params[:, DEFENSE_EV],
        defender_level,
        params[:, DEFENSE_NATURE],
    )
    defense = _modify(defense, params[:, DEFENSE_MOD])

    base = (((2 * attacker_level) // 5 + 2) * params[:, BASE_POWER] * attack) // defense
    base = base // 50 + 2
    base = (base * (100 - rolls)) // 100
    base = _modify(base, params[:, STAB_MOD])

    positive = jnp.maximum(params[:, TYPE_MOD], 0)
    negative = jnp.maximum(-params[:, TYPE_MOD], 0)
    base = jnp.where(
        params[:, TYPE_MOD] >= 0,
        base * jnp.left_shift(jnp.int32(1), positive),
        base // jnp.left_shift(jnp.int32(1), negative),
    )
    base = _modify(base, params[:, BURN_MOD])
    base = _modify(base, params[:, FINAL_MOD])
    return jnp.maximum(jnp.int32(1), base)


@jax.jit
def damage_score(params: jax.Array, rolls: jax.Array) -> jax.Array:
    return jnp.sum(damage_batch(params, rolls), dtype=jnp.int32)


@jax.jit
def weighted_damage_score(
    params: jax.Array,
    rolls: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    return jnp.sum(damage_batch(params, rolls) * weights, dtype=jnp.int32)
