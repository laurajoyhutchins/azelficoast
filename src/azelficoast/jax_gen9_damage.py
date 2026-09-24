"""JAX lowering of the narrow Gen 9 damage kernel."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from azelficoast.gen9_damage import DamageContext, compile_numeric_context

LEVEL = 0
BASE_POWER = 1
ATTACK_BASE = 2
ATTACK_IV = 3
ATTACK_EV = 4
ATTACK_NATURE = 5
DEFENSE_BASE = 6
DEFENSE_IV = 7
DEFENSE_EV = 8
DEFENSE_NATURE = 9
ATTACK_MOD = 10
STAB_MOD = 11
TYPE_MOD = 12
BURN_MOD = 13
FINAL_MOD = 14
CATEGORY = 15
PARAM_WIDTH = 16


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
    level = params[:, LEVEL]
    attack = _ordinary_stat(
        params[:, ATTACK_BASE],
        params[:, ATTACK_IV],
        params[:, ATTACK_EV],
        level,
        params[:, ATTACK_NATURE],
    )
    attack = _modify(attack, params[:, ATTACK_MOD])
    defense = _ordinary_stat(
        params[:, DEFENSE_BASE],
        params[:, DEFENSE_IV],
        params[:, DEFENSE_EV],
        level,
        params[:, DEFENSE_NATURE],
    )

    base = (((2 * level) // 5 + 2) * params[:, BASE_POWER] * attack) // defense
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
