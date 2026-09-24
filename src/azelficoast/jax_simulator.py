"""JAX lowering for the dependency-aware simulator reference microkernel."""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from azelficoast.simulator_ir import (
    ITEM_CHOICE_SPECS,
    EffectOp,
    EffectSpec,
    StateField,
    World,
)

STATE_WIDTH = len(StateField)


def worlds_to_array(worlds: Sequence[World]) -> jax.Array:
    return jnp.asarray(
        np.asarray([world.values for world in worlds], dtype=np.int32),
        dtype=jnp.int32,
    )


def array_to_worlds(states: jax.Array) -> tuple[World, ...]:
    values = np.asarray(states, dtype=np.int32)
    return tuple(World(tuple(int(value) for value in row)) for row in values)


def _protect_value(states: jax.Array) -> jax.Array:
    move = states[:, int(StateField.OPPONENT_MOVE)]
    current = states[:, int(StateField.OWN_PROTECTED)]
    return jnp.where(move != 0, jnp.int32(1), current)


def _damage_hp(states: jax.Array, rolls: jax.Array) -> jax.Array:
    hp = states[:, int(StateField.OWN_HP)]
    item = states[:, int(StateField.OPPONENT_ITEM)]
    base_damage = jnp.where(item == ITEM_CHOICE_SPECS, 90, 60)
    return jnp.maximum(jnp.int32(0), hp - base_damage - rolls)


@jax.jit
def protect_batch(states: jax.Array) -> jax.Array:
    return states.at[:, int(StateField.OWN_PROTECTED)].set(_protect_value(states))


@jax.jit
def special_damage_batch(states: jax.Array, rolls: jax.Array) -> jax.Array:
    return states.at[:, int(StateField.OWN_HP)].set(_damage_hp(states, rolls))


def step_batch(
    spec: EffectSpec,
    states: jax.Array,
    rolls: jax.Array,
) -> jax.Array:
    if spec.op is EffectOp.PROTECT_BLOCK:
        return protect_batch(states)
    if spec.op is EffectOp.SPECIAL_DAMAGE:
        return special_damage_batch(states, rolls)
    raise ValueError(f"unsupported JAX effect op {spec.op!r}")


@jax.jit
def protect_score(states: jax.Array) -> jax.Array:
    return jnp.sum(_protect_value(states), dtype=jnp.int32)


@jax.jit
def weighted_protect_score(states: jax.Array, weights: jax.Array) -> jax.Array:
    return jnp.sum(_protect_value(states) * weights, dtype=jnp.int32)


@jax.jit
def special_damage_score(states: jax.Array, rolls: jax.Array) -> jax.Array:
    return jnp.sum(_damage_hp(states, rolls), dtype=jnp.int32)


@jax.jit
def weighted_special_damage_score(
    states: jax.Array,
    rolls: jax.Array,
    weights: jax.Array,
) -> jax.Array:
    return jnp.sum(_damage_hp(states, rolls) * weights, dtype=jnp.int32)
