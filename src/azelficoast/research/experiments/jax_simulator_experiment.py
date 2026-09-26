"""Correctness and CPU throughput experiment for the first JAX simulator lowering."""

from __future__ import annotations

import statistics
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from azelficoast.research.mechanics.jax_simulator import (
    array_to_worlds,
    protect_batch,
    protect_score,
    special_damage_batch,
    special_damage_score,
    weighted_protect_score,
    weighted_special_damage_score,
    worlds_to_array,
)
from azelficoast.research.mechanics.simulator_ir import (
    ITEM_CHOICE_SCARF,
    ITEM_CHOICE_SPECS,
    MOVE_AURA_SPHERE,
    PROTECT_BLOCK,
    SPECIAL_DAMAGE,
    RandomField,
    RandomInput,
    StateField,
    World,
    execute_direct,
    transition,
)


def _world(item: int, bench_signature: int) -> World:
    return World.from_values(
        {
            StateField.OWN_HP: 180,
            StateField.OPPONENT_HP: 100,
            StateField.OWN_SPEED: 206,
            StateField.OPPONENT_SPEED: 220,
            StateField.OPPONENT_ITEM: item,
            StateField.OPPONENT_MOVE: MOVE_AURA_SPHERE,
            StateField.BENCH_SIGNATURE: bench_signature,
        }
    )


def correctness_check() -> dict[str, object]:
    worlds = tuple(
        _world(item, bench_signature)
        for item in (ITEM_CHOICE_SCARF, ITEM_CHOICE_SPECS)
        for bench_signature in range(16)
    )
    states = worlds_to_array(worlds)

    protect_random = RandomInput.from_values({RandomField.DAMAGE_ROLL: 0})
    scalar_protect = execute_direct(PROTECT_BLOCK, worlds, protect_random)
    jax_protect = array_to_worlds(protect_batch(states))

    rolls_np = np.asarray([index % 16 for index in range(len(worlds))], dtype=np.int32)
    rolls = jnp.asarray(rolls_np)
    scalar_damage = tuple(
        world.apply(
            transition(
                SPECIAL_DAMAGE,
                world,
                RandomInput.from_values(
                    {RandomField.DAMAGE_ROLL: int(roll)}
                ),
            )
        )
        for world, roll in zip(worlds, rolls_np, strict=True)
    )
    jax_damage = array_to_worlds(special_damage_batch(states, rolls))

    return {
        "world_count": len(worlds),
        "protect_exact": jax_protect == scalar_protect,
        "damage_exact": jax_damage == scalar_damage,
    }


def _benchmark_states(
    world_count: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if world_count < 2 or world_count % 2:
        raise ValueError("world_count must be an even integer >= 2")

    half = world_count // 2
    states_np = np.zeros((world_count, len(StateField)), dtype=np.int32)
    states_np[:, int(StateField.OWN_HP)] = 180
    states_np[:, int(StateField.OPPONENT_HP)] = 100
    states_np[:, int(StateField.OPPONENT_MOVE)] = MOVE_AURA_SPHERE
    states_np[:half, int(StateField.OPPONENT_ITEM)] = ITEM_CHOICE_SCARF
    states_np[half:, int(StateField.OPPONENT_ITEM)] = ITEM_CHOICE_SPECS
    states_np[:half, int(StateField.BENCH_SIGNATURE)] = np.arange(half, dtype=np.int32)
    states_np[half:, int(StateField.BENCH_SIGNATURE)] = np.arange(half, dtype=np.int32)

    states = jax.device_put(states_np)
    rolls = jax.device_put(np.full((world_count,), 7, dtype=np.int32))

    protect_representatives = jax.device_put(states_np[[0]])
    protect_weights = jax.device_put(np.asarray([world_count], dtype=np.int32))

    damage_representatives = jax.device_put(states_np[[0, half]])
    damage_weights = jax.device_put(np.asarray([half, half], dtype=np.int32))

    return (
        states,
        rolls,
        protect_representatives,
        protect_weights,
        damage_representatives,
        damage_weights,
    )


def _median_ms(call: Callable[[], jax.Array], repeats: int) -> tuple[float, int]:
    warm = call()
    warm.block_until_ready()

    samples: list[float] = []
    last_value = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        value = call()
        value.block_until_ready()
        end = time.perf_counter_ns()
        samples.append((end - start) / 1_000_000)
        last_value = int(np.asarray(value))
    return statistics.median(samples), last_value


def benchmark_size(world_count: int, repeats: int = 15) -> dict[str, object]:
    (
        states,
        rolls,
        protect_representatives,
        protect_weights,
        damage_representatives,
        damage_weights,
    ) = _benchmark_states(world_count)

    protect_direct_ms, protect_direct_score = _median_ms(
        lambda: protect_score(states),
        repeats,
    )
    protect_reduced_ms, protect_reduced_score = _median_ms(
        lambda: weighted_protect_score(protect_representatives, protect_weights),
        repeats,
    )

    damage_direct_ms, damage_direct_score = _median_ms(
        lambda: special_damage_score(states, rolls),
        repeats,
    )
    reduced_rolls = jax.device_put(np.asarray([7, 7], dtype=np.int32))
    damage_reduced_ms, damage_reduced_score = _median_ms(
        lambda: weighted_special_damage_score(
            damage_representatives,
            reduced_rolls,
            damage_weights,
        ),
        repeats,
    )

    protect_speedup = protect_direct_ms / protect_reduced_ms
    damage_speedup = damage_direct_ms / damage_reduced_ms

    return {
        "world_count": world_count,
        "protect": {
            "classes": 1,
            "direct_median_ms": protect_direct_ms,
            "reduced_median_ms": protect_reduced_ms,
            "logical_speedup": protect_speedup,
            "direct_worlds_per_second": world_count / (protect_direct_ms / 1000),
            "reduced_logical_worlds_per_second": world_count / (protect_reduced_ms / 1000),
            "score_equal": protect_direct_score == protect_reduced_score,
        },
        "damage": {
            "classes": 2,
            "direct_median_ms": damage_direct_ms,
            "reduced_median_ms": damage_reduced_ms,
            "logical_speedup": damage_speedup,
            "direct_worlds_per_second": world_count / (damage_direct_ms / 1000),
            "reduced_logical_worlds_per_second": world_count / (damage_reduced_ms / 1000),
            "score_equal": damage_direct_score == damage_reduced_score,
        },
    }


def run_experiment() -> dict[str, object]:
    correctness = correctness_check()
    benchmarks = [
        benchmark_size(world_count)
        for world_count in (2048, 32768, 524288)
    ]
    largest = benchmarks[-1]

    passed = (
        correctness["protect_exact"] is True
        and correctness["damage_exact"] is True
        and all(row["protect"]["score_equal"] for row in benchmarks)
        and all(row["damage"]["score_equal"] for row in benchmarks)
        and largest["damage"]["logical_speedup"] > 1.0
    )

    return {
        "schema": "azelficoast.jax-simulator-experiment",
        "schema_version": 1,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "partition_build_timed": False,
        "correctness": correctness,
        "benchmarks": benchmarks,
        "passed": passed,
        "non_claims": [
            "the benchmark lowers the reference microkernel, not full Pokemon Showdown mechanics",
            "dependency partition construction is outside the timed region",
            "reduced throughput counts logical worlds represented by dependency classes",
            "GitHub-hosted evidence is CPU evidence, not GPU evidence",
        ],
    }

