"""JAX correctness and partition-cost benchmark for the real Gen 9 damage kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from azelficoast.research.mechanics.gen9_damage import DamageContext, compile_numeric_context, damage
from azelficoast.research.mechanics.jax_gen9_damage import (
    contexts_to_array,
    damage_batch,
    damage_score,
    weighted_damage_score,
)
from azelficoast.research.verification.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    ShowdownDamageCorpusError,
    _context,
)


def _load_contexts(path: Path) -> tuple[DamageContext, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ShowdownDamageCorpusError("fixture document must be an object")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise ShowdownDamageCorpusError("fixture Showdown revision does not match the pinned oracle")
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
        raise ShowdownDamageCorpusError("fixture document lacks fixtures")
    contexts = tuple(
        _context(fixture["context"])
        for fixture in fixtures
        if isinstance(fixture, Mapping) and isinstance(fixture.get("context"), Mapping)
    )
    if not contexts:
        raise ShowdownDamageCorpusError("fixture document has no contexts")
    return contexts


def correctness_check(contexts: Sequence[DamageContext]) -> dict[str, object]:
    expanded_contexts = tuple(
        context
        for context in contexts
        for _ in range(16)
    )
    rolls_np = np.tile(np.arange(16, dtype=np.int32), len(contexts))
    params = contexts_to_array(expanded_contexts)
    rolls = jnp.asarray(rolls_np)

    actual = np.asarray(damage_batch(params, rolls), dtype=np.int32)
    expected = np.asarray(
        [
            damage(context, roll)
            for context in contexts
            for roll in range(16)
        ],
        dtype=np.int32,
    )
    return {
        "scenario_count": len(contexts),
        "roll_case_count": len(expected),
        "exact": bool(np.array_equal(actual, expected)),
    }


def _logical_worlds(
    contexts: Sequence[DamageContext],
    world_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    compiled = np.asarray(
        [compile_numeric_context(context) for context in contexts],
        dtype=np.int32,
    )
    class_count = len(contexts) * 16
    if world_count < class_count:
        raise ValueError("world_count must be at least the number of damage classes")

    params = np.empty((world_count, compiled.shape[1]), dtype=np.int32)
    rolls = np.empty(world_count, dtype=np.int32)
    for index in range(world_count):
        class_index = index % class_count
        scenario_index = class_index // 16
        roll = class_index % 16
        params[index] = compiled[scenario_index]
        rolls[index] = roll
    return params, rolls


def _partition(
    params: np.ndarray,
    rolls: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.concatenate([params, rolls[:, None]], axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    return (
        unique[:, :-1].astype(np.int32, copy=False),
        unique[:, -1].astype(np.int32, copy=False),
        counts.astype(np.int32, copy=False),
    )


def _median_ms(call: Callable[[], int], repeats: int) -> tuple[float, int]:
    samples: list[float] = []
    result = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples), result


def benchmark_size(
    contexts: Sequence[DamageContext],
    world_count: int,
    repeats: int = 9,
) -> dict[str, object]:
    params_np, rolls_np = _logical_worlds(contexts, world_count)

    params_device = jax.device_put(params_np)
    rolls_device = jax.device_put(rolls_np)
    warm = damage_score(params_device, rolls_device)
    warm.block_until_ready()

    def direct_call() -> int:
        value = damage_score(params_device, rolls_device)
        value.block_until_ready()
        return int(np.asarray(value))

    direct_ms, direct_score = _median_ms(direct_call, repeats)

    unique_params, unique_rolls, weights = _partition(params_np, rolls_np)
    unique_params_device = jax.device_put(unique_params)
    unique_rolls_device = jax.device_put(unique_rolls)
    weights_device = jax.device_put(weights)
    warm_reduced = weighted_damage_score(
        unique_params_device,
        unique_rolls_device,
        weights_device,
    )
    warm_reduced.block_until_ready()

    def prepartitioned_call() -> int:
        value = weighted_damage_score(
            unique_params_device,
            unique_rolls_device,
            weights_device,
        )
        value.block_until_ready()
        return int(np.asarray(value))

    prepartitioned_ms, prepartitioned_score = _median_ms(prepartitioned_call, repeats)

    def inclusive_call() -> int:
        p, r, w = _partition(params_np, rolls_np)
        value = weighted_damage_score(
            jax.device_put(p),
            jax.device_put(r),
            jax.device_put(w),
        )
        value.block_until_ready()
        return int(np.asarray(value))

    inclusive_ms, inclusive_score = _median_ms(inclusive_call, repeats)

    return {
        "world_count": world_count,
        "dependency_classes": int(len(unique_rolls)),
        "direct_median_ms": direct_ms,
        "prepartitioned_reduced_median_ms": prepartitioned_ms,
        "prepartitioned_logical_speedup": direct_ms / prepartitioned_ms,
        "partition_inclusive_reduced_median_ms": inclusive_ms,
        "partition_inclusive_logical_speedup": direct_ms / inclusive_ms,
        "score_equal": (
            direct_score == prepartitioned_score == inclusive_score
        ),
    }


def run_experiment(fixtures: Path) -> dict[str, object]:
    contexts = _load_contexts(fixtures)
    correctness = correctness_check(contexts)
    benchmarks = [
        benchmark_size(contexts, world_count)
        for world_count in (4096, 65536, 524288)
    ]

    passed = (
        correctness["exact"] is True
        and all(row["score_equal"] for row in benchmarks)
    )
    return {
        "schema": "azelficoast.gen9-damage-jax-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "jax_version": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "correctness": correctness,
        "benchmarks": benchmarks,
        "passed": passed,
        "non_claims": [
            "the kernel covers only the preregistered ordinary Gen 9 damage subset",
            "type-chart exponents are fixture/compiler inputs, not yet generated runtime tables",
            "the inclusive partition benchmark uses numpy.unique on the CPU as a deliberately simple baseline",
            "GitHub-hosted benchmark evidence is CPU evidence, not GPU evidence",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_experiment(args.fixtures)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
