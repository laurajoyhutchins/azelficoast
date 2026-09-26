"""Experiment: persistent class-native beliefs versus raw-world regrouping."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import jax
import numpy as np

from azelficoast.research.mechanics.class_native_belief import (
    ClassNativeBelief,
    ProjectionMap,
    build_factor_support,
    compile_bench_projection,
    compile_damage_projection,
    compile_full_projection,
    compile_protect_projection,
    filter_exact_damage_observation,
    materialize_projection_ids,
    project_belief,
    uniform_belief,
)
from azelficoast.research.mechanics.gen9_damage import DamageContext, compile_numeric_context, damage
from azelficoast.research.mechanics.jax_gen9_damage import damage_score, weighted_damage_score
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
        raise ShowdownDamageCorpusError(
            "fixture Showdown revision does not match the pinned oracle"
        )
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
        raise ShowdownDamageCorpusError("fixture document lacks fixtures")
    contexts = tuple(
        _context(fixture["context"])
        for fixture in fixtures
        if isinstance(fixture, Mapping)
        and isinstance(fixture.get("context"), Mapping)
    )
    if not contexts:
        raise ShowdownDamageCorpusError("fixture document has no contexts")
    return contexts


def _projection_damage_inputs(
    belief: ClassNativeBelief,
    projection: ProjectionMap,
    contexts: Sequence[DamageContext],
) -> tuple[np.ndarray, np.ndarray]:
    representatives = projection.representative_indices
    context_indices = belief.support.context_index[representatives]
    rolls = belief.support.roll[representatives].astype(np.int32, copy=False)
    params = np.asarray(
        [
            compile_numeric_context(contexts[int(index)])
            for index in context_indices
        ],
        dtype=np.int32,
    )
    return params, rolls


def _weighted_score(
    belief: ClassNativeBelief,
    projection: ProjectionMap,
    contexts: Sequence[DamageContext],
) -> int:
    projected = project_belief(belief, projection)
    params, rolls = _projection_damage_inputs(belief, projection, contexts)
    value = weighted_damage_score(
        jax.device_put(params),
        jax.device_put(rolls),
        jax.device_put(projected.weights.astype(np.int32)),
    )
    value.block_until_ready()
    return int(np.asarray(value))


def _raw_score(
    belief: ClassNativeBelief,
    contexts: Sequence[DamageContext],
) -> int:
    per_class = np.fromiter(
        (
            damage(contexts[int(context_index)], int(roll))
            for context_index, roll in zip(
                belief.support.context_index,
                belief.support.roll,
                strict=True,
            )
        ),
        dtype=np.int64,
        count=belief.support.class_count,
    )
    return int(np.dot(per_class, belief.weights))


def _materialize_damage_arrays(
    belief: ClassNativeBelief,
    contexts: Sequence[DamageContext],
) -> tuple[np.ndarray, np.ndarray]:
    context_rows = np.asarray(
        [compile_numeric_context(context) for context in contexts],
        dtype=np.int32,
    )
    expanded_context_index = np.repeat(
        belief.support.context_index,
        belief.weights.astype(np.int64, copy=False),
    )
    expanded_roll = np.repeat(
        belief.support.roll,
        belief.weights.astype(np.int64, copy=False),
    ).astype(np.int32, copy=False)
    return context_rows[expanded_context_index], expanded_roll


def _raw_regroup_score(
    belief: ClassNativeBelief,
    contexts: Sequence[DamageContext],
) -> int:
    # Deliberately recreate one row per logical world, then rediscover the same
    # dependency classes with a generic sort/unique path. This is the architecture
    # falsified by the prior experiment and retained here as a direct comparison.
    context_rows = np.asarray(
        [compile_numeric_context(context) for context in contexts],
        dtype=np.int32,
    )
    expanded_context_index = np.repeat(
        belief.support.context_index,
        belief.weights.astype(np.int64, copy=False),
    )
    expanded_roll = np.repeat(
        belief.support.roll,
        belief.weights.astype(np.int64, copy=False),
    )
    params = context_rows[expanded_context_index]
    keys = np.concatenate([params[:, :-1], expanded_roll[:, None]], axis=1)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    unique_params = np.empty((len(unique), params.shape[1]), dtype=np.int32)
    unique_params[:, :-1] = unique[:, :-1]
    # CATEGORY is not part of the damage dependency key. Recover any compatible
    # representative category metadata; the JAX kernel does not read it.
    unique_params[:, -1] = 0
    rolls = unique[:, -1].astype(np.int32, copy=False)
    value = weighted_damage_score(
        jax.device_put(unique_params),
        jax.device_put(rolls),
        jax.device_put(counts.astype(np.int32, copy=False)),
    )
    value.block_until_ready()
    return int(np.asarray(value))


def _median_ms(call: Callable[[], int], repeats: int) -> tuple[float, int]:
    samples: list[float] = []
    result = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples), result


def _projection_evidence(
    belief: ClassNativeBelief,
    contexts: Sequence[DamageContext],
) -> dict[str, object]:
    protect = compile_protect_projection(belief.support)
    damage_projection = compile_damage_projection(belief.support, contexts)
    bench = compile_bench_projection(belief.support)
    full = compile_full_projection(belief.support)

    sequence = [
        project_belief(belief, protect).active_classes,
        project_belief(belief, damage_projection).active_classes,
        project_belief(belief, bench).active_classes,
        project_belief(belief, full).active_classes,
        project_belief(belief, protect).active_classes,
    ]
    totals = [
        project_belief(belief, projection).logical_world_count
        for projection in (protect, damage_projection, bench, full)
    ]

    expanded_damage_ids = materialize_projection_ids(belief, damage_projection)
    raw_damage_counts = np.bincount(
        expanded_damage_ids,
        minlength=damage_projection.class_count,
    ).astype(np.int64, copy=False)
    compressed_damage_counts = project_belief(
        belief,
        damage_projection,
    ).weights

    return {
        "canonical_classes": belief.support.class_count,
        "protect_classes": protect.class_count,
        "damage_classes": damage_projection.class_count,
        "bench_classes": bench.class_count,
        "full_classes": full.class_count,
        "split_merge_sequence": sequence,
        "world_count_preserved": all(
            total == belief.logical_world_count for total in totals
        ),
        "damage_projection_exact": bool(
            np.array_equal(raw_damage_counts, compressed_damage_counts)
        ),
    }


def _observation_evidence(
    belief: ClassNativeBelief,
    contexts: Sequence[DamageContext],
) -> dict[str, object]:
    observed_damage = damage(contexts[0], 7)
    posterior = filter_exact_damage_observation(
        belief,
        contexts,
        observed_damage,
    )

    raw_context = np.repeat(
        belief.support.context_index,
        belief.weights.astype(np.int64, copy=False),
    )
    raw_roll = np.repeat(
        belief.support.roll,
        belief.weights.astype(np.int64, copy=False),
    )
    raw_matches = sum(
        damage(contexts[int(context_index)], int(roll)) == observed_damage
        for context_index, roll in zip(raw_context, raw_roll, strict=True)
    )

    return {
        "observed_damage": observed_damage,
        "prior_worlds": belief.logical_world_count,
        "posterior_worlds": posterior.logical_world_count,
        "raw_posterior_worlds": raw_matches,
        "exact": posterior.logical_world_count == raw_matches,
        "informative": 0 < posterior.logical_world_count < belief.logical_world_count,
    }


def benchmark(
    contexts: Sequence[DamageContext],
    logical_world_count: int,
    *,
    bench_variants: int = 8,
    repeats: int = 3,
) -> dict[str, object]:
    support = build_factor_support(
        len(contexts),
        bench_variants=bench_variants,
        rolls=16,
    )
    belief = uniform_belief(support, logical_world_count)
    projection = compile_damage_projection(support, contexts)
    bad_projection = compile_damage_projection(
        support,
        contexts,
        include_attack_modifier=False,
    )

    expected = _raw_score(belief, contexts)
    correct_score = _weighted_score(belief, projection, contexts)
    bad_score = _weighted_score(belief, bad_projection, contexts)

    raw_params, raw_rolls = _materialize_damage_arrays(belief, contexts)
    raw_params_device = jax.device_put(raw_params)
    raw_rolls_device = jax.device_put(raw_rolls)
    warm_direct = damage_score(raw_params_device, raw_rolls_device)
    warm_direct.block_until_ready()

    def direct_jax_call() -> int:
        value = damage_score(raw_params_device, raw_rolls_device)
        value.block_until_ready()
        return int(np.asarray(value))

    direct_jax_ms, direct_jax_score = _median_ms(direct_jax_call, repeats)

    # The class-native timing is deliberately stricter than direct JAX: projection,
    # representative assembly, and host-to-device transfer are paid on every call.
    _weighted_score(belief, projection, contexts)
    class_native_ms, class_native_score = _median_ms(
        lambda: _weighted_score(belief, projection, contexts),
        repeats,
    )
    raw_regroup_ms, raw_regroup_score = _median_ms(
        lambda: _raw_regroup_score(belief, contexts),
        repeats,
    )

    return {
        "logical_world_count": logical_world_count,
        "canonical_classes": support.class_count,
        "damage_classes": projection.class_count,
        "bad_damage_classes": bad_projection.class_count,
        "direct_jax_median_ms": direct_jax_ms,
        "class_native_median_ms": class_native_ms,
        "raw_regroup_median_ms": raw_regroup_ms,
        "speedup_vs_direct_jax": direct_jax_ms / class_native_ms,
        "speedup_vs_raw_regroup": raw_regroup_ms / class_native_ms,
        "score_equal": (
            expected
            == correct_score
            == direct_jax_score
            == class_native_score
            == raw_regroup_score
        ),
        "missing_attack_modifier_negative_control_detected": bad_score != expected,
    }


def run_experiment(fixtures: Path) -> dict[str, object]:
    contexts = _load_contexts(fixtures)
    support = build_factor_support(
        len(contexts),
        bench_variants=8,
        rolls=16,
    )
    belief = uniform_belief(support, 65536)

    projections = _projection_evidence(belief, contexts)
    observation = _observation_evidence(belief, contexts)
    benchmarks = [
        benchmark(contexts, logical_world_count)
        for logical_world_count in (4096, 65536, 524288)
    ]

    largest = benchmarks[-1]
    passed = (
        projections["world_count_preserved"] is True
        and projections["damage_projection_exact"] is True
        and projections["split_merge_sequence"] == [1, 192, 8, 1536, 1]
        and observation["exact"] is True
        and observation["informative"] is True
        and all(row["score_equal"] for row in benchmarks)
        and all(
            row["missing_attack_modifier_negative_control_detected"]
            for row in benchmarks
        )
        and largest["speedup_vs_direct_jax"] > 10.0
        and largest["speedup_vs_raw_regroup"] > 10.0
    )

    return {
        "schema": "azelficoast.class-native-belief-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "projection_evidence": projections,
        "observation_evidence": observation,
        "benchmarks": benchmarks,
        "passed": passed,
        "non_claims": [
            "canonical support still enumerates semantic context/bench/roll combinations",
            "the experiment does not prove arbitrary hidden variables factor cleanly",
            "the projection compiler currently operates on a small canonical support on the host",
            "the raw-regroup baseline is intentionally generic numpy.unique over materialized worlds",
            "GitHub-hosted timing is CPU evidence, not GPU evidence",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
