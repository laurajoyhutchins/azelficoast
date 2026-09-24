"""Hosted calibration and held-out validation for adaptive simulator dispatch."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import jax
import numpy as np

from azelficoast.adaptive_execution import (
    ExecutionCostProfile,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
)
from azelficoast.class_native_belief import (
    ClassNativeBelief,
    ProjectionMap,
    build_factor_support,
    compile_damage_projection,
    damage_dependency_tuple,
    project_belief,
)
from azelficoast.gen9_damage import DamageContext, compile_numeric_context
from azelficoast.jax_gen9_damage import damage_score, weighted_damage_score
from azelficoast.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    ShowdownDamageCorpusError,
    _context,
)


@dataclass(frozen=True)
class Treatment:
    logical_world_count: int
    active_context_count: int
    bench_variants: int

    @property
    def name(self) -> str:
        return (
            f"n{self.logical_world_count}-"
            f"contexts{self.active_context_count}-"
            f"bench{self.bench_variants}"
        )


TRAINING_TREATMENTS = (
    Treatment(2048, 2, 1),
    Treatment(4096, 12, 1),
    Treatment(8192, 4, 4),
    Treatment(16384, 8, 2),
    Treatment(32768, 12, 8),
    Treatment(65536, 3, 16),
    Treatment(131072, 6, 8),
    Treatment(262144, 12, 16),
)

HELD_OUT_TREATMENTS = (
    Treatment(3072, 3, 2),
    Treatment(6144, 12, 2),
    Treatment(12288, 6, 4),
    Treatment(24576, 12, 4),
    Treatment(49152, 4, 8),
    Treatment(98304, 10, 8),
    Treatment(196608, 6, 16),
    Treatment(524288, 12, 8),
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
    if len(contexts) < max(t.active_context_count for t in (*TRAINING_TREATMENTS, *HELD_OUT_TREATMENTS)):
        raise ShowdownDamageCorpusError("fixture corpus lacks enough distinct damage contexts")
    return contexts


def _effect_signature(contexts: Sequence[DamageContext]) -> str:
    payload = json.dumps(
        {
            "effect": "gen9-damage",
            "showdown_commit": PINNED_SHOWDOWN_COMMIT,
            "dependencies": [damage_dependency_tuple(context) for context in contexts],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sparse_uniform_belief(
    contexts: Sequence[DamageContext],
    treatment: Treatment,
) -> tuple[ClassNativeBelief, ProjectionMap]:
    support = build_factor_support(
        len(contexts),
        bench_variants=treatment.bench_variants,
        rolls=16,
    )
    active = support.context_index < treatment.active_context_count
    active_count = int(np.count_nonzero(active))
    if treatment.logical_world_count < active_count:
        raise ValueError(
            f"{treatment.name} has fewer logical worlds than active canonical classes"
        )

    weights = np.zeros(support.class_count, dtype=np.int64)
    quotient, remainder = divmod(treatment.logical_world_count, active_count)
    active_indices = np.flatnonzero(active)
    weights[active_indices] = quotient
    if remainder:
        weights[active_indices[:remainder]] += 1

    belief = ClassNativeBelief(support=support, weights=weights)
    projection = compile_damage_projection(support, contexts)
    return belief, projection


def _active_projected_inputs(
    belief: ClassNativeBelief,
    projection: ProjectionMap,
    contexts: Sequence[DamageContext],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = project_belief(belief, projection)
    active = projected.weights > 0
    representatives = projection.representative_indices[active]
    context_indices = belief.support.context_index[representatives]
    params = np.asarray(
        [
            compile_numeric_context(contexts[int(context_index)])
            for context_index in context_indices
        ],
        dtype=np.int32,
    )
    rolls = belief.support.roll[representatives].astype(np.int32, copy=False)
    weights = projected.weights[active].astype(np.int32, copy=False)
    return params, rolls, weights


def _materialize_direct_inputs(
    belief: ClassNativeBelief,
    contexts: Sequence[DamageContext],
) -> tuple[np.ndarray, np.ndarray]:
    active = belief.weights > 0
    context_index = np.repeat(
        belief.support.context_index[active],
        belief.weights[active],
    )
    rolls = np.repeat(
        belief.support.roll[active],
        belief.weights[active],
    ).astype(np.int32, copy=False)
    context_rows = np.asarray(
        [compile_numeric_context(context) for context in contexts],
        dtype=np.int32,
    )
    return context_rows[context_index], rolls


def _median_ms(call: Callable[[], int], repeats: int = 5) -> tuple[float, int]:
    samples: list[float] = []
    result = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples), result


def _benchmark_treatment(
    contexts: Sequence[DamageContext],
    treatment: Treatment,
    *,
    backend: str,
    effect_signature: str,
) -> dict[str, object]:
    belief, projection = _sparse_uniform_belief(contexts, treatment)

    direct_params, direct_rolls = _materialize_direct_inputs(belief, contexts)
    direct_params_device = jax.device_put(direct_params)
    direct_rolls_device = jax.device_put(direct_rolls)
    direct_warm = damage_score(direct_params_device, direct_rolls_device)
    direct_warm.block_until_ready()

    def direct_call() -> int:
        value = damage_score(direct_params_device, direct_rolls_device)
        value.block_until_ready()
        return int(np.asarray(value))

    direct_ms, direct_score = _median_ms(direct_call)

    # Projection is precompiled from the effect signature and canonical support. The
    # timed projected call still pays weight projection, active-class compaction,
    # representative assembly, and transfer to the accelerator.
    def projected_call() -> int:
        params, rolls, weights = _active_projected_inputs(
            belief,
            projection,
            contexts,
        )
        value = weighted_damage_score(
            jax.device_put(params),
            jax.device_put(rolls),
            jax.device_put(weights),
        )
        value.block_until_ready()
        return int(np.asarray(value))

    projected_warm = projected_call()
    projected_ms, projected_score = _median_ms(projected_call)

    projected = project_belief(belief, projection)
    features = ExecutionFeatures(
        backend=backend,
        effect_signature=effect_signature,
        logical_world_count=belief.logical_world_count,
        active_canonical_classes=belief.active_canonical_classes,
        active_projected_classes=projected.active_classes,
    )

    return {
        "treatment": treatment.name,
        "logical_world_count": features.logical_world_count,
        "active_canonical_classes": features.active_canonical_classes,
        "active_projected_classes": features.active_projected_classes,
        "direct_median_ms": direct_ms,
        "projected_median_ms": projected_ms,
        "oracle_path": (
            ExecutionPath.PROJECTED.value
            if projected_ms < direct_ms
            else ExecutionPath.DIRECT.value
        ),
        "score_equal": direct_score == projected_score == projected_warm,
    }


def _nonnegative_least_squares(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Solve tiny non-negative least squares by enumerating active coefficient faces."""
    columns = x.shape[1]
    best: np.ndarray | None = None
    best_error = float("inf")
    for mask in itertools.product((False, True), repeat=columns):
        if not any(mask):
            candidate = np.zeros(columns, dtype=np.float64)
        else:
            indices = np.flatnonzero(mask)
            partial, *_ = np.linalg.lstsq(x[:, indices], y, rcond=None)
            if np.any(partial < 0):
                continue
            candidate = np.zeros(columns, dtype=np.float64)
            candidate[indices] = partial
        residual = x @ candidate - y
        error = float(np.dot(residual, residual))
        if error < best_error:
            best = candidate
            best_error = error
    if best is None:
        raise RuntimeError("non-negative cost fit found no feasible profile")
    return best


def _fit_profile(
    rows: Sequence[Mapping[str, object]],
    *,
    backend: str,
    effect_signature: str,
) -> ExecutionCostProfile:
    direct_x = np.asarray(
        [
            [1.0, float(row["logical_world_count"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    direct_y = np.asarray(
        [float(row["direct_median_ms"]) for row in rows],
        dtype=np.float64,
    )
    direct = _nonnegative_least_squares(direct_x, direct_y)

    projected_x = np.asarray(
        [
            [
                1.0,
                float(row["active_canonical_classes"]),
                float(row["active_projected_classes"]),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    projected_y = np.asarray(
        [float(row["projected_median_ms"]) for row in rows],
        dtype=np.float64,
    )
    projected = _nonnegative_least_squares(projected_x, projected_y)

    return ExecutionCostProfile(
        backend=backend,
        effect_signature=effect_signature,
        direct_intercept_ms=float(direct[0]),
        direct_per_world_ms=float(direct[1]),
        projected_intercept_ms=float(projected[0]),
        projected_per_canonical_class_ms=float(projected[1]),
        projected_per_execution_class_ms=float(projected[2]),
    )


def _profile_record(profile: ExecutionCostProfile) -> dict[str, object]:
    return {
        "backend": profile.backend,
        "effect_signature": profile.effect_signature,
        "direct": {
            "intercept_ms": profile.direct_intercept_ms,
            "per_world_ms": profile.direct_per_world_ms,
        },
        "projected": {
            "intercept_ms": profile.projected_intercept_ms,
            "per_active_canonical_class_ms": profile.projected_per_canonical_class_ms,
            "per_active_execution_class_ms": profile.projected_per_execution_class_ms,
        },
    }


def _evaluate_held_out(
    profile: ExecutionCostProfile,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    oracle_total = 0.0
    adaptive_total = 0.0
    direct_total = 0.0
    projected_total = 0.0
    correct = 0
    chosen_paths: set[str] = set()

    for row in rows:
        features = ExecutionFeatures(
            backend=profile.backend,
            effect_signature=profile.effect_signature,
            logical_world_count=int(row["logical_world_count"]),
            active_canonical_classes=int(row["active_canonical_classes"]),
            active_projected_classes=int(row["active_projected_classes"]),
        )
        decision = choose_execution_path(profile, features)
        direct_ms = float(row["direct_median_ms"])
        projected_ms = float(row["projected_median_ms"])
        oracle_path = str(row["oracle_path"])
        chosen_ms = (
            projected_ms
            if decision.path is ExecutionPath.PROJECTED
            else direct_ms
        )
        oracle_ms = min(direct_ms, projected_ms)

        oracle_total += oracle_ms
        adaptive_total += chosen_ms
        direct_total += direct_ms
        projected_total += projected_ms
        chosen_paths.add(decision.path.value)
        correct += int(decision.path.value == oracle_path)

        records.append(
            {
                **dict(row),
                "chosen_path": decision.path.value,
                "predicted_direct_ms": decision.predicted_direct_ms,
                "predicted_projected_ms": decision.predicted_projected_ms,
                "choice_matches_oracle": decision.path.value == oracle_path,
                "chosen_over_oracle": chosen_ms / oracle_ms,
            }
        )

    return {
        "rows": records,
        "choice_accuracy": correct / len(rows),
        "chosen_paths": sorted(chosen_paths),
        "oracle_total_ms": oracle_total,
        "adaptive_total_ms": adaptive_total,
        "always_direct_total_ms": direct_total,
        "always_projected_total_ms": projected_total,
        "adaptive_over_oracle": adaptive_total / oracle_total,
        "speedup_vs_always_direct": direct_total / adaptive_total,
        "speedup_vs_always_projected": projected_total / adaptive_total,
    }


def run_experiment(fixtures: Path) -> dict[str, object]:
    contexts = _load_contexts(fixtures)
    backend = jax.default_backend()
    effect_signature = _effect_signature(contexts)

    training = [
        _benchmark_treatment(
            contexts,
            treatment,
            backend=backend,
            effect_signature=effect_signature,
        )
        for treatment in TRAINING_TREATMENTS
    ]
    profile = _fit_profile(
        training,
        backend=backend,
        effect_signature=effect_signature,
    )

    held_out_rows = [
        _benchmark_treatment(
            contexts,
            treatment,
            backend=backend,
            effect_signature=effect_signature,
        )
        for treatment in HELD_OUT_TREATMENTS
    ]
    held_out = _evaluate_held_out(profile, held_out_rows)

    training_names = {row["treatment"] for row in training}
    held_out_names = {row["treatment"] for row in held_out_rows}
    disjoint = training_names.isdisjoint(held_out_names)

    passed = (
        disjoint
        and all(bool(row["score_equal"]) for row in training)
        and all(bool(row["score_equal"]) for row in held_out_rows)
        and held_out["choice_accuracy"] >= 0.75
        and held_out["adaptive_over_oracle"] <= 1.15
        and held_out["chosen_paths"] == ["direct", "projected"]
        and held_out["speedup_vs_always_direct"] > 1.0
        and held_out["speedup_vs_always_projected"] > 1.0
    )

    return {
        "schema": "azelficoast.adaptive-execution-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "jax_version": jax.__version__,
        "backend": backend,
        "effect_signature": effect_signature,
        "training_and_held_out_disjoint": disjoint,
        "training": training,
        "profile": _profile_record(profile),
        "held_out": held_out,
        "passed": passed,
        "acceptance": {
            "minimum_held_out_choice_accuracy": 0.75,
            "maximum_held_out_latency_vs_oracle": 1.15,
            "must_choose_both_paths": True,
            "must_beat_always_direct": True,
            "must_beat_always_projected": True,
        },
        "non_claims": [
            "the fitted coefficients are specific to this backend and effect signature",
            "the hosted profile is experimental evidence, not a portable production constant",
            "the direct timed path receives pre-materialized device-resident worlds",
            "the projected path pays projection, compaction, representative assembly, and transfer",
            "GPU crossover behavior is not established by this CPU experiment",
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
