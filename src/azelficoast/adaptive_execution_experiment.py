"""Compare the original absolute cost model with a robust relative cost model."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

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


@dataclass(frozen=True)
class RelativeModelShape:
    canonical_term: bool
    quadratic_world_term: bool

    @property
    def name(self) -> str:
        pieces = ["linear-world"]
        if self.quadratic_world_term:
            pieces.append("saturation")
        pieces.append("execution-classes")
        if self.canonical_term:
            pieces.append("canonical-classes")
        return "+".join(pieces)

    @property
    def complexity(self) -> int:
        return 3 + int(self.canonical_term) + int(self.quadratic_world_term)


TRAINING_TREATMENTS = (
    Treatment(2048, 2, 1),
    Treatment(4096, 12, 1),
    Treatment(6144, 4, 4),
    Treatment(8192, 8, 1),
    Treatment(12288, 3, 8),
    Treatment(16384, 12, 2),
    Treatment(24576, 6, 4),
    Treatment(32768, 12, 8),
    Treatment(65536, 10, 4),
    Treatment(131072, 6, 8),
    Treatment(262144, 12, 16),
    Treatment(524288, 12, 8),
)

HELD_OUT_TREATMENTS = (
    Treatment(3072, 3, 2),
    Treatment(5120, 10, 1),
    Treatment(7168, 12, 2),
    Treatment(10240, 5, 4),
    Treatment(14336, 9, 2),
    Treatment(20480, 12, 4),
    Treatment(28672, 4, 8),
    Treatment(40960, 8, 8),
    Treatment(57344, 12, 8),
    Treatment(98304, 10, 8),
    Treatment(196608, 6, 16),
    Treatment(393216, 8, 16),
)

MODEL_SHAPES = (
    RelativeModelShape(canonical_term=False, quadratic_world_term=False),
    RelativeModelShape(canonical_term=True, quadratic_world_term=False),
    RelativeModelShape(canonical_term=False, quadratic_world_term=True),
    RelativeModelShape(canonical_term=True, quadratic_world_term=True),
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
    required = max(
        treatment.active_context_count
        for treatment in (*TRAINING_TREATMENTS, *HELD_OUT_TREATMENTS)
    )
    if len(contexts) < required:
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
    return belief, compile_damage_projection(support, contexts)


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


def _median_and_mad(samples: Sequence[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return median, mad


def _benchmark_treatment(
    contexts: Sequence[DamageContext],
    treatment: Treatment,
    *,
    repeats: int = 9,
) -> dict[str, object]:
    belief, projection = _sparse_uniform_belief(contexts, treatment)
    projected = project_belief(belief, projection)

    direct_params, direct_rolls = _materialize_direct_inputs(belief, contexts)
    direct_params_device = jax.device_put(direct_params)
    direct_rolls_device = jax.device_put(direct_rolls)

    def direct_call() -> int:
        value = damage_score(direct_params_device, direct_rolls_device)
        value.block_until_ready()
        return int(np.asarray(value))

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

    direct_warm = direct_call()
    projected_warm = projected_call()

    direct_samples: list[float] = []
    projected_samples: list[float] = []
    direct_score = direct_warm
    projected_score = projected_warm

    for repeat in range(repeats):
        calls = (
            (projected_call, projected_samples, "projected"),
            (direct_call, direct_samples, "direct"),
        )
        if repeat % 2:
            calls = tuple(reversed(calls))
        for call, samples, name in calls:
            start = time.perf_counter_ns()
            value = call()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            if name == "direct":
                direct_score = value
            else:
                projected_score = value

    direct_ms, direct_mad = _median_and_mad(direct_samples)
    projected_ms, projected_mad = _median_and_mad(projected_samples)
    delta_samples = [
        projected_value - direct_value
        for projected_value, direct_value in zip(
            projected_samples,
            direct_samples,
            strict=True,
        )
    ]
    delta_ms, delta_mad = _median_and_mad(delta_samples)

    return {
        "treatment": treatment.name,
        "logical_world_count": belief.logical_world_count,
        "active_canonical_classes": belief.active_canonical_classes,
        "active_projected_classes": projected.active_classes,
        "direct_median_ms": direct_ms,
        "direct_mad_ms": direct_mad,
        "projected_median_ms": projected_ms,
        "projected_mad_ms": projected_mad,
        "projected_minus_direct_median_ms": delta_ms,
        "projected_minus_direct_mad_ms": delta_mad,
        "oracle_path": (
            ExecutionPath.PROJECTED.value
            if delta_ms < 0
            else ExecutionPath.DIRECT.value
        ),
        "score_equal": direct_score == projected_score == direct_warm == projected_warm,
    }


def _nonnegative_least_squares(
    x: np.ndarray,
    y: np.ndarray,
    *,
    row_weights: np.ndarray | None = None,
) -> np.ndarray:
    if row_weights is None:
        row_weights = np.ones(len(y), dtype=np.float64)
    sqrt_weights = np.sqrt(row_weights)

    scale = np.max(np.abs(x), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    scaled = x / scale
    weighted_x = scaled * sqrt_weights[:, None]
    weighted_y = y * sqrt_weights

    columns = x.shape[1]
    best: np.ndarray | None = None
    best_error = float("inf")
    for mask in itertools.product((False, True), repeat=columns):
        if not any(mask):
            candidate_scaled = np.zeros(columns, dtype=np.float64)
        else:
            indices = np.flatnonzero(mask)
            partial, *_ = np.linalg.lstsq(
                weighted_x[:, indices],
                weighted_y,
                rcond=None,
            )
            if np.any(partial < 0):
                continue
            candidate_scaled = np.zeros(columns, dtype=np.float64)
            candidate_scaled[indices] = partial
        residual = weighted_x @ candidate_scaled - weighted_y
        error = float(np.dot(residual, residual))
        if error < best_error:
            best = candidate_scaled / scale
            best_error = error

    if best is None:
        raise RuntimeError("non-negative cost fit found no feasible profile")
    return best


def _noise_weights(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    noise = np.asarray(
        [
            max(float(row["projected_minus_direct_mad_ms"]), 1e-6)
            for row in rows
        ],
        dtype=np.float64,
    )
    floor = max(0.01, float(np.median(noise)) * 0.5)
    effective = np.maximum(noise, floor)
    return 1.0 / effective


def _relative_design(
    rows: Sequence[Mapping[str, object]],
    shape: RelativeModelShape,
) -> np.ndarray:
    columns = []
    for row in rows:
        worlds = float(row["logical_world_count"])
        values = [1.0, -worlds]
        if shape.quadratic_world_term:
            values.append(-(worlds * worlds))
        if shape.canonical_term:
            values.append(float(row["active_canonical_classes"]))
        values.append(float(row["active_projected_classes"]))
        columns.append(values)
    return np.asarray(columns, dtype=np.float64)


def _fit_relative_coefficients(
    rows: Sequence[Mapping[str, object]],
    shape: RelativeModelShape,
) -> tuple[float, float, float, float, float]:
    x = _relative_design(rows, shape)
    y = np.asarray(
        [float(row["projected_minus_direct_median_ms"]) for row in rows],
        dtype=np.float64,
    )
    fitted = list(
        _nonnegative_least_squares(
            x,
            y,
            row_weights=_noise_weights(rows),
        )
    )

    fixed = fitted.pop(0)
    per_world = fitted.pop(0)
    per_world_squared = fitted.pop(0) if shape.quadratic_world_term else 0.0
    per_canonical = fitted.pop(0) if shape.canonical_term else 0.0
    per_execution = fitted.pop(0)
    if fitted:
        raise RuntimeError("relative model coefficient layout mismatch")
    return (
        float(fixed),
        float(per_world),
        float(per_world_squared),
        float(per_canonical),
        float(per_execution),
    )


def _predict_relative(
    coefficients: tuple[float, float, float, float, float],
    row: Mapping[str, object],
) -> float:
    fixed, per_world, per_world_squared, per_canonical, per_execution = coefficients
    worlds = float(row["logical_world_count"])
    return (
        fixed
        - per_world * worlds
        - per_world_squared * worlds * worlds
        + per_canonical * float(row["active_canonical_classes"])
        + per_execution * float(row["active_projected_classes"])
    )


def _leave_one_out_errors(
    rows: Sequence[Mapping[str, object]],
    shape: RelativeModelShape,
) -> list[float]:
    errors = []
    for index, row in enumerate(rows):
        training = [candidate for j, candidate in enumerate(rows) if j != index]
        coefficients = _fit_relative_coefficients(training, shape)
        errors.append(
            abs(
                _predict_relative(coefficients, row)
                - float(row["projected_minus_direct_median_ms"])
            )
        )
    return errors


def _fit_uncertainty_model(
    rows: Sequence[Mapping[str, object]],
    leave_one_out_errors: Sequence[float],
) -> tuple[float, float, float]:
    targets = np.asarray(
        [
            error + float(row["projected_minus_direct_mad_ms"])
            for error, row in zip(leave_one_out_errors, rows, strict=True)
        ],
        dtype=np.float64,
    )
    x = np.asarray(
        [
            [1.0, float(row["logical_world_count"])]
            for row in rows
        ],
        dtype=np.float64,
    )
    fitted = _nonnegative_least_squares(x, targets)
    predicted = x @ fitted
    ratios = np.divide(
        targets,
        np.maximum(predicted, 1e-9),
    )
    safety_scale = max(1.0, float(np.quantile(ratios, 0.90)))
    fitted *= safety_scale
    covered = float(np.mean((x @ fitted) >= targets))
    return float(fitted[0]), float(fitted[1]), covered


def _fit_relative_profile(
    rows: Sequence[Mapping[str, object]],
    *,
    backend: str,
    effect_signature: str,
) -> tuple[ExecutionCostProfile, dict[str, object]]:
    candidates = []
    for shape in MODEL_SHAPES:
        errors = _leave_one_out_errors(rows, shape)
        candidates.append(
            {
                "shape": shape,
                "errors": errors,
                "mae": statistics.fmean(errors),
            }
        )

    best_mae = min(float(candidate["mae"]) for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if float(candidate["mae"]) <= best_mae * 1.05
    ]
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["shape"].complexity,
            candidate["shape"].quadratic_world_term,
            candidate["shape"].canonical_term,
        ),
    )
    shape = selected["shape"]
    errors = selected["errors"]
    coefficients = _fit_relative_coefficients(rows, shape)
    uncertainty_fixed, uncertainty_per_world, uncertainty_coverage = (
        _fit_uncertainty_model(rows, errors)
    )

    fixed, per_world, per_world_squared, per_canonical, per_execution = coefficients
    profile = ExecutionCostProfile(
        backend=backend,
        effect_signature=effect_signature,
        projected_fixed_overhead_ms=fixed,
        direct_per_world_ms=per_world,
        direct_per_world_squared_ms=per_world_squared,
        projected_per_canonical_class_ms=per_canonical,
        projected_per_execution_class_ms=per_execution,
        uncertainty_fixed_ms=uncertainty_fixed,
        uncertainty_per_world_ms=uncertainty_per_world,
        calibrated_max_logical_world_count=max(
            int(row["logical_world_count"]) for row in rows
        ),
        calibrated_max_canonical_classes=max(
            int(row["active_canonical_classes"]) for row in rows
        ),
        calibrated_max_projected_classes=max(
            int(row["active_projected_classes"]) for row in rows
        ),
    )
    return profile, {
        "selected_shape": shape.name,
        "selected_complexity": shape.complexity,
        "leave_one_out_mae_ms": {
            candidate["shape"].name: candidate["mae"]
            for candidate in candidates
        },
        "best_leave_one_out_mae_ms": best_mae,
        "selected_leave_one_out_mae_ms": selected["mae"],
        "uncertainty_training_coverage": uncertainty_coverage,
    }


def _fit_absolute_baseline(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    direct_x = np.asarray(
        [[1.0, float(row["logical_world_count"])] for row in rows],
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
    return {
        "direct_intercept_ms": float(direct[0]),
        "direct_per_world_ms": float(direct[1]),
        "projected_intercept_ms": float(projected[0]),
        "projected_per_canonical_class_ms": float(projected[1]),
        "projected_per_execution_class_ms": float(projected[2]),
    }


def _baseline_delta(
    baseline: Mapping[str, float],
    row: Mapping[str, object],
) -> float:
    direct = (
        baseline["direct_intercept_ms"]
        + baseline["direct_per_world_ms"] * float(row["logical_world_count"])
    )
    projected = (
        baseline["projected_intercept_ms"]
        + baseline["projected_per_canonical_class_ms"]
        * float(row["active_canonical_classes"])
        + baseline["projected_per_execution_class_ms"]
        * float(row["active_projected_classes"])
    )
    return projected - direct


def _evaluate(
    rows: Sequence[Mapping[str, object]],
    *,
    profile: ExecutionCostProfile,
    baseline: Mapping[str, float],
) -> dict[str, object]:
    candidate_records: list[dict[str, object]] = []
    baseline_records: list[dict[str, object]] = []

    oracle_total = 0.0
    direct_total = 0.0
    projected_total = 0.0
    candidate_total = 0.0
    baseline_total = 0.0
    candidate_errors: list[float] = []
    baseline_errors: list[float] = []
    candidate_correct = 0
    baseline_correct = 0
    candidate_worst = 1.0
    baseline_worst = 1.0
    candidate_paths: set[str] = set()
    guarded_cases = 0

    for row in rows:
        measured_delta = float(row["projected_minus_direct_median_ms"])
        direct_ms = float(row["direct_median_ms"])
        projected_ms = float(row["projected_median_ms"])
        oracle_path = str(row["oracle_path"])
        oracle_ms = min(direct_ms, projected_ms)

        features = ExecutionFeatures(
            backend=profile.backend,
            effect_signature=profile.effect_signature,
            logical_world_count=int(row["logical_world_count"]),
            active_canonical_classes=int(row["active_canonical_classes"]),
            active_projected_classes=int(row["active_projected_classes"]),
        )
        decision = choose_execution_path(profile, features)
        candidate_ms = (
            projected_ms
            if decision.path is ExecutionPath.PROJECTED
            else direct_ms
        )
        candidate_ratio = candidate_ms / oracle_ms
        candidate_worst = max(candidate_worst, candidate_ratio)
        candidate_total += candidate_ms
        candidate_paths.add(decision.path.value)
        candidate_correct += int(decision.path.value == oracle_path)
        guarded_cases += int(decision.within_uncertainty_guard)
        candidate_errors.append(
            abs(
                decision.predicted_projected_minus_direct_ms
                - measured_delta
            )
        )
        candidate_records.append(
            {
                **dict(row),
                "chosen_path": decision.path.value,
                "predicted_delta_ms": decision.predicted_projected_minus_direct_ms,
                "uncertainty_guard_ms": decision.uncertainty_guard_ms,
                "within_uncertainty_guard": decision.within_uncertainty_guard,
                "choice_matches_oracle": decision.path.value == oracle_path,
                "chosen_over_oracle": candidate_ratio,
            }
        )

        baseline_delta = _baseline_delta(baseline, row)
        baseline_path = (
            ExecutionPath.PROJECTED
            if baseline_delta < 0
            else ExecutionPath.DIRECT
        )
        baseline_ms = (
            projected_ms
            if baseline_path is ExecutionPath.PROJECTED
            else direct_ms
        )
        baseline_ratio = baseline_ms / oracle_ms
        baseline_worst = max(baseline_worst, baseline_ratio)
        baseline_total += baseline_ms
        baseline_correct += int(baseline_path.value == oracle_path)
        baseline_errors.append(abs(baseline_delta - measured_delta))
        baseline_records.append(
            {
                **dict(row),
                "chosen_path": baseline_path.value,
                "predicted_delta_ms": baseline_delta,
                "choice_matches_oracle": baseline_path.value == oracle_path,
                "chosen_over_oracle": baseline_ratio,
            }
        )

        oracle_total += oracle_ms
        direct_total += direct_ms
        projected_total += projected_ms

    candidate = {
        "rows": candidate_records,
        "choice_accuracy": candidate_correct / len(rows),
        "delta_mae_ms": statistics.fmean(candidate_errors),
        "adaptive_total_ms": candidate_total,
        "adaptive_over_oracle": candidate_total / oracle_total,
        "worst_case_over_oracle": candidate_worst,
        "chosen_paths": sorted(candidate_paths),
        "guarded_case_count": guarded_cases,
        "speedup_vs_always_direct": direct_total / candidate_total,
        "speedup_vs_always_projected": projected_total / candidate_total,
    }
    old = {
        "rows": baseline_records,
        "choice_accuracy": baseline_correct / len(rows),
        "delta_mae_ms": statistics.fmean(baseline_errors),
        "adaptive_total_ms": baseline_total,
        "adaptive_over_oracle": baseline_total / oracle_total,
        "worst_case_over_oracle": baseline_worst,
    }
    return {
        "candidate": candidate,
        "baseline": old,
        "oracle_total_ms": oracle_total,
        "always_direct_total_ms": direct_total,
        "always_projected_total_ms": projected_total,
    }


def _profile_record(profile: ExecutionCostProfile) -> dict[str, object]:
    return {
        "backend": profile.backend,
        "effect_signature": profile.effect_signature,
        "projected_fixed_overhead_ms": profile.projected_fixed_overhead_ms,
        "direct_per_world_ms": profile.direct_per_world_ms,
        "direct_per_world_squared_ms": profile.direct_per_world_squared_ms,
        "projected_per_canonical_class_ms": profile.projected_per_canonical_class_ms,
        "projected_per_execution_class_ms": profile.projected_per_execution_class_ms,
        "uncertainty_fixed_ms": profile.uncertainty_fixed_ms,
        "uncertainty_per_world_ms": profile.uncertainty_per_world_ms,
        "calibrated_max_logical_world_count": profile.calibrated_max_logical_world_count,
        "calibrated_max_canonical_classes": profile.calibrated_max_canonical_classes,
        "calibrated_max_projected_classes": profile.calibrated_max_projected_classes,
    }


def run_experiment(fixtures: Path) -> dict[str, object]:
    contexts = _load_contexts(fixtures)
    backend = jax.default_backend()
    effect_signature = _effect_signature(contexts)

    training = [
        _benchmark_treatment(contexts, treatment)
        for treatment in TRAINING_TREATMENTS
    ]
    profile, selection = _fit_relative_profile(
        training,
        backend=backend,
        effect_signature=effect_signature,
    )
    baseline = _fit_absolute_baseline(training)

    held_out_rows = [
        _benchmark_treatment(contexts, treatment)
        for treatment in HELD_OUT_TREATMENTS
    ]
    evaluation = _evaluate(
        held_out_rows,
        profile=profile,
        baseline=baseline,
    )
    candidate = evaluation["candidate"]
    old = evaluation["baseline"]

    training_names = {row["treatment"] for row in training}
    held_out_names = {row["treatment"] for row in held_out_rows}
    disjoint = training_names.isdisjoint(held_out_names)

    prediction_improvement = (
        candidate["delta_mae_ms"] / old["delta_mae_ms"]
        if old["delta_mae_ms"] > 0
        else 1.0
    )
    regret_not_worse = (
        candidate["adaptive_over_oracle"]
        <= old["adaptive_over_oracle"] + 0.01
    )

    passed = (
        disjoint
        and all(bool(row["score_equal"]) for row in training)
        and all(bool(row["score_equal"]) for row in held_out_rows)
        and candidate["choice_accuracy"] >= 0.75
        and candidate["adaptive_over_oracle"] <= 1.10
        and candidate["worst_case_over_oracle"] <= 1.20
        and candidate["chosen_paths"] == ["direct", "projected"]
        and candidate["speedup_vs_always_direct"] > 1.0
        and candidate["speedup_vs_always_projected"] > 1.0
        and prediction_improvement <= 0.95
        and regret_not_worse
    )

    return {
        "schema": "azelficoast.adaptive-execution-experiment",
        "schema_version": 2,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "jax_version": jax.__version__,
        "backend": backend,
        "effect_signature": effect_signature,
        "training_and_held_out_disjoint": disjoint,
        "training": training,
        "profile": _profile_record(profile),
        "model_selection": selection,
        "absolute_baseline_profile": baseline,
        "held_out": evaluation,
        "candidate_delta_mae_relative_to_baseline": prediction_improvement,
        "candidate_regret_not_worse_than_baseline": regret_not_worse,
        "passed": passed,
        "acceptance": {
            "minimum_choice_accuracy": 0.75,
            "maximum_aggregate_latency_vs_oracle": 1.10,
            "maximum_worst_case_latency_vs_oracle": 1.20,
            "maximum_delta_mae_relative_to_old_model": 0.95,
            "maximum_regret_increase_vs_old_model": 0.01,
            "must_choose_both_paths": True,
            "must_beat_always_direct": True,
            "must_beat_always_projected": True,
        },
        "non_claims": [
            "the profile is specific to this backend and effect signature",
            "the hosted coefficients are calibration evidence, not portable constants",
            "the direct timed path receives pre-materialized device-resident worlds",
            "the projected path pays projection, compaction, representative assembly, and transfer",
            "the uncertainty envelope is diagnostic, not a probabilistic confidence interval",
            "the quadratic term is a bounded empirical saturation approximation",
            "GPU and native-backend crossover behavior remain separately calibratable",
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
