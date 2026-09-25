"""Select and validate a structural adaptive-execution cost model."""

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
    current_jax_execution_target,
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
class CostModelShape:
    direct_quadratic_term: bool
    projected_canonical_term: bool
    direct_saturation_term: bool = False

    def __post_init__(self) -> None:
        if self.direct_quadratic_term and self.direct_saturation_term:
            raise ValueError("direct quadratic and saturation terms are alternative shapes")

    @property
    def name(self) -> str:
        if self.direct_quadratic_term:
            direct = "direct-linear+quadratic"
        elif self.direct_saturation_term:
            direct = "direct-linear+saturation"
        else:
            direct = "direct-linear"
        projected = (
            "projected-canonical+execution"
            if self.projected_canonical_term
            else "projected-execution"
        )
        return f"{direct}__{projected}"

    @property
    def complexity(self) -> int:
        return (
            4
            + int(self.direct_quadratic_term)
            + int(self.direct_saturation_term)
            + int(self.projected_canonical_term)
        )


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
    CostModelShape(False, False),
    CostModelShape(False, True),
    CostModelShape(True, False),
    CostModelShape(True, True),
    CostModelShape(False, False, True),
    CostModelShape(False, True, True),
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


def _execution_target_signature() -> str:
    return current_jax_execution_target()[1]


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


def _noise_weights(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> np.ndarray:
    noise = np.asarray(
        [max(float(row[field]), 1e-6) for row in rows],
        dtype=np.float64,
    )
    floor = max(0.01, float(np.median(noise)) * 0.5)
    return 1.0 / np.maximum(noise, floor)


def _direct_saturation_start_worlds(
    rows: Sequence[Mapping[str, object]],
    shape: CostModelShape,
) -> int:
    if not shape.direct_saturation_term:
        return 0
    worlds = sorted(int(row["logical_world_count"]) for row in rows)
    # The upper median is derived only from calibration features, never timings.
    # It gives the saturation shape a fixed low-work region while allowing a
    # second non-negative slope to absorb high-work cache/backend saturation.
    return worlds[len(worlds) // 2]


def _direct_design(
    rows: Sequence[Mapping[str, object]],
    shape: CostModelShape,
    saturation_start_worlds: int,
) -> np.ndarray:
    values = []
    for row in rows:
        worlds = float(row["logical_world_count"])
        columns = [1.0, worlds]
        if shape.direct_quadratic_term:
            columns.append(worlds * worlds)
        if shape.direct_saturation_term:
            columns.append(max(0.0, worlds - saturation_start_worlds))
        values.append(columns)
    return np.asarray(values, dtype=np.float64)


def _projected_design(
    rows: Sequence[Mapping[str, object]],
    shape: CostModelShape,
) -> np.ndarray:
    values = []
    for row in rows:
        columns = [1.0]
        if shape.projected_canonical_term:
            columns.append(float(row["active_canonical_classes"]))
        columns.append(float(row["active_projected_classes"]))
        values.append(columns)
    return np.asarray(values, dtype=np.float64)


def _fit_structural_coefficients(
    rows: Sequence[Mapping[str, object]],
    shape: CostModelShape,
    *,
    noise_weighted: bool,
) -> tuple[float, float, float, float, float, float, float, float]:
    saturation_start_worlds = _direct_saturation_start_worlds(rows, shape)
    direct = _nonnegative_least_squares(
        _direct_design(rows, shape, saturation_start_worlds),
        np.asarray(
            [float(row["direct_median_ms"]) for row in rows],
            dtype=np.float64,
        ),
        row_weights=(
            _noise_weights(rows, "direct_mad_ms")
            if noise_weighted
            else None
        ),
    )
    projected = _nonnegative_least_squares(
        _projected_design(rows, shape),
        np.asarray(
            [float(row["projected_median_ms"]) for row in rows],
            dtype=np.float64,
        ),
        row_weights=(
            _noise_weights(rows, "projected_mad_ms")
            if noise_weighted
            else None
        ),
    )

    direct_intercept = float(direct[0])
    direct_per_world = float(direct[1])
    direct_index = 2
    direct_quadratic = 0.0
    if shape.direct_quadratic_term:
        direct_quadratic = float(direct[direct_index])
        direct_index += 1
    direct_saturation = 0.0
    if shape.direct_saturation_term:
        direct_saturation = float(direct[direct_index])

    projected_intercept = float(projected[0])
    if shape.projected_canonical_term:
        projected_canonical = float(projected[1])
        projected_execution = float(projected[2])
    else:
        projected_canonical = 0.0
        projected_execution = float(projected[1])
    return (
        direct_intercept,
        direct_per_world,
        direct_quadratic,
        float(saturation_start_worlds),
        direct_saturation,
        projected_intercept,
        projected_canonical,
        projected_execution,
    )


def _predict_costs(
    coefficients: tuple[float, float, float, float, float, float, float, float],
    row: Mapping[str, object],
) -> tuple[float, float]:
    (
        direct_intercept,
        direct_per_world,
        direct_quadratic,
        saturation_start_worlds,
        direct_saturation,
        projected_intercept,
        projected_canonical,
        projected_execution,
    ) = coefficients
    worlds = float(row["logical_world_count"])
    saturated_worlds = max(0.0, worlds - saturation_start_worlds)
    direct = (
        direct_intercept
        + direct_per_world * worlds
        + direct_quadratic * worlds * worlds
        + direct_saturation * saturated_worlds
    )
    projected = (
        projected_intercept
        + projected_canonical * float(row["active_canonical_classes"])
        + projected_execution * float(row["active_projected_classes"])
    )
    return direct, projected


def _leave_one_out_metrics(
    rows: Sequence[Mapping[str, object]],
    shape: CostModelShape,
) -> dict[str, object]:
    errors: list[float] = []
    correct = 0
    for index, row in enumerate(rows):
        training = [candidate for j, candidate in enumerate(rows) if j != index]
        coefficients = _fit_structural_coefficients(
            training,
            shape,
            noise_weighted=True,
        )
        direct, projected = _predict_costs(coefficients, row)
        predicted_delta = projected - direct
        measured_delta = float(row["projected_minus_direct_median_ms"])
        errors.append(abs(predicted_delta - measured_delta))
        correct += int((predicted_delta < 0) == (measured_delta < 0))
    return {
        "errors": errors,
        "mae": statistics.fmean(errors),
        "choice_accuracy": correct / len(rows),
    }


def _fixed_class_crossover_bracket(
    rows: Sequence[Mapping[str, object]],
) -> tuple[int, int, int, int]:
    by_shape: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    for row in rows:
        shape = (
            int(row["active_canonical_classes"]),
            int(row["active_projected_classes"]),
        )
        by_shape.setdefault(shape, []).append(row)

    candidates: list[tuple[int, int, int, int, int]] = []
    for (canonical, projected), shape_rows in by_shape.items():
        direct_worlds = sorted(
            int(row["logical_world_count"])
            for row in shape_rows
            if str(row["oracle_path"]) == ExecutionPath.DIRECT.value
        )
        projected_worlds = sorted(
            int(row["logical_world_count"])
            for row in shape_rows
            if str(row["oracle_path"]) == ExecutionPath.PROJECTED.value
        )
        if not direct_worlds or not projected_worlds:
            continue

        direct_max = max(direct_worlds)
        projected_min = min(projected_worlds)
        if direct_max >= projected_min:
            continue

        candidates.append(
            (
                len(shape_rows),
                canonical,
                projected,
                direct_max,
                projected_min,
            )
        )

    if not candidates:
        return (0, 0, 0, 0)

    _row_count, canonical, projected, direct_max, projected_min = max(
        candidates,
        key=lambda candidate: (
            candidate[0],
            candidate[1],
            candidate[2],
        ),
    )
    return (canonical, projected, direct_max, projected_min)


def _fit_cost_profile(
    rows: Sequence[Mapping[str, object]],
    *,
    backend: str,
    target_signature: str,
    effect_signature: str,
) -> tuple[ExecutionCostProfile, dict[str, object]]:
    candidates = []
    for shape in MODEL_SHAPES:
        metrics = _leave_one_out_metrics(rows, shape)
        candidates.append(
            {
                "shape": shape,
                **metrics,
            }
        )

    best_accuracy = max(
        float(candidate["choice_accuracy"])
        for candidate in candidates
    )
    accuracy_eligible = [
        candidate
        for candidate in candidates
        if float(candidate["choice_accuracy"]) == best_accuracy
    ]
    best_candidate = min(
        accuracy_eligible,
        key=lambda candidate: float(candidate["mae"]),
    )
    best_mae = float(best_candidate["mae"])

    # The dispatcher consumes the sign of the cost difference. Once cross-validated
    # path accuracy ties, extra curve complexity must not be justified merely by
    # fitting large, decision-irrelevant latency magnitudes far from the crossover.
    minimum_complexity = min(
        candidate["shape"].complexity
        for candidate in accuracy_eligible
    )
    simplest = [
        candidate
        for candidate in accuracy_eligible
        if candidate["shape"].complexity == minimum_complexity
    ]
    selected = min(
        simplest,
        key=lambda candidate: (
            float(candidate["mae"]),
            candidate["shape"].direct_quadratic_term,
            candidate["shape"].direct_saturation_term,
            candidate["shape"].projected_canonical_term,
        ),
    )
    shape = selected["shape"]
    coefficients = _fit_structural_coefficients(
        rows,
        shape,
        noise_weighted=True,
    )

    guarded_errors = [
        error + float(row["projected_minus_direct_mad_ms"])
        for error, row in zip(selected["errors"], rows, strict=True)
    ]
    uncertainty_guard = float(np.quantile(guarded_errors, 0.90))

    (
        direct_intercept,
        direct_per_world,
        direct_quadratic,
        direct_saturation_start_worlds,
        direct_saturation,
        projected_intercept,
        projected_canonical,
        projected_execution,
    ) = coefficients
    (
        crossover_canonical,
        crossover_projected,
        crossover_direct_max,
        crossover_projected_min,
    ) = _fixed_class_crossover_bracket(rows)
    profile = ExecutionCostProfile(
        backend=backend,
        target_signature=target_signature,
        effect_signature=effect_signature,
        direct_intercept_ms=direct_intercept,
        direct_per_world_ms=direct_per_world,
        direct_per_world_squared_ms=direct_quadratic,
        projected_intercept_ms=projected_intercept,
        projected_per_canonical_class_ms=projected_canonical,
        projected_per_execution_class_ms=projected_execution,
        uncertainty_guard_ms=uncertainty_guard,
        calibrated_max_logical_world_count=max(
            int(row["logical_world_count"]) for row in rows
        ),
        calibrated_max_canonical_classes=max(
            int(row["active_canonical_classes"]) for row in rows
        ),
        calibrated_max_projected_classes=max(
            int(row["active_projected_classes"]) for row in rows
        ),
        direct_saturation_start_worlds=int(direct_saturation_start_worlds),
        direct_saturation_per_world_ms=direct_saturation,
        crossover_canonical_classes=crossover_canonical,
        crossover_projected_classes=crossover_projected,
        crossover_direct_max_worlds=crossover_direct_max,
        crossover_projected_min_worlds=crossover_projected_min,
    )
    return profile, {
        "selected_shape": shape.name,
        "selected_complexity": shape.complexity,
        "leave_one_out": {
            candidate["shape"].name: {
                "delta_mae_ms": candidate["mae"],
                "choice_accuracy": candidate["choice_accuracy"],
            }
            for candidate in candidates
        },
        "best_choice_accuracy": best_accuracy,
        "best_delta_mae_among_best_accuracy_ms": best_mae,
        "minimum_complexity_at_best_accuracy": minimum_complexity,
        "selection_policy": "choice-accuracy_then_complexity_then-delta-mae",
        "selected_delta_mae_ms": selected["mae"],
        "selected_choice_accuracy": selected["choice_accuracy"],
        "selected_direct_saturation_start_worlds": int(
            direct_saturation_start_worlds
        ),
        "fixed_class_crossover_bracket": {
            "canonical_classes": crossover_canonical,
            "projected_classes": crossover_projected,
            "direct_max_worlds": crossover_direct_max,
            "projected_min_worlds": crossover_projected_min,
        },
        "uncertainty_guard_ms": uncertainty_guard,
    }


def _fit_absolute_baseline(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, float]:
    shape = CostModelShape(
        direct_quadratic_term=False,
        projected_canonical_term=True,
    )
    coefficients = _fit_structural_coefficients(
        rows,
        shape,
        noise_weighted=False,
    )
    (
        direct_intercept,
        direct_per_world,
        _direct_quadratic,
        _direct_saturation_start,
        _direct_saturation,
        projected_intercept,
        projected_canonical,
        projected_execution,
    ) = coefficients
    return {
        "direct_intercept_ms": direct_intercept,
        "direct_per_world_ms": direct_per_world,
        "projected_intercept_ms": projected_intercept,
        "projected_per_canonical_class_ms": projected_canonical,
        "projected_per_execution_class_ms": projected_execution,
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
            target_signature=profile.target_signature,
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
                "predicted_direct_ms": decision.predicted_direct_ms,
                "predicted_projected_ms": decision.predicted_projected_ms,
                "predicted_delta_ms": decision.predicted_projected_minus_direct_ms,
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
        "target_signature": profile.target_signature,
        "effect_signature": profile.effect_signature,
        "direct_intercept_ms": profile.direct_intercept_ms,
        "direct_per_world_ms": profile.direct_per_world_ms,
        "direct_per_world_squared_ms": profile.direct_per_world_squared_ms,
        "direct_saturation_start_worlds": profile.direct_saturation_start_worlds,
        "direct_saturation_per_world_ms": profile.direct_saturation_per_world_ms,
        "crossover_canonical_classes": profile.crossover_canonical_classes,
        "crossover_projected_classes": profile.crossover_projected_classes,
        "crossover_direct_max_worlds": profile.crossover_direct_max_worlds,
        "crossover_projected_min_worlds": profile.crossover_projected_min_worlds,
        "projected_intercept_ms": profile.projected_intercept_ms,
        "projected_per_canonical_class_ms": profile.projected_per_canonical_class_ms,
        "projected_per_execution_class_ms": profile.projected_per_execution_class_ms,
        "uncertainty_guard_ms": profile.uncertainty_guard_ms,
        "calibrated_max_logical_world_count": profile.calibrated_max_logical_world_count,
        "calibrated_max_canonical_classes": profile.calibrated_max_canonical_classes,
        "calibrated_max_projected_classes": profile.calibrated_max_projected_classes,
    }


def run_experiment(fixtures: Path) -> dict[str, object]:
    contexts = _load_contexts(fixtures)
    backend = jax.default_backend()
    target_signature = _execution_target_signature()
    effect_signature = _effect_signature(contexts)

    training = [
        _benchmark_treatment(contexts, treatment)
        for treatment in TRAINING_TREATMENTS
    ]
    profile, selection = _fit_cost_profile(
        training,
        backend=backend,
        target_signature=target_signature,
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
        "schema_version": 4,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "jax_version": jax.__version__,
        "backend": backend,
        "target_signature": target_signature,
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
            "the uncertainty guard is diagnostic, not a probabilistic confidence interval",
            "quadratic and hinge direct terms are alternative bounded saturation approximations",
            "the hinge knee is the upper median calibration world count and is timing-independent",
            "a fixed-class monotone crossover bracket is used only when calibration collapses to one workload dimension",
            "native and GPU backends remain separately calibratable",
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
