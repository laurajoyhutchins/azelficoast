"""Pinned-Showdown ordered-attack, secondary, and adaptive-dispatch experiment."""

from __future__ import annotations

import ctypes
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Mapping

import jax
import numpy as np

import azelficoast.research.mechanics.gen9_attack as gen9_attack
import azelficoast.research.mechanics.gen9_damage as gen9_damage
import azelficoast.research.mechanics.gen9_ordered_attack as gen9_ordered_attack
from azelficoast.research.adaptive_execution_experiment import (
    _evaluate,
    _execution_target_signature,
    _fit_absolute_baseline,
    _fit_cost_profile,
    _profile_record,
)
from azelficoast.research.mechanics.gen9_attack import AttackTransitionContext
from azelficoast.research.mechanics.gen9_ordered_attack import (
    COMPILED_ORDERED_ATTACK_CONTEXT_WIDTH,
    OrderedAttackContext,
    ordered_attack_dependency_signature,
    ordered_attack_transition,
    ordered_attack_transition_numeric,
    compile_ordered_attack_context,
    unpack_ordered_attack_transition,
)
from azelficoast.research.mechanics.jax_gen9_ordered_attack import (
    ordered_attack_score,
    ordered_attack_transition_batch,
    weighted_ordered_attack_score,
)
from azelficoast.research.mechanics.ordered_attack_belief import (
    OrderedAttackBelief,
    OrderedAttackProjection,
    build_ordered_attack_support,
    compile_ordered_attack_projection,
    project_ordered_attack_belief,
    uniform_ordered_attack_belief,
)
from azelficoast.research.mechanics.ordered_attack_compiler import build_ordered_attack_library
from azelficoast.research.verification.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    _context as damage_context_from_mapping,
)

REPEATS = 7
TRAINING_WORLD_COUNTS = (
    2048, 3072, 4096, 6144, 8192, 12288, 24576, 49152, 98304, 196608,
)
# Frozen before hosted execution. These points are disjoint from training and
# concentrated around the crossover suggested by the previous attack transition.
CONFIRMATION_WORLD_COUNTS = (
    2304, 2560, 3584, 5120, 7168, 10240, 18432, 36864, 73728,
)


class OrderedAttackExperimentError(ValueError):
    pass


def _load(path: Path) -> tuple[Mapping[str, object], ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "azelficoast.showdown-ordered-attack-fixtures":
        raise OrderedAttackExperimentError("unexpected ordered-attack fixture schema")
    if document.get("schema_version") != 1:
        raise OrderedAttackExperimentError("unexpected ordered-attack fixture version")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise OrderedAttackExperimentError("wrong pinned Showdown revision")
    fixtures = tuple(document.get("fixtures") or ())
    if len(fixtures) != int(document.get("fixture_count") or 0):
        raise OrderedAttackExperimentError("ordered-attack fixture matrix is incomplete")
    return fixtures


def _context(fixture: Mapping[str, object]) -> OrderedAttackContext:
    before = fixture["before"]
    return OrderedAttackContext(
        attack=AttackTransitionContext(
            damage=damage_context_from_mapping(fixture["context"]),
            accuracy=int(fixture["move_accuracy"]),
            attacker_hp=int(before["attacker_hp"]),
            attacker_max_hp=int(before["attacker_max_hp"]),
            defender_hp=int(before["defender_hp"]),
            move_pp=int(before["move_pp"]),
        ),
        attacker_priority=int(fixture["move_priority"]),
        opponent_priority=int(fixture["opponent_priority"]),
        attacker_speed=int(before["attacker_speed"]),
        opponent_speed=int(before["opponent_speed"]),
        secondary_chance=int(fixture["secondary_chance"]),
        defender_spd_stage=int(before["defender_spd_stage"]),
    )


def _expected_packed(fixture: Mapping[str, object]) -> int:
    after = fixture["after"]
    return gen9_ordered_attack.OrderedAttackTransition(
        defender_hp=int(after["defender_hp"]),
        attacker_hp=int(after["attacker_hp"]),
        move_pp=int(after["move_pp"]),
        defender_spd_stage=int(after["defender_spd_stage"]),
        attacker_acted_first=bool(after["attacker_acted_first"]),
    ).packed


class NativeOrderedAttack:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path))
        pointer = ctypes.POINTER(ctypes.c_int32)
        self.library.az_ordered_attack_batch.argtypes = [
            pointer, pointer, pointer, pointer, pointer, pointer, ctypes.c_int64,
        ]
        self.library.az_ordered_attack_batch.restype = None

    @staticmethod
    def _pointer(values: np.ndarray):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def batch(self, params, order, accuracy, damage, secondary) -> np.ndarray:
        params = np.ascontiguousarray(params, dtype=np.int32)
        order = np.ascontiguousarray(order, dtype=np.int32)
        accuracy = np.ascontiguousarray(accuracy, dtype=np.int32)
        damage = np.ascontiguousarray(damage, dtype=np.int32)
        secondary = np.ascontiguousarray(secondary, dtype=np.int32)
        out = np.empty(len(params), dtype=np.int32)
        self.library.az_ordered_attack_batch(
            self._pointer(params),
            self._pointer(order),
            self._pointer(accuracy),
            self._pointer(damage),
            self._pointer(secondary),
            self._pointer(out),
            len(params),
        )
        return out


def _correctness(fixtures, native: NativeOrderedAttack) -> dict[str, object]:
    contexts = tuple(_context(fixture) for fixture in fixtures)
    params = np.asarray([compile_ordered_attack_context(context) for context in contexts], dtype=np.int32)
    order = np.asarray([fixture["order_tie_roll"] for fixture in fixtures], dtype=np.int32)
    accuracy = np.asarray([fixture["accuracy_roll"] for fixture in fixtures], dtype=np.int32)
    damage = np.asarray([fixture["damage_roll"] for fixture in fixtures], dtype=np.int32)
    secondary = np.asarray([fixture["secondary_roll"] for fixture in fixtures], dtype=np.int32)
    expected = np.asarray([_expected_packed(fixture) for fixture in fixtures], dtype=np.int32)

    python = np.asarray([
        ordered_attack_transition_numeric(
            tuple(int(value) for value in params[index]),
            int(order[index]),
            int(accuracy[index]),
            int(damage[index]),
            int(secondary[index]),
        )
        for index in range(len(fixtures))
    ], dtype=np.int32)
    native_actual = native.batch(params, order, accuracy, damage, secondary)
    jax_actual = np.asarray(ordered_attack_transition_batch(
        jax.device_put(params),
        jax.device_put(order),
        jax.device_put(accuracy),
        jax.device_put(damage),
        jax.device_put(secondary),
    ), dtype=np.int32)

    mismatches = []
    for index, fixture in enumerate(fixtures):
        if int(expected[index]) == int(python[index]):
            continue
        mismatches.append({
            "case": fixture["case"],
            "move": fixture["move"],
            "item": fixture["item"],
            "order_tie_roll": int(fixture["order_tie_roll"]),
            "damage_roll": int(fixture["damage_roll"]),
            "secondary_roll": int(fixture["secondary_roll"]),
            "expected": _state_dict(unpack_ordered_attack_transition(int(expected[index]))),
            "actual": _state_dict(unpack_ordered_attack_transition(int(python[index]))),
            "rng_requests": fixture["rng_requests"],
            "transition_log": fixture["transition_log"],
        })
        if len(mismatches) >= 12:
            break

    return {
        "fixture_count": len(fixtures),
        "python_exact": bool(np.array_equal(python, expected)),
        "native_exact": bool(np.array_equal(native_actual, expected)),
        "jax_exact": bool(np.array_equal(jax_actual, expected)),
        "backends_equal": bool(np.array_equal(python, native_actual) and np.array_equal(python, jax_actual)),
        "mismatch_examples": mismatches,
    }


def _state_dict(state) -> dict[str, object]:
    return {
        "defender_hp": state.defender_hp,
        "attacker_hp": state.attacker_hp,
        "move_pp": state.move_pp,
        "defender_spd_stage": state.defender_spd_stage,
        "attacker_acted_first": state.attacker_acted_first,
    }


def _semantic_evidence(fixtures) -> dict[str, object]:
    by_case = {}
    for case in ("speed-fast", "speed-slow", "speed-tie", "priority"):
        rows = [fixture for fixture in fixtures if fixture["case"] == case]
        by_case[case] = sorted({bool(row["after"]["attacker_acted_first"]) for row in rows})

    shadow = [fixture for fixture in fixtures if fixture["move_id"] == "shadowball"]
    secondary_applies = all(
        int(row["after"]["defender_spd_stage"]) == -1
        for row in shadow
        if int(row["secondary_roll"]) < 20
    )
    secondary_misses = all(
        int(row["after"]["defender_spd_stage"]) == 0
        for row in shadow
        if int(row["secondary_roll"]) >= 20
    )
    return {
        "order_outcomes": by_case,
        "fast_first": by_case["speed-fast"] == [True],
        "slow_second": by_case["speed-slow"] == [False],
        "tie_both_orders": by_case["speed-tie"] == [False, True],
        "priority_overrides_speed": by_case["priority"] == [True],
        "secondary_under_20_applies": secondary_applies,
        "secondary_at_or_over_20_does_not_apply": secondary_misses,
    }


def _benchmark_contexts(fixtures) -> tuple[OrderedAttackContext, ...]:
    selected = {}
    for fixture in fixtures:
        if (
            fixture["case"] == "speed-tie"
            and int(fixture["bench_signature"]) == 0
            and int(fixture["damage_roll"]) == 0
            and int(fixture["secondary_roll"]) == 0
        ):
            selected.setdefault(str(fixture["item"]), _context(fixture))
    required = {"None", "Choice Specs", "Life Orb"}
    if set(selected) != required:
        raise OrderedAttackExperimentError("benchmark contexts lack item matrix")
    return tuple(selected[item] for item in ("None", "Choice Specs", "Life Orb"))


def _projection_evidence(contexts) -> tuple[dict[str, object], object, OrderedAttackProjection]:
    support = build_ordered_attack_support(
        len(contexts),
        bench_variants=4,
        order_tie_rolls=2,
        accuracy_rolls=1,
        damage_rolls=16,
        secondary_rolls=21,
    )
    good = compile_ordered_attack_projection(support, contexts)
    missing_order = compile_ordered_attack_projection(
        support, contexts, include_order_tie_roll=False
    )
    missing_secondary = compile_ordered_attack_projection(
        support, contexts, include_secondary_roll=False
    )
    belief = uniform_ordered_attack_belief(support, support.class_count)

    def score(projection: OrderedAttackProjection) -> int:
        projected = project_ordered_attack_belief(belief, projection)
        total = 0
        for class_id, representative in enumerate(projection.representative_indices):
            weight = int(projected.weights[class_id])
            if not weight:
                continue
            context = contexts[int(support.context_index[representative])]
            total += ordered_attack_transition(
                context,
                order_tie_roll=int(support.order_tie_roll[representative]),
                accuracy_roll=int(support.accuracy_roll[representative]),
                damage_roll=int(support.damage_roll[representative]),
                secondary_roll=int(support.secondary_roll[representative]),
            ).packed * weight
        return total

    direct = sum(
        ordered_attack_transition(
            contexts[int(context_index)],
            order_tie_roll=int(order_roll),
            accuracy_roll=int(accuracy_roll),
            damage_roll=int(damage_roll),
            secondary_roll=int(secondary_roll),
        ).packed
        for context_index, order_roll, accuracy_roll, damage_roll, secondary_roll in zip(
            support.context_index,
            support.order_tie_roll,
            support.accuracy_roll,
            support.damage_roll,
            support.secondary_roll,
            strict=True,
        )
    )
    evidence = {
        "canonical_classes": support.class_count,
        "execution_classes": good.class_count,
        "effect_signature": good.effect_signature,
        "signature_matches_transition": good.effect_signature == ordered_attack_dependency_signature(),
        "direct_equals_projected": direct == score(good),
        "missing_order_tie_changes_result": direct != score(missing_order),
        "missing_secondary_roll_changes_result": direct != score(missing_secondary),
        "missing_order_classes": missing_order.class_count,
        "missing_secondary_classes": missing_secondary.class_count,
    }
    return evidence, support, good


def _benchmark_support(contexts):
    return build_ordered_attack_support(
        len(contexts),
        bench_variants=1,
        order_tie_rolls=2,
        accuracy_rolls=1,
        damage_rolls=16,
        secondary_rolls=21,
    )


def _materialize_direct(belief: OrderedAttackBelief, contexts):
    support = belief.support
    context_index = np.repeat(support.context_index, belief.weights)
    rows = np.asarray([compile_ordered_attack_context(context) for context in contexts], dtype=np.int32)
    return (
        rows[context_index],
        np.repeat(support.order_tie_roll, belief.weights).astype(np.int32, copy=False),
        np.repeat(support.accuracy_roll, belief.weights).astype(np.int32, copy=False),
        np.repeat(support.damage_roll, belief.weights).astype(np.int32, copy=False),
        np.repeat(support.secondary_roll, belief.weights).astype(np.int32, copy=False),
    )


def _projected_inputs(belief, projection, contexts):
    projected = project_ordered_attack_belief(belief, projection)
    active = projected.weights > 0
    representatives = projection.representative_indices[active]
    rows = np.asarray([compile_ordered_attack_context(context) for context in contexts], dtype=np.int32)
    return (
        rows[belief.support.context_index[representatives]],
        belief.support.order_tie_roll[representatives].astype(np.int32, copy=False),
        belief.support.accuracy_roll[representatives].astype(np.int32, copy=False),
        belief.support.damage_roll[representatives].astype(np.int32, copy=False),
        belief.support.secondary_roll[representatives].astype(np.int32, copy=False),
        projected.weights[active].astype(np.int32, copy=False),
    )


def _median_and_mad(samples):
    median = statistics.median(samples)
    return median, statistics.median(abs(value - median) for value in samples)


def _benchmark_row(contexts, support, projection, world_count):
    belief = uniform_ordered_attack_belief(support, world_count)
    projected = project_ordered_attack_belief(belief, projection)
    direct = _materialize_direct(belief, contexts)
    direct_device = tuple(jax.device_put(value) for value in direct)

    def direct_call():
        value = ordered_attack_score(*direct_device)
        value.block_until_ready()
        return int(np.asarray(value))

    def projected_call():
        values = _projected_inputs(belief, projection, contexts)
        *inputs, weights = values
        value = weighted_ordered_attack_score(
            *(jax.device_put(v) for v in inputs),
            jax.device_put(weights),
        )
        value.block_until_ready()
        return int(np.asarray(value))

    direct_warm = direct_call()
    projected_warm = projected_call()
    direct_samples = []
    projected_samples = []
    direct_score = direct_warm
    projected_score = projected_warm
    for repeat in range(REPEATS):
        calls = (
            (projected_call, projected_samples, "projected"),
            (direct_call, direct_samples, "direct"),
        )
        if repeat % 2:
            calls = tuple(reversed(calls))
        for call, samples, name in calls:
            start = time.perf_counter_ns()
            result = call()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            if name == "direct":
                direct_score = result
            else:
                projected_score = result

    direct_ms, direct_mad = _median_and_mad(direct_samples)
    projected_ms, projected_mad = _median_and_mad(projected_samples)
    deltas = [p - d for p, d in zip(projected_samples, direct_samples, strict=True)]
    delta_ms, delta_mad = _median_and_mad(deltas)
    return {
        "treatment": f"worlds-{world_count}",
        "logical_world_count": belief.logical_world_count,
        "active_canonical_classes": belief.active_canonical_classes,
        "active_projected_classes": projected.active_classes,
        "direct_median_ms": direct_ms,
        "direct_mad_ms": direct_mad,
        "projected_median_ms": projected_ms,
        "projected_mad_ms": projected_mad,
        "projected_minus_direct_median_ms": delta_ms,
        "projected_minus_direct_mad_ms": delta_mad,
        "oracle_path": "projected" if delta_ms < 0 else "direct",
        "score_equal": direct_score == projected_score == direct_warm == projected_warm,
    }


def run_experiment(fixtures_path: Path) -> dict[str, object]:
    fixtures = _load(fixtures_path)
    contexts = _benchmark_contexts(fixtures)
    effect_signature = ordered_attack_dependency_signature()
    target_signature = _execution_target_signature()
    backend = jax.default_backend()

    with tempfile.TemporaryDirectory(prefix="azelficoast-ordered-attack-") as temp_dir:
        native_build = build_ordered_attack_library(
            Path(gen9_damage.__file__),
            Path(gen9_attack.__file__),
            Path(gen9_ordered_attack.__file__),
            Path(temp_dir) / "ordered_attack.so",
            context_width=COMPILED_ORDERED_ATTACK_CONTEXT_WIDTH,
        )
        correctness = _correctness(fixtures, NativeOrderedAttack(native_build.library))

    semantics = _semantic_evidence(fixtures)
    projection, _, _ = _projection_evidence(contexts)
    support = _benchmark_support(contexts)
    execution_projection = compile_ordered_attack_projection(support, contexts)

    training = [
        _benchmark_row(contexts, support, execution_projection, count)
        for count in TRAINING_WORLD_COUNTS
    ]
    profile, selection = _fit_cost_profile(
        training,
        backend=backend,
        target_signature=target_signature,
        effect_signature=effect_signature,
    )
    baseline = _fit_absolute_baseline(training)
    confirmation_rows = [
        _benchmark_row(contexts, support, execution_projection, count)
        for count in CONFIRMATION_WORLD_COUNTS
    ]
    evaluation = _evaluate(confirmation_rows, profile=profile, baseline=baseline)
    candidate = evaluation["candidate"]
    disjoint = set(TRAINING_WORLD_COUNTS).isdisjoint(CONFIRMATION_WORLD_COUNTS)

    passed = (
        correctness["python_exact"]
        and correctness["native_exact"]
        and correctness["jax_exact"]
        and correctness["backends_equal"]
        and semantics["fast_first"]
        and semantics["slow_second"]
        and semantics["tie_both_orders"]
        and semantics["priority_overrides_speed"]
        and semantics["secondary_under_20_applies"]
        and semantics["secondary_at_or_over_20_does_not_apply"]
        and projection["signature_matches_transition"]
        and projection["direct_equals_projected"]
        and projection["missing_order_tie_changes_result"]
        and projection["missing_secondary_roll_changes_result"]
        and disjoint
        and all(bool(row["score_equal"]) for row in training)
        and all(bool(row["score_equal"]) for row in candidate["rows"])
        and candidate["choice_accuracy"] >= 0.75
        and candidate["adaptive_over_oracle"] <= 1.10
        and candidate["worst_case_over_oracle"] <= 1.20
    )

    return {
        "schema": "azelficoast.ordered-attack-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "backend": backend,
        "jax_version": jax.__version__,
        "target_signature": target_signature,
        "effect_signature": effect_signature,
        "generated_c_bytes": len(native_build.c_source.encode("utf-8")),
        "correctness": correctness,
        "semantics": semantics,
        "projection": projection,
        "training_and_confirmation_disjoint": disjoint,
        "training": training,
        "profile": _profile_record(profile),
        "model_selection": selection,
        "confirmation": evaluation,
        "passed": passed,
        "non_claims": [
            "the opponent action is a no-op; this is not yet a complete two-action turn scheduler",
            "the only modeled secondary is a bounded one-stage SpD drop",
            "accuracy and damage semantics are inherited from the previously validated whole-attack transition",
            "hosted timing is execution-target-specific CPU evidence",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
