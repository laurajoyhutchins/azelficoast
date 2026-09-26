"""Pinned-Showdown experiment for the bounded two-attack turn."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

import jax
import numpy as np

import azelficoast.research.mechanics.gen9_damage as gen9_damage
import azelficoast.research.mechanics.gen9_two_attack_turn as gen9_two_attack_turn
from azelficoast.core.costing import ExecutionPath
from azelficoast.research.experiments.adaptive_execution_experiment import (
    _evaluate,
    _execution_target_signature,
    _fit_absolute_baseline,
    _fit_cost_profile,
    _profile_record,
)
from azelficoast.research.mechanics.gen9_attack import AttackTransitionContext
from azelficoast.research.mechanics.gen9_two_attack_turn import (
    COMPILED_TWO_ATTACK_TURN_CONTEXT_WIDTH,
    P1_ACTION_ATTACK,
    P1_ACTION_PROTECT,
    TwoAttackTurn,
    TwoAttackTurnContext,
    compile_two_attack_turn_context,
    two_attack_turn_dependency_signature,
    two_attack_turn_numeric,
    unpack_two_attack_turn,
)
from azelficoast.research.mechanics.jax_gen9_two_attack_turn import (
    two_attack_turn_batch,
    two_attack_turn_score,
    weighted_two_attack_turn_score,
)
from azelficoast.research.verification.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    _context as damage_context_from_mapping,
)
from azelficoast.research.mechanics.two_attack_turn_belief import (
    TwoAttackTurnBelief,
    TwoAttackTurnProjection,
    build_two_attack_turn_support,
    compile_two_attack_turn_projection,
    project_two_attack_turn_belief,
    uniform_two_attack_turn_belief,
)
from azelficoast.research.mechanics.two_attack_turn_compiler import build_two_attack_turn_library
from azelficoast.research.mechanics.two_attack_turn_execution import execute_two_attack_turn_jax

REPEATS = 7
TRAINING_WORLD_COUNTS = (
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    24576,
    49152,
    98304,
    196608,
)
CONFIRMATION_WORLD_COUNTS = (
    2304,
    2560,
    3584,
    5120,
    7168,
    10240,
    18432,
    36864,
    73728,
)


class TwoAttackTurnExperimentError(ValueError):
    pass


def _load(path: Path) -> tuple[Mapping[str, object], ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "azelficoast.showdown-two-attack-turn-fixtures":
        raise TwoAttackTurnExperimentError("unexpected two-attack fixture schema")
    if document.get("schema_version") != 1:
        raise TwoAttackTurnExperimentError("unexpected two-attack fixture version")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise TwoAttackTurnExperimentError("wrong pinned Showdown revision")
    fixtures = tuple(document.get("fixtures") or ())
    if len(fixtures) != int(document.get("fixture_count") or 0):
        raise TwoAttackTurnExperimentError("two-attack fixture matrix is incomplete")
    return fixtures


def _context(fixture: Mapping[str, object]) -> TwoAttackTurnContext:
    before = fixture["before"]
    p1_damage_raw = fixture["p1_context"]
    if not isinstance(p1_damage_raw, Mapping):
        raise TwoAttackTurnExperimentError("fixture lacks p1 damage context")
    if fixture.get("p1_action_kind") == "protect":
        # Protect has no damage semantics, but the shared compiled context keeps a
        # fixed-width attack slot. Preserve Showdown-derived numeric fields while
        # normalizing only the parser-discriminant category to an inert admitted
        # damage shape. The Protect execution/dependency branches never read it.
        p1_damage_raw = {
            **p1_damage_raw,
            "category": "Special",
            "base_power": 0,
        }

    return TwoAttackTurnContext(
        p1_attack=AttackTransitionContext(
            damage=damage_context_from_mapping(p1_damage_raw),
            accuracy=int(fixture["p1_accuracy"]),
            attacker_hp=int(before["p1_hp"]),
            attacker_max_hp=int(before["p1_max_hp"]),
            defender_hp=int(before["p2_hp"]),
            move_pp=int(before["p1_pp"]),
        ),
        p2_attack=AttackTransitionContext(
            damage=damage_context_from_mapping(fixture["p2_context"]),
            accuracy=int(fixture["p2_accuracy"]),
            attacker_hp=int(before["p2_hp"]),
            attacker_max_hp=int(before["p2_max_hp"]),
            defender_hp=int(before["p1_hp"]),
            move_pp=int(before["p2_pp"]),
        ),
        p1_priority=int(fixture["p1_priority"]),
        p2_priority=int(fixture["p2_priority"]),
        p1_speed=int(before["p1_speed"]),
        p2_speed=int(before["p2_speed"]),
        p1_spa_drop_chance=int(fixture["p1_secondary_chance"]),
        p2_spa_stage=int(before["p2_spa_stage"]),
        p1_action_kind=(
            P1_ACTION_PROTECT
            if fixture.get("p1_action_kind") == "protect"
            else P1_ACTION_ATTACK
        ),
    )


def _expected_packed(fixture: Mapping[str, object]) -> int:
    after = fixture["after"]
    return TwoAttackTurn(
        p1_hp=int(after["p1_hp"]),
        p2_hp=int(after["p2_hp"]),
        p1_pp=int(after["p1_pp"]),
        p2_pp=int(after["p2_pp"]),
        p2_spa_stage=int(after["p2_spa_stage"]),
        p1_acted=bool(after["p1_acted"]),
        p2_acted=bool(after["p2_acted"]),
    ).packed


class NativeTwoAttackTurn:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path))
        pointer = ctypes.POINTER(ctypes.c_int32)
        self.library.az_two_attack_turn_batch.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
        ]
        self.library.az_two_attack_turn_batch.restype = None

    @staticmethod
    def _pointer(values: np.ndarray):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def batch(
        self,
        params,
        order,
        p1_accuracy,
        p1_damage,
        p1_secondary,
        p2_accuracy,
        p2_damage,
    ) -> np.ndarray:
        params = np.ascontiguousarray(params, dtype=np.int32)
        order = np.ascontiguousarray(order, dtype=np.int32)
        p1_accuracy = np.ascontiguousarray(p1_accuracy, dtype=np.int32)
        p1_damage = np.ascontiguousarray(p1_damage, dtype=np.int32)
        p1_secondary = np.ascontiguousarray(p1_secondary, dtype=np.int32)
        p2_accuracy = np.ascontiguousarray(p2_accuracy, dtype=np.int32)
        p2_damage = np.ascontiguousarray(p2_damage, dtype=np.int32)
        out = np.empty(len(params), dtype=np.int32)
        self.library.az_two_attack_turn_batch(
            self._pointer(params),
            self._pointer(order),
            self._pointer(p1_accuracy),
            self._pointer(p1_damage),
            self._pointer(p1_secondary),
            self._pointer(p2_accuracy),
            self._pointer(p2_damage),
            self._pointer(out),
            len(params),
        )
        return out


def _state_dict(state: TwoAttackTurn) -> dict[str, object]:
    return {
        "p1_hp": state.p1_hp,
        "p2_hp": state.p2_hp,
        "p1_pp": state.p1_pp,
        "p2_pp": state.p2_pp,
        "p2_spa_stage": state.p2_spa_stage,
        "p1_acted": state.p1_acted,
        "p2_acted": state.p2_acted,
    }


def _correctness(
    fixtures: Sequence[Mapping[str, object]],
    native: NativeTwoAttackTurn,
) -> dict[str, object]:
    contexts = tuple(_context(fixture) for fixture in fixtures)
    params = np.asarray(
        [compile_two_attack_turn_context(context) for context in contexts],
        dtype=np.int32,
    )
    order = np.asarray([fixture["order_tie_roll"] for fixture in fixtures], dtype=np.int32)
    p1_accuracy = np.asarray(
        [fixture["p1_accuracy_roll"] for fixture in fixtures],
        dtype=np.int32,
    )
    p1_damage = np.asarray(
        [fixture["p1_damage_roll"] for fixture in fixtures],
        dtype=np.int32,
    )
    p1_secondary = np.asarray(
        [fixture["p1_secondary_roll"] for fixture in fixtures],
        dtype=np.int32,
    )
    p2_accuracy = np.asarray(
        [fixture["p2_accuracy_roll"] for fixture in fixtures],
        dtype=np.int32,
    )
    p2_damage = np.asarray(
        [fixture["p2_damage_roll"] for fixture in fixtures],
        dtype=np.int32,
    )
    expected = np.asarray([_expected_packed(fixture) for fixture in fixtures], dtype=np.int32)

    python = np.asarray(
        [
            two_attack_turn_numeric(
                tuple(int(value) for value in params[index]),
                int(order[index]),
                int(p1_accuracy[index]),
                int(p1_damage[index]),
                int(p1_secondary[index]),
                int(p2_accuracy[index]),
                int(p2_damage[index]),
            )
            for index in range(len(fixtures))
        ],
        dtype=np.int32,
    )
    native_actual = native.batch(
        params,
        order,
        p1_accuracy,
        p1_damage,
        p1_secondary,
        p2_accuracy,
        p2_damage,
    )
    jax_actual = np.asarray(
        two_attack_turn_batch(
            jax.device_put(params),
            jax.device_put(order),
            jax.device_put(p1_accuracy),
            jax.device_put(p1_damage),
            jax.device_put(p1_secondary),
            jax.device_put(p2_accuracy),
            jax.device_put(p2_damage),
        ),
        dtype=np.int32,
    )

    mismatches = []
    for index, fixture in enumerate(fixtures):
        if int(expected[index]) == int(python[index]):
            continue
        mismatches.append(
            {
                "order_case": fixture["order_case"],
                "hp_case": fixture["hp_case"],
                "p2_item": fixture["p2_item"],
                "order_tie_roll": int(fixture["order_tie_roll"]),
                "p1_damage_roll": int(fixture["p1_damage_roll"]),
                "p1_secondary_roll": int(fixture["p1_secondary_roll"]),
                "p2_damage_roll": int(fixture["p2_damage_roll"]),
                "expected": _state_dict(unpack_two_attack_turn(int(expected[index]))),
                "actual": _state_dict(unpack_two_attack_turn(int(python[index]))),
                "rng_requests": fixture["rng_requests"],
                "transition_log": fixture["transition_log"],
            }
        )
        if len(mismatches) >= 12:
            break

    return {
        "fixture_count": len(fixtures),
        "python_exact": bool(np.array_equal(python, expected)),
        "native_exact": bool(np.array_equal(native_actual, expected)),
        "jax_exact": bool(np.array_equal(jax_actual, expected)),
        "backends_equal": bool(
            np.array_equal(python, native_actual)
            and np.array_equal(python, jax_actual)
        ),
        "mismatch_examples": mismatches,
    }


def _fixture(
    fixtures: Sequence[Mapping[str, object]],
    *,
    order_case: str,
    hp_case: str = "full",
    p2_item: str = "Choice Specs",
    tie_roll: int = 0,
    p1_damage_roll: int = 0,
    p2_damage_roll: int = 0,
    secondary_roll: int = 0,
) -> Mapping[str, object]:
    return next(
        fixture
        for fixture in fixtures
        if fixture["order_case"] == order_case
        and fixture["hp_case"] == hp_case
        and fixture["p2_item"] == p2_item
        and int(fixture["bench_signature"]) == 0
        and int(fixture["order_tie_roll"]) == tie_roll
        and int(fixture["p1_damage_roll"]) == p1_damage_roll
        and int(fixture["p2_damage_roll"]) == p2_damage_roll
        and int(fixture["p1_secondary_roll"]) == secondary_roll
    )


def _semantic_evidence(fixtures: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_order = {}
    for case in ("p1-fast", "p2-fast", "speed-tie", "p1-priority", "protect"):
        rows = [
            fixture
            for fixture in fixtures
            if fixture["order_case"] == case and fixture["hp_case"] == "full"
        ]
        by_order[case] = sorted({str(row["after"]["first_actor"]) for row in rows})

    p1_ko_rows = [
        fixture
        for fixture in fixtures
        if fixture["order_case"] == "p1-fast" and fixture["hp_case"] == "p2-low"
    ]
    p2_ko_rows = [
        fixture
        for fixture in fixtures
        if fixture["order_case"] == "p2-fast" and fixture["hp_case"] == "p1-low"
    ]

    before_drop = _fixture(fixtures, order_case="p1-fast", secondary_roll=0)
    before_no_drop = _fixture(fixtures, order_case="p1-fast", secondary_roll=30)
    after_drop = _fixture(fixtures, order_case="p2-fast", secondary_roll=0)
    after_no_drop = _fixture(fixtures, order_case="p2-fast", secondary_roll=30)
    protect_rows = [
        fixture
        for fixture in fixtures
        if fixture["order_case"] == "protect" and fixture["hp_case"] == "full"
    ]
    protect_outcomes = {
        (
            int(row["after"]["p1_hp"]),
            int(row["after"]["p2_hp"]),
            int(row["after"]["p1_pp"]),
            int(row["after"]["p2_pp"]),
            int(row["after"]["p2_spa_stage"]),
            bool(row["after"]["p1_acted"]),
            bool(row["after"]["p2_acted"]),
        )
        for row in protect_rows
    }
    protect_damage_rng_absent = all(
        not any(request.get("from") == 16 for request in row["rng_requests"])
        for row in protect_rows
    )

    return {
        "first_actors": by_order,
        "p1_fast_first": by_order["p1-fast"] == ["p1"],
        "p2_fast_first": by_order["p2-fast"] == ["p2"],
        "tie_both_orders": by_order["speed-tie"] == ["p1", "p2"],
        "priority_overrides_speed": by_order["p1-priority"] == ["p1"],
        "protect_priority_first": by_order["protect"] == ["p1"],
        "protect_blocks_opposing_damage": bool(protect_rows)
        and all(
            int(row["after"]["p1_hp"]) == int(row["before"]["p1_hp"])
            and int(row["after"]["p2_hp"]) == int(row["before"]["p2_hp"])
            and int(row["after"]["p1_pp"]) == 4
            and int(row["after"]["p2_pp"]) == 4
            and bool(row["after"]["p1_acted"])
            and bool(row["after"]["p2_acted"])
            for row in protect_rows
        ),
        "protect_hidden_item_and_rng_invariant": (
            len(protect_outcomes) == 1 and protect_damage_rng_absent
        ),
        "p1_ko_cancels_p2": bool(p1_ko_rows)
        and all(
            bool(row["after"]["p1_acted"])
            and not bool(row["after"]["p2_acted"])
            and int(row["after"]["p2_pp"]) == 5
            for row in p1_ko_rows
        ),
        "p2_ko_cancels_p1": bool(p2_ko_rows)
        and all(
            bool(row["after"]["p2_acted"])
            and not bool(row["after"]["p1_acted"])
            and int(row["after"]["p1_pp"]) == 5
            for row in p2_ko_rows
        ),
        "pre_attack_drop_changes_later_damage": (
            int(before_drop["after"]["p2_spa_stage"]) == -1
            and int(before_no_drop["after"]["p2_spa_stage"]) == 0
            and int(before_drop["after"]["p1_hp"]) > int(before_no_drop["after"]["p1_hp"])
        ),
        "post_attack_drop_not_retroactive": (
            int(after_drop["after"]["p2_spa_stage"]) == -1
            and int(after_no_drop["after"]["p2_spa_stage"]) == 0
            and int(after_drop["after"]["p1_hp"]) == int(after_no_drop["after"]["p1_hp"])
        ),
    }


def _protect_projection_evidence(
    fixtures: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    contexts_by_item = {}
    for fixture in fixtures:
        if (
            fixture["order_case"] == "protect"
            and fixture["hp_case"] == "full"
            and int(fixture["bench_signature"]) == 0
            and int(fixture["order_tie_roll"]) == 0
            and int(fixture["p1_damage_roll"]) == 0
            and int(fixture["p2_damage_roll"]) == 0
            and int(fixture["p1_secondary_roll"]) == 99
        ):
            contexts_by_item.setdefault(str(fixture["p2_item"]), _context(fixture))

    required = {"None", "Choice Specs"}
    if set(contexts_by_item) != required:
        raise TwoAttackTurnExperimentError(
            "Protect projection treatment lacks p2 item matrix"
        )
    contexts = tuple(
        contexts_by_item[item] for item in ("None", "Choice Specs")
    )
    support = build_two_attack_turn_support(
        len(contexts),
        bench_variants=2,
        order_tie_rolls=2,
        p1_damage_rolls=4,
        p1_secondary_rolls=31,
        p2_damage_rolls=4,
    )
    projection = compile_two_attack_turn_projection(support, contexts)
    belief = uniform_two_attack_turn_belief(support, support.class_count)
    projected = project_two_attack_turn_belief(belief, projection)

    params = [
        compile_two_attack_turn_context(context)
        for context in contexts
    ]
    direct = [
        two_attack_turn_numeric(
            params[int(context_index)],
            int(order_roll),
            0,
            int(p1_damage_roll),
            int(secondary_roll),
            0,
            int(p2_damage_roll),
        )
        for context_index, order_roll, p1_damage_roll, secondary_roll, p2_damage_roll in zip(
            support.context_index,
            support.order_tie_roll,
            support.p1_damage_roll,
            support.p1_secondary_roll,
            support.p2_damage_roll,
            strict=True,
        )
    ]
    representative_outputs = [
        two_attack_turn_numeric(
            params[int(support.context_index[representative])],
            int(support.order_tie_roll[representative]),
            0,
            int(support.p1_damage_roll[representative]),
            int(support.p1_secondary_roll[representative]),
            0,
            int(support.p2_damage_roll[representative]),
        )
        for representative in projection.representative_indices
    ]
    expanded = [
        representative_outputs[int(class_id)]
        for class_id in projection.class_ids
    ]

    return {
        "canonical_classes": support.class_count,
        "execution_classes": projection.class_count,
        "active_execution_classes": projected.active_classes,
        "effect_signature": projection.effect_signature,
        "direct_equals_projected": direct == expanded,
        "opponent_item_collapsed": projection.class_count == 1,
        "opponent_damage_rng_collapsed": projection.class_count == 1,
        "reduction_factor": support.class_count / projection.class_count,
    }


def _benchmark_contexts(
    fixtures: Sequence[Mapping[str, object]],
) -> tuple[TwoAttackTurnContext, ...]:
    selected = {}
    for fixture in fixtures:
        if (
            fixture["order_case"] == "speed-tie"
            and fixture["hp_case"] == "full"
            and int(fixture["bench_signature"]) == 0
            and int(fixture["order_tie_roll"]) == 0
            and int(fixture["p1_damage_roll"]) == 0
            and int(fixture["p2_damage_roll"]) == 0
            and int(fixture["p1_secondary_roll"]) == 0
        ):
            selected.setdefault(str(fixture["p2_item"]), _context(fixture))
    required = {"None", "Choice Specs"}
    if set(selected) != required:
        raise TwoAttackTurnExperimentError("benchmark contexts lack p2 item matrix")
    return tuple(selected[item] for item in ("None", "Choice Specs"))


def _projection_evidence(
    contexts: Sequence[TwoAttackTurnContext],
) -> tuple[dict[str, object], object, TwoAttackTurnProjection]:
    support = build_two_attack_turn_support(
        len(contexts),
        bench_variants=2,
        order_tie_rolls=2,
        p1_damage_rolls=4,
        p1_secondary_rolls=31,
        p2_damage_rolls=4,
    )
    good = compile_two_attack_turn_projection(support, contexts)
    missing_order = compile_two_attack_turn_projection(
        support,
        contexts,
        include_order_tie=False,
    )
    missing_secondary = compile_two_attack_turn_projection(
        support,
        contexts,
        include_secondary=False,
    )
    belief = uniform_two_attack_turn_belief(support, support.class_count)
    params = [
        compile_two_attack_turn_context(context)
        for context in contexts
    ]

    def score(projection: TwoAttackTurnProjection) -> int:
        projected = project_two_attack_turn_belief(belief, projection)
        total = 0
        for class_id, representative in enumerate(projection.representative_indices):
            weight = int(projected.weights[class_id])
            if not weight:
                continue
            total += two_attack_turn_numeric(
                params[int(support.context_index[representative])],
                int(support.order_tie_roll[representative]),
                0,
                int(support.p1_damage_roll[representative]),
                int(support.p1_secondary_roll[representative]),
                0,
                int(support.p2_damage_roll[representative]),
            ) * weight
        return total

    direct = sum(
        two_attack_turn_numeric(
            params[int(context_index)],
            int(order_roll),
            0,
            int(p1_damage_roll),
            int(secondary_roll),
            0,
            int(p2_damage_roll),
        )
        for context_index, order_roll, p1_damage_roll, secondary_roll, p2_damage_roll in zip(
            support.context_index,
            support.order_tie_roll,
            support.p1_damage_roll,
            support.p1_secondary_roll,
            support.p2_damage_roll,
            strict=True,
        )
    )
    evidence = {
        "canonical_classes": support.class_count,
        "execution_classes": good.class_count,
        "effect_signature": good.effect_signature,
        "signature_matches_transition": (
            good.effect_signature == two_attack_turn_dependency_signature()
        ),
        "direct_equals_projected": direct == score(good),
        "missing_order_tie_changes_result": direct != score(missing_order),
        "missing_secondary_changes_result": direct != score(missing_secondary),
        "missing_order_classes": missing_order.class_count,
        "missing_secondary_classes": missing_secondary.class_count,
    }
    return evidence, support, good


def _benchmark_support(contexts: Sequence[TwoAttackTurnContext]):
    return build_two_attack_turn_support(
        len(contexts),
        bench_variants=1,
        order_tie_rolls=2,
        p1_damage_rolls=4,
        p1_secondary_rolls=31,
        p2_damage_rolls=4,
    )


def _materialize_direct(
    belief: TwoAttackTurnBelief,
    contexts: Sequence[TwoAttackTurnContext],
):
    support = belief.support
    context_index = np.repeat(support.context_index, belief.weights)
    rows = np.asarray(
        [compile_two_attack_turn_context(context) for context in contexts],
        dtype=np.int32,
    )
    return (
        rows[context_index],
        np.repeat(support.order_tie_roll, belief.weights).astype(np.int32, copy=False),
        np.zeros(belief.logical_world_count, dtype=np.int32),
        np.repeat(support.p1_damage_roll, belief.weights).astype(np.int32, copy=False),
        np.repeat(support.p1_secondary_roll, belief.weights).astype(np.int32, copy=False),
        np.zeros(belief.logical_world_count, dtype=np.int32),
        np.repeat(support.p2_damage_roll, belief.weights).astype(np.int32, copy=False),
    )


def _projected_inputs(
    belief: TwoAttackTurnBelief,
    projection: TwoAttackTurnProjection,
    contexts: Sequence[TwoAttackTurnContext],
):
    projected = project_two_attack_turn_belief(belief, projection)
    active = projected.weights > 0
    representatives = projection.representative_indices[active]
    rows = np.asarray(
        [compile_two_attack_turn_context(context) for context in contexts],
        dtype=np.int32,
    )
    count = len(representatives)
    return (
        rows[belief.support.context_index[representatives]],
        belief.support.order_tie_roll[representatives].astype(np.int32, copy=False),
        np.zeros(count, dtype=np.int32),
        belief.support.p1_damage_roll[representatives].astype(np.int32, copy=False),
        belief.support.p1_secondary_roll[representatives].astype(np.int32, copy=False),
        np.zeros(count, dtype=np.int32),
        belief.support.p2_damage_roll[representatives].astype(np.int32, copy=False),
        projected.weights[active].astype(np.int32, copy=False),
    )


def _median_and_mad(samples):
    median = statistics.median(samples)
    return median, statistics.median(abs(value - median) for value in samples)


def _benchmark_row(
    contexts: Sequence[TwoAttackTurnContext],
    support,
    projection,
    world_count: int,
) -> dict[str, object]:
    belief = uniform_two_attack_turn_belief(support, world_count)
    projected = project_two_attack_turn_belief(belief, projection)
    direct = _materialize_direct(belief, contexts)
    direct_device = tuple(jax.device_put(value) for value in direct)

    def direct_call():
        value = two_attack_turn_score(*direct_device)
        value.block_until_ready()
        return int(np.asarray(value))

    def projected_call():
        values = _projected_inputs(belief, projection, contexts)
        *inputs, weights = values
        value = weighted_two_attack_turn_score(
            *(jax.device_put(value) for value in inputs),
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
    deltas = [
        projected_value - direct_value
        for projected_value, direct_value in zip(
            projected_samples,
            direct_samples,
            strict=True,
        )
    ]
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


def _preexecution_quotient_evidence(
    contexts: Sequence[TwoAttackTurnContext],
    support,
    projection: TwoAttackTurnProjection,
    profile,
) -> dict[str, object]:
    """Compare one held-out full-world execution with pre-execution quotienting."""

    world_count = 73_728
    belief = uniform_two_attack_turn_belief(support, world_count)

    direct = execute_two_attack_turn_jax(
        belief,
        projection,
        contexts,
        force_path=ExecutionPath.DIRECT,
    )
    projected = execute_two_attack_turn_jax(
        belief,
        projection,
        contexts,
        force_path=ExecutionPath.PROJECTED,
    )
    adaptive = execute_two_attack_turn_jax(
        belief,
        projection,
        contexts,
        profile=profile,
    )

    return {
        "logical_world_count": world_count,
        "successor_histogram_exact": projected.histogram() == direct.histogram(),
        "adaptive_histogram_exact": adaptive.histogram() == direct.histogram(),
        "direct_transition_evaluations": direct.certificate[
            "transition_evaluations"
        ],
        "projected_transition_evaluations": projected.certificate[
            "transition_evaluations"
        ],
        "transition_evaluation_reduction_factor": (
            int(direct.certificate["transition_evaluations"])
            / int(projected.certificate["transition_evaluations"])
        ),
        "logical_reduction_fraction": projected.certificate[
            "logical_reduction_fraction"
        ],
        "active_canonical_classes": projected.certificate[
            "active_canonical_classes"
        ],
        "active_execution_classes": projected.certificate[
            "active_execution_classes"
        ],
        "effect_signature": projected.certificate["effect_signature"],
        "projection_binding_hash": projected.certificate[
            "projection_binding_hash"
        ],
        "adaptive_path": adaptive.path.value,
        "adaptive_decision": {
            "predicted_direct_ms": adaptive.decision.predicted_direct_ms,
            "predicted_projected_ms": adaptive.decision.predicted_projected_ms,
            "within_uncertainty_guard": adaptive.decision.within_uncertainty_guard,
        }
        if adaptive.decision is not None
        else None,
    }


def run_experiment(fixtures_path: Path) -> dict[str, object]:
    fixtures = _load(fixtures_path)
    contexts = _benchmark_contexts(fixtures)
    effect_signature = two_attack_turn_dependency_signature()
    target_signature = _execution_target_signature()
    backend = jax.default_backend()

    with tempfile.TemporaryDirectory(prefix="azelficoast-two-attack-turn-") as temp_dir:
        native_build = build_two_attack_turn_library(
            Path(gen9_damage.__file__),
            Path(gen9_two_attack_turn.__file__),
            Path(temp_dir) / "two_attack_turn.so",
            context_width=COMPILED_TWO_ATTACK_TURN_CONTEXT_WIDTH,
        )
        correctness = _correctness(
            fixtures,
            NativeTwoAttackTurn(native_build.library),
        )

    semantics = _semantic_evidence(fixtures)
    protect_projection = _protect_projection_evidence(fixtures)
    projection, _, _ = _projection_evidence(contexts)
    support = _benchmark_support(contexts)
    execution_projection = compile_two_attack_turn_projection(support, contexts)

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
    evaluation = _evaluate(
        confirmation_rows,
        profile=profile,
        baseline=baseline,
    )
    candidate = evaluation["candidate"]
    disjoint = set(TRAINING_WORLD_COUNTS).isdisjoint(CONFIRMATION_WORLD_COUNTS)
    both_paths = set(candidate["chosen_paths"]) == {"direct", "projected"}
    preexecution = _preexecution_quotient_evidence(
        contexts,
        support,
        execution_projection,
        profile,
    )

    passed = (
        correctness["python_exact"]
        and correctness["native_exact"]
        and correctness["jax_exact"]
        and correctness["backends_equal"]
        and semantics["p1_fast_first"]
        and semantics["p2_fast_first"]
        and semantics["tie_both_orders"]
        and semantics["priority_overrides_speed"]
        and semantics["protect_priority_first"]
        and semantics["protect_blocks_opposing_damage"]
        and semantics["protect_hidden_item_and_rng_invariant"]
        and protect_projection["direct_equals_projected"]
        and protect_projection["opponent_item_collapsed"]
        and protect_projection["opponent_damage_rng_collapsed"]
        and protect_projection["effect_signature"] == effect_signature
        and semantics["p1_ko_cancels_p2"]
        and semantics["p2_ko_cancels_p1"]
        and semantics["pre_attack_drop_changes_later_damage"]
        and semantics["post_attack_drop_not_retroactive"]
        and projection["signature_matches_transition"]
        and projection["direct_equals_projected"]
        and projection["missing_order_tie_changes_result"]
        and projection["missing_secondary_changes_result"]
        and disjoint
        and all(bool(row["score_equal"]) for row in training)
        and all(bool(row["score_equal"]) for row in candidate["rows"])
        and both_paths
        and candidate["choice_accuracy"] >= 0.75
        and candidate["adaptive_over_oracle"] <= 1.10
        and candidate["worst_case_over_oracle"] <= 1.20
        and preexecution["successor_histogram_exact"]
        and preexecution["adaptive_histogram_exact"]
        and preexecution["projected_transition_evaluations"]
        < preexecution["direct_transition_evaluations"]
        and preexecution["effect_signature"] == effect_signature
    )

    return {
        "schema": "azelficoast.two-attack-turn-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "backend": backend,
        "jax_version": jax.__version__,
        "target_signature": target_signature,
        "effect_signature": effect_signature,
        "generated_c_bytes": len(native_build.c_source.encode("utf-8")),
        "correctness": correctness,
        "semantics": semantics,
        "protect_projection": protect_projection,
        "projection": projection,
        "training_and_confirmation_disjoint": disjoint,
        "training": training,
        "profile": _profile_record(profile),
        "model_selection": selection,
        "confirmation": evaluation,
        "preexecution_quotient": preexecution,
        "passed": passed,
        "non_claims": [
            "this is a bounded two-action singles turn with an attacking p1 or first-use Protect, not a complete battle turn scheduler",
            "p1's only modeled secondary is a Moonblast-shaped one-stage SpA drop",
            "only ordinary first-use Protect is modeled; consecutive-use probability and bypass mechanics remain outside the transition",
            "switching, immunities, redirection, multihit, contact hooks, statuses, and residual effects remain outside the transition",
            "hosted timing is execution-target-specific CPU evidence",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    args = parser.parse_args(argv)
    result = run_experiment(args.fixtures)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
