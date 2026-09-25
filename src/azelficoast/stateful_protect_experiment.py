"""Differential experiment for history-dependent repeated Protect."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.gen9_attack import AttackTransitionContext
from azelficoast.gen9_two_attack_turn import (
    P1_ACTION_PROTECT,
    TwoAttackTurnContext,
)
from azelficoast.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    _context as damage_context_from_mapping,
)
from azelficoast.stateful_protect_turn import (
    PROTECT_ROLL_DENOMINATOR,
    PROTECT_STALL_COUNTERS,
    StatefulProtectWorld,
    compile_stateful_protect_projection,
    protect_succeeds,
    stateful_protect_dependency_key,
    stateful_protect_dependency_signature,
    stateful_protect_turn,
)

SCHEMA = "azelficoast.showdown-stateful-protect-fixtures"
SCHEMA_VERSION = 1


class StatefulProtectExperimentError(ValueError):
    """Raised when repeated-Protect oracle evidence is malformed."""


def _turn_context(fixture: Mapping[str, Any]) -> TwoAttackTurnContext:
    before = fixture.get("before")
    p1_raw = fixture.get("p1_context")
    p2_raw = fixture.get("p2_context")
    if not isinstance(before, Mapping):
        raise StatefulProtectExperimentError("fixture lacks before state")
    if not isinstance(p1_raw, Mapping) or not isinstance(p2_raw, Mapping):
        raise StatefulProtectExperimentError("fixture lacks damage contexts")

    # The fixed-width compiled turn retains an inert attack slot for Protect.
    p1_damage = damage_context_from_mapping(
        {
            **p1_raw,
            "category": "Special",
            "base_power": 0,
        }
    )
    p2_damage = damage_context_from_mapping(p2_raw)

    return TwoAttackTurnContext(
        p1_attack=AttackTransitionContext(
            damage=p1_damage,
            accuracy=100,
            attacker_hp=int(before["p1_hp"]),
            attacker_max_hp=int(before["p1_hp"]),
            defender_hp=int(before["p2_hp"]),
            move_pp=int(before["p1_pp"]),
        ),
        p2_attack=AttackTransitionContext(
            damage=p2_damage,
            accuracy=100,
            attacker_hp=int(before["p2_hp"]),
            attacker_max_hp=int(before["p2_hp"]),
            defender_hp=int(before["p1_hp"]),
            move_pp=int(before["p2_pp"]),
        ),
        p1_priority=int(fixture["p1_priority"]),
        p2_priority=int(fixture["p2_priority"]),
        p1_speed=int(fixture["p1_speed"]),
        p2_speed=int(fixture["p2_speed"]),
        p1_spa_drop_chance=0,
        p2_spa_stage=int(before["p2_spa_stage"]),
        p1_action_kind=P1_ACTION_PROTECT,
    )


def _observed_successor(fixture: Mapping[str, Any]) -> tuple[int, ...]:
    after = fixture.get("after")
    if not isinstance(after, Mapping):
        raise StatefulProtectExperimentError("fixture lacks after state")
    return (
        int(after["p1_hp"]),
        int(after["p2_hp"]),
        int(after["p1_pp"]),
        int(after["p2_pp"]),
        int(after["p2_spa_stage"]),
        int(bool(after["p1_acted"])),
        int(bool(after["p2_acted"])),
        int(after["stall_counter"]),
        int(bool(after["protect_succeeded"])),
    )


def _predicted_successor(
    context: TwoAttackTurnContext,
    fixture: Mapping[str, Any],
) -> tuple[int, ...]:
    outcome = stateful_protect_turn(
        context,
        stall_counter=int(fixture["stall_counter"]),
        protect_roll=int(fixture["canonical_protect_roll"]),
        p2_accuracy_roll=0,
        p2_damage_roll=int(fixture["p2_damage_roll"]),
    )
    turn = outcome.turn
    return (
        turn.p1_hp,
        turn.p2_hp,
        turn.p1_pp,
        turn.p2_pp,
        turn.p2_spa_stage,
        int(turn.p1_acted),
        int(turn.p2_acted),
        outcome.stall_counter_after,
        int(outcome.protect_succeeded),
    )


def _rng_shape_is_exact(fixture: Mapping[str, Any]) -> bool:
    requests = fixture.get("rng_requests")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        return False
    kinds = [
        str(request["kind"])
        for request in requests
        if isinstance(request, Mapping) and isinstance(request.get("kind"), str)
    ]
    counter = int(fixture["stall_counter"])
    succeeded = bool(fixture["expected_success"])

    stall_requests = [
        request
        for request in requests
        if isinstance(request, Mapping) and request.get("kind") == "stall"
    ]
    if counter == 1:
        if stall_requests:
            return False
    elif len(stall_requests) != 1:
        return False
    else:
        stall = stall_requests[0]
        if (
            int(stall.get("numerator", -1)) != 1
            or int(stall.get("denominator", -1)) != counter
            or int(stall.get("value", -1))
            != int(fixture["showdown_protect_roll"])
        ):
            return False

    damage_calls = kinds.count("damage")
    return damage_calls == (0 if succeeded else 1)


def analyze_document(document: Mapping[str, Any]) -> dict[str, object]:
    if document.get("schema") != SCHEMA:
        raise StatefulProtectExperimentError("unexpected fixture schema")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise StatefulProtectExperimentError("unexpected fixture schema version")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise StatefulProtectExperimentError("Showdown revision is not pinned")
    if document.get("canonical_roll_denominator") != PROTECT_ROLL_DENOMINATOR:
        raise StatefulProtectExperimentError("unexpected canonical Protect roll lattice")

    raw_fixtures = document.get("fixtures")
    if not isinstance(raw_fixtures, Sequence) or isinstance(
        raw_fixtures, (str, bytes)
    ):
        raise StatefulProtectExperimentError("document lacks fixtures")
    fixtures = [fixture for fixture in raw_fixtures if isinstance(fixture, Mapping)]
    if len(fixtures) != len(raw_fixtures):
        raise StatefulProtectExperimentError("fixture list contains non-objects")
    if len(fixtures) != int(document.get("fixture_count") or -1):
        raise StatefulProtectExperimentError("fixture_count does not match fixtures")

    contexts = [_turn_context(fixture) for fixture in fixtures]
    mismatches: list[dict[str, object]] = []
    rng_exact = True

    for index, (fixture, context) in enumerate(zip(fixtures, contexts, strict=True)):
        observed = _observed_successor(fixture)
        predicted = _predicted_successor(context, fixture)
        rng_exact = rng_exact and _rng_shape_is_exact(fixture)
        if observed != predicted:
            mismatches.append(
                {
                    "index": index,
                    "counter": fixture["stall_counter"],
                    "canonical_roll": fixture["canonical_protect_roll"],
                    "item": fixture["p2_item"],
                    "damage_roll": fixture["p2_damage_roll"],
                    "observed": observed,
                    "predicted": predicted,
                }
            )

    item_contexts: dict[str, TwoAttackTurnContext] = {}
    for fixture, context in zip(fixtures, contexts, strict=True):
        item_contexts.setdefault(str(fixture["p2_item"]), context)
    if set(item_contexts) != {"None", "Choice Specs"}:
        raise StatefulProtectExperimentError("fixture item matrix is incomplete")
    projection_contexts = (
        item_contexts["None"],
        item_contexts["Choice Specs"],
    )

    worlds = [
        StatefulProtectWorld(
            context_index=context_index,
            stall_counter=counter,
            protect_roll=protect_roll,
            p2_accuracy_roll=0,
            p2_damage_roll=damage_roll,
        )
        for counter in PROTECT_STALL_COUNTERS
        for context_index in range(2)
        for damage_roll in (0, 15)
        for protect_roll in range(PROTECT_ROLL_DENOMINATOR)
    ]
    projection = compile_stateful_protect_projection(projection_contexts, worlds)

    direct = [
        stateful_protect_turn(
            projection_contexts[world.context_index],
            stall_counter=world.stall_counter,
            protect_roll=world.protect_roll,
            p2_accuracy_roll=world.p2_accuracy_roll,
            p2_damage_roll=world.p2_damage_roll,
        ).successor_key
        for world in worlds
    ]
    representative_outputs = [
        direct[int(index)]
        for index in projection.representative_indices
    ]
    expanded = [
        representative_outputs[int(class_id)]
        for class_id in projection.class_ids
    ]

    success_class_counts: dict[str, int] = {}
    failure_class_counts: dict[str, int] = {}
    for counter in PROTECT_STALL_COUNTERS:
        success_ids = {
            int(projection.class_ids[index])
            for index, world in enumerate(worlds)
            if world.stall_counter == counter
            and protect_succeeds(counter, world.protect_roll)
        }
        failure_ids = {
            int(projection.class_ids[index])
            for index, world in enumerate(worlds)
            if world.stall_counter == counter
            and not protect_succeeds(counter, world.protect_roll)
        }
        success_class_counts[str(counter)] = len(success_ids)
        failure_class_counts[str(counter)] = len(failure_ids)

    fixture_success_keys: dict[int, set[tuple[int, ...]]] = {}
    fixture_failure_keys: dict[int, set[tuple[int, ...]]] = {}
    for fixture, context in zip(fixtures, contexts, strict=True):
        counter = int(fixture["stall_counter"])
        key = stateful_protect_dependency_key(
            context,
            stall_counter=counter,
            protect_roll=int(fixture["canonical_protect_roll"]),
            p2_accuracy_roll=0,
            p2_damage_roll=int(fixture["p2_damage_roll"]),
        )
        target = (
            fixture_success_keys
            if bool(fixture["expected_success"])
            else fixture_failure_keys
        )
        target.setdefault(counter, set()).add(key)

    full_world_count = len(worlds)
    class_count = projection.class_count
    exact_projection = direct == expanded
    repeated_counters = PROTECT_STALL_COUNTERS[1:]

    passed = (
        len(mismatches) == 0
        and rng_exact
        and exact_projection
        and full_world_count == 20_412
        and class_count == 31
        and all(success_class_counts[str(counter)] == 1 for counter in PROTECT_STALL_COUNTERS)
        and failure_class_counts["1"] == 0
        and all(failure_class_counts[str(counter)] == 4 for counter in repeated_counters)
        and all(len(fixture_success_keys.get(counter, set())) == 1 for counter in PROTECT_STALL_COUNTERS)
        and all(len(fixture_failure_keys.get(counter, set())) == 4 for counter in repeated_counters)
    )

    return {
        "schema": "azelficoast.stateful-protect-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "effect_signature": stateful_protect_dependency_signature(),
        "fixture_count": len(fixtures),
        "showdown_exact": len(mismatches) == 0,
        "rng_request_shape_exact": rng_exact,
        "mismatches": mismatches[:5],
        "full_support": {
            "logical_world_count": full_world_count,
            "execution_class_count": class_count,
            "reduction_factor": full_world_count / class_count,
            "direct_equals_projected": exact_projection,
            "success_class_counts": success_class_counts,
            "failure_class_counts": failure_class_counts,
        },
        "interpretation": {
            "success": (
                "Repeated Protect success removes blocked opponent item and damage-RNG "
                "dependencies while preserving the next stall counter."
            ),
            "failure": (
                "Repeated Protect failure clears stall state and restores ordinary "
                "opponent damage dependencies."
            ),
        },
        "passed": passed,
        "non_claims": [
            "The bounded opponent action is an ordinary Protect-blockable special attack.",
            "Protect-bypassing moves and abilities remain outside the model.",
            "Other stalling moves are not yet represented as compiled action kinds.",
            "Residual and end-turn mechanics remain outside the two-action transition.",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    document = json.loads(args.fixtures.read_text(encoding="utf-8"))
    result = analyze_document(document)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
