"""Pinned-Showdown treatment for bounded voluntary switch-before-attack semantics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.gen9_attack import AttackTransitionContext
from azelficoast.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    _context as damage_context_from_mapping,
)
from azelficoast.voluntary_switch_turn import (
    VoluntarySwitchContext,
    VoluntarySwitchWorld,
    compile_voluntary_switch_projection,
    voluntary_switch_dependency_signature,
    voluntary_switch_turn,
)

SCHEMA = "azelficoast.showdown-voluntary-switch-fixtures"
SCHEMA_VERSION = 1


class VoluntarySwitchExperimentError(ValueError):
    """Raised when voluntary-switch oracle evidence is malformed."""


def _switch_context(fixture: Mapping[str, Any]) -> VoluntarySwitchContext:
    before = fixture.get("before")
    raw_damage = fixture.get("context")
    if not isinstance(before, Mapping):
        raise VoluntarySwitchExperimentError("fixture lacks before state")
    if not isinstance(raw_damage, Mapping):
        raise VoluntarySwitchExperimentError("fixture lacks damage context")

    damage = damage_context_from_mapping(raw_damage)
    attack = AttackTransitionContext(
        damage=damage,
        accuracy=int(fixture["move_accuracy"]),
        attacker_hp=int(before["opponent_hp"]),
        attacker_max_hp=int(before["opponent_max_hp"]),
        defender_hp=int(before["incoming_hp"]),
        move_pp=int(before["opponent_move_pp"]),
    )
    return VoluntarySwitchContext(
        outgoing_slot=int(before["outgoing_slot"]),
        incoming_slot=int(before["incoming_slot"]),
        outgoing_hp=int(before["outgoing_hp"]),
        outgoing_max_hp=int(before["outgoing_max_hp"]),
        incoming_max_hp=int(before["incoming_max_hp"]),
        opponent_attack=attack,
    )


def _observed_successor(fixture: Mapping[str, Any]) -> tuple[int, ...]:
    after = fixture.get("after")
    if not isinstance(after, Mapping):
        raise VoluntarySwitchExperimentError("fixture lacks after state")
    return (
        int(after["active_slot"]),
        int(after["bench_slot"]),
        int(after["active_hp"]),
        int(after["active_max_hp"]),
        int(after["bench_hp"]),
        int(after["bench_max_hp"]),
        int(after["opponent_hp"]),
        int(after["opponent_move_pp"]),
        int(bool(after["hit"])),
        int(bool(after["active_fainted"])),
        int(bool(after["opponent_fainted"])),
    )


def _predicted_successor(
    context: VoluntarySwitchContext,
    fixture: Mapping[str, Any],
) -> tuple[int, ...]:
    return voluntary_switch_turn(
        context,
        accuracy_roll=int(fixture["accuracy_roll"]),
        damage_roll=int(fixture["damage_roll"]),
    ).successor_key


def _rng_shape_is_exact(fixture: Mapping[str, Any]) -> bool:
    requests = fixture.get("rng_requests")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        return False

    accuracy_requests = [
        request
        for request in requests
        if isinstance(request, Mapping) and request.get("kind") == "accuracy"
    ]
    damage_requests = [
        request
        for request in requests
        if isinstance(request, Mapping) and request.get("kind") == "damage"
    ]
    if len(accuracy_requests) != 1:
        return False
    if int(accuracy_requests[0].get("value", -1)) != int(
        fixture["accuracy_roll"]
    ):
        return False

    hit = int(fixture["accuracy_roll"]) < int(fixture["move_accuracy"])
    if len(damage_requests) != int(hit):
        return False
    if hit and int(damage_requests[0].get("value", -1)) != int(
        fixture["damage_roll"]
    ):
        return False
    return True


def _context_matrix(
    fixtures: Sequence[Mapping[str, Any]],
) -> tuple[VoluntarySwitchContext, ...]:
    by_key: dict[tuple[int, str], VoluntarySwitchContext] = {}
    for fixture in fixtures:
        key = (
            int(fixture["incoming_slot"]),
            str(fixture["opponent_item"]),
        )
        by_key.setdefault(key, _switch_context(fixture))

    ordered_keys = (
        (1, "None"),
        (1, "Choice Specs"),
        (2, "None"),
        (2, "Choice Specs"),
    )
    if set(by_key) != set(ordered_keys):
        raise VoluntarySwitchExperimentError("fixture switch/item matrix is incomplete")
    return tuple(by_key[key] for key in ordered_keys)


def analyze_document(document: Mapping[str, Any]) -> dict[str, object]:
    if document.get("schema") != SCHEMA:
        raise VoluntarySwitchExperimentError("unexpected fixture schema")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise VoluntarySwitchExperimentError("unexpected fixture schema version")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise VoluntarySwitchExperimentError("Showdown revision is not pinned")

    raw_fixtures = document.get("fixtures")
    if not isinstance(raw_fixtures, Sequence) or isinstance(
        raw_fixtures, (str, bytes)
    ):
        raise VoluntarySwitchExperimentError("document lacks fixtures")
    fixtures = [fixture for fixture in raw_fixtures if isinstance(fixture, Mapping)]
    if len(fixtures) != len(raw_fixtures):
        raise VoluntarySwitchExperimentError("fixture list contains non-objects")
    if len(fixtures) != int(document.get("fixture_count") or -1):
        raise VoluntarySwitchExperimentError("fixture_count does not match fixtures")

    mismatches: list[dict[str, object]] = []
    rng_exact = True
    ordering_exact = True
    for index, fixture in enumerate(fixtures):
        context = _switch_context(fixture)
        observed = _observed_successor(fixture)
        predicted = _predicted_successor(context, fixture)
        rng_exact = rng_exact and _rng_shape_is_exact(fixture)

        before = fixture["before"]
        after = fixture["after"]
        ordering_exact = ordering_exact and (
            int(after["active_slot"]) == int(before["incoming_slot"])
            and int(after["bench_slot"]) == int(before["outgoing_slot"])
            and int(after["bench_hp"]) == int(before["outgoing_hp"])
        )
        if observed != predicted:
            mismatches.append(
                {
                    "index": index,
                    "incoming_slot": fixture["incoming_slot"],
                    "opponent_item": fixture["opponent_item"],
                    "accuracy_roll": fixture["accuracy_roll"],
                    "damage_roll": fixture["damage_roll"],
                    "observed": observed,
                    "predicted": predicted,
                }
            )

    contexts = _context_matrix(fixtures)
    worlds = [
        VoluntarySwitchWorld(
            context_index=context_index,
            accuracy_roll=accuracy_roll,
            damage_roll=damage_roll,
            untouched_bench_signature=untouched_bench_signature,
        )
        for context_index in range(len(contexts))
        for accuracy_roll in (0, 99)
        for damage_roll in range(16)
        for untouched_bench_signature in range(16)
    ]
    projection = compile_voluntary_switch_projection(contexts, worlds)

    direct = [
        voluntary_switch_turn(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
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

    miss_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if world.accuracy_roll == 99
    }
    hit_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if world.accuracy_roll == 0
    }

    factor_collapsed = True
    by_execution_world: dict[tuple[int, int, int], set[int]] = {}
    for index, world in enumerate(worlds):
        key = (
            world.context_index,
            world.accuracy_roll,
            world.damage_roll,
        )
        by_execution_world.setdefault(key, set()).add(
            int(projection.class_ids[index])
        )
    factor_collapsed = all(
        len(class_ids) == 1
        for class_ids in by_execution_world.values()
    )

    target_miss_ids: dict[int, set[int]] = {1: set(), 2: set()}
    for index, world in enumerate(worlds):
        if world.accuracy_roll != 99:
            continue
        target_slot = contexts[world.context_index].incoming_slot
        target_miss_ids[target_slot].add(int(projection.class_ids[index]))

    full_world_count = len(worlds)
    class_count = projection.class_count
    exact_projection = direct == expanded
    passed = (
        len(fixtures) == 48
        and len(mismatches) == 0
        and rng_exact
        and ordering_exact
        and full_world_count == 2048
        and class_count == 66
        and len(miss_ids) == 2
        and len(hit_ids) == 64
        and miss_ids.isdisjoint(hit_ids)
        and factor_collapsed
        and len(target_miss_ids[1]) == 1
        and len(target_miss_ids[2]) == 1
        and target_miss_ids[1].isdisjoint(target_miss_ids[2])
        and exact_projection
    )

    return {
        "schema": "azelficoast.voluntary-switch-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "effect_signature": voluntary_switch_dependency_signature(),
        "fixture_count": len(fixtures),
        "showdown_exact": len(mismatches) == 0,
        "switch_before_attack_exact": ordering_exact,
        "rng_request_shape_exact": rng_exact,
        "mismatches": mismatches[:5],
        "full_support": {
            "logical_world_count": full_world_count,
            "execution_class_count": class_count,
            "reduction_factor": full_world_count / class_count,
            "direct_equals_projected": exact_projection,
            "miss_class_count": len(miss_ids),
            "hit_class_count": len(hit_ids),
            "untouched_bench_factor_collapsed": factor_collapsed,
            "incoming_identity_preserved": target_miss_ids[1].isdisjoint(
                target_miss_ids[2]
            ),
        },
        "interpretation": {
            "ordering": (
                "The selected bench Pokémon becomes active before the opposing "
                "ordinary attack resolves."
            ),
            "miss": (
                "On an attack miss, opponent item and damage RNG disappear from "
                "the execution quotient while incoming identity remains."
            ),
            "hit": (
                "On a hit, the incoming Pokémon's defensive state, opponent item, "
                "and damage roll remain execution-relevant."
            ),
        },
        "passed": passed,
        "non_claims": [
            "Entry hazards are excluded.",
            "Switch-in and switch-out abilities and items are excluded.",
            "Forced switches, pivot moves, trapping, and residual effects are excluded.",
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
