"""Pinned-Showdown treatment for Intimidate in the switch-in event chain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.research.mechanics.gen9_attack import AttackTransitionContext
from azelficoast.research.verification.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    _context as damage_context_from_mapping,
)
from azelficoast.research.mechanics.switch_hazard_turn import (
    SwitchHazardContext,
    resolve_entry_hazards,
)
from azelficoast.research.mechanics.switch_intimidate_turn import (
    SwitchIntimidateContext,
    SwitchIntimidateWorld,
    compile_switch_intimidate_projection,
    switch_intimidate_dependency_signature,
    switch_intimidate_turn,
)
from azelficoast.research.mechanics.voluntary_switch_turn import VoluntarySwitchContext

SCHEMA = "azelficoast.showdown-switch-intimidate-fixtures"
SCHEMA_VERSION = 1


class SwitchIntimidateExperimentError(ValueError):
    """Raised when Intimidate switch oracle evidence is malformed."""


def _context(fixture: Mapping[str, Any]) -> SwitchIntimidateContext:
    before = fixture.get("before")
    raw_damage = fixture.get("context")
    if not isinstance(before, Mapping):
        raise SwitchIntimidateExperimentError("fixture lacks before state")
    if not isinstance(raw_damage, Mapping):
        raise SwitchIntimidateExperimentError("fixture lacks damage context")

    damage = damage_context_from_mapping(raw_damage)
    switch = VoluntarySwitchContext(
        outgoing_slot=int(before["outgoing_slot"]),
        incoming_slot=int(before["incoming_slot"]),
        outgoing_hp=int(before["outgoing_hp"]),
        outgoing_max_hp=int(before["outgoing_max_hp"]),
        incoming_max_hp=int(before["incoming_max_hp"]),
        opponent_attack=AttackTransitionContext(
            damage=damage,
            accuracy=int(fixture["move_accuracy"]),
            attacker_hp=int(before["opponent_hp"]),
            attacker_max_hp=int(before["opponent_max_hp"]),
            defender_hp=int(before["incoming_hp"]),
            move_pp=int(before["opponent_move_pp"]),
        ),
    )
    hazards = SwitchHazardContext(
        switch=switch,
        stealth_rock=bool(before["stealth_rock"]),
        spikes_layers=int(before["spikes_layers"]),
        incoming_has_heavy_duty_boots=bool(
            before["incoming_has_heavy_duty_boots"]
        ),
        incoming_grounded=bool(before["incoming_grounded"]),
        stealth_rock_type_mod=int(before["stealth_rock_type_mod"]),
    )
    return SwitchIntimidateContext(
        switch_hazards=hazards,
        incoming_has_intimidate=str(before["incoming_ability"]) == "Intimidate",
        opponent_attack_stage=int(before["opponent_attack_stage"]),
    )


def _observed_successor(fixture: Mapping[str, Any]) -> tuple[int, ...]:
    after = fixture.get("after")
    if not isinstance(after, Mapping):
        raise SwitchIntimidateExperimentError("fixture lacks after state")
    return (
        int(after["active_slot"]),
        int(after["bench_slot"]),
        int(after["active_hp"]),
        int(after["active_max_hp"]),
        int(after["bench_hp"]),
        int(after["bench_max_hp"]),
        int(after["opponent_hp"]),
        int(after["opponent_move_pp"]),
        int(after["hazard_damage"]),
        int(bool(after["hazard_fainted"])),
        int(bool(after["intimidate_activated"])),
        int(bool(after["intimidate_changed_stage"])),
        int(after["opponent_attack_stage"]),
        int(bool(after["attack_executed"])),
        int(bool(after["hit"])),
        int(bool(after["active_fainted"])),
        int(bool(after["opponent_fainted"])),
    )


def _predicted_successor(
    context: SwitchIntimidateContext,
    fixture: Mapping[str, Any],
) -> tuple[int, ...]:
    return switch_intimidate_turn(
        context,
        accuracy_roll=int(fixture["accuracy_roll"]),
        damage_roll=int(fixture["damage_roll"]),
    ).successor_key


def _rng_shape_is_exact(fixture: Mapping[str, Any]) -> bool:
    requests = fixture.get("rng_requests")
    after = fixture.get("after")
    if not isinstance(requests, Sequence) or isinstance(requests, (str, bytes)):
        return False
    if not isinstance(after, Mapping):
        return False

    kinds = [
        str(request["kind"])
        for request in requests
        if isinstance(request, Mapping) and isinstance(request.get("kind"), str)
    ]
    hazard_fainted = bool(after["hazard_fainted"])
    attack_executed = bool(after["attack_executed"])
    hit = bool(after["hit"])

    if hazard_fainted:
        return (
            attack_executed
            and kinds.count("accuracy") == 0
            and kinds.count("damage") == 0
        )
    if not attack_executed or kinds.count("accuracy") != 1:
        return False
    if kinds.count("damage") != int(hit):
        return False

    accuracy_request = next(
        request
        for request in requests
        if isinstance(request, Mapping) and request.get("kind") == "accuracy"
    )
    if int(accuracy_request.get("value", -1)) != int(fixture["accuracy_roll"]):
        return False

    if hit:
        damage_request = next(
            request
            for request in requests
            if isinstance(request, Mapping) and request.get("kind") == "damage"
        )
        if int(damage_request.get("value", -1)) != int(fixture["damage_roll"]):
            return False
    return True


def _hazard_events_are_exact(
    fixture: Mapping[str, Any],
    context: SwitchIntimidateContext,
) -> bool:
    events = fixture.get("hazard_damage_events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return False

    expected = resolve_entry_hazards(context.switch_hazards)
    observed_by_effect: dict[str, int] = {}
    for event in events:
        if not isinstance(event, Mapping):
            return False
        effect = str(event.get("effect"))
        if effect not in {"stealthrock", "spikes"}:
            return False
        observed_by_effect[effect] = observed_by_effect.get(effect, 0) + int(
            event.get("damage", 0)
        )

    return (
        observed_by_effect.get("stealthrock", 0)
        == expected.stealth_rock_damage
        and observed_by_effect.get("spikes", 0) == expected.spikes_damage
        and sum(observed_by_effect.values()) == expected.total_damage
    )


def _event_order_is_exact(fixture: Mapping[str, Any]) -> bool:
    log = fixture.get("transition_log")
    after = fixture.get("after")
    before = fixture.get("before")
    if not isinstance(log, Sequence) or isinstance(log, (str, bytes)):
        return False
    if not isinstance(after, Mapping) or not isinstance(before, Mapping):
        return False

    lines = [str(line) for line in log]
    hazard_indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("|-damage|") and (
            "[from] Stealth Rock" in line or "[from] Spikes" in line
        )
    ]
    ability_indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("|-ability|") and "|Intimidate|" in line
    ]

    expects_intimidate = (
        str(before["incoming_ability"]) == "Intimidate"
        and not bool(after["hazard_fainted"])
    )
    if expects_intimidate != bool(ability_indexes):
        return False
    if bool(after["hazard_fainted"]) and ability_indexes:
        return False
    if hazard_indexes and ability_indexes and max(hazard_indexes) >= min(ability_indexes):
        return False
    return True


def _intimidate_state_is_exact(fixture: Mapping[str, Any]) -> bool:
    before = fixture.get("before")
    after = fixture.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False

    initial = int(before["opponent_attack_stage"])
    has_intimidate = str(before["incoming_ability"]) == "Intimidate"
    hazard_fainted = bool(after["hazard_fainted"])

    if has_intimidate and not hazard_fainted:
        expected_stage = max(-6, initial - 1)
        return (
            bool(after["intimidate_activated"])
            and bool(after["intimidate_changed_stage"]) == (expected_stage != initial)
            and int(after["opponent_attack_stage"]) == expected_stage
        )
    return (
        not bool(after["intimidate_activated"])
        and not bool(after["intimidate_changed_stage"])
        and int(after["opponent_attack_stage"]) == initial
    )


def _fixture(
    fixtures: Sequence[Mapping[str, Any]],
    *,
    ability: str,
    opponent_item: str,
    initial_attack_stage: int,
    hp_case: str,
    incoming_item: str = "None",
    hazard_case: str = "stealth-rock-plus-spikes",
    accuracy_roll: int = 0,
    damage_roll: int = 0,
) -> Mapping[str, Any]:
    matches = [
        fixture
        for fixture in fixtures
        if fixture["ability"] == ability
        and fixture["incoming_item"] == incoming_item
        and fixture["opponent_item"] == opponent_item
        and int(fixture["initial_attack_stage"]) == initial_attack_stage
        and fixture["hazard_case"] == hazard_case
        and fixture["hp_case"] == hp_case
        and int(fixture["accuracy_roll"]) == accuracy_roll
        and int(fixture["damage_roll"]) == damage_roll
    ]
    if len(matches) != 1:
        raise SwitchIntimidateExperimentError(
            "fixture matrix does not contain exactly one requested treatment"
        )
    return matches[0]


def _projection_evidence(
    fixtures: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    specs = (
        ("Flash Fire", "None", 0, "full"),
        ("Flash Fire", "Choice Band", 0, "full"),
        ("Intimidate", "None", 0, "full"),
        ("Intimidate", "Choice Band", 0, "full"),
        ("Intimidate", "None", 1, "full"),
        ("Intimidate", "Choice Band", 1, "full"),
        ("Flash Fire", "None", 0, "low"),
        ("Intimidate", "Choice Band", 0, "low"),
    )
    contexts = tuple(
        _context(
            _fixture(
                fixtures,
                ability=ability,
                opponent_item=opponent_item,
                initial_attack_stage=stage,
                hp_case=hp_case,
            )
        )
        for ability, opponent_item, stage, hp_case in specs
    )

    worlds = [
        SwitchIntimidateWorld(
            context_index=context_index,
            accuracy_roll=accuracy_roll,
            damage_roll=damage_roll,
            untouched_bench_signature=untouched,
        )
        for context_index in range(len(contexts))
        for accuracy_roll in (0, 99)
        for damage_roll in range(16)
        for untouched in range(16)
    ]
    projection = compile_switch_intimidate_projection(contexts, worlds)

    direct = [
        switch_intimidate_turn(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
        ).successor_key
        for world in worlds
    ]
    representatives = [
        direct[int(index)]
        for index in projection.representative_indices
    ]
    expanded = [
        representatives[int(class_id)]
        for class_id in projection.class_ids
    ]

    ko_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if resolve_entry_hazards(contexts[world.context_index].switch_hazards).fainted
    }
    survivor_miss_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if not resolve_entry_hazards(contexts[world.context_index].switch_hazards).fainted
        and world.accuracy_roll == 99
    }
    survivor_hit_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if not resolve_entry_hazards(contexts[world.context_index].switch_hazards).fainted
        and world.accuracy_roll == 0
    }

    by_execution_world: dict[tuple[int, int, int], set[int]] = {}
    for index, world in enumerate(worlds):
        key = (world.context_index, world.accuracy_roll, world.damage_roll)
        by_execution_world.setdefault(key, set()).add(
            int(projection.class_ids[index])
        )
    factored = all(len(ids) == 1 for ids in by_execution_world.values())

    return {
        "logical_world_count": len(worlds),
        "execution_class_count": projection.class_count,
        "reduction_factor": len(worlds) / projection.class_count,
        "direct_equals_projected": direct == expanded,
        "hazard_ko_class_count": len(ko_ids),
        "survivor_miss_class_count": len(survivor_miss_ids),
        "survivor_hit_class_count": len(survivor_hit_ids),
        "untouched_bench_factor_collapsed": factored,
        "effect_signature": projection.effect_signature,
    }


def analyze_document(document: Mapping[str, Any]) -> dict[str, object]:
    if document.get("schema") != SCHEMA:
        raise SwitchIntimidateExperimentError("unexpected fixture schema")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise SwitchIntimidateExperimentError("unexpected fixture schema version")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise SwitchIntimidateExperimentError("Showdown revision is not pinned")

    raw_fixtures = document.get("fixtures")
    if not isinstance(raw_fixtures, Sequence) or isinstance(
        raw_fixtures, (str, bytes)
    ):
        raise SwitchIntimidateExperimentError("document lacks fixtures")
    fixtures = [fixture for fixture in raw_fixtures if isinstance(fixture, Mapping)]
    if len(fixtures) != len(raw_fixtures):
        raise SwitchIntimidateExperimentError("fixture list contains non-objects")
    if len(fixtures) != int(document.get("fixture_count") or -1):
        raise SwitchIntimidateExperimentError("fixture_count does not match fixtures")

    mismatches: list[dict[str, object]] = []
    rng_exact = True
    hazard_events_exact = True
    event_order_exact = True
    intimidate_state_exact = True
    hazard_ko_skips_intimidate = 0
    boots_enable_intimidate = 0

    for index, fixture in enumerate(fixtures):
        context = _context(fixture)
        observed = _observed_successor(fixture)
        predicted = _predicted_successor(context, fixture)

        rng_exact = rng_exact and _rng_shape_is_exact(fixture)
        hazard_events_exact = hazard_events_exact and _hazard_events_are_exact(
            fixture,
            context,
        )
        event_order_exact = event_order_exact and _event_order_is_exact(fixture)
        intimidate_state_exact = (
            intimidate_state_exact and _intimidate_state_is_exact(fixture)
        )

        before = fixture["before"]
        after = fixture["after"]
        if (
            str(before["incoming_ability"]) == "Intimidate"
            and bool(after["hazard_fainted"])
            and not bool(after["intimidate_activated"])
        ):
            hazard_ko_skips_intimidate += 1

        if (
            str(before["incoming_ability"]) == "Intimidate"
            and bool(before["incoming_has_heavy_duty_boots"])
            and fixture["hazard_case"] == "stealth-rock-plus-spikes"
            and fixture["hp_case"] == "low"
            and int(after["hazard_damage"]) == 0
            and bool(after["intimidate_activated"])
        ):
            boots_enable_intimidate += 1

        if observed != predicted:
            mismatches.append(
                {
                    "index": index,
                    "ability": fixture["ability"],
                    "incoming_item": fixture["incoming_item"],
                    "opponent_item": fixture["opponent_item"],
                    "initial_attack_stage": fixture["initial_attack_stage"],
                    "hazard_case": fixture["hazard_case"],
                    "hp_case": fixture["hp_case"],
                    "accuracy_roll": fixture["accuracy_roll"],
                    "damage_roll": fixture["damage_roll"],
                    "observed": observed,
                    "predicted": predicted,
                }
            )

    projection = _projection_evidence(fixtures)
    passed = (
        len(fixtures) == 256
        and len(mismatches) == 0
        and rng_exact
        and hazard_events_exact
        and event_order_exact
        and intimidate_state_exact
        and hazard_ko_skips_intimidate > 0
        and boots_enable_intimidate > 0
        and projection["logical_world_count"] == 4096
        and projection["execution_class_count"] == 100
        and projection["hazard_ko_class_count"] == 1
        and projection["survivor_miss_class_count"] == 3
        and projection["survivor_hit_class_count"] == 96
        and projection["untouched_bench_factor_collapsed"]
        and projection["direct_equals_projected"]
        and projection["effect_signature"] == switch_intimidate_dependency_signature()
    )

    return {
        "schema": "azelficoast.switch-intimidate-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "effect_signature": switch_intimidate_dependency_signature(),
        "fixture_count": len(fixtures),
        "showdown_exact": len(mismatches) == 0,
        "hazard_damage_events_exact": hazard_events_exact,
        "hazards_before_intimidate_exact": event_order_exact,
        "intimidate_stage_state_exact": intimidate_state_exact,
        "rng_request_shape_exact": rng_exact,
        "hazard_ko_skips_intimidate_fixture_count": hazard_ko_skips_intimidate,
        "boots_enable_intimidate_fixture_count": boots_enable_intimidate,
        "mismatches": mismatches[:8],
        "projection": projection,
        "passed": passed,
        "non_claims": [
            "Intimidate blockers, Substitute, reflection, and boost inversion are excluded.",
            "Ability suppression, Trace, Skill Swap, and Mold Breaker interactions are excluded.",
            "Special queued attacks and other switch-in effects are excluded.",
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
