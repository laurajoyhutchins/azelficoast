"""Pinned-Showdown treatment for switch-in hazards before an opposing attack."""

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
    SwitchHazardWorld,
    compile_switch_hazard_projection,
    resolve_entry_hazards,
    switch_hazard_dependency_signature,
    switch_hazard_turn,
)
from azelficoast.research.mechanics.voluntary_switch_turn import VoluntarySwitchContext

SCHEMA = "azelficoast.showdown-switch-hazard-fixtures"
SCHEMA_VERSION = 1


class SwitchHazardExperimentError(ValueError):
    """Raised when switch-hazard oracle evidence is malformed."""


def _context(fixture: Mapping[str, Any]) -> SwitchHazardContext:
    before = fixture.get("before")
    raw_damage = fixture.get("context")
    if not isinstance(before, Mapping):
        raise SwitchHazardExperimentError("fixture lacks before state")
    if not isinstance(raw_damage, Mapping):
        raise SwitchHazardExperimentError("fixture lacks attack context")

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
    return SwitchHazardContext(
        switch=switch,
        stealth_rock=bool(before["stealth_rock"]),
        spikes_layers=int(before["spikes_layers"]),
        incoming_has_heavy_duty_boots=bool(
            before["incoming_has_heavy_duty_boots"]
        ),
        incoming_grounded=bool(before["incoming_grounded"]),
        stealth_rock_type_mod=int(before["stealth_rock_type_mod"]),
    )


def _observed_successor(fixture: Mapping[str, Any]) -> tuple[int, ...]:
    after = fixture.get("after")
    if not isinstance(after, Mapping):
        raise SwitchHazardExperimentError("fixture lacks after state")
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
        int(bool(after["attack_executed"])),
        int(bool(after["hit"])),
        int(bool(after["active_fainted"])),
        int(bool(after["opponent_fainted"])),
    )


def _predicted_successor(
    context: SwitchHazardContext,
    fixture: Mapping[str, Any],
) -> tuple[int, ...]:
    return switch_hazard_turn(
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
    attack_executed = bool(after["attack_executed"])
    hazard_fainted = bool(after["hazard_fainted"])
    hit = bool(after["hit"])

    if hazard_fainted:
        return (
            attack_executed
            and kinds.count("accuracy") == 0
            and kinds.count("damage") == 0
        )

    if not attack_executed:
        return False

    if kinds.count("accuracy") != 1:
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


def _hazard_event_shape_is_exact(
    fixture: Mapping[str, Any],
    context: SwitchHazardContext,
) -> bool:
    events = fixture.get("hazard_damage_events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        return False

    expected = resolve_entry_hazards(context)
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


def _fixture(
    fixtures: Sequence[Mapping[str, Any]],
    *,
    incoming_case: str,
    incoming_item: str,
    opponent_item: str,
    hazard_case: str,
    hp_case: str,
    accuracy_roll: int = 0,
    damage_roll: int = 0,
) -> Mapping[str, Any]:
    matches = [
        fixture
        for fixture in fixtures
        if fixture["incoming_case"] == incoming_case
        and fixture["incoming_item"] == incoming_item
        and fixture["opponent_item"] == opponent_item
        and fixture["hazard_case"] == hazard_case
        and fixture["hp_case"] == hp_case
        and int(fixture["accuracy_roll"]) == accuracy_roll
        and int(fixture["damage_roll"]) == damage_roll
    ]
    if len(matches) != 1:
        raise SwitchHazardExperimentError(
            "fixture matrix does not contain exactly one requested treatment"
        )
    return matches[0]


def _projection_evidence(
    fixtures: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    context_specs = (
        ("lapras", "None", "None", "stealth-rock-plus-spikes", "full"),
        ("lapras", "None", "Choice Specs", "stealth-rock-plus-spikes", "full"),
        (
            "lapras",
            "Heavy-Duty Boots",
            "None",
            "stealth-rock-plus-spikes",
            "full",
        ),
        (
            "lapras",
            "Heavy-Duty Boots",
            "Choice Specs",
            "stealth-rock-plus-spikes",
            "full",
        ),
        ("charizard", "None", "None", "stealth-rock-plus-spikes", "full"),
        (
            "charizard",
            "None",
            "Choice Specs",
            "stealth-rock-plus-spikes",
            "full",
        ),
        ("lapras", "None", "None", "stealth-rock-plus-spikes", "low"),
        (
            "lapras",
            "None",
            "Choice Specs",
            "stealth-rock-plus-spikes",
            "low",
        ),
    )
    contexts = tuple(
        _context(
            _fixture(
                fixtures,
                incoming_case=incoming_case,
                incoming_item=incoming_item,
                opponent_item=opponent_item,
                hazard_case=hazard_case,
                hp_case=hp_case,
            )
        )
        for (
            incoming_case,
            incoming_item,
            opponent_item,
            hazard_case,
            hp_case,
        ) in context_specs
    )

    worlds = [
        SwitchHazardWorld(
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
    projection = compile_switch_hazard_projection(contexts, worlds)

    direct = [
        switch_hazard_turn(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
        ).successor_key
        for world in worlds
    ]
    representative_outputs = [
        direct[int(index)] for index in projection.representative_indices
    ]
    expanded = [
        representative_outputs[int(class_id)]
        for class_id in projection.class_ids
    ]

    hazard_ko_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if resolve_entry_hazards(contexts[world.context_index]).fainted
    }
    survivor_miss_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if not resolve_entry_hazards(contexts[world.context_index]).fainted
        and world.accuracy_roll == 99
    }
    survivor_hit_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if not resolve_entry_hazards(contexts[world.context_index]).fainted
        and world.accuracy_roll == 0
    }

    factored = True
    by_mechanics_world: dict[tuple[int, int, int], set[int]] = {}
    for index, world in enumerate(worlds):
        key = (world.context_index, world.accuracy_roll, world.damage_roll)
        by_mechanics_world.setdefault(key, set()).add(
            int(projection.class_ids[index])
        )
    factored = all(len(ids) == 1 for ids in by_mechanics_world.values())

    return {
        "logical_world_count": len(worlds),
        "execution_class_count": projection.class_count,
        "reduction_factor": len(worlds) / projection.class_count,
        "direct_equals_projected": direct == expanded,
        "hazard_ko_class_count": len(hazard_ko_ids),
        "survivor_miss_class_count": len(survivor_miss_ids),
        "survivor_hit_class_count": len(survivor_hit_ids),
        "untouched_bench_factor_collapsed": factored,
        "effect_signature": projection.effect_signature,
    }


def analyze_document(document: Mapping[str, Any]) -> dict[str, object]:
    if document.get("schema") != SCHEMA:
        raise SwitchHazardExperimentError("unexpected fixture schema")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise SwitchHazardExperimentError("unexpected fixture schema version")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise SwitchHazardExperimentError("Showdown revision is not pinned")

    raw_fixtures = document.get("fixtures")
    if not isinstance(raw_fixtures, Sequence) or isinstance(
        raw_fixtures, (str, bytes)
    ):
        raise SwitchHazardExperimentError("document lacks fixtures")
    fixtures = [fixture for fixture in raw_fixtures if isinstance(fixture, Mapping)]
    if len(fixtures) != len(raw_fixtures):
        raise SwitchHazardExperimentError("fixture list contains non-objects")
    if len(fixtures) != int(document.get("fixture_count") or -1):
        raise SwitchHazardExperimentError("fixture_count does not match fixtures")

    mismatches: list[dict[str, object]] = []
    rng_exact = True
    hazard_events_exact = True
    switch_order_exact = True

    hazard_ko_count = 0
    boots_bypass_count = 0
    airborne_spikes_bypass_count = 0

    for index, fixture in enumerate(fixtures):
        context = _context(fixture)
        observed = _observed_successor(fixture)
        predicted = _predicted_successor(context, fixture)

        rng_exact = rng_exact and _rng_shape_is_exact(fixture)
        hazard_events_exact = hazard_events_exact and _hazard_event_shape_is_exact(
            fixture,
            context,
        )

        before = fixture["before"]
        after = fixture["after"]
        switch_order_exact = switch_order_exact and (
            int(after["active_slot"]) == int(before["incoming_slot"])
            and int(after["bench_slot"]) == int(before["outgoing_slot"])
            and int(after["bench_hp"]) == int(before["outgoing_hp"])
        )

        if bool(after["hazard_fainted"]):
            hazard_ko_count += 1
            if not bool(after["attack_executed"]):
                mismatches.append(
                    {
                        "index": index,
                        "reason": "hazard-faint did not consume queued move action",
                    }
                )
            if int(after["opponent_move_pp"]) != int(before["opponent_move_pp"]) - 1:
                mismatches.append(
                    {
                        "index": index,
                        "reason": "hazard-faint queued move did not consume exactly one PP",
                    }
                )

        if bool(before["incoming_has_heavy_duty_boots"]):
            if int(after["hazard_damage"]) == 0:
                boots_bypass_count += 1

        if (
            fixture["incoming_case"] == "charizard"
            and fixture["hazard_case"].startswith("spikes")
            and not bool(before["incoming_has_heavy_duty_boots"])
            and int(after["hazard_damage"]) == 0
        ):
            airborne_spikes_bypass_count += 1

        if observed != predicted:
            mismatches.append(
                {
                    "index": index,
                    "incoming_case": fixture["incoming_case"],
                    "incoming_item": fixture["incoming_item"],
                    "opponent_item": fixture["opponent_item"],
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
        len(fixtures) == 384
        and len(mismatches) == 0
        and rng_exact
        and hazard_events_exact
        and switch_order_exact
        and hazard_ko_count > 0
        and boots_bypass_count > 0
        and airborne_spikes_bypass_count > 0
        and projection["logical_world_count"] == 4096
        and projection["execution_class_count"] == 100
        and projection["hazard_ko_class_count"] == 1
        and projection["survivor_miss_class_count"] == 3
        and projection["survivor_hit_class_count"] == 96
        and projection["untouched_bench_factor_collapsed"]
        and projection["direct_equals_projected"]
        and projection["effect_signature"] == switch_hazard_dependency_signature()
    )

    return {
        "schema": "azelficoast.switch-hazard-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "effect_signature": switch_hazard_dependency_signature(),
        "fixture_count": len(fixtures),
        "showdown_exact": len(mismatches) == 0,
        "switch_before_hazards_before_attack_exact": switch_order_exact,
        "rng_request_shape_exact": rng_exact,
        "hazard_damage_events_exact": hazard_events_exact,
        "hazard_ko_fixture_count": hazard_ko_count,
        "boots_bypass_fixture_count": boots_bypass_count,
        "airborne_spikes_bypass_fixture_count": airborne_spikes_bypass_count,
        "mismatches": mismatches[:8],
        "projection": projection,
        "passed": passed,
        "non_claims": [
            "Magic Guard and item suppression remain outside this bounded model.",
            "Toxic Spikes and Sticky Web remain outside this bounded model.",
            "Switch-in/out abilities and items other than Heavy-Duty Boots are excluded.",
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
