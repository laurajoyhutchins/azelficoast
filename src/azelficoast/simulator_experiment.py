"""Finite dependency-collapsing experiment for the simulator IR."""

from __future__ import annotations

import json

from azelficoast.simulator_ir import (
    ITEM_CHOICE_SCARF,
    ITEM_CHOICE_SPECS,
    MOVE_MOONBLAST,
    PROTECT_BLOCK,
    SPECIAL_DAMAGE,
    DependencyViolation,
    EffectSpec,
    RandomField,
    RandomInput,
    StateField,
    World,
    collapse_worlds,
    execute_direct,
    expand_collapsed,
    field_mask,
)


def _worlds(bench_variants: int = 2048) -> tuple[World, ...]:
    worlds = []
    for item in (ITEM_CHOICE_SCARF, ITEM_CHOICE_SPECS):
        for bench_signature in range(bench_variants):
            worlds.append(
                World.from_values(
                    {
                        StateField.OWN_HP: 180,
                        StateField.OPPONENT_HP: 100,
                        StateField.OWN_SPEED: 206,
                        StateField.OPPONENT_SPEED: 180,
                        StateField.OPPONENT_ITEM: item,
                        StateField.OPPONENT_MOVE: MOVE_MOONBLAST,
                        StateField.BENCH_SIGNATURE: bench_signature,
                    }
                )
            )
    return tuple(worlds)


def run_experiment(bench_variants: int = 2048) -> dict[str, object]:
    worlds = _worlds(bench_variants)
    random = RandomInput.from_values({RandomField.DAMAGE_ROLL: 7})

    protect = collapse_worlds(PROTECT_BLOCK, worlds, random)
    damage = collapse_worlds(SPECIAL_DAMAGE, worlds, random)

    protect_exact = expand_collapsed(worlds, protect) == execute_direct(
        PROTECT_BLOCK, worlds, random
    )
    damage_exact = expand_collapsed(worlds, damage) == execute_direct(
        SPECIAL_DAMAGE, worlds, random
    )

    missing_dependency = EffectSpec(
        name="misdeclared-choice-special-damage",
        op=SPECIAL_DAMAGE.op,
        reads=field_mask(StateField.OWN_HP),
        writes=SPECIAL_DAMAGE.writes,
        random=SPECIAL_DAMAGE.random,
    )
    negative_control_detected = False
    try:
        collapse_worlds(missing_dependency, worlds, random)
    except DependencyViolation:
        negative_control_detected = True

    passed = (
        protect_exact
        and damage_exact
        and protect.transition_count == 1
        and damage.transition_count == 2
        and negative_control_detected
    )
    return {
        "schema": "azelficoast.simulator-dependency-experiment",
        "schema_version": 1,
        "world_count": len(worlds),
        "protect": {
            "unique_transitions": protect.transition_count,
            "reduction_factor": protect.reduction_factor,
            "exact": protect_exact,
        },
        "choice_special_damage": {
            "unique_transitions": damage.transition_count,
            "reduction_factor": damage.reduction_factor,
            "exact": damage_exact,
        },
        "negative_control_detected": negative_control_detected,
        "passed": passed,
    }


def main() -> int:
    result = run_experiment()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
