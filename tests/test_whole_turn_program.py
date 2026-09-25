from __future__ import annotations

import copy

import pytest

from azelficoast.real_belief_trace import analyze_oracle
from azelficoast.whole_turn_program import (
    WholeTurnProgramError,
    compile_whole_turn_programs,
    execute_whole_turn_program,
    program_for_action,
)


def _oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": f"{item}-{noise}",
            "weight": 0.25,
            "hidden": {
                "opponent.active.item": item,
                "irrelevant.noise": noise,
            },
        }
        for item in ("Scarf", "Specs")
        for noise in (1, 2)
    ]
    transitions: list[dict[str, object]] = []
    for world in worlds:
        hidden = world["hidden"]
        assert isinstance(hidden, dict)
        item = str(hidden["opponent.active.item"])
        noise = int(hidden["irrelevant.noise"])
        transitions.extend(
            [
                {
                    "world_id": world["world_id"],
                    "action": "protect",
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"kind": "blocked"},
                            "successor": {"own_hp": 100, "opponent_hp": 100},
                            # Value evidence is intentionally not transition semantics.
                            "continuations": {"later": float(noise)},
                        }
                    ],
                },
                {
                    "world_id": world["world_id"],
                    "action": "attack",
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"kind": "damage"},
                            "successor": {
                                "own_hp": 100,
                                "opponent_hp": 70 if item == "Scarf" else 50,
                            },
                            "continuations": {"later": float(noise)},
                        }
                    ],
                },
            ]
        )
    return {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "whole-turn-fixture",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["protect", "attack"],
        "dependency_candidates": [
            "opponent.active.item",
            "irrelevant.noise",
        ],
        "declared_reads": {
            "protect": ["opponent.active.item", "irrelevant.noise"],
            "attack": ["opponent.active.item", "irrelevant.noise"],
        },
        "transitions": transitions,
    }


def test_compiler_separates_whole_turn_semantics_from_value_evidence() -> None:
    program_set = compile_whole_turn_programs(_oracle())

    protect = program_for_action(program_set, "protect")
    attack = program_for_action(program_set, "attack")

    assert protect["dependency_fields"] == []
    assert protect["classes_out"] == 1
    assert attack["dependency_fields"] == ["opponent.active.item"]
    assert attack["classes_out"] == 2
    assert attack["worlds_in"] == 4
    assert attack["reduction_fraction"] == 0.5
    assert str(attack["effect_signature"]).startswith("sha256:")

    trace = analyze_oracle(_oracle())
    reports = {row["action"]: row for row in trace["actions"]}
    assert reports["protect"]["dependency_signature"]["classes_out"] == 1
    assert reports["attack"]["dependency_signature"]["classes_out"] == 2
    assert reports["attack"]["dependency_signature"]["effect_signature"] == (
        attack["effect_signature"]
    )


def test_compiled_program_applies_new_posterior_mass_without_reexecuting_members() -> None:
    program_set = compile_whole_turn_programs(_oracle())
    posterior = {
        "Scarf-1": 1.0,
        "Scarf-2": 2.0,
        "Specs-1": 3.0,
        "Specs-2": 4.0,
    }

    execution = execute_whole_turn_program(
        program_set,
        action="attack",
        posterior=posterior,
    )

    assert execution["logical_world_count"] == 4
    assert execution["execution_class_count"] == 2
    assert execution["transition_evaluations"] == 2
    assert sum(edge["mass"] for edge in execution["edges"]) == pytest.approx(1.0)

    by_hp = {
        int(edge["successor"]["opponent_hp"]): float(edge["mass"])
        for edge in execution["edges"]
    }
    assert by_hp == pytest.approx({70: 0.3, 50: 0.7})


def test_compiler_is_invariant_to_chance_enumeration_order() -> None:
    oracle = _oracle()
    transitions = oracle["transitions"]
    assert isinstance(transitions, list)

    for transition in transitions:
        assert isinstance(transition, dict)
        if transition["action"] != "protect":
            continue
        outcome = transition["outcomes"][0]
        assert isinstance(outcome, dict)
        transition["outcomes"] = [
            {
                **copy.deepcopy(outcome),
                "probability": 0.5,
                "observation": {"kind": "blocked", "sample": 1},
            },
            {
                **copy.deepcopy(outcome),
                "probability": 0.5,
                "observation": {"kind": "blocked", "sample": 2},
            },
        ]

    first = compile_whole_turn_programs(oracle)
    reversed_oracle = copy.deepcopy(oracle)
    reversed_transitions = reversed_oracle["transitions"]
    assert isinstance(reversed_transitions, list)
    for transition in reversed_transitions:
        if isinstance(transition, dict) and transition["action"] == "protect":
            transition["outcomes"] = list(reversed(transition["outcomes"]))

    second = compile_whole_turn_programs(reversed_oracle)
    assert program_for_action(first, "protect") == program_for_action(
        second,
        "protect",
    )


def test_execution_rejects_posterior_support_drift() -> None:
    program_set = compile_whole_turn_programs(_oracle())

    with pytest.raises(
        WholeTurnProgramError,
        match="support differs",
    ):
        execute_whole_turn_program(
            program_set,
            action="attack",
            posterior={
                "Scarf-1": 0.5,
                "Specs-1": 0.5,
            },
        )
