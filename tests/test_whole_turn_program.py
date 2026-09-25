from __future__ import annotations

import copy

import pytest

from azelficoast.real_belief_trace import analyze_oracle
from azelficoast.whole_turn_program import (
    WholeTurnProgramError,
    compile_whole_turn_programs,
    execute_whole_turn_program,
    program_for_action,
    verify_whole_turn_program_set,
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


def _instrumented_branching_oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": f"{mode}-{detail}",
            "weight": 0.25,
            "hidden": {"mode": mode, "detail": detail},
        }
        for mode in ("a", "b")
        for detail in ("x", "y")
    ]
    transitions: list[dict[str, object]] = []
    for world in worlds:
        hidden = world["hidden"]
        assert isinstance(hidden, dict)
        mode = str(hidden["mode"])
        detail = str(hidden["detail"])
        reads = ["mode"] if mode == "a" else ["mode", "detail"]
        successor = (
            {"branch": "a"}
            if mode == "a"
            else {"branch": "b", "detail": detail}
        )
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "turn",
                "outcomes": [
                    {
                        "probability": 1.0,
                        "observation": {"branch": mode},
                        "successor": successor,
                        "transition_reads": reads,
                    }
                ],
            }
        )
    return {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "instrumented-branching",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["turn"],
        "dependency_candidates": ["mode", "detail"],
        "declared_reads": {"turn": ["mode", "detail"]},
        "transitions": transitions,
    }


def test_dynamic_read_refinement_keeps_branch_local_dependencies_local() -> None:
    program = program_for_action(
        compile_whole_turn_programs(
            _instrumented_branching_oracle(),
            partition_strategy="dynamic_reads",
        ),
        "turn",
    )

    assert program["partition_method"] == "dynamic-read-refinement"
    assert program["worlds_in"] == 4
    assert program["classes_out"] == 3
    assert program["representative_world_count"] == 3
    assert program["dependency_fields"] == ["detail", "mode"]

    members = sorted(
        sorted(row["member_world_ids"])
        for row in program["classes"]
    )
    assert members == [["a-x", "a-y"], ["b-x"], ["b-y"]]

    a_class = next(
        row for row in program["classes"] if row["member_world_ids"] == ["a-x", "a-y"]
    )
    assert a_class["read_fields"] == ["mode"]


def test_dynamic_read_refinement_fails_closed_on_missing_instrumentation() -> None:
    oracle = _instrumented_branching_oracle()
    transitions = oracle["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        if transition["world_id"] != "b-x":
            continue
        outcomes = transition["outcomes"]
        assert isinstance(outcomes, list)
        outcome = outcomes[0]
        assert isinstance(outcome, dict)
        outcome["transition_reads"] = ["mode"]

    with pytest.raises(
        WholeTurnProgramError,
        match="instrumented transition reads are incomplete",
    ):
        compile_whole_turn_programs(
            oracle,
            partition_strategy="dynamic_reads",
        )


def test_direct_oracle_verifies_read_refined_program() -> None:
    oracle = _instrumented_branching_oracle()
    program_set = compile_whole_turn_programs(
        oracle,
        partition_strategy="dynamic_reads",
    )

    certificate = verify_whole_turn_program_set(program_set, oracle)

    assert certificate["world_count"] == 4
    assert certificate["action_count"] == 1
    assert certificate["verified_class_count"] == 3
    assert certificate["representative_world_executions"] == 3
    assert certificate["exhaustive_world_action_product"] == 4
    assert certificate["saved_world_action_evaluations"] == 1
    assert certificate["reduction_fraction"] == pytest.approx(0.25)


def test_direct_oracle_rejects_semantically_invalid_lazy_class() -> None:
    oracle = _instrumented_branching_oracle()
    program_set = compile_whole_turn_programs(
        oracle,
        partition_strategy="dynamic_reads",
    )
    program = program_for_action(program_set, "turn")
    classes = program["classes"]
    assert isinstance(classes, list)

    bx = next(row for row in classes if row["member_world_ids"] == ["b-x"])
    bx["member_world_ids"] = ["b-x", "b-y"]
    program["classes"] = [
        row for row in classes if row["member_world_ids"] != ["b-y"]
    ]
    program["classes_out"] = 2

    with pytest.raises(
        WholeTurnProgramError,
        match="merges semantically different world",
    ):
        verify_whole_turn_program_set(program_set, oracle)


def test_semantic_analysis_does_not_inherit_conservative_runtime_reads() -> None:
    oracle = _instrumented_branching_oracle()

    semantic = program_for_action(compile_whole_turn_programs(oracle), "turn")
    read_refined = program_for_action(
        compile_whole_turn_programs(
            oracle,
            partition_strategy="dynamic_reads",
        ),
        "turn",
    )

    assert semantic["partition_method"] == "finite-support-minimal-semantics"
    assert semantic["classes_out"] == 4
    assert read_refined["partition_method"] == "dynamic-read-refinement"
    assert read_refined["classes_out"] == 3


def test_whole_turn_program_distinguishes_wait_from_terminal_leaf() -> None:
    oracle = _oracle()
    transitions = oracle["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        if transition["action"] != "protect":
            continue
        outcomes = transition["outcomes"]
        assert isinstance(outcomes, list)
        outcome = outcomes[0]
        assert isinstance(outcome, dict)
        outcome["observation"] = {"request": {"wait": True}}
        outcome["terminal_utility"] = 123.0

    program = program_for_action(compile_whole_turn_programs(oracle), "protect")

    assert {
        tuple(outcome["legal_actions"])
        for row in program["classes"]
        for outcome in row["outcomes"]
    } == {("<wait>",)}
