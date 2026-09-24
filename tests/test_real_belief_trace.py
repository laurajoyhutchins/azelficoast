from __future__ import annotations

import copy

import pytest

from azelficoast.real_belief_trace import BeliefTraceError, analyze_oracle


def _oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": "scarf-a",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Scarf", "noise": 1},
        },
        {
            "world_id": "scarf-b",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Scarf", "noise": 2},
        },
        {
            "world_id": "specs-a",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Specs", "noise": 1},
        },
        {
            "world_id": "specs-b",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Specs", "noise": 2},
        },
    ]
    transitions = []
    for world in worlds:
        item = world["hidden"]["opponent.active.item"]
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "wait",
                "outcomes": [
                    {
                        "probability": 0.5,
                        "observation": {"kind": "same", "roll": "low"},
                        "successor": {"hp": 10},
                        "continuations": {
                            "fast": 4 if item == "Specs" else -4,
                            "safe": 1,
                        },
                    },
                    {
                        "probability": 0.5,
                        "observation": {"kind": "same", "roll": "high"},
                        "successor": {"hp": 9},
                        "continuations": {
                            "fast": 4 if item == "Specs" else -4,
                            "safe": 1,
                        },
                    },
                ],
            }
        )
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "reveal",
                "outcomes": [
                    {
                        "probability": 1.0,
                        "observation": {"kind": item},
                        "successor": {"hp": 9},
                        "continuations": {
                            "fast": 3 if item == "Specs" else -3,
                            "safe": 0,
                        },
                    }
                ],
            }
        )
    return {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "real",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["wait", "reveal"],
        "dependency_candidates": ["opponent.active.item"],
        # wait's immediate transition is item-independent even though its future
        # continuation value is not; reveal's observation depends on the item.
        "declared_reads": {"wait": [], "reveal": ["opponent.active.item"]},
        "transitions": transitions,
    }


def test_information_set_policy_cannot_branch_on_hidden_world_or_future_chance() -> None:
    result = analyze_oracle(_oracle())

    assert result["determinization"]["chosen_action"] == "wait"
    assert result["public_belief"]["chosen_action"] == "reveal"
    assert result["policy_disagreement"] is True
    assert result["hypothesis_supported"] is True
    assert result["experiment_valid"] is True
    assert result["strategy_fusion_observation_count"] > 0

    wait = next(row for row in result["actions"] if row["action"] == "wait")
    assert wait["dependency_signature"]["empirically_required_reads"] == []
    assert wait["dependency_signature"]["classes_out"] == 1
    assert wait["dependency_signature"]["observable_classes_out"] == 2

    # Each chance observation still retains both hidden item worlds. The policy
    # may react to the observed roll, but not to which hidden item generated it.
    assert all(
        {member["world_id"].split("-")[0] for member in belief["members"]}
        == {"scarf", "specs"}
        for belief in wait["successor_beliefs"]
    )


def test_missing_declared_dependency_fails_closed() -> None:
    document = _oracle()
    declared = document["declared_reads"]
    assert isinstance(declared, dict)
    declared["reveal"] = []

    with pytest.raises(BeliefTraceError, match="missing from declaration"):
        analyze_oracle(document)


def test_incomplete_transition_matrix_fails_closed() -> None:
    document = copy.deepcopy(_oracle())
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    transitions.pop()

    with pytest.raises(BeliefTraceError, match="omitted root transitions"):
        analyze_oracle(document)

def test_determinization_cannot_condition_on_unobserved_chance() -> None:
    document = {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "chance-alias",
        "showdown_commit": "pinned",
        "worlds": [
            {
                "world_id": "only-world",
                "weight": 1.0,
                "hidden": {},
            }
        ],
        "legal_actions": ["hold"],
        "dependency_candidates": [],
        "declared_reads": {"hold": []},
        "transitions": [
            {
                "world_id": "only-world",
                "action": "hold",
                "outcomes": [
                    {
                        "probability": 0.5,
                        "observation": {"same": True},
                        "successor": {"public": "same"},
                        "continuations": {"a": 4, "b": 0},
                    },
                    {
                        "probability": 0.5,
                        "observation": {"same": True},
                        "successor": {"public": "same"},
                        "continuations": {"a": 0, "b": 4},
                    },
                ],
            }
        ],
    }

    result = analyze_oracle(document)

    assert result["determinization"]["value"] == 2
    assert result["public_belief"]["value"] == 2
    assert result["policy_disagreement"] is False
    assert result["strategy_fusion_observation_count"] == 0

    [action] = result["actions"]
    assert action["determinization_continuations"] == [
        {
            "world_id": "only-world",
            "outcome_indices": [0, 1],
            "observation_hash": action["determinization_continuations"][0][
                "observation_hash"
            ],
            "choice": "a",
            "value": 2.0,
        }
    ]
    [public] = action["public_belief_continuations"]
    assert public["world_aware_choices"] == ["a"]
    assert public["strategy_fusion_possible"] is False

