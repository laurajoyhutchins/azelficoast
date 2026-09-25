from __future__ import annotations

import copy

import pytest

from azelficoast.search.decision_relevance import (
    DecisionRelevanceError,
    analyze_quotiented_oracle,
    decision_relevance_quotient,
)
from azelficoast.real_belief_trace import analyze_oracle


def _strategy_fusion_oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": f"{item}-{noise}",
            "weight": 0.25,
            "hidden": {
                "opponent.active.item": item,
                "noise": noise,
            },
        }
        for item in ("Scarf", "Specs")
        for noise in (1, 2)
    ]
    transitions = []
    for world in worlds:
        item = world["hidden"]["opponent.active.item"]
        transitions.extend(
            [
                {
                    "world_id": world["world_id"],
                    "action": "wait",
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"same": True},
                            "successor": {"same": True},
                            "continuations": {
                                "fast": 4.0 if item == "Specs" else -4.0,
                                "safe": 1.0,
                            },
                        }
                    ],
                },
                {
                    "world_id": world["world_id"],
                    "action": "reveal",
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"item": item},
                            "successor": {"revealed": True},
                            "continuations": {
                                "fast": 3.0 if item == "Specs" else -3.0,
                                "safe": 0.0,
                            },
                        }
                    ],
                },
            ]
        )
    return {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "quotient-test",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["wait", "reveal"],
        "dependency_candidates": ["opponent.active.item", "noise"],
        "declared_reads": {
            "wait": ["opponent.active.item"],
            "reveal": ["opponent.active.item"],
        },
        "transitions": transitions,
    }


def test_exact_quotient_removes_irrelevant_hidden_dimension() -> None:
    document = _strategy_fusion_oracle()

    full = analyze_oracle(document)
    quotient_trace, certificate = analyze_quotiented_oracle(document)

    assert certificate["decision_fields"] == ["opponent.active.item"]
    assert certificate["worlds_in"] == 4
    assert certificate["classes_out"] == 2
    assert certificate["world_reduction"] == 2
    assert certificate["reduction_fraction"] == 0.5
    assert certificate["belief_branching_required"] is True
    assert all(len(row["member_world_ids"]) == 2 for row in certificate["classes"])

    assert quotient_trace["source_world_count"] == 4
    assert quotient_trace["world_count"] == 2
    assert quotient_trace["public_belief"] == full["public_belief"]
    assert quotient_trace["determinization"] == full["determinization"]
    assert quotient_trace["policy_disagreement"] == full["policy_disagreement"]
    assert (
        quotient_trace["strategy_fusion_observation_count"]
        == full["strategy_fusion_observation_count"]
    )


def test_exact_quotient_collapses_belief_when_all_hidden_state_is_irrelevant() -> None:
    document = {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "irrelevant",
        "showdown_commit": "pinned",
        "worlds": [
            {"world_id": "a", "weight": 0.4, "hidden": {"noise": 1}},
            {"world_id": "b", "weight": 0.6, "hidden": {"noise": 2}},
        ],
        "legal_actions": ["hold"],
        "dependency_candidates": ["noise"],
        "declared_reads": {"hold": []},
        "transitions": [
            {
                "world_id": world_id,
                "action": "hold",
                "outcomes": [
                    {
                        "probability": 1.0,
                        "observation": {"same": True},
                        "successor": {"same": True},
                        "terminal_utility": 2.0,
                    }
                ],
            }
            for world_id in ("a", "b")
        ],
    }

    certificate, quotient = decision_relevance_quotient(document)
    trace, runtime_certificate = analyze_quotiented_oracle(document)

    assert certificate == runtime_certificate
    assert certificate["decision_fields"] == []
    assert certificate["classes_out"] == 1
    assert certificate["belief_branching_required"] is False
    assert len(quotient["worlds"]) == 1
    assert quotient["worlds"][0]["hidden"] == {}
    assert trace["source_world_count"] == 2
    assert trace["world_count"] == 1
    assert trace["public_belief"]["chosen_action"] == "hold"
    assert trace["public_belief"]["value"] == 2.0


def test_quotient_is_invariant_to_chance_outcome_enumeration_order() -> None:
    document = {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "chance-order",
        "showdown_commit": "pinned",
        "worlds": [
            {"world_id": "a", "weight": 0.5, "hidden": {"noise": 1}},
            {"world_id": "b", "weight": 0.5, "hidden": {"noise": 2}},
        ],
        "legal_actions": ["hold"],
        "dependency_candidates": ["noise"],
        "declared_reads": {"hold": []},
        "transitions": [],
    }
    outcomes = [
        {
            "probability": 0.5,
            "observation": {"roll": "low"},
            "successor": {"hp": 10},
            "continuations": {"a": 2.0, "b": 0.0},
        },
        {
            "probability": 0.5,
            "observation": {"roll": "high"},
            "successor": {"hp": 9},
            "continuations": {"a": 0.0, "b": 2.0},
        },
    ]
    document["transitions"] = [
        {"world_id": "a", "action": "hold", "outcomes": outcomes},
        {
            "world_id": "b",
            "action": "hold",
            "outcomes": list(reversed(copy.deepcopy(outcomes))),
        },
    ]

    certificate, _ = decision_relevance_quotient(document)

    assert certificate["decision_fields"] == []
    assert certificate["classes_out"] == 1


def test_quotient_fails_closed_when_declared_candidates_cannot_explain_semantics() -> None:
    document = _strategy_fusion_oracle()
    document["dependency_candidates"] = ["noise"]
    for world in document["worlds"]:
        world["hidden"]["noise"] = 1

    with pytest.raises(
        DecisionRelevanceError,
        match="cannot explain bounded decision semantics",
    ):
        decision_relevance_quotient(document)
