from __future__ import annotations

import copy

import pytest

from azelficoast.research.depth import DepthStudyError, summarize_pair
from azelficoast.research.verification.real_belief_trace import analyze_oracle


def _deep_oracle() -> dict[str, object]:
    worlds = [
        {"world_id": "red", "weight": 0.5, "hidden": {"item": "red"}},
        {"world_id": "blue", "weight": 0.5, "hidden": {"item": "blue"}},
    ]
    transitions: list[dict[str, object]] = []
    for world in worlds:
        world_id = str(world["world_id"])
        for action in ("safe", "trap"):
            if action == "safe":
                leaf = {"hold": 2.0}
            elif world_id == "red":
                leaf = {"red": 3.0, "blue": 0.0}
            else:
                leaf = {"red": 0.0, "blue": 3.0}
            transitions.append(
                {
                    "world_id": world_id,
                    "action": action,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"root": "same"},
                            "successor": {"public": "same"},
                            "continuation_transitions": {
                                "continue": [
                                    {
                                        "probability": 1.0,
                                        "observation": {"second": "same"},
                                        "continuations": leaf,
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "paired-depth",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["safe", "trap"],
        "dependency_candidates": ["item"],
        "declared_reads": {"safe": [], "trap": []},
        "transitions": transitions,
    }


def _shallow_oracle() -> dict[str, object]:
    document = copy.deepcopy(_deep_oracle())
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        outcomes = transition["outcomes"]
        assert isinstance(outcomes, list)
        outcome = outcomes[0]
        assert isinstance(outcome, dict)
        outcome.pop("continuation_transitions")
        outcome["continuations"] = {"continue": 2.0}
    return document


def test_paired_depth_endpoints_promote_bias_and_regret_over_policy_flip() -> None:
    shallow = analyze_oracle(_shallow_oracle())
    deeper = analyze_oracle(_deep_oracle())

    result = summarize_pair(shallow, deeper)

    assert result["shallow"]["max_strategy_fusion_value_advantage"] == 0.0
    assert result["shallow"]["determinization_public_regret"] == 0.0
    assert result["shallow"]["policy_disagreement"] is False

    assert result["deeper"]["max_strategy_fusion_value_advantage"] == 1.5
    assert result["deeper"]["determinization_public_regret"] == 0.5
    assert result["deeper"]["policy_disagreement"] is True

    assert result["paired_change"] == {
        "max_strategy_fusion_value_advantage": 1.5,
        "determinization_public_regret": 0.5,
    }


def test_paired_depth_requires_same_root_action_set() -> None:
    shallow = analyze_oracle(_shallow_oracle())
    deeper_document = _deep_oracle()
    deeper_document["legal_actions"] = ["safe"]
    transitions = deeper_document["transitions"]
    assert isinstance(transitions, list)
    deeper_document["transitions"] = [
        transition
        for transition in transitions
        if isinstance(transition, dict) and transition["action"] == "safe"
    ]
    deeper = analyze_oracle(deeper_document)

    with pytest.raises(DepthStudyError, match="identical root actions"):
        summarize_pair(shallow, deeper)


def test_paired_depth_requires_exactly_one_added_horizon() -> None:
    shallow = analyze_oracle(_shallow_oracle())
    deeper = analyze_oracle(_shallow_oracle())

    with pytest.raises(DepthStudyError, match="exactly two"):
        summarize_pair(shallow, deeper)
