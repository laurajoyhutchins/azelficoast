from __future__ import annotations

import pytest

from azelficoast.matched_pilot_evaluator import (
    MatchedPilotEvaluatorError,
    expand_posterior,
    training_targets,
)


def _source() -> dict[str, object]:
    return {
        "schema": "azelficoast.frozen-evaluator-source-state",
        "schema_version": 1,
        "eventual_battle_outcome": 1.0,
        "public_state": {
            "legal_actions": ["attack", "switch"],
            "turn": 8,
        },
        "search_target": {
            "selected_action": "attack",
            "value": 4.5,
        },
        "posterior_components": [
            {
                "hidden": {
                    "opponent.active.item": "Choice Band",
                    "opponent.active.ability": "Unseen Fist",
                },
                "provenance": {
                    "generator_rounds": 2048,
                    "generator_count": 12,
                },
                "weight": 0.25,
                "exact_hp_values": [51, 52, 53],
            },
            {
                "hidden": {
                    "opponent.active.item": "Choice Scarf",
                    "opponent.active.ability": "Unseen Fist",
                },
                "provenance": {
                    "generator_rounds": 2048,
                    "generator_count": 36,
                },
                "weight": 0.75,
                "exact_hp_values": [51, 52, 53],
            },
        ],
    }


def test_expand_posterior_materializes_compact_joint_worlds() -> None:
    posterior = expand_posterior(_source())

    assert posterior["conditioned_on_public_history"] is True
    assert posterior["realized_hidden_state_revealed"] is False
    assert len(posterior["worlds"]) == 6
    assert {
        world["hidden"]["opponent.active.exact_hp"]
        for world in posterior["worlds"]
    } == {51, 52, 53}
    assert {
        world["hidden"]["opponent.active.item"]
        for world in posterior["worlds"]
    } == {"Choice Band", "Choice Scarf"}


def test_training_targets_combine_outcome_and_normalized_exact_search() -> None:
    selected, value, policy, metadata = training_targets(_source())

    assert selected == "attack"
    assert value == pytest.approx(0.7857142857142857)
    assert policy == (1.0, 0.0)
    assert metadata["search_return"] == 4.5
    assert metadata["normalized_search_return"] == pytest.approx(
        0.5714285714285714
    )


def test_training_targets_reject_nonlegal_search_action() -> None:
    source = _source()
    target = source["search_target"]
    assert isinstance(target, dict)
    target["selected_action"] = "cheat"

    with pytest.raises(MatchedPilotEvaluatorError, match="nonlegal"):
        training_targets(source)
