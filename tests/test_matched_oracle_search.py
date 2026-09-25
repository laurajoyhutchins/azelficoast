from __future__ import annotations

import copy

import pytest

from azelficoast.matched_comparison import freeze_packet, settle_packet
from azelficoast.matched_oracle_search import (
    MatchedSearchExecutionError,
    execute_method,
)
from azelficoast.real_belief_trace import analyze_oracle


def _oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": "scarf",
            "weight": 0.5,
            "hidden": {"opponent.active.item": "Scarf"},
        },
        {
            "world_id": "specs",
            "weight": 0.5,
            "hidden": {"opponent.active.item": "Specs"},
        },
    ]
    transitions: list[dict[str, object]] = []
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
                            "observation": {"kind": "same"},
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
                            "observation": {"kind": item},
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
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["wait", "reveal"],
        "dependency_candidates": ["opponent.active.item"],
        "declared_reads": {"wait": [], "reveal": ["opponent.active.item"]},
        "transitions": transitions,
    }


def _posterior(oracle: dict[str, object]) -> dict[str, object]:
    worlds = oracle["worlds"]
    assert isinstance(worlds, list)
    return {
        "treatment": "generator_faithful",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": copy.deepcopy(worlds),
    }


def _plan(*, limit: int = 4) -> dict[str, object]:
    return {
        "schema": "azelficoast.matched-search-comparison-plan",
        "schema_version": 1,
        "posterior_treatments": ["generator_faithful"],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": limit,
        },
        "opponent_model": "fixed_observed_response",
        "depths": [1],
        "confirmatory_predictors": [
            "hidden_item_entropy_bits",
            "persistent_branch_count",
        ],
        "cluster_unit": "battle_tag",
        "showdown_commit": "pinned",
        "inference": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1729,
        },
    }


def _packet(
    oracle: dict[str, object],
    posterior: dict[str, object],
    *,
    limit: int = 4,
) -> dict[str, object]:
    return freeze_packet(
        plan=_plan(limit=limit),
        state={
            "fixture_id": "fixture",
            "battle_tag": "battle",
            "public_state": {"turn": 8},
            "legal_actions": ["wait", "reveal"],
            "predictors": {
                "hidden_item_entropy_bits": 1.0,
                "persistent_branch_count": 1,
            },
        },
        posterior=posterior,
        posterior_treatment="generator_faithful",
        depth=1,
    )


def test_independent_receipts_match_existing_exact_analyzer() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)

    det = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=oracle,
        method="determinization",
    )
    info = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=oracle,
        method="information_set",
    )
    reference = analyze_oracle(oracle)

    assert det["root_values"] == reference["determinization"]["root_values"]
    assert info["root_values"] == reference["public_belief"]["root_values"]
    assert det["chosen_action"] == reference["determinization"]["chosen_action"]
    assert info["chosen_action"] == reference["public_belief"]["chosen_action"]
    assert det["consumed"] == info["consumed"] == 4
    assert (
        det["transition_oracle_digest"]
        == info["transition_oracle_digest"]
    )

    settled = settle_packet(packet=packet, receipts=[det, info])
    assert settled["matched_authorized_compute"] is True
    assert settled["compute_consumed"] == {
        "determinization": 4,
        "information_set": 4,
    }
    assert settled["policy_disagreement"] is True


def test_executor_fails_before_search_when_budget_cannot_cover_frozen_matrix() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior, limit=3)

    with pytest.raises(MatchedSearchExecutionError, match="requires 4 transitions"):
        execute_method(
            packet=packet,
            posterior=posterior,
            oracle=oracle,
            method="determinization",
        )


def test_executor_rejects_posterior_hidden_state_drift() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    drifted = copy.deepcopy(posterior)
    worlds = drifted["worlds"]
    assert isinstance(worlds, list)
    first = worlds[0]
    assert isinstance(first, dict)
    hidden = first["hidden"]
    assert isinstance(hidden, dict)
    hidden["opponent.active.item"] = "Band"

    with pytest.raises(
        MatchedSearchExecutionError,
        match="posterior artifact does not match frozen packet",
    ):
        execute_method(
            packet=packet,
            posterior=drifted,
            oracle=oracle,
            method="information_set",
        )


def test_depth_one_executor_rejects_deeper_evidence() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    transitions = oracle["transitions"]
    assert isinstance(transitions, list)
    first = transitions[0]
    assert isinstance(first, dict)
    outcomes = first["outcomes"]
    assert isinstance(outcomes, list)
    outcome = outcomes[0]
    assert isinstance(outcome, dict)
    outcome["continuation_transitions"] = {
        "continue": [
            {
                "probability": 1.0,
                "observation": {"later": True},
                "continuations": {"finish": 0.0},
            }
        ]
    }
    outcome.pop("continuations")

    with pytest.raises(
        MatchedSearchExecutionError,
        match="cannot consume deeper",
    ):
        execute_method(
            packet=packet,
            posterior=posterior,
            oracle=oracle,
            method="determinization",
        )
