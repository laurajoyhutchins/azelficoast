from __future__ import annotations

import copy

import pytest

from azelficoast.belief_evaluator import BeliefEvaluatorSpec, BeliefPrediction
from azelficoast.matched_comparison import freeze_packet, settle_packet
from azelficoast.matched_oracle_search import (
    MatchedSearchExecutionError,
    execute_method,
)


class _FakeEvaluator:
    spec = BeliefEvaluatorSpec(
        public_width=8,
        world_width=8,
        action_width=8,
        hidden_width=8,
        world_hidden_width=8,
    )
    identity = {
        "schema": "azelficoast.belief-policy-value-evaluator",
        "schema_version": 1,
        "checkpoint_digest": "sha256:" + "a" * 64,
        "observability": "public_belief_only",
        "architecture": "weighted_deep_sets_policy_value",
        "spec": spec.as_dict(),
    }

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, inputs) -> BeliefPrediction:
        self.calls += 1
        value = 1.0 - max(inputs.world_weights)
        probability = 1.0 / len(inputs.legal_actions)
        probabilities = tuple(probability for _ in inputs.legal_actions)
        return BeliefPrediction(
            value=value,
            legal_actions=inputs.legal_actions,
            probabilities=probabilities,
            selected_action=min(inputs.legal_actions),
            policy_margin=0.0,
            policy_entropy_bits=0.0,
        )


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
        common_successor = {
            "turn": 9,
            "request_state": "move",
            "p1": [{"species": "rotom", "hp": 61, "maxhp": 100}],
            "p2_active": {"species": "garchomp", "hp": "<unchanged>"},
        }
        transitions.extend(
            [
                {
                    "world_id": world["world_id"],
                    "action": "wait",
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"kind": "same"},
                            "successor": common_successor,
                            "continuations": {
                                "fast": 4000.0 if item == "Specs" else -4000.0,
                                "safe": 1000.0,
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
                            "successor": {
                                **common_successor,
                                "revealed_item": item,
                            },
                            "continuations": {
                                "fast": 3000.0 if item == "Specs" else -3000.0,
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
        "schema_version": 2,
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
        "evaluator": dict(_FakeEvaluator.identity),
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


def test_executor_uses_frozen_learned_evaluator_and_accounts_calls() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    det_evaluator = _FakeEvaluator()
    info_evaluator = _FakeEvaluator()

    det = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=oracle,
        method="determinization",
        evaluator=det_evaluator,
    )
    info = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=oracle,
        method="information_set",
        evaluator=info_evaluator,
    )

    assert det["root_values"] == {"wait": 0.0, "reveal": 0.0}
    assert info["root_values"] == {"wait": 0.5, "reveal": 0.0}
    assert det["chosen_action"] == "reveal"
    assert info["chosen_action"] == "wait"
    assert det["consumed"] == info["consumed"] == 4
    assert det["evaluator_calls"] == det_evaluator.calls == 4
    assert info["evaluator_calls"] == info_evaluator.calls == 3
    assert (
        det["evaluator_checkpoint_digest"]
        == info["evaluator_checkpoint_digest"]
        == _FakeEvaluator.identity["checkpoint_digest"]
    )
    assert (
        det["transition_oracle_digest"]
        == info["transition_oracle_digest"]
    )

    settled = settle_packet(packet=packet, receipts=[det, info])
    assert settled["matched_authorized_compute"] is True
    assert settled["matched_evaluator_checkpoint"] is True
    assert settled["compute_consumed"] == {
        "determinization": 4,
        "information_set": 4,
    }
    assert settled["evaluator_calls"] == {
        "determinization": 4,
        "information_set": 3,
    }
    assert settled["policy_disagreement"] is True


def test_executor_ignores_historical_leaf_utility_values() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    baseline = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=oracle,
        method="information_set",
        evaluator=_FakeEvaluator(),
    )

    changed = copy.deepcopy(oracle)
    transitions = changed["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        outcomes = transition["outcomes"]
        assert isinstance(outcomes, list)
        for outcome in outcomes:
            assert isinstance(outcome, dict)
            continuations = outcome["continuations"]
            assert isinstance(continuations, dict)
            for action in list(continuations):
                continuations[action] = 1e12 if action == "fast" else -1e12

    changed_receipt = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=changed,
        method="information_set",
        evaluator=_FakeEvaluator(),
    )

    assert changed_receipt["root_values"] == baseline["root_values"]
    assert changed_receipt["chosen_action"] == baseline["chosen_action"]
    assert changed_receipt["evaluator_calls"] == baseline["evaluator_calls"]


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
            evaluator=_FakeEvaluator(),
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
            evaluator=_FakeEvaluator(),
        )


def test_executor_rejects_evaluator_checkpoint_drift() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    evaluator = _FakeEvaluator()
    evaluator.identity = {
        **_FakeEvaluator.identity,
        "checkpoint_digest": "sha256:" + "b" * 64,
    }

    with pytest.raises(
        MatchedSearchExecutionError,
        match="loaded evaluator identity differs",
    ):
        execute_method(
            packet=packet,
            posterior=posterior,
            oracle=oracle,
            method="information_set",
            evaluator=evaluator,
        )
    assert evaluator.calls == 0


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
            evaluator=_FakeEvaluator(),
        )
