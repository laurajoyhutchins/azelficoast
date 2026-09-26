from __future__ import annotations

import copy

import pytest

from azelficoast.belief.evaluator import BeliefEvaluatorSpec, BeliefPrediction
from azelficoast.research.matched_comparison import (\n    MatchedComparisonError,\n    freeze_packet,\n    settle_packet,\n)
from azelficoast.research.matched_search import (
    MatchedSearchExecutionError,
    execute_method,
)
from azelficoast.core.whole_turn_program import compile_whole_turn_programs


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


class _BatchFakeEvaluator(_FakeEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.batches = 0

    def predict_values(self, inputs) -> tuple[float, ...]:
        self.batches += 1
        self.calls += len(inputs)
        return tuple(1.0 - max(row.world_weights) for row in inputs)


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
                            "hidden_reads": [],
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
                            "hidden_reads": ["opponent.active.item"],
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
        "schema": "azelficoast.core.transition-oracle",
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

    program = compile_whole_turn_programs(oracle)
    det = execute_method(
        packet=packet,
        posterior=posterior,
        transition_program=program,
        method="determinization",
        evaluator=det_evaluator,
    )
    info = execute_method(
        packet=packet,
        posterior=posterior,
        transition_program=program,
        method="information_set",
        evaluator=info_evaluator,
    )

    assert det["root_values"] == {"wait": 0.0, "reveal": 0.0}
    assert info["root_values"] == {"wait": 0.5, "reveal": 0.0}
    assert det["chosen_action"] == "reveal"
    assert info["chosen_action"] == "wait"
    assert det["consumed"] == info["consumed"] == 3
    assert det["evaluator_calls"] == det_evaluator.calls == 4
    assert info["evaluator_calls"] == info_evaluator.calls == 3
    assert det["evaluator_batches"] == 4
    assert info["evaluator_batches"] == 3
    assert (
        det["evaluator_checkpoint_digest"]
        == info["evaluator_checkpoint_digest"]
        == _FakeEvaluator.identity["checkpoint_digest"]
    )
    assert (
        det["transition_program_digest"]
        == info["transition_program_digest"]
    )

    settled = settle_packet(packet=packet, receipts=[det, info])
    assert settled["matched_authorized_compute"] is True
    assert settled["matched_evaluator_checkpoint"] is True
    assert settled["matched_transition_program"] is True
    assert settled["compute_consumed"] == {
        "determinization": 3,
        "information_set": 3,
    }
    assert settled["evaluator_calls"] == {
        "determinization": 4,
        "information_set": 3,
    }
    assert settled["evaluator_batches"] == {
        "determinization": 4,
        "information_set": 3,
    }
    assert det["resource_accounting"]["verified_execution_classes_consumed"] == 3
    assert det["resource_accounting"]["evaluator_calls"] == 4
    assert det["resource_accounting"]["transition_program_generation_included"] is False
    assert det["resource_accounting"]["transition_program_verification_included"] is False
    assert det["resource_accounting"]["posterior_construction_included"] is False
    assert det["resource_accounting"]["search_wall_ms"] >= 0.0
    assert settled["resource_accounting"]["determinization"] == det["resource_accounting"]
    assert settled["resource_accounting"]["information_set"] == info["resource_accounting"]
    assert settled["policy_disagreement"] is True


def test_executor_batches_complete_successor_frontier_without_changing_semantics() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    evaluator = _BatchFakeEvaluator()

    result = execute_method(
        packet=packet,
        posterior=posterior,
        transition_program=compile_whole_turn_programs(oracle),
        method="information_set",
        evaluator=evaluator,
    )

    assert result["root_values"] == {"wait": 0.5, "reveal": 0.0}
    assert result["chosen_action"] == "wait"
    assert result["evaluator_calls"] == evaluator.calls == 3
    assert result["evaluator_batches"] == evaluator.batches == 1
    assert result["resource_accounting"]["evaluator_calls"] == 3
    assert result["resource_accounting"]["evaluator_batches"] == 1


def test_executor_ignores_historical_leaf_utility_values() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    baseline_program = compile_whole_turn_programs(oracle)
    baseline = execute_method(
        packet=packet,
        posterior=posterior,
        transition_program=baseline_program,
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

    changed_program = compile_whole_turn_programs(changed)
    assert changed_program == baseline_program
    changed_receipt = execute_method(
        packet=packet,
        posterior=posterior,
        transition_program=changed_program,
        method="information_set",
        evaluator=_FakeEvaluator(),
    )

    assert changed_receipt["root_values"] == baseline["root_values"]
    assert changed_receipt["chosen_action"] == baseline["chosen_action"]
    assert changed_receipt["evaluator_calls"] == baseline["evaluator_calls"]


def test_executor_is_invariant_to_opaque_world_id_renaming_and_order() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)

    renamed = copy.deepcopy(oracle)
    renamed_posterior = copy.deepcopy(posterior)
    id_map = {"scarf": "omega", "specs": "alpha"}
    for source in (renamed, renamed_posterior):
        worlds = source["worlds"]
        assert isinstance(worlds, list)
        for world in worlds:
            assert isinstance(world, dict)
            world["world_id"] = id_map[str(world["world_id"])]
        worlds.reverse()
    transitions = renamed["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        transition["world_id"] = id_map[str(transition["world_id"])]
    transitions.reverse()
    renamed_packet = _packet(renamed, renamed_posterior)

    for method in ("determinization", "information_set"):
        baseline = execute_method(
            packet=packet,
            posterior=posterior,
            oracle=oracle,
            method=method,
            evaluator=_FakeEvaluator(),
        )
        result = execute_method(
            packet=renamed_packet,
            posterior=renamed_posterior,
            oracle=renamed,
            method=method,
            evaluator=_FakeEvaluator(),
        )

        assert result["root_values"] == baseline["root_values"]
        assert result["chosen_action"] == baseline["chosen_action"]


def test_executor_dependency_evidence_ignores_unread_hidden_perturbation() -> None:
    baseline_oracle = _oracle()
    baseline_posterior = _posterior(baseline_oracle)
    baseline_oracle["dependency_candidates"].append("noise")
    baseline_worlds = baseline_oracle["worlds"]
    baseline_posterior_worlds = baseline_posterior["worlds"]
    assert isinstance(baseline_worlds, list)
    assert isinstance(baseline_posterior_worlds, list)
    for index, (oracle_world, posterior_world) in enumerate(
        zip(baseline_worlds, baseline_posterior_worlds, strict=True)
    ):
        oracle_world["hidden"]["noise"] = index
        posterior_world["hidden"]["noise"] = index
    baseline_packet = _packet(baseline_oracle, baseline_posterior)
    baseline = execute_method(
        packet=baseline_packet,
        posterior=baseline_posterior,
        oracle=baseline_oracle,
        method="information_set",
        evaluator=_FakeEvaluator(),
    )

    changed_oracle = copy.deepcopy(baseline_oracle)
    changed_posterior = copy.deepcopy(baseline_posterior)
    changed_worlds = changed_oracle["worlds"]
    changed_posterior_worlds = changed_posterior["worlds"]
    assert isinstance(changed_worlds, list)
    assert isinstance(changed_posterior_worlds, list)
    for oracle_world, posterior_world in zip(
        changed_worlds, changed_posterior_worlds, strict=True
    ):
        oracle_world["hidden"]["noise"] += 100
        posterior_world["hidden"]["noise"] += 100
    changed_packet = _packet(changed_oracle, changed_posterior)
    changed = execute_method(
        packet=changed_packet,
        posterior=changed_posterior,
        oracle=changed_oracle,
        method="information_set",
        evaluator=_FakeEvaluator(),
    )

    assert changed["root_values"] == baseline["root_values"]
    assert changed["chosen_action"] == baseline["chosen_action"]
    assert changed["mechanics_evidence_digest"] == baseline["mechanics_evidence_digest"]


def test_executor_is_invariant_to_equivalent_support_splitting() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)

    split_oracle = copy.deepcopy(oracle)
    split_posterior = copy.deepcopy(posterior)
    for document in (split_oracle, split_posterior):
        worlds = document["worlds"]
        assert isinstance(worlds, list)
        first = next(world for world in worlds if world["world_id"] == "scarf")
        first["weight"] = 0.25
        worlds.append({**first, "world_id": "scarf-copy", "weight": 0.25})
    transitions = split_oracle["transitions"]
    assert isinstance(transitions, list)
    copies = [
        {**transition, "world_id": "scarf-copy"}
        for transition in transitions
        if transition["world_id"] == "scarf"
    ]
    transitions.extend(copies)
    split_packet = _packet(split_oracle, split_posterior)

    for method in ("determinization", "information_set"):
        baseline = execute_method(
            packet=packet,
            posterior=posterior,
            oracle=oracle,
            method=method,
            evaluator=_FakeEvaluator(),
        )
        result = execute_method(
            packet=split_packet,
            posterior=split_posterior,
            oracle=split_oracle,
            method=method,
            evaluator=_FakeEvaluator(),
        )

        assert result["root_values"] == baseline["root_values"]
        assert result["chosen_action"] == baseline["chosen_action"]


def test_executor_fails_before_search_when_budget_cannot_cover_frozen_matrix() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior, limit=2)
    program = compile_whole_turn_programs(oracle)

    with pytest.raises(MatchedSearchExecutionError, match="requires 3 transitions"):
        execute_method(
            packet=packet,
            posterior=posterior,
            transition_program=program,
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
            transition_program=compile_whole_turn_programs(oracle),
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
            transition_program=compile_whole_turn_programs(oracle),
            method="information_set",
            evaluator=evaluator,
        )
    assert evaluator.calls == 0


def test_executor_fails_closed_when_dependency_evidence_is_incomplete() -> None:
    oracle = _oracle()
    posterior = _posterior(oracle)
    packet = _packet(oracle, posterior)
    transitions = oracle["transitions"]
    assert isinstance(transitions, list)
    first_outcomes = transitions[0]["outcomes"]
    assert isinstance(first_outcomes, list)
    first_outcomes[0].pop("hidden_reads")

    with pytest.raises(MatchedSearchExecutionError, match="lacks hidden-read evidence"):
        execute_method(
            packet=packet,
            posterior=posterior,
            oracle=oracle,
            method="information_set",
            evaluator=_FakeEvaluator(),
        )

    missing_candidates = copy.deepcopy(_oracle())
    missing_posterior = _posterior(missing_candidates)
    missing_packet = _packet(missing_candidates, missing_posterior)
    missing_candidates.pop("dependency_candidates")
    with pytest.raises(MatchedSearchExecutionError, match="dependency_candidates list"):
        execute_method(
            packet=missing_packet,
            posterior=missing_posterior,
            oracle=missing_candidates,
            method="information_set",
            evaluator=_FakeEvaluator(),
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
            evaluator=_FakeEvaluator(),
        )


def _material_plan(depth: int) -> dict[str, object]:
    return {
        "schema": "azelficoast.matched-search-comparison-plan",
        "schema_version": 2,
        "posterior_treatments": ["generator_faithful"],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": 32,
        },
        "opponent_model": "fixed_observed_response",
        "depths": [depth],
        "confirmatory_predictors": [
            "hidden_item_entropy_bits",
            "persistent_branch_count",
        ],
        "cluster_unit": "battle_tag",
        "showdown_commit": "pinned",
        "chance_treatment": "shared_frozen_horizon_complete_oracle",
        "evaluator": {
            "schema": "azelficoast.material-utility-evaluator",
            "schema_version": 1,
            "checkpoint_digest": "sha256:" + "d" * 64,
            "observability": "public_belief_only",
            "architecture": "exact-material-utility",
            "spec": {
                "utility": (
                    "sum own team HP fractions minus opposing active HP fraction"
                )
            },
        },
        "inference": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1729,
        },
    }


def _material_oracle(depth: int) -> dict[str, object]:
    oracle = copy.deepcopy(_oracle())
    oracle["mechanics"] = {"continuation_decision_horizons": depth}
    if depth == 1:
        return oracle

    transitions = oracle["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        outcomes = transition["outcomes"]
        assert isinstance(outcomes, list)
        for outcome in outcomes:
            assert isinstance(outcome, dict)
            continuations = outcome.pop("continuations")
            assert isinstance(continuations, dict)
            outcome["continuation_transitions"] = {
                action: [
                    {
                        "probability": 1.0,
                        "observation": {"next": action},
                        "terminal_utility": value,
                    }
                ]
                for action, value in continuations.items()
            }
    return oracle


def _material_packet(
    oracle: dict[str, object],
    posterior: dict[str, object],
    *,
    depth: int,
) -> dict[str, object]:
    return freeze_packet(
        plan=_material_plan(depth),
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
        depth=depth,
    )


@pytest.mark.parametrize("depth", [1, 2])
def test_material_oracle_executor_settles_matched_depth_receipts(depth: int) -> None:
    oracle = _material_oracle(depth)
    posterior = _posterior(oracle)
    packet = _material_packet(oracle, posterior, depth=depth)

    receipts = [
        execute_method(
            packet=packet,
            posterior=posterior,
            oracle=oracle,
            method=method,
            evaluator=None,
        )
        for method in ("determinization", "information_set")
    ]

    assert receipts[0]["consumed"] == receipts[1]["consumed"] > 0
    assert receipts[0]["evaluator_calls"] == receipts[1]["evaluator_calls"] == 0
    assert receipts[0]["transition_program_source"] == "verified-material-oracle"
    assert (
        receipts[0]["transition_program_digest"]
        == receipts[1]["transition_program_digest"]
    )
    settled = settle_packet(packet=packet, receipts=receipts)
    assert settled["depth"] == depth
    assert settled["matched_authorized_compute"] is True
    assert settled["matched_transition_program"] is True


def test_depth_two_material_identity_includes_nested_continuations() -> None:
    oracle = _material_oracle(2)
    posterior = _posterior(oracle)
    packet = _material_packet(oracle, posterior, depth=2)

    det = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=oracle,
        method="determinization",
        evaluator=None,
    )

    drifted = copy.deepcopy(oracle)
    transitions = drifted["transitions"]
    assert isinstance(transitions, list)
    outcomes = transitions[0]["outcomes"]
    assert isinstance(outcomes, list)
    nested = outcomes[0]["continuation_transitions"]
    assert isinstance(nested, dict)
    first_action = sorted(nested)[0]
    branch_rows = nested[first_action]
    assert isinstance(branch_rows, list)
    branch_rows[0]["terminal_utility"] = 999999.0

    info = execute_method(
        packet=packet,
        posterior=posterior,
        oracle=drifted,
        method="information_set",
        evaluator=None,
    )
    assert det["transition_program_digest"] != info["transition_program_digest"]
    with pytest.raises(MatchedComparisonError):
        settle_packet(packet=packet, receipts=[det, info])
