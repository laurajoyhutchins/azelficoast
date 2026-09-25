from __future__ import annotations

import pytest

from azelficoast.matched_comparison import freeze_packet, settle_packet
from azelficoast.matched_population import (
    MatchedPopulationError,
    aggregate_population,
)


def _plan() -> dict[str, object]:
    return {
        "schema": "azelficoast.matched-search-comparison-plan",
        "schema_version": 2,
        "posterior_treatments": ["generator_faithful", "practical"],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": 100,
        },
        "opponent_model": "fixed_observed_response",
        "depths": [1, 2],
        "confirmatory_predictors": [
            "hidden_item_entropy_bits",
            "persistent_branch_count",
        ],
        "cluster_unit": "battle_tag",
        "showdown_commit": "pinned",
        "evaluator": {
            "schema": "azelficoast.belief-policy-value-evaluator",
            "schema_version": 1,
            "checkpoint_digest": "sha256:" + "a" * 64,
            "observability": "public_belief_only",
            "architecture": "weighted_deep_sets_policy_value",
            "spec": {"hidden_width": 256},
        },
        "inference": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1729,
        },
    }


def _packet(fixture: str, battle: str, treatment: str, depth: int) -> dict[str, object]:
    return freeze_packet(
        plan=_plan(),
        state={
            "fixture_id": fixture,
            "battle_tag": battle,
            "public_state": {"turn": depth},
            "legal_actions": ["a", "b"],
            "predictors": {
                "hidden_item_entropy_bits": 1.0,
                "persistent_branch_count": 2,
            },
        },
        posterior={
            "treatment": treatment,
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "worlds": [
                {
                    "world_id": "w",
                    "weight": 1.0,
                    "hidden": {"item": "Choice Band"},
                }
            ],
        },
        posterior_treatment=treatment,
        depth=depth,
    )


def _result(fixture: str, battle: str, treatment: str, depth: int) -> dict[str, object]:
    packet = _packet(fixture, battle, treatment, depth)
    return settle_packet(
        packet=packet,
        receipts=[
            {
                "schema": "azelficoast.matched-search-receipt",
                "schema_version": 4,
                "packet_digest": packet["packet_digest"],
                "matched_spec_digest": packet["matched_spec_digest"],
                "method": "determinization",
                "input_digest": packet["input_digest"],
                "posterior_digest": packet["posterior_digest"],
                "posterior_semantic_digest": packet["posterior_semantic_digest"],
                "evaluator_digest": packet["evaluator_digest"],
                "evaluator_checkpoint_digest": packet["evaluator"]["checkpoint_digest"],
                "mechanics_identity_digest": packet["mechanics_identity_digest"],
                "mechanics_evidence_digest": "e" * 64,
                "showdown_commit": packet["showdown_commit"],
                "chance_treatment": packet["chance_treatment"],
                "transition_oracle_digest": "f" * 64,
                "budget_unit_definition": "one transition_evaluation per verified whole-turn execution class consumed",
                "evaluator_call_unit_definition": "one learned value prediction per successor public information set",
                "evaluator_calls": 3,
                "transition_program_digest": "program-sha256",
                "transition_artifact_digest": "f" * 64,
                "transition_program_source": "verified-transition-program",
                "compute_budget": packet["compute_budget"],
                "consumed": 90,
                "chosen_action": "a",
                "root_values": {"a": 0.7, "b": 0.6},
            },
            {
                "schema": "azelficoast.matched-search-receipt",
                "schema_version": 4,
                "packet_digest": packet["packet_digest"],
                "matched_spec_digest": packet["matched_spec_digest"],
                "method": "information_set",
                "input_digest": packet["input_digest"],
                "posterior_digest": packet["posterior_digest"],
                "posterior_semantic_digest": packet["posterior_semantic_digest"],
                "evaluator_digest": packet["evaluator_digest"],
                "evaluator_checkpoint_digest": packet["evaluator"]["checkpoint_digest"],
                "mechanics_identity_digest": packet["mechanics_identity_digest"],
                "mechanics_evidence_digest": "e" * 64,
                "showdown_commit": packet["showdown_commit"],
                "chance_treatment": packet["chance_treatment"],
                "transition_oracle_digest": "f" * 64,
                "budget_unit_definition": "one transition_evaluation per verified whole-turn execution class consumed",
                "evaluator_call_unit_definition": "one learned value prediction per successor public information set",
                "evaluator_calls": 3,
                "transition_program_digest": "program-sha256",
                "transition_artifact_digest": "f" * 64,
                "transition_program_source": "verified-transition-program",
                "compute_budget": packet["compute_budget"],
                "consumed": 90,
                "chosen_action": "b",
                "root_values": {"a": 0.5, "b": 0.55},
            },
        ],
    )


def test_population_requires_complete_state_posterior_depth_matrix() -> None:
    cohort = {
        "schema": "azelficoast.matched-search-population-cohort",
        "schema_version": 1,
        "selected": [
            {"fixture_id": "f1", "battle_tag": "battle-1"},
            {"fixture_id": "f2", "battle_tag": "battle-2"},
        ],
    }
    rows = [
        _result(fixture, battle, treatment, depth)
        for fixture, battle in (("f1", "battle-1"), ("f2", "battle-2"))
        for treatment in ("generator_faithful", "practical")
        for depth in (1, 2)
    ]

    aggregate = aggregate_population(plan=_plan(), cohort=cohort, results=rows)

    assert aggregate["matrix_complete"] is True
    assert aggregate["matched_input"] is True
    assert aggregate["matched_authorized_compute"] is True
    assert aggregate["matched_evaluator"] is True
    assert aggregate["evaluator"] == _plan()["evaluator"]
    assert len(aggregate["figure_rows"]) == 4
    row = aggregate["figure_rows"][0]
    assert row["state_count"] == 2
    assert row["battle_count"] == 2
    assert abs(row["mean_value_optimism"] - 0.2) < 1e-12
    assert abs(row["mean_regret"] - 0.05) < 1e-12
    assert row["policy_disagreement_rate"] == 1.0


def test_population_refuses_missing_treatment_cell() -> None:
    cohort = {
        "schema": "azelficoast.matched-search-population-cohort",
        "schema_version": 1,
        "selected": [{"fixture_id": "f1", "battle_tag": "battle-1"}],
    }
    rows = [
        _result("f1", "battle-1", treatment, depth)
        for treatment in ("generator_faithful", "practical")
        for depth in (1, 2)
    ]
    rows.pop()

    with pytest.raises(MatchedPopulationError, match="matrix is incomplete"):
        aggregate_population(plan=_plan(), cohort=cohort, results=rows)
