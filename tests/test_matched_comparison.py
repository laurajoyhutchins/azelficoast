from __future__ import annotations

import pytest

from azelficoast.research.matched_comparison import (
    COMPUTE_BUDGET_UNIT_DEFINITION,
    EVALUATOR_CALL_UNIT_DEFINITION,
    MatchedComparisonError,
    _sha256,
    freeze_packet,
    settle_packet,
)


def _plan() -> dict[str, object]:
    return {
        "schema": "azelficoast.matched-search-comparison-plan",
        "schema_version": 2,
        "posterior_treatments": [
            "oracle",
            "generator_faithful",
            "practical",
        ],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": 4096,
        },
        "opponent_model": "fixed_observed_response",
        "depths": [1, 2],
        "confirmatory_predictors": [
            "hidden_item_entropy_bits",
            "incoming_ko_roll_probability_gap",
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


def _state() -> dict[str, object]:
    return {
        "fixture_id": "fixture-1",
        "battle_tag": "battle-1",
        "public_state": {"turn": 8, "active": "Tinkaton"},
        "legal_actions": ["protect", "attack"],
        "predictors": {
            "hidden_item_entropy_bits": 1.0,
            "incoming_ko_roll_probability_gap": 0.25,
            "persistent_branch_count": 2,
        },
    }


def _posterior(*, revealed: bool = False) -> dict[str, object]:
    return {
        "treatment": "generator_faithful",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": revealed,
        "worlds": [
            {"world_id": "band", "weight": 0.6, "hidden": {"item": "band"}},
            {"world_id": "scarf", "weight": 0.4, "hidden": {"item": "scarf"}},
        ],
    }


def _receipt(
    packet: dict[str, object],
    method: str,
    *,
    chosen_action: str,
    root_values: dict[str, float],
    consumed: int = 3000,
) -> dict[str, object]:
    spec = packet["matched_spec"]
    assert isinstance(spec, dict)
    return {
        "schema": "azelficoast.matched-search-receipt",
        "schema_version": 4,
        "packet_digest": packet["packet_digest"],
        "matched_spec_digest": packet["matched_spec_digest"],
        "method": method,
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
        "transition_artifact_digest": "f" * 64,
        "transition_program_source": "verified-transition-program",
        "budget_unit_definition": COMPUTE_BUDGET_UNIT_DEFINITION,
        "evaluator_call_unit_definition": EVALUATOR_CALL_UNIT_DEFINITION,
        "evaluator_calls": 7,
        "transition_program_digest": "program-sha256",
        "compute_budget": packet["compute_budget"],
        "consumed": consumed,
        "chosen_action": chosen_action,
        "root_values": root_values,
    }


def test_freeze_packet_gives_both_methods_identical_input_and_budget() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=2,
    )

    assert [row["method"] for row in packet["work"]] == [
        "determinization",
        "information_set",
    ]
    assert len({row["input_digest"] for row in packet["work"]}) == 1
    assert len({row["evaluator_digest"] for row in packet["work"]}) == 1
    assert packet["evaluator"]["observability"] == "public_belief_only"
    assert (
        len(
            {
                (
                    row["compute_budget"]["unit"],
                    row["compute_budget"]["authorized"],
                )
                for row in packet["work"]
            }
        )
        == 1
    )
    assert packet["opponent_model"] == "fixed_observed_response"
    assert packet["posterior_support"]["support_size"] == 2
    assert abs(packet["posterior_support"]["effective_sample_size"] - 1.923076923076923) < 1e-12


def test_freeze_packet_accepts_bounded_move_distribution_opponent_model() -> None:
    plan = _plan()
    plan["opponent_model"] = "repeat-last-or-uniform-legal-moves"

    packet = freeze_packet(
        plan=plan,
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )

    assert packet["opponent_model"] == "repeat-last-or-uniform-legal-moves"
    assert packet["matched_spec"]["opponent_model"] == (
        "repeat-last-or-uniform-legal-moves"
    )


def test_freeze_packet_carries_one_explicit_matched_experiment_spec() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=2,
    )

    spec = packet["matched_spec"]
    assert spec["fixture_id"] == "fixture-1"
    assert spec["battle_tag"] == "battle-1"
    assert set(spec["legal_actions"]) == {"protect", "attack"}
    assert spec["mechanics_identity"]["revision"] == "pinned"
    assert spec["posterior_semantic_digest"] == packet["posterior_semantic_digest"]
    assert spec["evaluator_identity_digest"] == packet["evaluator_digest"]
    assert spec["compute_budget"] == packet["compute_budget"]
    assert spec["chance_treatment"] == "shared_frozen_transition_oracle"


def test_packet_state_digest_cannot_drift_from_frozen_public_spec() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    packet["state_digest"] = "different"
    unsigned = dict(packet)
    unsigned.pop("packet_digest")
    packet["packet_digest"] = _sha256(unsigned)

    with pytest.raises(MatchedComparisonError, match="public state differs"):
        settle_packet(packet=packet, receipts=[])


def test_freeze_packet_rejects_realized_hidden_state_as_oracle() -> None:
    with pytest.raises(
        MatchedComparisonError,
        match="may not reveal the realized hidden state",
    ):
        freeze_packet(
            plan=_plan(),
            state=_state(),
            posterior=_posterior(revealed=True),
            posterior_treatment="generator_faithful",
            depth=1,
        )


def test_settle_packet_measures_bias_and_regret_under_matched_budget() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    result = settle_packet(
        packet=packet,
        receipts=[
            _receipt(
                packet,
                "determinization",
                chosen_action="protect",
                root_values={"protect": 0.70, "attack": 0.60},
            ),
            _receipt(
                packet,
                "information_set",
                chosen_action="attack",
                root_values={"protect": 0.50, "attack": 0.55},
                consumed=3000,
            ),
        ],
    )

    assert result["matched_input"] is True
    assert result["packet_digest"] == packet["packet_digest"]
    assert result["input_digest"] == packet["input_digest"]
    assert result["state_digest"] == packet["state_digest"]
    assert result["posterior_digest"] == packet["posterior_digest"]
    assert result["posterior_support"] == packet["posterior_support"]
    assert result["showdown_commit"] == packet["showdown_commit"]
    assert result["legal_actions"] == packet["legal_actions"]
    assert result["matched_authorized_compute"] is True
    assert result["matched_evaluator"] is True
    assert result["matched_evaluator_checkpoint"] is True
    assert result["matched_transition_program"] is True
    assert result["transition_program_digest"] == "program-sha256"
    assert result["evaluator"] == _plan()["evaluator"]
    assert result["evaluator_checkpoint_digest"] == _plan()["evaluator"]["checkpoint_digest"]
    assert result["evaluator_calls"] == {
        "determinization": 7,
        "information_set": 7,
    }
    assert result["evaluator_batches"] == {
        "determinization": 7,
        "information_set": 7,
    }
    assert abs(result["max_determinization_value_optimism"] - 0.20) < 1e-12
    assert abs(result["information_set_regret_of_determinization_action"] - 0.05) < 1e-12
    assert result["policy_disagreement"] is True
    assert result["compute_consumed"] == {
        "determinization": 3000,
        "information_set": 3000,
    }


def test_settle_packet_rejects_input_or_budget_drift() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    det = _receipt(
        packet,
        "determinization",
        chosen_action="protect",
        root_values={"protect": 0.7, "attack": 0.6},
    )
    info = _receipt(
        packet,
        "information_set",
        chosen_action="attack",
        root_values={"protect": 0.5, "attack": 0.55},
    )

    bad_packet = dict(info)
    bad_packet["packet_digest"] = "other"
    with pytest.raises(MatchedComparisonError, match="another frozen packet"):
        settle_packet(packet=packet, receipts=[det, bad_packet])

    bad_input = dict(info)
    bad_input["input_digest"] = "other"
    with pytest.raises(MatchedComparisonError, match="another frozen input"):
        settle_packet(packet=packet, receipts=[det, bad_input])

    over_budget = dict(info)
    over_budget["consumed"] = 4097
    with pytest.raises(MatchedComparisonError, match="exceeded"):
        settle_packet(packet=packet, receipts=[det, over_budget])


def test_settle_packet_rejects_different_counted_consumption() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    det = _receipt(
        packet,
        "determinization",
        chosen_action="protect",
        root_values={"protect": 0.7, "attack": 0.6},
        consumed=3000,
    )
    info = _receipt(
        packet,
        "information_set",
        chosen_action="attack",
        root_values={"protect": 0.5, "attack": 0.55},
        consumed=3001,
    )

    with pytest.raises(MatchedComparisonError, match="different counted compute"):
        settle_packet(packet=packet, receipts=[det, info])


def test_settle_packet_rejects_resource_receipt_disagreeing_with_measured_counts() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    det = _receipt(
        packet,
        "determinization",
        chosen_action="protect",
        root_values={"protect": 0.7, "attack": 0.6},
    )
    info = _receipt(
        packet,
        "information_set",
        chosen_action="attack",
        root_values={"protect": 0.5, "attack": 0.55},
    )
    resource = {
        "verified_execution_classes_consumed": 3000,
        "evaluator_calls": 7,
        "executor_preparation_wall_ms": 1.0,
        "search_wall_ms": 2.0,
        "executor_wall_ms": 3.0,
        "transition_program_generation_included": False,
        "transition_program_verification_included": False,
        "posterior_construction_included": False,
        "scope_note": "executor-local timings",
    }
    det["resource_accounting"] = dict(resource)
    info["resource_accounting"] = {**resource, "verified_execution_classes_consumed": 3001}

    with pytest.raises(MatchedComparisonError, match="resource accounting disagrees"):
        settle_packet(packet=packet, receipts=[det, info])


def test_settle_packet_rejects_evaluator_drift() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    det = _receipt(
        packet,
        "determinization",
        chosen_action="protect",
        root_values={"protect": 0.7, "attack": 0.6},
    )
    info = _receipt(
        packet,
        "information_set",
        chosen_action="attack",
        root_values={"protect": 0.5, "attack": 0.55},
    )
    info["evaluator_digest"] = "different"

    with pytest.raises(MatchedComparisonError, match="another evaluator"):
        settle_packet(packet=packet, receipts=[det, info])


def test_settle_packet_rejects_posterior_mechanics_and_chance_drift() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    det = _receipt(
        packet,
        "determinization",
        chosen_action="protect",
        root_values={"protect": 0.7, "attack": 0.6},
    )
    info = _receipt(
        packet,
        "information_set",
        chosen_action="attack",
        root_values={"protect": 0.5, "attack": 0.55},
    )
    det["transition_oracle_digest"] = "shared-oracle"
    info["transition_oracle_digest"] = "different-oracle"
    det["transition_artifact_digest"] = "shared-oracle"
    info["transition_artifact_digest"] = "different-oracle"

    with pytest.raises(MatchedComparisonError, match="different frozen mechanics evidence"):
        settle_packet(packet=packet, receipts=[det, info])

    info["transition_oracle_digest"] = "shared-oracle"
    info["transition_artifact_digest"] = "shared-oracle"
    info["posterior_semantic_digest"] = "different-posterior"
    with pytest.raises(MatchedComparisonError, match="another posterior"):
        settle_packet(packet=packet, receipts=[det, info])

    info["posterior_semantic_digest"] = packet["posterior_semantic_digest"]
    info["showdown_commit"] = "other-revision"
    with pytest.raises(MatchedComparisonError, match="another mechanics revision"):
        settle_packet(packet=packet, receipts=[det, info])


def test_settle_packet_rejects_root_action_set_drift() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    det = _receipt(
        packet,
        "determinization",
        chosen_action="protect",
        root_values={"protect": 0.7, "attack": 0.6},
    )
    info = _receipt(
        packet,
        "information_set",
        chosen_action="attack",
        root_values={"protect": 0.5, "attack": 0.55, "switch": 0.2},
    )

    with pytest.raises(MatchedComparisonError, match="root values do not cover"):
        settle_packet(packet=packet, receipts=[det, info])


def test_settle_packet_rejects_checkpoint_or_evaluator_call_drift() -> None:
    packet = freeze_packet(
        plan=_plan(),
        state=_state(),
        posterior=_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )
    det = _receipt(
        packet,
        "determinization",
        chosen_action="protect",
        root_values={"protect": 0.7, "attack": 0.6},
    )
    info = _receipt(
        packet,
        "information_set",
        chosen_action="attack",
        root_values={"protect": 0.5, "attack": 0.55},
    )

    bad_checkpoint = dict(info)
    bad_checkpoint["evaluator_checkpoint_digest"] = "sha256:" + "b" * 64
    with pytest.raises(MatchedComparisonError, match="another evaluator checkpoint"):
        settle_packet(packet=packet, receipts=[det, bad_checkpoint])

    bad_calls = dict(info)
    bad_calls["evaluator_calls"] = -1
    with pytest.raises(MatchedComparisonError, match="evaluator call count"):
        settle_packet(packet=packet, receipts=[det, bad_calls])

    bad_program = dict(info)
    bad_program["transition_program_digest"] = "other-program"
    with pytest.raises(MatchedComparisonError, match="different transition programs"):
        settle_packet(packet=packet, receipts=[det, bad_program])

    bad_unit = dict(info)
    bad_unit["budget_unit_definition"] = "caller-declared computation"
    with pytest.raises(MatchedComparisonError, match="unsupported compute-budget unit"):
        settle_packet(packet=packet, receipts=[det, bad_unit])
