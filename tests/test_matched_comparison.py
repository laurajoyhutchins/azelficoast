from __future__ import annotations

import pytest

from azelficoast.matched_comparison import (
    MatchedComparisonError,
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
            {"world_id": "band", "weight": 0.6},
            {"world_id": "scarf", "weight": 0.4},
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
    return {
        "method": method,
        "input_digest": packet["input_digest"],
        "evaluator_digest": packet["evaluator_digest"],
        "evaluator_checkpoint_digest": packet["evaluator"]["checkpoint_digest"],
        "evaluator_calls": 7,
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
    assert len(
        {
            (
                row["compute_budget"]["unit"],
                row["compute_budget"]["authorized"],
            )
            for row in packet["work"]
        }
    ) == 1
    assert packet["opponent_model"] == "fixed_observed_response"


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
                consumed=2800,
            ),
        ],
    )

    assert result["matched_input"] is True
    assert result["matched_authorized_compute"] is True
    assert result["matched_evaluator"] is True
    assert result["matched_evaluator_checkpoint"] is True
    assert result["evaluator"] == _plan()["evaluator"]
    assert result["evaluator_checkpoint_digest"] == _plan()["evaluator"]["checkpoint_digest"]
    assert result["evaluator_calls"] == {
        "determinization": 7,
        "information_set": 7,
    }
    assert abs(result["max_determinization_value_optimism"] - 0.20) < 1e-12
    assert (
        abs(result["information_set_regret_of_determinization_action"] - 0.05)
        < 1e-12
    )
    assert result["policy_disagreement"] is True
    assert result["compute_consumed"] == {
        "determinization": 3000,
        "information_set": 2800,
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

    bad_input = dict(info)
    bad_input["input_digest"] = "other"
    with pytest.raises(MatchedComparisonError, match="another frozen input"):
        settle_packet(packet=packet, receipts=[det, bad_input])

    over_budget = dict(info)
    over_budget["consumed"] = 4097
    with pytest.raises(MatchedComparisonError, match="exceeded"):
        settle_packet(packet=packet, receipts=[det, over_budget])


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
