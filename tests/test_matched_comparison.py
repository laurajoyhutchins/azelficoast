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
        "schema_version": 1,
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
