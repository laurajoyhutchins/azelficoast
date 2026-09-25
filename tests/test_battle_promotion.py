from __future__ import annotations

import pytest

from azelficoast.belief.battle_promotion import (
    BATTLE_PROMOTION_SCHEMA,
    BattlePromotionError,
    BattlePromotionPolicy,
    CANDIDATE_PRIMARY_MODE,
    INCUMBENT_PRIMARY_MODE,
    settle_battle_panel,
)


CANDIDATE = "sha256:candidate"
INCUMBENT = "sha256:incumbent"


def _row(index: int, mode: str, candidate_outcome: str):
    primary_outcome = candidate_outcome
    if mode == INCUMBENT_PRIMARY_MODE:
        if candidate_outcome == "win":
            primary_outcome = "loss"
        elif candidate_outcome == "loss":
            primary_outcome = "win"
    return {
        "battle_tag": f"battle-{index}",
        "mode": mode,
        "won": True if primary_outcome == "win" else False if primary_outcome == "loss" else None,
        "lost": True if primary_outcome == "loss" else False if primary_outcome == "win" else None,
        "candidate_checkpoint_digest": CANDIDATE,
        "incumbent_checkpoint_digest": INCUMBENT,
    }


def _panel(wins: int, losses: int, ties: int = 0):
    outcomes = ["win"] * wins + ["loss"] * losses + ["tie"] * ties
    half = len(outcomes) // 2
    return [
        _row(
            index,
            CANDIDATE_PRIMARY_MODE if index < half else INCUMBENT_PRIMARY_MODE,
            outcome,
        )
        for index, outcome in enumerate(outcomes)
    ]


def test_strong_candidate_passes_exact_battle_gate() -> None:
    policy = BattlePromotionPolicy(
        expected_battles=32,
        max_superiority_p_value=0.10,
        min_decisive_fraction=0.75,
    )
    settled = settle_battle_panel(
        _panel(21, 11),
        candidate_checkpoint_digest=CANDIDATE,
        incumbent_checkpoint_digest=INCUMBENT,
        policy=policy,
    )

    assert settled["schema"] == BATTLE_PROMOTION_SCHEMA
    assert settled["admitted"] is True
    assert settled["candidate_score_rate"] == 21 / 32
    assert settled["one_sided_superiority_p_value"] <= 0.10


def test_even_record_does_not_count_as_improvement() -> None:
    settled = settle_battle_panel(
        _panel(16, 16),
        candidate_checkpoint_digest=CANDIDATE,
        incumbent_checkpoint_digest=INCUMBENT,
    )

    assert settled["admitted"] is False
    assert "candidate_wins_more_than_loses" in settled["failed_checks"]
    assert "superiority_p_value" in settled["failed_checks"]


def test_panel_fails_closed_on_checkpoint_drift() -> None:
    rows = _panel(21, 11)
    rows[0]["candidate_checkpoint_digest"] = "sha256:other"

    with pytest.raises(BattlePromotionError, match="candidate checkpoint identity"):
        settle_battle_panel(
            rows,
            candidate_checkpoint_digest=CANDIDATE,
            incumbent_checkpoint_digest=INCUMBENT,
        )


def test_panel_requires_exact_side_balance() -> None:
    rows = _panel(21, 11)
    rows[-1]["mode"] = CANDIDATE_PRIMARY_MODE

    settled = settle_battle_panel(
        rows,
        candidate_checkpoint_digest=CANDIDATE,
        incumbent_checkpoint_digest=INCUMBENT,
    )

    assert settled["admitted"] is False
    assert settled["checks"]["side_balance"] is False
