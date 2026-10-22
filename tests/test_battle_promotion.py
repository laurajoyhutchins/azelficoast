from __future__ import annotations

import hashlib
import json

import pytest

from azelficoast.belief.battle_promotion import (
    BATTLE_PROMOTION_SCHEMA,
    BattlePromotionError,
    BattlePromotionPolicy,
    CANDIDATE_PRIMARY_MODE,
    INCUMBENT_PRIMARY_MODE,
    settle_battle_panel,
    verify_battle_panel_evidence,
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
        "search_policy_margin": 1.0,
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("won", 0),
        ("won", 1),
        ("lost", 0),
        ("lost", 1),
        ("won", 0.0),
        ("lost", 1.0),
    ],
)
def test_panel_rejects_numeric_outcomes_before_promotion(
    field: str, value: int | float,
) -> None:
    rows = _panel(21, 11)
    rows[0][field] = value

    with pytest.raises(BattlePromotionError, match="boolean or null"):
        settle_battle_panel(
            rows,
            candidate_checkpoint_digest=CANDIDATE,
            incumbent_checkpoint_digest=INCUMBENT,
        )


@pytest.mark.parametrize(
    ("won", "lost", "outcome"),
    [
        (True, False, "win"),
        (False, True, "loss"),
        (False, False, "tie"),
        (None, None, "tie"),
    ],
)
def test_native_boolean_and_null_outcomes_are_preserved(
    won: bool | None, lost: bool | None, outcome: str,
) -> None:
    from azelficoast.belief.battle_promotion import _candidate_outcome

    assert _candidate_outcome({
        "mode": CANDIDATE_PRIMARY_MODE, "won": won, "lost": lost,
    }) == outcome


def test_numeric_win_must_not_change_promotion_decision_to_a_tie() -> None:
    admitted = settle_battle_panel(
        _panel(17, 9, 6),
        candidate_checkpoint_digest=CANDIDATE,
        incumbent_checkpoint_digest=INCUMBENT,
    )
    rejected = settle_battle_panel(
        _panel(16, 9, 7),
        candidate_checkpoint_digest=CANDIDATE,
        incumbent_checkpoint_digest=INCUMBENT,
    )

    assert admitted["one_sided_superiority_p_value"] == pytest.approx(0.08431877195835114)
    assert admitted["admitted"] is True
    assert rejected["one_sided_superiority_p_value"] == pytest.approx(0.11476147174835205)
    assert rejected["admitted"] is False

    rows = _panel(17, 9, 6)
    rows[0]["won"] = 0
    # Previously the integer passed set-membership validation but was
    # classified as a tie. That changed the default-policy p-value from
    # 0.0843 (admitted) to 0.1148 (rejected).
    with pytest.raises(BattlePromotionError, match="boolean or null"):
        settle_battle_panel(
            rows,
            candidate_checkpoint_digest=CANDIDATE,
            incumbent_checkpoint_digest=INCUMBENT,
        )


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


def test_panel_fails_closed_on_deployment_routing_drift() -> None:
    rows = _panel(21, 11)
    rows[0]["search_policy_margin"] = 0.0

    with pytest.raises(BattlePromotionError, match="search policy margin drifted"):
        settle_battle_panel(
            rows,
            candidate_checkpoint_digest=CANDIDATE,
            incumbent_checkpoint_digest=INCUMBENT,
            search_policy_margin=1.0,
        )


def test_panel_verification_reopens_prefixed_raw_results_digest(
    tmp_path,
) -> None:
    rows = _panel(21, 11)
    results = tmp_path / "results.jsonl"
    results.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    settled = settle_battle_panel(
        rows,
        candidate_checkpoint_digest=CANDIDATE,
        incumbent_checkpoint_digest=INCUMBENT,
        search_policy_margin=1.0,
    )
    evidence = {
        **settled,
        "results": str(results),
        "results_digest": "sha256:" + hashlib.sha256(results.read_bytes()).hexdigest(),
    }

    verified = verify_battle_panel_evidence(evidence)

    assert verified["admitted"] is True
    assert verified["deployment"]["search_policy_margin"] == 1.0
