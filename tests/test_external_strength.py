from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import pytest

from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT
from azelficoast.research.hosted.external_strength import (
    _battle_identity,
    _send_sequential_challenges,
    _unit_spec,
)
from azelficoast.research.external_strength import (
    ExternalStrengthContractError,
    contract_readiness,
    exact_binomial_superiority_p_value,
    load_contract,
    settle_external_panel,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments" / "data" / "external-playing-strength-contract.json"


def _contract() -> dict[str, object]:
    return load_contract(CONTRACT)


def _rows(*, wins_per_shard: int = 60, losses_per_shard: int = 39) -> list[dict[str, object]]:
    contract = _contract()
    rows: list[dict[str, object]] = []
    outcome_template = (
        ["win"] * wins_per_shard
        + ["loss"] * losses_per_shard
        + ["tie"] * (100 - wins_per_shard - losses_per_shard)
    )
    for shard in range(10):
        for index, outcome in enumerate(outcome_template):
            direction = (
                "foul_play_challenges" if index < 50 else "azelficoast_challenges"
            )
            rows.append(
                {
                    "battle_tag": f"battle-{index:03d}",
                    "format": "gen9randombattle",
                    "shard": shard,
                    "direction": direction,
                    "player_role": "p2" if index < 50 else "p1",
                    "outcome": outcome,
                    "azelficoast_git_sha": "a" * 40,
                    "evaluator_digest": contract["azelficoast"]["evaluator_source"]["checkpoint_digest"],
                    "showdown_revision": PINNED_SHOWDOWN_COMMIT,
                    "opponent": "foul-play",
                    "opponent_revision": contract["opponent"]["revision"],
                }
            )
    return rows


def test_external_strength_contract_freezes_headline_claim_before_outcomes() -> None:
    readiness = contract_readiness(_contract())
    assert readiness["passed"] is True
    assert readiness["planned_battles"] == 1000
    assert readiness["shard_count"] == 10
    assert readiness["evaluator_digest"] == (
        "sha256:9a4b86802f676db276b5f9b988a3d3badda028b5ac6610b2fa7c424b93255a0c"
    )
    assert readiness["opponent_revision"] == "6c467c081e862fb321adb405355beb41aba8e226"
    assert readiness["claim_scope"] == "playing_strength_only_unmatched_compute"


def test_contract_rejects_optional_stopping_or_unpinned_opponent() -> None:
    contract = _contract()
    stopped = copy.deepcopy(contract)
    stopped["no_optional_stopping"] = False
    with pytest.raises(ExternalStrengthContractError, match="optional stopping"):
        validate_contract(stopped)

    floating = copy.deepcopy(contract)
    floating["opponent"]["revision"] = "main"
    with pytest.raises(ExternalStrengthContractError, match="full lowercase Git SHA"):
        validate_contract(floating)


def test_complete_superior_panel_settles_and_binds_raw_digest() -> None:
    result = settle_external_panel(_contract(), _rows())
    assert result["complete_panel"] is True
    assert result["superiority_established"] is True
    assert result["battle_count"] == 1000
    assert result["wins"] == 600
    assert result["losses"] == 390
    assert result["ties"] == 10
    assert result["player_roles"] == {"p1": 500, "p2": 500}
    assert result["directions"] == {
        "azelficoast_challenges": 500,
        "foul_play_challenges": 500,
    }
    assert result["raw_results_digest"].startswith("sha256:")


def test_complete_non_superior_panel_is_preserved_as_negative_result() -> None:
    result = settle_external_panel(
        _contract(),
        _rows(wins_per_shard=49, losses_per_shard=50),
    )
    assert result["complete_panel"] is True
    assert result["superiority_established"] is False
    assert result["battle_count"] == 1000


def test_incomplete_duplicate_or_identity_drifted_panels_fail_closed() -> None:
    rows = _rows()
    with pytest.raises(ExternalStrengthContractError, match="exactly 1000"):
        settle_external_panel(_contract(), rows[:-1])

    duplicate = _rows()
    duplicate[1]["battle_tag"] = duplicate[0]["battle_tag"]
    with pytest.raises(ExternalStrengthContractError, match="duplicate battle"):
        settle_external_panel(_contract(), duplicate)

    drifted = _rows()
    drifted[-1]["opponent_revision"] = "b" * 40
    with pytest.raises(ExternalStrengthContractError, match="opponent identity drifted"):
        settle_external_panel(_contract(), drifted)


def test_exact_binomial_superiority_is_one_sided() -> None:
    assert exact_binomial_superiority_p_value(5, 5) > 0.5
    assert exact_binomial_superiority_p_value(9, 1) < 0.02


def test_split_units_bind_raw_showdown_tags_to_unique_battle_identities() -> None:
    assert _battle_identity(0, "battle-gen9randombattle-1") == (
        "unit-00:battle-gen9randombattle-1"
    )
    assert _battle_identity(1, "battle-gen9randombattle-1") == (
        "unit-01:battle-gen9randombattle-1"
    )


def test_external_execution_units_preserve_frozen_shard_balance() -> None:
    contract = _contract()
    totals: dict[tuple[int, str], int] = {}
    unit_count = int(contract["shard_count"]) * 4
    for unit in range(unit_count):
        shard, direction, battles = _unit_spec(
            unit,
            unit_count=unit_count,
            contract=contract,
        )
        assert battles == 25
        key = (shard, direction)
        totals[key] = totals.get(key, 0) + battles

    for shard in range(int(contract["shard_count"])):
        assert totals[(shard, "foul_play_challenges")] == 50
        assert totals[(shard, "azelficoast_challenges")] == 50


def test_external_opponent_challenges_are_sent_only_after_each_battle_settles() -> None:
    class FakePlayer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        async def send_challenges(self, opponent: str, n_challenges: int) -> None:
            self.calls.append((opponent, n_challenges))

    player = FakePlayer()
    asyncio.run(
        _send_sequential_challenges(
            player,  # type: ignore[arg-type]
            "FoulBench00",
            battles=4,
        )
    )
    assert player.calls == [
        ("FoulBench00", 1),
        ("FoulBench00", 1),
        ("FoulBench00", 1),
        ("FoulBench00", 1),
    ]
