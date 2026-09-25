"""Deterministic battle-outcome evidence for evaluator promotion."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

BATTLE_PROMOTION_SCHEMA = "azelficoast.battle-promotion-panel"
BATTLE_PROMOTION_SCHEMA_VERSION = 1
CANDIDATE_PRIMARY_MODE = "promotion:candidate-primary"
INCUMBENT_PRIMARY_MODE = "promotion:incumbent-primary"


class BattlePromotionError(ValueError):
    """Raised when promotion battle evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class BattlePromotionPolicy:
    expected_battles: int = 32
    max_superiority_p_value: float = 0.10
    min_decisive_fraction: float = 0.75

    def __post_init__(self) -> None:
        if (
            not isinstance(self.expected_battles, int)
            or isinstance(self.expected_battles, bool)
            or self.expected_battles < 2
            or self.expected_battles % 2 != 0
        ):
            raise BattlePromotionError("expected_battles must be an even integer of at least 2")
        if (
            not math.isfinite(self.max_superiority_p_value)
            or not 0.0 < self.max_superiority_p_value < 1.0
        ):
            raise BattlePromotionError("max_superiority_p_value must be within (0, 1)")
        if (
            not math.isfinite(self.min_decisive_fraction)
            or not 0.0 < self.min_decisive_fraction <= 1.0
        ):
            raise BattlePromotionError("min_decisive_fraction must be within (0, 1]")

    def as_record(self) -> dict[str, float | int]:
        return {
            "expected_battles": self.expected_battles,
            "max_superiority_p_value": self.max_superiority_p_value,
            "min_decisive_fraction": self.min_decisive_fraction,
        }


def _candidate_outcome(row: Mapping[str, Any]) -> str:
    won = row.get("won")
    lost = row.get("lost")
    if won not in {True, False, None} or lost not in {True, False, None}:
        raise BattlePromotionError("battle result won/lost fields must be boolean or null")
    if won is True and lost is True:
        raise BattlePromotionError("battle result cannot be both won and lost")

    mode = row.get("mode")
    if mode not in {CANDIDATE_PRIMARY_MODE, INCUMBENT_PRIMARY_MODE}:
        raise BattlePromotionError(f"unexpected promotion battle mode {mode!r}")

    primary = "win" if won is True else "loss" if lost is True else "tie"
    if mode == CANDIDATE_PRIMARY_MODE:
        return primary
    if primary == "win":
        return "loss"
    if primary == "loss":
        return "win"
    return "tie"


def _one_sided_superiority_p_value(wins: int, losses: int) -> float:
    """Exact P[X >= wins] for X~Binomial(wins+losses, 0.5)."""

    decisive = wins + losses
    if decisive <= 0:
        return 1.0
    numerator = sum(math.comb(decisive, value) for value in range(wins, decisive + 1))
    return numerator / (2**decisive)


def settle_battle_panel(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_checkpoint_digest: str,
    incumbent_checkpoint_digest: str,
    policy: BattlePromotionPolicy = BattlePromotionPolicy(),
) -> dict[str, Any]:
    """Settle a side-balanced candidate-vs-incumbent superiority experiment."""

    if not candidate_checkpoint_digest or not incumbent_checkpoint_digest:
        raise BattlePromotionError("checkpoint digests must be non-empty")
    if candidate_checkpoint_digest == incumbent_checkpoint_digest:
        raise BattlePromotionError("candidate and incumbent checkpoint digests must differ")
    if len(records) != policy.expected_battles:
        raise BattlePromotionError(
            f"expected {policy.expected_battles} battle results, found {len(records)}"
        )

    seen_tags: set[str] = set()
    role_counts = {CANDIDATE_PRIMARY_MODE: 0, INCUMBENT_PRIMARY_MODE: 0}
    outcomes = {"win": 0, "loss": 0, "tie": 0}
    for row in records:
        battle_tag = row.get("battle_tag")
        if not isinstance(battle_tag, str) or not battle_tag:
            raise BattlePromotionError("promotion battle lacks a battle_tag")
        if battle_tag in seen_tags:
            raise BattlePromotionError(f"duplicate promotion battle {battle_tag!r}")
        seen_tags.add(battle_tag)

        if row.get("candidate_checkpoint_digest") != candidate_checkpoint_digest:
            raise BattlePromotionError("candidate checkpoint identity drifted in battle panel")
        if row.get("incumbent_checkpoint_digest") != incumbent_checkpoint_digest:
            raise BattlePromotionError("incumbent checkpoint identity drifted in battle panel")
        mode = row.get("mode")
        if mode not in role_counts:
            raise BattlePromotionError(f"unexpected promotion battle mode {mode!r}")
        role_counts[str(mode)] += 1
        outcomes[_candidate_outcome(row)] += 1

    expected_per_role = policy.expected_battles // 2
    side_balance = all(count == expected_per_role for count in role_counts.values())
    decisive = outcomes["win"] + outcomes["loss"]
    minimum_decisive = math.ceil(policy.expected_battles * policy.min_decisive_fraction)
    p_value = _one_sided_superiority_p_value(outcomes["win"], outcomes["loss"])
    score_rate = (
        outcomes["win"] + 0.5 * outcomes["tie"]
    ) / policy.expected_battles

    checks = {
        "side_balance": side_balance,
        "minimum_decisive_battles": decisive >= minimum_decisive,
        "candidate_wins_more_than_loses": outcomes["win"] > outcomes["loss"],
        "superiority_p_value": p_value <= policy.max_superiority_p_value,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema": BATTLE_PROMOTION_SCHEMA,
        "schema_version": BATTLE_PROMOTION_SCHEMA_VERSION,
        "candidate_checkpoint_digest": candidate_checkpoint_digest,
        "incumbent_checkpoint_digest": incumbent_checkpoint_digest,
        "policy": policy.as_record(),
        "battle_count": len(records),
        "role_counts": dict(sorted(role_counts.items())),
        "candidate_outcomes": outcomes,
        "candidate_score_rate": score_rate,
        "decisive_battle_count": decisive,
        "one_sided_superiority_p_value": p_value,
        "checks": checks,
        "failed_checks": failed,
        "admitted": not failed,
        "pairing": "side-balanced-unpaired-random-battle",
    }
