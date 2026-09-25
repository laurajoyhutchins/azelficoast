"""Deterministic competence-debt accounting for self-improvement curricula."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from azelficoast.live.corpus import DecisionFixture

COMPETENCE_LEDGER_SCHEMA = "azelficoast.competence-ledger"
COMPETENCE_LEDGER_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CompetenceClaim:
    """One observable capability cell exercised by a decision state."""

    claim_id: str
    family: str
    weight: float

    def __post_init__(self) -> None:
        if not self.claim_id or not self.family:
            raise ValueError("competence claim identity must be non-empty")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("competence claim weight must be positive and finite")


def _bucket_turn(state: Mapping[str, Any]) -> str:
    raw = state.get("turn")
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 0:
        return "unknown"
    if raw <= 4:
        return "opening"
    if raw <= 12:
        return "middle"
    return "late"


def _bucket_actions(count: int) -> str:
    if count <= 1:
        return "forced"
    if count <= 4:
        return "narrow"
    if count <= 8:
        return "broad"
    return "wide"


def _bucket_uncertainty(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.25:
        return "medium"
    return "low"


def _bucket_revealed_opponents(state: Mapping[str, Any]) -> str:
    raw = state.get("opponent_team")
    count = len(raw) if isinstance(raw, Mapping) else 0
    if count <= 1:
        return "one"
    if count <= 3:
        return "partial"
    return "many"


def competence_claims(
    fixture: DecisionFixture,
    signals: Mapping[str, Any],
) -> tuple[CompetenceClaim, ...]:
    """Project public decision evidence into small, stable competence cells."""

    action_count = len(fixture.legal_actions)
    uncertainty = signals.get("uncertainty", 0.0)
    if isinstance(uncertainty, bool) or not isinstance(uncertainty, (int, float)):
        uncertainty = 0.0
    uncertainty = max(0.0, min(1.0, float(uncertainty)))

    search_count = signals.get("search_count", 0)
    fallback_count = signals.get("fallback_count", 0)
    searched = isinstance(search_count, int) and not isinstance(search_count, bool) and search_count > 0
    fallback = isinstance(fallback_count, int) and not isinstance(fallback_count, bool) and fallback_count > 0

    state = fixture.state
    claims = [
        CompetenceClaim(
            f"battle-phase:{_bucket_turn(state)}",
            "battle-phase",
            1.0,
        ),
        CompetenceClaim(
            f"action-space:{_bucket_actions(action_count)}",
            "action-space",
            1.0,
        ),
        CompetenceClaim(
            f"policy-uncertainty:{_bucket_uncertainty(uncertainty)}",
            "policy-uncertainty",
            3.0 if uncertainty >= 0.75 else 1.5 if uncertainty >= 0.25 else 0.75,
        ),
        CompetenceClaim(
            "routing:search" if searched else "routing:direct",
            "routing",
            3.0 if searched else 0.75,
        ),
        CompetenceClaim(
            f"opponent-revelation:{_bucket_revealed_opponents(state)}",
            "opponent-revelation",
            1.0,
        ),
    ]
    if fallback:
        claims.append(CompetenceClaim("routing:fallback", "routing", 4.0))
    if bool(state.get("force_switch")):
        claims.append(CompetenceClaim("choice:forced-switch", "choice", 2.0))
    elif state.get("available_switches"):
        claims.append(CompetenceClaim("choice:voluntary-switch", "choice", 1.5))
    if bool(state.get("can_tera")):
        claims.append(CompetenceClaim("choice:tera-available", "choice", 1.25))

    source_kinds = signals.get("source_kinds", ())
    if (
        isinstance(source_kinds, Sequence)
        and not isinstance(source_kinds, (str, bytes))
    ):
        for source_kind in source_kinds:
            if isinstance(source_kind, str) and source_kind:
                claims.append(
                    CompetenceClaim(
                        f"evidence-source:{source_kind}",
                        "evidence-source",
                        1.5,
                    )
                )

    by_id = {claim.claim_id: claim for claim in claims}
    return tuple(by_id[claim_id] for claim_id in sorted(by_id))


def build_competence_ledger(
    rows: Sequence[tuple[DecisionFixture, Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize observable evidence coverage without consulting hidden truth."""

    counts: Counter[str] = Counter()
    descriptors: dict[str, CompetenceClaim] = {}
    for fixture, _control, signals in rows:
        for claim in competence_claims(fixture, signals):
            counts[claim.claim_id] += 1
            descriptors[claim.claim_id] = claim

    claims = []
    for claim_id in sorted(descriptors):
        descriptor = descriptors[claim_id]
        evidence_count = counts[claim_id]
        claims.append(
            {
                "claim_id": claim_id,
                "family": descriptor.family,
                "weight": descriptor.weight,
                "evidence_count": evidence_count,
                "debt": descriptor.weight / math.sqrt(1.0 + evidence_count),
            }
        )
    return {
        "schema": COMPETENCE_LEDGER_SCHEMA,
        "schema_version": COMPETENCE_LEDGER_SCHEMA_VERSION,
        "decision_count": len(rows),
        "claim_count": len(claims),
        "claims": claims,
    }


def curriculum_priority(
    fixture: DecisionFixture,
    signals: Mapping[str, Any],
    *,
    ledger: Mapping[str, Any],
    selected_claim_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Estimate marginal competence debt retired per public branching cost."""

    debt_by_id = {
        str(row.get("claim_id")): float(row.get("debt", 0.0))
        for row in ledger.get("claims", ())
        if isinstance(row, Mapping)
    }
    claims = competence_claims(fixture, signals)
    contributions = {}
    for claim in claims:
        already_selected = int(selected_claim_counts.get(claim.claim_id, 0))
        contributions[claim.claim_id] = debt_by_id.get(claim.claim_id, 0.0) / (
            1.0 + already_selected
        )

    marginal_value = sum(contributions.values())
    expected_cost = math.sqrt(max(1, len(fixture.legal_actions)))
    priority = marginal_value / expected_cost
    return {
        "claim_ids": [claim.claim_id for claim in claims],
        "marginal_debt": marginal_value,
        "expected_cost": expected_cost,
        "priority": priority,
        "contributions": dict(sorted(contributions.items())),
    }
