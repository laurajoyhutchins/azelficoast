from __future__ import annotations

from azelficoast.belief.competence import (
    build_competence_ledger,
    competence_claims,
    curriculum_priority,
)
from azelficoast.live.corpus import DecisionFixture


def _fixture(fixture_id: str, *, turn: int, actions: int, switches: bool = False):
    state = {
        "turn": turn,
        "legal_actions": [f"move {index}" for index in range(actions)],
        "opponent_team": {"one": {"species": "A"}},
    }
    if switches:
        state["available_switches"] = ["B"]
    return DecisionFixture(
        fixture_id=fixture_id,
        state=state,
        protocol_prefix=(),
        control_decisions=(),
    )


def test_claims_are_public_and_structural() -> None:
    fixture = _fixture("late", turn=18, actions=6, switches=True)

    claims = competence_claims(
        fixture,
        {
            "search_count": 1,
            "fallback_count": 0,
            "uncertainty": 0.9,
        },
    )

    ids = {claim.claim_id for claim in claims}
    assert "battle-phase:late" in ids
    assert "action-space:broad" in ids
    assert "policy-uncertainty:high" in ids
    assert "routing:search" in ids
    assert "choice:voluntary-switch" in ids


def test_ledger_makes_rare_weighted_claims_more_indebted() -> None:
    hard = _fixture("hard", turn=18, actions=6)
    easy = _fixture("easy", turn=2, actions=2)
    rows = [
        (hard, {}, {"search_count": 1, "fallback_count": 0, "uncertainty": 0.9}),
        (easy, {}, {"search_count": 0, "fallback_count": 0, "uncertainty": 0.1}),
        (easy, {}, {"search_count": 0, "fallback_count": 0, "uncertainty": 0.1}),
    ]

    ledger = build_competence_ledger(rows)
    debt = {row["claim_id"]: row["debt"] for row in ledger["claims"]}

    assert debt["routing:search"] > debt["routing:direct"]
    assert ledger["decision_count"] == 3


def test_priority_decays_after_selecting_same_competence_cell() -> None:
    fixture = _fixture("hard", turn=18, actions=4)
    signals = {"search_count": 1, "fallback_count": 0, "uncertainty": 0.9}
    ledger = build_competence_ledger([(fixture, {}, signals)])

    first = curriculum_priority(
        fixture,
        signals,
        ledger=ledger,
        selected_claim_counts={},
    )
    repeated = curriculum_priority(
        fixture,
        signals,
        ledger=ledger,
        selected_claim_counts={
            claim_id: 1 for claim_id in first["claim_ids"]
        },
    )

    assert first["priority"] > repeated["priority"]
    assert first["marginal_debt"] > repeated["marginal_debt"]


def test_source_provenance_is_a_competence_dimension() -> None:
    fixture = _fixture("public", turn=7, actions=3)
    claims = competence_claims(
        fixture,
        {
            "search_count": 0,
            "fallback_count": 0,
            "uncertainty": 0.2,
            "source_kinds": ["public-showdown-replay"],
        },
    )

    assert "evidence-source:public-showdown-replay" in {
        claim.claim_id for claim in claims
    }
