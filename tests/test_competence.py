from __future__ import annotations

from azelficoast.belief.competence import (
    allocate_evidence_budget,
    build_competence_ledger,
    competence_claims,
    curriculum_priority,
    evidence_source_debt,
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


def test_hard_source_claim_records_where_search_pressure_came_from() -> None:
    fixture = _fixture("hard-source", turn=12, actions=5)
    claims = competence_claims(
        fixture,
        {
            "search_count": 1,
            "fallback_count": 0,
            "uncertainty": 0.9,
            "source_kinds": ["generated-dirty-tricks"],
        },
    )

    ids = {claim.claim_id for claim in claims}
    assert "evidence-source:generated-dirty-tricks" in ids
    assert "evidence-source-hard:generated-dirty-tricks" in ids


def test_source_debt_distinguishes_unseen_from_seen_easy_source() -> None:
    fixture = _fixture("easy-source", turn=2, actions=2)
    ledger = build_competence_ledger(
        [
            (
                fixture,
                {},
                {
                    "search_count": 0,
                    "fallback_count": 0,
                    "uncertainty": 0.1,
                    "source_kinds": ["generated-simple-heuristics"],
                },
            )
        ]
    )

    seen = evidence_source_debt(ledger, "generated-simple-heuristics")
    unseen = evidence_source_debt(ledger, "generated-dirty-tricks")

    assert seen["hard_state_debt"] == 0.0
    assert unseen["hard_state_debt"] == 3.0
    assert unseen["acquisition_debt"] > seen["acquisition_debt"]


def test_evidence_budget_follows_source_debt_and_preserves_total() -> None:
    ledger = {
        "claims": [
            {
                "claim_id": "evidence-source:generated-a",
                "evidence_count": 10,
                "debt": 0.2,
            },
            {
                "claim_id": "evidence-source-hard:generated-a",
                "evidence_count": 1,
                "debt": 2.0,
            },
            {
                "claim_id": "evidence-source:generated-b",
                "evidence_count": 10,
                "debt": 0.2,
            },
        ]
    }

    allocation = allocate_evidence_budget(
        ["generated-a", "generated-b"],
        10,
        ledger=ledger,
    )

    assert sum(allocation) == 10
    assert allocation[0] > allocation[1]


def test_evidence_budget_rotates_equal_remainders() -> None:
    assert allocate_evidence_budget(
        ["a", "b", "c", "d"],
        2,
        ledger=None,
        rotation=0,
    ) == (1, 1, 0, 0)
    assert allocate_evidence_budget(
        ["a", "b", "c", "d"],
        2,
        ledger=None,
        rotation=2,
    ) == (0, 0, 1, 1)
