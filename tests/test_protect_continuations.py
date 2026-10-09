from __future__ import annotations

from azelficoast.live.corpus import DecisionFixture
from azelficoast.research.studies.protect_continuations import mine_protect_candidates


def _fixture(*, protect: bool) -> DecisionFixture:
    legal_actions = ["/choose move tackle"]
    if protect:
        legal_actions.append("/choose move protect")
    return DecisionFixture(
        fixture_id="protect" if protect else "no-protect",
        state={"legal_actions": legal_actions},
        protocol_prefix=(),
        control_decisions=(),
    )


def test_only_protect_fixtures_reach_hidden_world_miner(monkeypatch) -> None:
    seen = {}

    def fake_mine(fixtures, **kwargs):
        fixtures = list(fixtures)
        seen["ids"] = [fixture.fixture_id for fixture in fixtures]
        seen["kwargs"] = kwargs
        return {
            "schema": "azelficoast.natural-fusion-candidates",
            "schema_version": 1,
            "fixture_count": len(fixtures),
            "persistent_only": True,
            "sampled_world_queries": 0,
            "candidate_count": 0,
            "persistent_candidate_count": 0,
            "candidates": [],
            "skipped": {},
        }

    monkeypatch.setattr(
        "azelficoast.research.studies.protect_continuations.mine_candidates",
        fake_mine,
    )

    result = mine_protect_candidates(
        [_fixture(protect=False), _fixture(protect=True)],
        showdown_root="/tmp/showdown",
        rounds=512,
    )

    assert seen["ids"] == ["protect"]
    assert seen["kwargs"]["persistent_only"] is True
    assert seen["kwargs"]["rounds"] == 512
    assert result["source_fixture_count"] == 2
    assert result["protect_fixture_count"] == 1
    assert result["current_speed_fork_count"] == 0
