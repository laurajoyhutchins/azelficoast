"""Bounded smoke cannot assert full scientific admission or weaken its contract."""

from __future__ import annotations

from pathlib import Path

import pytest
from runpy import run_path

from azelficoast.live.corpus import DecisionFixture, _fixture_id, write_corpus

_SMOKE = run_path(str(Path(__file__).resolve().parents[1] / "tools" / "smoke_posterior_study.py"))
SmokeError = _SMOKE["SmokeError"]
_load_fixture_window = _SMOKE["_load_fixture_window"]
smoke_limits = _SMOKE["smoke_limits"]


@pytest.mark.parametrize(
    ("fixtures", "rounds", "sample_keys", "screen_rounds"),
    [
        (0, 128, 2, 64),
        (4097, 128, 2, 64),
        (32, 0, 2, 64),
        (32, 2049, 2, 64),
        (32, 128, 0, 64),
        (32, 128, 33, 64),
        (32, 128, 2, 513),
        (True, 128, 2, 64),
    ],
)
def test_smoke_bounds_fail_closed(
    fixtures: int, rounds: int, sample_keys: int, screen_rounds: int
) -> None:
    with pytest.raises(SmokeError):
        smoke_limits(fixtures, rounds, sample_keys, screen_rounds)


def test_small_smoke_is_permitted_but_not_a_scientific_cohort() -> None:
    smoke_limits(128, 64, 1, 64)
    source = (
        Path(__file__).resolve().parents[1]
        / "tools" / "smoke_posterior_study.py"
    ).read_text(encoding="utf-8")
    assert '"certified": False' in source
    assert '"scientific_population_admitted": False' in source
    assert '"execution_plan_issued": False' in source
    assert "freeze_issue_69_population(" not in source
    assert "compile_execution_plan(" not in source



def test_window_decodes_only_bounded_authenticated_fixture_records(
    tmp_path: Path,
) -> None:
    fixtures: list[DecisionFixture] = []
    for n in range(3):
        state = {"turn": n, "legal_actions": ["/choose move tackle"]}
        fixture_id = _fixture_id(state, ())
        fixtures.append(
            DecisionFixture(
                fixture_id=fixture_id,
                state=state,
                protocol_prefix=(),
                control_decisions=(),
            )
        )
    corpus = tmp_path / "corpus.jsonl"
    write_corpus(fixtures, corpus)
    selected, total = _load_fixture_window(corpus, limit=1, expected_total=3)
    assert total == 3
    assert len(selected) == 1
    assert selected[0].fixture_id in {fixture.fixture_id for fixture in fixtures}
    with pytest.raises(SmokeError, match="fixture count"):
        _load_fixture_window(corpus, limit=1, expected_total=2)


def test_smoke_key_offset_is_explicit_and_bounded() -> None:
    smoke_limits(1024, 2048, 16, 512, 64)
    with pytest.raises(SmokeError, match="key_offset"):
        smoke_limits(1024, 2048, 16, 512, -1)
    with pytest.raises(SmokeError, match="key_offset"):
        smoke_limits(1024, 2048, 16, 512, 4097)



def test_mechanics_smoke_summary_rejects_drift_and_missing_cases() -> None:
    check = _SMOKE["_verified_mechanics_summary"]
    valid = {
        "schema": "azelficoast.public-belief-speed-fork-mechanics",
        "schema_version": 1,
        "showdown_commit": "pinned",
        "rounds": 512,
        "case_count": 1,
        "strict_execution_fork_count": 0,
        "cases": [{"fixture_id": "example"}],
    }
    assert check(valid, candidate_count=1, rounds=512, showdown_commit="pinned") == {
        "case_count": 1,
        "strict_execution_fork_count": 0,
    }
    for change in (
        {"showdown_commit": "stale"},
        {"rounds": 1},
        {"case_count": 2},
        {"strict_execution_fork_count": 2},
        {"cases": []},
    ):
        with pytest.raises(SmokeError):
            check(
                {**valid, **change},
                candidate_count=1,
                rounds=512,
                showdown_commit="pinned",
            )
