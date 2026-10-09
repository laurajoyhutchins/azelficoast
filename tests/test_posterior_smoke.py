"""Bounded smoke cannot assert full scientific admission or weaken its contract."""

from __future__ import annotations

import pytest

from tools.smoke_posterior_study import SmokeError, smoke_limits


@pytest.mark.parametrize(
    ("fixtures", "rounds", "sample_keys", "screen_rounds"),
    [
        (0, 128, 2, 64),
        (4097, 128, 2, 64),
        (32, 0, 2, 64),
        (32, 2049, 2, 64),
        (32, 128, 0, 64),
        (32, 128, 5, 64),
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
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "tools" / "smoke_posterior_study.py"
    ).read_text(encoding="utf-8")
    assert '"certified": False' in source
    assert '"scientific_population_admitted": False' in source
    assert '"execution_plan_issued": False' in source
    assert "freeze_issue_69_population(" not in source
    assert "compile_execution_plan(" not in source
