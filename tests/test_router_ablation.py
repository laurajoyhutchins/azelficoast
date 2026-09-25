from __future__ import annotations

from azelficoast.router_ablation import (
    RouterCase,
    summarize_observed_routes,
    threshold_sweep,
)


def _cases() -> list[RouterCase]:
    return [
        RouterCase(
            case_id="confident",
            policy_margin=0.8,
            direct_regret=0.02,
            search_regret=0.01,
            fallback_regret=0.20,
            actual_route="direct",
            direct_outcome=1.0,
            search_outcome=1.0,
            fallback_outcome=0.0,
        ),
        RouterCase(
            case_id="uncertain",
            policy_margin=0.1,
            direct_regret=0.30,
            search_regret=0.05,
            fallback_regret=0.40,
            actual_route="search",
            direct_outcome=0.0,
            search_outcome=1.0,
            fallback_outcome=0.0,
        ),
        RouterCase(
            case_id="unavailable",
            policy_margin=0.05,
            direct_regret=0.25,
            search_regret=None,
            fallback_regret=0.15,
            actual_route="fallback",
            direct_outcome=0.0,
            fallback_outcome=1.0,
        ),
    ]


def test_observed_summary_keeps_route_conditioned_regret_separate() -> None:
    summary = summarize_observed_routes(_cases())

    assert summary["routes"]["direct"]["case_count"] == 1
    assert summary["routes"]["search"]["mean_regret"] == 0.05
    assert summary["routes"]["fallback"]["mean_regret"] == 0.15
    assert summary["always_search_coverage_rate"] == 2 / 3


def test_threshold_sweep_exposes_search_and_fallback_mixture() -> None:
    rows = threshold_sweep(_cases(), thresholds=[0.0, 0.2, 1.0])

    assert rows[0]["search_count"] == 0
    assert rows[0]["fallback_count"] == 0
    assert rows[1]["search_count"] == 1
    assert rows[1]["fallback_count"] == 1
    assert rows[2]["search_count"] == 2
    assert rows[2]["fallback_count"] == 1
    assert rows[1]["mean_regret"] < rows[0]["mean_regret"]
