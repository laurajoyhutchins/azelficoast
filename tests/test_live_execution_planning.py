from __future__ import annotations

from azelficoast.live.execution_planning import (
    CACHE_MODE_EXACT,
    CACHE_MODE_FRESH,
    CACHE_MODE_PROJECTION,
    TransitionRouteHistory,
)


def _producer(
    *,
    exact_hits: int,
    exact_misses: int,
    exact_hit_ms: float,
    exact_miss_ms: float,
    projection_hits: int,
    projection_misses: int,
    projection_hit_ms: float,
    projection_miss_ms: float,
    fresh_count: int,
    fresh_ms: float,
) -> dict[str, object]:
    return {
        "physical_cost_observations": {
            "exact_cache_hit_count": exact_hits,
            "exact_cache_hit_total_ms": exact_hit_ms,
            "exact_cache_miss_count": exact_misses,
            "exact_cache_miss_total_ms": exact_miss_ms,
            "projection_cache_hit_count": projection_hits,
            "projection_cache_hit_total_ms": projection_hit_ms,
            "projection_cache_miss_count": projection_misses,
            "projection_cache_miss_total_ms": projection_miss_ms,
            "fresh_execution_count": fresh_count,
            "fresh_execution_total_ms": fresh_ms,
        }
    }


def test_transition_route_cold_start_measures_full_cache_route() -> None:
    history = TransitionRouteHistory()

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_PROJECTION
    assert explanation["selection_basis"] == "cold-start-measurement"


def test_transition_route_prefers_projection_when_measured_reuse_is_cheap() -> None:
    history = TransitionRouteHistory()
    assert history.observe(
        _producer(
            exact_hits=80,
            exact_misses=20,
            exact_hit_ms=8.0,
            exact_miss_ms=2.0,
            projection_hits=15,
            projection_misses=5,
            projection_hit_ms=3.0,
            projection_miss_ms=0.5,
            fresh_count=5,
            fresh_ms=10.0,
        )
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_PROJECTION
    assert explanation["selection_basis"] == "measured-locality-and-latency"
    assert explanation["selected_implementation"] == "showdown:projected-delta-cache"


def test_transition_route_can_skip_projection_when_its_measured_lookup_is_worse() -> None:
    history = TransitionRouteHistory()
    assert history.observe(
        _producer(
            exact_hits=80,
            exact_misses=20,
            exact_hit_ms=8.0,
            exact_miss_ms=2.0,
            projection_hits=0,
            projection_misses=20,
            projection_hit_ms=0.0,
            projection_miss_ms=8.0,
            fresh_count=20,
            fresh_ms=40.0,
        )
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_EXACT
    assert explanation["selected_implementation"] == "showdown:exact-cache"


def test_transition_route_can_choose_fresh_when_cache_locality_is_bad() -> None:
    history = TransitionRouteHistory()
    assert history.observe(
        _producer(
            exact_hits=0,
            exact_misses=10,
            exact_hit_ms=0.0,
            exact_miss_ms=5.0,
            projection_hits=0,
            projection_misses=10,
            projection_hit_ms=0.0,
            projection_miss_ms=5.0,
            fresh_count=10,
            fresh_ms=10.0,
        )
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_FRESH
    assert explanation["selected_implementation"] == "showdown:fresh"


def test_transition_route_remeasures_locality_periodically() -> None:
    history = TransitionRouteHistory(
        exact_hits=0,
        exact_misses=160,
        exact_hit_total_ms=0.0,
        exact_miss_total_ms=80.0,
        projection_hits=0,
        projection_misses=160,
        projection_hit_total_ms=0.0,
        projection_miss_total_ms=80.0,
        fresh_executions=160,
        fresh_execution_total_ms=160.0,
        observations=16,
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_PROJECTION
    assert explanation["selection_basis"] == "periodic-locality-refresh"


def test_transition_route_rejects_partial_timing_observations() -> None:
    history = TransitionRouteHistory()

    assert not history.observe(
        {"physical_cost_observations": {"fresh_execution_count": 3}}
    )
    assert history.observations == 0
