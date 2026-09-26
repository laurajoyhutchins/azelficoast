from __future__ import annotations

from azelficoast.live.execution_planning import (
    CACHE_MODE_FRESH,
    CACHE_MODE_PROJECTION,
    CACHE_MODE_PROJECTION_FIRST,
    TransitionRouteHistory,
)


def _producer(
    *,
    cache_mode: str,
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
    route_count: int,
    route_ms: float,
) -> dict[str, object]:
    return {
        "transition_cache_mode": cache_mode,
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
            "route_execution_count": route_count,
            "route_total_ms": route_ms,
        },
    }


def test_transition_route_cold_start_measures_both_cache_orders() -> None:
    history = TransitionRouteHistory()

    first_mode, first = history.plan(semantic_signature="sha256:turn")
    assert first_mode == CACHE_MODE_PROJECTION
    assert first["cache_probe_order"] == ["exact-cache", "projected-delta"]
    assert first["selection_basis"] == "cold-start-order-measurement"

    assert history.observe(
        _producer(
            cache_mode=CACHE_MODE_PROJECTION,
            exact_hits=8,
            exact_misses=2,
            exact_hit_ms=0.8,
            exact_miss_ms=0.2,
            projection_hits=1,
            projection_misses=1,
            projection_hit_ms=0.1,
            projection_miss_ms=0.1,
            fresh_count=1,
            fresh_ms=2.0,
            route_count=10,
            route_ms=3.2,
        )
    )

    second_mode, second = history.plan(semantic_signature="sha256:turn")
    assert second_mode == CACHE_MODE_PROJECTION_FIRST
    assert second["cache_probe_order"] == ["projected-delta", "exact-cache"]
    assert second["selection_basis"] == "cold-start-order-measurement"


def test_transition_route_prefers_projection_first_when_measured_route_is_cheaper() -> None:
    history = TransitionRouteHistory()
    assert history.observe(
        _producer(
            cache_mode=CACHE_MODE_PROJECTION,
            exact_hits=2,
            exact_misses=8,
            exact_hit_ms=0.4,
            exact_miss_ms=0.8,
            projection_hits=6,
            projection_misses=2,
            projection_hit_ms=1.2,
            projection_miss_ms=0.2,
            fresh_count=2,
            fresh_ms=4.0,
            route_count=10,
            route_ms=6.6,
        )
    )
    assert history.observe(
        _producer(
            cache_mode=CACHE_MODE_PROJECTION_FIRST,
            exact_hits=1,
            exact_misses=1,
            exact_hit_ms=0.2,
            exact_miss_ms=0.1,
            projection_hits=8,
            projection_misses=2,
            projection_hit_ms=0.8,
            projection_miss_ms=0.2,
            fresh_count=1,
            fresh_ms=2.0,
            route_count=10,
            route_ms=3.3,
        )
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_PROJECTION_FIRST
    assert explanation["selection_basis"] == "measured-route-order-cost"
    assert explanation["selected_implementation"] == (
        "showdown:projection-then-exact-cache"
    )
    assert explanation["cache_probe_order"] == [
        "projected-delta",
        "exact-cache",
    ]


def test_transition_route_prefers_exact_first_when_measured_route_is_cheaper() -> None:
    history = TransitionRouteHistory()
    assert history.observe(
        _producer(
            cache_mode=CACHE_MODE_PROJECTION,
            exact_hits=8,
            exact_misses=2,
            exact_hit_ms=0.4,
            exact_miss_ms=0.1,
            projection_hits=1,
            projection_misses=1,
            projection_hit_ms=0.1,
            projection_miss_ms=0.1,
            fresh_count=1,
            fresh_ms=2.0,
            route_count=10,
            route_ms=2.7,
        )
    )
    assert history.observe(
        _producer(
            cache_mode=CACHE_MODE_PROJECTION_FIRST,
            exact_hits=6,
            exact_misses=2,
            exact_hit_ms=1.2,
            exact_miss_ms=0.2,
            projection_hits=2,
            projection_misses=8,
            projection_hit_ms=0.4,
            projection_miss_ms=1.6,
            fresh_count=2,
            fresh_ms=4.0,
            route_count=10,
            route_ms=7.4,
        )
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_PROJECTION
    assert explanation["selected_implementation"] == (
        "showdown:exact-then-projection-cache"
    )


def test_transition_route_can_still_choose_fresh_over_both_orders() -> None:
    history = TransitionRouteHistory()
    assert history.observe(
        _producer(
            cache_mode=CACHE_MODE_PROJECTION,
            exact_hits=0,
            exact_misses=10,
            exact_hit_ms=0.0,
            exact_miss_ms=10.0,
            projection_hits=0,
            projection_misses=10,
            projection_hit_ms=0.0,
            projection_miss_ms=10.0,
            fresh_count=10,
            fresh_ms=10.0,
            route_count=10,
            route_ms=30.0,
        )
    )
    assert history.observe(
        _producer(
            cache_mode=CACHE_MODE_PROJECTION_FIRST,
            exact_hits=0,
            exact_misses=10,
            exact_hit_ms=0.0,
            exact_miss_ms=10.0,
            projection_hits=0,
            projection_misses=10,
            projection_hit_ms=0.0,
            projection_miss_ms=10.0,
            fresh_count=10,
            fresh_ms=10.0,
            route_count=10,
            route_ms=30.0,
        )
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_FRESH
    assert explanation["selected_implementation"] == "showdown:fresh"


def test_transition_route_remeasures_less_sampled_order_periodically() -> None:
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
        exact_first_route_units=160,
        exact_first_route_total_ms=320.0,
        projection_first_route_units=80,
        projection_first_route_total_ms=120.0,
        observations=16,
    )

    mode, explanation = history.plan(semantic_signature="sha256:turn")

    assert mode == CACHE_MODE_PROJECTION_FIRST
    assert explanation["selection_basis"] == "periodic-order-refresh"


def test_transition_route_rejects_partial_timing_observations() -> None:
    history = TransitionRouteHistory()

    assert not history.observe(
        {"physical_cost_observations": {"fresh_execution_count": 3}}
    )
    assert history.observations == 0
