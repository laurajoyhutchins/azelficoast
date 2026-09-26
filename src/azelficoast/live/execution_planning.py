"""Measured physical planning for the persistent live Showdown transition route."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping

from azelficoast.core.costing import (
    CacheTierCost,
    LocalityEvidence,
    estimate_cache_route,
)
from azelficoast.core.planning import (
    LogicalOperator,
    OperatorImplementation,
    choose_operator_implementation,
    explain_operator_plan,
)

CACHE_MODE_FRESH = "fresh"
CACHE_MODE_EXACT = "exact"
CACHE_MODE_PROJECTION = "projection"
CACHE_MODE_PROJECTION_FIRST = "projection-first"
_CACHE_MODES = {
    CACHE_MODE_FRESH,
    CACHE_MODE_EXACT,
    CACHE_MODE_PROJECTION,
    CACHE_MODE_PROJECTION_FIRST,
}


@dataclass
class TransitionRouteHistory:
    """Aggregate measured cache locality and latency across live search turns."""

    exact_hits: int = 0
    exact_misses: int = 0
    exact_hit_total_ms: float = 0.0
    exact_miss_total_ms: float = 0.0
    projection_hits: int = 0
    projection_misses: int = 0
    projection_hit_total_ms: float = 0.0
    projection_miss_total_ms: float = 0.0
    fresh_executions: int = 0
    fresh_execution_total_ms: float = 0.0
    exact_first_route_units: int = 0
    exact_first_route_total_ms: float = 0.0
    projection_first_route_units: int = 0
    projection_first_route_total_ms: float = 0.0
    observations: int = 0

    @staticmethod
    def _count(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @staticmethod
    def _duration(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        number = float(value)
        if not isfinite(number) or number < 0:
            return None
        return number

    def observe(self, producer: Mapping[str, Any]) -> bool:
        """Accumulate one complete producer timing observation.

        Malformed or partial observations are ignored rather than contaminating future
        route decisions.
        """

        raw = producer.get("physical_cost_observations")
        if not isinstance(raw, Mapping):
            return False

        exact_hits = self._count(raw.get("exact_cache_hit_count"))
        exact_misses = self._count(raw.get("exact_cache_miss_count"))
        exact_hit_total_ms = self._duration(raw.get("exact_cache_hit_total_ms"))
        exact_miss_total_ms = self._duration(raw.get("exact_cache_miss_total_ms"))
        projection_hits = self._count(raw.get("projection_cache_hit_count"))
        projection_misses = self._count(raw.get("projection_cache_miss_count"))
        projection_hit_total_ms = self._duration(
            raw.get("projection_cache_hit_total_ms")
        )
        projection_miss_total_ms = self._duration(
            raw.get("projection_cache_miss_total_ms")
        )
        fresh_executions = self._count(raw.get("fresh_execution_count"))
        fresh_execution_total_ms = self._duration(
            raw.get("fresh_execution_total_ms")
        )
        route_units = self._count(raw.get("route_execution_count"))
        route_total_ms = self._duration(raw.get("route_total_ms"))
        cache_mode = producer.get("transition_cache_mode")

        if (
            exact_hits is None
            or exact_misses is None
            or exact_hit_total_ms is None
            or exact_miss_total_ms is None
            or projection_hits is None
            or projection_misses is None
            or projection_hit_total_ms is None
            or projection_miss_total_ms is None
            or fresh_executions is None
            or fresh_execution_total_ms is None
        ):
            return False
        if (route_units is None) != (route_total_ms is None):
            return False
        if cache_mode is not None and cache_mode not in _CACHE_MODES:
            return False

        self.exact_hits += exact_hits
        self.exact_misses += exact_misses
        self.exact_hit_total_ms += exact_hit_total_ms
        self.exact_miss_total_ms += exact_miss_total_ms
        self.projection_hits += projection_hits
        self.projection_misses += projection_misses
        self.projection_hit_total_ms += projection_hit_total_ms
        self.projection_miss_total_ms += projection_miss_total_ms
        self.fresh_executions += fresh_executions
        self.fresh_execution_total_ms += fresh_execution_total_ms
        if route_units is not None and route_total_ms is not None:
            if cache_mode == CACHE_MODE_PROJECTION:
                self.exact_first_route_units += route_units
                self.exact_first_route_total_ms += route_total_ms
            elif cache_mode == CACHE_MODE_PROJECTION_FIRST:
                self.projection_first_route_units += route_units
                self.projection_first_route_total_ms += route_total_ms
        self.observations += 1
        return True

    @staticmethod
    def _mean(total: float, count: int, *, fallback: float) -> float:
        return total / count if count > 0 else fallback

    def plan(self, *, semantic_signature: str) -> tuple[str, Mapping[str, Any]]:
        """Choose an exact transition route and, when measured, its cache-probe order."""

        if not semantic_signature:
            raise ValueError("semantic signature must be non-empty")

        if self.exact_first_route_units <= 0:
            return (
                CACHE_MODE_PROJECTION,
                {
                    "schema": "azelficoast.live-transition-route-plan",
                    "schema_version": 2,
                    "logical_operator": LogicalOperator.TRANSITION.value,
                    "semantic_signature": semantic_signature,
                    "selected_implementation": "showdown:exact-then-projection-cache",
                    "cache_mode": CACHE_MODE_PROJECTION,
                    "cache_probe_order": ["exact-cache", "projected-delta"],
                    "selection_basis": "cold-start-order-measurement",
                    "history_observations": self.observations,
                },
            )

        if self.projection_first_route_units <= 0:
            return (
                CACHE_MODE_PROJECTION_FIRST,
                {
                    "schema": "azelficoast.live-transition-route-plan",
                    "schema_version": 2,
                    "logical_operator": LogicalOperator.TRANSITION.value,
                    "semantic_signature": semantic_signature,
                    "selected_implementation": "showdown:projection-then-exact-cache",
                    "cache_mode": CACHE_MODE_PROJECTION_FIRST,
                    "cache_probe_order": ["projected-delta", "exact-cache"],
                    "selection_basis": "cold-start-order-measurement",
                    "history_observations": self.observations,
                },
            )

        if self.fresh_executions <= 0:
            return (
                CACHE_MODE_PROJECTION,
                {
                    "schema": "azelficoast.live-transition-route-plan",
                    "schema_version": 2,
                    "logical_operator": LogicalOperator.TRANSITION.value,
                    "semantic_signature": semantic_signature,
                    "selected_implementation": "showdown:exact-then-projection-cache",
                    "cache_mode": CACHE_MODE_PROJECTION,
                    "cache_probe_order": ["exact-cache", "projected-delta"],
                    "selection_basis": "fresh-cost-measurement",
                    "history_observations": self.observations,
                },
            )

        if self.observations > 0 and self.observations % 16 == 0:
            refresh_mode = (
                CACHE_MODE_PROJECTION
                if self.exact_first_route_units <= self.projection_first_route_units
                else CACHE_MODE_PROJECTION_FIRST
            )
            refresh_order = (
                ["exact-cache", "projected-delta"]
                if refresh_mode == CACHE_MODE_PROJECTION
                else ["projected-delta", "exact-cache"]
            )
            refresh_name = (
                "showdown:exact-then-projection-cache"
                if refresh_mode == CACHE_MODE_PROJECTION
                else "showdown:projection-then-exact-cache"
            )
            return (
                refresh_mode,
                {
                    "schema": "azelficoast.live-transition-route-plan",
                    "schema_version": 2,
                    "logical_operator": LogicalOperator.TRANSITION.value,
                    "semantic_signature": semantic_signature,
                    "selected_implementation": refresh_name,
                    "cache_mode": refresh_mode,
                    "cache_probe_order": refresh_order,
                    "selection_basis": "periodic-order-refresh",
                    "history_observations": self.observations,
                },
            )

        fresh_ms = self.fresh_execution_total_ms / self.fresh_executions
        exact_miss_ms = self._mean(
            self.exact_miss_total_ms,
            self.exact_misses,
            fallback=0.0,
        )
        exact_hit_ms = self._mean(
            self.exact_hit_total_ms,
            self.exact_hits,
            fallback=exact_miss_ms,
        )

        exact_tier = CacheTierCost(
            name="exact-cache",
            locality=LocalityEvidence(
                hits=self.exact_hits,
                misses=self.exact_misses,
            ),
            hit_cost_ms=exact_hit_ms,
            miss_cost_ms=exact_miss_ms,
        )
        exact_route = estimate_cache_route((exact_tier,), fallback_ms=fresh_ms)

        exact_first_ms = (
            self.exact_first_route_total_ms / self.exact_first_route_units
        )
        projection_first_ms = (
            self.projection_first_route_total_ms
            / self.projection_first_route_units
        )

        plan = choose_operator_implementation(
            (
                OperatorImplementation(
                    operator=LogicalOperator.TRANSITION,
                    name="showdown:fresh",
                    semantic_signature=semantic_signature,
                    predicted_ms=fresh_ms,
                    evidence={"route": [], "fallback": "fresh-showdown"},
                ),
                OperatorImplementation(
                    operator=LogicalOperator.TRANSITION,
                    name="showdown:exact-cache",
                    semantic_signature=semantic_signature,
                    predicted_ms=exact_route.expected_ms,
                    evidence={
                        "route": list(exact_route.tier_names),
                        "fallback": "fresh-showdown",
                        "hit_probabilities": list(exact_route.hit_probabilities),
                        "remaining_miss_probability": (
                            exact_route.remaining_miss_probability
                        ),
                    },
                ),
                OperatorImplementation(
                    operator=LogicalOperator.TRANSITION,
                    name="showdown:exact-then-projection-cache",
                    semantic_signature=semantic_signature,
                    predicted_ms=exact_first_ms,
                    evidence={
                        "route": ["exact-cache", "projected-delta"],
                        "measurement": "observed-route-cost-per-execution",
                        "sample_units": self.exact_first_route_units,
                    },
                ),
                OperatorImplementation(
                    operator=LogicalOperator.TRANSITION,
                    name="showdown:projection-then-exact-cache",
                    semantic_signature=semantic_signature,
                    predicted_ms=projection_first_ms,
                    evidence={
                        "route": ["projected-delta", "exact-cache"],
                        "measurement": "observed-route-cost-per-execution",
                        "sample_units": self.projection_first_route_units,
                    },
                ),
            )
        )
        modes = {
            "showdown:fresh": CACHE_MODE_FRESH,
            "showdown:exact-cache": CACHE_MODE_EXACT,
            "showdown:exact-then-projection-cache": CACHE_MODE_PROJECTION,
            "showdown:projection-then-exact-cache": CACHE_MODE_PROJECTION_FIRST,
        }
        selected_mode = modes[plan.selected.name]
        if selected_mode not in _CACHE_MODES:
            raise RuntimeError("planner selected an unknown transition cache mode")

        explanation = explain_operator_plan(plan)
        explanation.update(
            {
                "schema_version": 2,
                "cache_mode": selected_mode,
                "cache_probe_order": {
                    CACHE_MODE_FRESH: [],
                    CACHE_MODE_EXACT: ["exact-cache"],
                    CACHE_MODE_PROJECTION: [
                        "exact-cache",
                        "projected-delta",
                    ],
                    CACHE_MODE_PROJECTION_FIRST: [
                        "projected-delta",
                        "exact-cache",
                    ],
                }[selected_mode],
                "selection_basis": "measured-route-order-cost",
                "history_observations": self.observations,
                "history": {
                    "exact_hits": self.exact_hits,
                    "exact_misses": self.exact_misses,
                    "projection_hits": self.projection_hits,
                    "projection_misses": self.projection_misses,
                    "fresh_executions": self.fresh_executions,
                    "exact_first_route_units": self.exact_first_route_units,
                    "projection_first_route_units": (
                        self.projection_first_route_units
                    ),
                    "exact_first_ms_per_execution": exact_first_ms,
                    "projection_first_ms_per_execution": projection_first_ms,
                },
            }
        )
        return selected_mode, explanation
