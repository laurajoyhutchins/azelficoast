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
_CACHE_MODES = {
    CACHE_MODE_FRESH,
    CACHE_MODE_EXACT,
    CACHE_MODE_PROJECTION,
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
        self.observations += 1
        return True

    @staticmethod
    def _mean(total: float, count: int, *, fallback: float) -> float:
        return total / count if count > 0 else fallback

    def plan(self, *, semantic_signature: str) -> tuple[str, Mapping[str, Any]]:
        """Choose fresh, exact-cache, or projected-delta execution from measured cost."""

        if not semantic_signature:
            raise ValueError("semantic signature must be non-empty")

        # One full-route observation gives the planner measurements for all misses and,
        # when locality exists immediately, cache-hit costs as well.
        if self.fresh_executions <= 0:
            return (
                CACHE_MODE_PROJECTION,
                {
                    "schema": "azelficoast.live-transition-route-plan",
                    "schema_version": 1,
                    "logical_operator": LogicalOperator.TRANSITION.value,
                    "semantic_signature": semantic_signature,
                    "selected_implementation": "showdown:projected-delta-cache",
                    "cache_mode": CACHE_MODE_PROJECTION,
                    "selection_basis": "cold-start-measurement",
                    "history_observations": self.observations,
                },
            )

        # A route that stops querying a cache would otherwise stop learning whether
        # locality has changed. Refresh the full route deterministically once every
        # sixteen observed program compilations.
        if self.observations > 0 and self.observations % 16 == 0:
            return (
                CACHE_MODE_PROJECTION,
                {
                    "schema": "azelficoast.live-transition-route-plan",
                    "schema_version": 1,
                    "logical_operator": LogicalOperator.TRANSITION.value,
                    "semantic_signature": semantic_signature,
                    "selected_implementation": "showdown:projected-delta-cache",
                    "cache_mode": CACHE_MODE_PROJECTION,
                    "selection_basis": "periodic-locality-refresh",
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
        projection_miss_ms = self._mean(
            self.projection_miss_total_ms,
            self.projection_misses,
            fallback=0.0,
        )
        projection_hit_ms = self._mean(
            self.projection_hit_total_ms,
            self.projection_hits,
            fallback=projection_miss_ms,
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
        projection_tier = CacheTierCost(
            name="projected-delta",
            locality=LocalityEvidence(
                hits=self.projection_hits,
                misses=self.projection_misses,
            ),
            hit_cost_ms=projection_hit_ms,
            miss_cost_ms=projection_miss_ms,
        )

        exact_route = estimate_cache_route((exact_tier,), fallback_ms=fresh_ms)
        projection_route = estimate_cache_route(
            (exact_tier, projection_tier),
            fallback_ms=fresh_ms,
        )

        exact_evidence = {
            "route": list(exact_route.tier_names),
            "fallback": "fresh-showdown",
            "hit_probabilities": list(exact_route.hit_probabilities),
            "remaining_miss_probability": exact_route.remaining_miss_probability,
        }
        projection_evidence = {
            "route": list(projection_route.tier_names),
            "fallback": "fresh-showdown",
            "hit_probabilities": list(projection_route.hit_probabilities),
            "remaining_miss_probability": projection_route.remaining_miss_probability,
        }
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
                    evidence=exact_evidence,
                ),
                OperatorImplementation(
                    operator=LogicalOperator.TRANSITION,
                    name="showdown:projected-delta-cache",
                    semantic_signature=semantic_signature,
                    predicted_ms=projection_route.expected_ms,
                    evidence=projection_evidence,
                ),
            )
        )
        modes = {
            "showdown:fresh": CACHE_MODE_FRESH,
            "showdown:exact-cache": CACHE_MODE_EXACT,
            "showdown:projected-delta-cache": CACHE_MODE_PROJECTION,
        }
        selected_mode = modes[plan.selected.name]
        if selected_mode not in _CACHE_MODES:
            raise RuntimeError("planner selected an unknown transition cache mode")

        explanation = explain_operator_plan(plan)
        explanation.update(
            {
                "cache_mode": selected_mode,
                "selection_basis": "measured-locality-and-latency",
                "history_observations": self.observations,
                "history": {
                    "exact_hits": self.exact_hits,
                    "exact_misses": self.exact_misses,
                    "projection_hits": self.projection_hits,
                    "projection_misses": self.projection_misses,
                    "fresh_executions": self.fresh_executions,
                },
            }
        )
        return selected_mode, explanation
