"""Counterfactual analysis for the learned confidence router."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence


Route = Literal["direct", "search", "fallback"]


class RouterAblationError(ValueError):
    """Raised when router-ablation evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class RouterCase:
    """One admitted decision with direct/search/fallback counterfactuals."""

    case_id: str
    policy_margin: float
    direct_regret: float
    search_regret: float | None
    fallback_regret: float
    actual_route: Route
    direct_outcome: float | None = None
    search_outcome: float | None = None
    fallback_outcome: float | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise RouterAblationError("case_id must be non-empty")
        if not math.isfinite(self.policy_margin) or not 0.0 <= self.policy_margin <= 1.0:
            raise RouterAblationError("policy_margin must be finite and within [0, 1]")
        for name, value in (
            ("direct_regret", self.direct_regret),
            ("fallback_regret", self.fallback_regret),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise RouterAblationError(f"{name} must be finite and non-negative")
        if self.search_regret is not None and (
            not math.isfinite(self.search_regret) or self.search_regret < 0.0
        ):
            raise RouterAblationError("search_regret must be finite and non-negative")
        if self.actual_route not in {"direct", "search", "fallback"}:
            raise RouterAblationError("actual_route is invalid")
        if self.actual_route == "search" and self.search_regret is None:
            raise RouterAblationError("searched case lacks search counterfactual")
        for name, value in (
            ("direct_outcome", self.direct_outcome),
            ("search_outcome", self.search_outcome),
            ("fallback_outcome", self.fallback_outcome),
        ):
            if value is not None and not math.isfinite(value):
                raise RouterAblationError(f"{name} must be finite when supplied")


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _regret_for_route(case: RouterCase, route: Route) -> float:
    if route == "direct":
        return case.direct_regret
    if route == "search":
        if case.search_regret is None:
            raise RouterAblationError(f"{case.case_id}: search counterfactual unavailable")
        return case.search_regret
    return case.fallback_regret


def _outcome_for_route(case: RouterCase, route: Route) -> float | None:
    if route == "direct":
        return case.direct_outcome
    if route == "search":
        return case.search_outcome
    return case.fallback_outcome


def summarize_observed_routes(cases: Sequence[RouterCase]) -> dict[str, object]:
    """Report regret/outcome conditioned on the route the live system actually took."""

    if not cases:
        raise RouterAblationError("at least one router case is required")
    rows: dict[str, dict[str, object]] = {}
    for route in ("direct", "search", "fallback"):
        selected = [case for case in cases if case.actual_route == route]
        regrets = [_regret_for_route(case, route) for case in selected]
        outcomes = [
            value
            for case in selected
            if (value := _outcome_for_route(case, route)) is not None
        ]
        rows[route] = {
            "case_count": len(selected),
            "mean_regret": _mean(regrets),
            "mean_outcome": _mean(outcomes),
            "outcome_case_count": len(outcomes),
        }

    return {
        "case_count": len(cases),
        "routes": rows,
        "overall_mean_regret": _mean(
            [_regret_for_route(case, case.actual_route) for case in cases]
        ),
        "always_direct_mean_regret": _mean([case.direct_regret for case in cases]),
        "always_search_mean_regret": _mean(
            [
                float(case.search_regret)
                for case in cases
                if case.search_regret is not None
            ]
        ),
        "always_search_coverage_rate": (
            sum(case.search_regret is not None for case in cases) / len(cases)
        ),
    }


def threshold_sweep(
    cases: Sequence[RouterCase],
    *,
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    """Evaluate confidence thresholds against direct/search counterfactual regret."""

    if not cases:
        raise RouterAblationError("at least one router case is required")
    if not thresholds:
        raise RouterAblationError("at least one confidence threshold is required")

    output: list[dict[str, object]] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise RouterAblationError("thresholds must be finite and within [0, 1]")

        routes: list[Route] = []
        regrets: list[float] = []
        outcomes: list[float] = []
        for case in cases:
            if case.policy_margin > threshold:
                route: Route = "direct"
            elif case.search_regret is not None:
                route = "search"
            else:
                route = "fallback"
            routes.append(route)
            regrets.append(_regret_for_route(case, route))
            outcome = _outcome_for_route(case, route)
            if outcome is not None:
                outcomes.append(outcome)

        output.append(
            {
                "threshold": threshold,
                "search_count": routes.count("search"),
                "direct_count": routes.count("direct"),
                "fallback_count": routes.count("fallback"),
                "search_rate": routes.count("search") / len(routes),
                "fallback_rate": routes.count("fallback") / len(routes),
                "mean_regret": _mean(regrets),
                "mean_outcome": _mean(outcomes),
                "outcome_case_count": len(outcomes),
            }
        )
    return output
