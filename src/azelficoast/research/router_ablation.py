"""Counterfactual analysis for the learned confidence router."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from azelficoast.belief.evaluator import build_evaluator_input
from azelficoast.search.transition_program import (
    TransitionProgramSearchError,
    search_transition_program,
)


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


def evaluate_admitted_counterfactual(
    *,
    case_id: str,
    public_state: Mapping[str, Any],
    legal_actions: Sequence[str],
    posterior: Mapping[str, Any],
    transition_program: Mapping[str, Any],
    evaluator: Any,
    search_gate: Any,
    fallback_action: str | None = None,
) -> RouterCase:
    """Evaluate direct and always-search choices on the same admitted support.

    The information-set search root values are the counterfactual value authority.
    This measures routing regret without changing the posterior, evaluator, mechanics
    program, or legal-action surface between the direct and searched routes.
    """

    actions = tuple(str(action) for action in legal_actions)
    if not actions or len(set(actions)) != len(actions):
        raise RouterAblationError("legal_actions must be unique non-empty strings")
    try:
        inputs = build_evaluator_input(
            public_state=public_state,
            posterior=posterior,
            legal_actions=actions,
            spec=evaluator.spec,
        )
        prediction = evaluator.predict(inputs)
    except Exception as error:
        raise RouterAblationError(f"root evaluator failed: {error}") from error
    if prediction.selected_action not in set(actions):
        raise RouterAblationError("direct evaluator returned a nonlegal action")

    search = None
    try:
        search = search_transition_program(
            program_set=transition_program,
            posterior=posterior,
            method="information_set",
            evaluator=evaluator,
        )
    except TransitionProgramSearchError:
        search = None

    if search is None:
        if fallback_action is None:
            actual_route: Route = "fallback" if search_gate.should_search(prediction) else "direct"
            fallback_regret = 0.0
        else:
            if fallback_action not in set(actions):
                raise RouterAblationError("fallback_action must be legal")
            actual_route = "fallback" if search_gate.should_search(prediction) else "direct"
            fallback_regret = 0.0
        return RouterCase(
            case_id=case_id,
            policy_margin=float(prediction.policy_margin),
            direct_regret=0.0,
            search_regret=None,
            fallback_regret=fallback_regret,
            actual_route=actual_route,
        )

    raw_values = search.get("root_values")
    if not isinstance(raw_values, Mapping) or set(map(str, raw_values)) != set(actions):
        raise RouterAblationError("search root values do not cover legal actions")
    values = {str(action): float(value) for action, value in raw_values.items()}
    if any(not math.isfinite(value) for value in values.values()):
        raise RouterAblationError("search root values must be finite")
    best = max(values.values())

    def regret(action: str) -> float:
        return max(0.0, best - values[action])

    direct_regret = regret(str(prediction.selected_action))
    searched_action = search.get("chosen_action")
    if not isinstance(searched_action, str) or searched_action not in values:
        raise RouterAblationError("search returned a nonlegal action")
    search_regret = regret(searched_action)

    if fallback_action is None:
        fallback_regret = direct_regret
    else:
        if fallback_action not in values:
            raise RouterAblationError("fallback_action must be legal")
        fallback_regret = regret(fallback_action)

    actual_route = (
        "search" if bool(search_gate.should_search(prediction)) else "direct"
    )
    return RouterCase(
        case_id=case_id,
        policy_margin=float(prediction.policy_margin),
        direct_regret=direct_regret,
        search_regret=search_regret,
        fallback_regret=fallback_regret,
        actual_route=actual_route,
    )
