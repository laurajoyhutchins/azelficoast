"""Execution boundary between search topology and numerical evaluation.

Search owns which successor information sets exist and how their values contribute to
root actions. Numerical backends own only the value of each already-constructed leaf.
This keeps information-flow semantics in ordinary Python while permitting one batched
JAX dispatch for dense learned evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence


class EvaluationFrontierError(ValueError):
    """Raised when a search frontier cannot be reduced safely."""


@dataclass(frozen=True, slots=True)
class EvaluationLeaf:
    """One public successor information set awaiting numerical evaluation."""

    public_state: Any
    posterior: Any
    legal_actions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.legal_actions or any(not action for action in self.legal_actions):
            raise EvaluationFrontierError("evaluation leaf must have legal actions")
        if len(set(self.legal_actions)) != len(self.legal_actions):
            raise EvaluationFrontierError("evaluation leaf legal actions must be unique")


@dataclass(frozen=True, slots=True)
class EvaluationContribution:
    """Linear contribution of one leaf value to one root action."""

    root_action: str
    leaf_index: int
    coefficient: float

    def __post_init__(self) -> None:
        if not self.root_action:
            raise EvaluationFrontierError("evaluation contribution lacks a root action")
        if self.leaf_index < 0:
            raise EvaluationFrontierError("evaluation contribution has a negative leaf index")
        if not math.isfinite(self.coefficient) or self.coefficient <= 0.0:
            raise EvaluationFrontierError(
                "evaluation contribution coefficient must be positive and finite"
            )


@dataclass(frozen=True, slots=True)
class EvaluationFrontier:
    """Complete leaf frontier plus the deterministic reduction back to root values."""

    root_actions: tuple[str, ...]
    leaves: tuple[EvaluationLeaf, ...]
    contributions: tuple[EvaluationContribution, ...]
    transition_evaluations: int

    def __post_init__(self) -> None:
        if not self.root_actions or any(not action for action in self.root_actions):
            raise EvaluationFrontierError("evaluation frontier must have root actions")
        if len(set(self.root_actions)) != len(self.root_actions):
            raise EvaluationFrontierError("evaluation frontier root actions must be unique")
        if not self.leaves:
            raise EvaluationFrontierError("evaluation frontier has no leaves")
        if self.transition_evaluations <= 0:
            raise EvaluationFrontierError(
                "evaluation frontier must consume positive transition work"
            )
        root_set = set(self.root_actions)
        for contribution in self.contributions:
            if contribution.root_action not in root_set:
                raise EvaluationFrontierError(
                    "evaluation contribution references an unknown root action"
                )
            if contribution.leaf_index >= len(self.leaves):
                raise EvaluationFrontierError(
                    "evaluation contribution references an unknown leaf"
                )

    def reduce(self, values: tuple[float, ...]) -> dict[str, float]:
        """Reduce backend leaf values into root-action values deterministically."""

        if len(values) != len(self.leaves):
            raise EvaluationFrontierError(
                "evaluation backend returned the wrong number of leaf values"
            )
        if any(not math.isfinite(value) for value in values):
            raise EvaluationFrontierError(
                "evaluation backend returned a non-finite leaf value"
            )
        root_values = {action: 0.0 for action in self.root_actions}
        for contribution in self.contributions:
            root_values[contribution.root_action] += (
                contribution.coefficient * values[contribution.leaf_index]
            )
        return root_values



@dataclass(frozen=True, slots=True)
class ActionValueInterval:
    """Certified root-action interval under a bounded leaf evaluator."""

    action: str
    lower: float
    upper: float
    evaluated_leaves: int
    total_leaves: int
    exact: bool

    def __post_init__(self) -> None:
        if not self.action:
            raise EvaluationFrontierError("action interval lacks an action")
        if (
            not math.isfinite(self.lower)
            or not math.isfinite(self.upper)
            or self.lower > self.upper
        ):
            raise EvaluationFrontierError("action interval bounds are invalid")
        if (
            self.evaluated_leaves < 0
            or self.total_leaves <= 0
            or self.evaluated_leaves > self.total_leaves
        ):
            raise EvaluationFrontierError("action interval leaf counts are invalid")
        if self.exact and self.evaluated_leaves != self.total_leaves:
            raise EvaluationFrontierError(
                "exact action interval must evaluate every action leaf"
            )

    def as_record(self) -> dict[str, object]:
        return {
            "action": self.action,
            "lower": self.lower,
            "upper": self.upper,
            "evaluated_leaves": self.evaluated_leaves,
            "total_leaves": self.total_leaves,
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class BoundedActionDecision:
    """Exact winner certificate with intervals for any unevaluated losers."""

    chosen_action: str
    chosen_value: float
    intervals: tuple[ActionValueInterval, ...]
    exact_root_values: tuple[tuple[str, float], ...]
    evaluated_leaf_count: int
    total_leaf_count: int
    evaluation_batches: int
    value_lower_bound: float
    value_upper_bound: float

    @property
    def pruned_leaf_count(self) -> int:
        return self.total_leaf_count - self.evaluated_leaf_count

    @property
    def pruned_actions(self) -> tuple[str, ...]:
        return tuple(
            interval.action
            for interval in self.intervals
            if not interval.exact
        )

    def as_record(self) -> dict[str, object]:
        return {
            "chosen_action": self.chosen_action,
            "chosen_value": self.chosen_value,
            "action_value_intervals": [
                interval.as_record() for interval in self.intervals
            ],
            "exact_root_values": {
                action: value for action, value in self.exact_root_values
            },
            "evaluated_leaf_count": self.evaluated_leaf_count,
            "total_leaf_count": self.total_leaf_count,
            "pruned_leaf_count": self.pruned_leaf_count,
            "pruned_actions": list(self.pruned_actions),
            "evaluation_batches": self.evaluation_batches,
            "value_bounds": {
                "lower": self.value_lower_bound,
                "upper": self.value_upper_bound,
            },
        }


def choose_bounded_action(
    *,
    root_actions: Sequence[str],
    leaf_action_index: Sequence[int],
    coefficients: Sequence[float],
    evaluate: Callable[[tuple[int, ...]], Sequence[float]],
    value_lower_bound: float,
    value_upper_bound: float,
    batch_size: int = 1,
) -> BoundedActionDecision:
    """Return the exact winning action while pruning certifiably dominated leaves.

    This is a winner-only physical operator. A pruned action retains an interval rather
    than an invented exact score. Pruning occurs only when the action's conservative
    upper bound is strictly below the conservative lower bound of a fully evaluated
    incumbent.
    """

    actions = tuple(root_actions)
    leaf_actions = tuple(leaf_action_index)
    weights = tuple(float(value) for value in coefficients)
    if not actions or any(not action for action in actions):
        raise EvaluationFrontierError("bounded decision requires root actions")
    if len(set(actions)) != len(actions):
        raise EvaluationFrontierError("bounded decision root actions must be unique")
    if not leaf_actions or len(leaf_actions) != len(weights):
        raise EvaluationFrontierError(
            "bounded decision leaf indices and coefficients must align"
        )
    if (
        not math.isfinite(value_lower_bound)
        or not math.isfinite(value_upper_bound)
        or value_lower_bound >= value_upper_bound
    ):
        raise EvaluationFrontierError("bounded evaluator range is invalid")
    if batch_size <= 0:
        raise EvaluationFrontierError("bounded evaluator batch size must be positive")

    by_action: list[list[tuple[int, float]]] = [[] for _ in actions]
    for leaf_index, (action_index, coefficient) in enumerate(
        zip(leaf_actions, weights, strict=True)
    ):
        if not 0 <= action_index < len(actions):
            raise EvaluationFrontierError(
                "bounded decision leaf references an unknown action"
            )
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise EvaluationFrontierError(
                "bounded decision coefficients must be positive and finite"
            )
        by_action[action_index].append((leaf_index, coefficient))
    if any(not rows for rows in by_action):
        raise EvaluationFrontierError(
            "bounded decision requires at least one leaf per action"
        )

    values: dict[int, float] = {}
    evaluation_batches = 0

    def evaluate_indices(indices: tuple[int, ...]) -> None:
        nonlocal evaluation_batches
        if not indices:
            return
        raw = tuple(float(value) for value in evaluate(indices))
        evaluation_batches += 1
        if len(raw) != len(indices):
            raise EvaluationFrontierError(
                "bounded evaluator returned the wrong number of values"
            )
        for leaf_index, value in zip(indices, raw, strict=True):
            if not math.isfinite(value):
                raise EvaluationFrontierError(
                    "bounded evaluator returned a non-finite value"
                )
            if value < value_lower_bound or value > value_upper_bound:
                raise EvaluationFrontierError(
                    "bounded evaluator returned a value outside its certified range"
                )
            values[leaf_index] = value

    def action_value(action_index: int) -> float:
        rows = sorted(by_action[action_index], key=lambda row: row[0])
        return math.fsum(
            coefficient * values[leaf_index]
            for leaf_index, coefficient in rows
        )

    def outward_interval(
        *,
        partial_terms: Sequence[float],
        remaining_mass: float,
    ) -> tuple[float, float]:
        partial = math.fsum(partial_terms)
        lower = math.nextafter(
            partial + remaining_mass * value_lower_bound,
            -math.inf,
        )
        upper = math.nextafter(
            partial + remaining_mass * value_upper_bound,
            math.inf,
        )
        return lower, upper

    exact_values: dict[int, float] = {}
    intervals: dict[int, ActionValueInterval] = {}

    seed_index = min(
        range(len(actions)),
        key=lambda index: (len(by_action[index]), actions[index]),
    )
    seed_rows = tuple(
        leaf_index
        for leaf_index, _ in sorted(
            by_action[seed_index],
            key=lambda row: (-row[1], row[0]),
        )
    )
    for offset in range(0, len(seed_rows), batch_size):
        evaluate_indices(seed_rows[offset : offset + batch_size])
    seed_value = action_value(seed_index)
    exact_values[seed_index] = seed_value
    intervals[seed_index] = ActionValueInterval(
        action=actions[seed_index],
        lower=seed_value,
        upper=seed_value,
        evaluated_leaves=len(by_action[seed_index]),
        total_leaves=len(by_action[seed_index]),
        exact=True,
    )
    incumbent_index = seed_index
    incumbent_value = seed_value

    remaining_actions = sorted(
        (index for index in range(len(actions)) if index != seed_index),
        key=lambda index: (len(by_action[index]), actions[index]),
    )
    for action_index in remaining_actions:
        ordered = sorted(
            by_action[action_index],
            key=lambda row: (-row[1], row[0]),
        )
        total_mass = math.fsum(coefficient for _, coefficient in ordered)
        evaluated_mass = 0.0
        partial_terms: list[float] = []
        evaluated_count = 0
        pruned = False

        for offset in range(0, len(ordered), batch_size):
            chunk = ordered[offset : offset + batch_size]
            indices = tuple(leaf_index for leaf_index, _ in chunk)
            evaluate_indices(indices)
            for leaf_index, coefficient in chunk:
                evaluated_mass = math.fsum((evaluated_mass, coefficient))
                partial_terms.append(coefficient * values[leaf_index])
                evaluated_count += 1

            remaining_mass = max(0.0, total_mass - evaluated_mass)
            lower, upper = outward_interval(
                partial_terms=partial_terms,
                remaining_mass=remaining_mass,
            )
            incumbent_lower = math.nextafter(incumbent_value, -math.inf)
            if evaluated_count < len(ordered) and upper < incumbent_lower:
                intervals[action_index] = ActionValueInterval(
                    action=actions[action_index],
                    lower=lower,
                    upper=upper,
                    evaluated_leaves=evaluated_count,
                    total_leaves=len(ordered),
                    exact=False,
                )
                pruned = True
                break

        if pruned:
            continue

        exact = action_value(action_index)
        exact_values[action_index] = exact
        intervals[action_index] = ActionValueInterval(
            action=actions[action_index],
            lower=exact,
            upper=exact,
            evaluated_leaves=len(ordered),
            total_leaves=len(ordered),
            exact=True,
        )
        if (
            exact > incumbent_value
            or (
                exact == incumbent_value
                and actions[action_index] < actions[incumbent_index]
            )
        ):
            incumbent_index = action_index
            incumbent_value = exact

    ordered_intervals = tuple(intervals[index] for index in range(len(actions)))
    exact_root_values = tuple(
        (actions[index], exact_values[index])
        for index in range(len(actions))
        if index in exact_values
    )
    return BoundedActionDecision(
        chosen_action=actions[incumbent_index],
        chosen_value=incumbent_value,
        intervals=ordered_intervals,
        exact_root_values=exact_root_values,
        evaluated_leaf_count=len(values),
        total_leaf_count=len(leaf_actions),
        evaluation_batches=evaluation_batches,
        value_lower_bound=value_lower_bound,
        value_upper_bound=value_upper_bound,
    )
