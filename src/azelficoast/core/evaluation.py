"""Execution boundary between search topology and numerical evaluation.

Search owns which successor information sets exist and how their values contribute to
root actions. Numerical backends own only the value of each already-constructed leaf.
This keeps information-flow semantics in ordinary Python while permitting one batched
JAX dispatch for dense learned evaluation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


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
