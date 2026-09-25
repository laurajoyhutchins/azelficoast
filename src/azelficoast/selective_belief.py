"""Deterministic routing between learned policy/value and exact belief search."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from azelficoast.belief.evaluator import BeliefPrediction


class SelectiveBeliefError(ValueError):
    """Raised when a selective-search routing contract is invalid."""


@dataclass(frozen=True)
class PolicyMarginSearchGate:
    """Search when the learned policy is not separated enough to trust directly.

    A threshold of 1.0 is deliberately conservative: every position with two or
    more legal actions searches. Lower thresholds must be calibrated and supplied
    explicitly before the learned model can bypass search.
    """

    search_if_margin_at_most: float = 1.0

    def __post_init__(self) -> None:
        threshold = float(self.search_if_margin_at_most)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise SelectiveBeliefError("policy-margin threshold must be within [0, 1]")

    def should_search(self, prediction: BeliefPrediction) -> bool:
        if len(prediction.legal_actions) <= 1:
            return False
        return prediction.policy_margin <= float(self.search_if_margin_at_most)

    def as_record(self) -> dict[str, Any]:
        return {
            "kind": "policy_margin",
            "search_if_margin_at_most": float(self.search_if_margin_at_most),
        }
