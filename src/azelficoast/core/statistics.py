"""Advisory cardinality statistics for physical planning.

Statistics may predict work; they never establish semantic equivalence or authorize a
computation. Exact validation and post-materialization cardinality checks remain the
authority.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import floor, isfinite
from typing import Sequence

from azelficoast.core.planning import LogicalOperator


@dataclass(frozen=True, slots=True)
class CategoricalDependency:
    """Weighted two-column dependency evidence for extended planner statistics."""

    left_name: str
    right_name: str
    total_weight: float
    left_distinct: int
    right_distinct: int
    joint_distinct: int
    total_variation_from_independence: float
    left_predicts_right_accuracy: float
    right_predicts_left_accuracy: float

    def __post_init__(self) -> None:
        if not self.left_name or not self.right_name:
            raise ValueError("categorical dependency names must be non-empty")
        if not isfinite(self.total_weight) or self.total_weight <= 0.0:
            raise ValueError("categorical dependency total weight must be positive")
        if min(self.left_distinct, self.right_distinct, self.joint_distinct) <= 0:
            raise ValueError("categorical dependency distinct counts must be positive")
        for value in (
            self.total_variation_from_independence,
            self.left_predicts_right_accuracy,
            self.right_predicts_left_accuracy,
        ):
            if not isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("categorical dependency metrics must be in [0, 1]")

    def as_record(self) -> dict[str, object]:
        return {
            "columns": [self.left_name, self.right_name],
            "total_weight": self.total_weight,
            "distinct": {
                self.left_name: self.left_distinct,
                self.right_name: self.right_distinct,
                "joint": self.joint_distinct,
            },
            "total_variation_from_independence": (
                self.total_variation_from_independence
            ),
            "functional_accuracy": {
                f"{self.left_name}->{self.right_name}": (
                    self.left_predicts_right_accuracy
                ),
                f"{self.right_name}->{self.left_name}": (
                    self.right_predicts_left_accuracy
                ),
            },
        }


def analyze_categorical_dependency(
    samples: Sequence[tuple[str, str, float]],
    *,
    left_name: str,
    right_name: str,
) -> CategoricalDependency:
    """Measure weighted departure from independence without asserting causality."""

    if not left_name or not right_name:
        raise ValueError("categorical dependency names must be non-empty")
    if not samples:
        raise ValueError("categorical dependency requires at least one sample")

    left_weights: dict[str, float] = defaultdict(float)
    right_weights: dict[str, float] = defaultdict(float)
    joint_weights: dict[tuple[str, str], float] = defaultdict(float)
    total = 0.0
    for left, right, weight in samples:
        if not left or not right:
            raise ValueError("categorical dependency values must be non-empty")
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("categorical dependency weights must be positive")
        left_weights[left] += weight
        right_weights[right] += weight
        joint_weights[(left, right)] += weight
        total += weight

    total_variation = 0.0
    for left, left_weight in left_weights.items():
        p_left = left_weight / total
        for right, right_weight in right_weights.items():
            p_right = right_weight / total
            p_joint = joint_weights.get((left, right), 0.0) / total
            total_variation += abs(p_joint - p_left * p_right)
    total_variation *= 0.5

    left_predicts_right = sum(
        max(
            joint_weights.get((left, right), 0.0)
            for right in right_weights
        )
        for left in left_weights
    ) / total
    right_predicts_left = sum(
        max(
            joint_weights.get((left, right), 0.0)
            for left in left_weights
        )
        for right in right_weights
    ) / total

    return CategoricalDependency(
        left_name=left_name,
        right_name=right_name,
        total_weight=total,
        left_distinct=len(left_weights),
        right_distinct=len(right_weights),
        joint_distinct=len(joint_weights),
        total_variation_from_independence=total_variation,
        left_predicts_right_accuracy=left_predicts_right,
        right_predicts_left_accuracy=right_predicts_left,
    )


@dataclass
class CardinalityHistogram:
    """Smoothed observed output/input ratio for one logical operator signature."""

    input_rows: int = 0
    output_rows: int = 0
    observations: int = 0
    prior_output_rows: float = 1.0
    prior_rejected_rows: float = 1.0

    def __post_init__(self) -> None:
        if self.input_rows < 0 or self.output_rows < 0 or self.observations < 0:
            raise ValueError("histogram counts must be non-negative")
        if self.output_rows > self.input_rows:
            raise ValueError("histogram output rows cannot exceed input rows")
        if (
            not isfinite(self.prior_output_rows)
            or not isfinite(self.prior_rejected_rows)
            or self.prior_output_rows <= 0.0
            or self.prior_rejected_rows <= 0.0
        ):
            raise ValueError("histogram priors must be finite and positive")

    @property
    def selectivity(self) -> float:
        rejected = self.input_rows - self.output_rows
        return (self.output_rows + self.prior_output_rows) / (
            self.output_rows
            + rejected
            + self.prior_output_rows
            + self.prior_rejected_rows
        )

    def observe(self, *, input_rows: int, output_rows: int) -> None:
        if input_rows < 0 or output_rows < 0:
            raise ValueError("cardinality observation must be non-negative")
        if output_rows > input_rows:
            raise ValueError("cardinality output cannot exceed input")
        self.input_rows += input_rows
        self.output_rows += output_rows
        self.observations += 1

    def estimate(self, input_rows: int) -> int:
        if input_rows < 0:
            raise ValueError("estimated input rows must be non-negative")
        if input_rows == 0:
            return 0
        estimate = floor(input_rows * self.selectivity + 0.5)
        return max(0, min(input_rows, estimate))


@dataclass(frozen=True, slots=True)
class CardinalityEstimate:
    operator: LogicalOperator
    signature: str
    input_rows: int
    estimated_output_rows: int
    estimated_selectivity: float
    observations: int

    def as_record(self) -> dict[str, object]:
        return {
            "logical_operator": self.operator.value,
            "signature": self.signature,
            "input_rows": self.input_rows,
            "estimated_output_rows": self.estimated_output_rows,
            "estimated_selectivity": self.estimated_selectivity,
            "observations": self.observations,
        }


class PlannerStatistics:
    """In-memory ANALYZE-style catalog keyed by logical operator and semantic signature."""

    def __init__(self) -> None:
        self._histograms: dict[
            tuple[LogicalOperator, str],
            CardinalityHistogram,
        ] = {}

    @staticmethod
    def _key(
        operator: LogicalOperator,
        signature: str,
    ) -> tuple[LogicalOperator, str]:
        if not signature:
            raise ValueError("statistics signature must be non-empty")
        return operator, signature

    def estimate(
        self,
        *,
        operator: LogicalOperator,
        signature: str,
        input_rows: int,
    ) -> CardinalityEstimate:
        key = self._key(operator, signature)
        histogram = self._histograms.get(key)
        if histogram is None:
            histogram = CardinalityHistogram()
        return CardinalityEstimate(
            operator=operator,
            signature=signature,
            input_rows=input_rows,
            estimated_output_rows=histogram.estimate(input_rows),
            estimated_selectivity=histogram.selectivity,
            observations=histogram.observations,
        )

    def observe(
        self,
        *,
        operator: LogicalOperator,
        signature: str,
        input_rows: int,
        output_rows: int,
    ) -> None:
        key = self._key(operator, signature)
        histogram = self._histograms.setdefault(key, CardinalityHistogram())
        histogram.observe(input_rows=input_rows, output_rows=output_rows)

    def histogram(
        self,
        *,
        operator: LogicalOperator,
        signature: str,
    ) -> CardinalityHistogram | None:
        return self._histograms.get(self._key(operator, signature))
