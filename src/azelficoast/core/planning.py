"""Logical and physical planning for finite partial-information computations.

Search owns the question being asked. Planning owns how an equivalent exact computation
is executed. This is deliberately a small relational-style surface rather than a second
search implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from azelficoast.core.costing import (
    ExecutionCostProfile,
    ExecutionDecision,
    ExecutionFeatures,
    choose_execution_path,
)

PLAN_EXPLAIN_SCHEMA = "azelficoast.core.execution-plan-explain"
PLAN_EXPLAIN_SCHEMA_VERSION = 1


class LogicalOperator(str, Enum):
    SCAN = "scan"
    FILTER = "filter"
    PROJECT = "project"
    PARTITION = "partition"
    TRANSITION = "transition"
    OBSERVE = "observe"
    UPDATE_BELIEF = "update_belief"
    EVALUATE = "evaluate"
    AGGREGATE = "aggregate"


@dataclass(frozen=True)
class LogicalPlan:
    operators: tuple[LogicalOperator, ...]

    def __post_init__(self) -> None:
        if not self.operators:
            raise ValueError("logical plan must contain at least one operator")


DEFAULT_DECISION_PLAN = LogicalPlan(
    operators=(
        LogicalOperator.SCAN,
        LogicalOperator.FILTER,
        LogicalOperator.PROJECT,
        LogicalOperator.PARTITION,
        LogicalOperator.TRANSITION,
        LogicalOperator.OBSERVE,
        LogicalOperator.UPDATE_BELIEF,
        LogicalOperator.EVALUATE,
        LogicalOperator.AGGREGATE,
    )
)


@dataclass(frozen=True)
class PhysicalPlan:
    logical: LogicalPlan
    features: ExecutionFeatures
    decision: ExecutionDecision


def choose_physical_plan(
    profile: ExecutionCostProfile,
    features: ExecutionFeatures,
    *,
    logical: LogicalPlan = DEFAULT_DECISION_PLAN,
) -> PhysicalPlan:
    """Choose an exact physical implementation for one logical computation."""

    return PhysicalPlan(
        logical=logical,
        features=features,
        decision=choose_execution_path(profile, features),
    )


def explain_physical_plan(plan: PhysicalPlan) -> dict[str, Any]:
    """Return deterministic evidence for why a physical path was selected."""

    features = plan.features
    decision = plan.decision
    return {
        "schema": PLAN_EXPLAIN_SCHEMA,
        "schema_version": PLAN_EXPLAIN_SCHEMA_VERSION,
        "logical_operators": [operator.value for operator in plan.logical.operators],
        "features": {
            "backend": features.backend,
            "target_signature": features.target_signature,
            "effect_signature": features.effect_signature,
            "logical_world_count": features.logical_world_count,
            "active_canonical_classes": features.active_canonical_classes,
            "active_projected_classes": features.active_projected_classes,
        },
        "physical": {
            "selected_path": decision.path.value,
            "predicted_direct_ms": decision.predicted_direct_ms,
            "predicted_projected_ms": decision.predicted_projected_ms,
            "predicted_projected_minus_direct_ms": (
                decision.predicted_projected_minus_direct_ms
            ),
            "uncertainty_guard_ms": decision.uncertainty_guard_ms,
            "within_uncertainty_guard": decision.within_uncertainty_guard,
        },
    }
