from __future__ import annotations

from azelficoast.core.costing import ExecutionCostProfile, ExecutionFeatures, ExecutionPath
from azelficoast.core.planning import (
    DEFAULT_DECISION_PLAN,
    LogicalOperator,
    LogicalPlan,
    choose_physical_plan,
    explain_physical_plan,
)


def _profile() -> ExecutionCostProfile:
    return ExecutionCostProfile(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:effect",
        direct_intercept_ms=0.1,
        direct_per_world_ms=0.001,
        direct_per_world_squared_ms=0.0,
        projected_intercept_ms=0.2,
        projected_per_canonical_class_ms=0.001,
        projected_per_execution_class_ms=0.001,
        uncertainty_guard_ms=0.05,
        calibrated_max_logical_world_count=100_000,
        calibrated_max_canonical_classes=10_000,
        calibrated_max_projected_classes=10_000,
    )


def _features() -> ExecutionFeatures:
    return ExecutionFeatures(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:effect",
        logical_world_count=10_000,
        active_canonical_classes=128,
        active_projected_classes=16,
    )


def test_default_logical_plan_exposes_relational_optimization_frontier() -> None:
    assert DEFAULT_DECISION_PLAN.operators == (
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


def test_physical_plan_wraps_existing_exact_cost_choice_without_changing_it() -> None:
    plan = choose_physical_plan(_profile(), _features())

    assert plan.decision.path is ExecutionPath.PROJECTED
    assert plan.features == _features()
    assert plan.logical == DEFAULT_DECISION_PLAN


def test_explain_is_deterministic_machine_readable_planner_evidence() -> None:
    explanation = explain_physical_plan(
        choose_physical_plan(_profile(), _features())
    )

    assert explanation["schema"] == "azelficoast.core.execution-plan-explain"
    assert explanation["schema_version"] == 1
    assert explanation["logical_operators"][0:4] == [
        "scan",
        "filter",
        "project",
        "partition",
    ]
    assert explanation["features"]["logical_world_count"] == 10_000
    assert explanation["physical"]["selected_path"] == "projected"
    assert explanation == explain_physical_plan(
        choose_physical_plan(_profile(), _features())
    )


def test_custom_logical_plan_does_not_change_physical_cost_semantics() -> None:
    logical = LogicalPlan(
        (
            LogicalOperator.SCAN,
            LogicalOperator.PROJECT,
            LogicalOperator.PARTITION,
            LogicalOperator.EVALUATE,
            LogicalOperator.AGGREGATE,
        )
    )

    planned = choose_physical_plan(_profile(), _features(), logical=logical)

    assert planned.logical is logical
    assert planned.decision.path is ExecutionPath.PROJECTED
