from __future__ import annotations

from azelficoast.core.costing import (
    CacheTierCost,
    ExecutionCostProfile,
    ExecutionFeatures,
    ExecutionPath,
    LocalityEvidence,
    estimate_cache_route,
)
from azelficoast.core.planning import (
    DEFAULT_DECISION_PLAN,
    LogicalOperator,
    LogicalPlan,
    OperatorImplementation,
    choose_operator_implementation,
    choose_physical_plan,
    explain_operator_plan,
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


def test_cache_route_estimate_uses_smoothed_locality_and_fallback_cost() -> None:
    exact = CacheTierCost(
        name="exact-cache",
        locality=LocalityEvidence(hits=8, misses=2),
        hit_cost_ms=0.03,
        miss_cost_ms=0.01,
    )
    projected = CacheTierCost(
        name="projected-delta",
        locality=LocalityEvidence(hits=3, misses=1),
        hit_cost_ms=0.08,
        miss_cost_ms=0.02,
    )

    estimate = estimate_cache_route((exact, projected), fallback_ms=2.0)

    assert estimate.tier_names == ("exact-cache", "projected-delta")
    assert 0.0 < estimate.remaining_miss_probability < 1.0
    assert estimate.expected_ms < 2.0


def test_operator_planner_compares_only_semantically_equivalent_candidates() -> None:
    candidates = (
        OperatorImplementation(
            operator=LogicalOperator.TRANSITION,
            name="fresh-showdown",
            semantic_signature="sha256:turn",
            predicted_ms=2.4,
            evidence={"kind": "fresh"},
        ),
        OperatorImplementation(
            operator=LogicalOperator.TRANSITION,
            name="exact-cache",
            semantic_signature="sha256:turn",
            predicted_ms=0.2,
            evidence={"kind": "cache"},
        ),
        OperatorImplementation(
            operator=LogicalOperator.TRANSITION,
            name="compiled-mechanics",
            semantic_signature="sha256:turn",
            predicted_ms=0.6,
            evidence={"kind": "verified-compiled"},
        ),
    )

    plan = choose_operator_implementation(candidates)
    explanation = explain_operator_plan(plan)

    assert plan.selected.name == "exact-cache"
    assert explanation["logical_operator"] == "transition"
    assert explanation["selected_implementation"] == "exact-cache"
    assert [row["name"] for row in explanation["candidates"]] == [
        "fresh-showdown",
        "exact-cache",
        "compiled-mechanics",
    ]


def test_operator_planner_keeps_jax_on_evaluate_operator() -> None:
    plan = choose_operator_implementation(
        (
            OperatorImplementation(
                operator=LogicalOperator.EVALUATE,
                name="jax-batch",
                semantic_signature="sha256:evaluator",
                predicted_ms=0.4,
                evidence={"batch": 64},
            ),
            OperatorImplementation(
                operator=LogicalOperator.EVALUATE,
                name="scalar-evaluator",
                semantic_signature="sha256:evaluator",
                predicted_ms=3.0,
                evidence={"batch": 1},
            ),
        )
    )

    assert plan.selected.name == "jax-batch"


def test_operator_planner_rejects_cross_operator_or_semantic_comparisons() -> None:
    transition = OperatorImplementation(
        operator=LogicalOperator.TRANSITION,
        name="fresh",
        semantic_signature="sha256:turn-a",
        predicted_ms=1.0,
        evidence={},
    )
    evaluate = OperatorImplementation(
        operator=LogicalOperator.EVALUATE,
        name="jax",
        semantic_signature="sha256:value",
        predicted_ms=0.5,
        evidence={},
    )

    try:
        choose_operator_implementation((transition, evaluate))
    except ValueError as error:
        assert "one logical operator" in str(error)
    else:
        raise AssertionError("cross-operator candidates must be rejected")
