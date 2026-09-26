"""Lower admitted decision SQL into the existing packed/JAX search data plane.

The SQL front end is intentionally narrower than SQLite itself. An admitted query may be
read-only SQL, but compiled execution is enabled only when the core recognizer assigns
the reviewed decision semantic identity. Equivalent SQL syntax may therefore share one
physical plan and statistics history, while unknown SQL remains parseable without
acquiring execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from azelficoast.belief.compiled_search import (
    PACKED_BOUNDED_EXECUTION_STAGES,
    PACKED_COMPILED_EXECUTION_STAGES,
    choose_packed_compiled_action_bounded,
    search_packed_compiled_transition_program,
)
from azelficoast.belief.packed_evaluator import PackedBeliefEvaluatorSpec
from azelficoast.belief.showdown_packing import ShowdownVocabulary
from azelficoast.belief.statistics import (
    PosteriorCorrelationProfile,
    posterior_correlation_profile,
)
from azelficoast.core.compiled_search import (
    compile_search_topology,
    estimate_search_cardinality_lower_bound,
)
from azelficoast.core.planning import DEFAULT_DECISION_PLAN, LogicalOperator, LogicalPlan
from azelficoast.core.statistics import CardinalityEstimate, PlannerStatistics
from azelficoast.core.sql import (
    BEST_ACTION_SQL,
    DECISION_BEST_ACTION_QUERY,
    DECISION_BEST_ACTION_SEMANTIC_ID,
    DECISION_EXPECTED_VALUE_QUERY,
    DECISION_QUERY_SEMANTIC_ID,
    DEFAULT_DECISION_SQL,
    PreparedSQLQuery,
    explain_sql_query,
    prepare_best_action_query,
    prepare_decision_query,
)

SQL_PACKED_PLAN_SCHEMA = "azelficoast.sql-packed-decision-plan"
SQL_PACKED_PLAN_SCHEMA_VERSION = 4
SQL_PACKED_SEARCH_SCHEMA = "azelficoast.sql-packed-partial-information-search"
SQL_PACKED_SEARCH_SCHEMA_VERSION = 3
SQL_PACKED_BEST_ACTION_SCHEMA = "azelficoast.sql-packed-best-action"
SQL_PACKED_BEST_ACTION_SCHEMA_VERSION = 1
SQL_AZELFICOAST_EXPLAIN_SCHEMA = "azelficoast.sql-optimizer-explain"
SQL_AZELFICOAST_EXPLAIN_SCHEMA_VERSION = 1


class SQLPackedLoweringError(ValueError):
    """Raised when admitted SQL has no reviewed packed/JAX lowering."""


@dataclass(frozen=True, slots=True)
class SQLPackedBinding:
    """Bind one logical SQL operator to reviewed physical machinery."""

    operator: LogicalOperator
    implementation: str
    fused_group: str

    def __post_init__(self) -> None:
        if not self.implementation:
            raise SQLPackedLoweringError("SQL physical implementation must be non-empty")
        if not self.fused_group:
            raise SQLPackedLoweringError("SQL fused group must be non-empty")


@dataclass(frozen=True, slots=True)
class SQLPackedDecisionPlan:
    """Exact physical lowering for one admitted decision query."""

    query_class: str
    result_contract: str
    sql_sha256: str
    semantic_identity: str
    equivalence_rule: str
    equivalence_scope: str
    logical: LogicalPlan
    bindings: tuple[SQLPackedBinding, ...]
    physical_execution_stages: tuple[str, ...]
    numeric_backend: str
    plan_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": SQL_PACKED_PLAN_SCHEMA,
            "schema_version": SQL_PACKED_PLAN_SCHEMA_VERSION,
            "query_class": self.query_class,
            "result_contract": self.result_contract,
            "sql_sha256": self.sql_sha256,
            "semantic_identity": self.semantic_identity,
            "equivalence_rule": self.equivalence_rule,
            "equivalence_scope": self.equivalence_scope,
            "logical_operators": [
                operator.value for operator in self.logical.operators
            ],
            "bindings": [
                {
                    "logical_operator": binding.operator.value,
                    "implementation": binding.implementation,
                    "fused_group": binding.fused_group,
                }
                for binding in self.bindings
            ],
            "physical_execution_stages": list(self.physical_execution_stages),
            "numeric_backend": self.numeric_backend,
            "plan_sha256": self.plan_sha256,
        }


_PACKED_BINDINGS = (
    SQLPackedBinding(
        LogicalOperator.SCAN,
        "pack_joint_posterior",
        "posterior-pack",
    ),
    SQLPackedBinding(
        LogicalOperator.FILTER,
        "pack_joint_posterior",
        "posterior-pack",
    ),
    SQLPackedBinding(
        LogicalOperator.PROJECT,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.PARTITION,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.TRANSITION,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.OBSERVE,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.UPDATE_BELIEF,
        "transport_posterior_mass",
        "belief-transport",
    ),
    SQLPackedBinding(
        LogicalOperator.EVALUATE,
        "predict_packed_shared_world_values",
        "packed-evaluator",
    ),
    SQLPackedBinding(
        LogicalOperator.AGGREGATE,
        "reduce_compiled_root_values",
        "root-reduction",
    ),
)

_BOUNDED_PACKED_BINDINGS = (
    SQLPackedBinding(
        LogicalOperator.SCAN,
        "pack_joint_posterior",
        "posterior-pack",
    ),
    SQLPackedBinding(
        LogicalOperator.FILTER,
        "pack_joint_posterior",
        "posterior-pack",
    ),
    SQLPackedBinding(
        LogicalOperator.PROJECT,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.PARTITION,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.TRANSITION,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.OBSERVE,
        "compile_search_topology",
        "authorized-topology",
    ),
    SQLPackedBinding(
        LogicalOperator.UPDATE_BELIEF,
        "transport_posterior_mass",
        "belief-transport",
    ),
    SQLPackedBinding(
        LogicalOperator.EVALUATE,
        "predict_packed_shared_world_values",
        "bounded-evaluate-reduce",
    ),
    SQLPackedBinding(
        LogicalOperator.AGGREGATE,
        "choose_bounded_action",
        "bounded-evaluate-reduce",
    ),
)


# This is the optimized physical order, not the source relational order. Topology
# compilation is independent of posterior weights, so the current packed path hoists it
# ahead of posterior packing and then performs the numeric JAX stages.
_SQL_PACKED_EXECUTION_STAGES = PACKED_COMPILED_EXECUTION_STAGES
_SQL_BOUNDED_EXECUTION_STAGES = PACKED_BOUNDED_EXECUTION_STAGES


def _plan_digest(material: Mapping[str, Any]) -> str:
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def compile_packed_sql_decision_query(
    query: PreparedSQLQuery,
) -> SQLPackedDecisionPlan:
    """Lower one reviewed decision SQL semantic class into packed/JAX machinery."""

    if query.equivalence_rule is None or query.equivalence_scope is None:
        raise SQLPackedLoweringError(
            "admitted SQL lacks reviewed equivalence evidence"
        )
    if query.logical != DEFAULT_DECISION_PLAN:
        raise SQLPackedLoweringError(
            "decision SQL logical plan differs from the reviewed lowering"
        )

    if query.semantic_identity == DECISION_QUERY_SEMANTIC_ID:
        if query.query_class != DECISION_EXPECTED_VALUE_QUERY:
            raise SQLPackedLoweringError(
                "expected-value semantic identity has the wrong query class"
            )
        bindings = _PACKED_BINDINGS
        stages = _SQL_PACKED_EXECUTION_STAGES
        numeric_backend = "jax-shared-packed-worlds"
        result_contract = "all-action-exact-values"
    elif query.semantic_identity == DECISION_BEST_ACTION_SEMANTIC_ID:
        if query.query_class != DECISION_BEST_ACTION_QUERY:
            raise SQLPackedLoweringError(
                "best-action semantic identity has the wrong query class"
            )
        bindings = _BOUNDED_PACKED_BINDINGS
        stages = _SQL_BOUNDED_EXECUTION_STAGES
        numeric_backend = "jax-shared-packed-worlds-bounded"
        result_contract = "winner-only-exact-action-and-value"
    else:
        raise SQLPackedLoweringError(
            "admitted SQL has no reviewed packed/JAX semantic identity"
        )

    if tuple(binding.operator for binding in bindings) != query.logical.operators:
        raise SQLPackedLoweringError(
            "packed SQL bindings do not cover the logical plan exactly"
        )

    material = {
        "schema": SQL_PACKED_PLAN_SCHEMA,
        "schema_version": SQL_PACKED_PLAN_SCHEMA_VERSION,
        "query_class": query.query_class,
        "result_contract": result_contract,
        "semantic_identity": query.semantic_identity,
        "logical_operators": [
            operator.value for operator in query.logical.operators
        ],
        "bindings": [
            {
                "logical_operator": binding.operator.value,
                "implementation": binding.implementation,
                "fused_group": binding.fused_group,
            }
            for binding in bindings
        ],
        "physical_execution_stages": list(stages),
        "numeric_backend": numeric_backend,
    }
    return SQLPackedDecisionPlan(
        query_class=query.query_class,
        result_contract=result_contract,
        sql_sha256=query.sql_sha256,
        semantic_identity=query.semantic_identity,
        equivalence_rule=query.equivalence_rule,
        equivalence_scope=query.equivalence_scope,
        logical=query.logical,
        bindings=bindings,
        physical_execution_stages=stages,
        numeric_backend=numeric_backend,
        plan_sha256=_plan_digest(material),
    )

def _optimizer_groups(plan: SQLPackedDecisionPlan) -> list[dict[str, Any]]:
    """Collapse reviewed operator bindings into explicit physical fusion groups."""

    groups: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for binding in plan.bindings:
        group = by_name.get(binding.fused_group)
        if group is None:
            group = {
                "name": binding.fused_group,
                "logical_operators": [],
                "physical_implementations": [],
            }
            by_name[binding.fused_group] = group
            groups.append(group)
        group["logical_operators"].append(binding.operator.value)
        if binding.implementation not in group["physical_implementations"]:
            group["physical_implementations"].append(binding.implementation)
    return groups


def _explain_prepared_sql_packed_transition_program(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    prepared: PreparedSQLQuery,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
    planner_statistics: PlannerStatistics | None = None,
) -> dict[str, Any]:
    """Explain one reviewed SQL/JAX plan without executing the evaluator.

    This is the programmatic equivalent of EXPLAIN AZELFICOAST. It parses and
    recognizes the SQL, validates the transition topology, computes cheap and realized
    cardinalities, and reports the reviewed physical bindings. It does not run JAX,
    invoke the learned evaluator, choose an action, or update planner statistics.
    """

    plan = compile_packed_sql_decision_query(prepared)

    raw_worlds = posterior.get("worlds")
    raw_actions = program_set.get("legal_actions")
    if not isinstance(raw_worlds, list) or not isinstance(raw_actions, list):
        raise SQLPackedLoweringError(
            "SQL planning requires explicit posterior worlds and legal actions"
        )

    lower_bound = estimate_search_cardinality_lower_bound(
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )

    filter_signature = (
        f"{plan.semantic_identity}:filter:hidden_worlds:active-positive:"
        f"{posterior.get('schema', 'unknown')}"
    )
    correlation_profile: PosteriorCorrelationProfile | None = None
    correlation_signature = "unprofiled"
    if planner_statistics is not None:
        correlation_profile = posterior_correlation_profile(posterior)
        correlation_signature = correlation_profile.signature

    partition_signature = (
        f"{plan.semantic_identity}:partition:{method}:"
        f"{program_set.get('schema', 'unknown')}:"
        f"correlation:{correlation_signature}"
    )

    filter_estimate: CardinalityEstimate | None = None
    partition_estimate: CardinalityEstimate | None = None
    if planner_statistics is not None:
        filter_estimate = planner_statistics.estimate(
            operator=LogicalOperator.FILTER,
            signature=filter_signature,
            input_rows=len(raw_worlds),
        )
        partition_estimate = planner_statistics.estimate(
            operator=LogicalOperator.PARTITION,
            signature=partition_signature,
            input_rows=filter_estimate.estimated_output_rows * len(raw_actions),
        )

    return {
        "schema": SQL_AZELFICOAST_EXPLAIN_SCHEMA,
        "schema_version": SQL_AZELFICOAST_EXPLAIN_SCHEMA_VERSION,
        "sql": explain_sql_query(prepared),
        "semantic": {
            "logical_operators": [
                operator.value for operator in plan.logical.operators
            ],
            "transition_program_digest": topology.program_digest,
            "compiled_topology_digest": topology.topology_digest,
            "method": method,
        },
        "optimizer": {
            "groups": _optimizer_groups(plan),
            "outcome_world_join": topology.outcome_world_join_plan.as_record(),
            "cardinality": {
                "lower_bound": lower_bound.as_record(),
                "realized": {
                    "actions": topology.action_count,
                    "worlds": topology.world_count,
                    "classes": topology.class_count,
                    "chance_edges": topology.edge_count,
                    "raw_chance_edges": (
                        topology.outcome_world_join_plan.raw_join_rows
                    ),
                    "observations": len(topology.observation_keys),
                    "successor_states": len(topology.successor_states),
                    "leaves": topology.leaf_count,
                    "dense_leaf_world_cells": (
                        topology.leaf_count * topology.world_count
                    ),
                },
                "forecast": {
                    "filter": (
                        filter_estimate.as_record()
                        if filter_estimate is not None
                        else None
                    ),
                    "partition": (
                        partition_estimate.as_record()
                        if partition_estimate is not None
                        else None
                    ),
                },
            },
            "extended_statistics": (
                correlation_profile.as_record()
                if correlation_profile is not None
                else None
            ),
        },
        "physical": plan.as_record(),
        "authority": {
            "mechanics": "verified-transition-program-outside-sql",
            "information_sets": "python-validated-compiled-topology",
            "posterior": "caller-supplied-admitted-posterior",
            "sql": "read-only-query-semantics",
            "numeric_execution": "not-executed",
        },
        "execution": {
            "performed": False,
            "evaluator_calls": 0,
            "action_selected": False,
            "planner_statistics_mutated": False,
        },
    }



def explain_sql_packed_transition_program(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    sql: str = DEFAULT_DECISION_SQL,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
    planner_statistics: PlannerStatistics | None = None,
) -> dict[str, Any]:
    """Explain the full expected-value SQL lowering without evaluator execution."""

    return _explain_prepared_sql_packed_transition_program(
        prepared=prepare_decision_query(sql),
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
        planner_statistics=planner_statistics,
    )


def explain_sql_packed_best_action(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    sql: str = BEST_ACTION_SQL,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
    planner_statistics: PlannerStatistics | None = None,
) -> dict[str, Any]:
    """Explain the winner-only bounded SQL lowering without evaluator execution."""

    return _explain_prepared_sql_packed_transition_program(
        prepared=prepare_best_action_query(sql),
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
        planner_statistics=planner_statistics,
    )


def search_sql_packed_transition_program(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    vocabulary: ShowdownVocabulary,
    evaluator_spec: PackedBeliefEvaluatorSpec,
    evaluator_params: Mapping[str, Any],
    sql: str = DEFAULT_DECISION_SQL,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
    planner_statistics: PlannerStatistics | None = None,
) -> dict[str, Any]:
    """Execute reviewed decision SQL through the existing packed/JAX machinery."""

    prepared = prepare_decision_query(sql)
    plan = compile_packed_sql_decision_query(prepared)

    raw_worlds = posterior.get("worlds")
    raw_actions = program_set.get("legal_actions")
    if not isinstance(raw_worlds, list) or not isinstance(raw_actions, list):
        raise SQLPackedLoweringError(
            "SQL planning requires explicit posterior worlds and legal actions"
        )
    scan_rows = len(raw_worlds)
    action_rows = len(raw_actions)

    filter_signature = (
        f"{plan.semantic_identity}:filter:hidden_worlds:active-positive:"
        f"{posterior.get('schema', 'unknown')}"
    )
    correlation_profile: PosteriorCorrelationProfile | None = None
    correlation_signature = "unprofiled"
    if planner_statistics is not None:
        correlation_profile = posterior_correlation_profile(posterior)
        correlation_signature = correlation_profile.signature

    partition_signature = (
        f"{plan.semantic_identity}:partition:{method}:"
        f"{program_set.get('schema', 'unknown')}:"
        f"correlation:{correlation_signature}"
    )

    filter_estimate: CardinalityEstimate | None = None
    partition_estimate: CardinalityEstimate | None = None
    if planner_statistics is not None:
        filter_estimate = planner_statistics.estimate(
            operator=LogicalOperator.FILTER,
            signature=filter_signature,
            input_rows=scan_rows,
        )
        partition_estimate = planner_statistics.estimate(
            operator=LogicalOperator.PARTITION,
            signature=partition_signature,
            input_rows=filter_estimate.estimated_output_rows * action_rows,
        )

    result = search_packed_compiled_transition_program(
        program_set=program_set,
        posterior=posterior,
        method=method,
        vocabulary=vocabulary,
        evaluator_spec=evaluator_spec,
        evaluator_params=evaluator_params,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )

    hand_built_stages = tuple(result.get("physical_execution_stages", ()))
    if hand_built_stages != plan.physical_execution_stages:
        raise SQLPackedLoweringError(
            "SQL-generated physical plan drifted from the packed/JAX execution path"
        )
    if result.get("numeric_backend") != plan.numeric_backend:
        raise SQLPackedLoweringError(
            "SQL-generated numeric backend drifted from packed/JAX execution"
        )

    compiled_shape = result.get("compiled_shape")
    if not isinstance(compiled_shape, Mapping):
        raise SQLPackedLoweringError("packed search did not report compiled cardinalities")
    actual_filter_rows = compiled_shape.get("worlds")
    actual_partition_groups = compiled_shape.get("classes")
    actual_action_rows = compiled_shape.get("actions")
    if (
        not isinstance(actual_filter_rows, int)
        or isinstance(actual_filter_rows, bool)
        or actual_filter_rows < 0
        or not isinstance(actual_partition_groups, int)
        or isinstance(actual_partition_groups, bool)
        or actual_partition_groups < 0
        or not isinstance(actual_action_rows, int)
        or isinstance(actual_action_rows, bool)
        or actual_action_rows < 0
    ):
        raise SQLPackedLoweringError("packed search reported invalid cardinalities")

    if planner_statistics is not None:
        planner_statistics.observe(
            operator=LogicalOperator.FILTER,
            signature=filter_signature,
            input_rows=scan_rows,
            output_rows=actual_filter_rows,
        )
        planner_statistics.observe(
            operator=LogicalOperator.PARTITION,
            signature=partition_signature,
            input_rows=actual_filter_rows * actual_action_rows,
            output_rows=actual_partition_groups,
        )

    forecast = {
        "filter": (
            filter_estimate.as_record() if filter_estimate is not None else None
        ),
        "partition": (
            partition_estimate.as_record()
            if partition_estimate is not None
            else None
        ),
    }
    observed = {
        "filter": {
            "input_rows": scan_rows,
            "output_rows": actual_filter_rows,
        },
        "partition": {
            "input_rows": actual_filter_rows * actual_action_rows,
            "output_rows": actual_partition_groups,
        },
    }
    error = {
        "filter_rows": (
            actual_filter_rows - filter_estimate.estimated_output_rows
            if filter_estimate is not None
            else None
        ),
        "partition_groups": (
            actual_partition_groups - partition_estimate.estimated_output_rows
            if partition_estimate is not None
            else None
        ),
    }

    return {
        **result,
        "schema": SQL_PACKED_SEARCH_SCHEMA,
        "schema_version": SQL_PACKED_SEARCH_SCHEMA_VERSION,
        "packed_search_schema": result["schema"],
        "packed_search_schema_version": result["schema_version"],
        "sql_query_sha256": prepared.sql_sha256,
        "sql_semantic_identity": plan.semantic_identity,
        "sql_equivalence_rule": plan.equivalence_rule,
        "sql_equivalence_scope": plan.equivalence_scope,
        "sql_physical_plan": plan.as_record(),
        "sql_cardinality_forecast": forecast,
        "sql_extended_statistics": (
            correlation_profile.as_record()
            if correlation_profile is not None
            else None
        ),
        "sql_cardinality_observed": observed,
        "sql_cardinality_error": error,
        "sql_execution_matches_hand_built_plan": (
            hand_built_stages == PACKED_COMPILED_EXECUTION_STAGES
        ),
    }



def search_sql_packed_best_action(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    vocabulary: ShowdownVocabulary,
    evaluator_spec: PackedBeliefEvaluatorSpec,
    evaluator_params: Mapping[str, Any],
    sql: str = BEST_ACTION_SQL,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Execute winner-only SQL through exact bounded packed/JAX pruning."""

    prepared = prepare_best_action_query(sql)
    plan = compile_packed_sql_decision_query(prepared)
    result = choose_packed_compiled_action_bounded(
        program_set=program_set,
        posterior=posterior,
        method=method,
        vocabulary=vocabulary,
        evaluator_spec=evaluator_spec,
        evaluator_params=evaluator_params,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
        batch_size=batch_size,
    )

    hand_built_stages = tuple(result.get("physical_execution_stages", ()))
    if hand_built_stages != plan.physical_execution_stages:
        raise SQLPackedLoweringError(
            "winner-only SQL physical plan drifted from bounded packed execution"
        )
    if result.get("numeric_backend") != plan.numeric_backend:
        raise SQLPackedLoweringError(
            "winner-only SQL numeric backend drifted from bounded packed execution"
        )
    if "root_values" in result:
        raise SQLPackedLoweringError(
            "winner-only execution must not expose fabricated exact loser values"
        )

    return {
        **result,
        "schema": SQL_PACKED_BEST_ACTION_SCHEMA,
        "schema_version": SQL_PACKED_BEST_ACTION_SCHEMA_VERSION,
        "bounded_search_schema": result["schema"],
        "bounded_search_schema_version": result["schema_version"],
        "sql_query_sha256": prepared.sql_sha256,
        "sql_semantic_identity": plan.semantic_identity,
        "sql_equivalence_rule": plan.equivalence_rule,
        "sql_equivalence_scope": plan.equivalence_scope,
        "sql_physical_plan": plan.as_record(),
        "sql_result": {
            "action_id": result["chosen_action"],
            "expected_value": result["chosen_value"],
        },
        "sql_execution_matches_hand_built_plan": True,
    }
