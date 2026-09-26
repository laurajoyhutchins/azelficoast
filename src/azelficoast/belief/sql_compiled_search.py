"""Lower admitted decision SQL into the existing packed/JAX search data plane.

The SQL front end is intentionally narrower than SQLite itself. An admitted query may be
read-only SQL, but compiled execution is enabled only for the exact decision query whose
logical semantics have a reviewed lowering. Unknown SQL therefore remains parseable and
explainable without silently acquiring execution semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from azelficoast.belief.compiled_search import (
    PACKED_COMPILED_EXECUTION_STAGES,
    search_packed_compiled_transition_program,
)
from azelficoast.belief.packed_evaluator import PackedBeliefEvaluatorSpec
from azelficoast.belief.showdown_packing import ShowdownVocabulary
from azelficoast.core.planning import DEFAULT_DECISION_PLAN, LogicalOperator, LogicalPlan
from azelficoast.core.sql import (
    DEFAULT_DECISION_SQL,
    PreparedDecisionQuery,
    prepare_decision_query,
)

SQL_PACKED_PLAN_SCHEMA = "azelficoast.sql-packed-decision-plan"
SQL_PACKED_PLAN_SCHEMA_VERSION = 1
SQL_PACKED_SEARCH_SCHEMA = "azelficoast.sql-packed-partial-information-search"
SQL_PACKED_SEARCH_SCHEMA_VERSION = 1


class SQLPackedLoweringError(ValueError):
    """Raised when admitted SQL has no exact packed/JAX lowering."""


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

    sql_sha256: str
    logical: LogicalPlan
    bindings: tuple[SQLPackedBinding, ...]
    physical_execution_stages: tuple[str, ...]
    numeric_backend: str
    plan_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": SQL_PACKED_PLAN_SCHEMA,
            "schema_version": SQL_PACKED_PLAN_SCHEMA_VERSION,
            "sql_sha256": self.sql_sha256,
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


_DEFAULT_SQL_SHA256 = "sha256:" + hashlib.sha256(
    DEFAULT_DECISION_SQL.encode("utf-8")
).hexdigest()

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

# This is the optimized physical order, not the source relational order. Topology
# compilation is independent of posterior weights, so the current packed path hoists it
# ahead of posterior packing and then performs the numeric JAX stages.
_SQL_PACKED_EXECUTION_STAGES = (
    "compile_search_topology",
    "pack_joint_posterior",
    "transport_posterior_mass",
    "predict_packed_shared_world_values",
    "reduce_compiled_root_values",
)


def _plan_digest(material: Mapping[str, Any]) -> str:
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def compile_packed_sql_decision_query(
    query: PreparedDecisionQuery,
) -> SQLPackedDecisionPlan:
    """Lower the reviewed SQL decision query into the packed/JAX physical path."""

    if query.sql_sha256 != _DEFAULT_SQL_SHA256 or query.sql != DEFAULT_DECISION_SQL:
        raise SQLPackedLoweringError(
            "admitted SQL has no reviewed packed/JAX lowering"
        )
    if query.logical != DEFAULT_DECISION_PLAN:
        raise SQLPackedLoweringError(
            "decision SQL logical plan differs from the reviewed lowering"
        )
    if tuple(binding.operator for binding in _PACKED_BINDINGS) != query.logical.operators:
        raise SQLPackedLoweringError(
            "packed SQL bindings do not cover the logical plan exactly"
        )

    material = {
        "schema": SQL_PACKED_PLAN_SCHEMA,
        "schema_version": SQL_PACKED_PLAN_SCHEMA_VERSION,
        "sql_sha256": query.sql_sha256,
        "logical_operators": [
            operator.value for operator in query.logical.operators
        ],
        "bindings": [
            {
                "logical_operator": binding.operator.value,
                "implementation": binding.implementation,
                "fused_group": binding.fused_group,
            }
            for binding in _PACKED_BINDINGS
        ],
        "physical_execution_stages": list(_SQL_PACKED_EXECUTION_STAGES),
        "numeric_backend": "jax-shared-packed-worlds",
    }
    return SQLPackedDecisionPlan(
        sql_sha256=query.sql_sha256,
        logical=query.logical,
        bindings=_PACKED_BINDINGS,
        physical_execution_stages=_SQL_PACKED_EXECUTION_STAGES,
        numeric_backend="jax-shared-packed-worlds",
        plan_sha256=_plan_digest(material),
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
) -> dict[str, Any]:
    """Execute reviewed decision SQL through the existing packed/JAX machinery."""

    prepared = prepare_decision_query(sql)
    plan = compile_packed_sql_decision_query(prepared)
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

    return {
        **result,
        "schema": SQL_PACKED_SEARCH_SCHEMA,
        "schema_version": SQL_PACKED_SEARCH_SCHEMA_VERSION,
        "packed_search_schema": result["schema"],
        "packed_search_schema_version": result["schema_version"],
        "sql_query_sha256": prepared.sql_sha256,
        "sql_physical_plan": plan.as_record(),
        "sql_execution_matches_hand_built_plan": (
            hand_built_stages == PACKED_COMPILED_EXECUTION_STAGES
        ),
    }
