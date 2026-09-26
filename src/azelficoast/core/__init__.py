"""Domain-neutral primitives for finite partial-information decision systems.

The core package intentionally contains no Pokémon, Showdown, or poke-env semantics.
Domain adapters may depend on this package; the core package must not depend on them.
"""

from azelficoast.core.contracts import (
    BeliefEvaluator,
    PartialInformationDomain,
    TransitionOracleProducer,
)
from azelficoast.core.costing import (
    ExecutionCostProfile,
    ExecutionDecision,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
)
from azelficoast.core.decision_relevance import (
    CERTIFICATE_SCHEMA,
    CERTIFICATE_SCHEMA_VERSION,
    DecisionRelevanceError,
    decision_relevance_quotient,
)
from azelficoast.core.program import (
    EXECUTION_SCHEMA,
    EXECUTION_SCHEMA_VERSION,
    PROGRAM_SET_SCHEMA,
    PROGRAM_SET_SCHEMA_VERSION,
    VERIFICATION_SCHEMA,
    VERIFICATION_SCHEMA_VERSION,
    TransitionProgramError,
    program_for_action,
)
from azelficoast.core.planning import (
    DEFAULT_DECISION_PLAN,
    PLAN_EXPLAIN_SCHEMA,
    PLAN_EXPLAIN_SCHEMA_VERSION,
    LogicalOperator,
    LogicalPlan,
    PhysicalPlan,
    choose_physical_plan,
    explain_physical_plan,
)
from azelficoast.core.search import (
    SEARCH_METHODS,
    SEARCH_SCHEMA,
    SEARCH_SCHEMA_VERSION,
    PartialInformationSearchError,
    search_transition_program,
)
from azelficoast.core.transition import (
    ORACLE_SCHEMA,
    ORACLE_SCHEMA_VERSION,
    canonical_json,
    sha256_json,
    transition_outcomes,
    validate_transition_oracle,
)

__all__ = [
    "DEFAULT_DECISION_PLAN",
    "ExecutionCostProfile",
    "ExecutionDecision",
    "ExecutionFeatures",
    "ExecutionPath",
    "LogicalOperator",
    "LogicalPlan",
    "PLAN_EXPLAIN_SCHEMA",
    "PLAN_EXPLAIN_SCHEMA_VERSION",
    "PhysicalPlan",
    "BeliefEvaluator",
    "CERTIFICATE_SCHEMA",
    "CERTIFICATE_SCHEMA_VERSION",
    "DecisionRelevanceError",
    "EXECUTION_SCHEMA",
    "EXECUTION_SCHEMA_VERSION",
    "ORACLE_SCHEMA",
    "ORACLE_SCHEMA_VERSION",
    "PROGRAM_SET_SCHEMA",
    "PROGRAM_SET_SCHEMA_VERSION",
    "PartialInformationDomain",
    "PartialInformationSearchError",
    "SEARCH_METHODS",
    "SEARCH_SCHEMA",
    "SEARCH_SCHEMA_VERSION",
    "TransitionOracleProducer",
    "TransitionProgramError",
    "VERIFICATION_SCHEMA",
    "VERIFICATION_SCHEMA_VERSION",
    "canonical_json",
    "choose_execution_path",
    "choose_physical_plan",
    "decision_relevance_quotient",
    "explain_physical_plan",
    "program_for_action",
    "search_transition_program",
    "sha256_json",
    "transition_outcomes",
    "validate_transition_oracle",
]
