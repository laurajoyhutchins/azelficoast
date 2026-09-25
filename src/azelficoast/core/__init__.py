"""Domain-neutral primitives for finite partial-information decision systems.

The core package intentionally contains no Pokémon, Showdown, or poke-env semantics.
Domain adapters may depend on this package; the core package must not depend on them.
"""

from azelficoast.core.contracts import (
    BeliefEvaluator,
    PartialInformationDomain,
    TransitionOracleProducer,
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
    "BeliefEvaluator",
    "ORACLE_SCHEMA",
    "ORACLE_SCHEMA_VERSION",
    "PartialInformationDomain",
    "TransitionOracleProducer",
    "canonical_json",
    "sha256_json",
    "transition_outcomes",
    "validate_transition_oracle",
]
