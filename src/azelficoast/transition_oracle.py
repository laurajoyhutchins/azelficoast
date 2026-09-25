"""Compatibility façade for the domain-neutral finite transition core.

New generic consumers should import :mod:`azelficoast.core.transition`. Existing
Azelficoast modules keep this path so research artifacts do not churn merely because
the reusable machinery moved.
"""

from azelficoast.core.transition import (
    ORACLE_SCHEMA,
    ORACLE_SCHEMA_VERSION,
    canonical_json,
    sha256_json,
    transition_outcomes,
    validate_transition_oracle,
)

validate_oracle_core = validate_transition_oracle

__all__ = [
    "ORACLE_SCHEMA",
    "ORACLE_SCHEMA_VERSION",
    "canonical_json",
    "sha256_json",
    "transition_outcomes",
    "validate_oracle_core",
    "validate_transition_oracle",
]
