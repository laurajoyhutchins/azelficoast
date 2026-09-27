"""Physical planning for compiled partial-information search.

This module owns cost and cardinality decisions.  It may inspect already-authorized
search structure, but it does not compile execution topology, transport posterior
mass, evaluate leaves, or choose actions.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from azelficoast.core.search import (
    PartialInformationSearchError,
    SEARCH_METHODS,
    _normalized_inputs,
    _validated_classes,
)
from azelficoast.core.transition import canonical_json

CARDINALITY_PLAN_SCHEMA = "azelficoast.core.compiled-search-cardinality-plan"
CARDINALITY_PLAN_SCHEMA_VERSION = 1
SEARCH_PATH_COMPILED = "compiled-jax"
SEARCH_PATH_PYTHON = "python-frontier"
OUTCOME_JOIN_PLAN_SCHEMA = "azelficoast.core.outcome-world-join-plan"
OUTCOME_JOIN_PLAN_SCHEMA_VERSION = 1
JOIN_ORDER_EXPAND_FIRST = "expand-outcomes-before-world-join"
JOIN_ORDER_AGGREGATE_FIRST = "aggregate-outcomes-before-world-join"


class CompiledSearchError(ValueError):
    """Raised when an authorized search topology cannot be compiled or transported."""


@dataclass(frozen=True, slots=True)
class OutcomeWorldJoinPlan:
    """Costed ordering for outcome aggregation versus the class-member join."""

    selected_order: str
    raw_outcome_rows: int
    grouped_outcome_rows: int
    raw_join_rows: int
    grouped_join_rows: int
    expand_first_work_units: int
    aggregate_first_work_units: int

    def __post_init__(self) -> None:
        if self.selected_order not in {
            JOIN_ORDER_EXPAND_FIRST,
            JOIN_ORDER_AGGREGATE_FIRST,
        }:
            raise ValueError("unknown outcome/world join order")
        for value in (
            self.raw_outcome_rows,
            self.grouped_outcome_rows,
            self.raw_join_rows,
            self.grouped_join_rows,
            self.expand_first_work_units,
            self.aggregate_first_work_units,
        ):
            if value <= 0:
                raise ValueError("outcome/world join cardinalities must be positive")
        if self.grouped_outcome_rows > self.raw_outcome_rows:
            raise ValueError("outcome grouping cannot increase outcome rows")
        if self.grouped_join_rows > self.raw_join_rows:
            raise ValueError("outcome grouping cannot increase join rows")

    @property
    def saved_join_rows(self) -> int:
        return self.raw_join_rows - self.grouped_join_rows

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": OUTCOME_JOIN_PLAN_SCHEMA,
            "schema_version": OUTCOME_JOIN_PLAN_SCHEMA_VERSION,
            "selected_order": self.selected_order,
            "raw_outcome_rows": self.raw_outcome_rows,
            "grouped_outcome_rows": self.grouped_outcome_rows,
            "raw_join_rows": self.raw_join_rows,
            "grouped_join_rows": self.grouped_join_rows,
            "expand_first_work_units": self.expand_first_work_units,
            "aggregate_first_work_units": self.aggregate_first_work_units,
            "saved_join_rows": self.saved_join_rows,
            "equivalence_rule": (
                "sum probabilities only for outcomes with identical public "
                "observation, public successor, and successor legal-action set"
            ),
        }


def _search_outcome_key(outcome: Mapping[str, Any]) -> str:
    legal = outcome.get("legal_actions")
    if not isinstance(legal, list):
        raise CompiledSearchError("transition outcome lacks legal actions")
    return canonical_json(
        {
            "observation": outcome.get("observation"),
            "successor": outcome.get("successor"),
            "legal_actions": sorted(set(str(action) for action in legal)),
        }
    )


def _group_search_equivalent_outcomes(
    outcomes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        key = _search_outcome_key(outcome)
        existing = grouped.get(key)
        if existing is None:
            legal = outcome.get("legal_actions")
            assert isinstance(legal, list)
            grouped[key] = {
                "probability": float(outcome["probability"]),
                "observation": copy.deepcopy(outcome.get("observation")),
                "successor": copy.deepcopy(dict(outcome["successor"])),
                "legal_actions": sorted(set(str(action) for action in legal)),
            }
        else:
            existing["probability"] = math.fsum(
                (
                    float(existing["probability"]),
                    float(outcome["probability"]),
                )
            )
    return tuple(grouped[key] for key in sorted(grouped))


def _plan_outcome_world_join(
    prepared_classes: Sequence[
        tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]
    ],
) -> OutcomeWorldJoinPlan:
    raw_outcome_rows = 0
    grouped_outcome_rows = 0
    raw_join_rows = 0
    grouped_join_rows = 0

    for row, grouped in prepared_classes:
        members = row["member_world_ids"]
        outcomes = row["outcomes"]
        member_count = len(members)
        raw_count = len(outcomes)
        grouped_count = len(grouped)
        raw_outcome_rows += raw_count
        grouped_outcome_rows += grouped_count
        raw_join_rows += member_count * raw_count
        grouped_join_rows += member_count * grouped_count

    expand_first_work_units = raw_join_rows
    aggregate_first_work_units = raw_outcome_rows + grouped_join_rows
    selected_order = (
        JOIN_ORDER_AGGREGATE_FIRST
        if aggregate_first_work_units < expand_first_work_units
        else JOIN_ORDER_EXPAND_FIRST
    )
    return OutcomeWorldJoinPlan(
        selected_order=selected_order,
        raw_outcome_rows=raw_outcome_rows,
        grouped_outcome_rows=grouped_outcome_rows,
        raw_join_rows=raw_join_rows,
        grouped_join_rows=grouped_join_rows,
        expand_first_work_units=expand_first_work_units,
        aggregate_first_work_units=aggregate_first_work_units,
    )


@dataclass(frozen=True, slots=True)
class SearchCardinality:
    """Structural search size used by the physical planner."""

    world_count: int
    class_count: int
    chance_edge_count: int
    leaf_count: int

    def __post_init__(self) -> None:
        for value in (
            self.world_count,
            self.class_count,
            self.chance_edge_count,
            self.leaf_count,
        ):
            if value <= 0:
                raise ValueError("search cardinalities must be positive")

    @property
    def dense_leaf_world_cells(self) -> int:
        return self.world_count * self.leaf_count

    def as_record(self) -> dict[str, int]:
        return {
            "worlds": self.world_count,
            "classes": self.class_count,
            "chance_edges": self.chance_edge_count,
            "leaves": self.leaf_count,
            "dense_leaf_world_cells": self.dense_leaf_world_cells,
        }


@dataclass(frozen=True, slots=True)
class CardinalityEnvelope:
    """Authorized structural domain for one dense compiled-search treatment."""

    max_world_count: int
    max_class_count: int
    max_chance_edge_count: int
    max_leaf_count: int
    max_dense_leaf_world_cells: int

    def __post_init__(self) -> None:
        for value in (
            self.max_world_count,
            self.max_class_count,
            self.max_chance_edge_count,
            self.max_leaf_count,
            self.max_dense_leaf_world_cells,
        ):
            if value <= 0:
                raise ValueError("cardinality envelope bounds must be positive")

    def violations(self, cardinality: SearchCardinality) -> tuple[str, ...]:
        rows = (
            ("worlds", cardinality.world_count, self.max_world_count),
            ("classes", cardinality.class_count, self.max_class_count),
            (
                "chance_edges",
                cardinality.chance_edge_count,
                self.max_chance_edge_count,
            ),
            ("leaves", cardinality.leaf_count, self.max_leaf_count),
            (
                "dense_leaf_world_cells",
                cardinality.dense_leaf_world_cells,
                self.max_dense_leaf_world_cells,
            ),
        )
        return tuple(name for name, actual, maximum in rows if actual > maximum)

    def as_record(self) -> dict[str, int]:
        return {
            "max_worlds": self.max_world_count,
            "max_classes": self.max_class_count,
            "max_chance_edges": self.max_chance_edge_count,
            "max_leaves": self.max_leaf_count,
            "max_dense_leaf_world_cells": self.max_dense_leaf_world_cells,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveCardinalityPlan:
    """Initial and revised physical choice around a measured cardinality boundary."""

    lower_bound: SearchCardinality
    envelope: CardinalityEnvelope
    initial_path: str
    final_path: str
    observed: SearchCardinality | None
    violations: tuple[str, ...]

    @property
    def replanned(self) -> bool:
        return self.initial_path != self.final_path

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": CARDINALITY_PLAN_SCHEMA,
            "schema_version": CARDINALITY_PLAN_SCHEMA_VERSION,
            "lower_bound": self.lower_bound.as_record(),
            "envelope": self.envelope.as_record(),
            "initial_path": self.initial_path,
            "final_path": self.final_path,
            "replanned": self.replanned,
            "observed": (
                self.observed.as_record() if self.observed is not None else None
            ),
            "violations": list(self.violations),
        }


def estimate_search_cardinality_lower_bound(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
) -> SearchCardinality:
    """Return cheap cardinality lower bounds before topology materialization.

    The estimate deliberately uses only bounds that cannot exceed the realized topology:
    every class contributes at least one chance edge; every root action contributes at
    least one information-set leaf; determinization contributes at least one leaf per
    root-action/world pair. This is a guard, not a hopeful selectivity prediction.
    """

    if method not in SEARCH_METHODS:
        raise CompiledSearchError(f"unknown search method {method!r}")
    try:
        actions, worlds_by_id, _ = _normalized_inputs(
            program_set=program_set,
            posterior=posterior,
            expected_program_schema=expected_program_schema,
            expected_program_schema_version=expected_program_schema_version,
        )
    except PartialInformationSearchError as error:
        raise CompiledSearchError(str(error)) from error

    class_count = 0
    outcome_rows = 0
    world_ids = set(worlds_by_id)
    for action in actions:
        try:
            classes = _validated_classes(
                program_set=program_set,
                action=action,
                world_ids=world_ids,
            )
        except PartialInformationSearchError as error:
            raise CompiledSearchError(str(error)) from error
        class_count += len(classes)
        outcome_rows += sum(
            len(_group_search_equivalent_outcomes(row["outcomes"]))
            for row in classes
        )

    leaf_lower_bound = len(actions)
    if method == "determinization":
        leaf_lower_bound *= len(worlds_by_id)

    return SearchCardinality(
        world_count=len(worlds_by_id),
        class_count=class_count,
        chance_edge_count=outcome_rows,
        leaf_count=leaf_lower_bound,
    )


def _cardinality_plan(
    *,
    lower_bound: SearchCardinality,
    envelope: CardinalityEnvelope,
    observed: SearchCardinality | None = None,
) -> AdaptiveCardinalityPlan:
    lower_violations = envelope.violations(lower_bound)
    if lower_violations:
        return AdaptiveCardinalityPlan(
            lower_bound=lower_bound,
            envelope=envelope,
            initial_path=SEARCH_PATH_PYTHON,
            final_path=SEARCH_PATH_PYTHON,
            observed=None,
            violations=lower_violations,
        )

    if observed is None:
        return AdaptiveCardinalityPlan(
            lower_bound=lower_bound,
            envelope=envelope,
            initial_path=SEARCH_PATH_COMPILED,
            final_path=SEARCH_PATH_COMPILED,
            observed=None,
            violations=(),
        )

    observed_violations = envelope.violations(observed)
    return AdaptiveCardinalityPlan(
        lower_bound=lower_bound,
        envelope=envelope,
        initial_path=SEARCH_PATH_COMPILED,
        final_path=(
            SEARCH_PATH_PYTHON
            if observed_violations
            else SEARCH_PATH_COMPILED
        ),
        observed=observed,
        violations=observed_violations,
    )
