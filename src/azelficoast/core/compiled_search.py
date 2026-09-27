"""Compiled execution topology for finite partial-information search.

Ordinary Python remains authoritative for semantic grouping:

- transition-program class coverage;
- public-observation identity;
- successor public-state equality;
- common successor legal actions;
- determinization versus information-set coupling.

This module lowers that already-authorized topology into dense incidence arrays. JAX may
then transport posterior mass and reduce evaluated leaf values without deciding which
worlds are equivalent or which observations belong to one information set.

The first implementation is deliberately research-only. The existing
`search_transition_program` remains the live/reference path until exact semantic
comparison evidence justifies promotion.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Mapping, Protocol, Sequence, cast

import numpy as np

from azelficoast.core.compiled_planning import (
    AdaptiveCardinalityPlan as AdaptiveCardinalityPlan,
    CardinalityEnvelope as CardinalityEnvelope,
    CompiledSearchError as CompiledSearchError,
    JOIN_ORDER_AGGREGATE_FIRST as JOIN_ORDER_AGGREGATE_FIRST,
    JOIN_ORDER_EXPAND_FIRST as JOIN_ORDER_EXPAND_FIRST,
    OutcomeWorldJoinPlan as OutcomeWorldJoinPlan,
    SEARCH_PATH_COMPILED as SEARCH_PATH_COMPILED,
    SEARCH_PATH_PYTHON as SEARCH_PATH_PYTHON,
    SearchCardinality as SearchCardinality,
    _cardinality_plan,
    estimate_search_cardinality_lower_bound as estimate_search_cardinality_lower_bound,
)
from azelficoast.core.compiled_topology import (
    COMPILED_TOPOLOGY_SCHEMA as COMPILED_TOPOLOGY_SCHEMA,
    COMPILED_TOPOLOGY_SCHEMA_VERSION as COMPILED_TOPOLOGY_SCHEMA_VERSION,
    CompiledSearchTopology as CompiledSearchTopology,
    compile_search_topology as compile_search_topology,
)
from azelficoast.core.evaluation import (
    EvaluationContribution,
    EvaluationFrontier,
    EvaluationFrontierError,
    EvaluationLeaf,
)
from azelficoast.core.search import _EvaluatorMeter, search_transition_program

COMPILED_SEARCH_SCHEMA = "azelficoast.core.compiled-partial-information-search"
COMPILED_SEARCH_SCHEMA_VERSION = 2


class _JaxModule(Protocol):
    """Typed surface used from JAX without making JAX part of the core contract."""

    def jit(self, function: Callable[..., Any]) -> Callable[..., Any]: ...


def _observed_cardinality(topology: "CompiledSearchTopology") -> SearchCardinality:
    return SearchCardinality(
        world_count=topology.world_count,
        class_count=topology.class_count,
        chance_edge_count=topology.edge_count,
        leaf_count=topology.leaf_count,
    )


@dataclass(frozen=True, slots=True)
class TransportedSearchMass:
    """Posterior mass after JAX transport through a compiled topology."""

    normalized_world_weights: np.ndarray
    leaf_mass: np.ndarray
    leaf_world_mass: np.ndarray
    leaf_world_weights: np.ndarray

    def __post_init__(self) -> None:
        if self.normalized_world_weights.ndim != 1:
            raise CompiledSearchError("world weights must be one-dimensional")
        if self.leaf_mass.ndim != 1:
            raise CompiledSearchError("leaf mass must be one-dimensional")
        if self.leaf_world_mass.ndim != 2 or self.leaf_world_weights.ndim != 2:
            raise CompiledSearchError("leaf/world mass tensors must be two-dimensional")
        expected = (len(self.leaf_mass), len(self.normalized_world_weights))
        if self.leaf_world_mass.shape != expected or self.leaf_world_weights.shape != expected:
            raise CompiledSearchError("transported leaf/world tensor shape is invalid")
        if np.any(~np.isfinite(self.leaf_mass)) or np.any(self.leaf_mass <= 0.0):
            raise CompiledSearchError("every compiled leaf must carry positive finite mass")
        if np.any(~np.isfinite(self.leaf_world_weights)):
            raise CompiledSearchError("transported posterior contains non-finite weights")


def _posterior_for_topology(
    topology: CompiledSearchTopology,
    posterior: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], np.ndarray]:
    raw_worlds = posterior.get("worlds")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise CompiledSearchError("posterior has no hidden-world support")

    worlds_by_id: dict[str, Mapping[str, Any]] = {}
    raw_weights: dict[str, float] = {}
    for raw_world in raw_worlds:
        if not isinstance(raw_world, Mapping):
            raise CompiledSearchError("posterior world must be an object")
        world_id = raw_world.get("world_id")
        if not isinstance(world_id, str) or not world_id:
            raise CompiledSearchError("posterior world has invalid identity")
        if world_id in worlds_by_id:
            raise CompiledSearchError("posterior world ids must be unique")
        hidden = raw_world.get("hidden")
        if not isinstance(hidden, Mapping):
            raise CompiledSearchError(
                f"{world_id}: posterior must retain correlated hidden state"
            )
        weight = raw_world.get("weight")
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise CompiledSearchError(
                f"{world_id}: posterior weight must be positive and finite"
            )
        worlds_by_id[world_id] = raw_world
        raw_weights[world_id] = float(weight)

    if set(worlds_by_id) != set(topology.world_ids):
        raise CompiledSearchError(
            "posterior support differs from compiled transition-program support"
        )
    total = math.fsum(raw_weights.values())
    weights = np.asarray(
        [raw_weights[world_id] / total for world_id in topology.world_ids],
        dtype=np.float32,
    )
    return worlds_by_id, weights


def _require_jax() -> tuple[_JaxModule, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise CompiledSearchError(
            "JAX is required for compiled search transport; install the simulator extra"
        ) from error
    return cast(_JaxModule, jax), jnp


@lru_cache(maxsize=None)
def _transport_kernel(leaf_count: int, world_count: int) -> Any:
    jax, jnp = _require_jax()

    @jax.jit
    def transport(
        world_weights: Any,
        edge_world_index: Any,
        edge_leaf_index: Any,
        edge_chance: Any,
    ) -> tuple[Any, Any, Any]:
        edge_mass = world_weights[edge_world_index] * edge_chance
        leaf_world_mass = jnp.zeros(
            (leaf_count, world_count),
            dtype=jnp.float32,
        ).at[edge_leaf_index, edge_world_index].add(edge_mass)
        leaf_mass = jnp.sum(leaf_world_mass, axis=1)
        denominator = jnp.maximum(leaf_mass[:, None], 1e-30)
        normalized = leaf_world_mass / denominator
        return leaf_mass, leaf_world_mass, normalized

    return transport


@lru_cache(maxsize=None)
def _reduction_kernel(action_count: int) -> Any:
    jax, jnp = _require_jax()

    @jax.jit
    def reduce(
        leaf_values: Any,
        leaf_mass: Any,
        leaf_action_index: Any,
    ) -> Any:
        weighted = leaf_values * leaf_mass
        return jnp.zeros((action_count,), dtype=jnp.float32).at[
            leaf_action_index
        ].add(weighted)

    return reduce


def transport_posterior_mass(
    topology: CompiledSearchTopology,
    posterior: Mapping[str, Any],
) -> TransportedSearchMass:
    """Transport posterior mass through fixed chance/observation incidences in JAX."""

    _, jnp = _require_jax()
    _, world_weights = _posterior_for_topology(topology, posterior)
    arrays = topology.as_numpy()
    raw_leaf_mass, raw_leaf_world_mass, raw_leaf_world_weights = _transport_kernel(
        topology.leaf_count,
        topology.world_count,
    )(
        jnp.asarray(world_weights, dtype=jnp.float32),
        jnp.asarray(arrays["edge_world_index"], dtype=jnp.int32),
        jnp.asarray(arrays["edge_leaf_index"], dtype=jnp.int32),
        jnp.asarray(arrays["edge_chance"], dtype=jnp.float32),
    )
    leaf_mass = np.asarray(raw_leaf_mass, dtype=np.float64)
    leaf_world_mass = np.asarray(raw_leaf_world_mass, dtype=np.float64)
    leaf_world_weights = np.asarray(raw_leaf_world_weights, dtype=np.float64)

    if leaf_mass.shape != (topology.leaf_count,):
        raise CompiledSearchError("compiled transport returned the wrong leaf shape")
    if abs(float(leaf_mass.sum()) - float(topology.action_count)) > 1e-5:
        raise CompiledSearchError(
            "compiled transport lost or duplicated root-action probability mass"
        )

    return TransportedSearchMass(
        normalized_world_weights=world_weights.astype(np.float64),
        leaf_mass=leaf_mass,
        leaf_world_mass=leaf_world_mass,
        leaf_world_weights=leaf_world_weights,
    )


def materialize_compiled_frontier(
    topology: CompiledSearchTopology,
    posterior: Mapping[str, Any],
) -> tuple[EvaluationFrontier, TransportedSearchMass]:
    """Build evaluator leaves from JAX-transported mass and Python-authorized metadata."""

    worlds_by_id, _ = _posterior_for_topology(topology, posterior)
    transported = transport_posterior_mass(topology, posterior)

    leaves: list[EvaluationLeaf] = []
    contributions: list[EvaluationContribution] = []
    for leaf_index in range(topology.leaf_count):
        mass = float(transported.leaf_mass[leaf_index])
        posterior_worlds: list[dict[str, Any]] = []
        for world_index, world_id in enumerate(topology.world_ids):
            weight = float(transported.leaf_world_weights[leaf_index, world_index])
            if weight <= 0.0:
                continue
            row = copy.deepcopy(dict(worlds_by_id[world_id]))
            row["weight"] = weight
            posterior_worlds.append(row)
        if not posterior_worlds:
            raise CompiledSearchError("compiled leaf has no posterior support")

        try:
            leaves.append(
                EvaluationLeaf(
                    public_state=copy.deepcopy(
                        dict(topology.leaf_public_states[leaf_index])
                    ),
                    posterior=tuple(posterior_worlds),
                    legal_actions=topology.leaf_legal_actions[leaf_index],
                )
            )
            contributions.append(
                EvaluationContribution(
                    root_action=topology.root_actions[
                        topology.leaf_action_index[leaf_index]
                    ],
                    leaf_index=leaf_index,
                    coefficient=mass,
                )
            )
        except EvaluationFrontierError as error:
            raise CompiledSearchError(str(error)) from error

    try:
        frontier = EvaluationFrontier(
            root_actions=topology.root_actions,
            leaves=tuple(leaves),
            contributions=tuple(contributions),
            transition_evaluations=topology.transition_evaluations,
        )
    except EvaluationFrontierError as error:
        raise CompiledSearchError(str(error)) from error
    return frontier, transported


def reduce_compiled_root_values(
    topology: CompiledSearchTopology,
    transported: TransportedSearchMass,
    leaf_values: Sequence[float],
) -> dict[str, float]:
    """Reduce fixed leaf values to root actions through JAX scatter-add."""

    if len(leaf_values) != topology.leaf_count:
        raise CompiledSearchError("leaf-value count does not match compiled topology")
    values = np.asarray(tuple(float(value) for value in leaf_values), dtype=np.float32)
    if np.any(~np.isfinite(values)):
        raise CompiledSearchError("leaf values must be finite")

    _, jnp = _require_jax()
    arrays = topology.as_numpy()
    raw = _reduction_kernel(topology.action_count)(
        jnp.asarray(values, dtype=jnp.float32),
        jnp.asarray(transported.leaf_mass, dtype=jnp.float32),
        jnp.asarray(arrays["leaf_action_index"], dtype=jnp.int32),
    )
    root = np.asarray(raw, dtype=np.float64)
    if root.shape != (topology.action_count,) or np.any(~np.isfinite(root)):
        raise CompiledSearchError("compiled root reduction returned invalid values")
    return {
        action: float(root[index])
        for index, action in enumerate(topology.root_actions)
    }


def _execute_compiled_topology(
    *,
    topology: CompiledSearchTopology,
    posterior: Mapping[str, Any],
    evaluator: Any,
) -> dict[str, Any]:
    frontier, transported = materialize_compiled_frontier(topology, posterior)
    meter = _EvaluatorMeter(evaluator)
    leaf_values = meter.values(frontier.leaves)
    root_values = reduce_compiled_root_values(
        topology,
        transported,
        leaf_values,
    )
    best = max(root_values.values())
    chosen_action = min(
        action for action, value in root_values.items() if value == best
    )
    return {
        "schema": COMPILED_SEARCH_SCHEMA,
        "schema_version": COMPILED_SEARCH_SCHEMA_VERSION,
        "method": topology.method,
        "transition_program_digest": topology.program_digest,
        "compiled_topology_digest": topology.topology_digest,
        "transition_evaluations": topology.transition_evaluations,
        "evaluator_calls": meter.calls,
        "evaluator_batches": meter.batches,
        "chosen_action": chosen_action,
        "root_values": root_values,
        "compiled_shape": {
            "actions": topology.action_count,
            "worlds": topology.world_count,
            "classes": topology.class_count,
            "chance_edges": topology.edge_count,
            "raw_chance_edges": topology.outcome_world_join_plan.raw_join_rows,
            "outcome_join_order": (
                topology.outcome_world_join_plan.selected_order
            ),
            "outcome_join_saved_rows": (
                topology.outcome_world_join_plan.saved_join_rows
            ),
            "observations": len(topology.observation_keys),
            "leaves": topology.leaf_count,
            "dense_leaf_world_cells": topology.leaf_count * topology.world_count,
        },
        "numeric_backend": "jax",
        "semantic_authority": "python-validated-transition-program",
    }


def search_transition_program_compiled(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    evaluator: Any,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
) -> dict[str, Any]:
    """Research path: compile topology, transport mass in JAX, then evaluate leaves."""

    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )
    return _execute_compiled_topology(
        topology=topology,
        posterior=posterior,
        evaluator=evaluator,
    )


def search_transition_program_adaptive(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    evaluator: Any,
    envelope: CardinalityEnvelope,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
) -> dict[str, Any]:
    """Choose dense compiled search only inside an explicit cardinality envelope.

    The planner first evaluates a proven lower bound. If it fits, Python compiles the
    semantic topology and the planner re-checks exact realized cardinalities before any
    dense JAX posterior transport. A realized shape surprise therefore changes only the
    physical implementation, never the logical search or evaluator semantics.
    """

    lower_bound = estimate_search_cardinality_lower_bound(
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )
    plan = _cardinality_plan(
        lower_bound=lower_bound,
        envelope=envelope,
    )
    if plan.final_path == SEARCH_PATH_PYTHON:
        result = search_transition_program(
            program_set=program_set,
            posterior=posterior,
            method=method,
            evaluator=evaluator,
            expected_program_schema=expected_program_schema,
            expected_program_schema_version=expected_program_schema_version,
        )
        return {
            **result,
            "physical_search_path": SEARCH_PATH_PYTHON,
            "cardinality_plan": plan.as_record(),
        }

    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )
    observed = _observed_cardinality(topology)
    plan = _cardinality_plan(
        lower_bound=lower_bound,
        envelope=envelope,
        observed=observed,
    )
    if plan.final_path == SEARCH_PATH_PYTHON:
        result = search_transition_program(
            program_set=program_set,
            posterior=posterior,
            method=method,
            evaluator=evaluator,
            expected_program_schema=expected_program_schema,
            expected_program_schema_version=expected_program_schema_version,
        )
        return {
            **result,
            "physical_search_path": SEARCH_PATH_PYTHON,
            "cardinality_plan": plan.as_record(),
        }

    result = _execute_compiled_topology(
        topology=topology,
        posterior=posterior,
        evaluator=evaluator,
    )
    return {
        **result,
        "physical_search_path": SEARCH_PATH_COMPILED,
        "cardinality_plan": plan.as_record(),
    }
