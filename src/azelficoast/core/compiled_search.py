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
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

from azelficoast.core.evaluation import (
    EvaluationContribution,
    EvaluationFrontier,
    EvaluationFrontierError,
    EvaluationLeaf,
)
from azelficoast.core.search import (
    PartialInformationSearchError,
    SEARCH_METHODS,
    _EvaluatorMeter,
    _normalized_inputs,
    _validated_classes,
)
from azelficoast.core.transition import canonical_json, sha256_json

COMPILED_TOPOLOGY_SCHEMA = "azelficoast.core.compiled-search-topology"
COMPILED_TOPOLOGY_SCHEMA_VERSION = 1
COMPILED_SEARCH_SCHEMA = "azelficoast.core.compiled-partial-information-search"
COMPILED_SEARCH_SCHEMA_VERSION = 1


class CompiledSearchError(ValueError):
    """Raised when an authorized search topology cannot be compiled or transported."""


@dataclass(frozen=True, slots=True)
class CompiledSearchTopology:
    """Dense incidence structure derived from one validated transition program.

    `world_to_class[action][world]` identifies the verified execution class used by
    that world. Chance edges then map an action/world/class to an already-authorized
    public-observation partition and a method-specific evaluation leaf.
    """

    method: str
    program_digest: str
    topology_digest: str
    root_actions: tuple[str, ...]
    world_ids: tuple[str, ...]
    world_to_class: tuple[tuple[int, ...], ...]
    class_action_index: tuple[int, ...]
    class_local_index: tuple[int, ...]
    edge_action_index: tuple[int, ...]
    edge_world_index: tuple[int, ...]
    edge_class_index: tuple[int, ...]
    edge_outcome_index: tuple[int, ...]
    edge_observation_index: tuple[int, ...]
    edge_successor_index: tuple[int, ...]
    edge_leaf_index: tuple[int, ...]
    edge_chance: tuple[float, ...]
    leaf_action_index: tuple[int, ...]
    leaf_observation_index: tuple[int, ...]
    leaf_successor_index: tuple[int, ...]
    leaf_conditioned_world_index: tuple[int, ...]
    leaf_public_states: tuple[Mapping[str, Any], ...]
    leaf_legal_actions: tuple[tuple[str, ...], ...]
    leaf_legal_mask: tuple[tuple[bool, ...], ...]
    observation_keys: tuple[str, ...]
    successor_states: tuple[Mapping[str, Any], ...]
    successor_action_vocabulary: tuple[str, ...]
    transition_evaluations: int

    def __post_init__(self) -> None:
        if self.method not in SEARCH_METHODS:
            raise CompiledSearchError(f"unknown search method {self.method!r}")
        if not self.root_actions or len(set(self.root_actions)) != len(self.root_actions):
            raise CompiledSearchError("compiled topology needs unique root actions")
        if not self.world_ids or len(set(self.world_ids)) != len(self.world_ids):
            raise CompiledSearchError("compiled topology needs unique hidden worlds")
        if len(self.world_to_class) != len(self.root_actions):
            raise CompiledSearchError("world-to-class action dimension is invalid")
        if any(len(row) != len(self.world_ids) for row in self.world_to_class):
            raise CompiledSearchError("world-to-class world dimension is invalid")
        class_count = len(self.class_action_index)
        if class_count <= 0 or len(self.class_local_index) != class_count:
            raise CompiledSearchError("compiled topology has invalid class metadata")
        if self.transition_evaluations != class_count:
            raise CompiledSearchError(
                "transition evaluation count must equal verified execution classes"
            )
        edge_count = len(self.edge_world_index)
        edge_fields = (
            self.edge_action_index,
            self.edge_class_index,
            self.edge_outcome_index,
            self.edge_observation_index,
            self.edge_successor_index,
            self.edge_leaf_index,
            self.edge_chance,
        )
        if edge_count <= 0 or any(len(field) != edge_count for field in edge_fields):
            raise CompiledSearchError("compiled chance-edge arrays have inconsistent sizes")
        leaf_count = len(self.leaf_action_index)
        leaf_fields = (
            self.leaf_observation_index,
            self.leaf_successor_index,
            self.leaf_conditioned_world_index,
            self.leaf_public_states,
            self.leaf_legal_actions,
            self.leaf_legal_mask,
        )
        if leaf_count <= 0 or any(len(field) != leaf_count for field in leaf_fields):
            raise CompiledSearchError("compiled leaf arrays have inconsistent sizes")
        if any(not actions for actions in self.leaf_legal_actions):
            raise CompiledSearchError("compiled leaf has no legal actions")
        if not self.successor_action_vocabulary:
            raise CompiledSearchError("compiled topology has no successor action vocabulary")
        if any(
            len(mask) != len(self.successor_action_vocabulary)
            for mask in self.leaf_legal_mask
        ):
            raise CompiledSearchError("compiled legal-action mask width is invalid")
        for actions, mask in zip(
            self.leaf_legal_actions,
            self.leaf_legal_mask,
            strict=True,
        ):
            recovered = tuple(
                action
                for action, allowed in zip(
                    self.successor_action_vocabulary,
                    mask,
                    strict=True,
                )
                if allowed
            )
            if recovered != actions:
                raise CompiledSearchError(
                    "compiled legal-action mask does not match semantic legal actions"
                )
        if any(not math.isfinite(chance) or chance <= 0.0 for chance in self.edge_chance):
            raise CompiledSearchError("compiled chance edges must be positive and finite")
        if any(
            not 0 <= index < len(self.root_actions)
            for index in self.edge_action_index + self.leaf_action_index
        ):
            raise CompiledSearchError("compiled topology references an unknown root action")
        if any(
            not 0 <= index < len(self.world_ids)
            for index in self.edge_world_index
        ):
            raise CompiledSearchError("compiled topology references an unknown world")
        if any(not 0 <= index < class_count for index in self.edge_class_index):
            raise CompiledSearchError("compiled topology references an unknown class")
        if any(not 0 <= index < leaf_count for index in self.edge_leaf_index):
            raise CompiledSearchError("compiled topology references an unknown leaf")
        if any(
            not 0 <= index < len(self.observation_keys)
            for index in self.edge_observation_index + self.leaf_observation_index
        ):
            raise CompiledSearchError(
                "compiled topology references an unknown observation partition"
            )
        if any(
            not 0 <= index < len(self.successor_states)
            for index in self.edge_successor_index + self.leaf_successor_index
        ):
            raise CompiledSearchError(
                "compiled topology references an unknown successor state"
            )

    @property
    def action_count(self) -> int:
        return len(self.root_actions)

    @property
    def world_count(self) -> int:
        return len(self.world_ids)

    @property
    def class_count(self) -> int:
        return len(self.class_action_index)

    @property
    def edge_count(self) -> int:
        return len(self.edge_world_index)

    @property
    def leaf_count(self) -> int:
        return len(self.leaf_action_index)

    def as_record(self) -> dict[str, Any]:
        """Return the content-addressed semantic topology record."""

        return {
            "schema": COMPILED_TOPOLOGY_SCHEMA,
            "schema_version": COMPILED_TOPOLOGY_SCHEMA_VERSION,
            "method": self.method,
            "program_digest": self.program_digest,
            "root_actions": list(self.root_actions),
            "world_ids": list(self.world_ids),
            "world_to_class": [list(row) for row in self.world_to_class],
            "class_action_index": list(self.class_action_index),
            "class_local_index": list(self.class_local_index),
            "edge_action_index": list(self.edge_action_index),
            "edge_world_index": list(self.edge_world_index),
            "edge_class_index": list(self.edge_class_index),
            "edge_outcome_index": list(self.edge_outcome_index),
            "edge_observation_index": list(self.edge_observation_index),
            "edge_successor_index": list(self.edge_successor_index),
            "edge_leaf_index": list(self.edge_leaf_index),
            "edge_chance": list(self.edge_chance),
            "leaf_action_index": list(self.leaf_action_index),
            "leaf_observation_index": list(self.leaf_observation_index),
            "leaf_successor_index": list(self.leaf_successor_index),
            "leaf_conditioned_world_index": list(self.leaf_conditioned_world_index),
            "leaf_public_states": [copy.deepcopy(dict(row)) for row in self.leaf_public_states],
            "leaf_legal_actions": [list(row) for row in self.leaf_legal_actions],
            "leaf_legal_mask": [list(row) for row in self.leaf_legal_mask],
            "observation_keys": list(self.observation_keys),
            "successor_states": [
                copy.deepcopy(dict(row)) for row in self.successor_states
            ],
            "successor_action_vocabulary": list(self.successor_action_vocabulary),
            "transition_evaluations": self.transition_evaluations,
            "claim": (
                "Python validated information-set topology; numeric backends may only "
                "transport mass and reduce values through these fixed incidences."
            ),
        }

    def as_numpy(self) -> dict[str, np.ndarray]:
        """Materialize the integer incidence arrays used by JAX."""

        return {
            "world_to_class": np.asarray(self.world_to_class, dtype=np.int32),
            "class_action_index": np.asarray(self.class_action_index, dtype=np.int32),
            "class_local_index": np.asarray(self.class_local_index, dtype=np.int32),
            "edge_action_index": np.asarray(self.edge_action_index, dtype=np.int32),
            "edge_world_index": np.asarray(self.edge_world_index, dtype=np.int32),
            "edge_class_index": np.asarray(self.edge_class_index, dtype=np.int32),
            "edge_outcome_index": np.asarray(self.edge_outcome_index, dtype=np.int32),
            "edge_observation_index": np.asarray(
                self.edge_observation_index,
                dtype=np.int32,
            ),
            "edge_successor_index": np.asarray(
                self.edge_successor_index,
                dtype=np.int32,
            ),
            "edge_leaf_index": np.asarray(self.edge_leaf_index, dtype=np.int32),
            "edge_chance": np.asarray(self.edge_chance, dtype=np.float32),
            "leaf_action_index": np.asarray(self.leaf_action_index, dtype=np.int32),
            "leaf_observation_index": np.asarray(
                self.leaf_observation_index,
                dtype=np.int32,
            ),
            "leaf_successor_index": np.asarray(
                self.leaf_successor_index,
                dtype=np.int32,
            ),
            "leaf_legal_mask": np.asarray(self.leaf_legal_mask, dtype=np.bool_),
            "leaf_conditioned_world_index": np.asarray(
                self.leaf_conditioned_world_index,
                dtype=np.int32,
            ),
        }


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


def _observation_partition(
    action_index: int,
    observation: Any,
) -> str:
    """Action-scoped public observation identity.

    The same wire observation after two different root actions is not one continuation
    information set because the player remembers which action was chosen.
    """

    return canonical_json(
        {
            "root_action_index": action_index,
            "observation": observation,
        }
    )


def compile_search_topology(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
) -> CompiledSearchTopology:
    """Compile Python-authorized search semantics into dense incidence arrays."""

    if method not in SEARCH_METHODS:
        raise CompiledSearchError(f"unknown search method {method!r}")
    try:
        actions, _, _ = _normalized_inputs(
            program_set=program_set,
            posterior=posterior,
            expected_program_schema=expected_program_schema,
            expected_program_schema_version=expected_program_schema_version,
        )
    except PartialInformationSearchError as error:
        raise CompiledSearchError(str(error)) from error

    world_ids = tuple(program_set["world_ids"])
    world_index = {world_id: index for index, world_id in enumerate(world_ids)}

    world_to_class: list[list[int]] = [
        [-1] * len(world_ids) for _ in actions
    ]
    class_action_index: list[int] = []
    class_local_index: list[int] = []

    # Edge payloads are first accumulated with semantic observation keys. Leaf indices
    # are assigned only after every edge is present and successor/legal consistency can
    # be checked over the complete authorized group.
    edge_rows: list[dict[str, Any]] = []
    observation_index_by_key: dict[str, int] = {}
    observation_keys: list[str] = []
    successor_index_by_key: dict[str, int] = {}
    successor_states: list[Mapping[str, Any]] = []

    for action_index, action in enumerate(actions):
        try:
            classes = _validated_classes(
                program_set=program_set,
                action=action,
                world_ids=set(world_ids),
            )
        except PartialInformationSearchError as error:
            raise CompiledSearchError(str(error)) from error

        for local_class_index, row in enumerate(classes):
            global_class_index = len(class_action_index)
            class_action_index.append(action_index)
            class_local_index.append(local_class_index)

            members = list(row["member_world_ids"])
            for world_id in members:
                index = world_index[world_id]
                if world_to_class[action_index][index] != -1:
                    raise CompiledSearchError(
                        f"{action}: world {world_id!r} mapped to multiple classes"
                    )
                world_to_class[action_index][index] = global_class_index

            for outcome_index, outcome in enumerate(row["outcomes"]):
                observation_key = _observation_partition(
                    action_index,
                    outcome.get("observation"),
                )
                observation_index = observation_index_by_key.get(observation_key)
                if observation_index is None:
                    observation_index = len(observation_keys)
                    observation_index_by_key[observation_key] = observation_index
                    observation_keys.append(observation_key)

                successor_key = canonical_json(outcome["successor"])
                successor_index = successor_index_by_key.get(successor_key)
                if successor_index is None:
                    successor_index = len(successor_states)
                    successor_index_by_key[successor_key] = successor_index
                    successor_states.append(
                        copy.deepcopy(dict(outcome["successor"]))
                    )

                chance = float(outcome["probability"])
                for world_id in members:
                    w_index = world_index[world_id]
                    if method == "information_set":
                        leaf_key: tuple[int, ...] = (observation_index,)
                    else:
                        leaf_key = (observation_index, w_index)
                    edge_rows.append(
                        {
                            "action_index": action_index,
                            "world_index": w_index,
                            "class_index": global_class_index,
                            "outcome_index": outcome_index,
                            "observation_index": observation_index,
                            "successor_index": successor_index,
                            "leaf_key": leaf_key,
                            "chance": chance,
                            "outcome": outcome,
                        }
                    )

    if any(index < 0 for row in world_to_class for index in row):
        raise CompiledSearchError("compiled world-to-class map is incomplete")

    members_by_leaf_key: dict[tuple[int, ...], list[dict[str, Any]]] = defaultdict(list)
    for edge in edge_rows:
        members_by_leaf_key[edge["leaf_key"]].append(edge)

    # Stable order is by action, then observation partition, then conditioned world for
    # determinization. Observation indices already include action identity.
    ordered_leaf_keys = sorted(
        members_by_leaf_key,
        key=lambda key: (key[0], key[1] if len(key) > 1 else -1),
    )
    leaf_index_by_key = {
        key: index for index, key in enumerate(ordered_leaf_keys)
    }

    leaf_action_index: list[int] = []
    leaf_observation_index: list[int] = []
    leaf_successor_index: list[int] = []
    leaf_conditioned_world_index: list[int] = []
    leaf_public_states: list[Mapping[str, Any]] = []
    leaf_legal_actions: list[tuple[str, ...]] = []

    for key in ordered_leaf_keys:
        members = members_by_leaf_key[key]
        action_indices = {int(member["action_index"]) for member in members}
        observation_indices = {
            int(member["observation_index"]) for member in members
        }
        if len(action_indices) != 1 or len(observation_indices) != 1:
            raise CompiledSearchError("compiled leaf crosses an authorized partition")

        successor_indices = {
            int(member["successor_index"]) for member in members
        }
        if len(successor_indices) != 1:
            raise CompiledSearchError(
                "one public observation mapped to multiple successor public states"
            )
        successor_index = next(iter(successor_indices))
        successor = copy.deepcopy(dict(successor_states[successor_index]))

        legal_sets = [set(member["outcome"]["legal_actions"]) for member in members]
        common_legal = set(legal_sets[0])
        for legal in legal_sets[1:]:
            common_legal &= legal
        if not common_legal:
            raise CompiledSearchError(
                "successor information set has no common legal action"
            )

        leaf_action_index.append(next(iter(action_indices)))
        leaf_observation_index.append(next(iter(observation_indices)))
        leaf_successor_index.append(successor_index)
        leaf_conditioned_world_index.append(key[1] if len(key) > 1 else -1)
        leaf_public_states.append(successor)
        leaf_legal_actions.append(tuple(sorted(common_legal)))

    successor_action_vocabulary = tuple(
        sorted({action for row in leaf_legal_actions for action in row})
    )
    leaf_legal_mask = [
        tuple(
            action in set(legal_actions)
            for action in successor_action_vocabulary
        )
        for legal_actions in leaf_legal_actions
    ]

    edge_action_index: list[int] = []
    edge_world_index: list[int] = []
    edge_class_index: list[int] = []
    edge_outcome_index: list[int] = []
    edge_observation_index: list[int] = []
    edge_successor_index: list[int] = []
    edge_leaf_index: list[int] = []
    edge_chance: list[float] = []

    for edge in edge_rows:
        edge_action_index.append(int(edge["action_index"]))
        edge_world_index.append(int(edge["world_index"]))
        edge_class_index.append(int(edge["class_index"]))
        edge_outcome_index.append(int(edge["outcome_index"]))
        edge_observation_index.append(int(edge["observation_index"]))
        edge_successor_index.append(int(edge["successor_index"]))
        edge_leaf_index.append(leaf_index_by_key[edge["leaf_key"]])
        edge_chance.append(float(edge["chance"]))

    material = {
        "schema": COMPILED_TOPOLOGY_SCHEMA,
        "schema_version": COMPILED_TOPOLOGY_SCHEMA_VERSION,
        "method": method,
        "program_digest": sha256_json(program_set),
        "root_actions": list(actions),
        "world_ids": list(world_ids),
        "world_to_class": world_to_class,
        "class_action_index": class_action_index,
        "class_local_index": class_local_index,
        "edge_action_index": edge_action_index,
        "edge_world_index": edge_world_index,
        "edge_class_index": edge_class_index,
        "edge_outcome_index": edge_outcome_index,
        "edge_observation_index": edge_observation_index,
        "edge_successor_index": edge_successor_index,
        "edge_leaf_index": edge_leaf_index,
        "edge_chance": edge_chance,
        "leaf_action_index": leaf_action_index,
        "leaf_observation_index": leaf_observation_index,
        "leaf_successor_index": leaf_successor_index,
        "leaf_conditioned_world_index": leaf_conditioned_world_index,
        "leaf_public_states": leaf_public_states,
        "leaf_legal_actions": [list(row) for row in leaf_legal_actions],
        "leaf_legal_mask": [list(row) for row in leaf_legal_mask],
        "observation_keys": observation_keys,
        "successor_states": successor_states,
        "successor_action_vocabulary": list(successor_action_vocabulary),
        "transition_evaluations": len(class_action_index),
    }
    topology_digest = sha256_json(material)

    return CompiledSearchTopology(
        method=method,
        program_digest=material["program_digest"],
        topology_digest=topology_digest,
        root_actions=tuple(actions),
        world_ids=world_ids,
        world_to_class=tuple(tuple(row) for row in world_to_class),
        class_action_index=tuple(class_action_index),
        class_local_index=tuple(class_local_index),
        edge_action_index=tuple(edge_action_index),
        edge_world_index=tuple(edge_world_index),
        edge_class_index=tuple(edge_class_index),
        edge_outcome_index=tuple(edge_outcome_index),
        edge_observation_index=tuple(edge_observation_index),
        edge_successor_index=tuple(edge_successor_index),
        edge_leaf_index=tuple(edge_leaf_index),
        edge_chance=tuple(edge_chance),
        leaf_action_index=tuple(leaf_action_index),
        leaf_observation_index=tuple(leaf_observation_index),
        leaf_successor_index=tuple(leaf_successor_index),
        leaf_conditioned_world_index=tuple(leaf_conditioned_world_index),
        leaf_public_states=tuple(leaf_public_states),
        leaf_legal_actions=tuple(leaf_legal_actions),
        leaf_legal_mask=tuple(leaf_legal_mask),
        observation_keys=tuple(observation_keys),
        successor_states=tuple(successor_states),
        successor_action_vocabulary=successor_action_vocabulary,
        transition_evaluations=len(class_action_index),
    )


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


def _require_jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise CompiledSearchError(
            "JAX is required for compiled search transport; install the simulator extra"
        ) from error
    return jax, jnp


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
        "method": method,
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
            "observations": len(topology.observation_keys),
            "leaves": topology.leaf_count,
        },
        "numeric_backend": "jax",
        "semantic_authority": "python-validated-transition-program",
    }
