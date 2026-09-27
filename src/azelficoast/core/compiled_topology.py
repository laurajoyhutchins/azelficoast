"""Authorized topology compilation for finite partial-information search.

This module owns the semantic lowering from validated transition programs to immutable
incidence topology. Numeric execution belongs elsewhere: JAX may transport posterior
mass and reduce values only after this module has fixed the authorized partitions.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from azelficoast.core.compiled_planning import (
    CompiledSearchError,
    JOIN_ORDER_AGGREGATE_FIRST,
    OutcomeWorldJoinPlan,
    _group_search_equivalent_outcomes,
    _plan_outcome_world_join,
)
from azelficoast.core.search import (
    PartialInformationSearchError,
    SEARCH_METHODS,
    _normalized_inputs,
    _validated_classes,
)
from azelficoast.core.transition import canonical_json, sha256_json

COMPILED_TOPOLOGY_SCHEMA = "azelficoast.core.compiled-search-topology"
COMPILED_TOPOLOGY_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class CompiledClassIncidence:
    """World-to-class incidence and verified class identity."""

    world_to_class: tuple[tuple[int, ...], ...]
    action_index: tuple[int, ...]
    local_index: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CompiledEdgeIncidence:
    """Chance-edge incidence columns for the authorized topology."""

    action_index: tuple[int, ...]
    world_index: tuple[int, ...]
    class_index: tuple[int, ...]
    outcome_index: tuple[int, ...]
    observation_index: tuple[int, ...]
    successor_index: tuple[int, ...]
    leaf_index: tuple[int, ...]
    chance: tuple[float, ...]

    def __post_init__(self) -> None:
        edge_count = len(self.world_index)
        columns = (
            self.action_index,
            self.class_index,
            self.outcome_index,
            self.observation_index,
            self.successor_index,
            self.leaf_index,
            self.chance,
        )
        if edge_count <= 0 or any(len(column) != edge_count for column in columns):
            raise CompiledSearchError("compiled chance-edge arrays have inconsistent sizes")
        if any(not math.isfinite(chance) or chance <= 0.0 for chance in self.chance):
            raise CompiledSearchError("compiled chance edges must be positive and finite")


@dataclass(frozen=True, slots=True)
class CompiledLeafIncidence:
    """Evaluation-leaf incidence and authorized successor semantics."""

    action_index: tuple[int, ...]
    observation_index: tuple[int, ...]
    successor_index: tuple[int, ...]
    conditioned_world_index: tuple[int, ...]
    public_states: tuple[Mapping[str, Any], ...]
    legal_actions: tuple[tuple[str, ...], ...]
    legal_mask: tuple[tuple[bool, ...], ...]

    def __post_init__(self) -> None:
        leaf_count = len(self.action_index)
        columns = (
            self.observation_index,
            self.successor_index,
            self.conditioned_world_index,
            self.public_states,
            self.legal_actions,
            self.legal_mask,
        )
        if leaf_count <= 0 or any(len(column) != leaf_count for column in columns):
            raise CompiledSearchError("compiled leaf arrays have inconsistent sizes")
        if any(not actions for actions in self.legal_actions):
            raise CompiledSearchError("compiled leaf has no legal actions")


@dataclass(frozen=True, slots=True)
class CompiledSearchTopology:
    """Immutable semantic topology consumed by numeric search backends."""

    method: str
    program_digest: str
    topology_digest: str
    outcome_world_join_plan: OutcomeWorldJoinPlan
    root_actions: tuple[str, ...]
    world_ids: tuple[str, ...]
    classes: CompiledClassIncidence
    edges: CompiledEdgeIncidence
    leaves: CompiledLeafIncidence
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
        if len(self.classes.world_to_class) != len(self.root_actions):
            raise CompiledSearchError("world-to-class action dimension is invalid")
        if any(len(row) != len(self.world_ids) for row in self.classes.world_to_class):
            raise CompiledSearchError("world-to-class world dimension is invalid")

        class_count = len(self.classes.action_index)
        if class_count <= 0 or len(self.classes.local_index) != class_count:
            raise CompiledSearchError("compiled topology has invalid class metadata")
        if self.transition_evaluations != class_count:
            raise CompiledSearchError(
                "transition evaluation count must equal verified execution classes"
            )

        if not self.successor_action_vocabulary:
            raise CompiledSearchError("compiled topology has no successor action vocabulary")
        if any(
            len(mask) != len(self.successor_action_vocabulary)
            for mask in self.leaves.legal_mask
        ):
            raise CompiledSearchError("compiled legal-action mask width is invalid")
        for actions, mask in zip(
            self.leaves.legal_actions,
            self.leaves.legal_mask,
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

        if any(
            not 0 <= index < len(self.root_actions)
            for index in self.edges.action_index + self.leaves.action_index
        ):
            raise CompiledSearchError("compiled topology references an unknown root action")
        if any(not 0 <= index < len(self.world_ids) for index in self.edges.world_index):
            raise CompiledSearchError("compiled topology references an unknown world")
        if any(not 0 <= index < class_count for index in self.edges.class_index):
            raise CompiledSearchError("compiled topology references an unknown class")
        if any(not 0 <= index < self.leaf_count for index in self.edges.leaf_index):
            raise CompiledSearchError("compiled topology references an unknown leaf")
        if any(
            not 0 <= index < len(self.observation_keys)
            for index in self.edges.observation_index + self.leaves.observation_index
        ):
            raise CompiledSearchError(
                "compiled topology references an unknown observation partition"
            )
        if any(
            not 0 <= index < len(self.successor_states)
            for index in self.edges.successor_index + self.leaves.successor_index
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
        return len(self.classes.action_index)

    @property
    def edge_count(self) -> int:
        return len(self.edges.world_index)

    @property
    def leaf_count(self) -> int:
        return len(self.leaves.action_index)

    # Compatibility views keep existing consumers source-compatible while the
    # authoritative representation remains grouped by incidence kind.
    @property
    def world_to_class(self) -> tuple[tuple[int, ...], ...]:
        return self.classes.world_to_class

    @property
    def class_action_index(self) -> tuple[int, ...]:
        return self.classes.action_index

    @property
    def class_local_index(self) -> tuple[int, ...]:
        return self.classes.local_index

    @property
    def edge_action_index(self) -> tuple[int, ...]:
        return self.edges.action_index

    @property
    def edge_world_index(self) -> tuple[int, ...]:
        return self.edges.world_index

    @property
    def edge_class_index(self) -> tuple[int, ...]:
        return self.edges.class_index

    @property
    def edge_outcome_index(self) -> tuple[int, ...]:
        return self.edges.outcome_index

    @property
    def edge_observation_index(self) -> tuple[int, ...]:
        return self.edges.observation_index

    @property
    def edge_successor_index(self) -> tuple[int, ...]:
        return self.edges.successor_index

    @property
    def edge_leaf_index(self) -> tuple[int, ...]:
        return self.edges.leaf_index

    @property
    def edge_chance(self) -> tuple[float, ...]:
        return self.edges.chance

    @property
    def leaf_action_index(self) -> tuple[int, ...]:
        return self.leaves.action_index

    @property
    def leaf_observation_index(self) -> tuple[int, ...]:
        return self.leaves.observation_index

    @property
    def leaf_successor_index(self) -> tuple[int, ...]:
        return self.leaves.successor_index

    @property
    def leaf_conditioned_world_index(self) -> tuple[int, ...]:
        return self.leaves.conditioned_world_index

    @property
    def leaf_public_states(self) -> tuple[Mapping[str, Any], ...]:
        return self.leaves.public_states

    @property
    def leaf_legal_actions(self) -> tuple[tuple[str, ...], ...]:
        return self.leaves.legal_actions

    @property
    def leaf_legal_mask(self) -> tuple[tuple[bool, ...], ...]:
        return self.leaves.legal_mask

    def as_record(self) -> dict[str, Any]:
        """Return the unchanged content-addressed semantic topology record."""

        return {
            "schema": COMPILED_TOPOLOGY_SCHEMA,
            "schema_version": COMPILED_TOPOLOGY_SCHEMA_VERSION,
            "method": self.method,
            "program_digest": self.program_digest,
            "outcome_world_join_plan": self.outcome_world_join_plan.as_record(),
            "root_actions": list(self.root_actions),
            "world_ids": list(self.world_ids),
            "world_to_class": [list(row) for row in self.classes.world_to_class],
            "class_action_index": list(self.classes.action_index),
            "class_local_index": list(self.classes.local_index),
            "edge_action_index": list(self.edges.action_index),
            "edge_world_index": list(self.edges.world_index),
            "edge_class_index": list(self.edges.class_index),
            "edge_outcome_index": list(self.edges.outcome_index),
            "edge_observation_index": list(self.edges.observation_index),
            "edge_successor_index": list(self.edges.successor_index),
            "edge_leaf_index": list(self.edges.leaf_index),
            "edge_chance": list(self.edges.chance),
            "leaf_action_index": list(self.leaves.action_index),
            "leaf_observation_index": list(self.leaves.observation_index),
            "leaf_successor_index": list(self.leaves.successor_index),
            "leaf_conditioned_world_index": list(self.leaves.conditioned_world_index),
            "leaf_public_states": [
                copy.deepcopy(dict(row)) for row in self.leaves.public_states
            ],
            "leaf_legal_actions": [list(row) for row in self.leaves.legal_actions],
            "leaf_legal_mask": [list(row) for row in self.leaves.legal_mask],
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
        """Materialize the unchanged dense-array interface used by JAX."""

        return {
            "world_to_class": np.asarray(self.classes.world_to_class, dtype=np.int32),
            "class_action_index": np.asarray(self.classes.action_index, dtype=np.int32),
            "class_local_index": np.asarray(self.classes.local_index, dtype=np.int32),
            "edge_action_index": np.asarray(self.edges.action_index, dtype=np.int32),
            "edge_world_index": np.asarray(self.edges.world_index, dtype=np.int32),
            "edge_class_index": np.asarray(self.edges.class_index, dtype=np.int32),
            "edge_outcome_index": np.asarray(self.edges.outcome_index, dtype=np.int32),
            "edge_observation_index": np.asarray(
                self.edges.observation_index,
                dtype=np.int32,
            ),
            "edge_successor_index": np.asarray(
                self.edges.successor_index,
                dtype=np.int32,
            ),
            "edge_leaf_index": np.asarray(self.edges.leaf_index, dtype=np.int32),
            "edge_chance": np.asarray(self.edges.chance, dtype=np.float32),
            "leaf_action_index": np.asarray(self.leaves.action_index, dtype=np.int32),
            "leaf_observation_index": np.asarray(
                self.leaves.observation_index,
                dtype=np.int32,
            ),
            "leaf_successor_index": np.asarray(
                self.leaves.successor_index,
                dtype=np.int32,
            ),
            "leaf_legal_mask": np.asarray(self.leaves.legal_mask, dtype=np.bool_),
            "leaf_conditioned_world_index": np.asarray(
                self.leaves.conditioned_world_index,
                dtype=np.int32,
            ),
        }


type _PreparedClass = tuple[Mapping[str, Any], tuple[dict[str, Any], ...]]
type _PreparedAction = tuple[str, list[_PreparedClass]]
type _LeafKey = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _EdgeRow:
    action_index: int
    world_index: int
    class_index: int
    outcome_index: int
    observation_index: int
    successor_index: int
    leaf_key: _LeafKey
    chance: float
    outcome: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _EdgeCompilation:
    world_to_class: list[list[int]]
    class_action_index: list[int]
    class_local_index: list[int]
    rows: list[_EdgeRow]
    observation_keys: list[str]
    successor_states: list[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class _LeafCompilation:
    index_by_key: dict[_LeafKey, int]
    action_index: list[int]
    observation_index: list[int]
    successor_index: list[int]
    conditioned_world_index: list[int]
    public_states: list[Mapping[str, Any]]
    legal_actions: list[tuple[str, ...]]
    legal_mask: list[tuple[bool, ...]]
    successor_action_vocabulary: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _EdgeColumns:
    action_index: list[int]
    world_index: list[int]
    class_index: list[int]
    outcome_index: list[int]
    observation_index: list[int]
    successor_index: list[int]
    leaf_index: list[int]
    chance: list[float]


def _observation_partition(
    action_index: int,
    observation: Any,
) -> str:
    """Return action-scoped public observation identity."""

    return canonical_json(
        {
            "root_action_index": action_index,
            "observation": observation,
        }
    )


def _validated_actions(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    expected_program_schema: str | None,
    expected_program_schema_version: int | None,
) -> Sequence[str]:
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
    return actions


def _prepare_classes(
    *,
    actions: Sequence[str],
    program_set: Mapping[str, Any],
    world_ids: tuple[str, ...],
) -> tuple[list[_PreparedAction], OutcomeWorldJoinPlan]:
    prepared_by_action: list[_PreparedAction] = []
    prepared_classes_flat: list[_PreparedClass] = []

    for action in actions:
        try:
            classes = _validated_classes(
                program_set=program_set,
                action=action,
                world_ids=set(world_ids),
            )
        except PartialInformationSearchError as error:
            raise CompiledSearchError(str(error)) from error

        prepared: list[_PreparedClass] = [
            (row, _group_search_equivalent_outcomes(row["outcomes"]))
            for row in classes
        ]
        prepared_by_action.append((action, prepared))
        prepared_classes_flat.extend(prepared)

    return prepared_by_action, _plan_outcome_world_join(prepared_classes_flat)


def _compile_edges(
    *,
    actions: Sequence[str],
    world_ids: tuple[str, ...],
    prepared_by_action: Sequence[_PreparedAction],
    join_plan: OutcomeWorldJoinPlan,
    method: str,
) -> _EdgeCompilation:
    world_index = {world_id: index for index, world_id in enumerate(world_ids)}
    world_to_class = [[-1] * len(world_ids) for _ in actions]
    class_action_index: list[int] = []
    class_local_index: list[int] = []
    rows: list[_EdgeRow] = []
    observation_index_by_key: dict[str, int] = {}
    observation_keys: list[str] = []
    successor_index_by_key: dict[str, int] = {}
    successor_states: list[Mapping[str, Any]] = []

    for action_index, (action, prepared_classes) in enumerate(prepared_by_action):
        for local_class_index, (row, grouped_outcomes) in enumerate(prepared_classes):
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

            selected_outcomes: Sequence[Mapping[str, Any]] = (
                grouped_outcomes
                if join_plan.selected_order == JOIN_ORDER_AGGREGATE_FIRST
                else row["outcomes"]
            )
            for outcome_index, outcome in enumerate(selected_outcomes):
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
                    successor_states.append(copy.deepcopy(dict(outcome["successor"])))

                chance = float(outcome["probability"])
                for world_id in members:
                    w_index = world_index[world_id]
                    leaf_key: _LeafKey = (
                        (observation_index,)
                        if method == "information_set"
                        else (observation_index, w_index)
                    )
                    rows.append(
                        _EdgeRow(
                            action_index=action_index,
                            world_index=w_index,
                            class_index=global_class_index,
                            outcome_index=outcome_index,
                            observation_index=observation_index,
                            successor_index=successor_index,
                            leaf_key=leaf_key,
                            chance=chance,
                            outcome=outcome,
                        )
                    )

    if any(index < 0 for row in world_to_class for index in row):
        raise CompiledSearchError("compiled world-to-class map is incomplete")

    return _EdgeCompilation(
        world_to_class=world_to_class,
        class_action_index=class_action_index,
        class_local_index=class_local_index,
        rows=rows,
        observation_keys=observation_keys,
        successor_states=successor_states,
    )


def _compile_leaves(edges: _EdgeCompilation) -> _LeafCompilation:
    members_by_leaf_key: dict[_LeafKey, list[_EdgeRow]] = defaultdict(list)
    for edge in edges.rows:
        members_by_leaf_key[edge.leaf_key].append(edge)

    ordered_leaf_keys = sorted(
        members_by_leaf_key,
        key=lambda key: (key[0], key[1] if len(key) > 1 else -1),
    )
    index_by_key = {key: index for index, key in enumerate(ordered_leaf_keys)}

    action_index: list[int] = []
    observation_index: list[int] = []
    successor_index: list[int] = []
    conditioned_world_index: list[int] = []
    public_states: list[Mapping[str, Any]] = []
    legal_actions: list[tuple[str, ...]] = []

    for key in ordered_leaf_keys:
        members = members_by_leaf_key[key]
        action_indices = {member.action_index for member in members}
        observation_indices = {member.observation_index for member in members}
        if len(action_indices) != 1 or len(observation_indices) != 1:
            raise CompiledSearchError("compiled leaf crosses an authorized partition")

        successor_indices = {member.successor_index for member in members}
        if len(successor_indices) != 1:
            raise CompiledSearchError(
                "one public observation mapped to multiple successor public states"
            )
        resolved_successor_index = next(iter(successor_indices))
        successor = copy.deepcopy(
            dict(edges.successor_states[resolved_successor_index])
        )

        legal_sets = [set(member.outcome["legal_actions"]) for member in members]
        common_legal = set(legal_sets[0])
        for legal in legal_sets[1:]:
            common_legal &= legal
        if not common_legal:
            raise CompiledSearchError(
                "successor information set has no common legal action"
            )

        action_index.append(next(iter(action_indices)))
        observation_index.append(next(iter(observation_indices)))
        successor_index.append(resolved_successor_index)
        conditioned_world_index.append(key[1] if len(key) > 1 else -1)
        public_states.append(successor)
        legal_actions.append(tuple(sorted(common_legal)))

    successor_action_vocabulary = tuple(
        sorted({action for row in legal_actions for action in row})
    )
    legal_mask = [
        tuple(action in set(actions) for action in successor_action_vocabulary)
        for actions in legal_actions
    ]

    return _LeafCompilation(
        index_by_key=index_by_key,
        action_index=action_index,
        observation_index=observation_index,
        successor_index=successor_index,
        conditioned_world_index=conditioned_world_index,
        public_states=public_states,
        legal_actions=legal_actions,
        legal_mask=legal_mask,
        successor_action_vocabulary=successor_action_vocabulary,
    )


def _edge_columns(
    edges: _EdgeCompilation,
    leaves: _LeafCompilation,
) -> _EdgeColumns:
    return _EdgeColumns(
        action_index=[edge.action_index for edge in edges.rows],
        world_index=[edge.world_index for edge in edges.rows],
        class_index=[edge.class_index for edge in edges.rows],
        outcome_index=[edge.outcome_index for edge in edges.rows],
        observation_index=[edge.observation_index for edge in edges.rows],
        successor_index=[edge.successor_index for edge in edges.rows],
        leaf_index=[leaves.index_by_key[edge.leaf_key] for edge in edges.rows],
        chance=[edge.chance for edge in edges.rows],
    )


def _topology_material(
    *,
    method: str,
    program_digest: str,
    actions: Sequence[str],
    world_ids: tuple[str, ...],
    join_plan: OutcomeWorldJoinPlan,
    edges: _EdgeCompilation,
    edge_columns: _EdgeColumns,
    leaves: _LeafCompilation,
) -> dict[str, Any]:
    return {
        "schema": COMPILED_TOPOLOGY_SCHEMA,
        "schema_version": COMPILED_TOPOLOGY_SCHEMA_VERSION,
        "method": method,
        "program_digest": program_digest,
        "outcome_world_join_plan": join_plan.as_record(),
        "root_actions": list(actions),
        "world_ids": list(world_ids),
        "world_to_class": edges.world_to_class,
        "class_action_index": edges.class_action_index,
        "class_local_index": edges.class_local_index,
        "edge_action_index": edge_columns.action_index,
        "edge_world_index": edge_columns.world_index,
        "edge_class_index": edge_columns.class_index,
        "edge_outcome_index": edge_columns.outcome_index,
        "edge_observation_index": edge_columns.observation_index,
        "edge_successor_index": edge_columns.successor_index,
        "edge_leaf_index": edge_columns.leaf_index,
        "edge_chance": edge_columns.chance,
        "leaf_action_index": leaves.action_index,
        "leaf_observation_index": leaves.observation_index,
        "leaf_successor_index": leaves.successor_index,
        "leaf_conditioned_world_index": leaves.conditioned_world_index,
        "leaf_public_states": leaves.public_states,
        "leaf_legal_actions": [list(row) for row in leaves.legal_actions],
        "leaf_legal_mask": [list(row) for row in leaves.legal_mask],
        "observation_keys": edges.observation_keys,
        "successor_states": edges.successor_states,
        "successor_action_vocabulary": list(leaves.successor_action_vocabulary),
        "transition_evaluations": len(edges.class_action_index),
    }


def compile_search_topology(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
) -> CompiledSearchTopology:
    """Compile validated search semantics into immutable incidence topology."""

    actions = _validated_actions(
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )
    world_ids = tuple(program_set["world_ids"])
    prepared_by_action, join_plan = _prepare_classes(
        actions=actions,
        program_set=program_set,
        world_ids=world_ids,
    )
    edges = _compile_edges(
        actions=actions,
        world_ids=world_ids,
        prepared_by_action=prepared_by_action,
        join_plan=join_plan,
        method=method,
    )
    leaves = _compile_leaves(edges)
    edge_columns = _edge_columns(edges, leaves)
    program_digest = sha256_json(program_set)
    material = _topology_material(
        method=method,
        program_digest=program_digest,
        actions=actions,
        world_ids=world_ids,
        join_plan=join_plan,
        edges=edges,
        edge_columns=edge_columns,
        leaves=leaves,
    )

    return CompiledSearchTopology(
        method=method,
        program_digest=program_digest,
        topology_digest=sha256_json(material),
        outcome_world_join_plan=join_plan,
        root_actions=tuple(actions),
        world_ids=world_ids,
        world_to_class=tuple(tuple(row) for row in edges.world_to_class),
        class_action_index=tuple(edges.class_action_index),
        class_local_index=tuple(edges.class_local_index),
        edge_action_index=tuple(edge_columns.action_index),
        edge_world_index=tuple(edge_columns.world_index),
        edge_class_index=tuple(edge_columns.class_index),
        edge_outcome_index=tuple(edge_columns.outcome_index),
        edge_observation_index=tuple(edge_columns.observation_index),
        edge_successor_index=tuple(edge_columns.successor_index),
        edge_leaf_index=tuple(edge_columns.leaf_index),
        edge_chance=tuple(edge_columns.chance),
        leaf_action_index=tuple(leaves.action_index),
        leaf_observation_index=tuple(leaves.observation_index),
        leaf_successor_index=tuple(leaves.successor_index),
        leaf_conditioned_world_index=tuple(leaves.conditioned_world_index),
        leaf_public_states=tuple(leaves.public_states),
        leaf_legal_actions=tuple(leaves.legal_actions),
        leaf_legal_mask=tuple(leaves.legal_mask),
        observation_keys=tuple(edges.observation_keys),
        successor_states=tuple(edges.successor_states),
        successor_action_vocabulary=leaves.successor_action_vocabulary,
        transition_evaluations=len(edges.class_action_index),
    )

