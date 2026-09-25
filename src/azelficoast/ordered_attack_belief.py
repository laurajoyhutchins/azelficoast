"""Class-native support specialized to the ordered-attack transition.

This module deliberately does not widen the canonical support used by the earlier
damage and whole-attack experiments. New RNG axes live only where their semantics
are actually needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from azelficoast.belief_projection import (
    aggregate_projected_weights,
    compile_projection_ids,
    uniform_integer_weights,
)
from azelficoast.gen9_ordered_attack import (
    OrderedAttackContext,
    ordered_attack_dependency_key,
    ordered_attack_dependency_signature,
)


@dataclass(frozen=True)
class OrderedAttackSupport:
    context_index: np.ndarray
    bench_signature: np.ndarray
    order_tie_roll: np.ndarray
    accuracy_roll: np.ndarray
    damage_roll: np.ndarray
    secondary_roll: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.context_index)
        columns = (
            self.bench_signature,
            self.order_tie_roll,
            self.accuracy_roll,
            self.damage_roll,
            self.secondary_roll,
        )
        if any(len(column) != size for column in columns):
            raise ValueError("ordered-attack support columns must have equal length")
        if any(column.dtype.kind not in "iu" for column in (self.context_index, *columns)):
            raise ValueError("ordered-attack support columns must be integer")

    @property
    def class_count(self) -> int:
        return len(self.context_index)


@dataclass(frozen=True)
class OrderedAttackBelief:
    support: OrderedAttackSupport
    weights: np.ndarray

    def __post_init__(self) -> None:
        if len(self.weights) != self.support.class_count:
            raise ValueError("belief weights must match ordered-attack support")
        if self.weights.dtype.kind not in "iu":
            raise ValueError("belief weights must be integer")
        if np.any(self.weights < 0):
            raise ValueError("belief weights must be non-negative")

    @property
    def logical_world_count(self) -> int:
        return int(self.weights.sum(dtype=np.int64))

    @property
    def active_canonical_classes(self) -> int:
        return int(np.count_nonzero(self.weights))


@dataclass(frozen=True)
class OrderedAttackProjection:
    name: str
    class_ids: np.ndarray
    representative_indices: np.ndarray
    effect_signature: str | None

    @property
    def class_count(self) -> int:
        return len(self.representative_indices)


@dataclass(frozen=True)
class ProjectedOrderedAttackBelief:
    projection: OrderedAttackProjection
    weights: np.ndarray

    @property
    def logical_world_count(self) -> int:
        return int(self.weights.sum(dtype=np.int64))

    @property
    def active_classes(self) -> int:
        return int(np.count_nonzero(self.weights))


def build_ordered_attack_support(
    context_count: int,
    *,
    bench_variants: int,
    order_tie_rolls: int = 2,
    accuracy_rolls: int = 100,
    damage_rolls: int = 16,
    secondary_rolls: int = 100,
) -> OrderedAttackSupport:
    dimensions = (
        context_count,
        bench_variants,
        order_tie_rolls,
        accuracy_rolls,
        damage_rolls,
        secondary_rolls,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("ordered-attack support dimensions must be positive")

    context: list[int] = []
    bench: list[int] = []
    order: list[int] = []
    accuracy: list[int] = []
    damage: list[int] = []
    secondary: list[int] = []
    for context_index in range(context_count):
        for bench_signature in range(bench_variants):
            for order_roll in range(order_tie_rolls):
                for accuracy_roll in range(accuracy_rolls):
                    for damage_roll in range(damage_rolls):
                        for secondary_roll in range(secondary_rolls):
                            context.append(context_index)
                            bench.append(bench_signature)
                            order.append(order_roll)
                            accuracy.append(accuracy_roll)
                            damage.append(damage_roll)
                            secondary.append(secondary_roll)
    return OrderedAttackSupport(
        context_index=np.asarray(context, dtype=np.int32),
        bench_signature=np.asarray(bench, dtype=np.int32),
        order_tie_roll=np.asarray(order, dtype=np.int32),
        accuracy_roll=np.asarray(accuracy, dtype=np.int32),
        damage_roll=np.asarray(damage, dtype=np.int32),
        secondary_roll=np.asarray(secondary, dtype=np.int32),
    )


def uniform_ordered_attack_belief(
    support: OrderedAttackSupport,
    logical_world_count: int,
) -> OrderedAttackBelief:
    return OrderedAttackBelief(
        support=support,
        weights=uniform_integer_weights(support.class_count, logical_world_count),
    )


def _compile_ids(
    support: OrderedAttackSupport,
    key_at: Callable[[int], tuple[int, ...]],
    *,
    name: str,
    effect_signature: str | None,
) -> OrderedAttackProjection:
    class_ids, representatives = compile_projection_ids(support.class_count, key_at)
    return OrderedAttackProjection(
        name=name,
        class_ids=class_ids,
        representative_indices=representatives,
        effect_signature=effect_signature,
    )


def compile_ordered_attack_projection(
    support: OrderedAttackSupport,
    contexts: Sequence[OrderedAttackContext],
    *,
    include_order_tie_roll: bool = True,
    include_secondary_roll: bool = True,
) -> OrderedAttackProjection:
    complete = include_order_tie_roll and include_secondary_roll
    return _compile_ids(
        support,
        lambda index: ordered_attack_dependency_key(
            contexts[int(support.context_index[index])],
            order_tie_roll=int(support.order_tie_roll[index]),
            accuracy_roll=int(support.accuracy_roll[index]),
            damage_roll=int(support.damage_roll[index]),
            secondary_roll=int(support.secondary_roll[index]),
            include_order_tie_roll=include_order_tie_roll,
            include_secondary_roll=include_secondary_roll,
        ),
        name="ordered-attack-transition" if complete else "ordered-attack-incomplete",
        effect_signature=ordered_attack_dependency_signature() if complete else None,
    )


def project_ordered_attack_belief(
    belief: OrderedAttackBelief,
    projection: OrderedAttackProjection,
) -> ProjectedOrderedAttackBelief:
    weights = aggregate_projected_weights(
        belief.weights,
        projection.class_ids,
        projection.class_count,
        support_class_count=belief.support.class_count,
        mismatch_message="projection does not match ordered-attack support",
    )
    return ProjectedOrderedAttackBelief(projection=projection, weights=weights)
