"""Class-native finite support for the bounded two-attack turn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from azelficoast.gen9_two_attack_turn import (
    TwoAttackTurnContext,
    two_attack_turn_dependency_key,
    two_attack_turn_dependency_signature,
)


@dataclass(frozen=True)
class TwoAttackTurnSupport:
    context_index: np.ndarray
    bench_signature: np.ndarray
    order_tie_roll: np.ndarray
    p1_damage_roll: np.ndarray
    p1_secondary_roll: np.ndarray
    p2_damage_roll: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.context_index)
        columns = (
            self.bench_signature,
            self.order_tie_roll,
            self.p1_damage_roll,
            self.p1_secondary_roll,
            self.p2_damage_roll,
        )
        if any(len(column) != size for column in columns):
            raise ValueError("two-attack support columns must have equal length")
        if any(column.dtype.kind not in "iu" for column in (self.context_index, *columns)):
            raise ValueError("two-attack support columns must be integer")

    @property
    def class_count(self) -> int:
        return len(self.context_index)


@dataclass(frozen=True)
class TwoAttackTurnBelief:
    support: TwoAttackTurnSupport
    weights: np.ndarray

    def __post_init__(self) -> None:
        if len(self.weights) != self.support.class_count:
            raise ValueError("belief weights must match two-attack support")
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
class TwoAttackTurnProjection:
    name: str
    class_ids: np.ndarray
    representative_indices: np.ndarray
    effect_signature: str | None

    @property
    def class_count(self) -> int:
        return len(self.representative_indices)


@dataclass(frozen=True)
class ProjectedTwoAttackTurnBelief:
    projection: TwoAttackTurnProjection
    weights: np.ndarray

    @property
    def logical_world_count(self) -> int:
        return int(self.weights.sum(dtype=np.int64))

    @property
    def active_classes(self) -> int:
        return int(np.count_nonzero(self.weights))


def build_two_attack_turn_support(
    context_count: int,
    *,
    bench_variants: int,
    order_tie_rolls: int = 2,
    p1_damage_rolls: int = 16,
    p1_secondary_rolls: int = 31,
    p2_damage_rolls: int = 16,
) -> TwoAttackTurnSupport:
    dimensions = (
        context_count,
        bench_variants,
        order_tie_rolls,
        p1_damage_rolls,
        p1_secondary_rolls,
        p2_damage_rolls,
    )
    if any(value <= 0 for value in dimensions):
        raise ValueError("two-attack support dimensions must be positive")

    context: list[int] = []
    bench: list[int] = []
    order: list[int] = []
    p1_damage: list[int] = []
    p1_secondary: list[int] = []
    p2_damage: list[int] = []
    for context_index in range(context_count):
        for bench_signature in range(bench_variants):
            for order_roll in range(order_tie_rolls):
                for first_damage_roll in range(p1_damage_rolls):
                    for secondary_roll in range(p1_secondary_rolls):
                        for second_damage_roll in range(p2_damage_rolls):
                            context.append(context_index)
                            bench.append(bench_signature)
                            order.append(order_roll)
                            p1_damage.append(first_damage_roll)
                            p1_secondary.append(secondary_roll)
                            p2_damage.append(second_damage_roll)

    return TwoAttackTurnSupport(
        context_index=np.asarray(context, dtype=np.int32),
        bench_signature=np.asarray(bench, dtype=np.int32),
        order_tie_roll=np.asarray(order, dtype=np.int32),
        p1_damage_roll=np.asarray(p1_damage, dtype=np.int32),
        p1_secondary_roll=np.asarray(p1_secondary, dtype=np.int32),
        p2_damage_roll=np.asarray(p2_damage, dtype=np.int32),
    )


def uniform_two_attack_turn_belief(
    support: TwoAttackTurnSupport,
    logical_world_count: int,
) -> TwoAttackTurnBelief:
    if logical_world_count < support.class_count:
        raise ValueError("logical world count must cover every canonical support class")
    quotient, remainder = divmod(logical_world_count, support.class_count)
    weights = np.full(support.class_count, quotient, dtype=np.int64)
    if remainder:
        weights[:remainder] += 1
    return TwoAttackTurnBelief(support=support, weights=weights)


def _compile_ids(
    support: TwoAttackTurnSupport,
    key_at: Callable[[int], tuple[int, ...]],
    *,
    name: str,
    effect_signature: str | None,
) -> TwoAttackTurnProjection:
    ids = np.empty(support.class_count, dtype=np.int32)
    representatives: list[int] = []
    class_by_key: dict[tuple[int, ...], int] = {}
    for index in range(support.class_count):
        key = key_at(index)
        class_id = class_by_key.get(key)
        if class_id is None:
            class_id = len(representatives)
            class_by_key[key] = class_id
            representatives.append(index)
        ids[index] = class_id

    return TwoAttackTurnProjection(
        name=name,
        class_ids=ids,
        representative_indices=np.asarray(representatives, dtype=np.int32),
        effect_signature=effect_signature,
    )


def compile_two_attack_turn_projection(
    support: TwoAttackTurnSupport,
    contexts: Sequence[TwoAttackTurnContext],
    *,
    include_order_tie: bool = True,
    include_secondary: bool = True,
) -> TwoAttackTurnProjection:
    complete = include_order_tie and include_secondary
    return _compile_ids(
        support,
        lambda index: two_attack_turn_dependency_key(
            contexts[int(support.context_index[index])],
            order_tie_roll=int(support.order_tie_roll[index]),
            p1_accuracy_roll=0,
            p1_damage_roll=int(support.p1_damage_roll[index]),
            p1_secondary_roll=int(support.p1_secondary_roll[index]),
            p2_accuracy_roll=0,
            p2_damage_roll=int(support.p2_damage_roll[index]),
            include_order_tie=include_order_tie,
            include_secondary=include_secondary,
        ),
        name="two-attack-turn" if complete else "two-attack-turn-incomplete",
        effect_signature=two_attack_turn_dependency_signature() if complete else None,
    )


def project_two_attack_turn_belief(
    belief: TwoAttackTurnBelief,
    projection: TwoAttackTurnProjection,
) -> ProjectedTwoAttackTurnBelief:
    if len(projection.class_ids) != belief.support.class_count:
        raise ValueError("projection does not match two-attack support")
    weights = np.zeros(projection.class_count, dtype=np.int64)
    active = belief.weights > 0
    np.add.at(weights, projection.class_ids[active], belief.weights[active])
    return ProjectedTwoAttackTurnBelief(projection=projection, weights=weights)
