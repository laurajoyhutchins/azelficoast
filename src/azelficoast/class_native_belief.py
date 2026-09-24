"""Class-native finite beliefs with compiled dependency projections.

The mutable belief carries integer multiplicities over canonical semantic support classes,
not one row per logical hidden world. Effect-specific execution partitions are compiled
from that support and can coarsen or refine it without rediscovering classes from raw
particles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from azelficoast.gen9_damage import (
    COMPILED_ATTACK_MOD_COLUMN,
    COMPILED_CATEGORY_COLUMN,
    DamageContext,
    compile_numeric_context,
    damage,
)


@dataclass(frozen=True)
class CanonicalSupport:
    context_index: np.ndarray
    bench_signature: np.ndarray
    roll: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.context_index)
        if len(self.bench_signature) != size or len(self.roll) != size:
            raise ValueError("canonical support columns must have equal length")
        if self.context_index.dtype.kind not in "iu":
            raise ValueError("context_index must be integer")
        if self.bench_signature.dtype.kind not in "iu":
            raise ValueError("bench_signature must be integer")
        if self.roll.dtype.kind not in "iu":
            raise ValueError("roll must be integer")

    @property
    def class_count(self) -> int:
        return len(self.context_index)


@dataclass(frozen=True)
class ClassNativeBelief:
    support: CanonicalSupport
    weights: np.ndarray

    def __post_init__(self) -> None:
        if len(self.weights) != self.support.class_count:
            raise ValueError("belief weights must match canonical support")
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
class ProjectionMap:
    name: str
    class_ids: np.ndarray
    representative_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.class_ids.dtype.kind not in "iu":
            raise ValueError("projection class_ids must be integer")
        if self.representative_indices.dtype.kind not in "iu":
            raise ValueError("projection representatives must be integer")
        if len(self.class_ids) and int(self.class_ids.min()) < 0:
            raise ValueError("projection class IDs must be non-negative")

    @property
    def class_count(self) -> int:
        return len(self.representative_indices)


@dataclass(frozen=True)
class ProjectedBelief:
    projection: ProjectionMap
    weights: np.ndarray

    @property
    def logical_world_count(self) -> int:
        return int(self.weights.sum(dtype=np.int64))

    @property
    def active_classes(self) -> int:
        return int(np.count_nonzero(self.weights))


def build_factor_support(
    context_count: int,
    *,
    bench_variants: int,
    rolls: int = 16,
) -> CanonicalSupport:
    if context_count <= 0 or bench_variants <= 0 or rolls <= 0:
        raise ValueError("support dimensions must be positive")
    context: list[int] = []
    bench: list[int] = []
    roll: list[int] = []
    for context_index in range(context_count):
        for bench_signature in range(bench_variants):
            for damage_roll in range(rolls):
                context.append(context_index)
                bench.append(bench_signature)
                roll.append(damage_roll)
    return CanonicalSupport(
        context_index=np.asarray(context, dtype=np.int32),
        bench_signature=np.asarray(bench, dtype=np.int32),
        roll=np.asarray(roll, dtype=np.int32),
    )


def uniform_belief(
    support: CanonicalSupport,
    logical_world_count: int,
) -> ClassNativeBelief:
    if logical_world_count < support.class_count:
        raise ValueError("logical world count must cover every canonical support class")
    quotient, remainder = divmod(logical_world_count, support.class_count)
    weights = np.full(support.class_count, quotient, dtype=np.int64)
    if remainder:
        weights[:remainder] += 1
    return ClassNativeBelief(support=support, weights=weights)


def _compile_ids(
    support: CanonicalSupport,
    key_at: Callable[[int], tuple[int, ...]],
    *,
    name: str,
) -> ProjectionMap:
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
    return ProjectionMap(
        name=name,
        class_ids=ids,
        representative_indices=np.asarray(representatives, dtype=np.int32),
    )


def compile_protect_projection(support: CanonicalSupport) -> ProjectionMap:
    return ProjectionMap(
        name="protect",
        class_ids=np.zeros(support.class_count, dtype=np.int32),
        representative_indices=np.asarray([0], dtype=np.int32),
    )


def compile_bench_projection(support: CanonicalSupport) -> ProjectionMap:
    return _compile_ids(
        support,
        lambda index: (int(support.bench_signature[index]),),
        name="bench",
    )


def damage_dependency_tuple(
    context: DamageContext,
    *,
    include_attack_modifier: bool = True,
) -> tuple[int, ...]:
    compiled = compile_numeric_context(context)
    # CATEGORY is metadata after semantic modifiers are compiled and is not read by
    # the numeric JAX damage kernel. The negative control optionally removes the
    # Choice Band/Specs-derived attack modifier.
    return tuple(
        value
        for column, value in enumerate(compiled)
        if column != COMPILED_CATEGORY_COLUMN and (include_attack_modifier or column != COMPILED_ATTACK_MOD_COLUMN)
    )


def compile_damage_projection(
    support: CanonicalSupport,
    contexts: Sequence[DamageContext],
    *,
    include_attack_modifier: bool = True,
) -> ProjectionMap:
    dependencies = tuple(
        damage_dependency_tuple(
            context,
            include_attack_modifier=include_attack_modifier,
        )
        for context in contexts
    )
    return _compile_ids(
        support,
        lambda index: (
            *dependencies[int(support.context_index[index])],
            int(support.roll[index]),
        ),
        name=(
            "damage"
            if include_attack_modifier
            else "damage-missing-attack-modifier"
        ),
    )


def compile_full_projection(support: CanonicalSupport) -> ProjectionMap:
    return ProjectionMap(
        name="full",
        class_ids=np.arange(support.class_count, dtype=np.int32),
        representative_indices=np.arange(support.class_count, dtype=np.int32),
    )


def project_belief(
    belief: ClassNativeBelief,
    projection: ProjectionMap,
) -> ProjectedBelief:
    if len(projection.class_ids) != belief.support.class_count:
        raise ValueError("projection does not match belief support")
    weights = np.zeros(projection.class_count, dtype=np.int64)
    active = belief.weights > 0
    np.add.at(
        weights,
        projection.class_ids[active],
        belief.weights[active],
    )
    return ProjectedBelief(projection=projection, weights=weights)


def filter_exact_damage_observation(
    belief: ClassNativeBelief,
    contexts: Sequence[DamageContext],
    observed_damage: int,
) -> ClassNativeBelief:
    keep = np.fromiter(
        (
            damage(
                contexts[int(context_index)],
                int(roll),
            ) == observed_damage
            for context_index, roll in zip(
                belief.support.context_index,
                belief.support.roll,
                strict=True,
            )
        ),
        dtype=np.bool_,
        count=belief.support.class_count,
    )
    return ClassNativeBelief(
        support=belief.support,
        weights=np.where(keep, belief.weights, 0).astype(np.int64, copy=False),
    )


def materialize_projection_ids(
    belief: ClassNativeBelief,
    projection: ProjectionMap,
) -> np.ndarray:
    """Reference-only expansion used to verify compressed projection semantics."""
    return np.repeat(projection.class_ids, belief.weights.astype(np.int64, copy=False))
