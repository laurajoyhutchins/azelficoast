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

from azelficoast.core.projection import (
    aggregate_projected_weights,
    compile_projection_ids,
    uniform_integer_weights,
)
from azelficoast.gen9_attack import (
    AttackTransitionContext,
    attack_transition_dependency_key,
    attack_transition_dependency_signature,
)
from azelficoast.gen9_damage import (
    DamageContext,
    damage,
    damage_dependency_tuple,
)


@dataclass(frozen=True)
class CanonicalSupport:
    context_index: np.ndarray
    bench_signature: np.ndarray
    accuracy_roll: np.ndarray
    roll: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.context_index)
        if (
            len(self.bench_signature) != size
            or len(self.accuracy_roll) != size
            or len(self.roll) != size
        ):
            raise ValueError("canonical support columns must have equal length")
        if self.context_index.dtype.kind not in "iu":
            raise ValueError("context_index must be integer")
        if self.bench_signature.dtype.kind not in "iu":
            raise ValueError("bench_signature must be integer")
        if self.accuracy_roll.dtype.kind not in "iu":
            raise ValueError("accuracy_roll must be integer")
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
    effect_signature: str | None = None

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
    accuracy_rolls: int = 1,
) -> CanonicalSupport:
    if context_count <= 0 or bench_variants <= 0 or rolls <= 0 or accuracy_rolls <= 0:
        raise ValueError("support dimensions must be positive")
    context: list[int] = []
    bench: list[int] = []
    accuracy: list[int] = []
    roll: list[int] = []
    for context_index in range(context_count):
        for bench_signature in range(bench_variants):
            for accuracy_roll in range(accuracy_rolls):
                for damage_roll in range(rolls):
                    context.append(context_index)
                    bench.append(bench_signature)
                    accuracy.append(accuracy_roll)
                    roll.append(damage_roll)
    return CanonicalSupport(
        context_index=np.asarray(context, dtype=np.int32),
        bench_signature=np.asarray(bench, dtype=np.int32),
        accuracy_roll=np.asarray(accuracy, dtype=np.int32),
        roll=np.asarray(roll, dtype=np.int32),
    )


def uniform_belief(
    support: CanonicalSupport,
    logical_world_count: int,
) -> ClassNativeBelief:
    return ClassNativeBelief(
        support=support,
        weights=uniform_integer_weights(support.class_count, logical_world_count),
    )


def _compile_ids(
    support: CanonicalSupport,
    key_at: Callable[[int], tuple[int, ...]],
    *,
    name: str,
) -> ProjectionMap:
    class_ids, representatives = compile_projection_ids(support.class_count, key_at)
    return ProjectionMap(
        name=name,
        class_ids=class_ids,
        representative_indices=representatives,
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



def compile_attack_projection(
    support: CanonicalSupport,
    contexts: Sequence[AttackTransitionContext],
    *,
    include_attack_modifier: bool = True,
) -> ProjectionMap:
    projection = _compile_ids(
        support,
        lambda index: attack_transition_dependency_key(
            contexts[int(support.context_index[index])],
            int(support.accuracy_roll[index]),
            int(support.roll[index]),
            include_attack_modifier=include_attack_modifier,
        ),
        name=(
            "attack-transition"
            if include_attack_modifier
            else "attack-transition-missing-attack-modifier"
        ),
    )
    return ProjectionMap(
        name=projection.name,
        class_ids=projection.class_ids,
        representative_indices=projection.representative_indices,
        effect_signature=(
            attack_transition_dependency_signature()
            if include_attack_modifier
            else None
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
    weights = aggregate_projected_weights(
        belief.weights,
        projection.class_ids,
        projection.class_count,
        support_class_count=belief.support.class_count,
        mismatch_message="projection does not match belief support",
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
