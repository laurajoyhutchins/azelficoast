"""Integer-weight projection machinery for finite hidden-world beliefs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ActiveProjection:
    """Active slice of a verified projection after zero-mass support is filtered."""

    support_class_count: int
    active_support_indices: np.ndarray
    global_class_ids: np.ndarray
    representative_indices: np.ndarray
    weights: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            self.active_support_indices,
            self.global_class_ids,
            self.representative_indices,
            self.weights,
        )
        if self.support_class_count <= 0:
            raise ValueError("support class count must be positive")
        if any(array.ndim != 1 for array in arrays):
            raise ValueError("active projection arrays must be one-dimensional")
        if any(array.dtype.kind not in "iu" for array in arrays):
            raise ValueError("active projection arrays must be integer")
        count = len(self.global_class_ids)
        if (
            len(self.representative_indices) != count
            or len(self.weights) != count
        ):
            raise ValueError("active projection class arrays must have equal length")
        if len(self.active_support_indices) > self.support_class_count:
            raise ValueError("active support cannot exceed canonical support")
        if np.any(self.weights <= 0):
            raise ValueError("active projection weights must be positive")

    @property
    def active_support_count(self) -> int:
        return len(self.active_support_indices)

    @property
    def active_class_count(self) -> int:
        return len(self.global_class_ids)

    @property
    def logical_world_count(self) -> int:
        return int(self.weights.sum(dtype=np.int64))

    @property
    def filtered_support_count(self) -> int:
        return self.support_class_count - self.active_support_count


def uniform_integer_weights(
    class_count: int,
    logical_world_count: int,
) -> np.ndarray:
    if class_count <= 0:
        raise ValueError("class count must be positive")
    if logical_world_count < class_count:
        raise ValueError("logical world count must cover every canonical support class")
    quotient, remainder = divmod(logical_world_count, class_count)
    weights = np.full(class_count, quotient, dtype=np.int64)
    if remainder:
        weights[:remainder] += 1
    return weights


def compile_projection_ids(
    class_count: int,
    key_at: Callable[[int], tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    if class_count <= 0:
        raise ValueError("class count must be positive")
    ids = np.empty(class_count, dtype=np.int32)
    representatives: list[int] = []
    class_by_key: dict[tuple[int, ...], int] = {}

    for index in range(class_count):
        key = key_at(index)
        class_id = class_by_key.get(key)
        if class_id is None:
            class_id = len(representatives)
            class_by_key[key] = class_id
            representatives.append(index)
        ids[index] = class_id

    return ids, np.asarray(representatives, dtype=np.int32)


def compact_active_projection(
    weights: np.ndarray,
    class_ids: np.ndarray,
    representative_indices: np.ndarray,
    *,
    support_class_count: int,
    mismatch_message: str,
) -> ActiveProjection:
    """Push zero-mass filtering ahead of projection aggregation.

    The supplied projection map remains authoritative. This function never derives new
    equivalence classes: it selects the already-verified global class IDs touched by the
    active posterior, aggregates only those rows, and preserves each class's canonical
    representative.
    """

    if support_class_count <= 0:
        raise ValueError("support class count must be positive")
    if weights.ndim != 1 or weights.dtype.kind not in "iu":
        raise ValueError("weights must be a one-dimensional integer array")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    if class_ids.ndim != 1 or class_ids.dtype.kind not in "iu":
        raise ValueError("projection class IDs must be a one-dimensional integer array")
    if (
        representative_indices.ndim != 1
        or representative_indices.dtype.kind not in "iu"
    ):
        raise ValueError(
            "projection representatives must be a one-dimensional integer array"
        )
    if len(weights) != support_class_count:
        raise ValueError("weight count must match support class count")
    if len(class_ids) != support_class_count:
        raise ValueError(mismatch_message)
    if len(class_ids):
        if int(class_ids.min()) < 0:
            raise ValueError("projection class IDs must be non-negative")
        if not len(representative_indices):
            raise ValueError("projection has no representatives")
        if int(class_ids.max()) >= len(representative_indices):
            raise ValueError("projection class ID has no representative")
    if len(representative_indices):
        if int(representative_indices.min()) < 0:
            raise ValueError("projection representatives must be non-negative")
        if int(representative_indices.max()) >= support_class_count:
            raise ValueError("projection representative is outside support")

    active_support_indices = np.flatnonzero(weights > 0).astype(
        np.int64,
        copy=False,
    )
    if not len(active_support_indices):
        empty = np.asarray([], dtype=np.int64)
        return ActiveProjection(
            support_class_count=support_class_count,
            active_support_indices=active_support_indices,
            global_class_ids=empty,
            representative_indices=empty,
            weights=empty,
        )

    touched_class_ids, inverse = np.unique(
        class_ids[active_support_indices],
        return_inverse=True,
    )
    compact_weights = np.zeros(len(touched_class_ids), dtype=np.int64)
    np.add.at(
        compact_weights,
        inverse,
        weights[active_support_indices].astype(np.int64, copy=False),
    )
    representatives = representative_indices[touched_class_ids].astype(
        np.int64,
        copy=False,
    )

    return ActiveProjection(
        support_class_count=support_class_count,
        active_support_indices=active_support_indices,
        global_class_ids=touched_class_ids.astype(np.int64, copy=False),
        representative_indices=representatives,
        weights=compact_weights,
    )


def aggregate_projected_weights(
    weights: np.ndarray,
    class_ids: np.ndarray,
    projection_class_count: int,
    *,
    support_class_count: int,
    mismatch_message: str,
) -> np.ndarray:
    if projection_class_count <= 0:
        raise ValueError("projection class count must be positive")
    if len(weights) != support_class_count:
        raise ValueError("weight count must match support class count")
    if len(class_ids) != support_class_count:
        raise ValueError(mismatch_message)
    projected = np.zeros(projection_class_count, dtype=np.int64)
    active = weights > 0
    np.add.at(projected, class_ids[active], weights[active])
    return projected
