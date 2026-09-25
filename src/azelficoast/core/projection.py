"""Integer-weight projection machinery for finite hidden-world beliefs."""

from __future__ import annotations

from typing import Callable

import numpy as np


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
