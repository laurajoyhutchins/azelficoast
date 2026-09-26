from __future__ import annotations

import numpy as np
import pytest

from azelficoast.core.projection import (
    aggregate_projected_weights,
    compact_active_projection,
)


def test_compaction_pushes_zero_mass_filter_before_projection_aggregation() -> None:
    support_count = 10_000
    class_ids = (np.arange(support_count, dtype=np.int32) % 1_000).astype(
        np.int32,
        copy=False,
    )
    representatives = np.arange(1_000, dtype=np.int32)
    weights = np.zeros(support_count, dtype=np.int64)
    active_indices = np.asarray([7, 1007, 9999], dtype=np.int64)
    weights[active_indices] = np.asarray([2, 3, 5], dtype=np.int64)

    full = aggregate_projected_weights(
        weights,
        class_ids,
        len(representatives),
        support_class_count=support_count,
        mismatch_message="projection mismatch",
    )
    compact = compact_active_projection(
        weights,
        class_ids,
        representatives,
        support_class_count=support_count,
        mismatch_message="projection mismatch",
    )

    full_active = np.flatnonzero(full > 0)
    assert np.array_equal(compact.active_support_indices, active_indices)
    assert np.array_equal(compact.global_class_ids, full_active)
    assert np.array_equal(compact.weights, full[full_active])
    assert np.array_equal(
        compact.representative_indices,
        representatives[full_active],
    )
    assert compact.active_support_count == 3
    assert compact.active_class_count == len(full_active)
    assert compact.filtered_support_count == support_count - 3
    assert compact.logical_world_count == 10


def test_compaction_preserves_global_projection_identity_not_active_row_identity() -> None:
    weights = np.asarray([0, 4, 0, 6], dtype=np.int64)
    class_ids = np.asarray([0, 0, 1, 1], dtype=np.int32)
    representatives = np.asarray([0, 2], dtype=np.int32)

    compact = compact_active_projection(
        weights,
        class_ids,
        representatives,
        support_class_count=4,
        mismatch_message="projection mismatch",
    )

    assert compact.global_class_ids.tolist() == [0, 1]
    assert compact.representative_indices.tolist() == [0, 2]
    assert compact.weights.tolist() == [4, 6]


def test_compaction_fails_closed_on_unbound_projection_class() -> None:
    with pytest.raises(ValueError, match="no representative"):
        compact_active_projection(
            np.asarray([1, 1], dtype=np.int64),
            np.asarray([0, 2], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            support_class_count=2,
            mismatch_message="projection mismatch",
        )


def test_empty_active_support_stays_empty_without_inventing_a_class() -> None:
    compact = compact_active_projection(
        np.zeros(4, dtype=np.int64),
        np.asarray([0, 0, 1, 1], dtype=np.int32),
        np.asarray([0, 2], dtype=np.int32),
        support_class_count=4,
        mismatch_message="projection mismatch",
    )

    assert compact.active_support_count == 0
    assert compact.active_class_count == 0
    assert compact.logical_world_count == 0
