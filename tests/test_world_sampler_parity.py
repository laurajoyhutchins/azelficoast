"""Concurrent sampler must reproduce exact serial pinned-Showdown outputs."""

from __future__ import annotations

import sys
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    try:
        from parity_world_sampler import ParallelParityError, verify_equal
        return ParallelParityError, verify_equal
    finally:
        sys.path.pop(0)


def test_identical_positive_and_negative_sampler_evidence() -> None:
    _, verify_equal = _load()
    records = [
        {"status": "sampled", "sha256": "abc", "matched": 12},
        {"status": "unsupported", "reason": "no compatible sets"},
    ]
    verify_equal(records, list(records))


def test_parallel_drift_fails_closed() -> None:
    import pytest

    ParallelParityError, verify_equal = _load()
    with pytest.raises(ParallelParityError, match="drift"):
        verify_equal(
            [{"status": "sampled", "sha256": "a", "matched": 1}],
            [{"status": "sampled", "sha256": "b", "matched": 1}],
        )
    with pytest.raises(ParallelParityError, match="count"):
        verify_equal([{"status": "unsupported"}], [])
