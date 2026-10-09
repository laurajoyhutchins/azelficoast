"""Concurrent sampler must reproduce exact serial pinned-Showdown outputs."""

from __future__ import annotations

import sys
from pathlib import Path


def _load():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    try:
        from parity_world_sampler import (
            ParallelParityError,
            verify_complete_document,
            verify_equal,
        )
        return ParallelParityError, verify_equal, verify_complete_document
    finally:
        sys.path.pop(0)


def test_identical_positive_and_negative_sampler_evidence() -> None:
    _, verify_equal, _ = _load()
    records = [
        {"status": "sampled", "sha256": "abc", "matched": 12},
        {"status": "unsupported", "reason": "no compatible sets"},
    ]
    verify_equal(records, list(records))


def test_parallel_drift_fails_closed() -> None:
    import pytest

    ParallelParityError, verify_equal, _ = _load()
    with pytest.raises(ParallelParityError, match="drift"):
        verify_equal(
            [{"status": "sampled", "sha256": "a", "matched": 1}],
            [{"status": "sampled", "sha256": "b", "matched": 1}],
        )
    with pytest.raises(ParallelParityError, match="count"):
        verify_equal([{"status": "unsupported"}], [])



def test_complete_candidate_document_parity_is_exact() -> None:
    import pytest

    ParallelParityError, _, verify_complete_document = _load()
    document = {
        "schema": "azelficoast.natural-fusion-candidates",
        "candidate_count": 1,
        "candidates": [{"fixture_id": "A", "item_counts": {"Scarf": 3}}],
        "excluded_fixtures": [{"fixture_id": "B", "reason": "unobserved"}],
    }
    assert len(verify_complete_document(document, dict(document))) == 64
    with pytest.raises(ParallelParityError, match="candidate/exclusion"):
        verify_complete_document(
            document,
            {**document, "excluded_fixtures": [{"fixture_id": "B", "reason": "changed"}]},
        )
