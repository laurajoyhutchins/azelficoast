from __future__ import annotations

import pytest

pytest.importorskip("jax")

from azelficoast.research.jax_simulator_experiment import benchmark_size, correctness_check


def test_jax_lowering_matches_scalar_reference() -> None:
    result = correctness_check()

    assert result == {
        "world_count": 32,
        "protect_exact": True,
        "damage_exact": True,
    }


def test_reduced_jax_execution_preserves_weighted_results() -> None:
    result = benchmark_size(2048, repeats=3)

    assert result["protect"]["classes"] == 1
    assert result["protect"]["score_equal"] is True
    assert result["damage"]["classes"] == 2
    assert result["damage"]["score_equal"] is True
