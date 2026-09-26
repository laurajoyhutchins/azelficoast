from __future__ import annotations

import pytest

from azelficoast.core.planning import LogicalOperator
from azelficoast.core.statistics import CardinalityHistogram, PlannerStatistics


def test_histogram_uses_smoothed_selectivity_and_learns_observed_reduction() -> None:
    histogram = CardinalityHistogram()

    assert histogram.selectivity == pytest.approx(0.5)
    assert histogram.estimate(10) == 5

    histogram.observe(input_rows=10, output_rows=8)

    assert histogram.observations == 1
    assert histogram.input_rows == 10
    assert histogram.output_rows == 8
    assert histogram.selectivity == pytest.approx(9 / 12)
    assert histogram.estimate(10) == 8


def test_histogram_rejects_impossible_cardinality_observation() -> None:
    histogram = CardinalityHistogram()

    with pytest.raises(ValueError, match="cannot exceed"):
        histogram.observe(input_rows=3, output_rows=4)


def test_statistics_are_isolated_by_operator_and_semantic_signature() -> None:
    statistics = PlannerStatistics()
    signature = "sha256:query"

    statistics.observe(
        operator=LogicalOperator.FILTER,
        signature=signature,
        input_rows=100,
        output_rows=20,
    )
    filter_estimate = statistics.estimate(
        operator=LogicalOperator.FILTER,
        signature=signature,
        input_rows=50,
    )
    partition_estimate = statistics.estimate(
        operator=LogicalOperator.PARTITION,
        signature=signature,
        input_rows=50,
    )

    assert filter_estimate.observations == 1
    assert filter_estimate.estimated_output_rows < 50
    assert partition_estimate.observations == 0
    assert partition_estimate.estimated_output_rows == 25


def test_statistics_never_claim_more_output_rows_than_input_rows() -> None:
    statistics = PlannerStatistics()
    for _ in range(10):
        statistics.observe(
            operator=LogicalOperator.FILTER,
            signature="all-pass",
            input_rows=10,
            output_rows=10,
        )

    estimate = statistics.estimate(
        operator=LogicalOperator.FILTER,
        signature="all-pass",
        input_rows=3,
    )
    assert estimate.estimated_output_rows <= 3
