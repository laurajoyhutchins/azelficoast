from __future__ import annotations

import pytest

from azelficoast.core.evaluation import (
    EvaluationContribution,
    EvaluationFrontier,
    EvaluationFrontierError,
    EvaluationLeaf,
)


def test_frontier_reduction_is_backend_independent() -> None:
    leaves = (
        EvaluationLeaf(public_state={"state": 1}, posterior=("a",), legal_actions=("x",)),
        EvaluationLeaf(public_state={"state": 2}, posterior=("b",), legal_actions=("y",)),
        EvaluationLeaf(public_state={"state": 3}, posterior=("c",), legal_actions=("z",)),
    )
    frontier = EvaluationFrontier(
        root_actions=("wait", "attack"),
        leaves=leaves,
        contributions=(
            EvaluationContribution("wait", 0, 0.25),
            EvaluationContribution("wait", 1, 0.75),
            EvaluationContribution("attack", 2, 1.0),
        ),
        transition_evaluations=2,
    )

    assert frontier.reduce((1.0, 0.0, 0.5)) == {
        "wait": 0.25,
        "attack": 0.5,
    }


def test_frontier_rejects_backend_shape_drift() -> None:
    frontier = EvaluationFrontier(
        root_actions=("wait",),
        leaves=(
            EvaluationLeaf(
                public_state={"state": 1},
                posterior=("a",),
                legal_actions=("x",),
            ),
        ),
        contributions=(EvaluationContribution("wait", 0, 1.0),),
        transition_evaluations=1,
    )

    with pytest.raises(EvaluationFrontierError, match="wrong number"):
        frontier.reduce(())
