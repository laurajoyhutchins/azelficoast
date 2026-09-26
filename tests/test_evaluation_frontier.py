from __future__ import annotations

import pytest

from azelficoast.core.evaluation import (
    EvaluationContribution,
    EvaluationFrontier,
    EvaluationFrontierError,
    EvaluationLeaf,
    choose_bounded_action,
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



def test_bounded_winner_prunes_only_certifiably_dominated_leaves() -> None:
    leaf_values = {
        0: 0.9,
        1: 0.9,
        2: -0.9,
        3: -0.9,
        4: -0.9,
        5: -0.8,
        6: -0.8,
    }
    calls: list[tuple[int, ...]] = []

    def evaluate(indices: tuple[int, ...]) -> tuple[float, ...]:
        calls.append(indices)
        return tuple(leaf_values[index] for index in indices)

    decision = choose_bounded_action(
        root_actions=("alpha", "beta", "gamma"),
        leaf_action_index=(0, 0, 1, 1, 1, 2, 2),
        coefficients=(0.5, 0.5, 0.7, 0.2, 0.1, 0.6, 0.4),
        evaluate=evaluate,
        value_lower_bound=-1.0,
        value_upper_bound=1.0,
    )

    assert decision.chosen_action == "alpha"
    assert decision.chosen_value == pytest.approx(0.9)
    assert decision.evaluated_leaf_count == 4
    assert decision.total_leaf_count == 7
    assert decision.pruned_leaf_count == 3
    assert decision.pruned_actions == ("beta", "gamma")
    assert dict(decision.exact_root_values) == {"alpha": pytest.approx(0.9)}
    intervals = {row.action: row for row in decision.intervals}
    assert intervals["alpha"].exact is True
    assert intervals["beta"].exact is False
    assert intervals["beta"].upper < decision.chosen_value
    assert intervals["gamma"].exact is False
    assert intervals["gamma"].upper < decision.chosen_value
    assert {index for call in calls for index in call} == {0, 1, 2, 5}


def test_bounded_winner_evaluates_overlapping_action_bounds_to_exact_tie() -> None:
    values = {0: 0.5, 1: 0.5, 2: 0.5}

    decision = choose_bounded_action(
        root_actions=("alpha", "beta"),
        leaf_action_index=(0, 1, 1),
        coefficients=(1.0, 0.9, 0.1),
        evaluate=lambda indices: tuple(values[index] for index in indices),
        value_lower_bound=-1.0,
        value_upper_bound=1.0,
    )

    assert decision.chosen_action == "alpha"
    assert decision.evaluated_leaf_count == decision.total_leaf_count == 3
    assert decision.pruned_actions == ()
    assert dict(decision.exact_root_values) == pytest.approx(
        {"alpha": 0.5, "beta": 0.5}
    )


def test_bounded_winner_rejects_value_outside_certified_range() -> None:
    with pytest.raises(EvaluationFrontierError, match="outside its certified range"):
        choose_bounded_action(
            root_actions=("alpha",),
            leaf_action_index=(0,),
            coefficients=(1.0,),
            evaluate=lambda indices: (1.01,),
            value_lower_bound=-1.0,
            value_upper_bound=1.0,
        )
