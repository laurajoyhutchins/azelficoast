from __future__ import annotations

import math

import pytest

from azelficoast.belief_evaluator import (
    BeliefEvaluatorSpec,
    build_evaluator_input,
    init_params,
    loss,
)
from azelficoast.belief_training import TrainingExample, init_adam, train_step


def test_one_adam_step_is_finite_and_updates_state() -> None:
    pytest.importorskip("jax")
    spec = BeliefEvaluatorSpec(
        public_width=12,
        world_width=10,
        action_width=8,
        hidden_width=9,
        world_hidden_width=7,
    )
    inputs = build_evaluator_input(
        public_state={"turn": 7, "tera_unused": True},
        posterior={
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "worlds": [
                {"world_id": "a", "weight": 0.6, "hidden": {"item": "scarf"}},
                {"world_id": "b", "weight": 0.4, "hidden": {"item": "specs"}},
            ],
        },
        legal_actions=("attack", "switch"),
        spec=spec,
    )
    params = init_params(spec, seed=13)
    example = TrainingExample(
        inputs=inputs,
        value_target=0.75,
        policy_target=(0.9, 0.1),
    )
    state = init_adam(params)
    before = float(
        loss(
            params,
            inputs,
            value_target=example.value_target,
            policy_target=example.policy_target,
        )
    )

    updated, next_state, objective = train_step(
        params,
        state,
        example,
        learning_rate=1e-3,
    )

    assert math.isfinite(objective)
    assert next_state.step == 1
    assert set(updated) == set(params)
    assert objective == pytest.approx(before, rel=1e-6)
