from __future__ import annotations

import math

import pytest

from azelficoast.belief_evaluator import (
    BeliefEvaluatorError,
    BeliefEvaluatorSpec,
    build_evaluator_input,
    evaluator_identity,
    forward,
    init_params,
    loss,
)


def _posterior() -> dict[str, object]:
    return {
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {
                "world_id": "scarf-world",
                "weight": 0.4,
                "opponent": {
                    "species": "Rotom-Wash",
                    "item": "choicescarf",
                    "moves": ["hydropump", "voltswitch"],
                },
            },
            {
                "world_id": "leftovers-world",
                "weight": 0.6,
                "opponent": {
                    "species": "Rotom-Wash",
                    "item": "leftovers",
                    "moves": ["hydropump", "willowisp"],
                },
            },
        ],
    }


def test_input_is_public_belief_not_realized_hidden_state() -> None:
    posterior = _posterior()
    posterior["realized_hidden_state_revealed"] = True

    with pytest.raises(BeliefEvaluatorError, match="revealed hidden state"):
        build_evaluator_input(
            public_state={"turn": 12},
            posterior=posterior,
            legal_actions=["attack", "switch"],
        )


def test_world_transport_ids_are_not_model_features() -> None:
    first = build_evaluator_input(
        public_state={"turn": 12, "tera_unused": True},
        posterior=_posterior(),
        legal_actions=["attack", "switch"],
    )
    changed = _posterior()
    changed["worlds"][0]["world_id"] = "arbitrary-renamed-id"
    changed["worlds"][1]["world_id"] = "another-arbitrary-id"
    second = build_evaluator_input(
        public_state={"turn": 12, "tera_unused": True},
        posterior=changed,
        legal_actions=["attack", "switch"],
    )

    assert first.world_features == second.world_features
    assert first.world_weights == second.world_weights
    assert abs(sum(first.world_weights) - 1.0) < 1e-12


def test_evaluator_identity_declares_public_belief_observability() -> None:
    identity = evaluator_identity(
        checkpoint_digest_value="sha256:" + "a" * 64,
        spec=BeliefEvaluatorSpec(),
    )

    assert identity["observability"] == "public_belief_only"
    assert identity["architecture"] == "weighted_deep_sets_policy_value"


def test_network_is_invariant_to_hidden_world_order() -> None:
    pytest.importorskip("jax")
    spec = BeliefEvaluatorSpec(
        public_width=16,
        world_width=12,
        action_width=8,
        hidden_width=10,
        world_hidden_width=9,
    )
    params = init_params(spec, seed=7)
    first = build_evaluator_input(
        public_state={"turn": 12, "active_hp_fraction": 0.61},
        posterior=_posterior(),
        legal_actions=["attack", "switch"],
        spec=spec,
    )
    reversed_posterior = _posterior()
    reversed_posterior["worlds"] = list(reversed(reversed_posterior["worlds"]))
    second = build_evaluator_input(
        public_state={"turn": 12, "active_hp_fraction": 0.61},
        posterior=reversed_posterior,
        legal_actions=["attack", "switch"],
        spec=spec,
    )

    first_value, first_logits = forward(params, first)
    second_value, second_logits = forward(params, second)

    assert float(first_value) == pytest.approx(float(second_value), abs=1e-6)
    assert [float(value) for value in first_logits] == pytest.approx(
        [float(value) for value in second_logits],
        abs=1e-6,
    )


def test_joint_policy_value_loss_is_finite() -> None:
    pytest.importorskip("jax")
    spec = BeliefEvaluatorSpec(
        public_width=16,
        world_width=12,
        action_width=8,
        hidden_width=10,
        world_hidden_width=9,
    )
    params = init_params(spec, seed=11)
    inputs = build_evaluator_input(
        public_state={"turn": 12, "active_hp_fraction": 0.61},
        posterior=_posterior(),
        legal_actions=["attack", "switch"],
        spec=spec,
    )

    objective = loss(
        params,
        inputs,
        value_target=0.4,
        policy_target=[0.8, 0.2],
    )

    assert math.isfinite(float(objective))
