"""Minimal JAX training machinery for the public-belief policy/value network."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from azelficoast.belief.evaluator import (
    BeliefEvaluatorError,
    _require_jax,
    loss,
)


class TrainingInput(Protocol):
    legal_actions: tuple[str, ...]


@dataclass(frozen=True)
class TrainingExample:
    """One supervised target over a frozen public-belief state."""

    inputs: TrainingInput
    value_target: float
    policy_target: tuple[float, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value_target)) or not -1.0 <= float(self.value_target) <= 1.0:
            raise BeliefEvaluatorError("value target must be finite and within [-1, 1]")
        if len(self.policy_target) != len(self.inputs.legal_actions):
            raise BeliefEvaluatorError("policy target must cover every legal action")
        if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in self.policy_target):
            raise BeliefEvaluatorError("policy target mass must be finite and non-negative")
        if sum(float(value) for value in self.policy_target) <= 0.0:
            raise BeliefEvaluatorError("policy target must contain positive mass")


@dataclass(frozen=True)
class AdamState:
    step: int
    first_moment: Mapping[str, Any]
    second_moment: Mapping[str, Any]


def init_adam(params: Mapping[str, Any]) -> AdamState:
    _, jnp = _require_jax()
    return AdamState(
        step=0,
        first_moment={name: jnp.zeros_like(value) for name, value in params.items()},
        second_moment={name: jnp.zeros_like(value) for name, value in params.items()},
    )


def train_step(
    params: Mapping[str, Any],
    state: AdamState,
    example: TrainingExample,
    *,
    learning_rate: float = 3e-4,
    policy_weight: float = 1.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    loss_function: Callable[..., Any] = loss,
) -> tuple[dict[str, Any], AdamState, float]:
    """Apply one deterministic Adam update without Flax or Optax."""
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise BeliefEvaluatorError("learning rate must be positive and finite")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise BeliefEvaluatorError("Adam beta values must be within [0, 1)")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise BeliefEvaluatorError("Adam epsilon must be positive and finite")

    jax, jnp = _require_jax()

    def objective(candidate: Mapping[str, Any]) -> Any:
        return loss_function(
            candidate,
            example.inputs,
            value_target=example.value_target,
            policy_target=example.policy_target,
            policy_weight=policy_weight,
        )

    raw_loss, gradients = jax.value_and_grad(objective)(params)
    step = state.step + 1
    first: dict[str, Any] = {}
    second: dict[str, Any] = {}
    updated: dict[str, Any] = {}
    correction1 = 1.0 - beta1**step
    correction2 = 1.0 - beta2**step
    for name in sorted(params):
        gradient = gradients[name]
        first[name] = beta1 * state.first_moment[name] + (1.0 - beta1) * gradient
        second[name] = beta2 * state.second_moment[name] + (1.0 - beta2) * gradient * gradient
        first_hat = first[name] / correction1
        second_hat = second[name] / correction2
        updated[name] = params[name] - learning_rate * first_hat / (
            jnp.sqrt(second_hat) + epsilon
        )

    value = float(raw_loss)
    if not math.isfinite(value):
        raise BeliefEvaluatorError("training produced a non-finite loss")
    return updated, AdamState(step, first, second), value


def train_examples(
    params: Mapping[str, Any],
    examples: Sequence[TrainingExample],
    *,
    epochs: int,
    learning_rate: float = 3e-4,
    policy_weight: float = 1.0,
    loss_function: Callable[..., Any] = loss,
) -> tuple[dict[str, Any], list[float]]:
    """Train in a deterministic caller-supplied example order."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise BeliefEvaluatorError("epochs must be a positive integer")
    if not examples:
        raise BeliefEvaluatorError("training requires at least one example")

    current = dict(params)
    state = init_adam(current)
    losses: list[float] = []
    for _ in range(epochs):
        for example in examples:
            current, state, objective = train_step(
                current,
                state,
                example,
                learning_rate=learning_rate,
                policy_weight=policy_weight,
                loss_function=loss_function,
            )
            losses.append(objective)
    return current, losses
