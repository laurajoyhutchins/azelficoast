from __future__ import annotations

import pytest

from azelficoast.belief_evaluator import BeliefPrediction
from azelficoast.selective_belief import PolicyMarginSearchGate, SelectiveBeliefError


def _prediction(margin: float, *, actions: tuple[str, ...] = ("a", "b")) -> BeliefPrediction:
    if len(actions) == 1:
        probabilities = (1.0,)
    else:
        probabilities = ((1.0 + margin) / 2.0, (1.0 - margin) / 2.0)
    return BeliefPrediction(
        value=0.0,
        legal_actions=actions,
        probabilities=probabilities,
        selected_action=actions[0],
        policy_margin=margin,
        policy_entropy_bits=0.5,
    )


def test_default_gate_keeps_search_for_ambiguous_positions() -> None:
    assert PolicyMarginSearchGate().should_search(_prediction(0.99)) is True


def test_explicit_margin_allows_only_separated_policy_to_bypass_search() -> None:
    gate = PolicyMarginSearchGate(search_if_margin_at_most=0.20)

    assert gate.should_search(_prediction(0.10)) is True
    assert gate.should_search(_prediction(0.21)) is False


def test_single_legal_action_never_spends_search_budget() -> None:
    gate = PolicyMarginSearchGate()

    assert gate.should_search(_prediction(1.0, actions=("only",))) is False


def test_gate_rejects_invalid_threshold() -> None:
    with pytest.raises(SelectiveBeliefError):
        PolicyMarginSearchGate(search_if_margin_at_most=1.01)
