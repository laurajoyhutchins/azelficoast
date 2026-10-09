from __future__ import annotations

from azelficoast.belief.evaluator import BeliefEvaluatorSpec, BeliefPrediction
from azelficoast.research.router_ablation import (
    RouterCase,
    evaluate_admitted_counterfactual,
    summarize_observed_routes,
    threshold_sweep,
)
from azelficoast.search.selective import PolicyMarginSearchGate


def _cases() -> list[RouterCase]:
    return [
        RouterCase(
            case_id="confident",
            policy_margin=0.8,
            direct_regret=0.02,
            search_regret=0.01,
            fallback_regret=0.20,
            actual_route="direct",
            direct_outcome=1.0,
            search_outcome=1.0,
            fallback_outcome=0.0,
        ),
        RouterCase(
            case_id="uncertain",
            policy_margin=0.1,
            direct_regret=0.30,
            search_regret=0.05,
            fallback_regret=0.40,
            actual_route="search",
            direct_outcome=0.0,
            search_outcome=1.0,
            fallback_outcome=0.0,
        ),
        RouterCase(
            case_id="unavailable",
            policy_margin=0.05,
            direct_regret=0.25,
            search_regret=None,
            fallback_regret=0.15,
            actual_route="fallback",
            direct_outcome=0.0,
            fallback_outcome=1.0,
        ),
    ]


def test_observed_summary_keeps_route_conditioned_regret_separate() -> None:
    summary = summarize_observed_routes(_cases())

    assert summary["routes"]["direct"]["case_count"] == 1
    assert summary["routes"]["search"]["mean_regret"] == 0.05
    assert summary["routes"]["fallback"]["mean_regret"] == 0.15
    assert summary["always_search_coverage_rate"] == 2 / 3


def test_threshold_sweep_exposes_search_and_fallback_mixture() -> None:
    rows = threshold_sweep(_cases(), thresholds=[0.0, 0.2, 1.0])

    assert rows[0]["search_count"] == 0
    assert rows[0]["fallback_count"] == 0
    assert rows[1]["search_count"] == 1
    assert rows[1]["fallback_count"] == 1
    assert rows[2]["search_count"] == 2
    assert rows[2]["fallback_count"] == 1
    assert rows[1]["mean_regret"] < rows[0]["mean_regret"]



class _CounterfactualEvaluator:
    spec = BeliefEvaluatorSpec(
        public_width=8,
        world_width=8,
        action_width=8,
        hidden_width=8,
        world_hidden_width=8,
    )

    def predict(self, inputs) -> BeliefPrediction:
        if inputs.legal_actions == ("risky", "safe"):
            return BeliefPrediction(
                value=0.0,
                legal_actions=inputs.legal_actions,
                probabilities=(0.9, 0.1),
                selected_action="risky",
                policy_margin=0.8,
                policy_entropy_bits=0.4,
            )
        value = 1.0 if "good" in inputs.legal_actions else 0.0
        probability = 1.0 / len(inputs.legal_actions)
        return BeliefPrediction(
            value=value,
            legal_actions=inputs.legal_actions,
            probabilities=tuple(probability for _ in inputs.legal_actions),
            selected_action=min(inputs.legal_actions),
            policy_margin=0.0,
            policy_entropy_bits=0.0,
        )


def _counterfactual_program() -> dict[str, object]:
    return {
        "schema": "azelficoast.core.transition-program-set",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "world_ids": ["w1", "w2"],
        "legal_actions": ["risky", "safe"],
        "dependency_candidates": [],
        "programs": [
            {
                "action": "risky",
                "classes_out": 1,
                "classes": [
                    {
                        "class_id": "risky",
                        "member_world_ids": ["w1", "w2"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"kind": "same"},
                                "successor": {"turn": 2},
                                "legal_actions": ["bad"],
                            }
                        ],
                    }
                ],
            },
            {
                "action": "safe",
                "classes_out": 1,
                "classes": [
                    {
                        "class_id": "safe",
                        "member_world_ids": ["w1", "w2"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"kind": "same"},
                                "successor": {"turn": 2},
                                "legal_actions": ["good"],
                            }
                        ],
                    }
                ],
            },
        ],
    }


def test_admitted_counterfactual_uses_same_support_for_direct_and_search() -> None:
    posterior = {
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {"world_id": "w1", "weight": 0.5, "hidden": {"item": "a"}},
            {"world_id": "w2", "weight": 0.5, "hidden": {"item": "b"}},
        ],
    }
    case = evaluate_admitted_counterfactual(
        case_id="fixture",
        public_state={"turn": 1},
        legal_actions=["risky", "safe"],
        posterior=posterior,
        transition_program=_counterfactual_program(),
        evaluator=_CounterfactualEvaluator(),
        search_gate=PolicyMarginSearchGate(search_if_margin_at_most=0.9),
        fallback_action="risky",
    )

    assert case.actual_route == "search"
    assert case.direct_regret == 1.0
    assert case.search_regret == 0.0
    assert case.fallback_regret == 1.0
