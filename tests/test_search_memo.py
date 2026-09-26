from __future__ import annotations

from typing import Any, Mapping, Sequence

from azelficoast.core.evaluation import EvaluationFrontier
from azelficoast.core.memo import SemanticMemo
from azelficoast.core.search import search_transition_program


class _Evaluator:
    def __init__(self, value: float) -> None:
        self.value_result = value
        self.calls = 0

    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        del public_state, posterior_worlds, legal_actions
        self.calls += 1
        return self.value_result


def _inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    posterior = {
        "worlds": [
            {"world_id": "a", "weight": 0.5, "hidden": {"kind": "a"}},
            {"world_id": "b", "weight": 0.5, "hidden": {"kind": "b"}},
        ]
    }
    program = {
        "schema": "example.transition-program-set",
        "schema_version": 1,
        "world_ids": ["a", "b"],
        "legal_actions": ["wait"],
        "programs": [
            {
                "action": "wait",
                "classes_out": 1,
                "classes": [
                    {
                        "member_world_ids": ["a", "b"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"same": True},
                                "successor": {"turn": 2},
                                "legal_actions": ["continue"],
                            }
                        ],
                    }
                ],
            }
        ],
    }
    return program, posterior


def test_frontier_materialization_reuses_semantics_but_not_evaluator_outputs() -> None:
    program, posterior = _inputs()
    memo = SemanticMemo[EvaluationFrontier](max_entries=8)
    first_evaluator = _Evaluator(1.0)
    second_evaluator = _Evaluator(2.0)

    first = search_transition_program(
        program_set=program,
        posterior=posterior,
        method="information_set",
        evaluator=first_evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
        frontier_memo=memo,
    )
    second = search_transition_program(
        program_set=program,
        posterior=posterior,
        method="information_set",
        evaluator=second_evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
        frontier_memo=memo,
    )

    assert first["frontier_memo_hit"] is False
    assert first["frontier_builds"] == 1
    assert second["frontier_memo_hit"] is True
    assert second["frontier_builds"] == 0
    assert first["frontier_group_identity"] == second["frontier_group_identity"]
    assert first["transition_evaluations"] == second["transition_evaluations"] == 1
    assert first_evaluator.calls == second_evaluator.calls == 1
    assert first["root_values"] == {"wait": 1.0}
    assert second["root_values"] == {"wait": 2.0}


def test_frontier_memo_misses_when_posterior_identity_changes() -> None:
    program, posterior = _inputs()
    memo = SemanticMemo[EvaluationFrontier](max_entries=8)
    evaluator = _Evaluator(1.0)

    first = search_transition_program(
        program_set=program,
        posterior=posterior,
        method="information_set",
        evaluator=evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
        frontier_memo=memo,
    )
    changed = {
        "worlds": [
            {"world_id": "a", "weight": 0.75, "hidden": {"kind": "a"}},
            {"world_id": "b", "weight": 0.25, "hidden": {"kind": "b"}},
        ]
    }
    second = search_transition_program(
        program_set=program,
        posterior=changed,
        method="information_set",
        evaluator=evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
        frontier_memo=memo,
    )

    assert first["frontier_memo_hit"] is False
    assert second["frontier_memo_hit"] is False
    assert first["frontier_group_identity"] != second["frontier_group_identity"]
