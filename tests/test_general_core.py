from __future__ import annotations

from typing import Any, Mapping, Sequence

from azelficoast.core.decision_relevance import decision_relevance_quotient
from azelficoast.core.search import search_transition_program
from azelficoast.core.transition import validate_transition_oracle
from azelficoast.transition_oracle import validate_oracle_core


class SupportSizeEvaluator:
    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        del public_state, legal_actions
        return 1.0 if len(posterior_worlds) == 1 else 0.0


def _generic_oracle() -> dict[str, Any]:
    worlds = [
        {"world_id": "w1", "weight": 0.2, "hidden": {"fault": "a", "serial": "x"}},
        {"world_id": "w2", "weight": 0.3, "hidden": {"fault": "a", "serial": "y"}},
        {"world_id": "w3", "weight": 0.5, "hidden": {"fault": "b", "serial": "z"}},
    ]
    transitions = []
    for world in worlds:
        fault = world["hidden"]["fault"]
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "act",
                "outcomes": [
                    {
                        "probability": 1.0,
                        "observation": {"alarm": fault == "b"},
                        "successor": {"stage": "done"},
                        "terminal_utility": 1.0 if fault == "b" else 0.0,
                    }
                ],
            }
        )
    return {
        "schema": "example.transition-oracle",
        "schema_version": 7,
        "worlds": worlds,
        "legal_actions": ["act"],
        "transitions": transitions,
        "dependency_candidates": ["fault", "serial"],
    }


def test_general_transition_contract_accepts_independent_schema() -> None:
    oracle = _generic_oracle()
    worlds, actions, transitions, candidates = validate_transition_oracle(
        oracle,
        error_type=ValueError,
        expected_schema="example.transition-oracle",
        expected_schema_version=7,
    )

    assert [world["world_id"] for world in worlds] == ["w1", "w2", "w3"]
    assert actions == ["act"]
    assert set(transitions) == {("w1", "act"), ("w2", "act"), ("w3", "act")}
    assert candidates == ["fault", "serial"]


def test_exact_quotient_is_not_pokemon_specific() -> None:
    certificate, quotient = decision_relevance_quotient(
        _generic_oracle(),
        expected_schema="example.transition-oracle",
        expected_schema_version=7,
        certificate_schema="example.decision-relevance",
        certificate_schema_version=3,
    )

    assert certificate["schema"] == "example.decision-relevance"
    assert certificate["schema_version"] == 3
    assert certificate["decision_fields"] == ["fault"]
    assert certificate["worlds_in"] == 3
    assert certificate["classes_out"] == 2
    assert sorted(world["weight"] for world in quotient["worlds"]) == [0.5, 0.5]
    assert quotient["schema"] == "example.transition-oracle"


def test_information_set_search_uses_only_available_information() -> None:
    posterior = {
        "worlds": [
            {"world_id": "w1", "weight": 0.5, "hidden": {"mode": "a"}},
            {"world_id": "w2", "weight": 0.5, "hidden": {"mode": "b"}},
        ]
    }
    program_set = {
        "schema": "example.transition-program",
        "schema_version": 1,
        "world_ids": ["w1", "w2"],
        "legal_actions": ["hide", "probe"],
        "programs": [
            {
                "action": "hide",
                "classes_out": 1,
                "classes": [
                    {
                        "member_world_ids": ["w1", "w2"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"signal": "none"},
                                "successor": {"stage": 1},
                                "legal_actions": ["finish"],
                            }
                        ],
                    }
                ],
            },
            {
                "action": "probe",
                "classes_out": 2,
                "classes": [
                    {
                        "member_world_ids": ["w1"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"signal": "a"},
                                "successor": {"stage": 1},
                                "legal_actions": ["finish"],
                            }
                        ],
                    },
                    {
                        "member_world_ids": ["w2"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"signal": "b"},
                                "successor": {"stage": 1},
                                "legal_actions": ["finish"],
                            }
                        ],
                    },
                ],
            },
        ],
    }
    evaluator = SupportSizeEvaluator()

    determinized = search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method="determinization",
        evaluator=evaluator,
        expected_program_schema="example.transition-program",
        expected_program_schema_version=1,
    )
    information_set = search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        evaluator=evaluator,
        expected_program_schema="example.transition-program",
        expected_program_schema_version=1,
    )

    assert determinized["chosen_action"] == "hide"
    assert information_set["chosen_action"] == "probe"
    assert determinized["root_values"]["hide"] == 1.0
    assert information_set["root_values"]["hide"] == 0.0


def test_legacy_transition_import_routes_to_general_core() -> None:
    assert validate_oracle_core is validate_transition_oracle
