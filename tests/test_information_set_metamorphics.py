from __future__ import annotations

import copy

from azelficoast.belief_evaluator import BeliefEvaluatorSpec, BeliefPrediction
from azelficoast.transition_program_search import search_transition_program


class _PublicContinuationEvaluator:
    spec = BeliefEvaluatorSpec(
        public_width=8,
        world_width=8,
        action_width=8,
        hidden_width=8,
        world_hidden_width=8,
    )

    def predict(self, inputs) -> BeliefPrediction:
        legal = inputs.legal_actions
        value = 1.0 if "win" in legal else 0.0
        probability = 1.0 / len(legal)
        return BeliefPrediction(
            value=value,
            legal_actions=legal,
            probabilities=tuple(probability for _ in legal),
            selected_action=min(legal),
            policy_margin=0.0,
            policy_entropy_bits=0.0,
        )


def _posterior() -> dict[str, object]:
    return {
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {"world_id": "w1", "weight": 0.6, "hidden": {"item": "scarf"}},
            {"world_id": "w2", "weight": 0.4, "hidden": {"item": "specs"}},
        ],
    }


def _program() -> dict[str, object]:
    return {
        "schema": "azelficoast.whole-turn-transition-program-set",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "world_ids": ["w1", "w2"],
        "legal_actions": ["attack", "protect"],
        "dependency_candidates": ["item"],
        "programs": [
            {
                "action": "attack",
                "effect_signature": "attack-effect",
                "dependency_fields": [],
                "partition_method": "finite-support-minimal-semantics",
                "representative_world_count": 1,
                "worlds_in": 2,
                "classes_out": 1,
                "world_reduction": 1,
                "reduction_fraction": 0.5,
                "partition_key_hash": "attack-partition",
                "classes": [
                    {
                        "class_id": "attack-class",
                        "read_fields": [],
                        "projection_key": [],
                        "representative_world_id": "w1",
                        "member_world_ids": ["w1", "w2"],
                        "semantic_hash": "attack-semantic",
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"kind": "same"},
                                "successor": {"turn": 2},
                                "legal_actions": ["win"],
                            }
                        ],
                    }
                ],
            },
            {
                "action": "protect",
                "effect_signature": "protect-effect",
                "dependency_fields": [],
                "partition_method": "finite-support-minimal-semantics",
                "representative_world_count": 1,
                "worlds_in": 2,
                "classes_out": 1,
                "world_reduction": 1,
                "reduction_fraction": 0.5,
                "partition_key_hash": "protect-partition",
                "classes": [
                    {
                        "class_id": "protect-class",
                        "read_fields": [],
                        "projection_key": [],
                        "representative_world_id": "w1",
                        "member_world_ids": ["w1", "w2"],
                        "semantic_hash": "protect-semantic",
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"kind": "same"},
                                "successor": {"turn": 2},
                                "legal_actions": ["lose"],
                            }
                        ],
                    }
                ],
            },
        ],
    }


def _search(program, posterior):
    return search_transition_program(
        program_set=program,
        posterior=posterior,
        method="information_set",
        evaluator=_PublicContinuationEvaluator(),
    )


def _rename_world_ids(program, posterior, mapping):
    transformed_program = copy.deepcopy(program)
    transformed_posterior = copy.deepcopy(posterior)
    transformed_program["world_ids"] = [
        mapping[world_id] for world_id in transformed_program["world_ids"]
    ]
    for row in transformed_program["programs"]:
        for execution_class in row["classes"]:
            execution_class["representative_world_id"] = mapping[
                execution_class["representative_world_id"]
            ]
            execution_class["member_world_ids"] = [
                mapping[world_id] for world_id in execution_class["member_world_ids"]
            ]
    for world in transformed_posterior["worlds"]:
        world["world_id"] = mapping[world["world_id"]]
    transformed_program["world_ids"].reverse()
    transformed_posterior["worlds"].reverse()
    return transformed_program, transformed_posterior


def _split_world(program, posterior):
    transformed_program = copy.deepcopy(program)
    transformed_posterior = copy.deepcopy(posterior)
    source = next(world for world in transformed_posterior["worlds"] if world["world_id"] == "w1")
    transformed_posterior["worlds"].remove(source)
    for suffix in ("a", "b"):
        clone = copy.deepcopy(source)
        clone["world_id"] = f"w1{suffix}"
        clone["weight"] = source["weight"] / 2
        transformed_posterior["worlds"].append(clone)

    transformed_program["world_ids"] = ["w1a", "w1b", "w2"]
    for row in transformed_program["programs"]:
        row["worlds_in"] = 3
        row["world_reduction"] = 2
        row["reduction_fraction"] = 2 / 3
        execution_class = row["classes"][0]
        execution_class["representative_world_id"] = "w1a"
        execution_class["member_world_ids"] = ["w1a", "w1b", "w2"]
    return transformed_program, transformed_posterior


def test_permuting_and_renaming_hidden_world_ids_is_semantically_inert() -> None:
    baseline = _search(_program(), _posterior())
    program, posterior = _rename_world_ids(
        _program(),
        _posterior(),
        {"w1": "alpha", "w2": "beta"},
    )
    mutated = _search(program, posterior)

    assert mutated["chosen_action"] == baseline["chosen_action"]
    assert mutated["root_values"] == baseline["root_values"]


def test_splitting_identical_world_into_two_half_weight_worlds_preserves_value() -> None:
    baseline = _search(_program(), _posterior())
    program, posterior = _split_world(_program(), _posterior())
    mutated = _search(program, posterior)

    assert mutated["chosen_action"] == baseline["chosen_action"]
    assert mutated["root_values"] == baseline["root_values"]


def test_merging_semantically_identical_split_worlds_preserves_value() -> None:
    split_program, split_posterior = _split_world(_program(), _posterior())
    baseline = _search(split_program, split_posterior)
    merged = _search(_program(), _posterior())

    assert merged["chosen_action"] == baseline["chosen_action"]
    assert merged["root_values"] == baseline["root_values"]


def test_unread_hidden_field_does_not_change_public_information_set_continuation() -> None:
    baseline = _search(_program(), _posterior())
    posterior = copy.deepcopy(_posterior())
    for index, world in enumerate(posterior["worlds"]):
        world["hidden"]["outside_verified_dependency_signature"] = {
            "nonce": index,
            "arbitrary": "changed",
        }
    mutated = _search(_program(), posterior)

    assert mutated["chosen_action"] == baseline["chosen_action"]
    assert mutated["root_values"] == baseline["root_values"]
