from __future__ import annotations

import copy

from azelficoast.belief.evaluator import BeliefEvaluatorSpec, BeliefPrediction
from azelficoast.core.mechanics import VerifiedTransitionProgramSet
from azelficoast.research.contracts import MechanicsIdentity, parse_belief_artifact
from azelficoast.research.typed_search import search_transition_program
from azelficoast.core.whole_turn_program import compile_whole_turn_programs


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


def _oracle(posterior: dict[str, object]) -> dict[str, object]:
    worlds = copy.deepcopy(posterior["worlds"])
    assert isinstance(worlds, list)
    transitions: list[dict[str, object]] = []
    for world in worlds:
        assert isinstance(world, dict)
        for action, legal_actions in (("attack", ["win"]), ("protect", ["lose"])):
            transitions.append(
                {
                    "world_id": world["world_id"],
                    "action": action,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"kind": "same"},
                            "successor": {"turn": 2},
                            "legal_actions": legal_actions,
                            "hidden_reads": [],
                        }
                    ],
                }
            )
    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["attack", "protect"],
        "dependency_candidates": ["item"],
        "declared_reads": {"attack": [], "protect": []},
        "transitions": transitions,
    }


def _program(posterior: dict[str, object] | None = None) -> dict[str, object]:
    return compile_whole_turn_programs(_oracle(posterior or _posterior()))


def _search(program, posterior):
    belief, transport_index = parse_belief_artifact(posterior)
    mechanics = VerifiedTransitionProgramSet.from_artifact(
        artifact=program,
        identity=MechanicsIdentity.from_showdown_commit("pinned"),
        fixture_id="fixture",
        legal_actions=("attack", "protect"),
        belief=belief,
        transport_index=transport_index,
    )
    return search_transition_program(
        mechanics=mechanics,
        belief=belief,
        transport_index=transport_index,
        method="information_set",
        evaluator=_PublicContinuationEvaluator(),
    )


def _rename_world_ids(program, posterior, mapping):
    transformed_posterior = copy.deepcopy(posterior)
    for world in transformed_posterior["worlds"]:
        world["world_id"] = mapping[world["world_id"]]
    transformed_posterior["worlds"].reverse()
    return _program(transformed_posterior), transformed_posterior


def _split_world(program, posterior):
    transformed_posterior = copy.deepcopy(posterior)
    source = next(world for world in transformed_posterior["worlds"] if world["world_id"] == "w1")
    transformed_posterior["worlds"].remove(source)
    for suffix in ("a", "b"):
        clone = copy.deepcopy(source)
        clone["world_id"] = f"w1{suffix}"
        clone["weight"] = source["weight"] / 2
        transformed_posterior["worlds"].append(clone)
    return _program(transformed_posterior), transformed_posterior


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
