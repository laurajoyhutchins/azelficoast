from __future__ import annotations

from dataclasses import replace

import pytest

from azelficoast.research.contracts import (
    BeliefInput,
    ComputeBudget,
    FrozenJSONObject,
    MatchedExperimentSpec,
    MechanicsIdentity,
    PublicDecisionInput,
    ResearchContractError,
    parse_belief_artifact,
    stable_digest,
)


def _public_state(public_state: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "fixture_id": "fixture-1",
        "battle_tag": "battle-1",
        "public_state": public_state or {"turn": 8, "active": "Tinkaton"},
        "legal_actions": ["protect", "attack"],
    }


def _posterior() -> dict[str, object]:
    return {
        "treatment": "generator_faithful",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {
                "world_id": "opaque-a",
                "weight": 0.6,
                "hidden": {"item": "Choice Band", "moves": ["U-turn"]},
                "provenance": {"generator_round": 1},
            },
            {
                "world_id": "opaque-b",
                "weight": 0.4,
                "hidden": {"item": "Choice Scarf", "moves": ["U-turn"]},
                "provenance": {"generator_round": 2},
            },
        ],
    }


def _matched_spec() -> MatchedExperimentSpec:
    public = PublicDecisionInput.from_record(
        _public_state(),
        mechanics_identity=MechanicsIdentity.from_showdown_commit("pinned"),
    )
    belief, _transport = parse_belief_artifact(_posterior())
    return MatchedExperimentSpec.build(
        public=public,
        belief=belief,
        evaluator_identity_digest="sha256:" + "a" * 64,
        chance_treatment="shared_frozen_transition_oracle",
        compute_budget=ComputeBudget(
            unit="transition_evaluations",
            authorized=4096,
        ),
        depth=1,
        opponent_model="fixed_observed_response",
    )


def test_public_decision_input_rejects_private_and_future_information() -> None:
    for field in (
        "realized_hidden_state",
        "future_observation",
        "opponent_private_item",
        "opaque_world_id",
    ):
        state = _public_state({"turn": 8, field: {"item": "Choice Scarf"}})
        with pytest.raises(ResearchContractError, match="public decision input"):
            PublicDecisionInput.from_record(
                state,
                mechanics_identity=MechanicsIdentity.from_showdown_commit("pinned"),
            )

    state = _public_state()
    state["realized_hidden_state"] = {"item": "Choice Scarf"}
    with pytest.raises(ResearchContractError, match="forbidden private or future field"):
        PublicDecisionInput.from_record(
            state,
            mechanics_identity=MechanicsIdentity.from_showdown_commit("pinned"),
        )


def test_public_decision_contract_rejects_private_data_even_when_constructed_directly() -> None:
    with pytest.raises(ResearchContractError, match="forbidden private or future field"):
        PublicDecisionInput(
            fixture_id="fixture-1",
            battle_tag="battle-1",
            public_state=FrozenJSONObject.from_mapping(
                {"realized_hidden_state": {"item": "Choice Scarf"}}
            ),
            legal_actions=("protect",),
            public_history_identity="history-1",
            mechanics_identity=MechanicsIdentity.from_showdown_commit("pinned"),
        )


def test_posterior_rejects_information_revealed_after_the_decision() -> None:
    posterior = _posterior()
    worlds = posterior["worlds"]
    assert isinstance(worlds, list)
    worlds[0] = {**worlds[0], "future_revealed_item": "Choice Scarf"}

    with pytest.raises(ResearchContractError, match="future-only field"):
        BeliefInput.from_record(posterior)

    posterior = _posterior()
    posterior["realized_hidden_state"] = {"item": "Choice Scarf"}
    with pytest.raises(ResearchContractError, match="forbidden private or future field"):
        BeliefInput.from_record(posterior)


def test_belief_identity_ignores_transport_ids_and_support_order() -> None:
    original = _posterior()
    renamed = _posterior()
    renamed_worlds = renamed["worlds"]
    assert isinstance(renamed_worlds, list)
    renamed_worlds[0] = {**renamed_worlds[0], "world_id": "renamed-a"}
    renamed_worlds[1] = {**renamed_worlds[1], "world_id": "renamed-b"}
    renamed_worlds.reverse()

    first, _ = parse_belief_artifact(original)
    second, _ = parse_belief_artifact(renamed)

    assert first.semantic_digest == second.semantic_digest
    assert first.model_worlds == second.model_worlds
    assert all("world_id" not in world.to_record() for world in first.model_worlds)


def test_belief_rejects_unrecognized_transport_identifier_fields() -> None:
    posterior = _posterior()
    worlds = posterior["worlds"]
    assert isinstance(worlds, list)
    worlds[0] = {**worlds[0], "opaque_world_id": "not-a-feature"}

    with pytest.raises(ResearchContractError, match="transport-only identifier"):
        BeliefInput.from_record(posterior)


def test_equivalent_support_splitting_has_one_canonical_semantic_atom() -> None:
    original = _posterior()
    split = _posterior()
    original_worlds = original["worlds"]
    split_worlds = split["worlds"]
    assert isinstance(original_worlds, list)
    assert isinstance(split_worlds, list)
    split_worlds[0] = {**split_worlds[0], "weight": 0.3}
    split_worlds.append(
        {
            **split_worlds[0],
            "world_id": "opaque-a-copy",
            "weight": 0.3,
        }
    )

    first, _ = parse_belief_artifact(original)
    second, transport = parse_belief_artifact(split)

    assert first.semantic_digest == second.semantic_digest
    assert len(second.model_worlds) == 2
    assert sum(world.weight for world in second.model_worlds) == pytest.approx(1.0)
    assert transport.semantic_identity_for("opaque-a-copy") == transport.semantic_identity_for(
        "opaque-a"
    )


def test_belief_weights_are_positive_finite_and_normalized() -> None:
    posterior = _posterior()
    worlds = posterior["worlds"]
    assert isinstance(worlds, list)
    worlds[0] = {**worlds[0], "weight": float("nan")}

    with pytest.raises(ResearchContractError, match="positive and finite"):
        BeliefInput.from_record(posterior)


@pytest.mark.parametrize(
    "changed",
    [
        {"posterior_semantic_digest": "different"},
        {"mechanics_identity": MechanicsIdentity.from_showdown_commit("other")},
        {"evaluator_identity_digest": "different"},
        {"chance_treatment": "independent_randomness"},
        {"compute_budget": ComputeBudget("transition_evaluations", 4097)},
    ],
)
def test_matched_spec_rejects_any_treatment_input_drift(
    changed: dict[str, object],
) -> None:
    spec = _matched_spec()
    altered = replace(spec, **changed)

    with pytest.raises(ResearchContractError, match="matched specification differs"):
        spec.require_compatible(altered)


def test_matched_spec_rejects_a_different_root_action_set() -> None:
    spec = _matched_spec()
    actions = ("protect",)
    state_digest = stable_digest(
        {
            "fixture_id": spec.fixture_id,
            "battle_tag": spec.battle_tag,
            "public_state": spec.public_state.to_record(),
            "legal_actions": list(actions),
        }
    )
    altered = replace(spec, legal_actions=actions, public_state_digest=state_digest)

    with pytest.raises(ResearchContractError, match="matched specification differs"):
        spec.require_compatible(altered)
