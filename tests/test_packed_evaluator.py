from __future__ import annotations

import copy
import hashlib
import json
import math

import pytest

from azelficoast.belief.evaluator import BeliefEvaluatorError, BeliefEvaluatorSpec
from azelficoast.belief.packed_evaluator import (
    PackedBeliefEvaluatorSpec,
    build_packed_evaluator_input,
    forward_packed,
    init_packed_params,
    packed_loss,
    predict_packed,
    predict_packed_values,
)
from azelficoast.belief.showdown_packing import ShowdownVocabulary
from azelficoast.belief.training import TrainingExample, init_adam, train_step
from azelficoast.research.experiments.evaluator_representation_experiment import (
    run_representation_comparison,
)
from azelficoast.research.training_records import TRAINING_SCHEMA, TRAINING_SCHEMA_VERSION


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vocabulary() -> ShowdownVocabulary:
    material: dict[str, object] = {
        "schema": "azelficoast.showdown-vocabulary",
        "schema_version": 1,
        "showdown_commit": "a" * 40,
        "generation": 9,
        "identity": {},
        "species": [
            {
                "id": "rotomwash",
                "num": 479,
                "forme_index": 1,
                "base_species_id": "rotom",
                "forme": "wash",
            },
            {
                "id": "gliscor",
                "num": 472,
                "forme_index": 0,
                "base_species_id": "gliscor",
                "forme": "",
            },
        ],
        "moves": [
            {"id": "hydropump", "num": 56},
            {"id": "voltswitch", "num": 521},
            {"id": "willowisp", "num": 261},
            {"id": "earthquake", "num": 89},
        ],
        "items": [
            {"id": "choicescarf", "num": 287},
            {"id": "leftovers", "num": 234},
            {"id": "toxicorb", "num": 272},
        ],
        "abilities": [
            {"id": "levitate", "num": 26},
            {"id": "poisonheal", "num": 90},
        ],
        "types": [
            {"id": "water", "index": 1},
            {"id": "ground", "index": 2},
        ],
        "natures": [
            {"id": "timid", "index": 1},
            {"id": "impish", "index": 2},
        ],
        "roles": [
            {"id": "fastpivot", "index": 1},
            {"id": "physicalwall", "index": 2},
        ],
    }
    return ShowdownVocabulary.from_record(
        {**material, "vocabulary_sha256": _digest(material)}
    )


def _member(
    *,
    species: str,
    ability: str,
    item: str,
    moves: list[str],
    tera: str,
    role: str,
    nature: str,
    was_lead: bool,
) -> dict[str, object]:
    return {
        "species": species,
        "level": 80,
        "gender": "N",
        "ability": ability,
        "item": item,
        "moves": moves,
        "tera_type": tera,
        "role": role,
        "nature": nature,
        "evs": {"hp": 84, "def": 84, "spe": 84},
        "ivs": {},
        "was_lead": was_lead,
    }


def _posterior() -> dict[str, object]:
    rotom = _member(
        species="Rotom-Wash",
        ability="Levitate",
        item="Choice Scarf",
        moves=["Hydro Pump", "Volt Switch"],
        tera="Water",
        role="Fast Pivot",
        nature="Timid",
        was_lead=True,
    )
    gliscor = _member(
        species="Gliscor",
        ability="Poison Heal",
        item="Toxic Orb",
        moves=["Earthquake"],
        tera="Ground",
        role="Physical Wall",
        nature="Impish",
        was_lead=False,
    )
    rotom_leftovers = {**rotom, "item": "Leftovers", "moves": ["Hydro Pump", "Will-O-Wisp"]}
    return {
        "schema": "azelficoast.joint-random-battle-posterior",
        "schema_version": 1,
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "support_status": "sufficient",
        "showdown_commit": "a" * 40,
        "construction": {
            "kind": "full-team-generator-rejection-particles",
            "posterior_treatment": "generator_faithful_joint_empirical",
            "preserves_joint_team_set_correlations": True,
        },
        "worlds": [
            {
                "world_id": "scarf",
                "weight": 0.4,
                "hidden": {"team": [rotom, gliscor]},
            },
            {
                "world_id": "leftovers",
                "weight": 0.6,
                "hidden": {"team": [rotom_leftovers, gliscor]},
            },
        ],
        "public_evidence": {
            "opponent_team_size": 2,
            "revealed": [{"species": "Rotom-Wash", "was_lead": True}],
        },
    }


def _inputs(posterior: dict[str, object] | None = None):
    vocabulary = _vocabulary()
    spec = PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=16,
        action_width=8,
        embedding_width=6,
        member_hidden_width=9,
        world_hidden_width=10,
        hidden_width=12,
    )
    inputs = build_packed_evaluator_input(
        public_state={"turn": 7, "weather": "rain"},
        posterior=posterior or _posterior(),
        legal_actions=("move:hydropump", "switch:gliscor"),
        vocabulary=vocabulary,
        spec=spec,
    )
    return vocabulary, spec, inputs


def test_packed_evaluator_consumes_native_ids_without_hidden_hash_features() -> None:
    _, spec, inputs = _inputs()

    assert inputs.species_num == ((479, 472), (479, 472))
    assert inputs.item_num == ((287, 272), (234, 272))
    assert inputs.move_num[0][0] == (56, 521, 0, 0)
    assert inputs.vocabulary_sha256 == spec.vocabulary_sha256
    assert inputs.world_weights == (0.4, 0.6)


def test_packed_evaluator_fails_closed_on_vocabulary_identity_drift() -> None:
    vocabulary, spec, _ = _inputs()
    changed = PackedBeliefEvaluatorSpec(
        **{
            **spec.as_dict(),
            "vocabulary_sha256": "sha256:" + "b" * 64,
        }
    )

    with pytest.raises(BeliefEvaluatorError, match="spec and vocabulary differ"):
        build_packed_evaluator_input(
            public_state={"turn": 7},
            posterior=_posterior(),
            legal_actions=("move:hydropump",),
            vocabulary=vocabulary,
            spec=changed,
        )


def test_packed_network_is_invariant_to_world_and_team_order() -> None:
    pytest.importorskip("jax")
    vocabulary, spec, baseline_inputs = _inputs()
    params = init_packed_params(spec, seed=17)

    reordered = copy.deepcopy(_posterior())
    worlds = reordered["worlds"]
    assert isinstance(worlds, list)
    worlds.reverse()
    for world in worlds:
        assert isinstance(world, dict)
        hidden = world["hidden"]
        assert isinstance(hidden, dict)
        team = hidden["team"]
        assert isinstance(team, list)
        team.reverse()

    reordered_inputs = build_packed_evaluator_input(
        public_state={"turn": 7, "weather": "rain"},
        posterior=reordered,
        legal_actions=("move:hydropump", "switch:gliscor"),
        vocabulary=vocabulary,
        spec=spec,
    )
    baseline = predict_packed(params, baseline_inputs)
    actual = predict_packed(params, reordered_inputs)

    assert actual.value == pytest.approx(baseline.value, abs=1e-6)
    assert dict(zip(actual.legal_actions, actual.probabilities, strict=True)) == pytest.approx(
        dict(zip(baseline.legal_actions, baseline.probabilities, strict=True)),
        abs=1e-6,
    )
    assert actual.selected_action == baseline.selected_action


def test_packed_network_is_invariant_to_equivalent_support_splitting() -> None:
    pytest.importorskip("jax")
    vocabulary, spec, baseline_inputs = _inputs()
    params = init_packed_params(spec, seed=23)

    split = copy.deepcopy(_posterior())
    worlds = split["worlds"]
    assert isinstance(worlds, list)
    first = worlds[0]
    assert isinstance(first, dict)
    first["weight"] = 0.1
    duplicate = copy.deepcopy(first)
    duplicate["world_id"] = "scarf-duplicate"
    duplicate["weight"] = 0.3
    worlds.append(duplicate)

    split_inputs = build_packed_evaluator_input(
        public_state={"turn": 7, "weather": "rain"},
        posterior=split,
        legal_actions=("move:hydropump", "switch:gliscor"),
        vocabulary=vocabulary,
        spec=spec,
    )
    baseline = predict_packed(params, baseline_inputs)
    actual = predict_packed(params, split_inputs)

    assert actual.value == pytest.approx(baseline.value, abs=1e-6)
    assert actual.probabilities == pytest.approx(baseline.probabilities, abs=1e-6)


def test_packed_batched_values_match_scalar_across_world_buckets() -> None:
    pytest.importorskip("jax")
    vocabulary, spec, first = _inputs()
    params = init_packed_params(spec, seed=29)

    wider = copy.deepcopy(_posterior())
    worlds = wider["worlds"]
    assert isinstance(worlds, list)
    worlds[0]["weight"] = 0.2
    worlds[1]["weight"] = 0.5
    third = copy.deepcopy(worlds[0])
    third["world_id"] = "third"
    third["weight"] = 0.3
    worlds.append(third)
    second = build_packed_evaluator_input(
        public_state={"turn": 8, "weather": "rain"},
        posterior=wider,
        legal_actions=("move:hydropump", "switch:gliscor"),
        vocabulary=vocabulary,
        spec=spec,
    )

    scalar = (
        float(forward_packed(params, first)[0]),
        float(forward_packed(params, second)[0]),
    )
    batched = predict_packed_values(params, (first, second))
    assert batched == pytest.approx(scalar, abs=1e-6)


def test_existing_adam_machinery_trains_packed_objective() -> None:
    pytest.importorskip("jax")
    _, spec, inputs = _inputs()
    params = init_packed_params(spec, seed=31)
    example = TrainingExample(
        inputs=inputs,
        value_target=0.5,
        policy_target=(0.8, 0.2),
    )
    state = init_adam(params)
    before = float(
        packed_loss(
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
        loss_function=packed_loss,
    )

    assert math.isfinite(objective)
    assert next_state.step == 1
    assert set(updated) == set(params)
    assert objective == pytest.approx(before, rel=1e-6)


def _training_record(*, record_id: str, split: str, value: float, selected: str):
    actions = ["move:hydropump", "switch:gliscor"]
    return {
        "schema": TRAINING_SCHEMA,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "record_id": record_id,
        "battle_id": "battle-" + record_id,
        "decision_index": 0,
        "split_group_id": "group-" + record_id,
        "split": split,
        "input": {
            "fixture_id": "fixture-" + record_id,
            "public_state": {"turn": 7, "weather": "rain"},
            "posterior": _posterior(),
            "posterior_digest": "sha256:" + "c" * 64,
            "posterior_treatment": "generator_faithful_joint_empirical",
            "legal_actions": actions,
        },
        "targets": {
            "value": {"public_belief_search_return": value},
            "policy": {
                "selected_action": selected,
                "action_probabilities": {
                    action: 1.0 if action == selected else 0.0
                    for action in actions
                },
            },
        },
        "provenance": {},
    }


def test_representation_comparison_uses_identical_frozen_records() -> None:
    pytest.importorskip("jax")
    vocabulary = _vocabulary()
    result = run_representation_comparison(
        (
            _training_record(
                record_id="train-a",
                split="train",
                value=0.5,
                selected="move:hydropump",
            ),
            _training_record(
                record_id="train-b",
                split="train",
                value=-0.25,
                selected="switch:gliscor",
            ),
            _training_record(
                record_id="validation-a",
                split="validation",
                value=0.1,
                selected="move:hydropump",
            ),
        ),
        vocabulary=vocabulary,
        epochs=1,
        seed=41,
        hashed_spec=BeliefEvaluatorSpec(
            public_width=16,
            world_width=12,
            action_width=8,
            hidden_width=12,
            world_hidden_width=10,
        ),
        packed_spec=PackedBeliefEvaluatorSpec.from_vocabulary(
            vocabulary,
            public_width=16,
            action_width=8,
            embedding_width=6,
            member_hidden_width=9,
            world_hidden_width=10,
            hidden_width=12,
        ),
    )

    assert result["matched"]["record_count"] == 3
    assert result["matched"]["train_count"] == 2
    assert result["matched"]["validation_count"] == 1
    assert result["matched"]["vocabulary_sha256"] == vocabulary.vocabulary_sha256
    assert result["hashed_control"]["dense_input_bytes"] > 0
    assert result["showdown_packed_treatment"]["dense_input_bytes"] > 0
    assert result["hashed_control"]["trained_validation"]["count"] == 1
    assert result["showdown_packed_treatment"]["trained_validation"]["count"] == 1
