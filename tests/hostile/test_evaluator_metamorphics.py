from __future__ import annotations

import copy

import pytest

from azelficoast.belief.evaluator import (
    BeliefEvaluatorSpec,
    build_evaluator_input,
    checkpoint_digest,
    forward,
    init_params,
    load_checkpoint,
    predict,
    write_checkpoint,
)
from hostile.fixtures import (
    EVALUATOR_OUTPUT_ABS_TOLERANCE,
    hostile_case,
    tiny_posterior,
    tiny_public_state,
)
from hostile.transforms import (
    merge_equivalent_worlds,
    permute_support,
    rename_world_ids,
    split_world,
)


SPEC = BeliefEvaluatorSpec(
    public_width=16,
    world_width=12,
    action_width=8,
    hidden_width=10,
    world_hidden_width=9,
)


def _prediction(posterior, actions=("move:a", "move:b")):
    params = init_params(SPEC, seed=1729)
    inputs = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=posterior,
        legal_actions=list(actions),
        spec=SPEC,
    )
    return predict(params, inputs)


@hostile_case(
    mutation="rename, reverse, split, merge, and oppositely correlate opaque IDs with two hidden semantics",
    expected="learned value and action-to-probability map agree within 1e-6 and selected action is exact",
    threat="world counting, support order, or a transport ID correlated with hidden semantics alters the public-belief decision",
    layer="real preprocessing plus JAX evaluator pooling",
    tier="simulator",
)
def test_prediction_is_invariant_to_semantic_support_representations() -> None:
    pytest.importorskip("jax")
    posterior = tiny_posterior()
    split = split_world(posterior, 0, (0.1, 0.15), identifiers=("a", "b"))
    variants = [
        rename_world_ids(posterior, ["world-92841", "world-negative-seven"]),
        rename_world_ids(posterior, ["good_world", "bad_world"]),
        rename_world_ids(posterior, ["bad_world", "good_world"]),
        permute_support(posterior, (1, 0)),
        split,
        merge_equivalent_worlds(split),
    ]
    transport_poison = copy.deepcopy(posterior)
    transport_poison["worlds"][0]["transport_id"] = "9f" * 128
    variants.append(transport_poison)
    expected = _prediction(posterior)
    expected_probabilities = dict(zip(expected.legal_actions, expected.probabilities, strict=True))

    for variant in variants:
        actual = _prediction(variant)
        actual_probabilities = dict(zip(actual.legal_actions, actual.probabilities, strict=True))
        assert actual.value == pytest.approx(
            expected.value, abs=EVALUATOR_OUTPUT_ABS_TOLERANCE, rel=0
        )
        assert actual_probabilities == pytest.approx(
            expected_probabilities,
            abs=EVALUATOR_OUTPUT_ABS_TOLERANCE,
            rel=0,
        )
        assert actual.selected_action == expected.selected_action


@hostile_case(
    mutation="reverse legal-action serialization order at the real preprocessing and JAX boundary",
    expected="logit/probability association follows action identity and selected action is unchanged",
    threat="the model output at position i is attached to another action after an order change",
    layer="learned evaluator action pooling and output association",
    tier="simulator",
)
def test_real_model_action_outputs_follow_semantic_action_keys() -> None:
    pytest.importorskip("jax")
    np = pytest.importorskip("numpy")
    posterior = tiny_posterior()
    actions = ("move:a", "switch:rotom")
    params = init_params(SPEC, seed=1729)

    def result_for(action_order):
        inputs = build_evaluator_input(
            public_state=tiny_public_state(),
            posterior=posterior,
            legal_actions=action_order,
            spec=SPEC,
        )
        prediction = predict(params, inputs)
        _, logits = forward(params, inputs)
        return prediction, tuple(float(value) for value in np.asarray(logits))

    baseline, baseline_raw_logits = result_for(actions)
    reversed_result, reversed_raw_logits = result_for(tuple(reversed(actions)))
    baseline_logits = dict(zip(baseline.legal_actions, baseline_raw_logits, strict=True))
    baseline_probabilities = dict(zip(baseline.legal_actions, baseline.probabilities, strict=True))
    reversed_logits = dict(zip(reversed_result.legal_actions, reversed_raw_logits, strict=True))
    reversed_probabilities = dict(
        zip(reversed_result.legal_actions, reversed_result.probabilities, strict=True)
    )

    assert reversed_logits == pytest.approx(
        baseline_logits, abs=EVALUATOR_OUTPUT_ABS_TOLERANCE, rel=0
    )
    assert reversed_probabilities == pytest.approx(
        baseline_probabilities, abs=EVALUATOR_OUTPUT_ABS_TOLERANCE, rel=0
    )
    assert reversed_result.selected_action == baseline.selected_action


@hostile_case(
    mutation="pool a four-world posterior with highly unequal weights and a split duplicate atom",
    expected="real preprocessing and JAX output equal the equivalent three-atom semantic support",
    threat="pooling depends on row count or floating-point accumulation rather than posterior mass",
    layer="learned evaluator posterior pooling",
    tier="simulator",
)
def test_multiworld_pooling_is_invariant_to_duplicate_support_rows() -> None:
    pytest.importorskip("jax")
    posterior = {
        "treatment": "oracle",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {"world_id": "a", "weight": 0.01, "hidden": {"item": "band"}},
            {"world_id": "b", "weight": 0.09, "hidden": {"item": "scarf"}},
            {"world_id": "c", "weight": 0.30, "hidden": {"item": "specs"}},
            {"world_id": "d", "weight": 0.60, "hidden": {"item": "leftovers"}},
        ],
    }
    split = copy.deepcopy(posterior)
    split["worlds"] = [
        {**posterior["worlds"][0], "world_id": "a-1", "weight": 0.004},
        {**posterior["worlds"][0], "world_id": "a-2", "weight": 0.006},
        *posterior["worlds"][1:],
    ]
    baseline = _prediction(posterior)
    actual = _prediction(split)

    assert actual.value == pytest.approx(baseline.value, abs=EVALUATOR_OUTPUT_ABS_TOLERANCE, rel=0)
    assert dict(zip(actual.legal_actions, actual.probabilities, strict=True)) == pytest.approx(
        dict(zip(baseline.legal_actions, baseline.probabilities, strict=True)),
        abs=EVALUATOR_OUTPUT_ABS_TOLERANCE,
        rel=0,
    )
    assert actual.selected_action == baseline.selected_action


@hostile_case(
    mutation="change parameter values while moving an identical parameter tree between containers",
    expected="path/container metadata has no effect, while a changed learned tensor changes checkpoint identity",
    threat="matched arms silently use different learned models or reject equivalent model serialization",
    layer="evaluator checkpoint identity",
)
def test_checkpoint_identity_tracks_parameters_and_model_spec_only(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    params = {
        "head.bias": np.asarray([0.25, -0.5], dtype=np.float32),
        "head.weight": np.asarray([[1.0, 2.0]], dtype=np.float32),
    }
    digest = checkpoint_digest(params, SPEC)
    copied = {name: value.copy() for name, value in params.items()}
    assert checkpoint_digest(copied, SPEC) == digest

    first = write_checkpoint(
        tmp_path / "host-a" / "evaluator",
        params,
        SPEC,
        metadata={"note": "training host A"},
    )
    second = write_checkpoint(
        tmp_path / "host-b" / "evaluator",
        copied,
        SPEC,
        metadata={"note": "training host B"},
    )
    assert first["evaluator"]["checkpoint_digest"] == digest
    assert second["evaluator"]["checkpoint_digest"] == digest
    loaded, loaded_spec, _ = load_checkpoint(tmp_path / "host-a" / "evaluator")
    assert checkpoint_digest(loaded, loaded_spec) == digest

    changed = {name: value.copy() for name, value in params.items()}
    changed["head.bias"][0] += 1.0
    assert checkpoint_digest(changed, SPEC) != digest
    different_name = write_checkpoint(
        tmp_path / "renamed-checkpoint",
        changed,
        SPEC,
        metadata={"note": "changed learned semantics"},
    )
    same_name_different_semantics = write_checkpoint(
        tmp_path / "host-c" / "evaluator",
        changed,
        SPEC,
        metadata={"note": "same basename, changed parameters"},
    )
    assert different_name["evaluator"]["checkpoint_digest"] != digest
    assert same_name_different_semantics["evaluator"]["checkpoint_digest"] != digest

    other_spec = BeliefEvaluatorSpec(
        public_width=17,
        world_width=12,
        action_width=8,
        hidden_width=10,
        world_hidden_width=9,
    )
    assert checkpoint_digest(params, other_spec) != digest
