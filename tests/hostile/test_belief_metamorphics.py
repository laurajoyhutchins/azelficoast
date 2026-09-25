from __future__ import annotations

from collections import defaultdict
from math import fsum
from typing import Any, Mapping

import pytest

from azelficoast.belief.evaluator import BeliefEvaluatorError, build_evaluator_input
from hostile.fixtures import (
    WEIGHT_NORMALIZATION_ABS_TOLERANCE,
    hostile_case,
    tiny_posterior,
    tiny_public_state,
)
from hostile.transforms import (
    merge_equivalent_worlds,
    permute_actions,
    permute_support,
    perturb_hidden_field,
    rename_world_ids,
    split_world,
)


def _weighted_feature_measure(posterior: Mapping[str, Any]) -> dict[tuple[float, ...], float]:
    inputs = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=posterior,
        legal_actions=["move:a", "move:b"],
    )
    mass_by_feature: dict[tuple[float, ...], list[float]] = defaultdict(list)
    for feature, weight in zip(inputs.world_features, inputs.world_weights, strict=True):
        mass_by_feature[feature].append(weight)
    return {feature: fsum(weights) for feature, weights in mass_by_feature.items()}


@hostile_case(
    mutation="replace every opaque world_id with unrelated strings and integers",
    expected="each semantic world's evaluator feature row and posterior mass are unchanged",
    threat="transport identity becomes an accidental feature correlated with hidden semantics",
    layer="belief/evaluator input",
)
def test_world_id_renaming_does_not_change_the_weighted_feature_measure() -> None:
    posterior = tiny_posterior()
    renamed = rename_world_ids(posterior, ["good-world-9", "world-negative-seventy-three"])

    assert _weighted_feature_measure(renamed) == _weighted_feature_measure(posterior)


@hostile_case(
    mutation="poison an explicitly transport-only field with a hidden-correlated value",
    expected="transport_id is either rejected or excluded from evaluator features",
    threat="transport plumbing smuggles hidden-world identity into a public decision",
    layer="belief/evaluator input",
)
def test_transport_id_poison_does_not_change_evaluator_features() -> None:
    posterior = tiny_posterior()
    poisoned = perturb_hidden_field(posterior, 0, ("transport_id",), 2**31)

    assert _weighted_feature_measure(poisoned) == _weighted_feature_measure(posterior)


@hostile_case(
    mutation="reverse and fixed-shuffle the posterior support order",
    expected="the weighted semantic support measure is identical",
    threat="container ordering silently changes which hidden world receives which mass",
    layer="belief/evaluator input",
)
@pytest.mark.parametrize("order", [(1, 0), (0, 1)])
def test_support_permutation_preserves_world_feature_mass(order: tuple[int, ...]) -> None:
    posterior = tiny_posterior()

    assert _weighted_feature_measure(permute_support(posterior, order)) == (
        _weighted_feature_measure(posterior)
    )


@hostile_case(
    mutation="reverse and fixed-shuffle a six-world posterior support",
    expected="preprocessed semantic rows, normalized weights, and weighted feature measure remain exact",
    threat="support-order accumulation or positional coupling changes a multiworld belief",
    layer="belief evaluator preprocessing",
)
@pytest.mark.parametrize(
    "order",
    [
        (5, 4, 3, 2, 1, 0),
        (3, 0, 5, 2, 1, 4),
        (1, 4, 0, 5, 3, 2),
    ],
)
def test_multiworld_support_permutations_preserve_exact_preprocessing(order) -> None:
    posterior = {
        "treatment": "oracle",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {
                "world_id": f"opaque-{index}",
                "weight": weight,
                "hidden": {"item": item, "speed_tier": index % 3},
            }
            for index, (weight, item) in enumerate(
                zip(
                    (0.01, 0.04, 0.10, 0.20, 0.25, 0.40),
                    ("band", "scarf", "specs", "leftovers", "boots", "berry"),
                    strict=True,
                )
            )
        ],
    }
    baseline = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=posterior,
        legal_actions=["move:a", "move:b"],
    )
    transformed = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=permute_support(posterior, order),
        legal_actions=["move:a", "move:b"],
    )

    assert transformed.world_features == baseline.world_features
    assert transformed.world_weights == baseline.world_weights
    assert _weighted_feature_measure(permute_support(posterior, order)) == (
        _weighted_feature_measure(posterior)
    )


@hostile_case(
    mutation="split one support atom into two identical semantic worlds with conserved mass",
    expected="weighted world-feature mass and pooled evaluator output are unchanged",
    threat="the implementation counts support rows instead of integrating posterior mass",
    layer="belief/evaluator pooling",
)
@pytest.mark.parametrize(
    ("base_mass", "parts", "ids"),
    [
        (0.25, (0.1, 0.15), ("split-a", "split-b")),
        (0.25, (0.0625, 0.0625, 0.0625, 0.0625), ("s0", "s1", "s2", "s3")),
        (0.4, (0.2, 0.2), ("equal-a", "equal-b")),
        (0.4, (0.39, 0.01), ("unequal-a", "unequal-b")),
        (0.4, (0.1, 0.1, 0.1, 0.1), ("four-a", "four-b", "four-c", "four-d")),
    ],
)
def test_equivalent_support_split_preserves_weighted_feature_mass(
    base_mass: float,
    parts: tuple[float, ...],
    ids: tuple[str, ...],
) -> None:
    posterior = tiny_posterior()
    posterior["worlds"][0]["weight"] = base_mass
    posterior["worlds"][1]["weight"] = 1.0 - base_mass

    split = split_world(posterior, 0, parts, identifiers=ids)

    assert _weighted_feature_measure(split) == _weighted_feature_measure(posterior)


@hostile_case(
    mutation="split one hidden-world atom into duplicate semantic support rows",
    expected="preprocessed semantic support coalesces to the exact same rows and weights",
    threat="support multiplicity changes the evaluator's entropy/effective-support features",
    layer="belief/evaluator preprocessing",
)
def test_evaluator_input_coalesces_equivalent_support_before_pooling() -> None:
    posterior = tiny_posterior()
    split = split_world(posterior, 0, (0.1, 0.15), identifiers=("a", "b"))
    baseline = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=posterior,
        legal_actions=["move:a", "move:b"],
    )

    actual = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=split,
        legal_actions=["move:a", "move:b"],
    )

    assert actual.world_features == baseline.world_features
    assert actual.world_weights == baseline.world_weights


@hostile_case(
    mutation="merge duplicate semantic support atoms while adding their posterior mass",
    expected="weighted world-feature mass is unchanged",
    threat="compressed support changes the represented posterior",
    layer="belief/evaluator input",
)
def test_equivalent_support_merge_preserves_weighted_feature_mass() -> None:
    posterior = tiny_posterior()
    duplicate = split_world(posterior, 0, (0.1, 0.15))

    assert _weighted_feature_measure(merge_equivalent_worlds(duplicate)) == (
        _weighted_feature_measure(posterior)
    )


@hostile_case(
    mutation="permute legal actions without changing their semantic IDs",
    expected="action feature association follows each action ID rather than its index",
    threat="positional coupling pairs one action with another action's evaluator output",
    layer="evaluator action boundary",
)
def test_action_permutation_preserves_action_feature_association() -> None:
    actions = ["move:a", "switch:rotom"]
    original = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=tiny_posterior(),
        legal_actions=actions,
    )
    reordered_actions = permute_actions(actions, (1, 0))
    reordered = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=tiny_posterior(),
        legal_actions=reordered_actions,
    )

    assert dict(zip(original.legal_actions, original.action_features, strict=True)) == dict(
        zip(reordered.legal_actions, reordered.action_features, strict=True)
    )


@hostile_case(
    mutation="insert the realized hidden world below public metadata and posterior-world payloads",
    expected="the public-belief constructor rejects every nested realized-world marker",
    threat="generic feature flattening makes an information-set search condition on the sampled world",
    layer="public belief/evaluator boundary",
)
@pytest.mark.parametrize(
    "location",
    [
        "explicit_public_field",
        "public_metadata",
        "nested_state",
        "evaluator_feature_payload",
        "transport_world_field",
        "posterior_annotation",
        "auxiliary_context",
    ],
)
def test_realized_world_payload_is_rejected_through_every_input_route(location: str) -> None:
    public_state = tiny_public_state()
    posterior = tiny_posterior()
    if location == "explicit_public_field":
        public_state["realized_hidden_state"] = {"item": "choiceband"}
    elif location == "public_metadata":
        public_state["metadata"] = {"realized_hidden_state": {"item": "choiceband"}}
    elif location == "nested_state":
        public_state["nested"] = {"actual_hidden_world": {"item": "choiceband"}}
    elif location == "evaluator_feature_payload":
        public_state["evaluator_features"] = {"sampled_world": "opaque:alpha"}
    elif location == "transport_world_field":
        posterior["worlds"][0]["transport_realized_hidden_world"] = "opaque:alpha"
    elif location == "posterior_annotation":
        posterior["annotation"] = {"actual_world": "opaque:alpha"}
    else:
        public_state["auxiliary_context"] = {"sampled_hidden_state": {"item": "band"}}

    with pytest.raises(BeliefEvaluatorError, match="realized|information"):
        build_evaluator_input(
            public_state=public_state,
            posterior=posterior,
            legal_actions=["move:a"],
        )


@hostile_case(
    mutation="scale otherwise identical positive posterior weights near floating overflow",
    expected="normalization remains finite and returns the same 2:1 probability ratio",
    threat="finite masses overflow during summation and silently erase all belief mass",
    layer="posterior validation/normalization",
)
def test_finite_large_weight_scale_normalizes_without_losing_mass() -> None:
    posterior = tiny_posterior()
    posterior["worlds"][0]["weight"] = 1.2e308
    posterior["worlds"][1]["weight"] = 0.6e308

    inputs = build_evaluator_input(
        public_state=tiny_public_state(),
        posterior=posterior,
        legal_actions=["move:a"],
    )

    assert inputs.world_weights == pytest.approx(
        (2 / 3, 1 / 3), abs=WEIGHT_NORMALIZATION_ABS_TOLERANCE, rel=0
    )
    assert fsum(inputs.world_weights) == pytest.approx(
        1.0, abs=WEIGHT_NORMALIZATION_ABS_TOLERANCE, rel=0
    )


@hostile_case(
    mutation="multiply all valid posterior weights by a positive constant",
    expected="normalized posterior weights are exactly unchanged",
    threat="different call sites silently treat proportional mass vectors differently",
    layer="posterior validation/normalization",
)
def test_positive_weight_rescaling_preserves_normalized_posterior() -> None:
    posterior = tiny_posterior()
    scaled = {
        **posterior,
        "worlds": [{**world, "weight": world["weight"] * 40} for world in posterior["worlds"]],
    }

    assert _weighted_feature_measure(scaled) == _weighted_feature_measure(posterior)


@hostile_case(
    mutation="swap which opaque transport label is associated with each private world, then add a public revealing observation",
    expected="before reveal evaluator inputs are identical; after reveal the public input and conditioned belief distinguish the worlds",
    threat="the evaluator conditions on realized private state too early or ignores legitimate public evidence",
    layer="public-belief input boundary",
)
def test_private_field_canary_is_blind_before_reveal_and_sensitive_after() -> None:
    posterior = tiny_posterior()
    public_state = tiny_public_state()
    swapped_transport = rename_world_ids(posterior, ["opaque:beta", "opaque:alpha"])

    before_a = build_evaluator_input(
        public_state=public_state,
        posterior=posterior,
        legal_actions=["move:a", "move:b"],
    )
    before_b = build_evaluator_input(
        public_state=public_state,
        posterior=swapped_transport,
        legal_actions=["move:a", "move:b"],
    )
    assert before_a == before_b

    revealed_state = {**public_state, "observed_item": "choiceband"}
    revealed_posterior = {
        **posterior,
        "worlds": [
            {**posterior["worlds"][0], "weight": 1.0},
        ],
    }
    after = build_evaluator_input(
        public_state=revealed_state,
        posterior=revealed_posterior,
        legal_actions=["move:a", "move:b"],
    )
    assert after.public_features != before_a.public_features
    assert after.world_features != before_a.world_features


@hostile_case(
    mutation="supply a negative, zero, NaN, or infinite world mass",
    expected="posterior construction raises a validation error before evaluation",
    threat="invalid probability mass is silently dropped, renormalized, or propagated",
    layer="posterior validation",
)
@pytest.mark.parametrize("weight", [-0.1, 0.0, float("nan"), float("inf")])
def test_invalid_world_mass_fails_closed(weight: float) -> None:
    posterior = tiny_posterior()
    posterior["worlds"][0]["weight"] = weight

    with pytest.raises(BeliefEvaluatorError, match="weights must be positive and finite"):
        build_evaluator_input(
            public_state=tiny_public_state(),
            posterior=posterior,
            legal_actions=["move:a"],
        )
