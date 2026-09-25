from __future__ import annotations

import copy

import pytest

from azelficoast.belief.treatments import (
    PosteriorTreatmentError,
    build_posterior,
    generator_faithful_posterior,
    oracle_posterior,
    practical_posterior,
)


def _oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": "band-a",
            "weight": 0.45,
            "hidden": {
                "opponent.active.item": "Choice Band",
                "opponent.active.ability": "A",
                "opponent.active.exact_hp": 100,
            },
            "provenance": {
                "generator_rounds": 2048,
                "generator_count": 900,
            },
        },
        {
            "world_id": "band-b",
            "weight": 0.15,
            "hidden": {
                "opponent.active.item": "Choice Band",
                "opponent.active.ability": "B",
                "opponent.active.exact_hp": 90,
            },
            "provenance": {
                "generator_rounds": 2048,
                "generator_count": 300,
            },
        },
        {
            "world_id": "scarf-a",
            "weight": 0.10,
            "hidden": {
                "opponent.active.item": "Choice Scarf",
                "opponent.active.ability": "A",
                "opponent.active.exact_hp": 100,
            },
            "provenance": {
                "generator_rounds": 2048,
                "generator_count": 200,
            },
        },
        {
            "world_id": "scarf-b",
            "weight": 0.30,
            "hidden": {
                "opponent.active.item": "Choice Scarf",
                "opponent.active.ability": "B",
                "opponent.active.exact_hp": 90,
            },
            "provenance": {
                "generator_rounds": 2048,
                "generator_count": 600,
            },
        },
    ]
    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "reconstruction": {
            "generator_rounds": 2048,
            "generator_matches": 2000,
        },
        "worlds": worlds,
        "legal_actions": ["a"],
        "transitions": [],
    }


def test_generator_faithful_preserves_correlated_world_mass() -> None:
    oracle = _oracle()
    posterior = generator_faithful_posterior(oracle)

    assert posterior["treatment"] == "generator_faithful"
    assert posterior["conditioned_on_public_history"] is True
    assert posterior["realized_hidden_state_revealed"] is False
    assert posterior["construction"]["preserves_joint_hidden_worlds"] is True
    assert [world["weight"] for world in posterior["worlds"]] == [
        0.45,
        0.15,
        0.10,
        0.30,
    ]


def test_practical_preserves_item_mass_but_discards_within_item_frequency() -> None:
    posterior = practical_posterior(_oracle())
    by_id = {world["world_id"]: world for world in posterior["worlds"]}

    assert posterior["treatment"] == "practical"
    assert abs(by_id["band-a"]["weight"] - 0.30) < 1e-12
    assert abs(by_id["band-b"]["weight"] - 0.30) < 1e-12
    assert abs(by_id["scarf-a"]["weight"] - 0.20) < 1e-12
    assert abs(by_id["scarf-b"]["weight"] - 0.20) < 1e-12
    assert posterior["construction"]["preserves_item_marginal_mass"] is True
    assert posterior["construction"]["preserves_joint_hidden_world_support"] is True


def test_oracle_treatment_fails_closed_on_sampled_generator_reconstruction() -> None:
    with pytest.raises(
        PosteriorTreatmentError,
        match="requires exact_conditional",
    ):
        oracle_posterior(_oracle())


def test_oracle_treatment_requires_explicit_exact_conditional_evidence() -> None:
    oracle = copy.deepcopy(_oracle())
    reconstruction = oracle["reconstruction"]
    assert isinstance(reconstruction, dict)
    reconstruction["posterior_authority"] = "exact_conditional"

    with pytest.raises(
        PosteriorTreatmentError,
        match="lacks exact conditional evidence",
    ):
        oracle_posterior(oracle)

    reconstruction["exact_conditional_evidence"] = {
        "generator_model": "enumerated-test-model",
        "conditioned_public_history_digest": "abc",
    }
    posterior = oracle_posterior(oracle)
    assert posterior["treatment"] == "oracle"
    assert posterior["construction"]["realized_hidden_state_used"] is False


def test_generator_faithful_rejects_unbound_world_provenance() -> None:
    oracle = _oracle()
    worlds = oracle["worlds"]
    assert isinstance(worlds, list)
    first = worlds[0]
    assert isinstance(first, dict)
    provenance = first["provenance"]
    assert isinstance(provenance, dict)
    provenance["generator_rounds"] = 1024

    with pytest.raises(
        PosteriorTreatmentError,
        match="not bound to the reconstruction",
    ):
        generator_faithful_posterior(oracle)



def test_support_preserving_prior_stress_treatments() -> None:
    flattened = build_posterior(_oracle(), treatment="flattened")
    sharpened = build_posterior(_oracle(), treatment="sharpened")

    assert flattened["treatment"] == "flattened"
    assert [world["weight"] for world in flattened["worlds"]] == [0.25] * 4
    assert flattened["robustness_treatment"]["support_changed"] is False

    expected = [0.45**2, 0.15**2, 0.10**2, 0.30**2]
    total = sum(expected)
    assert sharpened["treatment"] == "sharpened"
    assert all(
        abs(world["weight"] - weight / total) < 1e-12
        for world, weight in zip(sharpened["worlds"], expected, strict=True)
    )
