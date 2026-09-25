from __future__ import annotations

import copy

import pytest

from azelficoast.joint_posterior import (
    JointPosteriorError,
    evaluator_posterior,
    validate_joint_posterior,
)


def _set(species: str, *, lead: bool, item: str = "leftovers") -> dict[str, object]:
    return {
        "species": species,
        "level": 80,
        "gender": "M",
        "ability": "pressure",
        "item": item,
        "moves": ["protect", "tackle"],
        "tera_type": "steel",
        "role": "Bulky Support",
        "nature": "Serious",
        "evs": {"hp": 85},
        "ivs": {"hp": 31},
        "was_lead": lead,
    }


def _posterior() -> dict[str, object]:
    first = {"team": [_set("zapdosgalar", lead=True, item="choicescarf"), _set("gougingfire", lead=False)]}
    second = {"team": [_set("zapdosgalar", lead=True, item="choiceband"), _set("swanna", lead=False)]}
    return {
        "schema": "azelficoast.joint-random-battle-posterior",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "construction": {
            "kind": "full-team-generator-rejection-particles",
            "preserves_joint_team_set_correlations": True,
        },
        "public_evidence": {
            "revealed": [
                {
                    "species": "zapdosgalar",
                    "level": 80,
                    "gender": "M",
                    "moves": ["tackle"],
                    "ability": "pressure",
                    "item": None,
                    "tera_type": None,
                }
            ]
        },
        "public_evidence_sha256": "evidence",
        "posterior_sha256": "posterior",
        "support_status": "sufficient",
        "worlds": [
            {"world_id": "a", "weight": 0.4, "sample_count": 4, "hidden": first},
            {"world_id": "b", "weight": 0.6, "sample_count": 6, "hidden": second},
        ],
    }


def test_joint_posterior_preserves_complete_team_particles() -> None:
    checked = validate_joint_posterior(_posterior())
    projected = evaluator_posterior(checked)

    assert len(projected["worlds"]) == 2
    assert projected["worlds"][0]["hidden"]["team"][0]["item"] == "choicescarf"
    assert projected["worlds"][0]["hidden"]["team"][1]["species"] == "gougingfire"
    assert "sample_count" not in projected["worlds"][0]


def test_joint_posterior_rejects_factorized_or_incomplete_construction() -> None:
    document = _posterior()
    document["construction"]["preserves_joint_team_set_correlations"] = False

    with pytest.raises(JointPosteriorError, match="does not preserve"):
        validate_joint_posterior(document)


def test_joint_posterior_rejects_particle_that_contradicts_public_move() -> None:
    document = _posterior()
    document["worlds"][0]["hidden"]["team"][0]["moves"] = ["protect"]

    with pytest.raises(JointPosteriorError, match="revealed moves"):
        validate_joint_posterior(document)


def test_joint_posterior_rejects_duplicate_species_within_particle() -> None:
    document = _posterior()
    duplicate = copy.deepcopy(document["worlds"][0]["hidden"]["team"][0])
    duplicate["was_lead"] = False
    document["worlds"][0]["hidden"]["team"][1] = duplicate

    with pytest.raises(JointPosteriorError, match="duplicate species"):
        validate_joint_posterior(document)


def test_joint_posterior_can_be_inspected_but_not_consumed_when_support_is_low() -> None:
    document = _posterior()
    document["support_status"] = "insufficient"

    validate_joint_posterior(document, require_sufficient_support=False)
    with pytest.raises(JointPosteriorError, match="insufficient"):
        validate_joint_posterior(document)
