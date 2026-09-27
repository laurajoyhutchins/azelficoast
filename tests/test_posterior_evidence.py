from __future__ import annotations

import copy

import pytest

from azelficoast.research.posterior_evidence import (
    PosteriorEvidenceError,
    bind_best_available_conditional,
)


def _worlds() -> list[dict[str, object]]:
    return [
        {
            "world_id": "a",
            "weight": 0.75,
            "hidden": {"opponent.active.item": "Band"},
        },
        {
            "world_id": "b",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Scarf"},
        },
    ]


def _oracle() -> dict[str, object]:
    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "reconstruction": {
            "generator_rounds": 65536,
            "generator_matches": 60000,
            "generator_seed_schedule": "diagonal-counter-[i,i,i,i]",
            "generator_seed_start": 0,
        },
        "worlds": _worlds(),
    }


def _reference() -> dict[str, object]:
    worlds = copy.deepcopy(_worlds())
    worlds[0]["weight"] = 0.6
    worlds[1]["weight"] = 0.4
    return {
        "schema": "azelficoast.live-belief-posterior",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "reconstruction": {
            "generator_rounds": 2048,
            "generator_matches": 1800,
            "generator_seed_schedule": "diagonal-counter-[i,i,i,i]",
            "generator_seed_start": 0,
        },
        "worlds": worlds,
    }


def _authority() -> dict[str, object]:
    return {
        "kind": "best_available_conditional",
        "oracle_generator_rounds": 65536,
        "generator_faithful_rounds": 2048,
        "generator_seed_schedule": "diagonal-counter-[i,i,i,i]",
        "realized_hidden_state_used": False,
    }


def test_best_available_conditional_binds_dense_and_reference_evidence() -> None:
    result = bind_best_available_conditional(
        oracle=_oracle(),
        generator_reference=_reference(),
        authority=_authority(),
    )
    reconstruction = result["reconstruction"]
    assert reconstruction["posterior_authority"] == "best_available_conditional"
    assert reconstruction["best_available_conditional_evidence"]["exact"] is False
    weights = reconstruction["generator_faithful_reference"]["world_weights"]
    assert weights == [
        {"world_id": "a", "weight": 0.6},
        {"world_id": "b", "weight": 0.4},
    ]


def test_conditional_binding_rejects_support_drift() -> None:
    reference = _reference()
    worlds = reference["worlds"]
    assert isinstance(worlds, list)
    worlds.pop()

    with pytest.raises(PosteriorEvidenceError, match="disagree on hidden-world support"):
        bind_best_available_conditional(
            oracle=_oracle(),
            generator_reference=reference,
            authority=_authority(),
        )


def test_conditional_binding_rejects_realized_hidden_state_authority() -> None:
    authority = _authority()
    authority["realized_hidden_state_used"] = True

    with pytest.raises(PosteriorEvidenceError, match="realized hidden state"):
        bind_best_available_conditional(
            oracle=_oracle(),
            generator_reference=_reference(),
            authority=authority,
        )
