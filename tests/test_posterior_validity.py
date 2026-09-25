from __future__ import annotations

import math

import pytest

from azelficoast.posterior_validity import (
    PosteriorValidityError,
    aggregate_realized_support,
    posterior_diagnostics,
    power_reweight_posterior,
    score_realized_support,
    widen_posterior_support,
)


def _posterior() -> dict[str, object]:
    return {
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {"world_id": "a", "weight": 0.75, "hidden": {"item": "scarf"}},
            {"world_id": "b", "weight": 0.25, "hidden": {"item": "specs"}},
        ],
    }


def test_diagnostics_report_support_concentration() -> None:
    diagnostics = posterior_diagnostics(_posterior())

    assert diagnostics["support_size"] == 2
    assert abs(diagnostics["entropy_bits"] - 0.8112781244591328) < 1e-12
    assert abs(diagnostics["effective_sample_size"] - 1.6) < 1e-12
    assert diagnostics["maximum_mass"] == 0.75
    assert diagnostics["minimum_mass"] == 0.25


def test_realized_support_scoring_keeps_omissions_visible() -> None:
    covered = score_realized_support(_posterior(), realized_world_id="b")
    omitted = score_realized_support(_posterior(), realized_world_id="missing")
    aggregate = aggregate_realized_support([covered, omitted])

    assert covered["realized_state_in_support"] is True
    assert covered["assigned_mass"] == 0.25
    assert covered["log_loss_if_covered"] == -math.log(0.25)
    assert omitted["realized_state_in_support"] is False
    assert omitted["assigned_mass"] == 0.0
    assert omitted["log_loss_if_covered"] is None
    assert aggregate["realized_support_coverage_rate"] == 0.5
    assert aggregate["omitted_realized_state_count"] == 1


def test_power_reweight_can_flatten_without_changing_support() -> None:
    flattened = power_reweight_posterior(
        _posterior(),
        exponent=0.0,
        treatment="flattened",
    )

    assert [row["world_id"] for row in flattened["worlds"]] == ["a", "b"]
    assert [row["weight"] for row in flattened["worlds"]] == [0.5, 0.5]
    assert flattened["robustness_treatment"]["support_changed"] is False


def test_widening_support_requires_a_new_transition_program() -> None:
    widened = widen_posterior_support(
        _posterior(),
        alternative={
            "worlds": [
                {"world_id": "a", "weight": 0.2, "hidden": {"item": "scarf"}},
                {"world_id": "c", "weight": 0.8, "hidden": {"item": "band"}},
            ]
        },
        alternative_mass=0.25,
    )

    masses = {row["world_id"]: row["weight"] for row in widened["worlds"]}
    assert set(masses) == {"a", "b", "c"}
    assert abs(sum(masses.values()) - 1.0) < 1e-12
    assert widened["robustness_treatment"]["support_changed"] is True
    assert widened["robustness_treatment"]["requires_new_verified_transition_program"] is True


def test_widening_rejects_world_id_reuse_for_different_hidden_state() -> None:
    with pytest.raises(PosteriorValidityError, match="reuses id"):
        widen_posterior_support(
            _posterior(),
            alternative={
                "worlds": [
                    {"world_id": "a", "weight": 1.0, "hidden": {"item": "band"}},
                ]
            },
            alternative_mass=0.2,
        )
