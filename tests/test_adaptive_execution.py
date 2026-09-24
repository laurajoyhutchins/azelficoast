from __future__ import annotations

import pytest

from azelficoast.adaptive_execution import (
    ExecutionCostProfile,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
)
from azelficoast.adaptive_execution_experiment import _fit_cost_profile


def _profile(*, guard: float = 0.2) -> ExecutionCostProfile:
    return ExecutionCostProfile(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:test",
        direct_intercept_ms=0.1,
        direct_per_world_ms=0.001,
        direct_per_world_squared_ms=0.0,
        projected_intercept_ms=0.4,
        projected_per_canonical_class_ms=0.0001,
        projected_per_execution_class_ms=0.001,
        uncertainty_guard_ms=guard,
        calibrated_max_logical_world_count=100_000,
        calibrated_max_canonical_classes=1_000,
        calibrated_max_projected_classes=500,
    )


def _features(
    worlds: int,
    *,
    canonical: int = 128,
    projected: int = 16,
) -> ExecutionFeatures:
    return ExecutionFeatures(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:test",
        logical_world_count=worlds,
        active_canonical_classes=canonical,
        active_projected_classes=projected,
    )


def test_structural_model_selects_direct_then_projected_as_multiplicity_grows() -> None:
    profile = _profile()

    small = choose_execution_path(profile, _features(100))
    large = choose_execution_path(profile, _features(10_000))

    assert small.path is ExecutionPath.DIRECT
    assert large.path is ExecutionPath.PROJECTED
    assert small.predicted_direct_ms < small.predicted_projected_ms
    assert large.predicted_projected_ms < large.predicted_direct_ms


def test_uncertainty_is_diagnostic_not_a_path_override() -> None:
    profile = _profile()
    features = _features(500, canonical=100, projected=10)

    decision = choose_execution_path(profile, features)

    assert decision.predicted_projected_minus_direct_ms < 0
    assert decision.within_uncertainty_guard is True
    assert decision.path is ExecutionPath.PROJECTED


def test_costs_are_monotone_in_their_work_dimensions() -> None:
    profile = _profile()
    base_features = _features(1000)
    more_worlds = _features(2000)
    more_canonical = _features(1000, canonical=256, projected=16)
    more_projected = _features(1000, canonical=128, projected=32)

    assert profile.estimate_direct_ms(more_worlds) > profile.estimate_direct_ms(base_features)
    assert profile.estimate_projected_ms(more_canonical) > profile.estimate_projected_ms(
        base_features
    )
    assert profile.estimate_projected_ms(more_projected) > profile.estimate_projected_ms(
        base_features
    )


def test_exact_prediction_tie_prefers_direct() -> None:
    profile = ExecutionCostProfile(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:test",
        direct_intercept_ms=1.0,
        direct_per_world_ms=0.0,
        direct_per_world_squared_ms=0.0,
        projected_intercept_ms=1.0,
        projected_per_canonical_class_ms=0.0,
        projected_per_execution_class_ms=0.0,
        uncertainty_guard_ms=0.1,
        calibrated_max_logical_world_count=100_000,
        calibrated_max_canonical_classes=1_000,
        calibrated_max_projected_classes=500,
    )
    assert choose_execution_path(profile, _features(100)).path is ExecutionPath.DIRECT


def test_profile_fails_closed_on_identity_or_calibration_domain_mismatch() -> None:
    profile = _profile()

    with pytest.raises(ValueError, match="backend"):
        choose_execution_path(
            profile,
            ExecutionFeatures(
                backend="gpu",
                target_signature="sha256:target",
                effect_signature="sha256:test",
                logical_world_count=100,
                active_canonical_classes=10,
                active_projected_classes=5,
            ),
        )

    with pytest.raises(ValueError, match="effect signature"):
        choose_execution_path(
            profile,
            ExecutionFeatures(
                backend="cpu",
                target_signature="sha256:target",
                effect_signature="sha256:other",
                logical_world_count=100,
                active_canonical_classes=10,
                active_projected_classes=5,
            ),
        )

    with pytest.raises(ValueError, match="target signature"):
        choose_execution_path(
            profile,
            ExecutionFeatures(
                backend="cpu",
                target_signature="sha256:other-target",
                effect_signature="sha256:test",
                logical_world_count=100,
                active_canonical_classes=10,
                active_projected_classes=5,
            ),
        )

    with pytest.raises(ValueError, match="calibrated"):
        choose_execution_path(profile, _features(100_001))


def test_cost_profiles_reject_negative_coefficients() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ExecutionCostProfile(
            backend="cpu",
            target_signature="sha256:target",
            effect_signature="sha256:test",
            direct_intercept_ms=0.0,
            direct_per_world_ms=-1.0,
            direct_per_world_squared_ms=0.0,
            projected_intercept_ms=0.0,
            projected_per_canonical_class_ms=0.0,
            projected_per_execution_class_ms=0.0,
            uncertainty_guard_ms=0.0,
            calibrated_max_logical_world_count=100,
            calibrated_max_canonical_classes=10,
            calibrated_max_projected_classes=5,
        )



def test_model_selection_prefers_simpler_shape_when_choice_accuracy_ties() -> None:
    # Frozen whole-attack calibration witness: quadratic shapes fit large-world
    # latency much more closely, but all shapes make the same number of
    # leave-one-out path decisions. Dispatch should therefore keep the simpler
    # linear boundary instead of spending complexity on irrelevant magnitude.
    observed = (
        (6144, 0.538380, 0.594276, 0.034118, 0.038052),
        (8192, 0.728377, 0.588846, 0.015900, 0.011110),
        (12288, 0.931238, 0.598484, 0.116800, 0.009537),
        (16384, 1.184323, 0.594385, 0.035737, 0.023874),
        (24576, 1.445684, 0.619863, 0.058089, 0.048751),
        (32768, 2.012328, 0.582323, 0.109917, 0.004950),
        (65536, 4.213745, 0.637075, 0.252663, 0.065984),
        (131072, 7.418286, 0.631304, 0.206898, 0.049693),
        (262144, 16.624287, 0.693781, 0.951895, 0.062286),
        (524288, 38.994405, 0.681559, 0.975217, 0.056927),
    )
    rows = [
        {
            "logical_world_count": worlds,
            "active_canonical_classes": 4800,
            "active_projected_classes": 49,
            "direct_median_ms": direct,
            "direct_mad_ms": direct_mad,
            "projected_median_ms": projected,
            "projected_mad_ms": projected_mad,
            "projected_minus_direct_median_ms": projected - direct,
            "projected_minus_direct_mad_ms": direct_mad + projected_mad,
        }
        for worlds, direct, projected, direct_mad, projected_mad in observed
    ]

    profile, selection = _fit_cost_profile(
        rows,
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:attack",
    )

    assert selection["selected_shape"] == "direct-linear__projected-execution"
    assert selection["selection_policy"] == (
        "choice-accuracy_then_complexity_then-delta-mae"
    )
    assert profile.direct_per_world_squared_ms == 0.0
