from __future__ import annotations

import pytest

from azelficoast.adaptive_execution import (
    ExecutionCostProfile,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
)


def _profile() -> ExecutionCostProfile:
    return ExecutionCostProfile(
        backend="cpu",
        effect_signature="sha256:test",
        projected_fixed_overhead_ms=0.4,
        direct_per_world_ms=0.001,
        direct_per_world_squared_ms=0.00000001,
        projected_per_canonical_class_ms=0.0001,
        projected_per_execution_class_ms=0.001,
        uncertainty_fixed_ms=0.05,
        uncertainty_per_world_ms=0.00001,
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
        effect_signature="sha256:test",
        logical_world_count=worlds,
        active_canonical_classes=canonical,
        active_projected_classes=projected,
    )


def test_relative_model_selects_direct_then_projected_as_multiplicity_grows() -> None:
    profile = _profile()

    small = choose_execution_path(profile, _features(100))
    large = choose_execution_path(profile, _features(10_000))

    assert small.path is ExecutionPath.DIRECT
    assert large.path is ExecutionPath.PROJECTED
    assert small.predicted_projected_minus_direct_ms > 0
    assert large.predicted_projected_minus_direct_ms < 0


def test_uncertainty_is_diagnostic_not_a_path_override() -> None:
    profile = _profile()
    features = _features(450, canonical=100, projected=10)

    decision = choose_execution_path(profile, features)

    assert decision.predicted_projected_minus_direct_ms < 0
    assert decision.within_uncertainty_guard is True
    assert decision.path is ExecutionPath.PROJECTED


def test_relative_cost_is_monotone_in_work_dimensions() -> None:
    profile = _profile()
    base = profile.estimate_projected_minus_direct_ms(_features(1000))
    more_worlds = profile.estimate_projected_minus_direct_ms(_features(2000))
    more_canonical = profile.estimate_projected_minus_direct_ms(
        _features(1000, canonical=256, projected=16)
    )
    more_projected = profile.estimate_projected_minus_direct_ms(
        _features(1000, canonical=128, projected=32)
    )

    assert more_worlds < base
    assert more_canonical > base
    assert more_projected > base


def test_profile_fails_closed_on_identity_or_calibration_domain_mismatch() -> None:
    profile = _profile()

    with pytest.raises(ValueError, match="backend"):
        choose_execution_path(
            profile,
            ExecutionFeatures(
                backend="gpu",
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
                effect_signature="sha256:other",
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
            effect_signature="sha256:test",
            projected_fixed_overhead_ms=0.0,
            direct_per_world_ms=-1.0,
            direct_per_world_squared_ms=0.0,
            projected_per_canonical_class_ms=0.0,
            projected_per_execution_class_ms=0.0,
            uncertainty_fixed_ms=0.0,
            uncertainty_per_world_ms=0.0,
            calibrated_max_logical_world_count=100,
            calibrated_max_canonical_classes=10,
            calibrated_max_projected_classes=5,
        )
