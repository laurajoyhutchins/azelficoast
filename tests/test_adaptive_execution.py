from __future__ import annotations

import pytest

from azelficoast.adaptive_execution import (
    ExecutionCostProfile,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
)


def _profile(*, guard: float = 0.05) -> ExecutionCostProfile:
    return ExecutionCostProfile(
        backend="cpu",
        effect_signature="sha256:test",
        projected_fixed_overhead_ms=0.4,
        direct_per_world_ms=0.001,
        projected_per_canonical_class_ms=0.0001,
        projected_per_execution_class_ms=0.001,
        decision_guard_ms=guard,
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
    assert large.predicted_projected_minus_direct_ms < -profile.decision_guard_ms


def test_uncertainty_guard_prefers_direct_near_crossover() -> None:
    profile = _profile(guard=0.2)
    features = _features(500, canonical=100, projected=10)

    decision = choose_execution_path(profile, features)

    assert decision.predicted_projected_minus_direct_ms < 0
    assert decision.predicted_projected_minus_direct_ms > -profile.decision_guard_ms
    assert decision.within_uncertainty_guard is True
    assert decision.path is ExecutionPath.DIRECT


def test_relative_cost_is_monotone_in_work_dimensions() -> None:
    profile = _profile(guard=0.0)
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


def test_profile_fails_closed_on_backend_or_effect_mismatch() -> None:
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


def test_cost_profiles_reject_negative_coefficients() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        ExecutionCostProfile(
            backend="cpu",
            effect_signature="sha256:test",
            projected_fixed_overhead_ms=0.0,
            direct_per_world_ms=-1.0,
            projected_per_canonical_class_ms=0.0,
            projected_per_execution_class_ms=0.0,
            decision_guard_ms=0.0,
        )
