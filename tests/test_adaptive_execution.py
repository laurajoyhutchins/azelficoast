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
        direct_intercept_ms=0.1,
        direct_per_world_ms=0.001,
        projected_intercept_ms=0.4,
        projected_per_canonical_class_ms=0.0001,
        projected_per_execution_class_ms=0.001,
    )


def _features(worlds: int) -> ExecutionFeatures:
    return ExecutionFeatures(
        backend="cpu",
        effect_signature="sha256:test",
        logical_world_count=worlds,
        active_canonical_classes=128,
        active_projected_classes=16,
    )


def test_dispatcher_selects_from_predicted_cost_not_fixed_population_threshold() -> None:
    profile = _profile()

    small = choose_execution_path(profile, _features(100))
    large = choose_execution_path(profile, _features(10_000))

    assert small.path is ExecutionPath.DIRECT
    assert large.path is ExecutionPath.PROJECTED
    assert small.predicted_direct_ms < small.predicted_projected_ms
    assert large.predicted_projected_ms < large.predicted_direct_ms


def test_exact_prediction_tie_prefers_direct() -> None:
    profile = ExecutionCostProfile(
        backend="cpu",
        effect_signature="sha256:test",
        direct_intercept_ms=1.0,
        direct_per_world_ms=0.0,
        projected_intercept_ms=1.0,
        projected_per_canonical_class_ms=0.0,
        projected_per_execution_class_ms=0.0,
    )
    assert choose_execution_path(profile, _features(100)).path is ExecutionPath.DIRECT


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
            direct_intercept_ms=0.0,
            direct_per_world_ms=-1.0,
            projected_intercept_ms=0.0,
            projected_per_canonical_class_ms=0.0,
            projected_per_execution_class_ms=0.0,
        )
