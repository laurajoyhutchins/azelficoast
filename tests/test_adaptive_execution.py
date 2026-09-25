from __future__ import annotations

import pytest

from azelficoast.core.costing import (
    ExecutionCostProfile,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
)


def _profile(
    *,
    guard: float = 0.2,
    saturation_start: int = 0,
    saturation_per_world: float = 0.0,
) -> ExecutionCostProfile:
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
        direct_saturation_start_worlds=saturation_start,
        direct_saturation_per_world_ms=saturation_per_world,
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


def test_direct_saturation_hinge_preserves_low_work_slope() -> None:
    linear = _profile()
    saturated = _profile(
        saturation_start=1000,
        saturation_per_world=0.002,
    )

    below = _features(500)
    above = _features(2000)

    assert saturated.estimate_direct_ms(below) == linear.estimate_direct_ms(below)
    assert saturated.estimate_direct_ms(above) > linear.estimate_direct_ms(above)


def test_positive_saturation_cost_requires_positive_knee() -> None:
    with pytest.raises(ValueError, match="saturation start"):
        _profile(saturation_start=0, saturation_per_world=0.001)


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




def test_absolute_baseline_mapping_survives_structural_tuple_growth() -> None:
    pytest.importorskip("jax")
    from azelficoast.research.adaptive_execution_experiment import (
        _baseline_delta,
        _fit_absolute_baseline,
    )

    rows = [
        {
            "logical_world_count": 1000,
            "active_canonical_classes": 10,
            "active_projected_classes": 5,
            "direct_median_ms": 2.0,
            "projected_median_ms": 4.5,
        },
        {
            "logical_world_count": 2000,
            "active_canonical_classes": 20,
            "active_projected_classes": 5,
            "direct_median_ms": 3.0,
            "projected_median_ms": 5.5,
        },
        {
            "logical_world_count": 3000,
            "active_canonical_classes": 10,
            "active_projected_classes": 10,
            "direct_median_ms": 4.0,
            "projected_median_ms": 6.0,
        },
        {
            "logical_world_count": 4000,
            "active_canonical_classes": 30,
            "active_projected_classes": 15,
            "direct_median_ms": 5.0,
            "projected_median_ms": 9.5,
        },
    ]
    baseline = _fit_absolute_baseline(rows)

    for row in rows:
        expected = row["projected_median_ms"] - row["direct_median_ms"]
        assert _baseline_delta(baseline, row) == pytest.approx(expected)


def test_saturation_shape_keeps_knee_feature_derived() -> None:
    pytest.importorskip("jax")
    from azelficoast.research.adaptive_execution_experiment import (
        CostModelShape,
        _direct_saturation_start_worlds,
    )

    rows = [
        {"logical_world_count": worlds}
        for worlds in (2048, 3072, 4096, 6144, 8192, 12288, 24576, 49152, 98304, 196608)
    ]
    shape = CostModelShape(
        direct_quadratic_term=False,
        projected_canonical_term=False,
        direct_saturation_term=True,
    )

    assert _direct_saturation_start_worlds(rows, shape) == 12288



def test_fixed_class_crossover_bracket_overrides_distorted_absolute_fit() -> None:
    profile = ExecutionCostProfile(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:test",
        direct_intercept_ms=1.0,
        direct_per_world_ms=0.001,
        direct_per_world_squared_ms=0.0,
        projected_intercept_ms=0.1,
        projected_per_canonical_class_ms=0.0,
        projected_per_execution_class_ms=0.0,
        uncertainty_guard_ms=0.1,
        calibrated_max_logical_world_count=10_000,
        calibrated_max_canonical_classes=100,
        calibrated_max_projected_classes=50,
        crossover_canonical_classes=10,
        crossover_projected_classes=5,
        crossover_direct_max_worlds=1000,
        crossover_projected_min_worlds=2000,
    )

    below_midpoint = choose_execution_path(
        profile,
        _features(1400, canonical=10, projected=5),
    )
    above_midpoint = choose_execution_path(
        profile,
        _features(1600, canonical=10, projected=5),
    )

    assert below_midpoint.predicted_projected_ms < below_midpoint.predicted_direct_ms
    assert below_midpoint.path is ExecutionPath.DIRECT
    assert above_midpoint.path is ExecutionPath.PROJECTED


def test_crossover_bracket_is_scoped_to_exact_class_shape() -> None:
    profile = ExecutionCostProfile(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:test",
        direct_intercept_ms=1.0,
        direct_per_world_ms=0.001,
        direct_per_world_squared_ms=0.0,
        projected_intercept_ms=0.1,
        projected_per_canonical_class_ms=0.0,
        projected_per_execution_class_ms=0.0,
        uncertainty_guard_ms=0.1,
        calibrated_max_logical_world_count=10_000,
        calibrated_max_canonical_classes=100,
        calibrated_max_projected_classes=50,
        crossover_canonical_classes=10,
        crossover_projected_classes=5,
        crossover_direct_max_worlds=1000,
        crossover_projected_min_worlds=2000,
    )

    different_shape = choose_execution_path(
        profile,
        _features(1400, canonical=11, projected=5),
    )

    assert different_shape.path is ExecutionPath.PROJECTED


def test_crossover_bracket_rejects_partial_or_reversed_configuration() -> None:
    kwargs = dict(
        backend="cpu",
        target_signature="sha256:target",
        effect_signature="sha256:test",
        direct_intercept_ms=0.0,
        direct_per_world_ms=0.001,
        direct_per_world_squared_ms=0.0,
        projected_intercept_ms=1.0,
        projected_per_canonical_class_ms=0.0,
        projected_per_execution_class_ms=0.0,
        uncertainty_guard_ms=0.0,
        calibrated_max_logical_world_count=10_000,
        calibrated_max_canonical_classes=100,
        calibrated_max_projected_classes=50,
    )
    with pytest.raises(ValueError, match="fully specified"):
        ExecutionCostProfile(
            **kwargs,
            crossover_canonical_classes=10,
        )
    with pytest.raises(ValueError, match="order direct below projected"):
        ExecutionCostProfile(
            **kwargs,
            crossover_canonical_classes=10,
            crossover_projected_classes=5,
            crossover_direct_max_worlds=2000,
            crossover_projected_min_worlds=1000,
        )


def test_fixed_class_crossover_bracket_requires_monotone_one_dimensional_labels() -> None:
    pytest.importorskip("jax")
    from azelficoast.research.adaptive_execution_experiment import _fixed_class_crossover_bracket

    monotone = [
        {
            "logical_world_count": 1000,
            "active_canonical_classes": 20,
            "active_projected_classes": 4,
            "oracle_path": "direct",
        },
        {
            "logical_world_count": 2000,
            "active_canonical_classes": 20,
            "active_projected_classes": 4,
            "oracle_path": "projected",
        },
    ]
    assert _fixed_class_crossover_bracket(monotone) == (20, 4, 1000, 2000)

    varying_classes = [
        {**monotone[0], "active_canonical_classes": 21},
        monotone[1],
    ]
    assert _fixed_class_crossover_bracket(varying_classes) == (0, 0, 0, 0)

    non_monotone = [
        monotone[0],
        monotone[1],
        {
            "logical_world_count": 3000,
            "active_canonical_classes": 20,
            "active_projected_classes": 4,
            "oracle_path": "direct",
        },
    ]
    assert _fixed_class_crossover_bracket(non_monotone) == (0, 0, 0, 0)
