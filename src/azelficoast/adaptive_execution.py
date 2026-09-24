"""Deterministic execution-path selection from a calibrated relative cost profile.

Dispatch needs the sign of projected_cost - direct_cost, not two independently fitted absolute
latency curves. The profile models that paired difference directly with monotone work terms.
Calibration uncertainty is reported separately from the path choice because both execution paths
are semantically exact; uncertainty about performance is not a reason to override the predicted
faster path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


class ExecutionPath(str, Enum):
    DIRECT = "direct"
    PROJECTED = "projected"


@dataclass(frozen=True)
class ExecutionFeatures:
    backend: str
    effect_signature: str
    logical_world_count: int
    active_canonical_classes: int
    active_projected_classes: int

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.effect_signature:
            raise ValueError("effect_signature must be non-empty")
        if self.logical_world_count <= 0:
            raise ValueError("logical_world_count must be positive")
        if self.active_canonical_classes <= 0:
            raise ValueError("active_canonical_classes must be positive")
        if self.active_projected_classes <= 0:
            raise ValueError("active_projected_classes must be positive")
        if self.active_projected_classes > self.active_canonical_classes:
            raise ValueError(
                "active_projected_classes cannot exceed active_canonical_classes"
            )


@dataclass(frozen=True)
class ExecutionCostProfile:
    backend: str
    effect_signature: str
    projected_fixed_overhead_ms: float
    direct_per_world_ms: float
    direct_per_world_squared_ms: float
    projected_per_canonical_class_ms: float
    projected_per_execution_class_ms: float
    uncertainty_fixed_ms: float
    uncertainty_per_world_ms: float
    calibrated_max_logical_world_count: int
    calibrated_max_canonical_classes: int
    calibrated_max_projected_classes: int

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.effect_signature:
            raise ValueError("effect_signature must be non-empty")
        coefficients = (
            self.projected_fixed_overhead_ms,
            self.direct_per_world_ms,
            self.direct_per_world_squared_ms,
            self.projected_per_canonical_class_ms,
            self.projected_per_execution_class_ms,
            self.uncertainty_fixed_ms,
            self.uncertainty_per_world_ms,
        )
        if any(not isfinite(value) or value < 0 for value in coefficients):
            raise ValueError("cost-profile coefficients must be finite and non-negative")
        bounds = (
            self.calibrated_max_logical_world_count,
            self.calibrated_max_canonical_classes,
            self.calibrated_max_projected_classes,
        )
        if any(value <= 0 for value in bounds):
            raise ValueError("calibration-domain bounds must be positive")

    def validate_features(self, features: ExecutionFeatures) -> None:
        if features.backend != self.backend:
            raise ValueError(
                f"cost profile backend {self.backend!r} does not match "
                f"{features.backend!r}"
            )
        if features.effect_signature != self.effect_signature:
            raise ValueError("cost profile effect signature does not match execution features")
        if features.logical_world_count > self.calibrated_max_logical_world_count:
            raise ValueError("logical world count is outside the calibrated cost-model domain")
        if features.active_canonical_classes > self.calibrated_max_canonical_classes:
            raise ValueError("canonical class count is outside the calibrated cost-model domain")
        if features.active_projected_classes > self.calibrated_max_projected_classes:
            raise ValueError("projected class count is outside the calibrated cost-model domain")

    def estimate_projected_minus_direct_ms(
        self,
        features: ExecutionFeatures,
    ) -> float:
        """Predict projected latency minus direct latency.

        The quadratic term is a bounded empirical saturation term. Its coefficient is
        non-negative, so increasing logical multiplicity still can only favor projection.
        Profiles refuse extrapolation beyond their calibrated domain.
        """
        self.validate_features(features)
        worlds = features.logical_world_count
        return (
            self.projected_fixed_overhead_ms
            - self.direct_per_world_ms * worlds
            - self.direct_per_world_squared_ms * worlds * worlds
            + self.projected_per_canonical_class_ms
            * features.active_canonical_classes
            + self.projected_per_execution_class_ms
            * features.active_projected_classes
        )

    def uncertainty_guard_ms(self, features: ExecutionFeatures) -> float:
        self.validate_features(features)
        return (
            self.uncertainty_fixed_ms
            + self.uncertainty_per_world_ms * features.logical_world_count
        )


@dataclass(frozen=True)
class ExecutionDecision:
    path: ExecutionPath
    predicted_projected_minus_direct_ms: float
    uncertainty_guard_ms: float
    within_uncertainty_guard: bool


def choose_execution_path(
    profile: ExecutionCostProfile,
    features: ExecutionFeatures,
) -> ExecutionDecision:
    delta = profile.estimate_projected_minus_direct_ms(features)
    guard = profile.uncertainty_guard_ms(features)
    # Both paths are semantically exact. Uncertainty is diagnostic, not a reason to
    # choose a path predicted to be slower. Exact predicted ties prefer direct.
    path = ExecutionPath.PROJECTED if delta < 0 else ExecutionPath.DIRECT
    return ExecutionDecision(
        path=path,
        predicted_projected_minus_direct_ms=delta,
        uncertainty_guard_ms=guard,
        within_uncertainty_guard=abs(delta) <= guard,
    )
