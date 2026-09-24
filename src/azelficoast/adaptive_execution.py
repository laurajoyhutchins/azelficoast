"""Deterministic execution-path selection from a calibrated structural cost profile.

Direct and projected execution do different work. The model keeps those costs separate:
direct work scales with logical worlds, while projected work scales with active semantic
classes. Optional terms are admitted only by calibration/model selection, and profiles
refuse to extrapolate beyond their measured domain.
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
    target_signature: str
    effect_signature: str
    logical_world_count: int
    active_canonical_classes: int
    active_projected_classes: int

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.target_signature:
            raise ValueError("target_signature must be non-empty")
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
    target_signature: str
    effect_signature: str
    direct_intercept_ms: float
    direct_per_world_ms: float
    direct_per_world_squared_ms: float
    projected_intercept_ms: float
    projected_per_canonical_class_ms: float
    projected_per_execution_class_ms: float
    uncertainty_guard_ms: float
    calibrated_max_logical_world_count: int
    calibrated_max_canonical_classes: int
    calibrated_max_projected_classes: int

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.target_signature:
            raise ValueError("target_signature must be non-empty")
        if not self.effect_signature:
            raise ValueError("effect_signature must be non-empty")
        coefficients = (
            self.direct_intercept_ms,
            self.direct_per_world_ms,
            self.direct_per_world_squared_ms,
            self.projected_intercept_ms,
            self.projected_per_canonical_class_ms,
            self.projected_per_execution_class_ms,
            self.uncertainty_guard_ms,
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
        if features.target_signature != self.target_signature:
            raise ValueError("cost profile target signature does not match execution features")
        if features.effect_signature != self.effect_signature:
            raise ValueError("cost profile effect signature does not match execution features")
        if features.logical_world_count > self.calibrated_max_logical_world_count:
            raise ValueError("logical world count is outside the calibrated cost-model domain")
        if features.active_canonical_classes > self.calibrated_max_canonical_classes:
            raise ValueError("canonical class count is outside the calibrated cost-model domain")
        if features.active_projected_classes > self.calibrated_max_projected_classes:
            raise ValueError("projected class count is outside the calibrated cost-model domain")

    def estimate_direct_ms(self, features: ExecutionFeatures) -> float:
        self.validate_features(features)
        worlds = features.logical_world_count
        return (
            self.direct_intercept_ms
            + self.direct_per_world_ms * worlds
            + self.direct_per_world_squared_ms * worlds * worlds
        )

    def estimate_projected_ms(self, features: ExecutionFeatures) -> float:
        self.validate_features(features)
        return (
            self.projected_intercept_ms
            + self.projected_per_canonical_class_ms
            * features.active_canonical_classes
            + self.projected_per_execution_class_ms
            * features.active_projected_classes
        )

    def estimate_projected_minus_direct_ms(
        self,
        features: ExecutionFeatures,
    ) -> float:
        return self.estimate_projected_ms(features) - self.estimate_direct_ms(features)


@dataclass(frozen=True)
class ExecutionDecision:
    path: ExecutionPath
    predicted_direct_ms: float
    predicted_projected_ms: float
    predicted_projected_minus_direct_ms: float
    uncertainty_guard_ms: float
    within_uncertainty_guard: bool


def choose_execution_path(
    profile: ExecutionCostProfile,
    features: ExecutionFeatures,
) -> ExecutionDecision:
    direct = profile.estimate_direct_ms(features)
    projected = profile.estimate_projected_ms(features)
    delta = projected - direct
    # Both paths are semantically exact. Uncertainty is diagnostic rather than a
    # reason to choose a path predicted to be slower. Exact predicted ties use direct.
    path = ExecutionPath.PROJECTED if delta < 0 else ExecutionPath.DIRECT
    return ExecutionDecision(
        path=path,
        predicted_direct_ms=direct,
        predicted_projected_ms=projected,
        predicted_projected_minus_direct_ms=delta,
        uncertainty_guard_ms=profile.uncertainty_guard_ms,
        within_uncertainty_guard=abs(delta) <= profile.uncertainty_guard_ms,
    )
