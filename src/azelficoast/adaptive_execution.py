"""Deterministic execution-path selection from a calibrated relative cost profile.

Dispatch needs the sign of projected_cost - direct_cost, not two independently fitted absolute
latency curves. The profile therefore models that paired difference directly with monotone,
non-negative work coefficients and a calibration-derived uncertainty guard.
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
    projected_per_canonical_class_ms: float
    projected_per_execution_class_ms: float
    decision_guard_ms: float

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.effect_signature:
            raise ValueError("effect_signature must be non-empty")
        coefficients = (
            self.projected_fixed_overhead_ms,
            self.direct_per_world_ms,
            self.projected_per_canonical_class_ms,
            self.projected_per_execution_class_ms,
            self.decision_guard_ms,
        )
        if any(not isfinite(value) or value < 0 for value in coefficients):
            raise ValueError("cost-profile coefficients must be finite and non-negative")

    def validate_features(self, features: ExecutionFeatures) -> None:
        if features.backend != self.backend:
            raise ValueError(
                f"cost profile backend {self.backend!r} does not match "
                f"{features.backend!r}"
            )
        if features.effect_signature != self.effect_signature:
            raise ValueError("cost profile effect signature does not match execution features")

    def estimate_projected_minus_direct_ms(
        self,
        features: ExecutionFeatures,
    ) -> float:
        """Predict projected latency minus direct latency.

        Increasing logical multiplicity can only make direct execution less attractive.
        Increasing canonical or execution classes can only make projection less attractive.
        """
        self.validate_features(features)
        return (
            self.projected_fixed_overhead_ms
            - self.direct_per_world_ms * features.logical_world_count
            + self.projected_per_canonical_class_ms
            * features.active_canonical_classes
            + self.projected_per_execution_class_ms
            * features.active_projected_classes
        )


@dataclass(frozen=True)
class ExecutionDecision:
    path: ExecutionPath
    predicted_projected_minus_direct_ms: float
    decision_guard_ms: float
    within_uncertainty_guard: bool


def choose_execution_path(
    profile: ExecutionCostProfile,
    features: ExecutionFeatures,
) -> ExecutionDecision:
    delta = profile.estimate_projected_minus_direct_ms(features)
    # Projection must clear the calibration uncertainty guard. Near the crossover,
    # prefer the simpler direct path rather than chase benchmark noise.
    path = (
        ExecutionPath.PROJECTED
        if delta < -profile.decision_guard_ms
        else ExecutionPath.DIRECT
    )
    return ExecutionDecision(
        path=path,
        predicted_projected_minus_direct_ms=delta,
        decision_guard_ms=profile.decision_guard_ms,
        within_uncertainty_guard=abs(delta) <= profile.decision_guard_ms,
    )
