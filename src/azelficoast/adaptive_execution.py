"""Deterministic execution-path selection from a calibrated cost profile.

The dispatcher contains no population-size threshold. A backend/effect-specific calibration
profile predicts direct and projected execution cost from the current belief geometry, and the
lower predicted cost wins. Calibration is separate from selection so production execution stays
deterministic and benchmark machinery stays outside the hot path.
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
    direct_intercept_ms: float
    direct_per_world_ms: float
    projected_intercept_ms: float
    projected_per_canonical_class_ms: float
    projected_per_execution_class_ms: float

    def __post_init__(self) -> None:
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if not self.effect_signature:
            raise ValueError("effect_signature must be non-empty")
        coefficients = (
            self.direct_intercept_ms,
            self.direct_per_world_ms,
            self.projected_intercept_ms,
            self.projected_per_canonical_class_ms,
            self.projected_per_execution_class_ms,
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

    def estimate_direct_ms(self, features: ExecutionFeatures) -> float:
        self.validate_features(features)
        return (
            self.direct_intercept_ms
            + self.direct_per_world_ms * features.logical_world_count
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


@dataclass(frozen=True)
class ExecutionDecision:
    path: ExecutionPath
    predicted_direct_ms: float
    predicted_projected_ms: float


def choose_execution_path(
    profile: ExecutionCostProfile,
    features: ExecutionFeatures,
) -> ExecutionDecision:
    direct = profile.estimate_direct_ms(features)
    projected = profile.estimate_projected_ms(features)
    # Exact ties intentionally choose the simpler direct path.
    path = ExecutionPath.PROJECTED if projected < direct else ExecutionPath.DIRECT
    return ExecutionDecision(
        path=path,
        predicted_direct_ms=direct,
        predicted_projected_ms=projected,
    )
