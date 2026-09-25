"""Deterministic execution-path selection from a calibrated structural cost profile.

Direct and projected execution do different work. The model keeps those costs separate:
direct work scales with logical worlds, while projected work scales with active semantic
classes. Optional terms are admitted only by calibration/model selection, and profiles
refuse to extrapolate beyond their measured domain.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from pathlib import Path


class ExecutionPath(str, Enum):
    DIRECT = "direct"
    PROJECTED = "projected"


def current_jax_execution_target() -> tuple[str, str]:
    """Return the current JAX backend and a hardware-bound calibration identity."""

    import jax

    cpu_model = platform.processor() or "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.partition(":")[2].strip()
                break

    devices = jax.devices()
    backend = jax.default_backend()
    payload = {
        "jax_backend": backend,
        "device_kinds": sorted(str(device.device_kind) for device in devices),
        "device_count": len(devices),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpu_count": os.cpu_count(),
        "cpu_model": cpu_model,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return backend, "sha256:" + hashlib.sha256(encoded).hexdigest()


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
    direct_saturation_start_worlds: int = 0
    direct_saturation_per_world_ms: float = 0.0
    crossover_canonical_classes: int = 0
    crossover_projected_classes: int = 0
    crossover_direct_max_worlds: int = 0
    crossover_projected_min_worlds: int = 0

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
            self.direct_saturation_per_world_ms,
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
        if self.direct_saturation_start_worlds < 0:
            raise ValueError("direct saturation start must be non-negative")
        if (
            self.direct_saturation_per_world_ms > 0
            and self.direct_saturation_start_worlds <= 0
        ):
            raise ValueError(
                "positive direct saturation cost requires a positive saturation start"
            )
        crossover = (
            self.crossover_canonical_classes,
            self.crossover_projected_classes,
            self.crossover_direct_max_worlds,
            self.crossover_projected_min_worlds,
        )
        if any(value < 0 for value in crossover):
            raise ValueError("crossover bracket fields must be non-negative")
        active_crossover = any(value > 0 for value in crossover)
        if active_crossover and not all(value > 0 for value in crossover):
            raise ValueError("crossover bracket must be either fully specified or absent")
        if (
            active_crossover
            and self.crossover_direct_max_worlds
            >= self.crossover_projected_min_worlds
        ):
            raise ValueError("crossover bracket must order direct below projected")

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
        saturated_worlds = max(0, worlds - self.direct_saturation_start_worlds)
        return (
            self.direct_intercept_ms
            + self.direct_per_world_ms * worlds
            + self.direct_per_world_squared_ms * worlds * worlds
            + self.direct_saturation_per_world_ms * saturated_worlds
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

    bracket_applies = (
        profile.crossover_direct_max_worlds > 0
        and features.active_canonical_classes == profile.crossover_canonical_classes
        and features.active_projected_classes == profile.crossover_projected_classes
    )
    if bracket_applies:
        boundary = (
            profile.crossover_direct_max_worlds
            + profile.crossover_projected_min_worlds
        ) / 2
        path = (
            ExecutionPath.DIRECT
            if features.logical_world_count < boundary
            else ExecutionPath.PROJECTED
        )
    else:
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
