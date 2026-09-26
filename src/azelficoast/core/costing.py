"""Calibrated structural costing for exact finite partial-information execution.

The cost model is domain-neutral: direct work scales with logical support while
projected work scales with active semantic classes. Profiles are identity-bound and
refuse to extrapolate outside their measured calibration domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Sequence


class ExecutionPath(str, Enum):
    DIRECT = "direct"
    PROJECTED = "projected"


@dataclass(frozen=True)
class LocalityEvidence:
    """Observed reuse evidence with a deterministic smoothed hit estimate."""

    hits: int
    misses: int
    prior_hits: float = 1.0
    prior_misses: float = 1.0

    def __post_init__(self) -> None:
        if self.hits < 0 or self.misses < 0:
            raise ValueError("locality counts must be non-negative")
        if (
            not isfinite(self.prior_hits)
            or not isfinite(self.prior_misses)
            or self.prior_hits <= 0
            or self.prior_misses <= 0
        ):
            raise ValueError("locality priors must be finite and positive")

    @property
    def attempts(self) -> int:
        return self.hits + self.misses

    @property
    def hit_probability(self) -> float:
        return (self.hits + self.prior_hits) / (
            self.attempts + self.prior_hits + self.prior_misses
        )


@dataclass(frozen=True)
class CacheTierCost:
    """One optional physical reuse tier in a fallback execution route."""

    name: str
    locality: LocalityEvidence
    hit_cost_ms: float
    miss_cost_ms: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("cache tier name must be non-empty")
        for value in (self.hit_cost_ms, self.miss_cost_ms):
            if not isfinite(value) or value < 0:
                raise ValueError("cache tier costs must be finite and non-negative")


@dataclass(frozen=True)
class CacheRouteEstimate:
    """Expected physical cost for an ordered cache route and terminal fallback."""

    tier_names: tuple[str, ...]
    expected_ms: float
    fallback_ms: float
    remaining_miss_probability: float
    hit_probabilities: tuple[float, ...]


def estimate_cache_route(
    tiers: Sequence[CacheTierCost],
    *,
    fallback_ms: float,
) -> CacheRouteEstimate:
    """Estimate an ordered cache route from measured locality and latency.

    Each tier is attempted only after every prior tier misses. A hit terminates the route.
    The terminal fallback is exact work such as a fresh simulator execution or an already
    verified compiled kernel.
    """

    if not isfinite(fallback_ms) or fallback_ms < 0:
        raise ValueError("fallback cost must be finite and non-negative")
    if len({tier.name for tier in tiers}) != len(tiers):
        raise ValueError("cache tier names must be unique")

    expected = 0.0
    remaining = 1.0
    probabilities: list[float] = []
    for tier in tiers:
        probability = tier.locality.hit_probability
        probabilities.append(probability)
        expected += remaining * (
            probability * tier.hit_cost_ms
            + (1.0 - probability) * tier.miss_cost_ms
        )
        remaining *= 1.0 - probability

    expected += remaining * fallback_ms
    return CacheRouteEstimate(
        tier_names=tuple(tier.name for tier in tiers),
        expected_ms=expected,
        fallback_ms=fallback_ms,
        remaining_miss_probability=remaining,
        hit_probabilities=tuple(probabilities),
    )


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
