"""Generic exact execution for dependency-quotiented transition programs.

A transition program owns semantics and how canonical belief support maps into batch
execution arguments. This module owns only execution-path selection, weighted outcome
aggregation, mass preservation, and evidence binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np

from azelficoast.core.costing import (
    ExecutionCostProfile,
    ExecutionDecision,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
)

CERTIFICATE_SCHEMA = "azelficoast.core.transition-execution-certificate"
CERTIFICATE_SCHEMA_VERSION = 1


class TransitionProgramError(ValueError):
    """Raised when an exact transition program cannot execute faithfully."""


@dataclass(frozen=True)
class TransitionBatch:
    """One exact batch of transition evaluations plus represented belief mass."""

    arguments: tuple[np.ndarray, ...]
    weights: np.ndarray

    def __post_init__(self) -> None:
        if self.weights.ndim != 1:
            raise ValueError("transition batch weights must be one-dimensional")
        if self.weights.dtype.kind not in "iu":
            raise ValueError("transition batch weights must be integer")
        if np.any(self.weights <= 0):
            raise ValueError("transition batch weights must be positive")


class TransitionProgram(Protocol):
    """Runtime contract for one exact dependency-bound transition."""

    name: str
    effect_signature: str
    projection_binding_hash: str
    logical_world_count: int
    active_canonical_classes: int
    active_execution_classes: int
    certificate_schema: str
    certificate_schema_version: int
    claim: str
    non_claim: str

    def direct_batch(self) -> TransitionBatch:
        """Materialize one transition evaluation per represented logical world."""

    def projected_batch(self) -> TransitionBatch:
        """Materialize one evaluation per active exact execution class."""

    def execute_batch(self, *arguments: np.ndarray) -> np.ndarray:
        """Execute one exact batch and return one integer successor state per row."""


@dataclass(frozen=True)
class WeightedTransitionOutcomes:
    """Weighted canonical successor states produced by one exact execution path."""

    packed_states: np.ndarray
    weights: np.ndarray
    path: ExecutionPath
    certificate: Mapping[str, object]
    decision: ExecutionDecision | None = None

    def __post_init__(self) -> None:
        if self.packed_states.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("successor states and weights must be one-dimensional")
        if self.packed_states.dtype.kind not in "iu":
            raise ValueError("packed successor states must be integer")
        if self.weights.dtype.kind not in "iu":
            raise ValueError("successor weights must be integer")
        if len(self.packed_states) != len(self.weights):
            raise ValueError("successor states and weights must have equal length")
        if np.any(self.weights <= 0):
            raise ValueError("successor weights must be positive")

    @property
    def total_weight(self) -> int:
        return int(self.weights.sum(dtype=np.int64))

    def histogram(self) -> dict[int, int]:
        return {
            int(state): int(weight)
            for state, weight in zip(self.packed_states, self.weights, strict=True)
        }


def _aggregate(
    packed_states: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(packed_states, dtype=np.int64)
    multiplicity = np.asarray(weights, dtype=np.int64)
    if packed.ndim != 1 or multiplicity.ndim != 1:
        raise TransitionProgramError(
            "batch executor must return one-dimensional states and weights"
        )
    if packed.shape != multiplicity.shape:
        raise TransitionProgramError(
            "batch executor result does not match execution weights"
        )
    unique, inverse = np.unique(packed, return_inverse=True)
    aggregate = np.zeros(len(unique), dtype=np.int64)
    np.add.at(aggregate, inverse, multiplicity)
    return unique.astype(np.int64, copy=False), aggregate


def execute_transition_program(
    program: TransitionProgram,
    *,
    backend: str,
    target_signature: str,
    profile: ExecutionCostProfile | None = None,
    force_path: ExecutionPath | None = None,
) -> WeightedTransitionOutcomes:
    """Execute a complete transition directly or through its exact quotient."""

    if not program.effect_signature:
        raise TransitionProgramError("transition program has no effect signature")
    if not program.projection_binding_hash:
        raise TransitionProgramError("transition program has no projection binding")
    if program.logical_world_count <= 0:
        raise TransitionProgramError("transition program represents no logical worlds")
    if program.active_canonical_classes <= 0:
        raise TransitionProgramError("transition program has no active canonical classes")
    if not 0 < program.active_execution_classes <= program.active_canonical_classes:
        raise TransitionProgramError(
            "transition program execution-class count is inconsistent"
        )
    if not backend:
        raise TransitionProgramError("backend must be non-empty")
    if not target_signature:
        raise TransitionProgramError("target signature must be non-empty")

    features = ExecutionFeatures(
        backend=backend,
        target_signature=target_signature,
        effect_signature=program.effect_signature,
        logical_world_count=program.logical_world_count,
        active_canonical_classes=program.active_canonical_classes,
        active_projected_classes=program.active_execution_classes,
    )

    decision: ExecutionDecision | None
    if force_path is None:
        if profile is None:
            raise TransitionProgramError(
                "calibrated cost profile is required when execution path is not forced"
            )
        decision = choose_execution_path(profile, features)
        path = decision.path
    else:
        if profile is not None:
            profile.validate_features(features)
        decision = None
        path = force_path

    if path is ExecutionPath.DIRECT:
        batch = program.direct_batch()
        expected_evaluations = program.logical_world_count
    elif path is ExecutionPath.PROJECTED:
        batch = program.projected_batch()
        expected_evaluations = program.active_execution_classes
    else:
        raise TransitionProgramError(f"unsupported execution path: {path!r}")

    if len(batch.weights) != expected_evaluations:
        raise TransitionProgramError(
            "execution batch size does not match the selected transition path"
        )
    if int(batch.weights.sum(dtype=np.int64)) != program.logical_world_count:
        raise TransitionProgramError(
            "execution batch lost or duplicated logical-world mass"
        )

    raw = np.asarray(program.execute_batch(*batch.arguments), dtype=np.int64)
    states, weights = _aggregate(raw, batch.weights)

    certificate = {
        "schema": program.certificate_schema,
        "schema_version": program.certificate_schema_version,
        "effect_signature": program.effect_signature,
        "projection_name": program.name,
        "projection_binding_hash": program.projection_binding_hash,
        "logical_world_count": program.logical_world_count,
        "active_canonical_classes": program.active_canonical_classes,
        "active_execution_classes": program.active_execution_classes,
        "execution_path": path.value,
        "transition_evaluations": expected_evaluations,
        "logical_world_reduction": (
            program.logical_world_count - program.active_execution_classes
        ),
        "logical_reduction_fraction": (
            1.0
            - program.active_execution_classes / program.logical_world_count
        ),
        "canonical_class_reduction": (
            program.active_canonical_classes - program.active_execution_classes
        ),
        "claim": program.claim,
        "non_claim": program.non_claim,
    }

    outcomes = WeightedTransitionOutcomes(
        packed_states=states,
        weights=weights,
        path=path,
        certificate=certificate,
        decision=decision,
    )
    if outcomes.total_weight != program.logical_world_count:
        raise TransitionProgramError(
            "successor distribution lost or duplicated logical-world mass"
        )
    return outcomes
