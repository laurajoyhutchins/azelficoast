"""Pre-execution quotienting for the bounded compiled two-attack turn.

This module promotes the class-native projection from benchmark plumbing into a reusable
execution surface. The dependency-bound projection is computed before transition
execution. Projected execution runs one representative per active execution class and
carries class multiplicity forward as successor-state weight.

The direct path remains available to the calibrated dispatcher and as an exact oracle.
Both paths return the same weighted successor-state representation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

from azelficoast.adaptive_execution import (
    ExecutionCostProfile,
    ExecutionDecision,
    ExecutionFeatures,
    ExecutionPath,
    choose_execution_path,
    current_jax_execution_target,
)
from azelficoast.gen9_two_attack_turn import (
    TwoAttackTurnContext,
    compile_two_attack_turn_context,
    two_attack_turn_dependency_signature,
)
from azelficoast.two_attack_turn_belief import (
    TwoAttackTurnBelief,
    TwoAttackTurnProjection,
    project_two_attack_turn_belief,
)

BatchExecutor = Callable[
    [
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    np.ndarray,
]

CERTIFICATE_SCHEMA = "azelficoast.preexecution-quotient-certificate"
CERTIFICATE_SCHEMA_VERSION = 1


class TwoAttackTurnExecutionError(ValueError):
    """Raised when an execution quotient is unbound or internally inconsistent."""


@dataclass(frozen=True)
class WeightedTurnOutcomes:
    """Weighted packed successor states produced by one exact execution path."""

    packed_states: np.ndarray
    weights: np.ndarray
    path: ExecutionPath
    certificate: Mapping[str, object]
    decision: ExecutionDecision | None = None

    def __post_init__(self) -> None:
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
            for state, weight in zip(
                self.packed_states,
                self.weights,
                strict=True,
            )
        }


def _projection_binding_hash(
    projection: TwoAttackTurnProjection,
    *,
    active_class_ids: np.ndarray,
) -> str:
    payload = {
        "effect_signature": projection.effect_signature,
        "active_class_ids": [int(value) for value in active_class_ids],
        "representative_indices": [
            int(projection.representative_indices[int(class_id)])
            for class_id in active_class_ids
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _compiled_contexts(
    contexts: Sequence[TwoAttackTurnContext],
) -> np.ndarray:
    if not contexts:
        raise TwoAttackTurnExecutionError("at least one turn context is required")
    return np.asarray(
        [compile_two_attack_turn_context(context) for context in contexts],
        dtype=np.int32,
    )


def _direct_inputs(
    belief: TwoAttackTurnBelief,
    contexts: Sequence[TwoAttackTurnContext],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    support = belief.support
    rows = _compiled_contexts(contexts)
    if np.any(support.context_index >= len(rows)):
        raise TwoAttackTurnExecutionError(
            "belief support references an unavailable turn context"
        )

    context_index = np.repeat(support.context_index, belief.weights)
    logical_world_count = belief.logical_world_count
    if logical_world_count <= 0:
        raise TwoAttackTurnExecutionError("belief contains no logical worlds")

    return (
        rows[context_index],
        np.repeat(support.order_tie_roll, belief.weights).astype(
            np.int32, copy=False
        ),
        np.zeros(logical_world_count, dtype=np.int32),
        np.repeat(support.p1_damage_roll, belief.weights).astype(
            np.int32, copy=False
        ),
        np.repeat(support.p1_secondary_roll, belief.weights).astype(
            np.int32, copy=False
        ),
        np.zeros(logical_world_count, dtype=np.int32),
        np.repeat(support.p2_damage_roll, belief.weights).astype(
            np.int32, copy=False
        ),
        np.ones(logical_world_count, dtype=np.int64),
    )


def _projected_inputs(
    belief: TwoAttackTurnBelief,
    projection: TwoAttackTurnProjection,
    contexts: Sequence[TwoAttackTurnContext],
) -> tuple[
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    np.ndarray,
]:
    support = belief.support
    rows = _compiled_contexts(contexts)
    if len(projection.class_ids) != support.class_count:
        raise TwoAttackTurnExecutionError(
            "projection does not cover canonical turn support"
        )
    if np.any(support.context_index >= len(rows)):
        raise TwoAttackTurnExecutionError(
            "belief support references an unavailable turn context"
        )

    projected = project_two_attack_turn_belief(belief, projection)
    active_class_ids = np.flatnonzero(projected.weights > 0).astype(
        np.int32, copy=False
    )
    if not len(active_class_ids):
        raise TwoAttackTurnExecutionError("belief contains no active execution class")

    representatives = projection.representative_indices[active_class_ids]
    count = len(representatives)
    inputs = (
        rows[support.context_index[representatives]],
        support.order_tie_roll[representatives].astype(np.int32, copy=False),
        np.zeros(count, dtype=np.int32),
        support.p1_damage_roll[representatives].astype(np.int32, copy=False),
        support.p1_secondary_roll[representatives].astype(np.int32, copy=False),
        np.zeros(count, dtype=np.int32),
        support.p2_damage_roll[representatives].astype(np.int32, copy=False),
        projected.weights[active_class_ids].astype(np.int64, copy=False),
    )
    return inputs, active_class_ids


def _aggregate(
    packed_states: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(packed_states, dtype=np.int64)
    multiplicity = np.asarray(weights, dtype=np.int64)
    if packed.ndim != 1 or multiplicity.ndim != 1:
        raise TwoAttackTurnExecutionError("batch executor must return one-dimensional states")
    if packed.shape != multiplicity.shape:
        raise TwoAttackTurnExecutionError(
            "batch executor result does not match execution weights"
        )
    unique, inverse = np.unique(packed, return_inverse=True)
    aggregate = np.zeros(len(unique), dtype=np.int64)
    np.add.at(aggregate, inverse, multiplicity)
    return unique.astype(np.int64, copy=False), aggregate


def execute_two_attack_turn_belief(
    belief: TwoAttackTurnBelief,
    projection: TwoAttackTurnProjection,
    contexts: Sequence[TwoAttackTurnContext],
    *,
    batch_executor: BatchExecutor,
    backend: str,
    target_signature: str,
    profile: ExecutionCostProfile | None = None,
    force_path: ExecutionPath | None = None,
) -> WeightedTurnOutcomes:
    """Execute a belief directly or through its dependency-bound exact quotient.

    If force_path is omitted, a calibrated profile is required and owns the path
    decision. An incomplete or unsigned projection is always rejected, even when
    direct execution would happen to be selected, because the execution features and
    certificate must remain bound to the exact effect contract.
    """

    effect_signature = two_attack_turn_dependency_signature()
    if projection.effect_signature != effect_signature:
        raise TwoAttackTurnExecutionError(
            "projection is not bound to the two-attack-turn dependency signature"
        )
    if not backend:
        raise TwoAttackTurnExecutionError("backend must be non-empty")
    if not target_signature:
        raise TwoAttackTurnExecutionError("target signature must be non-empty")

    projected = project_two_attack_turn_belief(belief, projection)
    active_class_ids = np.flatnonzero(projected.weights > 0).astype(
        np.int32, copy=False
    )
    features = ExecutionFeatures(
        backend=backend,
        target_signature=target_signature,
        effect_signature=effect_signature,
        logical_world_count=belief.logical_world_count,
        active_canonical_classes=belief.active_canonical_classes,
        active_projected_classes=projected.active_classes,
    )

    decision: ExecutionDecision | None
    if force_path is None:
        if profile is None:
            raise TwoAttackTurnExecutionError(
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
        *inputs, execution_weights = _direct_inputs(belief, contexts)
    elif path is ExecutionPath.PROJECTED:
        projected_inputs, active_class_ids = _projected_inputs(
            belief,
            projection,
            contexts,
        )
        *inputs, execution_weights = projected_inputs
    else:
        raise TwoAttackTurnExecutionError(f"unsupported execution path: {path!r}")

    raw = np.asarray(batch_executor(*inputs), dtype=np.int64)
    states, weights = _aggregate(raw, execution_weights)

    representative_count = projected.active_classes
    transition_evaluations = (
        belief.logical_world_count
        if path is ExecutionPath.DIRECT
        else representative_count
    )
    certificate = {
        "schema": CERTIFICATE_SCHEMA,
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "effect_signature": effect_signature,
        "projection_name": projection.name,
        "projection_binding_hash": _projection_binding_hash(
            projection,
            active_class_ids=active_class_ids,
        ),
        "logical_world_count": belief.logical_world_count,
        "active_canonical_classes": belief.active_canonical_classes,
        "active_execution_classes": representative_count,
        "execution_path": path.value,
        "transition_evaluations": transition_evaluations,
        "logical_world_reduction": belief.logical_world_count - representative_count,
        "logical_reduction_fraction": (
            1.0 - representative_count / belief.logical_world_count
        ),
        "canonical_class_reduction": (
            belief.active_canonical_classes - representative_count
        ),
        "claim": (
            "Projected execution evaluates one representative for each active class "
            "of the dependency-bound two-attack-turn projection and carries exact "
            "class multiplicity into the successor distribution."
        ),
        "non_claim": (
            "The certificate applies only to the bounded two-attack-turn effect "
            "identified by effect_signature."
        ),
    }

    outcomes = WeightedTurnOutcomes(
        packed_states=states,
        weights=weights,
        path=path,
        certificate=certificate,
        decision=decision,
    )
    if outcomes.total_weight != belief.logical_world_count:
        raise TwoAttackTurnExecutionError(
            "successor distribution lost or duplicated logical-world mass"
        )
    return outcomes



def execute_two_attack_turn_jax(
    belief: TwoAttackTurnBelief,
    projection: TwoAttackTurnProjection,
    contexts: Sequence[TwoAttackTurnContext],
    *,
    profile: ExecutionCostProfile | None = None,
    force_path: ExecutionPath | None = None,
) -> WeightedTurnOutcomes:
    """Execute through the compiled JAX lowering on the current calibrated target."""

    import jax

    from azelficoast.jax_gen9_two_attack_turn import two_attack_turn_batch

    backend, target_signature = current_jax_execution_target()

    def batch_executor(
        params,
        order,
        p1_accuracy,
        p1_damage,
        p1_secondary,
        p2_accuracy,
        p2_damage,
    ):
        value = two_attack_turn_batch(
            jax.device_put(params),
            jax.device_put(order),
            jax.device_put(p1_accuracy),
            jax.device_put(p1_damage),
            jax.device_put(p1_secondary),
            jax.device_put(p2_accuracy),
            jax.device_put(p2_damage),
        )
        value.block_until_ready()
        return np.asarray(value, dtype=np.int64)

    return execute_two_attack_turn_belief(
        belief,
        projection,
        contexts,
        batch_executor=batch_executor,
        backend=backend,
        target_signature=target_signature,
        profile=profile,
        force_path=force_path,
    )
