"""Pre-execution quotienting for the bounded compiled two-attack turn.

The two-attack turn is now an adapter to the generic transition-program runtime. Its
mechanics-specific job is to prepare direct and dependency-projected batches; generic
execution owns path selection, mass preservation, aggregation, and certificates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from azelficoast.core.costing import ExecutionCostProfile, ExecutionPath
from azelficoast.core.projection import ActiveProjection, compact_active_projection
from azelficoast.research.adaptive_execution import current_jax_execution_target
from azelficoast.research.mechanics.gen9_two_attack_turn import (
    TwoAttackTurnContext,
    compile_two_attack_turn_context,
    two_attack_turn_dependency_signature,
)
from azelficoast.research.mechanics.transition_program import (
    TransitionBatch,
    TransitionProgramError,
    WeightedTransitionOutcomes,
    execute_transition_program,
)
from azelficoast.research.mechanics.two_attack_turn_belief import (
    TwoAttackTurnBelief,
    TwoAttackTurnProjection,
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

TwoAttackTurnExecutionError = TransitionProgramError
WeightedTurnOutcomes = WeightedTransitionOutcomes


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


def _active_projection(
    belief: TwoAttackTurnBelief,
    projection: TwoAttackTurnProjection,
) -> ActiveProjection:
    return compact_active_projection(
        belief.weights,
        projection.class_ids,
        projection.representative_indices,
        support_class_count=belief.support.class_count,
        mismatch_message="projection does not match two-attack support",
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
    if np.any(support.context_index >= len(rows)):
        raise TwoAttackTurnExecutionError(
            "belief support references an unavailable turn context"
        )

    active = _active_projection(belief, projection)
    if not active.active_class_count:
        raise TwoAttackTurnExecutionError("belief contains no active execution class")

    representatives = active.representative_indices
    count = len(representatives)
    inputs = (
        rows[support.context_index[representatives]],
        support.order_tie_roll[representatives].astype(np.int32, copy=False),
        np.zeros(count, dtype=np.int32),
        support.p1_damage_roll[representatives].astype(np.int32, copy=False),
        support.p1_secondary_roll[representatives].astype(np.int32, copy=False),
        np.zeros(count, dtype=np.int32),
        support.p2_damage_roll[representatives].astype(np.int32, copy=False),
        active.weights.astype(np.int64, copy=False),
    )
    return inputs, active.global_class_ids


@dataclass(frozen=True)
class _TwoAttackTurnProgram:
    belief: TwoAttackTurnBelief
    projection: TwoAttackTurnProjection
    contexts: Sequence[TwoAttackTurnContext]
    batch_executor: BatchExecutor

    certificate_schema: str = CERTIFICATE_SCHEMA
    certificate_schema_version: int = CERTIFICATE_SCHEMA_VERSION
    claim: str = (
        "Projected execution evaluates one representative for each active class "
        "of the dependency-bound two-attack-turn projection and carries exact "
        "class multiplicity into the successor distribution."
    )
    non_claim: str = (
        "The certificate applies only to the bounded two-attack-turn effect "
        "identified by effect_signature."
    )

    @property
    def name(self) -> str:
        return self.projection.name

    @property
    def effect_signature(self) -> str:
        return two_attack_turn_dependency_signature()

    @property
    def logical_world_count(self) -> int:
        return self.belief.logical_world_count

    @property
    def active_canonical_classes(self) -> int:
        return self.belief.active_canonical_classes

    @property
    def _active_projection(self) -> ActiveProjection:
        return _active_projection(self.belief, self.projection)

    @property
    def _active_class_ids(self) -> np.ndarray:
        return self._active_projection.global_class_ids

    @property
    def active_execution_classes(self) -> int:
        return self._active_projection.active_class_count

    @property
    def projection_binding_hash(self) -> str:
        return _projection_binding_hash(
            self.projection,
            active_class_ids=self._active_class_ids,
        )

    def direct_batch(self) -> TransitionBatch:
        *inputs, weights = _direct_inputs(self.belief, self.contexts)
        return TransitionBatch(arguments=tuple(inputs), weights=weights)

    def projected_batch(self) -> TransitionBatch:
        inputs, _ = _projected_inputs(self.belief, self.projection, self.contexts)
        *arguments, weights = inputs
        return TransitionBatch(arguments=tuple(arguments), weights=weights)

    def execute_batch(self, *arguments: np.ndarray) -> np.ndarray:
        return self.batch_executor(*arguments)


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
    """Execute the bounded turn through the generic exact transition runtime."""

    effect_signature = two_attack_turn_dependency_signature()
    if projection.effect_signature != effect_signature:
        raise TwoAttackTurnExecutionError(
            "projection is not bound to the two-attack-turn dependency signature"
        )

    program = _TwoAttackTurnProgram(
        belief=belief,
        projection=projection,
        contexts=contexts,
        batch_executor=batch_executor,
    )
    return execute_transition_program(
        program,
        backend=backend,
        target_signature=target_signature,
        profile=profile,
        force_path=force_path,
    )


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

    from azelficoast.research.mechanics.jax_gen9_two_attack_turn import two_attack_turn_batch

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
