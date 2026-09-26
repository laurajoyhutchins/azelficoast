"""Pokémon-specific fusion of compiled search topology and packed belief evaluation.

This path keeps semantic authority in the existing transition-program validator and
Showdown-bound posterior pack. It removes per-leaf hidden-world object construction from
numerical search:

    verified program -> compiled topology
    joint posterior -> one packed hidden-world tensor
    topology + prior -> JAX [leaf, world] conditional weights
    successor public states -> fixed public feature matrix
    shared packed worlds + leaf weights -> one JAX evaluator dispatch
    leaf values + leaf masses -> JAX root reduction

The module is research-only until equivalence and performance evidence justify routing
live exact search through it.
"""

from __future__ import annotations

from typing import Any, Mapping

from azelficoast.belief.evaluator import BeliefEvaluatorError, hashed_features
from azelficoast.belief.packed_evaluator import (
    PACKED_VALUE_LOWER_BOUND,
    PACKED_VALUE_UPPER_BOUND,
    PackedBeliefEvaluatorSpec,
    predict_packed_shared_world_values,
)
from azelficoast.belief.showdown_packing import (
    ShowdownPackingError,
    ShowdownVocabulary,
    pack_joint_posterior,
)
from azelficoast.core.evaluation import choose_bounded_action
from azelficoast.core.compiled_search import (
    COMPILED_SEARCH_SCHEMA,
    COMPILED_SEARCH_SCHEMA_VERSION,
    CompiledSearchError,
    compile_search_topology,
    reduce_compiled_root_values,
    transport_posterior_mass,
)
from azelficoast.research.contracts import PublicSuccessorState, ResearchContractError

PACKED_COMPILED_SEARCH_SCHEMA = "azelficoast.packed-compiled-partial-information-search"
PACKED_COMPILED_SEARCH_SCHEMA_VERSION = 2
PACKED_BOUNDED_DECISION_SCHEMA = "azelficoast.packed-bounded-best-action"
PACKED_BOUNDED_DECISION_SCHEMA_VERSION = 1
PACKED_COMPILED_EXECUTION_STAGES = (
    "compile_search_topology",
    "pack_joint_posterior",
    "transport_posterior_mass",
    "predict_packed_shared_world_values",
    "reduce_compiled_root_values",
)


def search_packed_compiled_transition_program(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    vocabulary: ShowdownVocabulary,
    evaluator_spec: PackedBeliefEvaluatorSpec,
    evaluator_params: Mapping[str, Any],
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
) -> dict[str, Any]:
    """Evaluate one exact root frontier without materializing per-leaf world objects."""

    if evaluator_spec.vocabulary_sha256 != vocabulary.vocabulary_sha256:
        raise BeliefEvaluatorError(
            "packed evaluator spec and Showdown vocabulary identities differ"
        )

    try:
        topology = compile_search_topology(
            program_set=program_set,
            posterior=posterior,
            method=method,
            expected_program_schema=expected_program_schema,
            expected_program_schema_version=expected_program_schema_version,
        )
        packed = pack_joint_posterior(posterior, vocabulary)
        transported = transport_posterior_mass(topology, posterior)
    except (CompiledSearchError, ShowdownPackingError) as error:
        raise BeliefEvaluatorError(str(error)) from error

    if packed.world_ids != topology.world_ids:
        raise BeliefEvaluatorError(
            "packed posterior ordering differs from compiled search support"
        )
    if packed.vocabulary_sha256 != evaluator_spec.vocabulary_sha256:
        raise BeliefEvaluatorError("packed posterior vocabulary identity drifted")

    public_features: list[tuple[float, ...]] = []
    try:
        for successor in topology.leaf_public_states:
            public = PublicSuccessorState.from_record(successor)
            public_features.append(
                hashed_features(
                    public.public_state.to_record(),
                    width=evaluator_spec.public_width,
                )
            )
    except ResearchContractError as error:
        raise BeliefEvaluatorError(str(error)) from error

    leaf_values = predict_packed_shared_world_values(
        evaluator_params,
        packed,
        public_features=public_features,
        leaf_world_weights=transported.leaf_world_weights,
        expected_vocabulary_sha256=evaluator_spec.vocabulary_sha256,
    )
    root_values = reduce_compiled_root_values(
        topology,
        transported,
        leaf_values,
    )
    best = max(root_values.values())
    chosen_action = min(
        action for action, value in root_values.items() if value == best
    )

    return {
        "schema": PACKED_COMPILED_SEARCH_SCHEMA,
        "schema_version": PACKED_COMPILED_SEARCH_SCHEMA_VERSION,
        "generic_compiled_search_schema": COMPILED_SEARCH_SCHEMA,
        "generic_compiled_search_schema_version": COMPILED_SEARCH_SCHEMA_VERSION,
        "method": method,
        "transition_program_digest": topology.program_digest,
        "compiled_topology_digest": topology.topology_digest,
        "posterior_source_digest": packed.source_digest,
        "vocabulary_sha256": packed.vocabulary_sha256,
        "transition_evaluations": topology.transition_evaluations,
        "evaluator_calls": topology.leaf_count,
        "evaluator_batches": 1,
        "chosen_action": chosen_action,
        "root_values": root_values,
        "compiled_shape": {
            "actions": topology.action_count,
            "worlds": topology.world_count,
            "classes": topology.class_count,
            "chance_edges": topology.edge_count,
            "raw_chance_edges": topology.outcome_world_join_plan.raw_join_rows,
            "outcome_join_order": (
                topology.outcome_world_join_plan.selected_order
            ),
            "outcome_join_saved_rows": (
                topology.outcome_world_join_plan.saved_join_rows
            ),
            "observations": len(topology.observation_keys),
            "successor_states": len(topology.successor_states),
            "successor_action_vocabulary": len(
                topology.successor_action_vocabulary
            ),
            "leaves": topology.leaf_count,
            "leaf_world_weight_cells": (
                topology.leaf_count * topology.world_count
            ),
        },
        "numeric_backend": "jax-shared-packed-worlds",
        "outcome_world_join_plan": (
            topology.outcome_world_join_plan.as_record()
        ),
        "physical_execution_stages": list(PACKED_COMPILED_EXECUTION_STAGES),
        "semantic_authority": (
            "python-validated-transition-program + pinned-showdown-vocabulary"
        ),
        "non_claim": (
            "Research path only; live routing remains on the existing search "
            "implementation until equivalence and latency evidence are admitted."
        ),
    }



def choose_packed_compiled_action_bounded(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    vocabulary: ShowdownVocabulary,
    evaluator_spec: PackedBeliefEvaluatorSpec,
    evaluator_params: Mapping[str, Any],
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
    batch_size: int = 1,
) -> dict[str, Any]:
    """Return only the exact best action, pruning bounded losing leaf evaluations.

    This surface deliberately does not return exact values for pruned actions. It may
    therefore serve winner-only decision semantics, but it cannot substitute for the
    full expected-value query, whose result contract includes every exact action value.
    """

    if evaluator_spec.vocabulary_sha256 != vocabulary.vocabulary_sha256:
        raise BeliefEvaluatorError(
            "packed evaluator spec and Showdown vocabulary identities differ"
        )

    try:
        topology = compile_search_topology(
            program_set=program_set,
            posterior=posterior,
            method=method,
            expected_program_schema=expected_program_schema,
            expected_program_schema_version=expected_program_schema_version,
        )
        packed = pack_joint_posterior(posterior, vocabulary)
        transported = transport_posterior_mass(topology, posterior)
    except (CompiledSearchError, ShowdownPackingError) as error:
        raise BeliefEvaluatorError(str(error)) from error

    if packed.world_ids != topology.world_ids:
        raise BeliefEvaluatorError(
            "packed posterior ordering differs from compiled search support"
        )
    if packed.vocabulary_sha256 != evaluator_spec.vocabulary_sha256:
        raise BeliefEvaluatorError("packed posterior vocabulary identity drifted")

    public_features: list[tuple[float, ...]] = []
    try:
        for successor in topology.leaf_public_states:
            public = PublicSuccessorState.from_record(successor)
            public_features.append(
                hashed_features(
                    public.public_state.to_record(),
                    width=evaluator_spec.public_width,
                )
            )
    except ResearchContractError as error:
        raise BeliefEvaluatorError(str(error)) from error

    def evaluate(indices: tuple[int, ...]) -> tuple[float, ...]:
        return predict_packed_shared_world_values(
            evaluator_params,
            packed,
            public_features=[public_features[index] for index in indices],
            leaf_world_weights=transported.leaf_world_weights[list(indices)],
            expected_vocabulary_sha256=evaluator_spec.vocabulary_sha256,
        )

    decision = choose_bounded_action(
        root_actions=topology.root_actions,
        leaf_action_index=topology.leaf_action_index,
        coefficients=tuple(float(value) for value in transported.leaf_mass),
        evaluate=evaluate,
        value_lower_bound=PACKED_VALUE_LOWER_BOUND,
        value_upper_bound=PACKED_VALUE_UPPER_BOUND,
        batch_size=batch_size,
    )

    return {
        "schema": PACKED_BOUNDED_DECISION_SCHEMA,
        "schema_version": PACKED_BOUNDED_DECISION_SCHEMA_VERSION,
        "method": method,
        "transition_program_digest": topology.program_digest,
        "compiled_topology_digest": topology.topology_digest,
        "posterior_source_digest": packed.source_digest,
        "vocabulary_sha256": packed.vocabulary_sha256,
        "transition_evaluations": topology.transition_evaluations,
        "evaluator_calls": decision.evaluated_leaf_count,
        "evaluator_batches": decision.evaluation_batches,
        "chosen_action": decision.chosen_action,
        "chosen_value": decision.chosen_value,
        "winner_certificate": decision.as_record(),
        "compiled_shape": {
            "actions": topology.action_count,
            "worlds": topology.world_count,
            "classes": topology.class_count,
            "chance_edges": topology.edge_count,
            "leaves": topology.leaf_count,
        },
        "numeric_backend": "jax-shared-packed-worlds-bounded",
        "semantic_authority": (
            "python-validated-transition-program + pinned-showdown-vocabulary "
            "+ tanh-certified-value-range"
        ),
        "non_claim": (
            "Winner-only result: pruned actions have certified value intervals, not "
            "exact root values; this cannot replace decision.expected_value."
        ),
    }
