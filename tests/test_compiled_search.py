from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

import pytest

from azelficoast.core.compiled_search import (
    CardinalityEnvelope,
    JOIN_ORDER_AGGREGATE_FIRST,
    JOIN_ORDER_EXPAND_FIRST,
    compile_search_topology,
    estimate_search_cardinality_lower_bound,
    materialize_compiled_frontier,
    reduce_compiled_root_values,
    search_transition_program_adaptive,
    search_transition_program_compiled,
)
from azelficoast.core.search import search_transition_program


class WeightedPayoffEvaluator:
    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        del legal_actions
        action = public_state["root_action"]
        return math.fsum(
            float(world["weight"]) * float(world["hidden"]["values"][action])
            for world in posterior_worlds
        )


def _inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    world_ids = ["w0", "w1", "w2"]
    posterior = {
        "worlds": [
            {
                "world_id": "w0",
                "weight": 0.2,
                "hidden": {"values": {"A": 1.0, "B": -0.5}},
            },
            {
                "world_id": "w1",
                "weight": 0.3,
                "hidden": {"values": {"A": -0.25, "B": 0.75}},
            },
            {
                "world_id": "w2",
                "weight": 0.5,
                "hidden": {"values": {"A": 0.5, "B": 0.25}},
            },
        ]
    }
    programs = [
        {
            "action": "A",
            "classes_out": 2,
            "classes": [
                {
                    "member_world_ids": ["w0", "w1"],
                    "outcomes": [
                        {
                            "probability": 0.25,
                            "observation": {"signal": "shared"},
                            "successor": {"root_action": "A", "stage": "shared"},
                            "legal_actions": ["continue", "switch"],
                        },
                        {
                            "probability": 0.75,
                            "observation": {"signal": "other"},
                            "successor": {"root_action": "A", "stage": "other"},
                            "legal_actions": ["continue"],
                        },
                    ],
                },
                {
                    "member_world_ids": ["w2"],
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"signal": "shared"},
                            "successor": {"root_action": "A", "stage": "shared"},
                            "legal_actions": ["continue"],
                        }
                    ],
                },
            ],
        },
        {
            "action": "B",
            "classes_out": 1,
            "classes": [
                {
                    "member_world_ids": world_ids,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"signal": "b"},
                            "successor": {"root_action": "B", "stage": "done"},
                            "legal_actions": ["continue"],
                        }
                    ],
                }
            ],
        },
    ]
    program_set = {
        "schema": "example.transition-program-set",
        "schema_version": 1,
        "world_ids": world_ids,
        "legal_actions": ["A", "B"],
        "programs": programs,
    }
    return program_set, posterior


@pytest.mark.parametrize("method", ["determinization", "information_set"])
def test_compiled_search_matches_reference_root_values_and_choice(method: str) -> None:
    pytest.importorskip("jax")
    program_set, posterior = _inputs()
    evaluator = WeightedPayoffEvaluator()

    reference = search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method=method,
        evaluator=evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    compiled = search_transition_program_compiled(
        program_set=program_set,
        posterior=posterior,
        method=method,
        evaluator=evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert compiled["transition_program_digest"] == reference["transition_program_digest"]
    assert compiled["transition_evaluations"] == reference["transition_evaluations"]
    assert compiled["evaluator_calls"] == reference["evaluator_calls"]
    assert compiled["chosen_action"] == reference["chosen_action"]
    assert compiled["root_values"] == pytest.approx(reference["root_values"], abs=1e-6)


def test_compiled_topology_exposes_authorized_world_class_and_observation_incidence() -> None:
    pytest.importorskip("jax")
    program_set, posterior = _inputs()
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert topology.world_to_class[0][0] == topology.world_to_class[0][1]
    assert topology.world_to_class[0][2] != topology.world_to_class[0][0]
    assert len(topology.observation_keys) == 3
    assert topology.class_count == 3
    assert topology.transition_evaluations == 3
    assert topology.edge_count == 8
    assert topology.leaf_count == 3
    assert set(topology.leaf_action_index) == {0, 1}
    assert len(topology.topology_digest) == 64
    assert all(character in "0123456789abcdef" for character in topology.topology_digest)


def test_jax_transport_preserves_action_mass_and_normalizes_leaf_posteriors() -> None:
    pytest.importorskip("jax")
    np = pytest.importorskip("numpy")
    program_set, posterior = _inputs()
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    frontier, transported = materialize_compiled_frontier(topology, posterior)

    assert transported.leaf_mass.sum() == pytest.approx(2.0, abs=1e-6)
    assert np.allclose(transported.leaf_world_mass.sum(axis=1), transported.leaf_mass)
    assert np.allclose(
        transported.leaf_world_weights.sum(axis=1),
        np.ones(topology.leaf_count),
        atol=1e-6,
    )
    per_action = {action: 0.0 for action in topology.root_actions}
    for contribution in frontier.contributions:
        per_action[contribution.root_action] += contribution.coefficient
    assert per_action == pytest.approx({"A": 1.0, "B": 1.0}, abs=1e-6)


def test_compiled_topology_is_independent_of_posterior_weights() -> None:
    pytest.importorskip("jax")
    program_set, posterior = _inputs()
    first = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    changed = copy.deepcopy(posterior)
    changed["worlds"][0]["weight"] = 0.6
    changed["worlds"][1]["weight"] = 0.2
    changed["worlds"][2]["weight"] = 0.2
    second = compile_search_topology(
        program_set=program_set,
        posterior=changed,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert second.topology_digest == first.topology_digest
    assert second.as_record() == first.as_record()

    first_frontier, first_mass = materialize_compiled_frontier(first, posterior)
    second_frontier, second_mass = materialize_compiled_frontier(second, changed)
    assert first_mass.leaf_mass.tolist() != second_mass.leaf_mass.tolist()
    assert first_frontier.root_actions == second_frontier.root_actions


def test_compiled_reduction_matches_frontier_reduce() -> None:
    pytest.importorskip("jax")
    program_set, posterior = _inputs()
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    frontier, transported = materialize_compiled_frontier(topology, posterior)
    values = tuple(0.125 * (index + 1) for index in range(topology.leaf_count))

    reference = frontier.reduce(values)
    compiled = reduce_compiled_root_values(topology, transported, values)

    assert compiled == pytest.approx(reference, abs=1e-6)


def test_compiler_rejects_observation_partition_with_multiple_public_successors() -> None:
    program_set, posterior = _inputs()
    bad = copy.deepcopy(program_set)
    # w2 shares observation "shared" with the first A class. If its public successor
    # differs, information-set compilation must fail before JAX sees the topology.
    bad["programs"][0]["classes"][1]["outcomes"][0]["successor"] = {
        "root_action": "A",
        "stage": "secretly-different",
    }

    with pytest.raises(
        ValueError,
        match="multiple successor public states",
    ):
        compile_search_topology(
            program_set=bad,
            posterior=posterior,
            method="information_set",
            expected_program_schema="example.transition-program-set",
            expected_program_schema_version=1,
        )


def test_compiled_path_preserves_deterministic_tie_breaking_for_exact_tie() -> None:
    pytest.importorskip("jax")
    program_set, posterior = _inputs()
    for world in posterior["worlds"]:
        world["hidden"]["values"] = {"A": 0.5, "B": 0.5}

    reference = search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    compiled = search_transition_program_compiled(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert reference["chosen_action"] == "A"
    assert compiled["chosen_action"] == "A"
    assert compiled["root_values"] == pytest.approx(reference["root_values"], abs=1e-6)



def test_cardinality_lower_bound_never_claims_realized_work_is_smaller() -> None:
    program_set, posterior = _inputs()
    for method in ("information_set", "determinization"):
        lower = estimate_search_cardinality_lower_bound(
            program_set=program_set,
            posterior=posterior,
            method=method,
            expected_program_schema="example.transition-program-set",
            expected_program_schema_version=1,
        )
        topology = compile_search_topology(
            program_set=program_set,
            posterior=posterior,
            method=method,
            expected_program_schema="example.transition-program-set",
            expected_program_schema_version=1,
        )

        assert lower.world_count == topology.world_count
        assert lower.class_count == topology.class_count
        assert lower.chance_edge_count <= topology.edge_count
        assert lower.leaf_count <= topology.leaf_count
        assert lower.dense_leaf_world_cells <= (
            topology.leaf_count * topology.world_count
        )


def test_adaptive_search_replans_before_dense_transport_on_cardinality_surprise() -> None:
    program_set, posterior = _inputs()
    reference = search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method="determinization",
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    result = search_transition_program_adaptive(
        program_set=program_set,
        posterior=posterior,
        method="determinization",
        evaluator=WeightedPayoffEvaluator(),
        envelope=CardinalityEnvelope(
            max_world_count=3,
            max_class_count=3,
            max_chance_edge_count=4,
            max_leaf_count=6,
            max_dense_leaf_world_cells=18,
        ),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert result["physical_search_path"] == "python-frontier"
    assert result["cardinality_plan"]["initial_path"] == "compiled-jax"
    assert result["cardinality_plan"]["final_path"] == "python-frontier"
    assert result["cardinality_plan"]["replanned"] is True
    assert "chance_edges" in result["cardinality_plan"]["violations"]
    assert result["cardinality_plan"]["observed"]["chance_edges"] == 8
    assert result["transition_evaluations"] == reference["transition_evaluations"]
    assert result["evaluator_calls"] == reference["evaluator_calls"]
    assert result["chosen_action"] == reference["chosen_action"]
    assert result["root_values"] == reference["root_values"]


def test_adaptive_search_rejects_compiled_path_from_lower_bound_without_jax() -> None:
    program_set, posterior = _inputs()
    result = search_transition_program_adaptive(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        evaluator=WeightedPayoffEvaluator(),
        envelope=CardinalityEnvelope(
            max_world_count=2,
            max_class_count=3,
            max_chance_edge_count=100,
            max_leaf_count=100,
            max_dense_leaf_world_cells=1000,
        ),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert result["physical_search_path"] == "python-frontier"
    assert result["cardinality_plan"]["initial_path"] == "python-frontier"
    assert result["cardinality_plan"]["observed"] is None
    assert result["cardinality_plan"]["violations"] == ["worlds"]


def test_adaptive_search_keeps_compiled_path_inside_observed_envelope() -> None:
    pytest.importorskip("jax")
    program_set, posterior = _inputs()
    result = search_transition_program_adaptive(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        evaluator=WeightedPayoffEvaluator(),
        envelope=CardinalityEnvelope(
            max_world_count=3,
            max_class_count=3,
            max_chance_edge_count=8,
            max_leaf_count=3,
            max_dense_leaf_world_cells=9,
        ),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert result["physical_search_path"] == "compiled-jax"
    assert result["cardinality_plan"]["replanned"] is False
    assert result["cardinality_plan"]["observed"] == {
        "worlds": 3,
        "classes": 3,
        "chance_edges": 8,
        "leaves": 3,
        "dense_leaf_world_cells": 9,
    }



def _duplicate_outcome_program(
    *,
    split_legal_surface: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    program_set, posterior = _inputs()
    changed = copy.deepcopy(program_set)
    row = changed["programs"][1]["classes"][0]
    template = copy.deepcopy(row["outcomes"][0])
    outcomes = []
    for index in range(8 if split_legal_surface else 4):
        outcome = copy.deepcopy(template)
        outcome["probability"] = 1.0 / (8 if split_legal_surface else 4)
        if split_legal_surface and index % 2:
            outcome["legal_actions"] = ["continue", "switch"]
        outcomes.append(outcome)
    row["outcomes"] = outcomes
    return changed, posterior


def test_join_planner_keeps_expand_first_when_grouping_cannot_pay_for_it() -> None:
    program_set, posterior = _inputs()
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    plan = topology.outcome_world_join_plan
    assert plan.selected_order == JOIN_ORDER_EXPAND_FIRST
    assert plan.raw_join_rows == plan.grouped_join_rows == topology.edge_count
    assert plan.aggregate_first_work_units > plan.expand_first_work_units
    assert plan.saved_join_rows == 0


@pytest.mark.parametrize("method", ["information_set", "determinization"])
def test_join_planner_pushes_exact_outcome_aggregation_before_world_join(
    method: str,
) -> None:
    pytest.importorskip("jax")
    program_set, posterior = _duplicate_outcome_program()
    reference = search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method=method,
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    compiled = search_transition_program_compiled(
        program_set=program_set,
        posterior=posterior,
        method=method,
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method=method,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    plan = topology.outcome_world_join_plan
    assert plan.selected_order == JOIN_ORDER_AGGREGATE_FIRST
    assert plan.raw_outcome_rows == 7
    assert plan.grouped_outcome_rows == 4
    assert plan.raw_join_rows == 17
    assert plan.grouped_join_rows == 8
    assert plan.expand_first_work_units == 17
    assert plan.aggregate_first_work_units == 15
    assert plan.saved_join_rows == 9
    assert topology.edge_count == 8
    assert topology.transition_evaluations == 3
    assert compiled["compiled_shape"]["raw_chance_edges"] == 17
    assert compiled["compiled_shape"]["chance_edges"] == 8
    assert compiled["compiled_shape"]["outcome_join_saved_rows"] == 9
    assert compiled["root_values"] == pytest.approx(reference["root_values"], abs=1e-6)
    assert compiled["chosen_action"] == reference["chosen_action"]


def test_outcome_aggregation_does_not_cross_successor_legal_action_surface() -> None:
    pytest.importorskip("jax")
    program_set, posterior = _duplicate_outcome_program(split_legal_surface=True)
    reference = search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    compiled = search_transition_program_compiled(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    plan = topology.outcome_world_join_plan
    assert plan.selected_order == JOIN_ORDER_AGGREGATE_FIRST
    assert plan.raw_outcome_rows == 11
    assert plan.grouped_outcome_rows == 5
    b_action_index = topology.root_actions.index("B")
    b_leaf_legal = {
        topology.leaf_legal_actions[index]
        for index, action_index in enumerate(topology.leaf_action_index)
        if action_index == b_action_index
    }
    assert b_leaf_legal == {("continue",)}
    assert compiled["root_values"] == pytest.approx(reference["root_values"], abs=1e-6)
    assert compiled["chosen_action"] == reference["chosen_action"]


def test_cardinality_lower_bound_remains_below_aggregated_physical_edges() -> None:
    program_set, posterior = _duplicate_outcome_program()
    lower = estimate_search_cardinality_lower_bound(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    topology = compile_search_topology(
        program_set=program_set,
        posterior=posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert lower.chance_edge_count == 4
    assert lower.chance_edge_count <= topology.edge_count == 8
