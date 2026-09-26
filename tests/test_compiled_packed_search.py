from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import pytest

from azelficoast.belief.compiled_search import (
    PACKED_COMPILED_EXECUTION_STAGES,
    search_packed_compiled_transition_program,
)
from azelficoast.belief.packed_evaluator import (
    PackedBeliefEvaluatorSpec,
    build_packed_evaluator_input,
    init_packed_params,
    predict_packed_values,
)
from azelficoast.belief.showdown_packing import ShowdownVocabulary
from azelficoast.belief.sql_compiled_search import (
    SQLPackedLoweringError,
    compile_packed_sql_decision_query,
    search_sql_packed_transition_program,
)
from azelficoast.core.compiled_search import compile_search_topology
from azelficoast.core.search import search_transition_program
from azelficoast.core.statistics import PlannerStatistics
from azelficoast.core.sql import DEFAULT_DECISION_SQL, prepare_decision_query


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vocabulary() -> ShowdownVocabulary:
    material: dict[str, object] = {
        "schema": "azelficoast.showdown-vocabulary",
        "schema_version": 1,
        "showdown_commit": "a" * 40,
        "generation": 9,
        "identity": {},
        "species": [
            {
                "id": "rotomwash",
                "num": 479,
                "forme_index": 1,
                "base_species_id": "rotom",
                "forme": "wash",
            }
        ],
        "moves": [
            {"id": "hydropump", "num": 56},
            {"id": "voltswitch", "num": 521},
            {"id": "willowisp", "num": 261},
        ],
        "items": [
            {"id": "choicescarf", "num": 287},
            {"id": "leftovers", "num": 234},
        ],
        "abilities": [{"id": "levitate", "num": 26}],
        "types": [{"id": "water", "index": 1}],
        "natures": [{"id": "timid", "index": 1}],
        "roles": [{"id": "fastpivot", "index": 1}],
    }
    return ShowdownVocabulary.from_record(
        {**material, "vocabulary_sha256": _digest(material)}
    )


def _member(*, item: str, moves: list[str]) -> dict[str, object]:
    return {
        "species": "Rotom-Wash",
        "level": 80,
        "gender": "N",
        "ability": "Levitate",
        "item": item,
        "moves": moves,
        "tera_type": "Water",
        "role": "Fast Pivot",
        "nature": "Timid",
        "evs": {"hp": 84, "spa": 84, "spe": 84},
        "ivs": {},
        "was_lead": True,
    }


def _posterior() -> dict[str, Any]:
    return {
        "schema": "azelficoast.joint-random-battle-posterior",
        "schema_version": 1,
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "support_status": "sufficient",
        "showdown_commit": "a" * 40,
        "construction": {
            "kind": "full-team-generator-rejection-particles",
            "posterior_treatment": "generator_faithful_joint_empirical",
            "preserves_joint_team_set_correlations": True,
        },
        "worlds": [
            {
                "world_id": "scarf",
                "weight": 0.4,
                "hidden": {
                    "team": [
                        _member(
                            item="Choice Scarf",
                            moves=["Hydro Pump", "Volt Switch"],
                        )
                    ]
                },
            },
            {
                "world_id": "leftovers",
                "weight": 0.6,
                "hidden": {
                    "team": [
                        _member(
                            item="Leftovers",
                            moves=["Hydro Pump", "Will-O-Wisp"],
                        )
                    ]
                },
            },
        ],
        "public_evidence": {
            "opponent_team_size": 1,
            "revealed": [{"species": "Rotom-Wash", "was_lead": True}],
        },
    }


def _program() -> dict[str, Any]:
    return {
        "schema": "example.transition-program-set",
        "schema_version": 1,
        "world_ids": ["scarf", "leftovers"],
        "legal_actions": ["hide", "reveal"],
        "programs": [
            {
                "action": "hide",
                "classes_out": 1,
                "classes": [
                    {
                        "member_world_ids": ["scarf", "leftovers"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"signal": "same"},
                                "successor": {
                                    "turn": 8,
                                    "weather": "rain",
                                    "root_action": "hide",
                                },
                                "legal_actions": ["move:a", "move:b"],
                            }
                        ],
                    }
                ],
            },
            {
                "action": "reveal",
                "classes_out": 2,
                "classes": [
                    {
                        "member_world_ids": ["scarf"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"signal": "scarf"},
                                "successor": {
                                    "turn": 8,
                                    "weather": "rain",
                                    "root_action": "reveal",
                                },
                                "legal_actions": ["move:a"],
                            }
                        ],
                    },
                    {
                        "member_world_ids": ["leftovers"],
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"signal": "leftovers"},
                                "successor": {
                                    "turn": 8,
                                    "weather": "rain",
                                    "root_action": "reveal",
                                },
                                "legal_actions": ["move:a", "switch:b"],
                            }
                        ],
                    },
                ],
            },
        ],
    }


class MaterializedPackedEvaluator:
    def __init__(
        self,
        *,
        posterior_template: Mapping[str, Any],
        vocabulary: ShowdownVocabulary,
        spec: PackedBeliefEvaluatorSpec,
        params: Mapping[str, Any],
    ) -> None:
        self.posterior_template = posterior_template
        self.vocabulary = vocabulary
        self.spec = spec
        self.params = params

    def values(self, leaves: Sequence[Any]) -> tuple[float, ...]:
        inputs = []
        for leaf in leaves:
            posterior = copy.deepcopy(dict(self.posterior_template))
            posterior["worlds"] = [copy.deepcopy(dict(world)) for world in leaf.posterior]
            inputs.append(
                build_packed_evaluator_input(
                    public_state=leaf.public_state,
                    posterior=posterior,
                    legal_actions=leaf.legal_actions,
                    vocabulary=self.vocabulary,
                    spec=self.spec,
                )
            )
        return predict_packed_values(self.params, inputs)


@pytest.mark.parametrize("method", ["determinization", "information_set"])
def test_shared_world_compiled_search_matches_materialized_packed_frontier(
    method: str,
) -> None:
    pytest.importorskip("jax")
    vocabulary = _vocabulary()
    spec = PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=16,
        action_width=8,
        embedding_width=6,
        member_hidden_width=9,
        world_hidden_width=10,
        hidden_width=12,
    )
    params = init_packed_params(spec, seed=43)
    posterior = _posterior()
    program = _program()

    reference = search_transition_program(
        program_set=program,
        posterior=posterior,
        method=method,
        evaluator=MaterializedPackedEvaluator(
            posterior_template=posterior,
            vocabulary=vocabulary,
            spec=spec,
            params=params,
        ),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    compiled = search_packed_compiled_transition_program(
        program_set=program,
        posterior=posterior,
        method=method,
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert compiled["transition_program_digest"] == reference["transition_program_digest"]
    assert compiled["transition_evaluations"] == reference["transition_evaluations"]
    assert compiled["evaluator_calls"] == reference["evaluator_calls"]
    assert compiled["evaluator_batches"] == 1
    assert compiled["chosen_action"] == reference["chosen_action"]
    assert compiled["root_values"] == pytest.approx(reference["root_values"], abs=1e-6)


def test_compiled_topology_carries_successor_and_legal_action_tensors() -> None:
    topology = compile_search_topology(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    arrays = topology.as_numpy()

    assert topology.successor_action_vocabulary == (
        "move:a",
        "move:b",
        "switch:b",
    )
    assert arrays["leaf_legal_mask"].shape == (
        topology.leaf_count,
        len(topology.successor_action_vocabulary),
    )
    assert len(topology.successor_states) == 2
    assert len(topology.edge_successor_index) == topology.edge_count
    assert len(topology.leaf_successor_index) == topology.leaf_count
    assert all(any(row) for row in topology.leaf_legal_mask)


def test_shared_world_compiled_search_reuses_topology_across_prior_weights() -> None:
    pytest.importorskip("jax")
    vocabulary = _vocabulary()
    spec = PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=16,
        action_width=8,
        embedding_width=6,
        member_hidden_width=9,
        world_hidden_width=10,
        hidden_width=12,
    )
    params = init_packed_params(spec, seed=47)
    program = _program()
    first_posterior = _posterior()
    second_posterior = _posterior()
    second_posterior["worlds"][0]["weight"] = 0.8
    second_posterior["worlds"][1]["weight"] = 0.2

    first_topology = compile_search_topology(
        program_set=program,
        posterior=first_posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    second_topology = compile_search_topology(
        program_set=program,
        posterior=second_posterior,
        method="information_set",
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    assert first_topology.topology_digest == second_topology.topology_digest

    first = search_packed_compiled_transition_program(
        program_set=program,
        posterior=first_posterior,
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    second = search_packed_compiled_transition_program(
        program_set=program,
        posterior=second_posterior,
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert first["compiled_topology_digest"] == second["compiled_topology_digest"]
    assert first["root_values"] != second["root_values"]



def test_sql_decision_query_lowers_to_current_packed_jax_physical_plan() -> None:
    prepared = prepare_decision_query(DEFAULT_DECISION_SQL)
    plan = compile_packed_sql_decision_query(prepared)

    assert plan.physical_execution_stages == PACKED_COMPILED_EXECUTION_STAGES
    assert [binding.operator.value for binding in plan.bindings] == [
        "scan",
        "filter",
        "project",
        "partition",
        "transition",
        "observe",
        "update_belief",
        "evaluate",
        "aggregate",
    ]
    assert plan.bindings[0].implementation == "pack_joint_posterior"
    assert plan.bindings[1].implementation == "pack_joint_posterior"
    assert plan.bindings[-2].implementation == "predict_packed_shared_world_values"
    assert plan.bindings[-1].implementation == "reduce_compiled_root_values"
    assert plan.numeric_backend == "jax-shared-packed-worlds"
    assert plan.plan_sha256.startswith("sha256:")


def test_sql_packed_lowering_fails_closed_for_unreviewed_semantic_change() -> None:
    changed = DEFAULT_DECISION_SQL.replace(
        "ORDER BY expected_value DESC",
        "ORDER BY expected_value ASC",
    )
    prepared = prepare_decision_query(changed)

    with pytest.raises(
        SQLPackedLoweringError,
        match="no reviewed packed/JAX lowering",
    ):
        compile_packed_sql_decision_query(prepared)


def test_sql_generated_plan_executes_same_packed_jax_path_as_hand_built_search() -> None:
    pytest.importorskip("jax")
    vocabulary = _vocabulary()
    spec = PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=16,
        action_width=8,
        embedding_width=6,
        member_hidden_width=9,
        world_hidden_width=10,
        hidden_width=12,
    )
    params = init_packed_params(spec, seed=53)
    posterior = _posterior()
    program = _program()

    hand_built = search_packed_compiled_transition_program(
        program_set=program,
        posterior=posterior,
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    generated = search_sql_packed_transition_program(
        program_set=program,
        posterior=posterior,
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert generated["compiled_topology_digest"] == hand_built["compiled_topology_digest"]
    assert generated["transition_evaluations"] == hand_built["transition_evaluations"]
    assert generated["evaluator_calls"] == hand_built["evaluator_calls"]
    assert generated["evaluator_batches"] == hand_built["evaluator_batches"] == 1
    assert generated["chosen_action"] == hand_built["chosen_action"]
    assert generated["root_values"] == pytest.approx(hand_built["root_values"], abs=1e-6)
    assert generated["physical_execution_stages"] == list(
        PACKED_COMPILED_EXECUTION_STAGES
    )
    assert generated["sql_physical_plan"]["physical_execution_stages"] == list(
        PACKED_COMPILED_EXECUTION_STAGES
    )
    assert generated["sql_execution_matches_hand_built_plan"] is True



def test_sql_planner_statistics_forecast_filter_and_partition_cardinality() -> None:
    pytest.importorskip("jax")
    vocabulary = _vocabulary()
    spec = PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=16,
        action_width=8,
        embedding_width=6,
        member_hidden_width=9,
        world_hidden_width=10,
        hidden_width=12,
    )
    params = init_packed_params(spec, seed=59)
    statistics = PlannerStatistics()

    first = search_sql_packed_transition_program(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        planner_statistics=statistics,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    second = search_sql_packed_transition_program(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        planner_statistics=statistics,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert first["sql_cardinality_forecast"]["filter"]["observations"] == 0
    assert first["sql_cardinality_forecast"]["partition"]["observations"] == 0
    assert first["sql_cardinality_observed"] == {
        "filter": {"input_rows": 2, "output_rows": 2},
        "partition": {"input_rows": 4, "output_rows": 3},
    }
    assert second["sql_cardinality_forecast"]["filter"]["observations"] == 1
    assert second["sql_cardinality_forecast"]["partition"]["observations"] == 1
    assert second["sql_cardinality_forecast"]["filter"]["estimated_output_rows"] == 2
    assert second["sql_cardinality_forecast"]["partition"]["estimated_output_rows"] == 3
    assert second["sql_cardinality_error"] == {
        "filter_rows": 0,
        "partition_groups": 0,
    }


def test_sql_planner_statistics_do_not_change_query_semantics() -> None:
    pytest.importorskip("jax")
    vocabulary = _vocabulary()
    spec = PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=16,
        action_width=8,
        embedding_width=6,
        member_hidden_width=9,
        world_hidden_width=10,
        hidden_width=12,
    )
    params = init_packed_params(spec, seed=61)
    statistics = PlannerStatistics()
    for _ in range(25):
        statistics.observe(
            operator=__import__(
                "azelficoast.core.planning",
                fromlist=["LogicalOperator"],
            ).LogicalOperator.FILTER,
            signature=(
                "sha256:"
                + hashlib.sha256(DEFAULT_DECISION_SQL.encode("utf-8")).hexdigest()
                + ":filter:hidden_worlds:active-positive:"
                + "azelficoast.joint-random-battle-posterior"
            ),
            input_rows=1000,
            output_rows=1,
        )

    baseline = search_sql_packed_transition_program(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    forecasted = search_sql_packed_transition_program(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        planner_statistics=statistics,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert forecasted["chosen_action"] == baseline["chosen_action"]
    assert forecasted["root_values"] == pytest.approx(
        baseline["root_values"],
        abs=1e-6,
    )
    assert forecasted["compiled_topology_digest"] == baseline["compiled_topology_digest"]
