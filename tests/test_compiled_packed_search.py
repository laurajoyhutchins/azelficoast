from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import pytest

import azelficoast.belief.compiled_search as compiled_search_module
from azelficoast.belief.compiled_search import (
    PACKED_COMPILED_EXECUTION_STAGES,
    choose_packed_compiled_action_bounded,
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
    explain_sql_packed_transition_program,
    search_sql_packed_transition_program,
)
from azelficoast.core.compiled_search import compile_search_topology
from azelficoast.core.planning import LogicalOperator
from azelficoast.core.search import search_transition_program
from azelficoast.core.statistics import PlannerStatistics
from azelficoast.core.sql import (
    DECISION_QUERY_SEMANTIC_ID,
    DEFAULT_DECISION_SQL,
    MAXIMIN_SQL,
    prepare_decision_query,
    prepare_policy_query,
)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


EQUIVALENT_INLINE_DECISION_SQL = """
SELECT
    t.action_id AS action_id,
    SUM(w.weight * e.value) AS expected_value
FROM evaluations AS e
JOIN transitions AS t ON e.successor_id = t.successor_id
JOIN hidden_worlds AS w ON t.world_id = w.world_id
JOIN legal_actions AS a ON a.action_id = t.action_id
WHERE w.weight > 0 AND w.active = 1
GROUP BY t.action_id
ORDER BY expected_value DESC, action_id ASC
""".strip()


PROGRAM_EQUIVALENT_DECISION_SQL = """
SELECT
    terms.action_id AS action_id,
    SUM((terms.weight * terms.value)) AS expected_value
FROM action_value_terms AS terms
GROUP BY terms.action_id
ORDER BY 2 DESC, 1 ASC
""".strip()


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



def test_sql_optimizer_explain_is_read_only_and_exposes_multi_stage_plan() -> None:
    statistics = PlannerStatistics()

    explanation = explain_sql_packed_transition_program(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        planner_statistics=statistics,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert explanation["schema"] == "azelficoast.sql-optimizer-explain"
    assert explanation["schema_version"] == 1
    assert explanation["execution"] == {
        "performed": False,
        "evaluator_calls": 0,
        "action_selected": False,
        "planner_statistics_mutated": False,
    }
    assert explanation["authority"]["mechanics"] == (
        "verified-transition-program-outside-sql"
    )
    assert explanation["authority"]["information_sets"] == (
        "python-validated-compiled-topology"
    )
    assert explanation["semantic"]["logical_operators"] == [
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
    assert [
        group["name"] for group in explanation["optimizer"]["groups"]
    ] == [
        "posterior-pack",
        "authorized-topology",
        "belief-transport",
        "packed-evaluator",
        "root-reduction",
    ]
    assert explanation["optimizer"]["outcome_world_join"]["selected_order"] == (
        "expand-outcomes-before-world-join"
    )
    assert explanation["optimizer"]["outcome_world_join"]["saved_join_rows"] == 0
    cardinality = explanation["optimizer"]["cardinality"]
    assert cardinality["lower_bound"]["worlds"] == 2
    assert cardinality["lower_bound"]["classes"] == 3
    assert cardinality["realized"]["actions"] == 2
    assert cardinality["realized"]["worlds"] == 2
    assert cardinality["realized"]["classes"] == 3
    assert cardinality["realized"]["leaves"] == 3
    assert cardinality["forecast"]["filter"]["observations"] == 0
    assert cardinality["forecast"]["partition"]["observations"] == 0
    assert explanation["physical"]["numeric_backend"] == (
        "jax-shared-packed-worlds"
    )


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
    assert plan.semantic_identity == DECISION_QUERY_SEMANTIC_ID
    assert plan.plan_sha256.startswith("sha256:")


def test_sql_packed_lowering_fails_closed_for_unreviewed_semantic_change() -> None:
    changed = DEFAULT_DECISION_SQL.replace(
        "ORDER BY expected_value DESC",
        "ORDER BY expected_value ASC",
    )
    prepared = prepare_decision_query(changed)

    with pytest.raises(
        SQLPackedLoweringError,
        match="no reviewed packed/JAX semantic identity",
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
            operator=LogicalOperator.FILTER,
            signature=(
                DECISION_QUERY_SEMANTIC_ID
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



def test_equivalent_sql_rewrites_share_one_packed_physical_plan_identity() -> None:
    canonical = compile_packed_sql_decision_query(
        prepare_decision_query(DEFAULT_DECISION_SQL)
    )
    rewritten = compile_packed_sql_decision_query(
        prepare_decision_query(EQUIVALENT_INLINE_DECISION_SQL)
    )

    assert rewritten.sql_sha256 != canonical.sql_sha256
    assert rewritten.semantic_identity == canonical.semantic_identity
    assert rewritten.semantic_identity == DECISION_QUERY_SEMANTIC_ID
    assert rewritten.plan_sha256 == canonical.plan_sha256
    assert rewritten.equivalence_scope == "reviewed-relational"
    assert rewritten.physical_execution_stages == canonical.physical_execution_stages

    program_equivalent = compile_packed_sql_decision_query(
        prepare_decision_query(PROGRAM_EQUIVALENT_DECISION_SQL)
    )
    assert program_equivalent.plan_sha256 == canonical.plan_sha256
    assert program_equivalent.semantic_identity == canonical.semantic_identity
    assert program_equivalent.equivalence_rule == "sqlite-program-equivalence"
    assert program_equivalent.equivalence_scope == "sqlite-version-bound"
    assert program_equivalent.as_record()["schema_version"] == 3


def test_equivalent_sql_executes_through_same_packed_jax_plan_and_statistics() -> None:
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
    params = init_packed_params(spec, seed=67)
    statistics = PlannerStatistics()

    canonical = search_sql_packed_transition_program(
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
    rewritten = search_sql_packed_transition_program(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        planner_statistics=statistics,
        sql=EQUIVALENT_INLINE_DECISION_SQL,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert rewritten["sql_query_sha256"] != canonical["sql_query_sha256"]
    assert rewritten["sql_semantic_identity"] == canonical["sql_semantic_identity"]
    assert rewritten["sql_equivalence_scope"] == "reviewed-relational"
    assert rewritten["sql_physical_plan"]["plan_sha256"] == (
        canonical["sql_physical_plan"]["plan_sha256"]
    )
    assert rewritten["compiled_topology_digest"] == canonical["compiled_topology_digest"]
    assert rewritten["chosen_action"] == canonical["chosen_action"]
    assert rewritten["root_values"] == pytest.approx(canonical["root_values"], abs=1e-6)
    assert rewritten["sql_cardinality_forecast"]["filter"]["observations"] == 1
    assert rewritten["sql_cardinality_forecast"]["partition"]["observations"] == 1



def test_sql_partition_statistics_are_conditioned_on_correlation_regime() -> None:
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
    params = init_packed_params(spec, seed=71)
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
    repeated = search_sql_packed_transition_program(
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

    assert first["sql_extended_statistics"]["signature"].startswith("sha256:")
    assert first["sql_cardinality_forecast"]["partition"]["observations"] == 0
    assert repeated["sql_cardinality_forecast"]["partition"]["observations"] == 1
    assert (
        first["sql_cardinality_forecast"]["partition"]["signature"]
        == repeated["sql_cardinality_forecast"]["partition"]["signature"]
    )
    assert first["chosen_action"] == repeated["chosen_action"]
    assert first["root_values"] == pytest.approx(repeated["root_values"], abs=1e-6)



def test_composable_sql_policy_does_not_inherit_expected_value_lowering() -> None:
    prepared = prepare_policy_query(MAXIMIN_SQL)

    assert prepared.semantic_identity is not None
    assert prepared.semantic_identity != DECISION_QUERY_SEMANTIC_ID
    assert prepared.logical is None
    with pytest.raises(
        SQLPackedLoweringError,
        match="no reviewed packed/JAX semantic identity",
    ):
        compile_packed_sql_decision_query(prepared)



def test_bounded_packed_winner_skips_certifiably_losing_leaf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("jax")
    vocabulary = _vocabulary()
    spec = PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=4,
        action_width=4,
        embedding_width=4,
        member_hidden_width=4,
        world_hidden_width=4,
        hidden_width=4,
    )
    params = init_packed_params(spec, seed=79)

    def fake_features(
        value: Mapping[str, Any],
        *,
        width: int,
    ) -> tuple[float, ...]:
        marker = 1.0 if value.get("root_action") == "hide" else -1.0
        return tuple(marker for _ in range(width))

    evaluated_rows: list[tuple[float, ...]] = []

    def fake_values(
        evaluator_params: Mapping[str, Any],
        packed: Any,
        *,
        public_features: Sequence[Sequence[float]],
        leaf_world_weights: Any,
        expected_vocabulary_sha256: str | None = None,
    ) -> tuple[float, ...]:
        del evaluator_params, packed, leaf_world_weights, expected_vocabulary_sha256
        rows = [tuple(float(item) for item in row) for row in public_features]
        evaluated_rows.extend(rows)
        return tuple(0.9 if row[0] > 0.0 else -0.9 for row in rows)

    monkeypatch.setattr(compiled_search_module, "hashed_features", fake_features)
    monkeypatch.setattr(
        compiled_search_module,
        "predict_packed_shared_world_values",
        fake_values,
    )

    full = search_packed_compiled_transition_program(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    full_rows = len(evaluated_rows)
    evaluated_rows.clear()

    bounded = choose_packed_compiled_action_bounded(
        program_set=_program(),
        posterior=_posterior(),
        method="information_set",
        vocabulary=vocabulary,
        evaluator_spec=spec,
        evaluator_params=params,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert full["chosen_action"] == bounded["chosen_action"] == "hide"
    assert bounded["chosen_value"] == pytest.approx(full["root_values"]["hide"])
    assert bounded["evaluator_calls"] < full["evaluator_calls"] == full_rows
    assert bounded["winner_certificate"]["pruned_leaf_count"] == 1
    assert bounded["winner_certificate"]["pruned_actions"] == ["reveal"]
    assert "root_values" not in bounded
    reveal_interval = next(
        row
        for row in bounded["winner_certificate"]["action_value_intervals"]
        if row["action"] == "reveal"
    )
    assert reveal_interval["exact"] is False
    assert reveal_interval["upper"] < bounded["chosen_value"]
