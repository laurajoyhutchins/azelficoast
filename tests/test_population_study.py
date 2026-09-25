from __future__ import annotations

from azelficoast.corpus import DecisionFixture
from azelficoast.research.population import (
    aggregate_results,
    freeze_population,
    summarize_trace,
)


def _fixture(fixture_id: str, battle_tag: str, hp: int = 100) -> DecisionFixture:
    return DecisionFixture(
        fixture_id=fixture_id,
        state={
            "turn": 8,
            "player": "Azelficoast",
            "opponent": "Rival",
            "active": {
                "species": "Tinkaton",
                "current_hp": hp,
                "max_hp": 100,
                "hp_fraction": hp / 100,
                "tera_type": "Steel",
            },
            "opponent_active": {
                "species": "Zapdos-Galar",
                "level": 77,
                "current_hp": 45,
                "max_hp": 100,
                "hp_fraction": 0.45,
                "item": None,
            },
            "team": {"p1: Tinkaton": {"species": "Tinkaton"}},
            "legal_actions": [
                "/choose move protect",
                "/choose move gigatonhammer",
                "/choose switch Lapras",
            ],
        },
        protocol_prefix=(
            (
                ("", "player", "p1", "Azelficoast"),
                ("", "player", "p2", "Rival"),
                ("", "teamsize", "p2", "6"),
                ("", "switch", "p2a: Lapras", "Lapras, L80", "100/100"),
                ("", "switch", "p2a: Zapdos-Galar", "Zapdos-Galar, L77", "100/100"),
                ("", "move", "p2a: Zapdos-Galar", "U-turn", "p1a: Tinkaton"),
            ),
        ),
        control_decisions=(
            {
                "battle_tag": battle_tag,
                "chosen_action": "/choose move protect",
            },
        ),
    )


def _candidate(fixture_id: str) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "active_species": "Tinkaton",
        "opponent_species": "Zapdos-Galar",
        "revealed_moves": ["uturn"],
        "locked_move": "U-turn",
        "is_lead": False,
        "item_counts": {"Choice Band": 50, "Choice Scarf": 50},
        "item_weights": {"Choice Band": 0.5, "Choice Scarf": 0.5},
        "persistent_protect_actions": [{"action": "/choose move protect"}],
        "persistent_switches": [],
        "sample_rounds": 2048,
    }


def _mechanics(fixture_id: str) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "by_item": {
            "Choice Band": {
                "worlds": [
                    {
                        "count": 50,
                        "incoming": {
                            "own_speed": 200,
                            "opponent_speed": 180,
                            "damage_min": 40,
                            "damage_max": 50,
                            "ko_rolls": 0,
                        },
                    }
                ]
            },
            "Choice Scarf": {
                "worlds": [
                    {
                        "count": 50,
                        "incoming": {
                            "own_speed": 200,
                            "opponent_speed": 260,
                            "damage_min": 70,
                            "damage_max": 90,
                            "ko_rolls": 8,
                        },
                    }
                ]
            },
        },
    }


def _plan(source_count: int, max_exact: int = 128) -> dict[str, object]:
    return {
        "schema": "azelficoast.natural-population-strategy-fusion-plan",
        "schema_version": 1,
        "source_artifact": {
            "decision_state_count": source_count,
            "battle_count": source_count,
        },
        "showdown_commit": "pinned",
        "admissibility": {
            "generator_rounds": 2048,
            "mechanics_screen_rounds": 512,
            "max_exact_states": max_exact,
            "overflow_selection": "hash",
        },
        "exact_treatment": {
            "positive_bias_tolerance": 1e-9,
        },
        "inference": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1729,
        },
    }


def test_freeze_population_uses_outcome_blind_hash_sampling(tmp_path) -> None:
    fixtures = [_fixture("fixture-a", "battle-a"), _fixture("fixture-b", "battle-b")]
    candidates = {
        "schema": "azelficoast.natural-fusion-candidates",
        "persistent_only": True,
        "candidates": [_candidate("fixture-a"), _candidate("fixture-b")],
        "excluded_fixtures": [],
    }
    mechanics = {
        "schema": "azelficoast.public-belief-speed-fork-mechanics",
        "showdown_commit": "pinned",
        "rounds": 512,
        "cases": [_mechanics("fixture-a"), _mechanics("fixture-b")],
    }

    result = freeze_population(
        plan=_plan(2, max_exact=1),
        candidates_document=candidates,
        mechanics_document=mechanics,
        fixtures=fixtures,
        output_dir=tmp_path,
    )

    assert result["selection_uses_policy_result"] is False
    assert result["eligible_count"] == 2
    assert result["selected_count"] == 1
    assert result["overflow_sampling_applied"] is True
    selected = result["selected"][0]
    assert selected["selection_key"]
    source = (tmp_path / selected["filename"]).read_text()
    assert '"selection_uses_policy_result": false' in source


def test_summarize_trace_measures_bias_and_corrected_regret() -> None:
    source = {
        "fixture_id": "fixture",
        "showdown_commit": "pinned",
        "population_selection": {
            "population_index": 1,
            "battle_tag": "battle",
            "selection_key": "abc",
            "selection_uses_policy_result": False,
            "predictors": {"turn": 8},
        },
    }
    trace = {
        "experiment_valid": True,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "world_count": 10,
        "legal_action_count": 2,
        "strategy_fusion_observation_count": 1,
        "policy_disagreement": True,
        "determinization": {
            "chosen_action": "a",
            "root_values": {"a": 0.7, "b": 0.6},
        },
        "public_belief": {
            "chosen_action": "b",
            "root_values": {"a": 0.5, "b": 0.55},
        },
    }

    result = summarize_trace(source=source, trace=trace)

    assert abs(result["max_strategy_fusion_value_advantage"] - 0.2) < 1e-12
    assert abs(result["determinization_public_regret"] - 0.05) < 1e-12
    assert result["policy_disagreement"] is True


def _result(
    index: int,
    fixture_id: str,
    battle_tag: str,
    *,
    bias: float,
    regret: float,
    disagreement: bool,
) -> dict[str, object]:
    return {
        "population_index": index,
        "fixture_id": fixture_id,
        "battle_tag": battle_tag,
        "max_strategy_fusion_value_advantage": bias,
        "determinization_public_regret": regret,
        "policy_disagreement": disagreement,
        "strategy_fusion_observation_count": int(bias > 0),
        "predictors": {
            "turn": 8 + index,
            "own_active_hp_fraction": 0.5,
            "opponent_public_hp_fraction": 0.4,
            "hidden_item_entropy_bits": 1.0,
            "incoming_ko_roll_probability_gap": 0.2 * index,
            "incoming_damage_fraction_gap": 0.1 * index,
            "minimum_hidden_item_mass": 0.4,
            "persistent_branch_count": 1,
            "legal_action_count": 3,
            "relative_move_order_changes": index == 1,
            "persistent_protect_present": True,
            "persistent_switch_present": False,
        },
    }


def test_aggregate_results_clusters_by_battle_and_retains_zeroes() -> None:
    rows = [
        _result(
            1,
            "a",
            "battle-one",
            bias=0.02,
            regret=0.01,
            disagreement=True,
        ),
        _result(
            2,
            "b",
            "battle-one",
            bias=0.0,
            regret=0.0,
            disagreement=False,
        ),
        _result(
            3,
            "c",
            "battle-two",
            bias=0.01,
            regret=0.0,
            disagreement=False,
        ),
    ]
    manifest = {
        "schema": "azelficoast.natural-population-strategy-fusion-cohort",
        "selected_count": 3,
        "source_decision_state_count": 3,
        "bounded_candidate_count": 3,
        "eligible_count": 3,
        "overflow_sampling_applied": False,
        "ineligible_reason_counts": {},
        "selected": [
            {"fixture_id": "a"},
            {"fixture_id": "b"},
            {"fixture_id": "c"},
        ],
    }

    result = aggregate_results(
        plan=_plan(3),
        manifest=manifest,
        rows=rows,
    )

    assert result["cohort"]["exact_battle_count"] == 2
    assert result["outcomes"]["positive_bias_count"] == 2
    assert result["outcomes"]["policy_disagreement_count"] == 1
    assert result["cluster_bootstrap"]["cluster_count"] == 2
    assert result["cluster_bootstrap"]["replicates"] == 20

    associations = result["predictor_associations"]
    assert set(associations) == {
        "max_strategy_fusion_value_advantage",
        "determinization_public_regret",
    }
    assert (
        associations["max_strategy_fusion_value_advantage"]["numeric_spearman"][
            "turn"
        ]["rho"]
        is not None
    )
    assert (
        associations["determinization_public_regret"]["numeric_spearman"]["turn"][
            "rho"
        ]
        is not None
    )
    regret_binary = associations["determinization_public_regret"][
        "binary_mean_contrast"
    ]["relative_move_order_changes"]
    assert regret_binary["true_n"] == 1
    assert regret_binary["false_n"] == 2
    assert regret_binary["true_mean"] == 0.01
    assert regret_binary["false_mean"] == 0.0
