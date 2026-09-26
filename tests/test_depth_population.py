from __future__ import annotations

from azelficoast.research.studies.depth_population import (
    aggregate_depth_results,
    select_depth_rows,
)


def _plan(max_exact: int = 3) -> dict[str, object]:
    return {
        "schema": "azelficoast.natural-depth-regret-plan",
        "schema_version": 1,
        "predictors_frozen_from_population_study": {
            "numeric": [
                {"name": "legal_action_count"},
                {"name": "incoming_ko_roll_probability_gap"},
                {"name": "opponent_public_hp_fraction"},
                {"name": "hidden_item_entropy_bits"},
            ],
            "binary": [
                {"name": "persistent_protect_present"},
                {"name": "persistent_switch_present"},
            ],
        },
        "admissibility": {"max_exact_states": max_exact},
        "inference": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 2718,
        },
    }


def _eligible(
    fixture_id: str,
    *,
    legal_actions: int,
    ko_gap: float,
    hp: float,
    entropy: float,
    protect: bool = False,
    switch: bool = False,
) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "battle_tag": f"battle-{fixture_id}",
        "filename": f"{fixture_id}.json",
        "predictors": {
            "legal_action_count": legal_actions,
            "incoming_ko_roll_probability_gap": ko_gap,
            "opponent_public_hp_fraction": hp,
            "hidden_item_entropy_bits": entropy,
            "persistent_protect_present": protect,
            "persistent_switch_present": switch,
        },
    }


def test_depth_selection_is_outcome_blind_and_preserves_lower_stratum() -> None:
    rows = [
        _eligible("a", legal_actions=12, ko_gap=1.0, hp=0.8, entropy=1.0),
        _eligible("b", legal_actions=10, ko_gap=0.8, hp=0.7, entropy=0.9, protect=True),
        _eligible("c", legal_actions=8, ko_gap=0.5, hp=0.6, entropy=0.8),
        _eligible("d", legal_actions=4, ko_gap=0.0, hp=0.2, entropy=0.2),
        _eligible("e", legal_actions=3, ko_gap=0.0, hp=0.1, entropy=0.1),
        _eligible("f", legal_actions=2, ko_gap=0.0, hp=0.05, entropy=0.05),
    ]

    selected, thresholds = select_depth_rows(rows, plan=_plan(max_exact=3))

    assert len(selected) == 3
    assert sum(row["depth_stratum"] == "predictor-enriched" for row in selected) == 2
    assert sum(row["depth_stratum"] == "representative-lower" for row in selected) == 1
    assert thresholds["legal_action_count"] > 4
    assert all("depth_selection_key" in row for row in selected)


def _result(
    index: int,
    fixture_id: str,
    battle_tag: str,
    stratum: str,
    *,
    deep_bias: float,
    deep_regret: float,
    bias_change: float,
    regret_change: float,
) -> dict[str, object]:
    return {
        "depth_index": index,
        "fixture_id": fixture_id,
        "battle_tag": battle_tag,
        "stratum": stratum,
        "shallow_value_bias": deep_bias - bias_change,
        "deep_value_bias": deep_bias,
        "value_bias_change": bias_change,
        "shallow_regret": deep_regret - regret_change,
        "deep_regret": deep_regret,
        "regret_change": regret_change,
        "deep_policy_disagreement": deep_regret > 0,
        "predictors": {
            "legal_action_count": 5 + index,
            "incoming_ko_roll_probability_gap": 0.1 * index,
            "opponent_public_hp_fraction": 0.2 * index,
            "hidden_item_entropy_bits": 0.5 + 0.1 * index,
            "persistent_protect_present": index == 1,
            "persistent_switch_present": index == 2,
        },
    }


def test_depth_aggregate_keeps_continuous_zeroes_and_paired_changes() -> None:
    rows = [
        _result(
            1,
            "a",
            "battle-one",
            "predictor-enriched",
            deep_bias=0.06,
            deep_regret=0.02,
            bias_change=0.04,
            regret_change=0.02,
        ),
        _result(
            2,
            "b",
            "battle-one",
            "predictor-enriched",
            deep_bias=0.02,
            deep_regret=0.0,
            bias_change=0.01,
            regret_change=0.0,
        ),
        _result(
            3,
            "c",
            "battle-two",
            "representative-lower",
            deep_bias=0.0,
            deep_regret=0.0,
            bias_change=0.0,
            regret_change=0.0,
        ),
    ]
    manifest = {
        "schema": "azelficoast.natural-depth-regret-cohort",
        "source_artifact": {"decision_state_count": 3},
        "source_decision_state_count": 3,
        "bounded_candidate_count": 3,
        "eligible_count": 3,
        "selected_count": 3,
        "stratum_counts": {
            "predictor-enriched": 2,
            "representative-lower": 1,
        },
        "selected": [
            {"fixture_id": "a"},
            {"fixture_id": "b"},
            {"fixture_id": "c"},
        ],
    }

    result = aggregate_depth_results(
        plan=_plan(),
        manifest=manifest,
        rows=rows,
    )

    assert result["outcomes"]["deep_regret"]["median"] == 0.0
    assert result["outcomes"]["deep_positive_regret_count"] == 1
    assert result["outcomes"]["value_bias_change"]["mean"] > 0
    assert result["cohort"]["exact_battle_count"] == 2
    assert result["by_stratum"]["representative-lower"]["deep_regret_mean"] == 0.0
    assert result["cluster_bootstrap"]["replicates"] == 20
    assert set(result["predictor_associations"]) == {
        "deep_value_bias",
        "deep_regret",
        "value_bias_change",
        "regret_change",
    }


def test_invariant_numeric_predictors_do_not_make_every_state_enriched() -> None:
    rows = [
        _eligible(
            "a",
            legal_actions=12,
            ko_gap=0.0,
            hp=0.1,
            entropy=0.7,
            switch=True,
        ),
        _eligible(
            "b",
            legal_actions=12,
            ko_gap=0.0,
            hp=0.2,
            entropy=0.7,
            switch=True,
        ),
        _eligible(
            "c",
            legal_actions=12,
            ko_gap=0.0,
            hp=0.9,
            entropy=1.0,
            switch=True,
        ),
    ]

    selected, thresholds = select_depth_rows(rows, plan=_plan(max_exact=3))

    assert thresholds["legal_action_count"] is None
    assert thresholds["incoming_ko_roll_probability_gap"] is None
    assert sum(row["depth_stratum"] == "predictor-enriched" for row in selected) == 1
    assert sum(row["depth_stratum"] == "representative-lower" for row in selected) == 2
