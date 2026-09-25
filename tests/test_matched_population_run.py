from __future__ import annotations

from azelficoast.matched_population_run import matched_cohort, run_state


def _plan() -> dict[str, object]:
    return {
        "schema": "azelficoast.matched-search-comparison-plan",
        "schema_version": 1,
        "posterior_treatments": ["generator_faithful", "practical"],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": 4,
        },
        "opponent_model": "fixed_observed_response",
        "depths": [1],
        "confirmatory_predictors": [
            "legal_action_count",
            "incoming_ko_roll_probability_gap",
            "opponent_public_hp_fraction",
        ],
        "cluster_unit": "battle_tag",
        "showdown_commit": "pinned",
        "inference": {
            "bootstrap_replicates": 20,
            "bootstrap_seed": 1729,
        },
    }


def _source() -> dict[str, object]:
    return {
        "fixture_id": "fixture",
        "fixture": {
            "fixture_id": "fixture",
            "state": {
                "turn": 8,
                "legal_actions": ["wait", "reveal"],
                "active": {"species": "Tinkaton"},
                "opponent_active": {"species": "Zapdos-Galar"},
            },
            "protocol_prefix": [],
            "control_decisions": [],
        },
    }


def _manifest() -> dict[str, object]:
    return {
        "schema": "azelficoast.natural-population-strategy-fusion-cohort",
        "schema_version": 1,
        "source_decision_state_count": 6485,
        "frozen_before_policy_values": True,
        "selection_uses_policy_result": False,
        "selected": [
            {
                "population_index": 1,
                "fixture_id": "fixture",
                "battle_tag": "battle",
                "predictors": {
                    "legal_action_count": 2,
                    "incoming_ko_roll_probability_gap": 0.5,
                    "opponent_public_hp_fraction": 0.53,
                },
            }
        ],
    }


def _oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": "band",
            "weight": 0.75,
            "hidden": {
                "opponent.active.item": "Choice Band",
                "opponent.active.ability": "A",
                "opponent.active.exact_hp": 100,
            },
            "provenance": {
                "generator_rounds": 2048,
                "generator_count": 1536,
            },
        },
        {
            "world_id": "scarf",
            "weight": 0.25,
            "hidden": {
                "opponent.active.item": "Choice Scarf",
                "opponent.active.ability": "B",
                "opponent.active.exact_hp": 90,
            },
            "provenance": {
                "generator_rounds": 2048,
                "generator_count": 512,
            },
        },
    ]
    transitions = []
    for world in worlds:
        item = world["hidden"]["opponent.active.item"]
        for action in ("wait", "reveal"):
            if action == "wait":
                observation = {"kind": "same"}
                continuations = {
                    "fast": 4.0 if item == "Choice Scarf" else -4.0,
                    "safe": 1.0,
                }
            else:
                observation = {"kind": item}
                continuations = {
                    "fast": 3.0 if item == "Choice Scarf" else -3.0,
                    "safe": 0.0,
                }
            transitions.append(
                {
                    "world_id": world["world_id"],
                    "action": action,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": observation,
                            "continuations": continuations,
                        }
                    ],
                }
            )
    return {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "fixture",
        "showdown_commit": "pinned",
        "reconstruction": {
            "generator_rounds": 2048,
            "generator_matches": 2048,
        },
        "worlds": worlds,
        "legal_actions": ["wait", "reveal"],
        "transitions": transitions,
    }


def test_run_state_executes_each_posterior_with_paired_equal_receipts() -> None:
    results = run_state(
        plan=_plan(),
        source=_source(),
        manifest=_manifest(),
        population_index=1,
        oracle=_oracle(),
    )

    assert [row["posterior_treatment"] for row in results] == [
        "generator_faithful",
        "practical",
    ]
    for row in results:
        assert row["matched_input"] is True
        assert row["matched_authorized_compute"] is True
        assert row["compute_consumed"] == {
            "determinization": 4,
            "information_set": 4,
        }
        assert row["transition_oracle_digest"]
        assert row["population_index"] == 1
        assert row["predictors"] == {
            "legal_action_count": 2,
            "incoming_ko_roll_probability_gap": 0.5,
            "opponent_public_hp_fraction": 0.53,
        }


def test_matched_cohort_preserves_outcome_blind_selection_identity() -> None:
    cohort = matched_cohort(_manifest())

    assert cohort["frozen_before_policy_values"] is True
    assert cohort["selection_uses_policy_result"] is False
    assert cohort["source_decision_state_count"] == 6485
    assert cohort["selected"] == [
        {
            "fixture_id": "fixture",
            "battle_tag": "battle",
            "population_index": 1,
        }
    ]
