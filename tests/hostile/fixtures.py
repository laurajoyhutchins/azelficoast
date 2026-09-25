from __future__ import annotations

from typing import Any

import pytest


WEIGHT_NORMALIZATION_ABS_TOLERANCE = 1e-15
EVALUATOR_OUTPUT_ABS_TOLERANCE = 1e-6
SEARCH_VALUE_ABS_TOLERANCE = 1e-12
PROBABILITY_CONSERVATION_ABS_TOLERANCE = 1e-12


def hostile_case(
    *,
    mutation: str,
    expected: str,
    threat: str,
    layer: str,
    tier: str = "fast",
):
    """Attach a machine-checkable statement of what each hostile case protects."""
    if tier not in {"fast", "simulator", "showdown"}:
        raise ValueError(f"unknown hostile test tier {tier!r}")

    def decorate(test):
        contract = (
            f"Mutation: {mutation}\n"
            f"Expected invariant: {expected}\n"
            f"Threat: {threat}\n"
            f"Layer: {layer}"
        )
        test.__doc__ = "\n".join(value for value in (test.__doc__, contract) if value)
        test = getattr(pytest.mark, f"hostile_{tier}")(test)
        return pytest.mark.hostile_contract(
            mutation=mutation,
            expected=expected,
            threat=threat,
            layer=layer,
        )(test)

    return decorate


def tiny_public_state() -> dict[str, Any]:
    return {
        "turn": 8,
        "active": {"species": "rotomwash", "hp_fraction": 0.61},
        "tera_unused": True,
    }


def tiny_posterior() -> dict[str, Any]:
    return {
        "treatment": "oracle",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {
                "world_id": "opaque-alpha",
                "weight": 0.25,
                "hidden": {
                    "item": "choiceband",
                    "moves": ["protect", "voltswitch"],
                    "tera_type": "water",
                },
            },
            {
                "world_id": "opaque-beta",
                "weight": 0.75,
                "hidden": {
                    "item": "leftovers",
                    "moves": ["protect", "willowisp"],
                    "tera_type": "water",
                },
            },
        ],
    }


def analytic_action_values() -> dict[str, Any]:
    """Hand-solvable world/action table; A=.25 and B=.75 under the stated mass."""
    return {
        "world_weights": {"w0": 0.25, "w1": 0.75},
        "values": {"w0": {"A": 1.0, "B": 0.0}, "w1": {"A": 0.0, "B": 1.0}},
        "expected": {"A": 0.25, "B": 0.75},
    }


def matched_plan() -> dict[str, Any]:
    return {
        "schema": "azelficoast.matched-search-comparison-plan",
        "schema_version": 2,
        "posterior_treatments": ["generator_faithful", "practical"],
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": 64,
        },
        "opponent_model": "fixed_observed_response",
        "depths": [1, 2],
        "confirmatory_predictors": ["hidden_item_entropy_bits", "persistent_branch_count"],
        "cluster_unit": "battle_tag",
        "showdown_commit": "pinned",
        "chance_treatment": "shared_frozen_transition_oracle",
        "evaluator": {
            "schema": "azelficoast.belief-policy-value-evaluator",
            "schema_version": 1,
            "checkpoint_digest": "sha256:" + "a" * 64,
            "observability": "public_belief_only",
            "architecture": "weighted_deep_sets_policy_value",
            "spec": {"hidden_width": 32},
        },
        "inference": {"bootstrap_replicates": 30, "bootstrap_seed": 1729},
    }


def matched_state(*, legal_actions: list[str] | None = None) -> dict[str, Any]:
    return {
        "fixture_id": "hostile-fixture-1",
        "battle_tag": "hostile-battle-1",
        "public_state": {"turn": 8, "active": "rotomwash", "hp_fraction": 0.61},
        "legal_actions": list(legal_actions or ["move:a", "move:b"]),
        "predictors": {"hidden_item_entropy_bits": 1.0, "persistent_branch_count": 2},
    }


def matched_posterior() -> dict[str, Any]:
    return {
        "treatment": "generator_faithful",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": [
            {"world_id": "transport:band", "weight": 0.25, "hidden": {"item": "band"}},
            {
                "world_id": "transport:leftovers",
                "weight": 0.75,
                "hidden": {"item": "leftovers"},
            },
        ],
    }
