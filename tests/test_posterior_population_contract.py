from __future__ import annotations

import json
from pathlib import Path

import pytest

from azelficoast.research.matched_population_run import matched_cohort
from azelficoast.research.posterior_population_contract import (
    PosteriorPopulationContractError,
    compile_execution_plan,
    contract_readiness,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "experiments" / "posterior-stratified-population-contract.json"


def _contract() -> dict[str, object]:
    document = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _evaluator() -> dict[str, object]:
    return {
        "schema": "azelficoast.belief-policy-value-evaluator",
        "schema_version": 1,
        "checkpoint_digest": "sha256:" + "a" * 64,
        "observability": "public_belief_only",
        "architecture": "contract-test",
        "spec": {"kind": "contract-test"},
    }


def _bindings(*, treatments: list[str] | None = None) -> dict[str, object]:
    selected = [
        {
            "fixture_id": f"fixture-{index:03d}",
            "battle_tag": f"battle-{index // 2:03d}",
        }
        for index in range(100)
    ]
    result: dict[str, object] = {
        "source_corpus": {
            "digest": "sha256:" + "b" * 64,
            "state_count": 500,
            "frozen_before_treatments": True,
        },
        "inclusion_ledger": {
            "digest": "sha256:" + "c" * 64,
            "frozen_before_treatments": True,
            "selection_uses_treatment_outcomes": False,
            "selected": selected,
        },
        "evaluator": _evaluator(),
        "compute_budget_per_method": 4096,
    }
    if treatments is not None:
        result["enabled_treatments"] = treatments
    return result


def test_issue_69_contract_is_repository_owned_and_preserves_unfinished_boundaries() -> None:
    checked = validate_contract(_contract())
    readiness = contract_readiness(checked)

    assert readiness["passed"] is True
    assert readiness["default_enabled_treatments"] == [
        "generator_faithful",
        "practical",
    ]
    assert readiness["completion_posterior_treatments"] == [
        "oracle",
        "generator_faithful",
        "practical",
    ]
    assert readiness["execution_depths"] == [1]
    assert readiness["required_depth_curve"] == [1, 2]
    assert readiness["oracle_authority_required"] is True


def test_compile_population_plan_materializes_complete_default_matrix() -> None:
    execution = compile_execution_plan(_contract(), _bindings())

    assert execution["selected_state_count"] == 100
    assert execution["selected_battle_count"] == 50
    assert execution["expected_result_count"] == 200
    assert execution["issue_69_complete_treatment_set"] is False
    plan = execution["matched_plan"]
    assert isinstance(plan, dict)
    assert plan["posterior_treatments"] == ["generator_faithful", "practical"]
    assert plan["depths"] == [1]


def test_oracle_stratum_requires_exact_conditional_authority() -> None:
    bindings = _bindings(
        treatments=["oracle", "generator_faithful", "practical"]
    )
    with pytest.raises(
        PosteriorPopulationContractError,
        match="oracle stratum lacks exact authority",
    ):
        compile_execution_plan(_contract(), bindings)

    bindings["oracle_authority"] = {
        "kind": "exact_conditional",
        "evidence_digest": "sha256:" + "d" * 64,
        "realized_hidden_state_used": False,
    }
    execution = compile_execution_plan(_contract(), bindings)
    assert execution["expected_result_count"] == 300
    assert execution["issue_69_complete_treatment_set"] is True


def test_population_plan_rejects_outcome_selected_or_too_small_ledgers() -> None:
    outcome_selected = _bindings()
    ledger = outcome_selected["inclusion_ledger"]
    assert isinstance(ledger, dict)
    ledger["selection_uses_treatment_outcomes"] = True
    with pytest.raises(PosteriorPopulationContractError, match="used treatment outcomes"):
        compile_execution_plan(_contract(), outcome_selected)

    too_small = _bindings()
    ledger = too_small["inclusion_ledger"]
    assert isinstance(ledger, dict)
    ledger["selected"] = ledger["selected"][:99]
    with pytest.raises(PosteriorPopulationContractError, match="requires 100"):
        compile_execution_plan(_contract(), too_small)


def test_frozen_ledger_projects_to_existing_population_settlement_schema() -> None:
    manifest = {
        "schema": "azelficoast.test-ledger",
        "schema_version": 1,
        "source_decision_state_count": 2,
        "frozen_before_policy_values": True,
        "selection_uses_policy_result": False,
        "selected": [
            {
                "fixture_id": "a",
                "battle_tag": "battle-1",
                "population_index": 1,
            },
            {
                "fixture_id": "b",
                "battle_tag": "battle-1",
                "population_index": 2,
            },
        ],
    }
    cohort = matched_cohort(manifest)
    assert cohort["schema"] == "azelficoast.matched-search-population-cohort"
    assert cohort["selected"] == [
        {"fixture_id": "a", "battle_tag": "battle-1", "population_index": 1},
        {"fixture_id": "b", "battle_tag": "battle-1", "population_index": 2},
    ]
