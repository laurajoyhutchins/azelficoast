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


def _bindings(*, treatments: list[str] | None = None) -> dict[str, object]:
    selected = [
        {
            "fixture_id": f"fixture-{index:03d}",
            "battle_tag": f"battle-{index // 2:03d}",
        }
        for index in range(100)
    ]
    contract = _contract()
    authority = contract["oracle_authority"]
    assert isinstance(authority, dict)
    evaluator = contract["evaluator"]
    assert isinstance(evaluator, dict)
    compute = contract["compute"]
    assert isinstance(compute, dict)
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
        "evaluator": evaluator,
        "compute_budget_per_method": compute["per_method_limit"],
        "oracle_authority": {
            "kind": authority["kind"],
            "evidence_digest": "sha256:" + "d" * 64,
            "realized_hidden_state_used": False,
        },
    }
    if treatments is not None:
        result["enabled_treatments"] = treatments
    return result


def test_issue_69_contract_is_complete_and_repository_owned() -> None:
    checked = validate_contract(_contract())
    readiness = contract_readiness(checked)

    assert readiness["passed"] is True
    assert readiness["default_enabled_treatments"] == [
        "oracle",
        "generator_faithful",
        "practical",
    ]
    assert readiness["completion_posterior_treatments"] == [
        "oracle",
        "generator_faithful",
        "practical",
    ]
    assert readiness["execution_depths"] == [1, 2]
    assert readiness["required_depth_curve"] == [1, 2]
    assert readiness["oracle_authority_kind"] == "best_available_conditional"
    assert readiness["source_artifact_id"] == 10866900218


def test_compile_population_plan_materializes_full_three_by_two_matrix() -> None:
    execution = compile_execution_plan(_contract(), _bindings())

    assert execution["selected_state_count"] == 100
    assert execution["selected_battle_count"] == 50
    assert execution["expected_result_count"] == 600
    assert execution["issue_69_complete_treatment_set"] is True
    plan = execution["matched_plan"]
    assert isinstance(plan, dict)
    assert plan["posterior_treatments"] == [
        "oracle",
        "generator_faithful",
        "practical",
    ]
    assert plan["depths"] == [1, 2]
    assert plan["chance_treatment"] == "shared_frozen_horizon_complete_oracle"


def test_oracle_stratum_requires_frozen_conditional_authority() -> None:
    bindings = _bindings()
    bindings.pop("oracle_authority")
    with pytest.raises(
        PosteriorPopulationContractError,
        match="lacks conditional authority",
    ):
        compile_execution_plan(_contract(), bindings)

    bindings = _bindings()
    authority = bindings["oracle_authority"]
    assert isinstance(authority, dict)
    authority["kind"] = "exact_conditional"
    with pytest.raises(
        PosteriorPopulationContractError,
        match="differs from the frozen contract",
    ):
        compile_execution_plan(_contract(), bindings)


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
    selected = ledger["selected"]
    assert isinstance(selected, list)
    ledger["selected"] = selected[:99]
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
