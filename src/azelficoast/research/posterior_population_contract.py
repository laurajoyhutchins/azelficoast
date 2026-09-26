"""Compile the posterior-stratified matched-population research contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from azelficoast.research.contracts import stable_digest
from azelficoast.research.matched_comparison import (
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    validate_plan,
)

SCHEMA = "azelficoast.posterior-stratified-population-contract"
SCHEMA_VERSION = 1
EXECUTION_SCHEMA = "azelficoast.posterior-stratified-population-execution-plan"
EXECUTION_SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHOWDOWN_REVISION_PATH = REPOSITORY_ROOT / "experiments" / "showdown-revision.txt"


class PosteriorPopulationContractError(ValueError):
    """Raised when issue #69 research authority is incomplete or drifts."""


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise PosteriorPopulationContractError(
            f"{field} must be sha256:<64 lowercase hex>"
        )
    return value


def _showdown_revision() -> str:
    revision = SHOWDOWN_REVISION_PATH.read_text(encoding="utf-8").strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise PosteriorPopulationContractError("repository Showdown revision is malformed")
    return revision


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the repository-owned scientific contract, without inventing bindings."""
    if contract.get("schema") != SCHEMA or contract.get("schema_version") != SCHEMA_VERSION:
        raise PosteriorPopulationContractError("unexpected posterior-population contract schema")
    if contract.get("issue") != 69:
        raise PosteriorPopulationContractError("posterior-population contract must bind issue #69")
    if contract.get("showdown_revision_source") != "experiments/showdown-revision.txt":
        raise PosteriorPopulationContractError("Showdown authority must come from repository contract")

    treatments = contract.get("completion_posterior_treatments")
    if treatments != ["oracle", "generator_faithful", "practical"]:
        raise PosteriorPopulationContractError("issue #69 requires the three frozen posterior strata")
    enabled = contract.get("default_enabled_treatments")
    if enabled != ["oracle", "generator_faithful", "practical"]:
        raise PosteriorPopulationContractError(
            "issue #69 completion requires all three posterior strata"
        )
    if contract.get("search_methods") != ["determinization", "information_set"]:
        raise PosteriorPopulationContractError("matched search methods drifted")
    if contract.get("execution_depths") != [1, 2]:
        raise PosteriorPopulationContractError("issue #69 execution must cover depths 1 and 2")
    if contract.get("required_depth_curve") != [1, 2]:
        raise PosteriorPopulationContractError("issue #69 depth-curve completion contract drifted")

    compute = contract.get("compute")
    if not isinstance(compute, Mapping) or compute.get("unit") != "transition_evaluations":
        raise PosteriorPopulationContractError("matched compute must use transition evaluations")
    if compute.get("same_authorized_limit_per_method") is not True:
        raise PosteriorPopulationContractError("paired methods must receive equal authorized compute")
    if contract.get("opponent_model") != "fixed_observed_response":
        raise PosteriorPopulationContractError("opponent-model boundary drifted")

    predictors = contract.get("confirmatory_predictors")
    if not isinstance(predictors, list) or len(predictors) != 3 or len(set(predictors)) != 3:
        raise PosteriorPopulationContractError("contract must freeze three unique predictors")

    population = contract.get("population")
    if not isinstance(population, Mapping):
        raise PosteriorPopulationContractError("population contract is missing")
    if int(population.get("minimum_selected_states", 0)) < 100:
        raise PosteriorPopulationContractError("population must contain at least hundreds of states")
    if population.get("source_corpus_frozen_before_treatments") is not True:
        raise PosteriorPopulationContractError("source corpus must freeze before treatments")
    if population.get("inclusion_ledger_frozen_before_treatments") is not True:
        raise PosteriorPopulationContractError("inclusion ledger must freeze before treatments")
    source_artifact = population.get("source_artifact")
    if not isinstance(source_artifact, Mapping):
        raise PosteriorPopulationContractError("population source artifact is missing")
    for field in (
        "artifact_id",
        "workflow_run_id",
        "battle_count",
        "decision_state_count",
    ):
        value = source_artifact.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise PosteriorPopulationContractError(
                f"population source artifact lacks positive {field}"
            )
    for field in ("artifact_digest", "decisions_sha256", "corpus_sha256"):
        value = source_artifact.get(field)
        if field == "artifact_digest":
            _digest(value, field="source artifact digest")
        elif (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise PosteriorPopulationContractError(
                f"population source artifact lacks valid {field}"
            )
    if population.get("persistent_only") is not False:
        raise PosteriorPopulationContractError(
            "issue #69 must retain non-persistent zero-effect controls"
        )
    maximum = population.get("max_selected_states")
    minimum = population.get("minimum_selected_states")
    if (
        not isinstance(maximum, int)
        or isinstance(maximum, bool)
        or maximum < int(minimum)
    ):
        raise PosteriorPopulationContractError(
            "max selected states must cover the preregistered minimum"
        )
    forbidden = population.get("forbidden_selection_fields")
    if not isinstance(forbidden, list) or not {
        "policy_disagreement", "value_bias", "regret", "battle_outcome"
    }.issubset(set(map(str, forbidden))):
        raise PosteriorPopulationContractError("selection-outcome firewall is incomplete")

    inference = contract.get("inference")
    if not isinstance(inference, Mapping) or inference.get("cluster_unit") != "battle_tag":
        raise PosteriorPopulationContractError("uncertainty must cluster by battle")
    replicates = inference.get("bootstrap_replicates")
    seed = inference.get("bootstrap_seed")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise PosteriorPopulationContractError("bootstrap replicates must be positive")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PosteriorPopulationContractError("bootstrap seed must be integral")

    oracle = contract.get("oracle_authority")
    if not isinstance(oracle, Mapping):
        raise PosteriorPopulationContractError("oracle posterior authority is missing")
    if (
        oracle.get("kind") not in {"exact_conditional", "best_available_conditional"}
        or oracle.get("evidence_required") is not True
        or oracle.get("realized_hidden_state_used") is not False
    ):
        raise PosteriorPopulationContractError(
            "oracle treatment must be explicitly conditional and leakage-free"
        )
    if oracle.get("kind") == "best_available_conditional":
        dense = oracle.get("oracle_generator_rounds")
        reference = oracle.get("generator_faithful_rounds")
        if (
            not isinstance(dense, int)
            or isinstance(dense, bool)
            or not isinstance(reference, int)
            or isinstance(reference, bool)
            or dense <= reference
            or reference < 1
        ):
            raise PosteriorPopulationContractError(
                "best-available conditional authority requires a denser oracle sweep"
            )
        if oracle.get("support_must_match_generator_faithful") is not True:
            raise PosteriorPopulationContractError(
                "conditional strata must share hidden support"
            )
        schedule = oracle.get("generator_seed_schedule")
        if not isinstance(schedule, str) or not schedule:
            raise PosteriorPopulationContractError("oracle authority lacks seed schedule")
        if oracle.get("generator_seed_start") != 0:
            raise PosteriorPopulationContractError("oracle seed sweep must start at zero")

    return dict(contract)


def compile_execution_plan(
    contract: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind frozen corpus/evaluator authority into existing matched-search machinery."""
    checked = validate_contract(contract)
    completion = tuple(str(value) for value in checked["completion_posterior_treatments"])
    raw_enabled = bindings.get("enabled_treatments", checked["default_enabled_treatments"])
    if not isinstance(raw_enabled, list) or not raw_enabled:
        raise PosteriorPopulationContractError("enabled treatments must be a non-empty list")
    enabled = tuple(str(value) for value in raw_enabled)
    if tuple(value for value in completion if value in enabled) != enabled:
        raise PosteriorPopulationContractError("enabled treatments must preserve contract order")
    if not {"generator_faithful", "practical"}.issubset(enabled):
        raise PosteriorPopulationContractError(
            "population execution must retain generator-faithful and practical strata"
        )

    source = bindings.get("source_corpus")
    ledger = bindings.get("inclusion_ledger")
    evaluator = bindings.get("evaluator")
    if not isinstance(source, Mapping) or not isinstance(ledger, Mapping):
        raise PosteriorPopulationContractError("frozen corpus and inclusion ledger are required")
    if not isinstance(evaluator, Mapping):
        raise PosteriorPopulationContractError("one frozen evaluator identity is required")
    source_digest = _digest(source.get("digest"), field="source corpus digest")
    ledger_digest = _digest(ledger.get("digest"), field="inclusion ledger digest")
    if source.get("frozen_before_treatments") is not True:
        raise PosteriorPopulationContractError("source corpus is not frozen before treatments")
    if ledger.get("frozen_before_treatments") is not True:
        raise PosteriorPopulationContractError("inclusion ledger is not frozen before treatments")
    if ledger.get("selection_uses_treatment_outcomes") is not False:
        raise PosteriorPopulationContractError("inclusion ledger used treatment outcomes")

    selected = ledger.get("selected")
    if not isinstance(selected, list):
        raise PosteriorPopulationContractError("inclusion ledger lacks selected states")
    minimum = int(checked["population"]["minimum_selected_states"])
    if len(selected) < minimum:
        raise PosteriorPopulationContractError(
            f"inclusion ledger has {len(selected)} states; contract requires {minimum}"
        )
    state_count = source.get("state_count")
    if not isinstance(state_count, int) or isinstance(state_count, bool) or state_count < len(selected):
        raise PosteriorPopulationContractError("source corpus cannot contain fewer states than ledger")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in selected:
        if not isinstance(row, Mapping):
            raise PosteriorPopulationContractError("selected ledger row must be an object")
        fixture_id = row.get("fixture_id")
        battle_tag = row.get("battle_tag")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise PosteriorPopulationContractError("selected row lacks fixture identity")
        if not isinstance(battle_tag, str) or not battle_tag:
            raise PosteriorPopulationContractError("selected row lacks battle identity")
        if fixture_id in seen:
            raise PosteriorPopulationContractError(f"duplicate selected fixture {fixture_id}")
        seen.add(fixture_id)
        rows.append({"fixture_id": fixture_id, "battle_tag": battle_tag})

    if "oracle" in enabled:
        authority = bindings.get("oracle_authority")
        if not isinstance(authority, Mapping):
            raise PosteriorPopulationContractError(
                "oracle stratum lacks conditional authority binding"
            )
        if authority.get("kind") != checked["oracle_authority"]["kind"]:
            raise PosteriorPopulationContractError(
                "oracle authority differs from the frozen contract"
            )
        _digest(authority.get("evidence_digest"), field="oracle authority evidence digest")
        if authority.get("realized_hidden_state_used") is not False:
            raise PosteriorPopulationContractError("oracle authority may not use realized hidden state")

    limit = bindings.get("compute_budget_per_method")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise PosteriorPopulationContractError("compute budget per method must be positive")

    matched_plan = {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "posterior_treatments": list(enabled),
        "compute_budget": {
            "unit": "transition_evaluations",
            "per_method_limit": limit,
        },
        "opponent_model": checked["opponent_model"],
        "depths": list(checked["execution_depths"]),
        "confirmatory_predictors": list(checked["confirmatory_predictors"]),
        "cluster_unit": checked["inference"]["cluster_unit"],
        "showdown_commit": _showdown_revision(),
        "evaluator": dict(evaluator),
        "chance_treatment": "shared_frozen_horizon_complete_oracle",
        "inference": {
            "bootstrap_replicates": checked["inference"]["bootstrap_replicates"],
            "bootstrap_seed": checked["inference"]["bootstrap_seed"],
        },
    }
    matched_plan = validate_plan(matched_plan)

    work = [
        {
            "fixture_id": row["fixture_id"],
            "battle_tag": row["battle_tag"],
            "posterior_treatment": treatment,
            "depth": depth,
        }
        for row in rows
        for treatment in enabled
        for depth in matched_plan["depths"]
    ]
    return {
        "schema": EXECUTION_SCHEMA,
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "contract_digest": stable_digest(checked),
        "source_corpus_digest": source_digest,
        "inclusion_ledger_digest": ledger_digest,
        "matched_plan": matched_plan,
        "selected_state_count": len(rows),
        "selected_battle_count": len({row["battle_tag"] for row in rows}),
        "expected_result_count": len(work),
        "work": work,
        "issue_69_complete_treatment_set": enabled == completion,
        "required_depth_curve": list(checked["required_depth_curve"]),
    }


def contract_readiness(contract: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_contract(contract)
    return {
        "schema": "azelficoast.posterior-stratified-population-contract-check",
        "schema_version": 1,
        "passed": True,
        "contract_digest": stable_digest(checked),
        "default_enabled_treatments": list(checked["default_enabled_treatments"]),
        "completion_posterior_treatments": list(checked["completion_posterior_treatments"]),
        "execution_depths": list(checked["execution_depths"]),
        "required_depth_curve": list(checked["required_depth_curve"]),
        "minimum_selected_states": int(checked["population"]["minimum_selected_states"]),
        "showdown_commit": _showdown_revision(),
        "oracle_authority_required": True,
        "oracle_authority_kind": checked["oracle_authority"]["kind"],
        "source_artifact_id": checked["population"]["source_artifact"]["artifact_id"],
    }
