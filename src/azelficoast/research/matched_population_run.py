"""Execute one frozen natural state across matched posterior/search treatments."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from azelficoast.belief.treatments import build_posterior
from azelficoast.research.matched_comparison import freeze_packet, settle_packet, validate_plan
from azelficoast.research.matched_search import execute_method

COHORT_SCHEMA = "azelficoast.matched-search-population-cohort"
COHORT_SCHEMA_VERSION = 1


class MatchedPopulationRunError(ValueError):
    """Raised when a natural-state treatment cannot bind to frozen evidence."""


def _selection(manifest: Mapping[str, Any], *, population_index: int) -> Mapping[str, Any]:
    selected = manifest.get("selected")
    if not isinstance(selected, list):
        raise MatchedPopulationRunError("population manifest lacks selected rows")
    matches = [
        row
        for row in selected
        if isinstance(row, Mapping)
        and int(row.get("population_index", -1)) == population_index
    ]
    if len(matches) != 1:
        raise MatchedPopulationRunError(
            f"population index {population_index} is not uniquely selected"
        )
    return matches[0]


def _state_from_source(
    *,
    source: Mapping[str, Any],
    selection: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    fixture = source.get("fixture")
    if not isinstance(fixture, Mapping):
        raise MatchedPopulationRunError("source lacks frozen fixture")
    fixture_id = fixture.get("fixture_id")
    if fixture_id != source.get("fixture_id") or fixture_id != selection.get("fixture_id"):
        raise MatchedPopulationRunError("source, fixture, and selection identities disagree")

    state = fixture.get("state")
    if not isinstance(state, Mapping):
        raise MatchedPopulationRunError("fixture lacks public state")
    legal_actions = state.get("legal_actions")
    if (
        not isinstance(legal_actions, list)
        or not legal_actions
        or not all(isinstance(action, str) and action for action in legal_actions)
    ):
        raise MatchedPopulationRunError("fixture lacks legal actions")

    predictors = selection.get("predictors")
    if not isinstance(predictors, Mapping):
        raise MatchedPopulationRunError("selection lacks frozen predictors")
    frozen_predictors: dict[str, Any] = {}
    for name in plan["confirmatory_predictors"]:
        if name not in predictors:
            raise MatchedPopulationRunError(
                f"selection lacks confirmatory predictor {name!r}"
            )
        frozen_predictors[str(name)] = predictors[name]

    battle_tag = selection.get("battle_tag")
    if not isinstance(battle_tag, str) or not battle_tag:
        raise MatchedPopulationRunError("selection lacks battle tag")

    return {
        "fixture_id": fixture_id,
        "battle_tag": battle_tag,
        "public_state": dict(state),
        "legal_actions": list(legal_actions),
        "predictors": frozen_predictors,
    }


def run_state(
    *,
    plan: Mapping[str, Any],
    source: Mapping[str, Any],
    manifest: Mapping[str, Any],
    population_index: int,
    transition_artifact: Mapping[str, Any],
    evaluator: Any,
) -> list[dict[str, Any]]:
    """Run every preregistered posterior/depth cell for one selected state."""
    checked_plan = validate_plan(plan)
    identity = getattr(evaluator, "identity", None)
    if not isinstance(identity, Mapping) or dict(identity) != checked_plan["evaluator"]:
        raise MatchedPopulationRunError(
            "loaded evaluator differs from the frozen matched-comparison plan"
        )

    selection = _selection(manifest, population_index=population_index)
    state = _state_from_source(source=source, selection=selection, plan=checked_plan)
    if transition_artifact.get("source_fixture_id") != state["fixture_id"]:
        raise MatchedPopulationRunError("transition artifact belongs to another selected state")

    results: list[dict[str, Any]] = []
    for treatment in checked_plan["posterior_treatments"]:
        posterior = build_posterior(transition_artifact, treatment=str(treatment))
        for depth in checked_plan["depths"]:
            packet = freeze_packet(
                plan=checked_plan,
                state=state,
                posterior=posterior,
                posterior_treatment=str(treatment),
                depth=int(depth),
            )
            receipts = [
                execute_method(
                    packet=packet,
                    posterior=posterior,
                    transition_program=transition_artifact,
                    method=method,
                    evaluator=evaluator,
                )
                for method in ("determinization", "information_set")
            ]
            result = settle_packet(packet=packet, receipts=receipts)
            result["population_index"] = population_index
            result["posterior_construction"] = dict(posterior["construction"])
            artifact_digest = receipts[0]["transition_artifact_digest"]
            if artifact_digest != receipts[1]["transition_artifact_digest"]:
                raise MatchedPopulationRunError(
                    "paired methods consumed different transition artifacts"
                )
            result["transition_artifact_digest"] = artifact_digest
            results.append(result)
    return results


def matched_cohort(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Project a frozen inclusion ledger into the population settlement contract."""
    selected = manifest.get("selected")
    if not isinstance(selected, list) or not selected:
        raise MatchedPopulationRunError("population manifest has no selected rows")
    rows: list[dict[str, Any]] = []
    for row in selected:
        if not isinstance(row, Mapping):
            raise MatchedPopulationRunError("population selection row must be an object")
        fixture_id = row.get("fixture_id")
        battle_tag = row.get("battle_tag")
        population_index = row.get("population_index")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise MatchedPopulationRunError("selected row lacks fixture identity")
        if not isinstance(battle_tag, str) or not battle_tag:
            raise MatchedPopulationRunError("selected row lacks battle identity")
        if not isinstance(population_index, int) or isinstance(population_index, bool):
            raise MatchedPopulationRunError("selected row lacks integral population index")
        rows.append(
            {
                "fixture_id": fixture_id,
                "battle_tag": battle_tag,
                "population_index": population_index,
            }
        )
    return {
        "schema": COHORT_SCHEMA,
        "schema_version": COHORT_SCHEMA_VERSION,
        "source_schema": manifest.get("schema"),
        "source_schema_version": manifest.get("schema_version"),
        "source_decision_state_count": manifest.get("source_decision_state_count"),
        "frozen_before_policy_values": manifest.get("frozen_before_policy_values"),
        "selection_uses_policy_result": manifest.get("selection_uses_policy_result"),
        "selected": rows,
    }
