"""Outcome-blind Protect/environment cohort selection for two-horizon search."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.corpus import DecisionFixture, load_corpus
from azelficoast.depth_population import (
    COHORT_SCHEMA,
    COHORT_SCHEMA_VERSION,
    DepthPopulationError,
    load_plan,
)
from azelficoast.population_study import _quantile, freeze_population

SELECTION_SALT = "protect-weather-field-v1:"


def _selection_key(fixture_id: str) -> str:
    return hashlib.sha256((SELECTION_SALT + fixture_id).encode()).hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DepthPopulationError(f"{path}: expected a JSON object")
    return value


def _environment_profile(source: Mapping[str, Any]) -> dict[str, Any]:
    fixture = source.get("fixture")
    if not isinstance(fixture, Mapping):
        raise DepthPopulationError("probe source lacks fixture")
    state = fixture.get("state")
    if not isinstance(state, Mapping):
        raise DepthPopulationError("probe source fixture lacks state")
    weather = state.get("weather")
    fields = state.get("fields")
    if not isinstance(weather, Mapping) or not isinstance(fields, Mapping):
        raise DepthPopulationError("fixture environment state is malformed")
    return {
        "environment_present": bool(weather or fields),
        "weather_present": bool(weather),
        "terrain_present": bool(fields),
        "weather": dict(weather),
        "fields": dict(fields),
    }


def _classify_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    all_sources: Path,
    cap: int,
    minimum_weather_states: int = 0,
    minimum_terrain_states: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not rows:
        raise DepthPopulationError("no exact-admissible states remain")
    if cap < 1:
        raise DepthPopulationError("max_exact_states must be positive")

    expanded: list[dict[str, Any]] = []
    environment_actions: list[float] = []
    all_actions: list[float] = []
    for raw in rows:
        row = dict(raw)
        predictors = row.get("predictors")
        if not isinstance(predictors, Mapping):
            raise DepthPopulationError("eligible row lacks predictors")
        legal_actions = predictors.get("legal_action_count")
        if not isinstance(legal_actions, (int, float)) or isinstance(legal_actions, bool):
            raise DepthPopulationError("eligible row lacks legal_action_count")

        source = _load_object(all_sources / str(row["filename"]))
        environment = _environment_profile(source)
        actions = float(legal_actions)
        all_actions.append(actions)
        if environment["environment_present"]:
            environment_actions.append(actions)

        row["environment"] = environment
        row["selection_key"] = _selection_key(str(row["fixture_id"]))
        expanded.append(row)

    if not environment_actions:
        raise DepthPopulationError("no weather/terrain states survived exact admission")

    q75 = _quantile(environment_actions, 0.75)
    median = _quantile(all_actions, 0.5)
    for row in expanded:
        predictors = row["predictors"]
        environment = row["environment"]
        protect = predictors.get("persistent_protect_present") is True
        actions = float(predictors["legal_action_count"])
        enriched = (
            environment["environment_present"]
            and protect
            and actions >= q75
        )
        row["depth_stratum"] = (
            "predictor-enriched" if enriched else "representative-lower"
        )
        row["comparison_rank"] = (
            0
            if environment["environment_present"] and not protect
            else 1
            if environment["environment_present"] and actions <= median
            else 2
            if not protect
            else 3
            if actions <= median
            else 4
        )

    enriched_rows = sorted(
        (row for row in expanded if row["depth_stratum"] == "predictor-enriched"),
        key=lambda row: (row["selection_key"], str(row["fixture_id"])),
    )
    comparator_rows = sorted(
        (
            row
            for row in expanded
            if row["depth_stratum"] == "representative-lower"
        ),
        key=lambda row: (
            int(row["comparison_rank"]),
            row["selection_key"],
            str(row["fixture_id"]),
        ),
    )

    target_total = min(cap, len(expanded))
    target_enriched = min(len(enriched_rows), math.ceil(target_total * 2 / 3))
    target_comparator = min(len(comparator_rows), target_total - target_enriched)

    required_comparators: list[dict[str, Any]] = []
    required_ids: set[str] = set()

    def reserve_environment(kind: str, minimum: int) -> None:
        if minimum < 0:
            raise DepthPopulationError("environment minimums must be nonnegative")
        count = 0
        for row in comparator_rows:
            if not row["environment"][kind]:
                continue
            fixture_id = str(row["fixture_id"])
            if fixture_id not in required_ids:
                required_comparators.append(row)
                required_ids.add(fixture_id)
            count += 1
            if count >= minimum:
                return

    reserve_environment("terrain_present", minimum_terrain_states)
    reserve_environment("weather_present", minimum_weather_states)

    target_comparator = max(target_comparator, len(required_comparators))
    if target_comparator > len(comparator_rows):
        target_comparator = len(comparator_rows)
    if target_enriched + target_comparator > target_total:
        target_enriched = max(0, target_total - target_comparator)

    selected_comparators = list(required_comparators)
    selected_ids = set(required_ids)
    for row in comparator_rows:
        if len(selected_comparators) >= target_comparator:
            break
        fixture_id = str(row["fixture_id"])
        if fixture_id in selected_ids:
            continue
        selected_comparators.append(row)
        selected_ids.add(fixture_id)

    remaining = target_total - target_enriched - len(selected_comparators)
    if remaining:
        extra_enriched = min(
            len(enriched_rows) - target_enriched,
            remaining,
        )
        target_enriched += extra_enriched
        remaining -= extra_enriched
    if remaining:
        raise DepthPopulationError("could not fill the preregistered cohort cap")

    selected = enriched_rows[:target_enriched] + selected_comparators
    selected.sort(key=lambda row: (row["selection_key"], str(row["fixture_id"])))
    return selected, {
        "environment_legal_action_q75": q75,
        "all_eligible_legal_action_median": median,
    }


def freeze_environment_cohort(
    *,
    plan: Mapping[str, Any],
    source_artifact: Mapping[str, Any],
    candidates_document: Mapping[str, Any],
    mechanics_document: Mapping[str, Any],
    fixtures: Sequence[DecisionFixture],
    scratch_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_count = int(source_artifact.get("decision_state_count", 0))
    if source_count != len(fixtures):
        raise DepthPopulationError(
            f"source metadata says {source_count} states; corpus has {len(fixtures)}"
        )

    admission = plan["admissibility"]
    base_plan = {
        "source_artifact": dict(source_artifact),
        "showdown_commit": plan["showdown_commit"],
        "admissibility": {
            "generator_rounds": int(admission["generator_rounds"]),
            "mechanics_screen_rounds": int(admission["mechanics_screen_rounds"]),
            "max_exact_states": max(1, source_count),
            "overflow_selection": "all eligible states before environment stratification",
        },
    }

    scratch = Path(scratch_dir)
    all_sources = scratch / "all-eligible"
    base_manifest = freeze_population(
        plan=base_plan,
        candidates_document=candidates_document,
        mechanics_document=mechanics_document,
        fixtures=fixtures,
        output_dir=all_sources,
    )
    selected, thresholds = _classify_rows(
        base_manifest["selected"],
        all_sources=all_sources,
        cap=int(admission["max_exact_states"]),
        minimum_weather_states=int(
            plan["fresh_population"].get("minimum_weather_states", 0)
        ),
        minimum_terrain_states=int(
            plan["fresh_population"].get("minimum_terrain_states", 0)
        ),
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        source = _load_object(all_sources / str(row["filename"]))
        predictors = dict(row["predictors"])
        environment = dict(row["environment"])
        predictors.update(
            {
                "environment_present": bool(environment["environment_present"]),
                "weather_present": bool(environment["weather_present"]),
                "terrain_present": bool(environment["terrain_present"]),
            }
        )
        source["depth_selection"] = {
            "schema": COHORT_SCHEMA,
            "schema_version": COHORT_SCHEMA_VERSION,
            "depth_index": index,
            "stratum": row["depth_stratum"],
            "selection_key": row["selection_key"],
            "predictors": predictors,
            "environment": environment,
            "thresholds": thresholds,
            "comparison_rank": int(row["comparison_rank"]),
            "selection_uses_policy_result": False,
        }
        filename = f"{index:03d}-{row['fixture_id']}.json"
        (destination / filename).write_text(
            json.dumps(source, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_rows.append(
            {
                "depth_index": index,
                "fixture_id": row["fixture_id"],
                "battle_tag": row["battle_tag"],
                "filename": filename,
                "stratum": row["depth_stratum"],
                "selection_key": row["selection_key"],
                "predictors": predictors,
                "environment": environment,
                "comparison_rank": int(row["comparison_rank"]),
            }
        )

    return {
        "schema": COHORT_SCHEMA,
        "schema_version": COHORT_SCHEMA_VERSION,
        "source_artifact": dict(source_artifact),
        "showdown_commit": plan["showdown_commit"],
        "selection_uses_policy_result": False,
        "selection_contract": "Protect + top-quartile branching among bounded-exact weather/terrain states; deterministic lower comparator",
        "source_decision_state_count": source_count,
        "bounded_candidate_count": int(base_manifest["bounded_candidate_count"]),
        "eligible_count": int(base_manifest["eligible_count"]),
        "ineligible_reason_counts": dict(base_manifest["ineligible_reason_counts"]),
        "thresholds": thresholds,
        "max_exact_states": int(admission["max_exact_states"]),
        "selected_count": len(selected_rows),
        "environment_selected_count": sum(
            bool(row["environment"]["environment_present"])
            for row in selected_rows
        ),
        "protect_environment_selected_count": sum(
            bool(row["environment"]["environment_present"])
            and row["predictors"]["persistent_protect_present"] is True
            for row in selected_rows
        ),
        "stratum_counts": {
            "predictor-enriched": sum(
                row["stratum"] == "predictor-enriched" for row in selected_rows
            ),
            "representative-lower": sum(
                row["stratum"] == "representative-lower" for row in selected_rows
            ),
        },
        "selected": selected_rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("source_metadata", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("mechanics", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--scratch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = load_plan(args.plan)
    result = freeze_environment_cohort(
        plan=plan,
        source_artifact=_load_object(args.source_metadata),
        candidates_document=_load_object(args.candidates),
        mechanics_document=_load_object(args.mechanics),
        fixtures=load_corpus(args.corpus),
        scratch_dir=args.scratch_dir,
        output_dir=args.output_dir,
    )
    args.manifest.write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
