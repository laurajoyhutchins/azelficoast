"""Fresh-cohort selection and paired aggregation for the depth-regret study."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.live.corpus import DecisionFixture, load_corpus
from azelficoast.research.depth import summarize_pair
from azelficoast.research.population import (
    _binary_contrast,
    _quantile,
    _spearman,
    freeze_population,
)

PLAN_SCHEMA = "azelficoast.natural-depth-regret-plan"
PLAN_SCHEMA_VERSION = 1
COHORT_SCHEMA = "azelficoast.natural-depth-regret-cohort"
COHORT_SCHEMA_VERSION = 1
RESULT_SCHEMA = "azelficoast.natural-depth-regret-result"
RESULT_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA = "azelficoast.natural-depth-regret-aggregate"
AGGREGATE_SCHEMA_VERSION = 1
SELECTION_SALT = "depth-regret-v1:"


class DepthPopulationError(ValueError):
    """Raised when fresh depth-regret evidence violates the frozen contract."""


def _load_object(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise DepthPopulationError(f"{path}: expected a JSON object")
    return document


def load_plan(path: str | Path) -> dict[str, Any]:
    plan = _load_object(path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        raise DepthPopulationError("unexpected depth-regret plan schema")
    for field in (
        "fresh_population",
        "admissibility",
        "cohort_selection",
        "paired_treatment",
        "inference",
    ):
        if not isinstance(plan.get(field), Mapping):
            raise DepthPopulationError(f"depth-regret plan lacks {field}")
    return plan


def _selection_key(fixture_id: str) -> str:
    return hashlib.sha256((SELECTION_SALT + fixture_id).encode()).hexdigest()


def _numeric_predictor_names(plan: Mapping[str, Any]) -> list[str]:
    frozen = plan.get("predictors_frozen_from_population_study")
    if not isinstance(frozen, Mapping):
        raise DepthPopulationError("plan lacks frozen predictors")
    rows = frozen.get("numeric")
    if not isinstance(rows, list):
        raise DepthPopulationError("plan numeric predictors are malformed")
    names = [
        str(row["name"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    ]
    if len(names) != len(rows):
        raise DepthPopulationError("plan numeric predictor lacks a name")
    return names


def _binary_predictor_names(plan: Mapping[str, Any]) -> list[str]:
    frozen = plan.get("predictors_frozen_from_population_study")
    if not isinstance(frozen, Mapping):
        raise DepthPopulationError("plan lacks frozen predictors")
    rows = frozen.get("binary")
    if not isinstance(rows, list):
        raise DepthPopulationError("plan binary predictors are malformed")
    names = [
        str(row["name"])
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("name"), str)
    ]
    if len(names) != len(rows):
        raise DepthPopulationError("plan binary predictor lacks a name")
    return names


def select_depth_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, float | None]]:
    if not rows:
        raise DepthPopulationError("cannot select from an empty eligible cohort")

    numeric = _numeric_predictor_names(plan)
    thresholds: dict[str, float | None] = {}
    for predictor in numeric:
        values = [
            float(row["predictors"][predictor])
            for row in rows
            if isinstance(row.get("predictors"), Mapping)
            and isinstance(row["predictors"].get(predictor), (int, float))
            and not isinstance(row["predictors"].get(predictor), bool)
        ]
        thresholds[predictor] = (
            _quantile(values, 0.75)
            if values and max(values) > min(values)
            else None
        )

    classified: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        predictors = row.get("predictors")
        if not isinstance(predictors, Mapping):
            raise DepthPopulationError("eligible row lacks predictors")

        numeric_hit = any(
            thresholds[predictor] is not None
            and isinstance(predictors.get(predictor), (int, float))
            and not isinstance(predictors.get(predictor), bool)
            and float(predictors[predictor]) >= float(thresholds[predictor])
            for predictor in numeric
        )
        row["depth_stratum"] = (
            "predictor-enriched" if numeric_hit else "representative-lower"
        )
        row["depth_selection_key"] = _selection_key(str(row["fixture_id"]))
        classified.append(row)

    enriched = sorted(
        (row for row in classified if row["depth_stratum"] == "predictor-enriched"),
        key=lambda row: (row["depth_selection_key"], str(row["fixture_id"])),
    )
    lower = sorted(
        (row for row in classified if row["depth_stratum"] == "representative-lower"),
        key=lambda row: (row["depth_selection_key"], str(row["fixture_id"])),
    )

    cap = int(plan["admissibility"]["max_exact_states"])
    if cap < 1:
        raise DepthPopulationError("max_exact_states must be positive")
    target_total = min(cap, len(classified))

    target_enriched = min(len(enriched), math.ceil(target_total * 2 / 3))
    target_lower = min(len(lower), target_total - target_enriched)
    remaining = target_total - target_enriched - target_lower
    if remaining:
        extra_enriched = min(len(enriched) - target_enriched, remaining)
        target_enriched += extra_enriched
        remaining -= extra_enriched
    if remaining:
        extra_lower = min(len(lower) - target_lower, remaining)
        target_lower += extra_lower
        remaining -= extra_lower
    if remaining:
        raise DepthPopulationError("could not satisfy frozen cohort size")

    selected = enriched[:target_enriched] + lower[:target_lower]
    selected.sort(
        key=lambda row: (row["depth_selection_key"], str(row["fixture_id"]))
    )
    return selected, thresholds


def freeze_depth_cohort(
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
            "overflow_selection": "all eligible states before depth stratification",
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
    eligible_rows = base_manifest["selected"]
    selected, thresholds = select_depth_rows(eligible_rows, plan=plan)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        source_path = all_sources / str(row["filename"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if not isinstance(source, dict):
            raise DepthPopulationError("eligible source is not an object")
        source["depth_selection"] = {
            "schema": COHORT_SCHEMA,
            "schema_version": COHORT_SCHEMA_VERSION,
            "depth_index": index,
            "stratum": row["depth_stratum"],
            "selection_key": row["depth_selection_key"],
            "predictors": dict(row["predictors"]),
            "quartile_thresholds": thresholds,
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
                "selection_key": row["depth_selection_key"],
                "predictors": dict(row["predictors"]),
            }
        )

    return {
        "schema": COHORT_SCHEMA,
        "schema_version": COHORT_SCHEMA_VERSION,
        "source_artifact": dict(source_artifact),
        "showdown_commit": plan["showdown_commit"],
        "selection_uses_policy_result": False,
        "source_decision_state_count": source_count,
        "bounded_candidate_count": int(base_manifest["bounded_candidate_count"]),
        "eligible_count": int(base_manifest["eligible_count"]),
        "ineligible_reason_counts": dict(base_manifest["ineligible_reason_counts"]),
        "quartile_thresholds": thresholds,
        "max_exact_states": int(admission["max_exact_states"]),
        "selected_count": len(selected_rows),
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


def summarize_selected_pair(
    *,
    source: Mapping[str, Any],
    shallow: Mapping[str, Any],
    deeper: Mapping[str, Any],
) -> dict[str, Any]:
    selection = source.get("depth_selection")
    if not isinstance(selection, Mapping):
        raise DepthPopulationError("source lacks frozen depth-selection metadata")
    if selection.get("selection_uses_policy_result") is not False:
        raise DepthPopulationError("depth selection is not outcome-blind")

    pair = summarize_pair(shallow, deeper)
    if pair.get("source_fixture_id") != source.get("fixture_id"):
        raise DepthPopulationError("paired trace fixture identity mismatch")

    shallow_result = pair["shallow"]
    deeper_result = pair["deeper"]
    change = pair["paired_change"]
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "depth_index": int(selection["depth_index"]),
        "fixture_id": source["fixture_id"],
        "battle_tag": str(
            source.get("population_selection", {}).get(
                "battle_tag",
                f"fixture:{source['fixture_id']}",
            )
        ),
        "stratum": str(selection["stratum"]),
        "selection_key": str(selection["selection_key"]),
        "predictors": dict(selection["predictors"]),
        "shallow_value_bias": float(
            shallow_result["max_strategy_fusion_value_advantage"]
        ),
        "deep_value_bias": float(
            deeper_result["max_strategy_fusion_value_advantage"]
        ),
        "value_bias_change": float(change["max_strategy_fusion_value_advantage"]),
        "shallow_regret": float(shallow_result["determinization_public_regret"]),
        "deep_regret": float(deeper_result["determinization_public_regret"]),
        "regret_change": float(change["determinization_public_regret"]),
        "shallow_policy_disagreement": bool(
            shallow_result["policy_disagreement"]
        ),
        "deep_policy_disagreement": bool(deeper_result["policy_disagreement"]),
        "shallow_determinization_action": shallow_result["determinization_action"],
        "deep_determinization_action": deeper_result["determinization_action"],
        "shallow_public_belief_action": shallow_result["public_belief_action"],
        "deep_public_belief_action": deeper_result["public_belief_action"],
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise DepthPopulationError("cannot summarize an empty value sequence")
    return {
        "mean": _mean(values),
        "median": _quantile(values, 0.5),
        "p90": _quantile(values, 0.9),
        "max": max(values),
    }


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_battle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battle[str(row["battle_tag"])].append(row)
    clusters = sorted(by_battle)
    if not clusters:
        raise DepthPopulationError("cannot bootstrap an empty cohort")

    metrics = (
        "deep_value_bias",
        "deep_regret",
        "value_bias_change",
        "regret_change",
    )
    samples: dict[str, list[float]] = {metric: [] for metric in metrics}
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled: list[Mapping[str, Any]] = []
        for _index in range(len(clusters)):
            sampled.extend(by_battle[clusters[rng.randrange(len(clusters))]])
        for metric in metrics:
            samples[metric].append(
                _mean([float(row[metric]) for row in sampled])
            )

    return {
        "cluster_count": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "mean_95pct_ci": {
            metric: [_quantile(values, 0.025), _quantile(values, 0.975)]
            for metric, values in samples.items()
        },
    }


def aggregate_depth_results(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if manifest.get("schema") != COHORT_SCHEMA:
        raise DepthPopulationError("unexpected depth cohort schema")
    expected = int(manifest["selected_count"])
    if expected < 1 or len(rows) != expected:
        raise DepthPopulationError(
            f"expected {expected} paired results, received {len(rows)}"
        )

    ordered = sorted(rows, key=lambda row: int(row["depth_index"]))
    if [int(row["depth_index"]) for row in ordered] != list(range(1, expected + 1)):
        raise DepthPopulationError("paired depth indices are incomplete or duplicated")
    expected_ids = [str(row["fixture_id"]) for row in manifest["selected"]]
    if [str(row["fixture_id"]) for row in ordered] != expected_ids:
        raise DepthPopulationError("paired results do not match the frozen cohort")

    metrics = (
        "shallow_value_bias",
        "deep_value_bias",
        "value_bias_change",
        "shallow_regret",
        "deep_regret",
        "regret_change",
    )
    outcomes = {
        metric: _summary([float(row[metric]) for row in ordered])
        for metric in metrics
    }
    outcomes["deep_positive_regret_count"] = sum(
        float(row["deep_regret"]) > 1e-12 for row in ordered
    )
    outcomes["deep_policy_disagreement_count"] = sum(
        bool(row["deep_policy_disagreement"]) for row in ordered
    )

    by_stratum: dict[str, Any] = {}
    for stratum in ("predictor-enriched", "representative-lower"):
        subset = [row for row in ordered if row.get("stratum") == stratum]
        if not subset:
            by_stratum[stratum] = {"count": 0}
            continue
        by_stratum[stratum] = {
            "count": len(subset),
            "deep_value_bias_mean": _mean(
                [float(row["deep_value_bias"]) for row in subset]
            ),
            "deep_regret_mean": _mean(
                [float(row["deep_regret"]) for row in subset]
            ),
            "value_bias_change_mean": _mean(
                [float(row["value_bias_change"]) for row in subset]
            ),
            "regret_change_mean": _mean(
                [float(row["regret_change"]) for row in subset]
            ),
        }

    numeric = _numeric_predictor_names(plan)
    binary = _binary_predictor_names(plan)
    association_outcomes = (
        "deep_value_bias",
        "deep_regret",
        "value_bias_change",
        "regret_change",
    )
    associations = {
        outcome: {
            "numeric_spearman": {
                predictor: _spearman(ordered, predictor, outcome)
                for predictor in numeric
            },
            "binary_mean_contrast": {
                predictor: _binary_contrast(ordered, predictor, outcome)
                for predictor in binary
            },
        }
        for outcome in association_outcomes
    }

    inference = plan["inference"]
    return {
        "schema": AGGREGATE_SCHEMA,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_artifact": dict(manifest["source_artifact"]),
        "cohort": {
            "source_decision_state_count": int(
                manifest["source_decision_state_count"]
            ),
            "bounded_candidate_count": int(manifest["bounded_candidate_count"]),
            "eligible_count": int(manifest["eligible_count"]),
            "selected_count": expected,
            "exact_battle_count": len(
                {str(row["battle_tag"]) for row in ordered}
            ),
            "stratum_counts": dict(manifest["stratum_counts"]),
        },
        "outcomes": outcomes,
        "by_stratum": by_stratum,
        "cluster_bootstrap": _cluster_bootstrap(
            ordered,
            replicates=int(inference["bootstrap_replicates"]),
            seed=int(inference["bootstrap_seed"]),
        ),
        "predictor_associations": associations,
        "results": [dict(row) for row in ordered],
    }


def _write_json(path: str | Path, document: Mapping[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(document, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("plan", type=Path)
    freeze.add_argument("source_metadata", type=Path)
    freeze.add_argument("candidates", type=Path)
    freeze.add_argument("mechanics", type=Path)
    freeze.add_argument("corpus", type=Path)
    freeze.add_argument("--scratch-dir", type=Path, required=True)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("source", type=Path)
    summarize.add_argument("shallow", type=Path)
    summarize.add_argument("deeper", type=Path)
    summarize.add_argument("--output", type=Path, required=True)

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("plan", type=Path)
    aggregate.add_argument("manifest", type=Path)
    aggregate.add_argument("result_dir", type=Path)
    aggregate.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        plan = load_plan(args.plan)
        result = freeze_depth_cohort(
            plan=plan,
            source_artifact=_load_object(args.source_metadata),
            candidates_document=_load_object(args.candidates),
            mechanics_document=_load_object(args.mechanics),
            fixtures=load_corpus(args.corpus),
            scratch_dir=args.scratch_dir,
            output_dir=args.output_dir,
        )
        _write_json(args.manifest, result)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "summarize":
        result = summarize_selected_pair(
            source=_load_object(args.source),
            shallow=_load_object(args.shallow),
            deeper=_load_object(args.deeper),
        )
        _write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "aggregate":
        plan = load_plan(args.plan)
        manifest = _load_object(args.manifest)
        rows = [
            _load_object(path)
            for path in sorted(args.result_dir.glob("summary-*.json"))
        ]
        result = aggregate_depth_results(
            plan=plan,
            manifest=manifest,
            rows=rows,
        )
        _write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0

    raise AssertionError(f"unexpected command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
