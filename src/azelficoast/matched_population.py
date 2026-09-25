"""Population settlement for matched search comparisons."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.matched_comparison import (
    RESULT_SCHEMA as COMPARISON_RESULT_SCHEMA,
    MatchedComparisonError,
    validate_plan,
)

COHORT_SCHEMA = "azelficoast.matched-search-population-cohort"
COHORT_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA = "azelficoast.matched-search-population-aggregate"
AGGREGATE_SCHEMA_VERSION = 1


class MatchedPopulationError(ValueError):
    """Raised when matched-comparison population evidence is incomplete."""


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedPopulationError(f"{path}: expected a JSON object")
    return value


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise MatchedPopulationError("cannot take a quantile of an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _cluster_interval(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
    replicates: int,
    seed: int,
) -> list[float]:
    by_battle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battle[str(row["battle_tag"])].append(row)
    battles = sorted(by_battle)
    if not battles:
        raise MatchedPopulationError("cannot bootstrap an empty stratum")

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(replicates):
        sampled: list[float] = []
        for _index in battles:
            battle = battles[rng.randrange(len(battles))]
            sampled.extend(float(row[field]) for row in by_battle[battle])
        samples.append(_mean(sampled))
    return [_quantile(samples, 0.025), _quantile(samples, 0.975)]


def aggregate_population(
    *,
    plan: Mapping[str, Any],
    cohort: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checked_plan = validate_plan(plan)
    if (
        cohort.get("schema") != COHORT_SCHEMA
        or cohort.get("schema_version") != COHORT_SCHEMA_VERSION
    ):
        raise MatchedPopulationError("unexpected matched-population cohort schema")

    selected = cohort.get("selected")
    if not isinstance(selected, list) or not selected:
        raise MatchedPopulationError("cohort must contain selected natural states")

    state_battles: dict[str, str] = {}
    for row in selected:
        if not isinstance(row, Mapping):
            raise MatchedPopulationError("cohort row must be an object")
        fixture_id = row.get("fixture_id")
        battle_tag = row.get("battle_tag")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise MatchedPopulationError("cohort row lacks fixture_id")
        if not isinstance(battle_tag, str) or not battle_tag:
            raise MatchedPopulationError("cohort row lacks battle_tag")
        if fixture_id in state_battles:
            raise MatchedPopulationError(f"duplicate cohort fixture {fixture_id}")
        state_battles[fixture_id] = battle_tag

    expected = {
        (fixture_id, treatment, int(depth))
        for fixture_id in state_battles
        for treatment in checked_plan["posterior_treatments"]
        for depth in checked_plan["depths"]
    }
    observed: dict[tuple[str, str, int], Mapping[str, Any]] = {}

    for row in results:
        if row.get("schema") != COMPARISON_RESULT_SCHEMA:
            raise MatchedPopulationError("unexpected per-state comparison result schema")
        if row.get("matched_input") is not True:
            raise MatchedPopulationError("population contains an unmatched input")
        if row.get("matched_authorized_compute") is not True:
            raise MatchedPopulationError("population contains unmatched compute")
        if row.get("matched_evaluator") is not True:
            raise MatchedPopulationError("population contains an unmatched evaluator")
        if row.get("evaluator") != checked_plan["evaluator"]:
            raise MatchedPopulationError("population mixed evaluator checkpoints")
        support = row.get("posterior_support")
        if not isinstance(support, Mapping):
            raise MatchedPopulationError("population result lacks posterior support diagnostics")
        for field in (
            "support_size",
            "entropy_bits",
            "effective_sample_size",
            "minimum_mass",
            "maximum_mass",
        ):
            value = support.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise MatchedPopulationError(
                    f"population result has invalid posterior support field {field}"
                )

        fixture_id = str(row.get("fixture_id"))
        treatment = str(row.get("posterior_treatment"))
        depth = int(row.get("depth", 0))
        key = (fixture_id, treatment, depth)
        if key in observed:
            raise MatchedPopulationError(f"duplicate population treatment {key!r}")
        if fixture_id not in state_battles:
            raise MatchedPopulationError(f"result references unselected fixture {fixture_id}")
        if row.get("battle_tag") != state_battles[fixture_id]:
            raise MatchedPopulationError(f"{fixture_id}: battle tag drifted")
        observed[key] = row

    missing = expected - set(observed)
    extra = set(observed) - expected
    if missing or extra:
        raise MatchedPopulationError(
            "population matrix is incomplete: "
            f"missing={sorted(missing)[:3]!r} extra={sorted(extra)[:3]!r}"
        )

    inference = checked_plan["inference"]
    replicates = int(inference["bootstrap_replicates"])
    seed = int(inference["bootstrap_seed"])
    figure_rows: list[dict[str, Any]] = []

    for treatment in checked_plan["posterior_treatments"]:
        for depth in checked_plan["depths"]:
            stratum = [
                observed[(fixture_id, treatment, int(depth))]
                for fixture_id in state_battles
            ]
            bias_field = "max_determinization_value_optimism"
            regret_field = "information_set_regret_of_determinization_action"
            disagreement_values = [
                float(bool(row["policy_disagreement"])) for row in stratum
            ]
            figure_rows.append(
                {
                    "posterior_treatment": treatment,
                    "depth": int(depth),
                    "state_count": len(stratum),
                    "battle_count": len(
                        {str(row["battle_tag"]) for row in stratum}
                    ),
                    "mean_value_optimism": _mean(
                        [float(row[bias_field]) for row in stratum]
                    ),
                    "mean_value_optimism_95pct_ci": _cluster_interval(
                        stratum,
                        field=bias_field,
                        replicates=replicates,
                        seed=seed,
                    ),
                    "mean_regret": _mean(
                        [float(row[regret_field]) for row in stratum]
                    ),
                    "mean_regret_95pct_ci": _cluster_interval(
                        stratum,
                        field=regret_field,
                        replicates=replicates,
                        seed=seed + 1,
                    ),
                    "policy_disagreement_rate": _mean(disagreement_values),
                    "mean_posterior_support_size": _mean(
                        [float(row["posterior_support"]["support_size"]) for row in stratum]
                    ),
                    "mean_posterior_entropy_bits": _mean(
                        [float(row["posterior_support"]["entropy_bits"]) for row in stratum]
                    ),
                    "mean_posterior_effective_sample_size": _mean(
                        [
                            float(row["posterior_support"]["effective_sample_size"])
                            for row in stratum
                        ]
                    ),
                    "mean_posterior_maximum_mass": _mean(
                        [float(row["posterior_support"]["maximum_mass"]) for row in stratum]
                    ),
                }
            )

    posterior_sensitivity_rows: list[dict[str, Any]] = []
    for depth in checked_plan["depths"]:
        rows_at_depth = [
            row for row in figure_rows if int(row["depth"]) == int(depth)
        ]
        optimism = [float(row["mean_value_optimism"]) for row in rows_at_depth]
        regret = [float(row["mean_regret"]) for row in rows_at_depth]
        disagreement = [
            float(row["policy_disagreement_rate"]) for row in rows_at_depth
        ]
        posterior_sensitivity_rows.append(
            {
                "depth": int(depth),
                "posterior_treatment_count": len(rows_at_depth),
                "posterior_treatments": [
                    str(row["posterior_treatment"]) for row in rows_at_depth
                ],
                "mean_value_optimism_range": [min(optimism), max(optimism)],
                "mean_value_optimism_span": max(optimism) - min(optimism),
                "mean_regret_range": [min(regret), max(regret)],
                "mean_regret_span": max(regret) - min(regret),
                "policy_disagreement_rate_range": [
                    min(disagreement),
                    max(disagreement),
                ],
                "policy_disagreement_rate_span": (
                    max(disagreement) - min(disagreement)
                ),
            }
        )

    return {
        "schema": AGGREGATE_SCHEMA,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "source_state_count": len(state_battles),
        "posterior_treatments": list(checked_plan["posterior_treatments"]),
        "depths": list(checked_plan["depths"]),
        "opponent_model": checked_plan["opponent_model"],
        "compute_budget": dict(checked_plan["compute_budget"]),
        "evaluator": dict(checked_plan["evaluator"]),
        "matrix_complete": True,
        "matched_input": True,
        "matched_authorized_compute": True,
        "matched_evaluator": True,
        "bootstrap": {
            "cluster_unit": "battle_tag",
            "replicates": replicates,
            "seed": seed,
        },
        "figure_rows": figure_rows,
        "posterior_sensitivity_rows": posterior_sensitivity_rows,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = [
        _load_object(path)
        for path in sorted(args.result_dir.glob("*.json"))
    ]
    try:
        result = aggregate_population(
            plan=_load_object(args.plan),
            cohort=_load_object(args.cohort),
            results=rows,
        )
    except MatchedComparisonError as error:
        raise MatchedPopulationError(str(error)) from error
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
