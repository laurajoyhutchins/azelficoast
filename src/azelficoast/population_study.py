"""Preregistered population analysis for natural strategy-fusion bias."""

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

from azelficoast.corpus import DecisionFixture, load_corpus
from azelficoast.fusion_search import FusionSearchError, _candidate_signals
from azelficoast.live_belief import build_probe_source

PLAN_SCHEMA = "azelficoast.natural-population-strategy-fusion-plan"
PLAN_SCHEMA_VERSION = 1
COHORT_SCHEMA = "azelficoast.natural-population-strategy-fusion-cohort"
COHORT_SCHEMA_VERSION = 1
RESULT_SCHEMA = "azelficoast.natural-population-strategy-fusion-result"
RESULT_SCHEMA_VERSION = 1
AGGREGATE_SCHEMA = "azelficoast.natural-population-strategy-fusion-aggregate"
AGGREGATE_SCHEMA_VERSION = 1
SELECTION_SALT = "population-v1:"


class PopulationStudyError(ValueError):
    """Raised when population evidence violates the preregistered contract."""


def _load_object(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise PopulationStudyError(f"{path}: expected a JSON object")
    return document


def load_plan(path: str | Path) -> dict[str, Any]:
    plan = _load_object(path)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        raise PopulationStudyError("unexpected population-study plan schema")
    for field in ("source_artifact", "admissibility", "exact_treatment", "inference"):
        if not isinstance(plan.get(field), Mapping):
            raise PopulationStudyError(f"population-study plan lacks {field}")
    return plan


def _battle_tag(fixture: DecisionFixture) -> str:
    tags = {
        str(control.get("battle_tag"))
        for control in fixture.control_decisions
        if isinstance(control, Mapping) and control.get("battle_tag")
    }
    if len(tags) == 1:
        return next(iter(tags))
    if not tags:
        return f"fixture:{fixture.fixture_id}"
    raise PopulationStudyError(
        f"{fixture.fixture_id}: controls span multiple battle tags"
    )


def _hp_fraction(view: Mapping[str, Any]) -> float | None:
    current = view.get("current_hp")
    maximum = view.get("max_hp")
    if not isinstance(current, (int, float)) or not isinstance(maximum, (int, float)):
        return None
    if maximum <= 0:
        return None
    return float(current) / float(maximum)


def _opponent_public_hp_fraction(view: Mapping[str, Any]) -> float | None:
    fraction = view.get("hp_fraction")
    if isinstance(fraction, (int, float)):
        return float(fraction)
    current = view.get("current_hp")
    maximum = view.get("max_hp")
    if (
        isinstance(current, (int, float))
        and isinstance(maximum, (int, float))
        and maximum > 0
    ):
        return float(current) / float(maximum)
    if isinstance(current, (int, float)) and 0 <= float(current) <= 100:
        return float(current) / 100.0
    return None


def _entropy(weights: Mapping[str, Any]) -> float:
    values = [float(value) for value in weights.values()]
    if not values or any(value <= 0 for value in values):
        raise PopulationStudyError("hidden-item weights must be positive")
    total = sum(values)
    if total <= 0:
        raise PopulationStudyError("hidden-item weights have no mass")
    return -sum((value / total) * math.log2(value / total) for value in values)


def _selection_key(fixture_id: str) -> str:
    return hashlib.sha256((SELECTION_SALT + fixture_id).encode("utf-8")).hexdigest()


def _predictors(
    fixture: DecisionFixture,
    candidate: Mapping[str, Any],
    signals: Any,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    active = fixture.state.get("active")
    opponent = fixture.state.get("opponent_active")
    if not isinstance(active, Mapping) or not isinstance(opponent, Mapping):
        raise PopulationStudyError("admitted fixture lacks active state")
    item_weights = candidate.get("item_weights")
    if not isinstance(item_weights, Mapping):
        raise PopulationStudyError("candidate lacks hidden-item weights")
    protects = candidate.get("persistent_protect_actions")
    switches = candidate.get("persistent_switches")
    if not isinstance(protects, list) or not isinstance(switches, list):
        raise PopulationStudyError("candidate persistent branches are malformed")

    return {
        "turn": int(fixture.state.get("turn", 0)),
        "own_active_hp_fraction": _hp_fraction(active),
        "opponent_public_hp_fraction": _opponent_public_hp_fraction(opponent),
        "hidden_item_entropy_bits": _entropy(item_weights),
        "relative_move_order_changes": bool(
            diagnostics["relative_move_order_changes"]
        ),
        "incoming_ko_roll_probability_gap": float(
            diagnostics["incoming_ko_roll_probability_gap"]
        ),
        "incoming_damage_fraction_gap": float(
            diagnostics["incoming_damage_fraction_gap"]
        ),
        "minimum_hidden_item_mass": float(
            diagnostics["minimum_hidden_item_mass"]
        ),
        "persistent_protect_present": bool(protects),
        "persistent_switch_present": bool(switches),
        "persistent_branch_count": int(signals.persistent_branch_count),
        "legal_action_count": int(signals.legal_action_count),
    }


def freeze_population(
    *,
    plan: Mapping[str, Any],
    candidates_document: Mapping[str, Any],
    mechanics_document: Mapping[str, Any],
    fixtures: Sequence[DecisionFixture],
    output_dir: str | Path,
) -> dict[str, Any]:
    admission = plan["admissibility"]
    if candidates_document.get("schema") != "azelficoast.natural-fusion-candidates":
        raise PopulationStudyError("unexpected candidate discovery schema")
    if candidates_document.get("persistent_only") is not True:
        raise PopulationStudyError("candidate discovery must be persistent-only")
    if (
        mechanics_document.get("schema")
        != "azelficoast.public-belief-speed-fork-mechanics"
    ):
        raise PopulationStudyError("unexpected mechanics-screen schema")
    if mechanics_document.get("showdown_commit") != plan["showdown_commit"]:
        raise PopulationStudyError("mechanics screen used another Showdown revision")

    source_count = int(plan["source_artifact"]["decision_state_count"])
    if len(fixtures) != source_count:
        raise PopulationStudyError(
            f"source corpus has {len(fixtures)} fixtures; preregistered {source_count}"
        )

    candidates = candidates_document.get("candidates")
    cases = mechanics_document.get("cases")
    if not isinstance(candidates, list) or not isinstance(cases, list):
        raise PopulationStudyError("discovery evidence lacks candidates or mechanics cases")

    fixture_by_id = {fixture.fixture_id: fixture for fixture in fixtures}
    case_by_id = {
        str(case["fixture_id"]): case
        for case in cases
        if isinstance(case, Mapping) and isinstance(case.get("fixture_id"), str)
    }
    eligible: list[dict[str, Any]] = []
    ineligible_reason_counts: dict[str, int] = {}
    seen_candidates: set[str] = set()

    def exclude(reason: str) -> None:
        ineligible_reason_counts[reason] = ineligible_reason_counts.get(reason, 0) + 1

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise PopulationStudyError("candidate must be an object")
        fixture_id = candidate.get("fixture_id")
        if not isinstance(fixture_id, str):
            raise PopulationStudyError("candidate lacks fixture ID")
        if fixture_id in seen_candidates:
            raise PopulationStudyError(f"duplicate candidate {fixture_id}")
        seen_candidates.add(fixture_id)

        fixture = fixture_by_id.get(fixture_id)
        mechanics = case_by_id.get(fixture_id)
        if fixture is None:
            raise PopulationStudyError(f"candidate {fixture_id} missing from source corpus")
        if mechanics is None:
            raise PopulationStudyError(f"candidate {fixture_id} lacks mechanics screen")

        active = fixture.state.get("active")
        active_hp = active.get("current_hp") if isinstance(active, Mapping) else None
        if not isinstance(active_hp, (int, float)) or active_hp <= 0:
            exclude("active-hp-nonpositive")
            continue

        source, status = build_probe_source(fixture)
        if source is None:
            exclude(f"live-admission:{status}")
            continue
        if status != "admitted":
            raise PopulationStudyError(
                "live admission returned source without admitted status"
            )

        weights = candidate.get("item_weights")
        if not isinstance(weights, Mapping):
            raise PopulationStudyError(f"{fixture_id}: candidate item weights missing")
        if set(map(str, weights)) != set(map(str, source.get("plausible_items", []))):
            exclude("item-support-mismatch")
            continue

        try:
            signals, diagnostics = _candidate_signals(candidate, mechanics, fixture)
        except FusionSearchError as error:
            raise PopulationStudyError(
                f"{fixture_id}: invalid mechanics evidence: {error}"
            ) from error

        eligible.append(
            {
                "fixture_id": fixture_id,
                "battle_tag": _battle_tag(fixture),
                "selection_key": _selection_key(fixture_id),
                "predictors": _predictors(
                    fixture,
                    candidate,
                    signals,
                    diagnostics,
                ),
                "candidate": dict(candidate),
                "source": source,
            }
        )

    eligible.sort(key=lambda row: (row["selection_key"], row["fixture_id"]))
    cap = int(admission["max_exact_states"])
    if cap < 1:
        raise PopulationStudyError("max_exact_states must be positive")
    selected = eligible[:cap]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_artifact = dict(plan["source_artifact"])
    selected_rows: list[dict[str, Any]] = []

    for index, row in enumerate(selected, start=1):
        source = dict(row["source"])
        candidate = row["candidate"]
        source["observed_opponent_moves"] = list(candidate["revealed_moves"])
        source["source_artifact"] = {
            **source_artifact,
            "population_index": index,
            "selection_fixture_id": row["fixture_id"],
        }
        source["population_selection"] = {
            "schema": COHORT_SCHEMA,
            "schema_version": COHORT_SCHEMA_VERSION,
            "frozen_before_policy_values": True,
            "selection_uses_policy_result": False,
            "population_index": index,
            "battle_tag": row["battle_tag"],
            "selection_key": row["selection_key"],
            "predictors": row["predictors"],
        }
        filename = f"{index:03d}-{row['fixture_id']}.json"
        (destination / filename).write_text(
            json.dumps(source, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_rows.append(
            {
                "population_index": index,
                "fixture_id": row["fixture_id"],
                "battle_tag": row["battle_tag"],
                "selection_key": row["selection_key"],
                "filename": filename,
                "predictors": row["predictors"],
            }
        )

    outside = source_count - len(candidates)
    if outside < 0:
        raise PopulationStudyError("candidate count exceeds source decision-state count")

    return {
        "schema": COHORT_SCHEMA,
        "schema_version": COHORT_SCHEMA_VERSION,
        "source_artifact": source_artifact,
        "showdown_commit": plan["showdown_commit"],
        "frozen_before_policy_values": True,
        "selection_uses_policy_result": False,
        "source_decision_state_count": source_count,
        "bounded_candidate_count": len(candidates),
        "source_outside_bounded_model_count": outside,
        "eligible_count": len(eligible),
        "ineligible_reason_counts": dict(sorted(ineligible_reason_counts.items())),
        "max_exact_states": cap,
        "overflow_sampling_applied": len(eligible) > cap,
        "overflow_selection": admission["overflow_selection"],
        "selected_count": len(selected_rows),
        "eligible_fixture_ids": [row["fixture_id"] for row in eligible],
        "selected": selected_rows,
    }


def summarize_trace(
    *,
    source: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    selection = source.get("population_selection")
    if not isinstance(selection, Mapping):
        raise PopulationStudyError("source lacks frozen population-selection metadata")
    if selection.get("selection_uses_policy_result") is not False:
        raise PopulationStudyError("population selection is not policy-blind")
    if trace.get("experiment_valid") is not True:
        raise PopulationStudyError("exact trace is not valid")
    if trace.get("source_fixture_id") != source.get("fixture_id"):
        raise PopulationStudyError("exact trace fixture identity mismatch")
    if trace.get("showdown_commit") != source.get("showdown_commit"):
        raise PopulationStudyError("exact trace Showdown revision mismatch")

    determinization = trace.get("determinization")
    public = trace.get("public_belief")
    if not isinstance(determinization, Mapping) or not isinstance(public, Mapping):
        raise PopulationStudyError("trace lacks search results")
    det_values = determinization.get("root_values")
    public_values = public.get("root_values")
    if not isinstance(det_values, Mapping) or not isinstance(public_values, Mapping):
        raise PopulationStudyError("trace lacks root-action values")
    if set(det_values) != set(public_values) or not det_values:
        raise PopulationStudyError(
            "search methods did not evaluate identical root actions"
        )

    gaps = {
        str(action): float(det_values[action]) - float(public_values[action])
        for action in det_values
    }
    max_gap = max(gaps.values())
    max_gap_action = min(action for action, gap in gaps.items() if gap == max_gap)
    public_best = max(float(value) for value in public_values.values())
    det_action = determinization.get("chosen_action")
    public_action = public.get("chosen_action")
    if not isinstance(det_action, str) or det_action not in public_values:
        raise PopulationStudyError(
            "determinization chose a non-evaluated root action"
        )
    if not isinstance(public_action, str) or public_action not in public_values:
        raise PopulationStudyError(
            "public-belief search chose a non-evaluated root action"
        )

    regret = public_best - float(public_values[det_action])
    if regret < -1e-10:
        raise PopulationStudyError("computed determinization regret is negative")
    regret = max(0.0, regret)

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "population_index": int(selection["population_index"]),
        "fixture_id": source["fixture_id"],
        "battle_tag": str(selection["battle_tag"]),
        "selection_key": str(selection["selection_key"]),
        "predictors": dict(selection["predictors"]),
        "world_count": int(trace["world_count"]),
        "legal_action_count": int(trace["legal_action_count"]),
        "strategy_fusion_observation_count": int(
            trace["strategy_fusion_observation_count"]
        ),
        "max_strategy_fusion_value_advantage": max_gap,
        "max_strategy_fusion_action": max_gap_action,
        "determinization_public_regret": regret,
        "policy_disagreement": bool(trace["policy_disagreement"]),
        "determinization_action": det_action,
        "public_belief_action": public_action,
        "determinization_root_values": {
            str(action): float(value) for action, value in det_values.items()
        },
        "public_belief_root_values": {
            str(action): float(value) for action, value in public_values.items()
        },
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = _mean(left)
    right_mean = _mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left, right, strict=True)
    )
    left_ss = sum((x - left_mean) ** 2 for x in left)
    right_ss = sum((y - right_mean) ** 2 for y in right)
    if left_ss <= 0 or right_ss <= 0:
        return None
    return numerator / math.sqrt(left_ss * right_ss)


def _spearman(rows: Sequence[Mapping[str, Any]], predictor: str) -> dict[str, Any]:
    pairs = [
        (
            float(row["predictors"][predictor]),
            float(row["max_strategy_fusion_value_advantage"]),
        )
        for row in rows
        if isinstance(row.get("predictors"), Mapping)
        and isinstance(row["predictors"].get(predictor), (int, float))
        and not isinstance(row["predictors"].get(predictor), bool)
    ]
    if len(pairs) < 2:
        return {"n": len(pairs), "rho": None}
    x, y = zip(*pairs, strict=True)
    return {
        "n": len(pairs),
        "rho": _pearson(_ranks(x), _ranks(y)),
    }


def _binary_contrast(
    rows: Sequence[Mapping[str, Any]],
    predictor: str,
) -> dict[str, Any]:
    yes = [
        float(row["max_strategy_fusion_value_advantage"])
        for row in rows
        if isinstance(row.get("predictors"), Mapping)
        and row["predictors"].get(predictor) is True
    ]
    no = [
        float(row["max_strategy_fusion_value_advantage"])
        for row in rows
        if isinstance(row.get("predictors"), Mapping)
        and row["predictors"].get(predictor) is False
    ]
    return {
        "true_n": len(yes),
        "false_n": len(no),
        "true_mean_bias": _mean(yes) if yes else None,
        "false_mean_bias": _mean(no) if no else None,
        "mean_bias_difference": _mean(yes) - _mean(no) if yes and no else None,
    }


def _cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
    tolerance: float,
) -> dict[str, Any]:
    by_battle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_battle[str(row["battle_tag"])].append(row)
    clusters = sorted(by_battle)
    if not clusters:
        raise PopulationStudyError("cannot bootstrap an empty cohort")

    rng = random.Random(seed)
    mean_bias_samples: list[float] = []
    disagreement_samples: list[float] = []
    positive_bias_samples: list[float] = []
    regret_samples: list[float] = []

    for _ in range(replicates):
        sampled: list[Mapping[str, Any]] = []
        for _cluster_index in range(len(clusters)):
            sampled.extend(by_battle[clusters[rng.randrange(len(clusters))]])
        biases = [
            float(row["max_strategy_fusion_value_advantage"])
            for row in sampled
        ]
        regrets = [float(row["determinization_public_regret"]) for row in sampled]
        mean_bias_samples.append(_mean(biases))
        regret_samples.append(_mean(regrets))
        disagreement_samples.append(
            _mean([float(bool(row["policy_disagreement"])) for row in sampled])
        )
        positive_bias_samples.append(
            _mean([float(value > tolerance) for value in biases])
        )

    def interval(samples: Sequence[float]) -> list[float]:
        return [_quantile(samples, 0.025), _quantile(samples, 0.975)]

    return {
        "cluster_count": len(clusters),
        "replicates": replicates,
        "seed": seed,
        "mean_bias_95pct_ci": interval(mean_bias_samples),
        "mean_determinization_public_regret_95pct_ci": interval(regret_samples),
        "policy_disagreement_rate_95pct_ci": interval(disagreement_samples),
        "positive_bias_rate_95pct_ci": interval(positive_bias_samples),
    }


def aggregate_results(
    *,
    plan: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if manifest.get("schema") != COHORT_SCHEMA:
        raise PopulationStudyError("unexpected cohort manifest schema")
    expected = int(manifest["selected_count"])
    if len(rows) != expected:
        raise PopulationStudyError(
            f"expected {expected} exact results, received {len(rows)}"
        )
    if expected < 1:
        raise PopulationStudyError("population treatment selected no exact states")

    ordered = sorted(rows, key=lambda row: int(row["population_index"]))
    expected_indices = list(range(1, expected + 1))
    observed_indices = [int(row["population_index"]) for row in ordered]
    if observed_indices != expected_indices:
        raise PopulationStudyError(
            "exact population indices are incomplete or duplicated"
        )
    manifest_ids = [str(row["fixture_id"]) for row in manifest["selected"]]
    if [str(row["fixture_id"]) for row in ordered] != manifest_ids:
        raise PopulationStudyError("exact results do not match the frozen cohort")

    tolerance = float(plan["exact_treatment"]["positive_bias_tolerance"])
    biases = [float(row["max_strategy_fusion_value_advantage"]) for row in ordered]
    regrets = [float(row["determinization_public_regret"]) for row in ordered]
    disagreements = [bool(row["policy_disagreement"]) for row in ordered]
    fusions = [int(row["strategy_fusion_observation_count"]) for row in ordered]

    numeric_predictors = [
        "turn",
        "own_active_hp_fraction",
        "opponent_public_hp_fraction",
        "hidden_item_entropy_bits",
        "incoming_ko_roll_probability_gap",
        "incoming_damage_fraction_gap",
        "minimum_hidden_item_mass",
        "persistent_branch_count",
        "legal_action_count",
    ]
    binary_predictors = [
        "relative_move_order_changes",
        "persistent_protect_present",
        "persistent_switch_present",
    ]

    inference = plan["inference"]
    bootstrap = _cluster_bootstrap(
        ordered,
        replicates=int(inference["bootstrap_replicates"]),
        seed=int(inference["bootstrap_seed"]),
        tolerance=tolerance,
    )

    return {
        "schema": AGGREGATE_SCHEMA,
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "plan_schema": plan["schema"],
        "source_artifact": dict(plan["source_artifact"]),
        "cohort": {
            "source_decision_state_count": int(
                manifest["source_decision_state_count"]
            ),
            "bounded_candidate_count": int(manifest["bounded_candidate_count"]),
            "eligible_count": int(manifest["eligible_count"]),
            "selected_count": expected,
            "overflow_sampling_applied": bool(
                manifest["overflow_sampling_applied"]
            ),
            "exact_battle_count": len(
                {str(row["battle_tag"]) for row in ordered}
            ),
            "ineligible_reason_counts": dict(
                manifest["ineligible_reason_counts"]
            ),
        },
        "outcomes": {
            "positive_bias_tolerance": tolerance,
            "positive_bias_count": sum(value > tolerance for value in biases),
            "positive_bias_rate": _mean(
                [float(value > tolerance) for value in biases]
            ),
            "strategy_fusion_present_count": sum(value > 0 for value in fusions),
            "policy_disagreement_count": sum(disagreements),
            "policy_disagreement_rate": _mean(
                [float(value) for value in disagreements]
            ),
            "mean_max_strategy_fusion_value_advantage": _mean(biases),
            "median_max_strategy_fusion_value_advantage": _quantile(biases, 0.5),
            "p90_max_strategy_fusion_value_advantage": _quantile(biases, 0.9),
            "max_strategy_fusion_value_advantage": max(biases),
            "mean_determinization_public_regret": _mean(regrets),
            "median_determinization_public_regret": _quantile(regrets, 0.5),
            "p90_determinization_public_regret": _quantile(regrets, 0.9),
            "max_determinization_public_regret": max(regrets),
        },
        "cluster_bootstrap": bootstrap,
        "predictor_associations": {
            "numeric_spearman": {
                predictor: _spearman(ordered, predictor)
                for predictor in numeric_predictors
            },
            "binary_mean_bias_contrast": {
                predictor: _binary_contrast(ordered, predictor)
                for predictor in binary_predictors
            },
        },
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
    freeze.add_argument("candidates", type=Path)
    freeze.add_argument("mechanics", type=Path)
    freeze.add_argument("corpus", type=Path)
    freeze.add_argument("--output-dir", type=Path, required=True)
    freeze.add_argument("--manifest", type=Path, required=True)

    summarize = commands.add_parser("summarize")
    summarize.add_argument("source", type=Path)
    summarize.add_argument("trace", type=Path)
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
        result = freeze_population(
            plan=plan,
            candidates_document=_load_object(args.candidates),
            mechanics_document=_load_object(args.mechanics),
            fixtures=load_corpus(args.corpus),
            output_dir=args.output_dir,
        )
        _write_json(args.manifest, result)
        print(json.dumps(result, sort_keys=True))
        return 0

    if args.command == "summarize":
        result = summarize_trace(
            source=_load_object(args.source),
            trace=_load_object(args.trace),
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
        result = aggregate_results(
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
