#!/usr/bin/env python3
"""Run a bounded multi-generation Azelficoast self-improvement ratchet pilot.

The experiment measures, for each incumbent generation:

1. held-out regret of the learned policy relative to verified information-set search;
2. the fraction of exact-search-capable held-out decisions that would still route to
   search at a fixed policy-margin threshold;
3. live battle strength against RandomPlayer at that same deployment threshold.

Training uses separate shadow-mode battles. One successfully searched decision per
training battle is retained so the teacher set cannot be dominated by long battles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping, Sequence

from azelficoast.belief.evaluator import (
    BeliefEvaluatorRuntime,
    BeliefEvaluatorSpec,
    init_params,
    write_checkpoint,
)
from azelficoast.belief.improvement import AdmissionPolicy
from azelficoast.belief.self_improvement import run_self_improvement_cycle
from azelficoast.showdown_damage_corpus import PINNED_SHOWDOWN_COMMIT

EXPERIMENT_SCHEMA = "azelficoast.self-improvement-ratchet-experiment"
EXPERIMENT_SCHEMA_VERSION = 1


class ExperimentError(RuntimeError):
    """Raised when the ratchet experiment cannot produce trustworthy evidence."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ExperimentError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise ExperimentError(
                    f"{path}:{line_number}: expected a JSON object"
                )
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest(value: Any) -> str:
    payload = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _battle_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        return None
    kind = record.get("kind")
    tag = record.get("room") if kind == "protocol" else record.get("battle_tag")
    if not isinstance(tag, str) or not tag:
        return None
    return run_id, tag


def _terminal_battles(records: Sequence[Mapping[str, Any]]) -> list[tuple[str, str]]:
    battles = {
        key
        for record in records
        if record.get("kind") == "terminal"
        if (key := _battle_key(record)) is not None
    }
    return sorted(
        battles,
        key=lambda key: hashlib.sha256(
            f"{key[0]}\0{key[1]}".encode("utf-8")
        ).hexdigest(),
    )


def _partition_trace(
    source: Path,
    *,
    heldout_path: Path,
    training_path: Path,
    heldout_fraction: float,
) -> dict[str, int]:
    records = _read_jsonl(source)
    battles = _terminal_battles(records)
    if len(battles) < 4:
        raise ExperimentError(
            f"shadow trace contains only {len(battles)} completed battles"
        )
    heldout_count = max(2, int(round(len(battles) * heldout_fraction)))
    heldout_count = min(heldout_count, len(battles) - 2)
    heldout = set(battles[:heldout_count])
    training = set(battles[heldout_count:])

    _write_jsonl(
        heldout_path,
        (
            record
            for record in records
            if (key := _battle_key(record)) is not None and key in heldout
        ),
    )
    _write_jsonl(
        training_path,
        (
            record
            for record in records
            if (key := _battle_key(record)) is not None and key in training
        ),
    )
    return {
        "total_battles": len(battles),
        "heldout_battles": len(heldout),
        "training_battles": len(training),
    }


def _belief_record(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    metadata = record.get("decision_metadata")
    if not isinstance(metadata, Mapping):
        return None
    belief = metadata.get("belief")
    return belief if isinstance(belief, Mapping) else None


def _diagnostics(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    belief = _belief_record(record)
    if not isinstance(belief, Mapping):
        return None
    diagnostics = belief.get("diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else None


def _successful_teacher_decision(record: Mapping[str, Any]) -> bool:
    if record.get("kind") != "decision":
        return False
    belief = _belief_record(record)
    diagnostics = _diagnostics(record)
    if not isinstance(belief, Mapping) or not isinstance(diagnostics, Mapping):
        return False
    return (
        belief.get("reason") == "transition-program-public-belief"
        and isinstance(diagnostics.get("learned_prediction"), Mapping)
        and isinstance(diagnostics.get("public_belief_root_values"), Mapping)
    )


def _thin_training_trace(source: Path, destination: Path) -> dict[str, Any]:
    records = _read_jsonl(source)
    selected: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in records:
        if not _successful_teacher_decision(record):
            continue
        key = _battle_key(record)
        if key is None:
            continue
        previous = selected.get(key)
        if previous is None or int(record.get("decision_index", -1)) > int(
            previous.get("decision_index", -1)
        ):
            selected[key] = record

    selected_events = {
        (
            str(record["run_id"]),
            str(record["battle_tag"]),
            int(record["event_index"]),
        )
        for record in selected.values()
    }
    retained: list[Mapping[str, Any]] = []
    for record in records:
        if record.get("kind") != "decision":
            retained.append(record)
            continue
        run_id = record.get("run_id")
        battle_tag = record.get("battle_tag")
        event_index = record.get("event_index")
        if (
            isinstance(run_id, str)
            and isinstance(battle_tag, str)
            and isinstance(event_index, int)
            and (run_id, battle_tag, event_index) in selected_events
        ):
            retained.append(record)

    _write_jsonl(destination, retained)
    return {
        "selected_decision_count": len(selected_events),
        "selected_battle_count": len(selected),
        "source_battle_count": len(_terminal_battles(records)),
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


def _shadow_metrics(path: Path, *, deployment_threshold: float) -> dict[str, Any]:
    records = _read_jsonl(path)
    decisions = [record for record in records if record.get("kind") == "decision"]
    predictions = 0
    exact_teacher = 0
    regrets: list[float] = []
    margins: list[float] = []
    agreements = 0
    would_search = 0

    for record in decisions:
        diagnostics = _diagnostics(record)
        if not isinstance(diagnostics, Mapping):
            continue
        prediction = diagnostics.get("learned_prediction")
        if not isinstance(prediction, Mapping):
            continue
        predictions += 1
        margin = prediction.get("policy_margin")
        if isinstance(margin, (int, float)) and not isinstance(margin, bool):
            numeric_margin = float(margin)
            if math.isfinite(numeric_margin):
                margins.append(numeric_margin)

        root_values = diagnostics.get("public_belief_root_values")
        if not isinstance(root_values, Mapping) or not root_values:
            continue
        selected_action = prediction.get("selected_action")
        if not isinstance(selected_action, str) or selected_action not in root_values:
            continue
        numeric_values: dict[str, float] = {}
        valid = True
        for action, raw_value in root_values.items():
            if (
                not isinstance(action, str)
                or isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
            ):
                valid = False
                break
            value = float(raw_value)
            if not math.isfinite(value):
                valid = False
                break
            numeric_values[action] = value
        if not valid or not numeric_values:
            continue

        exact_teacher += 1
        best_value = max(numeric_values.values())
        best_action = min(
            action
            for action, value in numeric_values.items()
            if abs(value - best_value) <= 1e-12
        )
        regrets.append(best_value - numeric_values[selected_action])
        agreements += int(selected_action == best_action)
        if isinstance(margin, (int, float)) and not isinstance(margin, bool):
            would_search += int(float(margin) <= deployment_threshold)

    return {
        "battle_count": len(_terminal_battles(records)),
        "decision_count": len(decisions),
        "learned_prediction_count": predictions,
        "exact_teacher_decision_count": exact_teacher,
        "exact_teacher_coverage": (
            exact_teacher / len(decisions) if decisions else None
        ),
        "teacher_policy_agreement": (
            agreements / exact_teacher if exact_teacher else None
        ),
        "mean_public_regret": mean(regrets) if regrets else None,
        "median_public_regret": median(regrets) if regrets else None,
        "p90_public_regret": _percentile(regrets, 0.90),
        "mean_policy_margin": mean(margins) if margins else None,
        "search_dependence_at_threshold": (
            would_search / exact_teacher if exact_teacher else None
        ),
        "deployment_threshold": deployment_threshold,
    }


def _route_metrics(path: Path) -> dict[str, Any]:
    records = _read_jsonl(path)
    decisions = [record for record in records if record.get("kind") == "decision"]
    reasons = Counter()
    selected_policies = Counter()
    for record in decisions:
        metadata = record.get("decision_metadata")
        if not isinstance(metadata, Mapping):
            continue
        selected_policy = metadata.get("selected_policy")
        if isinstance(selected_policy, str):
            selected_policies[selected_policy] += 1
        belief = metadata.get("belief")
        if isinstance(belief, Mapping) and isinstance(belief.get("reason"), str):
            reasons[str(belief["reason"])] += 1
    direct = reasons["learned-public-belief"]
    searched = reasons["transition-program-public-belief"]
    fallback = selected_policies["simple-heuristics"]
    return {
        "decision_count": len(decisions),
        "direct_learned_count": direct,
        "verified_search_count": searched,
        "heuristic_fallback_count": fallback,
        "direct_learned_rate": direct / len(decisions) if decisions else None,
        "verified_search_rate": searched / len(decisions) if decisions else None,
        "heuristic_fallback_rate": fallback / len(decisions) if decisions else None,
        "belief_reason_counts": dict(sorted(reasons.items())),
    }


def _wilson_interval(successes: int, trials: int) -> tuple[float, float] | None:
    if trials <= 0:
        return None
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _battle_metrics(path: Path) -> dict[str, Any]:
    rows = _read_jsonl(path)
    wins = sum(row.get("won") is True for row in rows)
    losses = sum(row.get("lost") is True for row in rows)
    ties = max(0, len(rows) - wins - losses)
    interval = _wilson_interval(wins, len(rows))
    return {
        "battle_count": len(rows),
        "wins": wins,
        "losses": losses,
        "ties_or_unresolved": ties,
        "win_rate": wins / len(rows) if rows else None,
        "score_rate": (wins + 0.5 * ties) / len(rows) if rows else None,
        "win_rate_wilson_95": list(interval) if interval is not None else None,
    }


def _checkpoint_digest(path: Path) -> str:
    _, _, manifest = BeliefEvaluatorRuntime.from_checkpoint(path).params, None, None
    runtime = BeliefEvaluatorRuntime.from_checkpoint(path)
    return str(runtime.identity["checkpoint_digest"])


def _run_battles(
    *,
    repo_root: Path,
    workspace: Path,
    showdown_root: Path | None,
    checkpoint: Path | None,
    search_threshold: float | None,
    battles: int,
    label: str,
) -> tuple[Path, Path]:
    directory = workspace / "battles" / label
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)
    results = directory / "results.jsonl"
    decisions = directory / "decisions.jsonl"
    replays = directory / "replays"

    command = [
        sys.executable,
        "-m",
        "azelficoast.live.harness",
        "--results",
        str(results),
        "--decisions",
        str(decisions),
        "--replays",
        str(replays),
    ]
    if showdown_root is not None:
        command.extend(["--showdown-root", str(showdown_root)])
    if checkpoint is not None:
        command.extend(["--evaluator-checkpoint", str(checkpoint)])
    if search_threshold is not None:
        command.extend(["--search-policy-margin", str(search_threshold)])
    command.extend(
        [
            "local",
            "--battles",
            str(battles),
            "--concurrency",
            str(min(4, battles)),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        text=True,
        capture_output=True,
        timeout=max(300, 90 * battles),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    (directory / "stdout.log").write_text(completed.stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise ExperimentError(
            f"battle batch {label!r} failed with {completed.returncode}: "
            f"{completed.stderr[-2000:]}"
        )
    if not results.is_file() or not decisions.is_file():
        raise ExperimentError(f"battle batch {label!r} did not produce expected evidence")
    produced = len(_read_jsonl(results))
    if produced != battles:
        raise ExperimentError(
            f"battle batch {label!r} produced {produced} results, expected {battles}"
        )
    return results, decisions


def _initialize_seed_checkpoint(directory: Path) -> Path:
    spec = BeliefEvaluatorSpec()
    params = init_params(spec, seed=20260925)
    write_checkpoint(
        directory,
        params,
        spec,
        metadata={
            "kind": "self-improvement-ratchet-experiment-seed",
            "seed": 20260925,
        },
    )
    return directory


def _delta(last: Any, first: Any) -> float | None:
    if not isinstance(last, (int, float)) or isinstance(last, bool):
        return None
    if not isinstance(first, (int, float)) or isinstance(first, bool):
        return None
    return float(last) - float(first)


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    showdown_root = Path(args.showdown_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    workspace = Path(args.workspace).resolve()
    shutil.rmtree(workspace, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)

    seed = _initialize_seed_checkpoint(workspace / "seed-checkpoint")
    promotion = workspace / "models" / "current.json"
    models_dir = workspace / "models" / "candidates"
    receipts_dir = workspace / "models" / "receipts"
    cycle_workspace = workspace / "self-improvement"
    current_checkpoint = seed
    cumulative_training_traces: list[Path] = []

    baseline_results, baseline_trace = _run_battles(
        repo_root=repo_root,
        workspace=workspace,
        showdown_root=None,
        checkpoint=None,
        search_threshold=None,
        battles=args.deployment_battles,
        label="baseline-simple-heuristics",
    )
    baseline = {
        "battle_strength": _battle_metrics(baseline_results),
        "routing": _route_metrics(baseline_trace),
    }

    generations: list[dict[str, Any]] = []
    for generation in range(args.generations):
        runtime = BeliefEvaluatorRuntime.from_checkpoint(current_checkpoint)
        checkpoint_digest = str(runtime.identity["checkpoint_digest"])

        shadow_results, shadow_trace = _run_battles(
            repo_root=repo_root,
            workspace=workspace,
            showdown_root=showdown_root,
            checkpoint=current_checkpoint,
            search_threshold=1.0,
            battles=args.shadow_battles,
            label=f"generation-{generation}-shadow",
        )
        generation_dir = workspace / "generation-data" / str(generation)
        heldout_trace = generation_dir / "heldout.jsonl"
        training_partition = generation_dir / "training-partition.jsonl"
        partition = _partition_trace(
            shadow_trace,
            heldout_path=heldout_trace,
            training_path=training_partition,
            heldout_fraction=args.heldout_fraction,
        )
        thinned = generation_dir / "training-thinned.jsonl"
        thinning = _thin_training_trace(training_partition, thinned)
        if thinning["selected_decision_count"] > 0:
            cumulative_training_traces.append(thinned)

        deployment_results, deployment_trace = _run_battles(
            repo_root=repo_root,
            workspace=workspace,
            showdown_root=showdown_root,
            checkpoint=current_checkpoint,
            search_threshold=args.deployment_threshold,
            battles=args.deployment_battles,
            label=f"generation-{generation}-deployment",
        )

        generation_record: dict[str, Any] = {
            "generation": generation,
            "checkpoint_digest": checkpoint_digest,
            "shadow_battle_strength": _battle_metrics(shadow_results),
            "heldout": _shadow_metrics(
                heldout_trace,
                deployment_threshold=args.deployment_threshold,
            ),
            "deployment_battle_strength": _battle_metrics(deployment_results),
            "deployment_routing": _route_metrics(deployment_trace),
            "partition": partition,
            "training_selection": thinning,
        }

        if generation < args.generations - 1:
            cycle_result: dict[str, Any] | None = None
            extra_batches: list[dict[str, Any]] = []
            for attempt in range(args.max_cycle_attempts):
                if not cumulative_training_traces:
                    status = "not-ready"
                    reason = "no-successfully-searched-training-decisions"
                    cycle_result = {"status": status, "reason": reason}
                else:
                    cycle_result = run_self_improvement_cycle(
                        cumulative_training_traces,
                        showdown_root=showdown_root,
                        incumbent_checkpoint=current_checkpoint,
                        workspace=cycle_workspace,
                        models_dir=models_dir,
                        receipts_dir=receipts_dir,
                        promotion_file=promotion,
                        teacher_compute_budget=args.teacher_budget,
                        split_seed=args.split_seed,
                        train_fraction=args.train_fraction,
                        validation_fraction=args.validation_fraction,
                        epochs=args.epochs,
                        learning_rate=args.learning_rate,
                        policy_weight=args.policy_weight,
                        value_target_source="public_belief_search_return",
                        admission_policy=AdmissionPolicy(
                            min_validation_total_improvement=0.0,
                            max_validation_value_mse_regression=0.0,
                            max_validation_policy_cross_entropy_regression=0.0,
                        ),
                    )
                if cycle_result.get("status") != "not-ready":
                    break
                if attempt + 1 >= args.max_cycle_attempts:
                    break

                _, extra_trace = _run_battles(
                    repo_root=repo_root,
                    workspace=workspace,
                    showdown_root=showdown_root,
                    checkpoint=current_checkpoint,
                    search_threshold=1.0,
                    battles=args.extra_training_battles,
                    label=f"generation-{generation}-extra-{attempt + 1}",
                )
                extra_thinned = (
                    generation_dir / f"extra-training-{attempt + 1}.jsonl"
                )
                extra_selection = _thin_training_trace(extra_trace, extra_thinned)
                if extra_selection["selected_decision_count"] > 0:
                    cumulative_training_traces.append(extra_thinned)
                extra_batches.append(extra_selection)

            generation_record["cycle"] = cycle_result
            generation_record["extra_training_batches"] = extra_batches
            if promotion.is_file():
                current_checkpoint = promotion

        generations.append(generation_record)

    first = generations[0]
    last = generations[-1]
    trends = {
        "mean_public_regret_delta": _delta(
            last["heldout"].get("mean_public_regret"),
            first["heldout"].get("mean_public_regret"),
        ),
        "search_dependence_delta": _delta(
            last["heldout"].get("search_dependence_at_threshold"),
            first["heldout"].get("search_dependence_at_threshold"),
        ),
        "deployment_win_rate_delta": _delta(
            last["deployment_battle_strength"].get("win_rate"),
            first["deployment_battle_strength"].get("win_rate"),
        ),
        "direct_learned_rate_delta": _delta(
            last["deployment_routing"].get("direct_learned_rate"),
            first["deployment_routing"].get("direct_learned_rate"),
        ),
        "distinct_checkpoint_count": len(
            {record["checkpoint_digest"] for record in generations}
        ),
        "promotion_count": sum(
            record.get("cycle", {}).get("status") == "promoted"
            for record in generations
            if isinstance(record.get("cycle"), Mapping)
        ),
    }

    result = {
        "schema": EXPERIMENT_SCHEMA,
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_kind": "bounded-multigeneration-pilot",
        "repository_revision": os.getenv("GITHUB_SHA"),
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "configuration": {
            "generations": args.generations,
            "shadow_battles_per_generation": args.shadow_battles,
            "deployment_battles_per_generation": args.deployment_battles,
            "heldout_fraction": args.heldout_fraction,
            "deployment_policy_margin_threshold": args.deployment_threshold,
            "teacher_compute_budget": args.teacher_budget,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "policy_weight": args.policy_weight,
            "max_cycle_attempts": args.max_cycle_attempts,
            "extra_training_battles": args.extra_training_battles,
        },
        "baseline": baseline,
        "generations": generations,
        "trends": trends,
        "limitations": [
            "Battle-strength samples are small and use unpaired random battles against RandomPlayer.",
            "Held-out regret is conditional on states where verified live information-set search succeeded.",
            "The information-set teacher uses the incumbent evaluator at its leaves, so this pilot measures recursive search distillation rather than an independent optimal-policy oracle.",
            "The pilot is intended to detect whether the ratchet moves at all, not to establish a final Elo or causal effect size.",
        ],
    }
    result["experiment_digest"] = _digest(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/azelficoast-self-improvement-ratchet"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/self-improvement-ratchet-results.json"),
    )
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--shadow-battles", type=int, default=8)
    parser.add_argument("--deployment-battles", type=int, default=8)
    parser.add_argument("--extra-training-battles", type=int, default=6)
    parser.add_argument("--max-cycle-attempts", type=int, default=2)
    parser.add_argument("--heldout-fraction", type=float, default=0.375)
    parser.add_argument("--deployment-threshold", type=float, default=0.20)
    parser.add_argument("--teacher-budget", type=int, default=65536)
    parser.add_argument("--split-seed", default="azelficoast.ratchet-pilot")
    parser.add_argument("--train-fraction", type=float, default=0.50)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.generations < 2:
        raise ExperimentError("at least two generations are required")
    if args.shadow_battles < 4 or args.deployment_battles < 1:
        raise ExperimentError("battle counts are too small for this pilot")
    if not 0.0 < args.heldout_fraction < 1.0:
        raise ExperimentError("heldout fraction must be within (0, 1)")
    if not 0.0 <= args.deployment_threshold <= 1.0:
        raise ExperimentError("deployment threshold must be within [0, 1]")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
