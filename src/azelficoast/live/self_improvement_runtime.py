"""Automatic curriculum acquisition and evaluator self-improvement."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_env.player import MaxBasePowerPlayer, Player, SimpleHeuristicsPlayer

from azelficoast.belief.battle_promotion import BattlePromotionPolicy
from azelficoast.belief.competence import allocate_evidence_budget, evidence_source_debt
from azelficoast.belief.improvement import AdmissionPolicy, promote_deferred_candidate
from azelficoast.belief.self_improvement import run_self_improvement_cycle
from azelficoast.live.battle_runtime import BATTLE_FORMAT, prepare_output_paths, run_local
from azelficoast.live.opponents import DirtyTricksPlayer
from azelficoast.live.player import AzelficoastPlayer
from azelficoast.live.promotion_runtime import (
    checkpoint_digest,
    file_sha256,
    run_promotion_panel,
)
from azelficoast.research.public_replays import PublicReplayError, import_public_replays


def _immutable_checkpoint(path: Path) -> Path:
    """Resolve a mutable promotion pointer to the checkpoint it names now."""

    if path.is_dir():
        return path
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot resolve incumbent checkpoint {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError(f"incumbent pointer {path} must contain a JSON object")
    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError(f"incumbent pointer {path} does not name a checkpoint")
    target = path.parent / checkpoint
    if not target.is_dir():
        raise ValueError(f"incumbent pointer {path} names missing checkpoint {target}")
    return target


def _training_opponent_specs(
    current_checkpoint: Path,
    archive_checkpoints: Sequence[Path],
) -> list[dict[str, Any]]:
    """Freeze the deterministic opponent population for one generation."""

    specs: list[dict[str, Any]] = [
        {"kind": "max-base-power"},
        {"kind": "simple-heuristics"},
        {"kind": "dirty-tricks"},
        {
            "kind": "incumbent",
            "checkpoint": str(current_checkpoint),
            "checkpoint_digest": checkpoint_digest(current_checkpoint),
        },
    ]
    for offset, checkpoint in enumerate(reversed(archive_checkpoints[-3:]), start=1):
        specs.append(
            {
                "kind": "archive",
                "archive_recency": offset,
                "checkpoint": str(checkpoint),
                "checkpoint_digest": checkpoint_digest(checkpoint),
            }
        )
    return specs


def _training_opponent(
    spec: Mapping[str, Any],
    *,
    concurrency: int,
    showdown_root: Path,
    belief_timeout: float,
    search_policy_margin: float,
) -> Player:
    kind = spec.get("kind")
    if kind == "max-base-power":
        return MaxBasePowerPlayer(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=concurrency,
        )
    if kind == "simple-heuristics":
        return SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=concurrency,
        )
    if kind == "dirty-tricks":
        return DirtyTricksPlayer(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=concurrency,
        )
    if kind in {"incumbent", "archive"}:
        checkpoint = spec.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"{kind} opponent lacks a checkpoint")
        return AzelficoastPlayer(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=concurrency,
            showdown_root=showdown_root,
            belief_timeout_seconds=belief_timeout,
            evaluator_checkpoint=Path(checkpoint),
            search_policy_margin=search_policy_margin,
        )
    raise ValueError(f"unsupported training opponent kind {kind!r}")


def adaptive_public_replay_count(
    maximum: int,
    *,
    ledger: Mapping[str, Any] | None,
    generated_source_kinds: Sequence[str],
) -> int:
    """Back off public acquisition only when its debt is below generated evidence."""

    if maximum <= 0:
        return 0
    if ledger is None or not generated_source_kinds:
        return maximum
    public_debt = float(
        evidence_source_debt(ledger, "public-showdown-replay")["acquisition_debt"]
    )
    generated_debts = [
        float(evidence_source_debt(ledger, source)["acquisition_debt"])
        for source in generated_source_kinds
    ]
    generated_mean = math.fsum(generated_debts) / len(generated_debts)
    if generated_mean <= 0.0 or public_debt >= generated_mean:
        return maximum
    if public_debt <= 0.0:
        return 0
    return max(1, math.ceil(maximum * public_debt / generated_mean))


def _competence_ledger_from_teacher_manifest(path: object) -> Mapping[str, Any] | None:
    if not isinstance(path, str) or not path:
        return None
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read teacher competence ledger: {error}") from error
    if not isinstance(document, Mapping):
        raise ValueError("teacher manifest must contain a JSON object")
    selection = document.get("selection")
    if not isinstance(selection, Mapping):
        return None
    ledger = selection.get("competence_ledger")
    return ledger if isinstance(ledger, Mapping) else None


async def run_automatic_self_improvement(args: argparse.Namespace) -> dict[str, object]:
    """Acquire mixed curriculum evidence and attempt bounded promotions."""

    if args.showdown_root is None:
        raise ValueError(
            "automatic training requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
        )

    traces: list[Path] = []
    current_checkpoint = _immutable_checkpoint(args.incumbent)
    archive_checkpoints: list[Path] = []
    generations: list[dict[str, object]] = []
    public_before: int | None = None
    competence_ledger: Mapping[str, Any] | None = None
    competence_ledger_manifest: str | None = None

    for index in range(args.generations):
        generation_number = index + 1
        generation_root = args.workspace / "generations" / f"{generation_number:04d}"
        if generation_root.exists():
            raise ValueError(
                f"generation output already exists and is immutable: {generation_root}"
            )
        decisions = generation_root / "decisions.jsonl"
        results = generation_root / "results.jsonl"
        replays = generation_root / "replays"
        manifest_path = generation_root / "generation.json"

        prepare_output_paths(results, decisions, replays)
        opponent_specs = _training_opponent_specs(
            current_checkpoint,
            archive_checkpoints,
        )
        generated_source_kinds = [
            f"generated-{spec['kind']}" for spec in opponent_specs
        ]
        allocations = allocate_evidence_budget(
            generated_source_kinds,
            args.battles_per_generation,
            ledger=competence_ledger,
            rotation=index,
        )
        public_replay_count = adaptive_public_replay_count(
            args.public_replays_per_generation,
            ledger=competence_ledger,
            generated_source_kinds=generated_source_kinds,
        )
        source_debts = [
            evidence_source_debt(competence_ledger, source)
            for source in generated_source_kinds
        ]
        public_source_debt = evidence_source_debt(
            competence_ledger,
            "public-showdown-replay",
        )
        acquisition_plan = {
            "kind": "competence-debt",
            "ledger_manifest": competence_ledger_manifest,
            "cold_start": competence_ledger is None,
            "generated_sources": [
                {
                    **dict(debt),
                    "battle_count": battle_count,
                }
                for debt, battle_count in zip(
                    source_debts,
                    allocations,
                    strict=True,
                )
            ],
            "public_source": {
                **dict(public_source_debt),
                "configured_max_replays": args.public_replays_per_generation,
                "selected_replays": public_replay_count,
            },
        }
        opponent_population: list[dict[str, object]] = []
        for spec, battle_count in zip(opponent_specs, allocations, strict=True):
            population_row = {
                **dict(spec),
                "battle_count": battle_count,
            }
            opponent_population.append(population_row)
            if battle_count == 0:
                continue
            opponent = _training_opponent(
                spec,
                concurrency=args.concurrency,
                showdown_root=args.showdown_root,
                belief_timeout=args.belief_timeout,
                search_policy_margin=args.battle_search_policy_margin,
            )
            trace_source = {
                "kind": f"generated-{spec['kind']}",
                "generation": generation_number,
                "opponent_kind": spec["kind"],
                **(
                    {"opponent_checkpoint_digest": spec["checkpoint_digest"]}
                    if isinstance(spec.get("checkpoint_digest"), str)
                    else {}
                ),
            }
            await run_local(
                battle_count,
                args.concurrency,
                results,
                decisions,
                replays,
                showdown_root=args.showdown_root,
                belief_timeout=args.belief_timeout,
                evaluator_checkpoint=current_checkpoint,
                search_policy_margin=args.battle_search_policy_margin,
                opponent=opponent,
                trace_source=trace_source,
                mode=f"training:{spec['kind']}",
                print_summary=False,
            )
        traces.append(decisions)

        public_curriculum: dict[str, object]
        if public_replay_count > 0:
            public_root = generation_root / "public-replays"
            try:
                public_result = await asyncio.to_thread(
                    import_public_replays,
                    showdown_root=args.showdown_root,
                    output_root=public_root,
                    max_battles=public_replay_count,
                    min_rating=args.public_replay_min_rating,
                    before=public_before,
                    strict=False,
                )
                public_curriculum = {
                    "status": "acquired",
                    **public_result,
                }
                next_before = public_result.get("next_before")
                if isinstance(next_before, int) and not isinstance(next_before, bool):
                    public_before = next_before
                public_trace = public_result.get("trace")
                if (
                    isinstance(public_trace, str)
                    and int(public_result.get("decision_count", 0)) > 0
                ):
                    traces.append(Path(public_trace))
            except (PublicReplayError, OSError, ValueError) as error:
                public_curriculum = {
                    "status": "unavailable",
                    "reason": type(error).__name__,
                    "detail": str(error)[-1000:],
                    "before": public_before,
                }
        else:
            public_curriculum = {
                "status": (
                    "disabled"
                    if args.public_replays_per_generation == 0
                    else "deferred-by-competence-debt"
                ),
                "selected_replays": public_replay_count,
            }

        receipt = run_self_improvement_cycle(
            traces,
            showdown_root=args.showdown_root,
            incumbent_checkpoint=current_checkpoint,
            workspace=args.workspace,
            models_dir=args.models_dir,
            receipts_dir=args.receipts_dir,
            promotion_file=args.promotion,
            teacher_compute_budget=args.teacher_budget,
            max_teacher_fixtures=args.max_teacher_fixtures,
            challenger_uncertainty_threshold=args.teacher_challenger_uncertainty,
            teacher_timeout_seconds=args.belief_timeout,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            policy_weight=args.policy_weight,
            value_target_source=args.value_target,
            admission_policy=AdmissionPolicy(
                min_validation_total_improvement=args.min_validation_improvement,
                max_validation_value_mse_regression=args.max_validation_value_regression,
                max_validation_policy_cross_entropy_regression=(
                    args.max_validation_policy_regression
                ),
                max_validation_posterior_stress_regression=(
                    args.max_validation_posterior_stress_regression
                ),
            ),
            defer_promotion=True,
        )

        next_competence_ledger = _competence_ledger_from_teacher_manifest(
            receipt.get("teacher_manifest")
        )
        next_competence_ledger_manifest = (
            str(receipt.get("teacher_manifest"))
            if isinstance(receipt.get("teacher_manifest"), str)
            else None
        )

        promoted = False
        promotion_panel: dict[str, Any] | None = None
        promotion_settlement: dict[str, Any] | None = None
        if receipt.get("status") == "candidate-admitted":
            improvement = receipt.get("improvement")
            if not isinstance(improvement, Mapping):
                raise ValueError("candidate admission lacks improvement evidence")
            candidate_checkpoint = improvement.get("candidate_checkpoint")
            if not isinstance(candidate_checkpoint, str) or not candidate_checkpoint:
                raise ValueError("candidate admission lacks immutable checkpoint")
            candidate_path = Path(candidate_checkpoint)
            promotion_panel = await run_promotion_panel(
                root=generation_root / "promotion-panel",
                candidate_checkpoint=candidate_path,
                incumbent_checkpoint=current_checkpoint,
                showdown_root=args.showdown_root,
                belief_timeout=args.belief_timeout,
                search_policy_margin=args.battle_search_policy_margin,
                concurrency=args.concurrency,
                policy=BattlePromotionPolicy(
                    expected_battles=args.promotion_battles,
                    max_superiority_p_value=args.promotion_alpha,
                ),
            )
            if promotion_panel["admitted"]:
                promotion_settlement = promote_deferred_candidate(
                    improvement,
                    battle_evidence=promotion_panel,
                    promotion_file=args.promotion,
                    receipts_dir=args.receipts_dir,
                )
                promoted = True
                archive_checkpoints.append(current_checkpoint)
                current_checkpoint = candidate_path

        generation = {
            "schema": "azelficoast.self-improvement-generation",
            "schema_version": 4,
            "generation": generation_number,
            "battle_count": args.battles_per_generation,
            "battle_search_policy_margin": args.battle_search_policy_margin,
            "opponent_population": opponent_population,
            "acquisition_plan": acquisition_plan,
            "public_curriculum": public_curriculum,
            "promotion_panel": promotion_panel,
            "promotion_settlement_receipt_digest": (
                promotion_settlement.get("receipt_digest")
                if promotion_settlement is not None
                else None
            ),
            "trace": str(decisions),
            "trace_digest": file_sha256(decisions),
            "results": str(results),
            "results_digest": file_sha256(results),
            "cycle_id": receipt.get("cycle_id"),
            "cycle_status": receipt.get("status"),
            "teacher_manifest": receipt.get("teacher_manifest"),
            "dataset_digest": receipt.get("dataset_digest"),
            "incumbent_checkpoint_digest": (
                receipt.get("inputs", {}).get("incumbent_checkpoint_digest")
                if isinstance(receipt.get("inputs"), dict)
                else None
            ),
            "promoted_checkpoint_digest": (
                receipt.get("improvement", {}).get("candidate_checkpoint_digest")
                if promoted and isinstance(receipt.get("improvement"), dict)
                else None
            ),
            "improvement_receipt_digest": (
                receipt.get("improvement", {}).get("receipt_digest")
                if isinstance(receipt.get("improvement"), dict)
                else None
            ),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(generation, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        generations.append(generation)
        competence_ledger = next_competence_ledger
        competence_ledger_manifest = next_competence_ledger_manifest

    return {
        "schema": "azelficoast.self-improvement-run",
        "schema_version": 4,
        "generation_count": len(generations),
        "generations": generations,
        "final_checkpoint": str(current_checkpoint),
        "archived_checkpoint_count": len(archive_checkpoints),
    }


