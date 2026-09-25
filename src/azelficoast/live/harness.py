"""Run Azelficoast battle and offline evaluation workflows."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import Player, RandomPlayer

from azelficoast.belief.coverage import summarize_traces
from azelficoast.belief.improvement import (
    VALUE_TARGET_SOURCES,
    AdmissionPolicy,
    improve_checkpoint,
)
from azelficoast.belief.self_improvement import run_self_improvement_cycle
from azelficoast.corpus import BUILTIN_POLICIES, build_corpus, evaluate_corpus
from azelficoast.live.player import AzelficoastPlayer
from azelficoast.research.training_records import build_training_dataset

BATTLE_FORMAT = "gen9randombattle"
DEFAULT_RESULTS = Path("artifacts/results.jsonl")
DEFAULT_DECISIONS = Path("artifacts/decisions.jsonl")
DEFAULT_REPLAYS = Path("artifacts/replays")
DEFAULT_CORPUS = Path("artifacts/corpus.jsonl")
DEFAULT_TRAINING = Path("artifacts/training.jsonl")
DEFAULT_EVALUATOR_MODELS = Path("artifacts/evaluators/candidates")
DEFAULT_EVALUATOR_RECEIPTS = Path("artifacts/evaluators/receipts")
DEFAULT_EVALUATOR_PROMOTION = Path("artifacts/evaluators/current.json")
DEFAULT_SELF_IMPROVEMENT = Path("artifacts/self-improvement")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="azelficoast",
        description="Pokemon Showdown evaluation harness for Azelficoast.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"append one JSON record per battle (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--decisions",
        type=Path,
        default=DEFAULT_DECISIONS,
        help=f"append protocol and decision evidence (default: {DEFAULT_DECISIONS})",
    )
    parser.add_argument(
        "--replays",
        type=Path,
        default=DEFAULT_REPLAYS,
        help=f"save replay HTML here (default: {DEFAULT_REPLAYS})",
    )
    parser.add_argument(
        "--showdown-root",
        type=Path,
        default=(Path(os.environ["AZELFICOAST_SHOWDOWN_ROOT"]) if os.getenv("AZELFICOAST_SHOWDOWN_ROOT") else None),
        help=(
            "built pinned Pokémon Showdown checkout for bounded live public-belief "
            "search; defaults to AZELFICOAST_SHOWDOWN_ROOT"
        ),
    )
    parser.add_argument(
        "--belief-timeout",
        type=_positive_float,
        default=float(os.getenv("AZELFICOAST_BELIEF_TIMEOUT_SECONDS", "20")),
        help="maximum seconds for one live public-belief probe (default: 20)",
    )
    parser.add_argument(
        "--evaluator-checkpoint",
        type=Path,
        default=(
            Path(os.environ["AZELFICOAST_EVALUATOR_CHECKPOINT"])
            if os.getenv("AZELFICOAST_EVALUATOR_CHECKPOINT")
            else None
        ),
        help=(
            "verified learned policy/value checkpoint; without this, live behavior "
            "remains exact public-belief search"
        ),
    )
    parser.add_argument(
        "--search-policy-margin",
        type=_unit_float,
        default=float(os.getenv("AZELFICOAST_SEARCH_POLICY_MARGIN", "1.0")),
        help=(
            "run exact search when learned top-two policy margin is at or below this "
            "threshold; 1.0 is conservative shadow mode (default: 1.0)"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local", help="battle a baseline on localhost:8000")
    local.add_argument("--battles", type=_positive_int, default=1)
    local.add_argument(
        "--concurrency",
        type=_positive_int,
        default=1,
        help="maximum simultaneous local battles (default: 1)",
    )

    challenge = subparsers.add_parser(
        "challenge",
        help="challenge a named user on the official Pokemon Showdown server",
    )
    challenge.add_argument("opponent")
    challenge.add_argument("--battles", type=_positive_int, default=1)
    challenge.add_argument(
        "--username",
        help="Showdown account name; defaults to SHOWDOWN_USERNAME",
    )

    accept = subparsers.add_parser(
        "accept",
        help="accept official Showdown challenges from one user or anyone",
    )
    accept.add_argument("--opponent", help="only accept this username; omit for anyone")
    accept.add_argument("--battles", type=_positive_int, default=1)
    accept.add_argument(
        "--username",
        help="Showdown account name; defaults to SHOWDOWN_USERNAME",
    )

    ladder = subparsers.add_parser("ladder", help="play on the official Showdown ladder")
    ladder.add_argument("--battles", type=_positive_int, default=1)
    ladder.add_argument(
        "--username",
        help="Showdown account name; defaults to SHOWDOWN_USERNAME",
    )

    corpus = subparsers.add_parser(
        "corpus",
        help="build or evaluate replayable frozen decision fixtures",
    )
    corpus_commands = corpus.add_subparsers(dest="corpus_command", required=True)

    corpus_build = corpus_commands.add_parser(
        "build",
        help="extract immutable fixtures from one or more decision traces",
    )
    corpus_build.add_argument("traces", nargs="+", type=Path)
    corpus_build.add_argument("--output", type=Path, default=DEFAULT_CORPUS)

    corpus_evaluate = corpus_commands.add_parser(
        "evaluate",
        help="run a deterministic policy over every frozen fixture",
    )
    corpus_evaluate.add_argument("corpus_path", type=Path)
    corpus_evaluate.add_argument(
        "--policy",
        required=True,
        help=(
            "built-in policy "
            f"({', '.join(sorted(BUILTIN_POLICIES))}) or local module:object"
        ),
    )
    corpus_evaluate.add_argument(
        "--output",
        type=Path,
        help="optional JSONL file for per-fixture evaluation results",
    )

    training = subparsers.add_parser(
        "training",
        help="build leakage-safe policy/value records from real decision traces",
    )
    training_commands = training.add_subparsers(
        dest="training_command",
        required=True,
    )
    training_build = training_commands.add_parser(
        "build",
        help="join real decision states with outcome and public-belief search targets",
    )
    training_build.add_argument("traces", nargs="+", type=Path)
    training_build.add_argument("--output", type=Path, default=DEFAULT_TRAINING)
    training_build.add_argument(
        "--search-packet",
        action="append",
        type=Path,
        default=[],
        help="frozen matched-search packet; repeat for multiple decisions",
    )
    training_build.add_argument(
        "--search-receipt",
        action="append",
        type=Path,
        default=[],
        help="matched-search method receipt; provide both methods for each packet",
    )
    training_build.add_argument(
        "--posterior",
        action="append",
        type=Path,
        default=[],
        help="posterior artifact referenced by a search packet; repeat as needed",
    )
    training_build.add_argument(
        "--split-seed",
        default="azelficoast.training-records",
    )
    training_build.add_argument("--train-fraction", type=_unit_float, default=0.8)
    training_build.add_argument(
        "--validation-fraction",
        type=_unit_float,
        default=0.1,
    )

    training_improve = training_commands.add_parser(
        "improve",
        help="train, evaluate, and evidence-gate one evaluator candidate",
    )
    training_improve.add_argument("dataset", type=Path)
    training_improve.add_argument(
        "--incumbent",
        type=Path,
        default=DEFAULT_EVALUATOR_PROMOTION,
        help="checkpoint directory or digest-bound promotion pointer",
    )
    training_improve.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_MODELS,
    )
    training_improve.add_argument(
        "--receipts-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_RECEIPTS,
    )
    training_improve.add_argument(
        "--promotion",
        type=Path,
        default=DEFAULT_EVALUATOR_PROMOTION,
        help="atomic pointer updated only when the candidate is admitted",
    )
    training_improve.add_argument("--epochs", type=_positive_int, default=1)
    training_improve.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    training_improve.add_argument("--policy-weight", type=_nonnegative_float, default=1.0)
    training_improve.add_argument(
        "--value-target",
        choices=VALUE_TARGET_SOURCES,
        default="public_belief_search_return",
    )
    training_improve.add_argument(
        "--min-validation-improvement",
        type=_nonnegative_float,
        default=1e-6,
    )
    training_improve.add_argument(
        "--max-validation-value-regression",
        type=_nonnegative_float,
        default=0.0,
    )
    training_improve.add_argument(
        "--max-validation-policy-regression",
        type=_nonnegative_float,
        default=0.0,
    )

    training_cycle = training_commands.add_parser(
        "cycle",
        help="derive teacher evidence from completed traces and run one gated model update",
    )
    training_cycle.add_argument("traces", nargs="+", type=Path)
    training_cycle.add_argument(
        "--incumbent",
        type=Path,
        default=DEFAULT_EVALUATOR_PROMOTION,
        help="checkpoint directory or digest-bound promotion pointer",
    )
    training_cycle.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_SELF_IMPROVEMENT,
    )
    training_cycle.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_MODELS,
    )
    training_cycle.add_argument(
        "--receipts-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_RECEIPTS,
    )
    training_cycle.add_argument(
        "--promotion",
        type=Path,
        default=DEFAULT_EVALUATOR_PROMOTION,
    )
    training_cycle.add_argument("--teacher-budget", type=_positive_int, default=4096)
    training_cycle.add_argument(
        "--max-teacher-fixtures",
        type=_positive_int,
        help="mine at most this many battle-diverse informative fixtures per cycle",
    )
    training_cycle.add_argument(
        "--split-seed",
        default="azelficoast.training-records",
    )
    training_cycle.add_argument("--train-fraction", type=_unit_float, default=0.8)
    training_cycle.add_argument("--validation-fraction", type=_unit_float, default=0.1)
    training_cycle.add_argument("--epochs", type=_positive_int, default=1)
    training_cycle.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    training_cycle.add_argument("--policy-weight", type=_nonnegative_float, default=1.0)
    training_cycle.add_argument(
        "--value-target",
        choices=VALUE_TARGET_SOURCES,
        default="public_belief_search_return",
    )
    training_cycle.add_argument(
        "--min-validation-improvement",
        type=_nonnegative_float,
        default=1e-6,
    )
    training_cycle.add_argument(
        "--max-validation-value-regression",
        type=_nonnegative_float,
        default=0.0,
    )
    training_cycle.add_argument(
        "--max-validation-policy-regression",
        type=_nonnegative_float,
        default=0.0,
    )

    training_auto = training_commands.add_parser(
        "auto",
        help="generate local battles and repeat the evidence-gated improvement cycle",
    )
    training_auto.add_argument(
        "--incumbent",
        type=Path,
        default=DEFAULT_EVALUATOR_PROMOTION,
        help="initial checkpoint directory or digest-bound promotion pointer",
    )
    training_auto.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_SELF_IMPROVEMENT,
    )
    training_auto.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_MODELS,
    )
    training_auto.add_argument(
        "--receipts-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_RECEIPTS,
    )
    training_auto.add_argument(
        "--promotion",
        type=Path,
        default=DEFAULT_EVALUATOR_PROMOTION,
    )
    training_auto.add_argument("--generations", type=_positive_int, default=1)
    training_auto.add_argument("--battles-per-generation", type=_positive_int, default=12)
    training_auto.add_argument("--concurrency", type=_positive_int, default=1)
    training_auto.add_argument("--teacher-budget", type=_positive_int, default=4096)
    training_auto.add_argument(
        "--max-teacher-fixtures",
        type=_positive_int,
        default=64,
        help="battle-diverse curriculum budget per candidate generation",
    )
    training_auto.add_argument(
        "--split-seed",
        default="azelficoast.training-records",
    )
    training_auto.add_argument("--train-fraction", type=_unit_float, default=0.8)
    training_auto.add_argument("--validation-fraction", type=_unit_float, default=0.1)
    training_auto.add_argument("--epochs", type=_positive_int, default=1)
    training_auto.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    training_auto.add_argument("--policy-weight", type=_nonnegative_float, default=1.0)
    training_auto.add_argument(
        "--value-target",
        choices=VALUE_TARGET_SOURCES,
        default="public_belief_search_return",
    )
    training_auto.add_argument(
        "--min-validation-improvement",
        type=_nonnegative_float,
        default=1e-6,
    )
    training_auto.add_argument(
        "--max-validation-value-regression",
        type=_nonnegative_float,
        default=0.0,
    )
    training_auto.add_argument(
        "--max-validation-policy-regression",
        type=_nonnegative_float,
        default=0.0,
    )

    coverage = subparsers.add_parser(
        "belief-coverage",
        help="summarize live public-belief routing and static admission coverage",
    )
    coverage.add_argument("traces", nargs="+", type=Path)

    return parser


def _resolve_live_credentials(username_override: str | None) -> tuple[str, str]:
    username = username_override or os.getenv("SHOWDOWN_USERNAME")
    password = os.getenv("SHOWDOWN_PASSWORD")
    missing = [
        name
        for name, value in (
            ("SHOWDOWN_USERNAME (or --username)", username),
            ("SHOWDOWN_PASSWORD", password),
        )
        if not value
    ]
    if missing:
        raise ValueError("missing live Showdown credentials: " + ", ".join(missing))
    return username, password


def _prepare_output_paths(results: Path, decisions: Path, replays: Path) -> None:
    results.parent.mkdir(parents=True, exist_ok=True)
    decisions.parent.mkdir(parents=True, exist_ok=True)
    replays.mkdir(parents=True, exist_ok=True)


def _live_player(
    username: str,
    password: str,
    replays: Path,
    decisions: Path,
    *,
    showdown_root: Path | None,
    belief_timeout: float,
    evaluator_checkpoint: Path | None,
    search_policy_margin: float,
) -> AzelficoastPlayer:
    return AzelficoastPlayer(
        account_configuration=AccountConfiguration(username, password),
        server_configuration=ShowdownServerConfiguration,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
        save_replays=str(replays),
        decision_log=decisions,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
        evaluator_checkpoint=evaluator_checkpoint,
        search_policy_margin=search_policy_margin,
    )


def _append_results(player: Player, path: Path, mode: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for battle in player.battles.values():
            record = {
                "observed_at": datetime.now(UTC).isoformat(),
                "mode": mode,
                "format": BATTLE_FORMAT,
                "battle_tag": battle.battle_tag,
                "players": list(battle.players),
                "player": battle.player_username,
                "won": battle.won,
                "lost": battle.lost,
                "rating": battle.rating,
                "opponent_rating": battle.opponent_rating,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _print_summary(player: Player) -> None:
    print(
        json.dumps(
            {
                "player": player.username,
                "finished": player.n_finished_battles,
                "won": player.n_won_battles,
                "lost": player.n_lost_battles,
                "tied": player.n_tied_battles,
            },
            sort_keys=True,
        )
    )


async def _run_local(
    battles: int,
    concurrency: int,
    results: Path,
    decisions: Path,
    replays: Path,
    *,
    showdown_root: Path | None,
    belief_timeout: float,
    evaluator_checkpoint: Path | None,
    search_policy_margin: float,
) -> None:
    player = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        save_replays=str(replays),
        decision_log=decisions,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
        evaluator_checkpoint=evaluator_checkpoint,
        search_policy_margin=search_policy_margin,
    )
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
    )
    await player.battle_against(opponent, n_battles=battles)
    _append_results(player, results, mode="local")
    _print_summary(player)


async def _run_live(args: argparse.Namespace) -> None:
    username, password = _resolve_live_credentials(args.username)
    player = _live_player(
        username,
        password,
        args.replays,
        args.decisions,
        showdown_root=args.showdown_root,
        belief_timeout=args.belief_timeout,
        evaluator_checkpoint=args.evaluator_checkpoint,
        search_policy_margin=args.search_policy_margin,
    )

    if args.command == "challenge":
        await player.send_challenges(args.opponent, n_challenges=args.battles)
        mode = f"challenge:{args.opponent}"
    elif args.command == "accept":
        await player.accept_challenges(args.opponent, n_challenges=args.battles)
        mode = f"accept:{args.opponent or '*'}"
    elif args.command == "ladder":
        await player.ladder(args.battles)
        mode = "ladder"
    else:
        raise AssertionError(f"unsupported live command: {args.command}")

    _append_results(player, args.results, mode=mode)
    _print_summary(player)


async def _async_main(args: argparse.Namespace) -> None:
    _prepare_output_paths(args.results, args.decisions, args.replays)
    if args.command == "local":
        await _run_local(
            args.battles,
            args.concurrency,
            args.results,
            args.decisions,
            args.replays,
            showdown_root=args.showdown_root,
            belief_timeout=args.belief_timeout,
            evaluator_checkpoint=args.evaluator_checkpoint,
            search_policy_margin=args.search_policy_margin,
        )
    else:
        await _run_live(args)


def _run_corpus(args: argparse.Namespace) -> None:
    if args.corpus_command == "build":
        summary = build_corpus(args.traces, args.output)
    elif args.corpus_command == "evaluate":
        summary = evaluate_corpus(args.corpus_path, args.policy, args.output)
    else:
        raise AssertionError(f"unsupported corpus command: {args.corpus_command}")
    print(json.dumps(summary, sort_keys=True))


async def _run_automatic_self_improvement(args: argparse.Namespace) -> dict[str, object]:
    """Generate fresh Random Battles, mine them, and attempt bounded promotions."""

    if args.showdown_root is None:
        raise ValueError(
            "automatic training requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
        )

    traces: list[Path] = []
    current_checkpoint = args.incumbent
    generations: list[dict[str, object]] = []

    for index in range(args.generations):
        generation_root = args.workspace / "generations" / f"{index + 1:04d}"
        decisions = generation_root / "decisions.jsonl"
        results = generation_root / "results.jsonl"
        replays = generation_root / "replays"
        manifest_path = generation_root / "generation.json"
        if decisions.exists() or results.exists() or manifest_path.exists():
            raise ValueError(
                f"generation output already exists and is immutable: {generation_root}"
            )

        _prepare_output_paths(results, decisions, replays)
        await _run_local(
            args.battles_per_generation,
            args.concurrency,
            results,
            decisions,
            replays,
            showdown_root=args.showdown_root,
            belief_timeout=args.belief_timeout,
            evaluator_checkpoint=current_checkpoint,
            search_policy_margin=args.search_policy_margin,
        )
        traces.append(decisions)

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
            ),
        )
        promoted = receipt.get("status") == "promoted"
        if promoted:
            current_checkpoint = args.promotion

        generation = {
            "schema": "azelficoast.self-improvement-generation",
            "schema_version": 1,
            "generation": index + 1,
            "battle_count": args.battles_per_generation,
            "trace": str(decisions),
            "results": str(results),
            "cycle_id": receipt.get("cycle_id"),
            "cycle_status": receipt.get("status"),
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
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(generation, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        generations.append(generation)

    return {
        "schema": "azelficoast.self-improvement-run",
        "schema_version": 1,
        "generation_count": len(generations),
        "generations": generations,
        "final_checkpoint": str(current_checkpoint),
    }


def _run_training(args: argparse.Namespace) -> None:
    if args.training_command == "build":
        summary = build_training_dataset(
            args.traces,
            args.output,
            search_packet_paths=args.search_packet,
            search_receipt_paths=args.search_receipt,
            posterior_paths=args.posterior,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
    elif args.training_command == "improve":
        summary = improve_checkpoint(
            args.dataset,
            incumbent_checkpoint=args.incumbent,
            models_dir=args.models_dir,
            receipts_dir=args.receipts_dir,
            promotion_file=args.promotion,
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
            ),
        )
    elif args.training_command == "cycle":
        if args.showdown_root is None:
            raise ValueError(
                "training cycle requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
            )
        summary = run_self_improvement_cycle(
            args.traces,
            showdown_root=args.showdown_root,
            incumbent_checkpoint=args.incumbent,
            workspace=args.workspace,
            models_dir=args.models_dir,
            receipts_dir=args.receipts_dir,
            promotion_file=args.promotion,
            teacher_compute_budget=args.teacher_budget,
            max_teacher_fixtures=args.max_teacher_fixtures,
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
            ),
        )
    elif args.training_command == "auto":
        summary = asyncio.run(_run_automatic_self_improvement(args))
    else:
        raise AssertionError(f"unsupported training command: {args.training_command}")
    print(json.dumps(summary, sort_keys=True))


def _run_belief_coverage(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_traces(args.traces), sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "corpus":
            _run_corpus(args)
        elif args.command == "training":
            _run_training(args)
        elif args.command == "belief-coverage":
            _run_belief_coverage(args)
        else:
            asyncio.run(_async_main(args))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
