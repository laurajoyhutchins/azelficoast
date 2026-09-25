"""Run Azelficoast battle and offline evaluation workflows."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import MaxBasePowerPlayer, Player, RandomPlayer, SimpleHeuristicsPlayer

from azelficoast.belief.coverage import summarize_traces
from azelficoast.belief.evaluator import BeliefEvaluatorRuntime
from azelficoast.belief.battle_promotion import (
    BattlePromotionPolicy,
    CANDIDATE_PRIMARY_MODE,
    INCUMBENT_PRIMARY_MODE,
    settle_battle_panel,
)
from azelficoast.belief.improvement import (
    VALUE_TARGET_SOURCES,
    AdmissionPolicy,
    improve_checkpoint,
    promote_deferred_candidate,
)
from azelficoast.belief.public_pretraining import run_public_pretraining
from azelficoast.belief.self_improvement import (
    DEFAULT_CHALLENGER_UNCERTAINTY_THRESHOLD,
    run_self_improvement_cycle,
)
from azelficoast.live.corpus import BUILTIN_POLICIES, build_corpus, evaluate_corpus
from azelficoast.live.opponents import DirtyTricksPlayer
from azelficoast.research.public_replays import PublicReplayError, import_public_replays
from azelficoast.live.player import AzelficoastPlayer
from azelficoast.research.training_records import build_training_dataset

BATTLE_FORMAT = "gen9randombattle"
DEFAULT_RESULTS = Path("artifacts/results.jsonl")
DEFAULT_DECISIONS = Path("artifacts/decisions.jsonl")
DEFAULT_REPLAYS = Path("artifacts/replays")
DEFAULT_CORPUS = Path("artifacts/corpus.jsonl")
DEFAULT_PUBLIC_CORPUS = Path("artifacts/public-replays")
DEFAULT_TRAINING = Path("artifacts/training.jsonl")
DEFAULT_PUBLIC_PRETRAINING = Path("artifacts/public-pretraining.jsonl")
DEFAULT_EVALUATOR_MODELS = Path("artifacts/evaluators/candidates")
DEFAULT_EVALUATOR_RECEIPTS = Path("artifacts/evaluators/receipts")
DEFAULT_EVALUATOR_PROMOTION = Path("artifacts/evaluators/current.json")
DEFAULT_SELF_IMPROVEMENT = Path("artifacts/self-improvement")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
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


def _open_unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be within (0, 1)")
    return parsed


def _positive_even_int(value: str) -> int:
    parsed = _positive_int(value)
    if parsed % 2 != 0:
        raise argparse.ArgumentTypeError("must be even")
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

    corpus_import = corpus_commands.add_parser(
        "import-public",
        help="freeze public Gen 9 Random Battle replays as player-perspective traces",
    )
    corpus_import.add_argument("--output-root", type=Path, default=DEFAULT_PUBLIC_CORPUS)
    corpus_import.add_argument("--max-battles", type=_positive_int, default=100)
    corpus_import.add_argument("--min-rating", type=int, default=0)
    corpus_import.add_argument(
        "--before",
        type=_positive_int,
        help="optional Showdown replay-search uploadtime cursor",
    )
    corpus_import.add_argument(
        "--strict",
        action="store_true",
        help="fail the import on the first unreconstructible replay instead of recording an exclusion",
    )

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

    training_bootstrap_public = training_commands.add_parser(
        "bootstrap-public",
        help="cold-start an evaluator from public human Random Battle decisions",
    )
    training_bootstrap_public.add_argument("traces", nargs="+", type=Path)
    training_bootstrap_public.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PUBLIC_PRETRAINING,
        help="frozen public imitation/value training dataset",
    )
    training_bootstrap_public.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_MODELS,
    )
    training_bootstrap_public.add_argument(
        "--receipts-dir",
        type=Path,
        default=DEFAULT_EVALUATOR_RECEIPTS,
    )
    training_bootstrap_public.add_argument(
        "--promotion",
        type=Path,
        default=DEFAULT_EVALUATOR_PROMOTION,
    )
    training_bootstrap_public.add_argument(
        "--split-seed",
        default="azelficoast.training-records",
    )
    training_bootstrap_public.add_argument("--train-fraction", type=_unit_float, default=0.8)
    training_bootstrap_public.add_argument(
        "--validation-fraction",
        type=_unit_float,
        default=0.1,
    )
    training_bootstrap_public.add_argument("--seed", type=int, default=0)
    training_bootstrap_public.add_argument("--epochs", type=_positive_int, default=1)
    training_bootstrap_public.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=3e-4,
    )
    training_bootstrap_public.add_argument(
        "--policy-weight",
        type=_nonnegative_float,
        default=1.0,
    )
    training_bootstrap_public.add_argument(
        "--min-validation-improvement",
        type=_nonnegative_float,
        default=1e-6,
    )
    training_bootstrap_public.add_argument(
        "--max-validation-value-regression",
        type=_nonnegative_float,
        default=0.0,
    )
    training_bootstrap_public.add_argument(
        "--max-validation-policy-regression",
        type=_nonnegative_float,
        default=0.0,
    )
    training_bootstrap_public.add_argument(
        "--max-validation-posterior-stress-regression",
        type=_nonnegative_float,
        default=0.0,
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
    training_improve.add_argument(
        "--max-validation-posterior-stress-regression",
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
        "--teacher-challenger-uncertainty",
        type=_unit_float,
        default=DEFAULT_CHALLENGER_UNCERTAINTY_THRESHOLD,
        help=(
            "run posterior-stress challenger teachers when public policy uncertainty "
            "reaches this threshold; search/fallback states always challenge"
        ),
    )
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
    training_cycle.add_argument(
        "--max-validation-posterior-stress-regression",
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
    training_auto.add_argument(
        "--public-replays-per-generation",
        type=_nonnegative_int,
        default=4,
        help=(
            "replenish the curriculum from this many public Random Battle replays "
            "per generation; use 0 to disable network acquisition"
        ),
    )
    training_auto.add_argument(
        "--public-replay-min-rating",
        type=_nonnegative_int,
        default=1500,
        help="minimum public replay rating admitted to automatic curriculum acquisition",
    )
    training_auto.add_argument(
        "--battle-search-policy-margin",
        type=_unit_float,
        default=0.0,
        help=(
            "exact-search routing threshold while generating battles; defaults to 0 "
            "because mined teacher states are searched offline after generation"
        ),
    )
    training_auto.add_argument(
        "--promotion-battles",
        type=_positive_even_int,
        default=32,
        help="side-balanced candidate-vs-incumbent battles required before promotion",
    )
    training_auto.add_argument(
        "--promotion-alpha",
        type=_open_unit_float,
        default=0.10,
        help="maximum one-sided exact superiority p-value for battle promotion",
    )
    training_auto.add_argument("--teacher-budget", type=_positive_int, default=4096)
    training_auto.add_argument(
        "--teacher-challenger-uncertainty",
        type=_unit_float,
        default=DEFAULT_CHALLENGER_UNCERTAINTY_THRESHOLD,
        help=(
            "run posterior-stress challenger teachers when public policy uncertainty "
            "reaches this threshold; search/fallback states always challenge"
        ),
    )
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
    training_auto.add_argument(
        "--max-validation-posterior-stress-regression",
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _prepare_output_paths(results: Path, decisions: Path, replays: Path) -> None:
    results.parent.mkdir(parents=True, exist_ok=True)
    decisions.parent.mkdir(parents=True, exist_ok=True)
    replays.mkdir(parents=True, exist_ok=True)


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


def _checkpoint_digest(path: Path) -> str:
    runtime = BeliefEvaluatorRuntime.from_checkpoint(path)
    return str(runtime.identity["checkpoint_digest"])


def _balanced_battle_allocation(
    battle_count: int,
    opponent_count: int,
    *,
    rotation: int = 0,
) -> tuple[int, ...]:
    if battle_count <= 0 or opponent_count <= 0:
        raise ValueError("battle and opponent counts must be positive")
    if not isinstance(rotation, int) or isinstance(rotation, bool):
        raise ValueError("battle allocation rotation must be an integer")
    quotient, remainder = divmod(battle_count, opponent_count)
    allocation = [quotient for _ in range(opponent_count)]
    start = rotation % opponent_count
    for offset in range(remainder):
        allocation[(start + offset) % opponent_count] += 1
    return tuple(allocation)


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
            "checkpoint_digest": _checkpoint_digest(current_checkpoint),
        },
    ]
    for offset, checkpoint in enumerate(reversed(archive_checkpoints[-3:]), start=1):
        specs.append(
            {
                "kind": "archive",
                "archive_recency": offset,
                "checkpoint": str(checkpoint),
                "checkpoint_digest": _checkpoint_digest(checkpoint),
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
    common = {
        "battle_format": BATTLE_FORMAT,
        "max_concurrent_battles": concurrency,
    }
    if kind == "max-base-power":
        return MaxBasePowerPlayer(**common)
    if kind == "simple-heuristics":
        return SimpleHeuristicsPlayer(**common)
    if kind == "dirty-tricks":
        return DirtyTricksPlayer(**common)
    if kind in {"incumbent", "archive"}:
        checkpoint = spec.get("checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError(f"{kind} opponent lacks a checkpoint")
        return AzelficoastPlayer(
            **common,
            showdown_root=showdown_root,
            belief_timeout_seconds=belief_timeout,
            evaluator_checkpoint=Path(checkpoint),
            search_policy_margin=search_policy_margin,
        )
    raise ValueError(f"unsupported training opponent kind {kind!r}")


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


def _append_results(
    player: Player,
    path: Path,
    mode: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
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
                **(dict(metadata) if metadata is not None else {}),
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
    opponent: Player | None = None,
    trace_source: Mapping[str, Any] | None = None,
    mode: str = "local",
    print_summary: bool = True,
    result_metadata: Mapping[str, Any] | None = None,
) -> None:
    player = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        save_replays=str(replays),
        decision_log=decisions,
        trace_source=trace_source,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
        evaluator_checkpoint=evaluator_checkpoint,
        search_policy_margin=search_policy_margin,
    )
    if opponent is None:
        opponent = RandomPlayer(
            battle_format=BATTLE_FORMAT,
            max_concurrent_battles=concurrency,
        )
    await player.battle_against(opponent, n_battles=battles)
    _append_results(player, results, mode=mode, metadata=result_metadata)
    if print_summary:
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
    elif args.corpus_command == "import-public":
        if args.showdown_root is None:
            raise ValueError(
                "public replay import requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
            )
        summary = import_public_replays(
            showdown_root=args.showdown_root,
            output_root=args.output_root,
            max_battles=args.max_battles,
            min_rating=args.min_rating,
            before=args.before,
            strict=args.strict,
        )
    elif args.corpus_command == "evaluate":
        summary = evaluate_corpus(args.corpus_path, args.policy, args.output)
    else:
        raise AssertionError(f"unsupported corpus command: {args.corpus_command}")
    print(json.dumps(summary, sort_keys=True))



def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


async def _run_promotion_panel(
    *,
    root: Path,
    candidate_checkpoint: Path,
    incumbent_checkpoint: Path,
    showdown_root: Path,
    belief_timeout: float,
    search_policy_margin: float,
    concurrency: int,
    policy: BattlePromotionPolicy,
) -> dict[str, Any]:
    """Run and settle a side-balanced candidate-vs-incumbent battle panel."""

    candidate_digest = _checkpoint_digest(candidate_checkpoint)
    incumbent_digest = _checkpoint_digest(incumbent_checkpoint)
    results = root / "results.jsonl"
    decisions = root / "decisions.jsonl"
    replays = root / "replays"
    evidence_path = root / "evidence.json"
    _prepare_output_paths(results, decisions, replays)

    common_metadata = {
        "candidate_checkpoint_digest": candidate_digest,
        "incumbent_checkpoint_digest": incumbent_digest,
    }
    half = policy.expected_battles // 2

    incumbent_opponent = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
        evaluator_checkpoint=incumbent_checkpoint,
        search_policy_margin=search_policy_margin,
    )
    await _run_local(
        half,
        concurrency,
        results,
        decisions,
        replays,
        showdown_root=showdown_root,
        belief_timeout=belief_timeout,
        evaluator_checkpoint=candidate_checkpoint,
        search_policy_margin=search_policy_margin,
        opponent=incumbent_opponent,
        trace_source={"kind": "promotion-candidate-primary"},
        mode=CANDIDATE_PRIMARY_MODE,
        print_summary=False,
        result_metadata=common_metadata,
    )

    candidate_opponent = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
        evaluator_checkpoint=candidate_checkpoint,
        search_policy_margin=search_policy_margin,
    )
    await _run_local(
        half,
        concurrency,
        results,
        decisions,
        replays,
        showdown_root=showdown_root,
        belief_timeout=belief_timeout,
        evaluator_checkpoint=incumbent_checkpoint,
        search_policy_margin=search_policy_margin,
        opponent=candidate_opponent,
        trace_source={"kind": "promotion-incumbent-primary"},
        mode=INCUMBENT_PRIMARY_MODE,
        print_summary=False,
        result_metadata=common_metadata,
    )

    evidence = settle_battle_panel(
        _read_jsonl_objects(results),
        candidate_checkpoint_digest=candidate_digest,
        incumbent_checkpoint_digest=incumbent_digest,
        policy=policy,
    )
    evidence = {
        **evidence,
        "results": str(results),
        "results_digest": _file_sha256(results),
        "decision_trace": str(decisions),
        "decision_trace_digest": _file_sha256(decisions),
    }
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return evidence


async def _run_automatic_self_improvement(args: argparse.Namespace) -> dict[str, object]:
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

        _prepare_output_paths(results, decisions, replays)
        opponent_specs = _training_opponent_specs(
            current_checkpoint,
            archive_checkpoints,
        )
        allocations = _balanced_battle_allocation(
            args.battles_per_generation,
            len(opponent_specs),
            rotation=index,
        )
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
            await _run_local(
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
        if args.public_replays_per_generation > 0:
            public_root = generation_root / "public-replays"
            try:
                public_result = await asyncio.to_thread(
                    import_public_replays,
                    showdown_root=args.showdown_root,
                    output_root=public_root,
                    max_battles=args.public_replays_per_generation,
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
            public_curriculum = {"status": "disabled"}

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
            promotion_panel = await _run_promotion_panel(
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
            "schema_version": 3,
            "generation": generation_number,
            "battle_count": args.battles_per_generation,
            "battle_search_policy_margin": args.battle_search_policy_margin,
            "opponent_population": opponent_population,
            "public_curriculum": public_curriculum,
            "promotion_panel": promotion_panel,
            "promotion_settlement_receipt_digest": (
                promotion_settlement.get("receipt_digest")
                if promotion_settlement is not None
                else None
            ),
            "trace": str(decisions),
            "trace_digest": _file_sha256(decisions),
            "results": str(results),
            "results_digest": _file_sha256(results),
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

    return {
        "schema": "azelficoast.self-improvement-run",
        "schema_version": 3,
        "generation_count": len(generations),
        "generations": generations,
        "final_checkpoint": str(current_checkpoint),
        "archived_checkpoint_count": len(archive_checkpoints),
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
    elif args.training_command == "bootstrap-public":
        if args.showdown_root is None:
            raise ValueError(
                "public pretraining requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
            )
        summary = run_public_pretraining(
            args.traces,
            showdown_root=args.showdown_root,
            dataset_path=args.output,
            models_dir=args.models_dir,
            receipts_dir=args.receipts_dir,
            promotion_file=args.promotion,
            posterior_timeout_seconds=args.belief_timeout,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            policy_weight=args.policy_weight,
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
                max_validation_posterior_stress_regression=(
                    args.max_validation_posterior_stress_regression
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
