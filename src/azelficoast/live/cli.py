"""Command-line schema for Azelficoast workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from azelficoast.belief.improvement import VALUE_TARGET_SOURCES
from azelficoast.belief.self_improvement import DEFAULT_CHALLENGER_UNCERTAINTY_THRESHOLD
from azelficoast.live.corpus import BUILTIN_POLICIES

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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be within [0, 1]")
    return parsed


def open_unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("must be within (0, 1)")
    return parsed


def positive_even_int(value: str) -> int:
    parsed = positive_int(value)
    if parsed % 2 != 0:
        raise argparse.ArgumentTypeError("must be even")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed



def _add_live_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    local = subparsers.add_parser("local", help="battle a baseline on localhost:8000")
    local.add_argument("--battles", type=positive_int, default=1)
    local.add_argument(
        "--concurrency",
        type=positive_int,
        default=1,
        help="maximum simultaneous local battles (default: 1)",
    )
    
    challenge = subparsers.add_parser(
        "challenge",
        help="challenge a named user on the official Pokemon Showdown server",
    )
    challenge.add_argument("opponent")
    challenge.add_argument("--battles", type=positive_int, default=1)
    challenge.add_argument(
        "--username",
        help="Showdown account name; defaults to SHOWDOWN_USERNAME",
    )
    
    accept = subparsers.add_parser(
        "accept",
        help="accept official Showdown challenges from one user or anyone",
    )
    accept.add_argument("--opponent", help="only accept this username; omit for anyone")
    accept.add_argument("--battles", type=positive_int, default=1)
    accept.add_argument(
        "--username",
        help="Showdown account name; defaults to SHOWDOWN_USERNAME",
    )
    
    ladder = subparsers.add_parser("ladder", help="play on the official Showdown ladder")
    ladder.add_argument("--battles", type=positive_int, default=1)
    ladder.add_argument(
        "--username",
        help="Showdown account name; defaults to SHOWDOWN_USERNAME",
    )
    
    


def _add_corpus_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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
    corpus_import.add_argument("--max-battles", type=positive_int, default=100)
    corpus_import.add_argument("--min-rating", type=int, default=0)
    corpus_import.add_argument(
        "--before",
        type=positive_int,
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
    
    


def _add_training_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
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
    training_build.add_argument("--train-fraction", type=unit_float, default=0.8)
    training_build.add_argument(
        "--validation-fraction",
        type=unit_float,
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
    training_bootstrap_public.add_argument("--train-fraction", type=unit_float, default=0.8)
    training_bootstrap_public.add_argument(
        "--validation-fraction",
        type=unit_float,
        default=0.1,
    )
    training_bootstrap_public.add_argument("--seed", type=int, default=0)
    training_bootstrap_public.add_argument("--epochs", type=positive_int, default=1)
    training_bootstrap_public.add_argument(
        "--learning-rate",
        type=positive_float,
        default=3e-4,
    )
    training_bootstrap_public.add_argument(
        "--policy-weight",
        type=nonnegative_float,
        default=1.0,
    )
    training_bootstrap_public.add_argument(
        "--min-validation-improvement",
        type=nonnegative_float,
        default=1e-6,
    )
    training_bootstrap_public.add_argument(
        "--max-validation-value-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_bootstrap_public.add_argument(
        "--max-validation-policy-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_bootstrap_public.add_argument(
        "--max-validation-posterior-stress-regression",
        type=nonnegative_float,
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
    training_improve.add_argument("--epochs", type=positive_int, default=1)
    training_improve.add_argument("--learning-rate", type=positive_float, default=3e-4)
    training_improve.add_argument("--policy-weight", type=nonnegative_float, default=1.0)
    training_improve.add_argument(
        "--value-target",
        choices=VALUE_TARGET_SOURCES,
        default="public_belief_search_return",
    )
    training_improve.add_argument(
        "--min-validation-improvement",
        type=nonnegative_float,
        default=1e-6,
    )
    training_improve.add_argument(
        "--max-validation-value-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_improve.add_argument(
        "--max-validation-policy-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_improve.add_argument(
        "--max-validation-posterior-stress-regression",
        type=nonnegative_float,
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
    training_cycle.add_argument("--teacher-budget", type=positive_int, default=4096)
    training_cycle.add_argument(
        "--teacher-challenger-uncertainty",
        type=unit_float,
        default=DEFAULT_CHALLENGER_UNCERTAINTY_THRESHOLD,
        help=(
            "run posterior-stress challenger teachers when public policy uncertainty "
            "reaches this threshold; search/fallback states always challenge"
        ),
    )
    training_cycle.add_argument(
        "--max-teacher-fixtures",
        type=positive_int,
        help="mine at most this many battle-diverse informative fixtures per cycle",
    )
    training_cycle.add_argument(
        "--split-seed",
        default="azelficoast.training-records",
    )
    training_cycle.add_argument("--train-fraction", type=unit_float, default=0.8)
    training_cycle.add_argument("--validation-fraction", type=unit_float, default=0.1)
    training_cycle.add_argument("--epochs", type=positive_int, default=1)
    training_cycle.add_argument("--learning-rate", type=positive_float, default=3e-4)
    training_cycle.add_argument("--policy-weight", type=nonnegative_float, default=1.0)
    training_cycle.add_argument(
        "--value-target",
        choices=VALUE_TARGET_SOURCES,
        default="public_belief_search_return",
    )
    training_cycle.add_argument(
        "--min-validation-improvement",
        type=nonnegative_float,
        default=1e-6,
    )
    training_cycle.add_argument(
        "--max-validation-value-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_cycle.add_argument(
        "--max-validation-policy-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_cycle.add_argument(
        "--max-validation-posterior-stress-regression",
        type=nonnegative_float,
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
    training_auto.add_argument("--generations", type=positive_int, default=1)
    training_auto.add_argument("--battles-per-generation", type=positive_int, default=12)
    training_auto.add_argument("--concurrency", type=positive_int, default=1)
    training_auto.add_argument(
        "--public-replays-per-generation",
        type=nonnegative_int,
        default=4,
        help=(
            "allow up to this many public Random Battle replays per generation; "
            "competence debt may reduce acquisition after the first generation; "
            "use 0 to disable network acquisition"
        ),
    )
    training_auto.add_argument(
        "--public-replay-min-rating",
        type=nonnegative_int,
        default=1500,
        help="minimum public replay rating admitted to automatic curriculum acquisition",
    )
    training_auto.add_argument(
        "--battle-search-policy-margin",
        type=unit_float,
        default=0.0,
        help=(
            "exact-search routing threshold while generating battles; defaults to 0 "
            "because mined teacher states are searched offline after generation"
        ),
    )
    training_auto.add_argument(
        "--promotion-battles",
        type=positive_even_int,
        default=32,
        help="side-balanced candidate-vs-incumbent battles required before promotion",
    )
    training_auto.add_argument(
        "--promotion-alpha",
        type=open_unit_float,
        default=0.10,
        help="maximum one-sided exact superiority p-value for battle promotion",
    )
    training_auto.add_argument("--teacher-budget", type=positive_int, default=4096)
    training_auto.add_argument(
        "--teacher-challenger-uncertainty",
        type=unit_float,
        default=DEFAULT_CHALLENGER_UNCERTAINTY_THRESHOLD,
        help=(
            "run posterior-stress challenger teachers when public policy uncertainty "
            "reaches this threshold; search/fallback states always challenge"
        ),
    )
    training_auto.add_argument(
        "--max-teacher-fixtures",
        type=positive_int,
        default=64,
        help="battle-diverse curriculum budget per candidate generation",
    )
    training_auto.add_argument(
        "--split-seed",
        default="azelficoast.training-records",
    )
    training_auto.add_argument("--train-fraction", type=unit_float, default=0.8)
    training_auto.add_argument("--validation-fraction", type=unit_float, default=0.1)
    training_auto.add_argument("--epochs", type=positive_int, default=1)
    training_auto.add_argument("--learning-rate", type=positive_float, default=3e-4)
    training_auto.add_argument("--policy-weight", type=nonnegative_float, default=1.0)
    training_auto.add_argument(
        "--value-target",
        choices=VALUE_TARGET_SOURCES,
        default="public_belief_search_return",
    )
    training_auto.add_argument(
        "--min-validation-improvement",
        type=nonnegative_float,
        default=1e-6,
    )
    training_auto.add_argument(
        "--max-validation-value-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_auto.add_argument(
        "--max-validation-policy-regression",
        type=nonnegative_float,
        default=0.0,
    )
    training_auto.add_argument(
        "--max-validation-posterior-stress-regression",
        type=nonnegative_float,
        default=0.0,
    )
    
    


def _add_coverage_command(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    coverage = subparsers.add_parser(
        "belief-coverage",
        help="summarize live public-belief routing and static admission coverage",
    )
    coverage.add_argument("traces", nargs="+", type=Path)
    
    


def build_parser() -> argparse.ArgumentParser:
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
        type=positive_float,
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
        type=unit_float,
        default=float(os.getenv("AZELFICOAST_SEARCH_POLICY_MARGIN", "1.0")),
        help=(
            "run exact search when learned top-two policy margin is at or below this "
            "threshold; 1.0 is conservative shadow mode (default: 1.0)"
        ),
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    
    _add_live_commands(subparsers)
    _add_corpus_commands(subparsers)
    _add_training_commands(subparsers)
    _add_coverage_command(subparsers)
    return parser
