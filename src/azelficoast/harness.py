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

from azelficoast.belief_coverage import summarize_traces
from azelficoast.corpus import BUILTIN_POLICIES, build_corpus, evaluate_corpus
from azelficoast.player import AzelficoastPlayer

BATTLE_FORMAT = "gen9randombattle"
DEFAULT_RESULTS = Path("artifacts/results.jsonl")
DEFAULT_DECISIONS = Path("artifacts/decisions.jsonl")
DEFAULT_REPLAYS = Path("artifacts/replays")
DEFAULT_CORPUS = Path("artifacts/corpus.jsonl")


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
) -> None:
    player = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        save_replays=str(replays),
        decision_log=decisions,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
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


def _run_belief_coverage(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_traces(args.traces), sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "corpus":
            _run_corpus(args)
        elif args.command == "belief-coverage":
            _run_belief_coverage(args)
        else:
            asyncio.run(_async_main(args))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
