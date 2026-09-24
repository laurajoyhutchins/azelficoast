"""Run Azelficoast against local or official Pokemon Showdown opponents."""

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

from azelficoast.player import AzelficoastPlayer

BATTLE_FORMAT = "gen9randombattle"
DEFAULT_RESULTS = Path("artifacts/results.jsonl")
DEFAULT_DECISIONS = Path("artifacts/decisions.jsonl")
DEFAULT_REPLAYS = Path("artifacts/replays")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
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

    subparsers = parser.add_subparsers(dest="command", required=True)

    local = subparsers.add_parser("local", help="battle a baseline on localhost:8000")
    local.add_argument("--battles", type=_positive_int, default=1)

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
) -> AzelficoastPlayer:
    return AzelficoastPlayer(
        account_configuration=AccountConfiguration(username, password),
        server_configuration=ShowdownServerConfiguration,
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
        save_replays=str(replays),
        decision_log=decisions,
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
    results: Path,
    decisions: Path,
    replays: Path,
) -> None:
    player = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
        save_replays=str(replays),
        decision_log=decisions,
    )
    opponent = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=1,
    )
    await player.battle_against(opponent, n_battles=battles)
    _append_results(player, results, mode="local")
    _print_summary(player)


async def _run_live(args: argparse.Namespace) -> None:
    username, password = _resolve_live_credentials(args.username)
    player = _live_player(username, password, args.replays, args.decisions)

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
        await _run_local(args.battles, args.results, args.decisions, args.replays)
    else:
        await _run_live(args)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        asyncio.run(_async_main(args))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
