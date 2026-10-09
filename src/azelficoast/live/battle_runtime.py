"""Battle execution and result recording for Azelficoast."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

from poke_env import AccountConfiguration, ShowdownServerConfiguration
from poke_env.player import Player, RandomPlayer

from azelficoast.live.player import AzelficoastPlayer

BATTLE_FORMAT = "gen9randombattle"


def resolve_live_credentials(username_override: str | None) -> tuple[str, str]:
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
    assert username is not None and password is not None
    return username, password


def prepare_output_paths(results: Path, decisions: Path, replays: Path) -> None:
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


async def run_local(
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


async def run_live(args: argparse.Namespace) -> None:
    username, password = resolve_live_credentials(args.username)
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


async def run_battle_command(args: argparse.Namespace) -> None:
    prepare_output_paths(args.results, args.decisions, args.replays)
    if args.command == "local":
        await run_local(
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
        await run_live(args)


