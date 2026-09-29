"""Hosted execution adapter for the external Foul Play benchmark."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from poke_env import AccountConfiguration, LocalhostServerConfiguration

from azelficoast.live.player import AzelficoastPlayer
from azelficoast.research.external_strength import load_contract, settle_external_panel
from azelficoast.research.hosted.common import (
    SHOWDOWN_ROOT,
    HostedResearchError,
    run,
    showdown_revision,
    start_showdown_server,
    write_json,
)

ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = ROOT / "experiments" / "data" / "external-playing-strength-contract.json"
OUTPUT_ROOT = Path("/tmp/external-playing-strength")
EXTERNAL_ROOT = Path("/tmp/foul-play")
EXTERNAL_VENV = Path("/tmp/foul-play-venv")
EVALUATOR_WORKTREE = Path("/tmp/azelficoast-benchmark-state")
EXECUTION_UNITS_PER_SHARD = 4
UNIT_TIMEOUT_SECONDS = 4 * 60 * 60


def _git_head() -> str:
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _restore_evaluator(contract: Mapping[str, Any]) -> Path:
    source = contract["azelficoast"]["evaluator_source"]
    commit = str(source["commit"])
    digest = str(source["checkpoint_digest"])
    shutil.rmtree(EVALUATOR_WORKTREE, ignore_errors=True)
    subprocess.run(
        ("git", "worktree", "prune"),
        check=True,
        stdout=subprocess.DEVNULL,
    )
    run(("git", "fetch", "--depth", "1", "origin", commit))
    run(("git", "worktree", "add", "--detach", str(EVALUATOR_WORKTREE), "FETCH_HEAD"))
    promotion_path = EVALUATOR_WORKTREE / str(source["promotion_path"])
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    if not isinstance(promotion, dict) or promotion.get("checkpoint_digest") != digest:
        raise HostedResearchError("training-state promotion does not name frozen evaluator")
    checkpoint = promotion.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise HostedResearchError("training-state promotion lacks checkpoint path")
    checkpoint_path = promotion_path.parent / checkpoint
    manifest = json.loads((checkpoint_path / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or not isinstance(manifest.get("evaluator"), dict)
        or manifest["evaluator"].get("checkpoint_digest") != digest
    ):
        raise HostedResearchError("frozen evaluator manifest digest drifted")
    return checkpoint_path


def _install_foul_play(contract: Mapping[str, Any]) -> Path:
    opponent = contract["opponent"]
    shutil.rmtree(EXTERNAL_ROOT, ignore_errors=True)
    shutil.rmtree(EXTERNAL_VENV, ignore_errors=True)
    run(("git", "init", str(EXTERNAL_ROOT)))
    run(("git", "-C", str(EXTERNAL_ROOT), "remote", "add", "origin", str(opponent["repository"])))
    run(("git", "-C", str(EXTERNAL_ROOT), "fetch", "--depth", "1", "origin", str(opponent["revision"])))
    run(("git", "-C", str(EXTERNAL_ROOT), "checkout", "--detach", "FETCH_HEAD"))
    actual = subprocess.run(
        ("git", "-C", str(EXTERNAL_ROOT), "rev-parse", "HEAD"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if actual != opponent["revision"]:
        raise HostedResearchError("Foul Play checkout identity drifted")
    run((sys.executable, "-m", "venv", str(EXTERNAL_VENV)))
    python = EXTERNAL_VENV / "bin" / "python"
    run((str(python), "-m", "pip", "install", "--upgrade", "pip==24.2"))
    run((str(python), "-m", "pip", "install", "-r", str(EXTERNAL_ROOT / "requirements.txt")))
    return python


def _foul_play_process(
    python: Path,
    contract: Mapping[str, Any],
    *,
    username: str,
    mode: str,
    opponent_username: str | None,
    battles: int,
    log_path: Path,
) -> tuple[subprocess.Popen[bytes], object]:
    args = [
        str(python),
        "run.py",
        "--websocket-uri",
        "local",
        "--ps-username",
        username,
        "--bot-mode",
        mode,
        "--pokemon-format",
        str(contract["format"]),
        "--run-count",
        str(battles),
        "--search-time-ms",
        str(contract["opponent"]["search_time_ms"]),
        "--search-parallelism",
        str(contract["opponent"]["search_parallelism"]),
        "--search-threads",
        str(contract["opponent"]["search_threads"]),
        "--log-level",
        "INFO",
    ]
    if opponent_username is not None:
        args.extend(["--user-to-challenge", opponent_username])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    process = subprocess.Popen(
        args,
        cwd=EXTERNAL_ROOT,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, handle


def _wait_for_log(process: subprocess.Popen[bytes], path: Path, needle: str) -> None:
    deadline = time.time() + 60
    while time.time() < deadline:
        if path.is_file() and needle in path.read_text(encoding="utf-8", errors="replace"):
            return
        if process.poll() is not None:
            break
        time.sleep(0.25)
    tail = path.read_text(encoding="utf-8", errors="replace")[-4000:] if path.is_file() else ""
    raise HostedResearchError(f"Foul Play did not become ready for challenges\n{tail}")


def _battle_identity(unit: int, source_tag: str) -> str:
    if not source_tag:
        raise HostedResearchError("external-strength battle tag is empty")
    return f"unit-{unit:02d}:{source_tag}"


def _phase_rows(
    player: AzelficoastPlayer,
    *,
    unit: int,
    shard: int,
    direction: str,
    contract: Mapping[str, Any],
    git_head: str,
) -> list[dict[str, Any]]:
    source = contract["azelficoast"]["evaluator_source"]
    opponent = contract["opponent"]
    rows: list[dict[str, Any]] = []
    for source_tag, battle in sorted(player.battles.items()):
        tag = _battle_identity(unit, source_tag)
        if battle.won:
            outcome = "win"
        elif battle.lost:
            outcome = "loss"
        else:
            outcome = "tie"
        rows.append(
            {
                "battle_tag": tag,
                "source_battle_tag": source_tag,
                "format": contract["format"],
                "shard": shard,
                "direction": direction,
                "player_role": battle.player_role,
                "outcome": outcome,
                "azelficoast_git_sha": git_head,
                "evaluator_digest": source["checkpoint_digest"],
                "showdown_revision": showdown_revision(),
                "opponent": "foul-play",
                "opponent_revision": opponent["revision"],
            }
        )
    return rows


async def _send_sequential_challenges(
    player: AzelficoastPlayer,
    opponent: str,
    *,
    battles: int,
) -> None:
    """Challenge only after the previous battle has fully settled.

    poke-env's multi-challenge helper sends the next challenge once the previous
    battle starts. Foul Play's accept loop consumes one battle at a time, so that
    eager challenge can be read and discarded by the active battle loop. One-battle
    calls wait for battle completion before returning and preserve the intended
    alternating challenge protocol.
    """
    for _ in range(battles):
        await player.send_challenges(opponent, n_challenges=1)


def _unit_spec(
    unit: int,
    *,
    unit_count: int,
    contract: Mapping[str, Any],
) -> tuple[int, str, int]:
    """Map bounded hosted work onto the frozen logical shard contract."""
    shard_count = int(contract["shard_count"])
    per_direction = int(contract["battles_per_direction_per_shard"])
    expected_units = shard_count * EXECUTION_UNITS_PER_SHARD
    if unit_count != expected_units or not 0 <= unit < unit_count:
        raise HostedResearchError(
            "hosted execution-unit matrix does not preserve the frozen shard contract"
        )
    if per_direction % 2:
        raise HostedResearchError(
            "frozen per-direction battle count must split evenly across hosted units"
        )
    shard = unit // EXECUTION_UNITS_PER_SHARD
    lane = unit % EXECUTION_UNITS_PER_SHARD
    direction = (
        "foul_play_challenges" if lane < 2 else "azelficoast_challenges"
    )
    return shard, direction, per_direction // 2


async def _run_panel_unit(
    unit: int,
    contract: Mapping[str, Any],
    *,
    shard: int,
    direction: str,
    battles: int,
    evaluator_checkpoint: Path,
    foul_play_python: Path,
) -> list[dict[str, Any]]:
    unit_root = OUTPUT_ROOT / f"unit-{unit:02d}"
    replays = unit_root / "replays"
    decisions = unit_root / "decisions.jsonl"
    replays.mkdir(parents=True, exist_ok=True)
    username = f"AzelfUnit{unit:02d}"
    external_username = f"FoulUnit{unit:02d}"
    player = AzelficoastPlayer(
        account_configuration=AccountConfiguration(username, None),
        server_configuration=LocalhostServerConfiguration,
        battle_format=str(contract["format"]),
        max_concurrent_battles=1,
        save_replays=str(replays),
        decision_log=decisions,
        showdown_root=SHOWDOWN_ROOT,
        belief_timeout_seconds=float(contract["azelficoast"]["belief_timeout_seconds"]),
        evaluator_checkpoint=evaluator_checkpoint,
        search_policy_margin=float(contract["azelficoast"]["search_policy_margin"]),
    )
    head = _git_head()
    log_path = unit_root / f"{direction}.log"

    if direction == "foul_play_challenges":
        task = asyncio.create_task(
            player.accept_challenges(external_username, n_challenges=battles)
        )
        process, handle = _foul_play_process(
            foul_play_python,
            contract,
            username=external_username,
            mode="challenge_user",
            opponent_username=username,
            battles=battles,
            log_path=log_path,
        )
        try:
            await asyncio.wait_for(task, timeout=UNIT_TIMEOUT_SECONDS)
            await asyncio.to_thread(process.wait, 120)
            if process.returncode != 0:
                raise HostedResearchError(
                    f"Foul Play challenge unit exited {process.returncode}"
                )
        finally:
            if process.poll() is None:
                process.terminate()
            handle.close()
    elif direction == "azelficoast_challenges":
        process, handle = _foul_play_process(
            foul_play_python,
            contract,
            username=external_username,
            mode="accept_challenge",
            opponent_username=None,
            battles=battles,
            log_path=log_path,
        )
        try:
            await asyncio.to_thread(
                _wait_for_log,
                process,
                log_path,
                f"Waiting for a {contract['format']} challenge",
            )
            await asyncio.wait_for(
                _send_sequential_challenges(
                    player,
                    external_username,
                    battles=battles,
                ),
                timeout=UNIT_TIMEOUT_SECONDS,
            )
            await asyncio.to_thread(process.wait, 120)
            if process.returncode != 0:
                raise HostedResearchError(
                    f"Foul Play accept unit exited {process.returncode}"
                )
        finally:
            if process.poll() is None:
                process.terminate()
            handle.close()
    else:
        raise HostedResearchError(f"unknown external-strength direction {direction!r}")

    rows = _phase_rows(
        player,
        unit=unit,
        shard=shard,
        direction=direction,
        contract=contract,
        git_head=head,
    )
    if len(rows) != battles:
        raise HostedResearchError(
            f"external-strength unit {unit} produced {len(rows)} battles; expected {battles}"
        )
    return rows


def run_unit(unit: int, *, unit_count: int) -> None:
    contract = load_contract(CONTRACT_PATH)
    shard, direction, battles = _unit_spec(
        unit,
        unit_count=unit_count,
        contract=contract,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    evaluator = _restore_evaluator(contract)
    foul_play_python = _install_foul_play(contract)
    showdown = start_showdown_server()
    try:
        rows = asyncio.run(
            _run_panel_unit(
                unit,
                contract,
                shard=shard,
                direction=direction,
                battles=battles,
                evaluator_checkpoint=evaluator,
                foul_play_python=foul_play_python,
            )
        )
    finally:
        showdown.terminate()
        try:
            showdown.wait(timeout=10)
        except subprocess.TimeoutExpired:
            showdown.kill()

    unit_root = OUTPUT_ROOT / f"unit-{unit:02d}"
    result_path = unit_root / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(
        unit_root / "metadata.json",
        {
            "schema": "azelficoast.external-playing-strength-unit",
            "schema_version": 1,
            "unit": unit,
            "shard": shard,
            "direction": direction,
            "battle_count": len(rows),
            "git_sha": _git_head(),
            "evaluator_digest": contract["azelficoast"]["evaluator_source"]["checkpoint_digest"],
            "showdown_revision": showdown_revision(),
            "opponent_revision": contract["opponent"]["revision"],
        },
        pretty=True,
    )


def aggregate() -> None:
    contract = load_contract(CONTRACT_PATH)
    rows: list[dict[str, Any]] = []
    paths = sorted(Path("/tmp/exact").glob("unit-*/results.jsonl"))
    expected_units = int(contract["shard_count"]) * EXECUTION_UNITS_PER_SHARD
    if len(paths) != expected_units:
        raise HostedResearchError(
            f"aggregate received {len(paths)} execution units; expected {expected_units}"
        )
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise HostedResearchError(f"{path} contains a non-object row")
                rows.append(row)
    result = settle_external_panel(contract, rows)
    write_json("/tmp/external-playing-strength-aggregate.json", result, pretty=True)
    print(json.dumps(result, sort_keys=True))
