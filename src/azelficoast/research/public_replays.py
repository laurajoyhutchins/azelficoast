"""Import public Pokémon Showdown Random Battle replays as frozen decision traces."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import subprocess
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_env import AccountConfiguration
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder

from azelficoast.live.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION, battle_view
from azelficoast.research.verification.showdown_damage_corpus import PINNED_SHOWDOWN_COMMIT

PUBLIC_REPLAY_SCHEMA = "azelficoast.public-replay-corpus"
PUBLIC_REPLAY_SCHEMA_VERSION = 1
DEFAULT_REPLAY_FORMAT = "gen9randombattle"
REPLAY_ORIGIN = "https://replay.pokemonshowdown.com"


class PublicReplayError(ValueError):
    """Raised when public replay evidence cannot be imported faithfully."""


class PublicReplayConflictError(PublicReplayError):
    """Raised when an immutable public replay artifact already differs."""


@dataclass(frozen=True)
class PublicReplay:
    replay_id: str
    payload: Mapping[str, Any]
    rating: int | None
    uploadtime: int | None

    @property
    def log(self) -> str:
        value = self.payload.get("log")
        if not isinstance(value, str) or not value:
            raise PublicReplayError(f"{self.replay_id}: replay has no public log")
        return value

    @property
    def inputlog(self) -> str:
        value = self.payload.get("inputlog")
        if not isinstance(value, str) or not value:
            raise PublicReplayError(
                f"{self.replay_id}: replay has no reconstructible inputlog"
            )
        return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _to_id(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _input_version(inputlog: str) -> str:
    for line in inputlog.splitlines():
        if line.startswith(">version "):
            version = line[len(">version ") :].strip()
            if version:
                return version
    raise PublicReplayError("replay inputlog does not declare its Showdown version")


def _get_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "azelficoast-public-corpus/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _search_url(format_id: str, before: int | None) -> str:
    query: dict[str, str] = {"format": format_id}
    if before is not None:
        query["before"] = str(before)
    return f"{REPLAY_ORIGIN}/search.json?{urllib.parse.urlencode(query)}"


def discover_public_replays(
    *,
    format_id: str = DEFAULT_REPLAY_FORMAT,
    max_battles: int = 100,
    min_rating: int = 0,
    before: int | None = None,
) -> list[dict[str, Any]]:
    """Discover a deterministic prefix of public replay metadata.

    Pokémon Showdown returns up to 51 rows. The 51st is a pagination
    sentinel; accepted rows come from the first 50 and the next cursor is the
    sentinel row's uploadtime.
    """
    if max_battles <= 0:
        raise PublicReplayError("max_battles must be positive")
    if min_rating < 0:
        raise PublicReplayError("min_rating must be non-negative")

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = before

    while len(selected) < max_battles:
        raw = _get_json(_search_url(format_id, cursor))
        if not isinstance(raw, list):
            raise PublicReplayError("Showdown replay search did not return a list")
        page = [row for row in raw[:50] if isinstance(row, Mapping)]
        if not page:
            break

        for row in page:
            replay_id = row.get("id")
            if not isinstance(replay_id, str) or not replay_id or replay_id in seen:
                continue
            seen.add(replay_id)
            rating = row.get("rating")
            if isinstance(rating, int) and not isinstance(rating, bool):
                numeric_rating = rating
            elif isinstance(rating, str) and rating.isdigit():
                numeric_rating = int(rating)
            else:
                numeric_rating = 0
            if numeric_rating < min_rating:
                continue
            selected.append(dict(row))
            if len(selected) >= max_battles:
                break

        if len(raw) <= 50:
            break
        sentinel = raw[-1]
        last_upload = sentinel.get("uploadtime") if isinstance(sentinel, Mapping) else None
        if not isinstance(last_upload, int) or last_upload <= 0 or last_upload == cursor:
            raise PublicReplayError("replay pagination did not expose a usable before cursor")
        cursor = last_upload

    return selected


def fetch_public_replay(metadata: Mapping[str, Any]) -> PublicReplay:
    replay_id = metadata.get("id")
    if not isinstance(replay_id, str) or not replay_id:
        raise PublicReplayError("replay metadata has no id")
    payload = _get_json(f"{REPLAY_ORIGIN}/{urllib.parse.quote(replay_id)}.json")
    if not isinstance(payload, Mapping):
        raise PublicReplayError(f"{replay_id}: replay endpoint did not return an object")

    payload_id = payload.get("id")
    if payload_id is not None and payload_id != replay_id:
        raise PublicReplayError(f"{replay_id}: replay endpoint returned a different id")
    format_id = payload.get("formatid")
    if format_id is None:
        raw_format = payload.get("format")
        format_id = _to_id(str(raw_format)) if raw_format is not None else None
    if format_id is not None and _to_id(str(format_id)) != DEFAULT_REPLAY_FORMAT:
        raise PublicReplayError(
            f"{replay_id}: expected {DEFAULT_REPLAY_FORMAT}, got {format_id!r}"
        )

    replay = PublicReplay(
        replay_id=replay_id,
        payload=dict(payload),
        rating=(
            metadata.get("rating")
            if isinstance(metadata.get("rating"), int)
            and not isinstance(metadata.get("rating"), bool)
            else None
        ),
        uploadtime=(
            metadata.get("uploadtime")
            if isinstance(metadata.get("uploadtime"), int)
            and not isinstance(metadata.get("uploadtime"), bool)
            else None
        ),
    )
    _ = replay.log
    inputlog = replay.inputlog
    source_version = _input_version(inputlog)
    if source_version != PINNED_SHOWDOWN_COMMIT:
        raise PublicReplayError(
            f"{replay_id}: source Showdown version {source_version} does not match "
            f"pinned reconstruction revision {PINNED_SHOWDOWN_COMMIT}"
        )
    return replay


def _write_immutable(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise PublicReplayConflictError(
                f"{path} already contains different public replay evidence"
            )
    else:
        path.write_bytes(content)
    return _sha256_bytes(content)


def freeze_public_replay(replay: PublicReplay, root: str | Path) -> dict[str, Any]:
    destination = Path(root) / "raw" / f"{replay.replay_id}.json"
    content = (_canonical(dict(replay.payload)) + "\n").encode("utf-8")
    digest = _write_immutable(destination, content)
    return {
        "replay_id": replay.replay_id,
        "raw_path": str(destination),
        "raw_sha256": digest,
        "rating": replay.rating,
        "uploadtime": replay.uploadtime,
        "source_locator": f"{REPLAY_ORIGIN}/{replay.replay_id}",
        "source_showdown_version": _input_version(replay.inputlog),
    }


def _showdown_revision(showdown_root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(showdown_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PublicReplayError(f"cannot read Showdown revision: {error}") from error
    if revision != PINNED_SHOWDOWN_COMMIT:
        raise PublicReplayError(
            "public replay reconstruction requires pinned Showdown "
            f"{PINNED_SHOWDOWN_COMMIT}, got {revision}"
        )
    return revision


def _bridge_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "replay_inputlog_to_streams.cjs"


def _replay_streams(showdown_root: Path, inputlog: str) -> dict[str, list[str]]:
    script = _bridge_script()
    if not script.is_file():
        raise PublicReplayError(f"missing replay bridge: {script}")
    with tempfile.TemporaryDirectory(prefix="azelficoast-replay-") as directory:
        input_path = Path(directory) / "input.log"
        input_path.write_text(inputlog, encoding="utf-8")
        try:
            result = subprocess.run(
                ["node", str(script), str(showdown_root), str(input_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            detail = getattr(error, "stderr", None) or str(error)
            raise PublicReplayError(f"Showdown replay bridge failed: {detail}") from error
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PublicReplayError("Showdown replay bridge returned invalid JSON") from error
    if not isinstance(document, Mapping):
        raise PublicReplayError("Showdown replay bridge returned a non-object")
    sides: dict[str, list[str]] = {}
    for side in ("p1", "p2"):
        raw = document.get(side)
        if not isinstance(raw, list) or not all(isinstance(chunk, str) for chunk in raw):
            raise PublicReplayError(f"Showdown replay bridge omitted {side} stream")
        sides[side] = list(raw)
    return sides


def _input_players(inputlog: str) -> dict[str, str]:
    players: dict[str, str] = {}
    for line in inputlog.splitlines():
        if not line.startswith(">player "):
            continue
        try:
            _, side, raw = line.split(" ", 2)
            document = json.loads(raw)
        except (ValueError, json.JSONDecodeError) as error:
            raise PublicReplayError(f"malformed inputlog player line: {line!r}") from error
        name = document.get("name") if isinstance(document, Mapping) else None
        if side in {"p1", "p2"} and isinstance(name, str) and name:
            players[side] = name
    if set(players) != {"p1", "p2"}:
        raise PublicReplayError("inputlog does not identify both players")
    return players


def _input_choices(inputlog: str, side: str) -> list[str]:
    prefix = f">{side} "
    choices: list[str] = []
    for line in inputlog.splitlines():
        if not line.startswith(prefix):
            continue
        choice = line[len(prefix) :].strip()
        if not choice or choice == "undo":
            continue
        choices.append(choice)
    return choices


def _winner(log: str) -> str | None:
    for line in log.splitlines():
        if line.startswith("|win|"):
            return line.split("|", 2)[2]
        if line == "|tie|":
            return None
    raise PublicReplayError("public replay log has no terminal result")


def _timestamp(uploadtime: int | None) -> str:
    if uploadtime is None:
        return "1970-01-01T00:00:00+00:00"
    return datetime.fromtimestamp(uploadtime, UTC).isoformat()


def _actionable_request(request: Mapping[str, Any]) -> bool:
    if request.get("wait") is True:
        return False
    force_switch = request.get("forceSwitch")
    forced = (
        any(bool(value) for value in force_switch)
        if isinstance(force_switch, list)
        else bool(force_switch)
    )
    return isinstance(request.get("active"), list) or forced


def _resolved_choice_key(choice: str, request: Mapping[str, Any]) -> tuple[str, str, bool]:
    words = choice.strip().split()
    if not words:
        raise PublicReplayError("empty recorded Showdown choice")
    kind = words[0].lower()
    tera = "terastallize" in {word.lower() for word in words[2:]}

    if kind == "move" and len(words) >= 2:
        move = words[1]
        if move.isdigit():
            active = request.get("active")
            if not isinstance(active, list) or not active or not isinstance(active[0], Mapping):
                raise PublicReplayError("numeric move choice has no active request")
            moves = active[0].get("moves")
            index = int(move) - 1
            if (
                not isinstance(moves, list)
                or index < 0
                or index >= len(moves)
                or not isinstance(moves[index], Mapping)
            ):
                raise PublicReplayError("numeric move choice is outside request move list")
            move = str(moves[index].get("id") or moves[index].get("move") or "")
        return ("move", _to_id(move), tera)

    if kind == "switch" and len(words) >= 2:
        target = " ".join(words[1:])
        if words[1].isdigit():
            side = request.get("side")
            pokemon = side.get("pokemon") if isinstance(side, Mapping) else None
            index = int(words[1]) - 1
            if (
                not isinstance(pokemon, list)
                or index < 0
                or index >= len(pokemon)
                or not isinstance(pokemon[index], Mapping)
            ):
                raise PublicReplayError("numeric switch choice is outside request team")
            ident = str(pokemon[index].get("ident") or "")
            target = ident.split(":", 1)[-1].strip()
        return ("switch", _to_id(target), False)

    raise PublicReplayError(f"unsupported recorded Showdown choice {choice!r}")


def _action_key(action: str) -> tuple[str, str, bool] | None:
    value = action.strip()
    if value.startswith("/choose "):
        value = value[len("/choose ") :]
    words = value.split()
    if len(words) < 2:
        return None
    kind = words[0].lower()
    tera = "terastallize" in {word.lower() for word in words[2:]}
    if kind == "move":
        return ("move", _to_id(words[1]), tera)
    if kind == "switch":
        modifiers = {"mega", "zmove", "dynamax", "terastallize"}
        target = " ".join(word for word in words[1:] if word.lower() not in modifiers)
        return ("switch", _to_id(target), False)
    return None


def _recorded_action(
    choice: str,
    request: Mapping[str, Any],
    legal_actions: Sequence[str],
) -> str:
    wanted = _resolved_choice_key(choice, request)
    matches = [action for action in legal_actions if _action_key(action) == wanted]
    if len(matches) != 1:
        raise PublicReplayError(
            f"recorded choice {choice!r} matched {len(matches)} legal actions"
        )
    return matches[0]


class _ReplayTraceReader(Player):
    """Offline poke-env reader. It parses player-specific Showdown output only."""

    def __init__(self, username: str, format_id: str):
        super().__init__(
            account_configuration=AccountConfiguration(username, None),
            battle_format=format_id,
            start_listening=False,
            log_level=logging.CRITICAL,
        )

        async def _noop(*args: Any, **kwargs: Any) -> None:
            return None

        self.ps_client.send_message = _noop

    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        maybe_default_order: bool = False,
    ) -> None:
        return None

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        raise PublicReplayError("offline replay reader must never choose a move")


def _protocol_messages(chunk: str) -> list[list[str]]:
    return [line.split("|") for line in chunk.splitlines() if line.startswith("|")]


def _request_from_chunk(chunk: str) -> Mapping[str, Any] | None:
    for line in reversed(chunk.splitlines()):
        if not line.startswith("|request|"):
            continue
        try:
            value = json.loads(line[len("|request|") :])
        except json.JSONDecodeError as error:
            raise PublicReplayError("invalid request JSON in reconstructed replay") from error
        if not isinstance(value, Mapping):
            raise PublicReplayError("reconstructed request is not an object")
        return value
    return None


async def _trace_side(
    replay: PublicReplay,
    *,
    side: str,
    username: str,
    chunks: Sequence[str],
) -> list[dict[str, Any]]:
    reader = _ReplayTraceReader(username, DEFAULT_REPLAY_FORMAT)
    battle_tag = f"battle-{replay.replay_id}"
    run_id = f"public-replay:{replay.replay_id}:{side}"
    observed_at = _timestamp(replay.uploadtime)
    choices = _input_choices(replay.inputlog, side)
    choice_index = 0
    event_index = 0
    protocol_index = 0
    decision_index = 0
    records: list[dict[str, Any]] = []

    reader.ps_client._battle_locks[battle_tag] = asyncio.Lock()
    battle = await reader._create_battle(f">{battle_tag}".split("-"))
    battle.logger = None

    for chunk in chunks:
        messages = _protocol_messages(chunk)
        if not messages:
            continue
        split_messages = [[f">{battle_tag}"], *messages]
        await reader._handle_battle_message(split_messages)

        records.append(
            {
                "schema": TRACE_SCHEMA,
                "schema_version": TRACE_SCHEMA_VERSION,
                "run_id": run_id,
                "event_index": event_index,
                "observed_at": observed_at,
                "kind": "protocol",
                "room": battle_tag,
                "protocol_index": protocol_index,
                "messages": messages,
                "source": {
                    "kind": "public-showdown-replay",
                    "replay_id": replay.replay_id,
                    "side": side,
                },
            }
        )
        event_index += 1
        protocol_index += 1

        request = _request_from_chunk(chunk)
        if request is None or not _actionable_request(request):
            continue
        if choice_index >= len(choices):
            raise PublicReplayError(
                f"{replay.replay_id}/{side}: request has no recorded input choice"
            )
        battle = reader.battles.get(battle_tag)
        if battle is None:
            raise PublicReplayError(
                f"{replay.replay_id}/{side}: poke-env did not retain reconstructed battle"
            )
        legal_actions = [order.message for order in getattr(battle, "valid_orders", ())]
        if not legal_actions:
            raise PublicReplayError(
                f"{replay.replay_id}/{side}: actionable request has no legal orders"
            )
        choice = choices[choice_index]
        choice_index += 1
        if choice in {"default", "forfeit"}:
            records.append(
                {
                    "schema": TRACE_SCHEMA,
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "run_id": run_id,
                    "event_index": event_index,
                    "observed_at": observed_at,
                    "kind": "decision_exclusion",
                    "battle_tag": battle_tag,
                    "request_index": choice_index - 1,
                    "reason": f"recorded-{choice}-has-no-exact-human-action-label",
                    "source": {
                        "kind": "public-showdown-replay",
                        "replay_id": replay.replay_id,
                        "side": side,
                    },
                }
            )
            event_index += 1
            continue
        action = _recorded_action(choice, request, legal_actions)
        records.append(
            {
                "schema": TRACE_SCHEMA,
                "schema_version": TRACE_SCHEMA_VERSION,
                "run_id": run_id,
                "event_index": event_index,
                "observed_at": observed_at,
                "kind": "decision",
                "battle_tag": battle_tag,
                "decision_index": decision_index,
                "state": battle_view(battle),
                "chosen_action": action,
                "decision_metadata": {
                    "selected_policy": "recorded-human",
                    "training_policy_authority": False,
                    "source_replay_id": replay.replay_id,
                    "source_side": side,
                    "source_rating": replay.rating,
                },
            }
        )
        event_index += 1
        decision_index += 1

    leftovers = [
        choice
        for choice in choices[choice_index:]
        if choice not in {"default", "forfeit"}
    ]
    if leftovers:
        raise PublicReplayError(
            f"{replay.replay_id}/{side}: {len(leftovers)} recorded choices were not consumed"
        )

    battle = reader.battles.get(battle_tag)
    if battle is None:
        raise PublicReplayError(f"{replay.replay_id}/{side}: reconstructed battle missing")
    winner = _winner(replay.log)
    won = winner == username if winner is not None else None
    records.append(
        {
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": run_id,
            "event_index": event_index,
            "observed_at": observed_at,
            "kind": "terminal",
            "battle_tag": battle_tag,
            "won": won,
            "lost": (not won) if won is not None else None,
            "tied": winner is None,
            "final_state": battle_view(battle),
            "source": {
                "kind": "public-showdown-replay",
                "replay_id": replay.replay_id,
                "side": side,
            },
        }
    )
    return records


async def reconstruct_replay_trace(
    replay: PublicReplay,
    *,
    showdown_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(showdown_root)
    _showdown_revision(root)
    players = _input_players(replay.inputlog)
    streams = _replay_streams(root, replay.inputlog)
    rows: list[dict[str, Any]] = []
    for side in ("p1", "p2"):
        rows.extend(
            await _trace_side(
                replay,
                side=side,
                username=players[side],
                chunks=streams[side],
            )
        )
    return rows


def _trace_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(_canonical(dict(row)) + "\n" for row in rows).encode("utf-8")


def import_public_replays(
    *,
    showdown_root: str | Path,
    output_root: str | Path,
    max_battles: int = 100,
    min_rating: int = 0,
    before: int | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Fetch, freeze, reconstruct, and emit public Random Battle decision traces."""
    root = Path(output_root)
    revision = _showdown_revision(Path(showdown_root))
    metadata = discover_public_replays(
        max_battles=max_battles,
        min_rating=min_rating,
        before=before,
    )

    rows: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for candidate in metadata:
        replay_id = str(candidate.get("id") or "")
        try:
            replay = fetch_public_replay(candidate)
            frozen = freeze_public_replay(replay, root)
            replay_rows = asyncio.run(
                reconstruct_replay_trace(replay, showdown_root=showdown_root)
            )
            rows.extend(replay_rows)
            admitted.append(
                {
                    **frozen,
                    "trace_record_count": len(replay_rows),
                    "decision_count": sum(
                        row.get("kind") == "decision" for row in replay_rows
                    ),
                }
            )
        except (PublicReplayError, OSError) as error:
            if strict:
                raise
            excluded.append(
                {
                    "replay_id": replay_id,
                    "reason": type(error).__name__,
                    "detail": str(error)[-1000:],
                }
            )

    trace_path = root / "decisions.jsonl"
    trace_digest = _write_immutable(trace_path, _trace_bytes(rows))
    manifest = {
        "schema": PUBLIC_REPLAY_SCHEMA,
        "schema_version": PUBLIC_REPLAY_SCHEMA_VERSION,
        "format": DEFAULT_REPLAY_FORMAT,
        "showdown_commit": revision,
        "query": {
            "max_battles": max_battles,
            "min_rating": min_rating,
            "before": before,
        },
        "provenance": {
            "source": "Pokemon Showdown public replay service",
            "source_locator": f"{REPLAY_ORIGIN}/search.json",
            "source_revision": "live replay service",
            "acquisition": "downloaded and deterministically replayed",
            "transformation": (
                "public replay JSON plus autogenerated-team inputlog replayed through "
                "pinned Showdown into player-perspective poke-env decision traces"
            ),
            "contains_third_party_assets": False,
            "contains_user_identifiers": True,
            "license_or_terms": "not asserted",
        },
        "discovered_count": len(metadata),
        "admitted_count": len(admitted),
        "excluded_count": len(excluded),
        "trace_record_count": len(rows),
        "decision_count": sum(row.get("kind") == "decision" for row in rows),
        "trace_path": str(trace_path),
        "trace_sha256": trace_digest,
        "admitted": admitted,
        "excluded": excluded,
    }
    manifest_path = root / "manifest.json"
    manifest_digest = _write_immutable(
        manifest_path, (_canonical(manifest) + "\n").encode("utf-8")
    )
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "trace": str(trace_path),
        "trace_sha256": trace_digest,
        "discovered_count": len(metadata),
        "admitted_count": len(admitted),
        "excluded_count": len(excluded),
        "decision_count": manifest["decision_count"],
    }
