"""Fetch and normalize public Pokémon Showdown replay observations."""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

REPLAY_URL = (
    "https://replay.pokemonshowdown.com/"
    "gen9randombattle-2405042449-9irailjicjrb4g5sr5r0v1j7tthvqudpw"
)
SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc"
SETS_URL = (
    "https://raw.githubusercontent.com/smogon/pokemon-showdown/"
    f"{SHOWDOWN_COMMIT}/data/random-battles/gen9/sets.json"
)


class ReplayError(ValueError):
    """Raised when public replay evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class ProtocolEvent:
    index: int
    turn: int
    kind: str
    fields: tuple[str, ...]


@dataclass
class PokemonObservation:
    side: str
    species: str
    level: int | None = None
    moves: list[tuple[int, str]] = field(default_factory=list)
    item_events: list[tuple[int, str, str]] = field(default_factory=list)
    hp_events: list[tuple[int, str]] = field(default_factory=list)


@dataclass(frozen=True)
class ReplayEvidence:
    replay_id: str
    format: str | None
    log: str


def _get_json(url: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "azelficoast-research/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def replay_json_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path.rstrip("/")
    if not path.endswith(".json"):
        path += ".json"
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def fetch_replay(url: str) -> ReplayEvidence:
    payload = _get_json(replay_json_url(url))
    if not isinstance(payload, dict):
        raise ReplayError("replay endpoint did not return an object")
    log = payload.get("log")
    if not isinstance(log, str) or not log:
        raise ReplayError("replay JSON does not contain a log string")

    replay_id = payload.get("id")
    if not isinstance(replay_id, str) or not replay_id:
        replay_id = urllib.parse.urlsplit(url).path.rstrip("/").rsplit("/", 1)[-1]

    format_name = payload.get("format")
    if format_name is not None and not isinstance(format_name, str):
        format_name = str(format_name)

    return ReplayEvidence(replay_id=replay_id, format=format_name, log=log)


def fetch_randbats_sets() -> dict[str, Any]:
    payload = _get_json(SETS_URL)
    if not isinstance(payload, dict):
        raise ReplayError("pinned Showdown sets data is not an object")
    return payload


def parse_protocol(log: str) -> tuple[ProtocolEvent, ...]:
    events: list[ProtocolEvent] = []
    turn = 0
    for index, raw_line in enumerate(log.splitlines()):
        if not raw_line.startswith("|"):
            continue
        fields = tuple(raw_line.split("|")[1:])
        if not fields:
            continue
        kind = fields[0]
        if kind == "turn" and len(fields) >= 2:
            try:
                turn = int(fields[1])
            except ValueError as error:
                raise ReplayError(f"invalid turn number: {fields[1]!r}") from error
        events.append(
            ProtocolEvent(index=index, turn=turn, kind=kind, fields=fields[1:])
        )
    return tuple(events)


def _side_from_actor(actor: str) -> str | None:
    actor = actor.strip()
    if not actor:
        return None
    slot = actor.split(":", 1)[0]
    if slot.startswith("p1"):
        return "p1"
    if slot.startswith("p2"):
        return "p2"
    return None


def _slot_from_actor(actor: str) -> str:
    return actor.split(":", 1)[0].strip()


def _species_from_details(details: str) -> tuple[str, int | None]:
    parts = [part.strip() for part in details.split(",")]
    species = parts[0]
    level = None
    for part in parts[1:]:
        if part.startswith("L") and part[1:].isdigit():
            level = int(part[1:])
            break
    return species, level


def collect_observations(
    events: Iterable[ProtocolEvent],
) -> dict[tuple[str, str], PokemonObservation]:
    active_slots: dict[str, tuple[str, str]] = {}
    observations: dict[tuple[str, str], PokemonObservation] = {}

    def ensure(side: str, species: str, level: int | None = None) -> PokemonObservation:
        key = (side, species)
        observation = observations.get(key)
        if observation is None:
            observation = PokemonObservation(side=side, species=species, level=level)
            observations[key] = observation
        elif observation.level is None and level is not None:
            observation.level = level
        return observation

    for event in events:
        fields = event.fields
        if event.kind in {"switch", "drag"} and len(fields) >= 2:
            actor, details = fields[0], fields[1]
            side = _side_from_actor(actor)
            if side is None:
                continue
            species, level = _species_from_details(details)
            key = (side, species)
            active_slots[_slot_from_actor(actor)] = key
            ensure(side, species, level)
            continue

        if not fields:
            continue
        actor = fields[0]
        side = _side_from_actor(actor)
        if side is None:
            continue
        key = active_slots.get(_slot_from_actor(actor))
        if key is None:
            continue
        observation = ensure(*key)

        if event.kind == "move" and len(fields) >= 2:
            observation.moves.append((event.turn, fields[1]))
        elif event.kind in {"-item", "-enditem"} and len(fields) >= 2:
            observation.item_events.append((event.turn, event.kind, fields[1]))
        elif event.kind in {"-damage", "-heal"} and len(fields) >= 2:
            observation.hp_events.append((event.turn, fields[1]))

    return observations


def event_window(
    events: Sequence[ProtocolEvent],
    *,
    start_turn: int,
    end_turn: int,
) -> list[dict[str, Any]]:
    return [
        {
            "index": event.index,
            "turn": event.turn,
            "kind": event.kind,
            "fields": list(event.fields),
        }
        for event in events
        if start_turn <= event.turn <= end_turn
    ]


def probe_replay(url: str = REPLAY_URL) -> dict[str, Any]:
    replay = fetch_replay(url)
    events = parse_protocol(replay.log)
    observations = collect_observations(events)

    infernape = [
        observation
        for observation in observations.values()
        if observation.species.lower() == "infernape"
    ]
    kingambit = [
        observation
        for observation in observations.values()
        if observation.species.lower() == "kingambit"
    ]
    rillaboom = [
        observation
        for observation in observations.values()
        if observation.species.lower() == "rillaboom"
    ]

    sets = fetch_randbats_sets()
    infernape_sets = sets.get("infernape")

    return {
        "schema": "azelficoast.replay-probe",
        "schema_version": 1,
        "replay_id": replay.replay_id,
        "format": replay.format,
        "showdown_commit": SHOWDOWN_COMMIT,
        "infernape": [
            {
                "side": item.side,
                "species": item.species,
                "level": item.level,
                "moves": item.moves,
                "item_events": item.item_events,
                "hp_events": item.hp_events,
            }
            for item in infernape
        ],
        "kingambit": [
            {
                "side": item.side,
                "level": item.level,
                "hp_events": item.hp_events,
            }
            for item in kingambit
        ],
        "rillaboom": [
            {
                "side": item.side,
                "level": item.level,
                "hp_events": item.hp_events,
            }
            for item in rillaboom
        ],
        "turns_17_to_21": event_window(events, start_turn=17, end_turn=21),
        "infernape_generator_entry": infernape_sets,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default=REPLAY_URL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(probe_replay(args.replay), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
