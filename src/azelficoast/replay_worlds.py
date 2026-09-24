"""Reconstruct hidden randbats worlds from public Pokémon Showdown evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from poke_env.data import GenData

REPLAY_URL = (
    "https://replay.pokemonshowdown.com/"
    "gen9randombattle-2405042449-9irailjicjrb4g5sr5r0v1j7tthvqudpw"
)

# Latest randbats generator revision before this replay was played in July 2025.
SHOWDOWN_COMMIT = "6397bfddb3db4e916dd792e03c43355f7366e8ab"
SETS_URL = (
    "https://raw.githubusercontent.com/smogon/pokemon-showdown/"
    f"{SHOWDOWN_COMMIT}/data/random-battles/gen9/sets.json"
)

RANDBATS_EV = 85
RANDBATS_IV = 31

ITEM_DAMAGE_MODELS: dict[str, tuple[Fraction, Fraction]] = {
    # (physical attack multiplier, final damage multiplier)
    "Choice Band": (Fraction(3, 2), Fraction(1, 1)),
    "Choice Scarf": (Fraction(1, 1), Fraction(1, 1)),
    "Life Orb": (Fraction(1, 1), Fraction(13, 10)),
}


class ReplayError(ValueError):
    """Raised when replay or hidden-world evidence cannot be interpreted safely."""


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


@dataclass(frozen=True)
class DamageObservation:
    turn: int
    attacker_species: str
    attacker_level: int
    move: str
    target_species: str
    target_level: int
    target_tera_type: str | None
    before_hp: int
    after_hp: int
    max_hp: int
    damage: int


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _to_id(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


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
    slot = _slot_from_actor(actor)
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


def _parse_hp(status: str, previous_max: int | None = None) -> tuple[int, int] | None:
    token = status.split(" ", 1)[0]
    if "/" in token:
        current_text, max_text = token.split("/", 1)
        if current_text.isdigit() and max_text.isdigit():
            return int(current_text), int(max_text)
    if token == "0" and previous_max is not None:
        return 0, previous_max
    return None


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


def extract_damage_observations(events: Sequence[ProtocolEvent]) -> tuple[DamageObservation, ...]:
    """Pair public move and HP messages without using later hidden information."""
    active: dict[str, tuple[str, int | None]] = {}
    hp: dict[str, tuple[int, int]] = {}
    tera: dict[str, str] = {}
    pending_move: tuple[int, str, int, str, str, int] | None = None
    observations: list[DamageObservation] = []

    for event in events:
        fields = event.fields

        if event.kind in {"switch", "drag"} and len(fields) >= 3:
            slot = _slot_from_actor(fields[0])
            species, level = _species_from_details(fields[1])
            active[slot] = (species, level)
            parsed_hp = _parse_hp(fields[2])
            if parsed_hp is not None:
                hp[slot] = parsed_hp
            pending_move = None
            continue

        if event.kind == "-terastallize" and len(fields) >= 2:
            tera[_slot_from_actor(fields[0])] = fields[1]
            continue

        if event.kind == "move" and len(fields) >= 3:
            attacker_slot = _slot_from_actor(fields[0])
            target_slot = _slot_from_actor(fields[2])
            attacker = active.get(attacker_slot)
            target = active.get(target_slot)
            if (
                attacker is not None
                and target is not None
                and attacker[1] is not None
                and target[1] is not None
            ):
                pending_move = (
                    event.turn,
                    attacker[0],
                    attacker[1],
                    fields[1],
                    target_slot,
                    target[1],
                )
            else:
                pending_move = None
            continue

        if event.kind in {"-damage", "-heal"} and len(fields) >= 2:
            slot = _slot_from_actor(fields[0])
            previous = hp.get(slot)
            parsed_hp = _parse_hp(
                fields[1],
                previous_max=previous[1] if previous is not None else None,
            )
            if parsed_hp is None:
                continue

            if event.kind == "-damage" and pending_move is not None and previous is not None:
                (
                    move_turn,
                    attacker_species,
                    attacker_level,
                    move,
                    target_slot,
                    target_level,
                ) = pending_move
                target = active.get(slot)
                if (
                    move_turn == event.turn
                    and slot == target_slot
                    and target is not None
                    and previous[1] == parsed_hp[1]
                    and parsed_hp[0] <= previous[0]
                ):
                    observations.append(
                        DamageObservation(
                            turn=event.turn,
                            attacker_species=attacker_species,
                            attacker_level=attacker_level,
                            move=move,
                            target_species=target[0],
                            target_level=target_level,
                            target_tera_type=tera.get(slot),
                            before_hp=previous[0],
                            after_hp=parsed_hp[0],
                            max_hp=parsed_hp[1],
                            damage=previous[0] - parsed_hp[0],
                        )
                    )
                    pending_move = None

            hp[slot] = parsed_hp

    return tuple(observations)


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


def load_world_sample(path: str | Path) -> dict[str, Any]:
    sample_path = Path(path)
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReplayError(f"{sample_path}: world sample must be an object")
    if payload.get("schema") != "azelficoast.showdown-world-sample":
        raise ReplayError(f"{sample_path}: unexpected world-sample schema")
    if payload.get("schema_version") != 1:
        raise ReplayError(f"{sample_path}: unsupported world-sample schema version")
    if not isinstance(payload.get("rounds"), int) or payload["rounds"] <= 0:
        raise ReplayError(f"{sample_path}: invalid rounds")
    if not isinstance(payload.get("matched"), int) or payload["matched"] <= 0:
        raise ReplayError(f"{sample_path}: invalid matched count")
    item_counts = payload.get("item_counts")
    if not isinstance(item_counts, dict) or not item_counts:
        raise ReplayError(f"{sample_path}: missing item counts")
    for item, count in item_counts.items():
        if not isinstance(item, str) or not isinstance(count, int) or count <= 0:
            raise ReplayError(f"{sample_path}: malformed item count")
    return payload


def _neutral_stat(base: int, level: int) -> int:
    return math.floor((2 * base + RANDBATS_IV + RANDBATS_EV // 4) * level / 100) + 5


def _effectiveness(data: GenData, attacking_type: str, defending_types: Sequence[str]) -> Fraction:
    result = Fraction(1, 1)
    for defending_type in defending_types:
        value = data.type_chart[defending_type.upper()][attacking_type.upper()]
        result *= Fraction(str(value))
    return result


def damage_rolls_for_item(
    observation: DamageObservation,
    *,
    item: str,
) -> tuple[int, ...]:
    """Calculate ordinary physical-damage rolls for a recognized sampled item."""
    if item not in ITEM_DAMAGE_MODELS:
        raise ReplayError(
            f"no damage model for sampled item {item!r}; refusing to guess"
        )

    data = GenData.from_gen(9)
    attacker_id = _to_id(observation.attacker_species)
    target_id = _to_id(observation.target_species)
    move_id = _to_id(observation.move)

    attacker = data.pokedex.get(attacker_id)
    target = data.pokedex.get(target_id)
    move = data.moves.get(move_id)
    if not isinstance(attacker, dict) or not isinstance(target, dict) or not isinstance(move, dict):
        raise ReplayError("species or move is missing from pinned poke-env mechanics data")
    if move.get("category") != "Physical":
        raise ReplayError("current replay conditioner supports physical damage only")

    base_power = move.get("basePower")
    move_type = move.get("type")
    if not isinstance(base_power, int) or not isinstance(move_type, str):
        raise ReplayError("move lacks ordinary base power/type")

    attack_multiplier, final_multiplier = ITEM_DAMAGE_MODELS[item]
    attack = math.floor(
        _neutral_stat(attacker["baseStats"]["atk"], observation.attacker_level)
        * attack_multiplier
    )
    defense = _neutral_stat(target["baseStats"]["def"], observation.target_level)

    base_damage = (
        math.floor(
            math.floor(
                (math.floor(2 * observation.attacker_level / 5) + 2)
                * base_power
                * attack
                / defense
            )
            / 50
        )
        + 2
    )

    stab = (
        Fraction(3, 2)
        if move_type.lower() in {type_.lower() for type_ in attacker["types"]}
        else Fraction(1, 1)
    )
    defending_types = (
        [observation.target_tera_type]
        if observation.target_tera_type is not None
        else list(target["types"])
    )
    effectiveness = _effectiveness(data, move_type, defending_types)

    rolls: list[int] = []
    for random_percent in range(85, 101):
        damage = base_damage
        for modifier in (stab, effectiveness, final_multiplier):
            damage = math.floor(damage * modifier)
        damage = math.floor(damage * random_percent / 100)
        rolls.append(damage)
    return tuple(rolls)


def infer_item_posterior(
    sample: Mapping[str, Any],
    observation: DamageObservation,
) -> dict[str, Any]:
    """Condition empirical generator item mass on one exact public damage observation."""
    species = sample.get("species")
    if not isinstance(species, str) or _to_id(species) != _to_id(observation.attacker_species):
        raise ReplayError("world sample species does not match damage attacker")

    item_counts = sample.get("item_counts")
    if not isinstance(item_counts, dict):
        raise ReplayError("world sample has no item counts")

    prior_total = sum(item_counts.values())
    prior = {
        item: count / prior_total
        for item, count in sorted(item_counts.items())
    }

    compatible: dict[str, dict[str, Any]] = {}
    for item, count in sorted(item_counts.items()):
        rolls = damage_rolls_for_item(observation, item=item)
        if observation.damage in rolls:
            compatible[item] = {
                "count": count,
                "prior_weight": count / prior_total,
                "damage_min": min(rolls),
                "damage_max": max(rolls),
                "matching_rolls": sum(roll == observation.damage for roll in rolls),
            }

    if not compatible:
        raise ReplayError("public damage observation eliminated every sampled item world")

    # Each ordinary damage roll is equiprobable, so update sample mass by likelihood.
    weighted = {
        item: details["count"] * details["matching_rolls"] / 16
        for item, details in compatible.items()
    }
    evidence_mass = sum(weighted.values())
    posterior = {
        item: weight / evidence_mass
        for item, weight in sorted(weighted.items())
    }

    return {
        "prior": prior,
        "compatible": compatible,
        "posterior": posterior,
    }


def _target_replay_observation(events: Sequence[ProtocolEvent]) -> DamageObservation:
    matches = [
        observation
        for observation in extract_damage_observations(events)
        if observation.turn == 19
        and observation.attacker_species == "Infernape"
        and observation.move == "Close Combat"
        and observation.target_species == "Kingambit"
    ]
    if len(matches) != 1:
        raise ReplayError(
            f"expected exactly one turn-19 Infernape Close Combat observation, got {len(matches)}"
        )
    observation = matches[0]
    if observation.target_tera_type != "Flying":
        raise ReplayError(
            f"expected Tera Flying Kingambit, got {observation.target_tera_type!r}"
        )
    return observation


def build_replay_belief(
    replay: ReplayEvidence,
    events: Sequence[ProtocolEvent],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic belief certificate from already-acquired public evidence."""
    observation = _target_replay_observation(events)
    inference = infer_item_posterior(sample, observation)

    evidence = {
        "replay_id": replay.replay_id,
        "format": replay.format,
        "showdown_commit": SHOWDOWN_COMMIT,
        "sample": {
            "species": sample["species"],
            "observed_moves": sample.get("observed_moves"),
            "seed_family": sample.get("seed_family"),
            "rounds": sample["rounds"],
            "matched": sample["matched"],
            "item_counts": sample["item_counts"],
        },
        "damage_observation": {
            "turn": observation.turn,
            "attacker": observation.attacker_species,
            "attacker_level": observation.attacker_level,
            "move": observation.move,
            "target": observation.target_species,
            "target_level": observation.target_level,
            "target_tera_type": observation.target_tera_type,
            "before_hp": observation.before_hp,
            "after_hp": observation.after_hp,
            "max_hp": observation.max_hp,
            "damage": observation.damage,
        },
        "posterior": inference["posterior"],
    }

    return {
        "schema": "azelficoast.replay-belief",
        "schema_version": 1,
        **evidence,
        "prior": inference["prior"],
        "compatible_worlds": inference["compatible"],
        "belief_sha256": _sha256(evidence),
    }


def reconstruct_replay_belief(
    world_sample_path: str | Path,
    replay_url: str = REPLAY_URL,
) -> dict[str, Any]:
    replay = fetch_replay(replay_url)
    events = parse_protocol(replay.log)
    sample = load_world_sample(world_sample_path)
    return build_replay_belief(replay, events, sample)


def probe_replay(url: str = REPLAY_URL) -> dict[str, Any]:
    replay = fetch_replay(url)
    events = parse_protocol(replay.log)
    observations = collect_observations(events)

    infernape = [
        observation
        for observation in observations.values()
        if observation.species.lower() == "infernape"
    ]

    sets = fetch_randbats_sets()
    infernape_sets = sets.get("infernape")
    damage = _target_replay_observation(events)

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
        "turn_19_damage": {
            "attacker": damage.attacker_species,
            "move": damage.move,
            "target": damage.target_species,
            "target_tera_type": damage.target_tera_type,
            "before_hp": damage.before_hp,
            "after_hp": damage.after_hp,
            "damage": damage.damage,
        },
        "turns_17_to_21": event_window(events, start_turn=17, end_turn=21),
        "infernape_generator_entry": infernape_sets,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default=REPLAY_URL)
    parser.add_argument(
        "--world-sample",
        type=Path,
        help="condition a Showdown-generated world sample on replay evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.world_sample is None:
        result = probe_replay(args.replay)
    else:
        result = reconstruct_replay_belief(args.world_sample, args.replay)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
