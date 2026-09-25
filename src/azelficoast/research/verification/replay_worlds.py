"""Reconstruct hidden randbats worlds from public Pokémon Showdown evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_env.data import GenData

REPLAY_URL = (
    "https://replay.pokemonshowdown.com/"
    "gen9randombattle-2405042449-9irailjicjrb4g5sr5r0v1j7tthvqudpw"
)

# Latest relevant randbats repository revision before the July 2025 replay.
SHOWDOWN_COMMIT = "6397bfddb3db4e916dd792e03c43355f7366e8ab"
POKE_ENV_VERSION = "0.16.1"
RANDBATS_EV = 85
RANDBATS_IV = 31

EXPECTED_GENERATOR_CONTEXT = {
    "format": "gen9randombattle",
    "teamDetails": {},
    "isLead": False,
    "isDoubles": False,
}

# (physical attack multiplier, final damage multiplier)
# Life Orb uses Showdown's historical fixed-point 5324/4096 modifier.
ITEM_DAMAGE_MODELS: dict[str, tuple[Fraction, Fraction]] = {
    "Choice Band": (Fraction(3, 2), Fraction(1, 1)),
    "Choice Scarf": (Fraction(1, 1), Fraction(1, 1)),
    "Life Orb": (Fraction(1, 1), Fraction(5324, 4096)),
}

NON_SEMANTIC_PROTOCOL_KINDS = {"", "t:", "-hint"}


class ReplayError(ValueError):
    """Raised when replay or hidden-world evidence cannot be interpreted safely."""


@dataclass(frozen=True)
class ProtocolEvent:
    index: int
    turn: int
    kind: str
    fields: tuple[str, ...]


@dataclass(frozen=True)
class ReplayEvidence:
    replay_id: str
    format: str | None
    log: str


@dataclass(frozen=True)
class DamageObservation:
    event_index: int
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


def extract_damage_observations(
    events: Sequence[ProtocolEvent],
) -> tuple[DamageObservation, ...]:
    """Pair public move and HP messages without consulting later hidden information."""
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

        if event.kind not in {"-damage", "-heal"} or len(fields) < 2:
            continue

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
                        event_index=event.index,
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


def _normalized_generator_context(value: object) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: item
        for key, item in value.items()
        if key not in {"publicLevel", "publicAbility"} or item is not None
    }


def load_world_sample(path: str | Path) -> dict[str, Any]:
    sample_path = Path(path)
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReplayError(f"{sample_path}: world sample must be an object")
    if payload.get("schema") != "azelficoast.showdown-world-sample":
        raise ReplayError(f"{sample_path}: unexpected world-sample schema")
    if payload.get("schema_version") != 1:
        raise ReplayError(f"{sample_path}: unsupported world-sample schema version")
    if payload.get("showdown_commit") != SHOWDOWN_COMMIT:
        raise ReplayError(
            f"{sample_path}: sample is not bound to Showdown {SHOWDOWN_COMMIT}"
        )
    if _normalized_generator_context(payload.get("generator_context")) != EXPECTED_GENERATOR_CONTEXT:
        raise ReplayError(f"{sample_path}: unexpected generator context")

    rounds = payload.get("rounds")
    matched = payload.get("matched")
    if not isinstance(rounds, int) or rounds <= 0:
        raise ReplayError(f"{sample_path}: invalid rounds")
    if not isinstance(matched, int) or matched <= 0 or matched > rounds:
        raise ReplayError(f"{sample_path}: invalid matched count")

    item_counts = payload.get("item_counts")
    if not isinstance(item_counts, dict) or not item_counts:
        raise ReplayError(f"{sample_path}: missing item counts")
    for item, count in item_counts.items():
        if not isinstance(item, str) or not isinstance(count, int) or count <= 0:
            raise ReplayError(f"{sample_path}: malformed item count")
    if sum(item_counts.values()) != matched:
        raise ReplayError(f"{sample_path}: item counts do not sum to matched worlds")

    return payload


def _neutral_stat(base: int, level: int) -> int:
    return math.floor((2 * base + RANDBATS_IV + RANDBATS_EV // 4) * level / 100) + 5


def _showdown_modify(value: int, modifier: Fraction) -> int:
    """Mirror historical Showdown's 12-bit Battle.modify for positive values."""
    fixed = math.floor(modifier.numerator * 4096 / modifier.denominator)
    return math.floor((math.floor(value * fixed) + 2048 - 1) / 4096)


def _effectiveness(
    data: GenData,
    attacking_type: str,
    defending_types: Sequence[str],
) -> Fraction:
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
    """Calculate the 16 ordinary damage rolls for one recognized item world."""
    if item not in ITEM_DAMAGE_MODELS:
        raise ReplayError(
            f"no damage model for sampled item {item!r}; refusing to guess"
        )

    data = GenData.from_gen(9)
    attacker = data.pokedex.get(_to_id(observation.attacker_species))
    target = data.pokedex.get(_to_id(observation.target_species))
    move = data.moves.get(_to_id(observation.move))
    if (
        not isinstance(attacker, dict)
        or not isinstance(target, dict)
        or not isinstance(move, dict)
    ):
        raise ReplayError("species or move is missing from pinned poke-env mechanics data")
    if move.get("category") != "Physical":
        raise ReplayError("current replay conditioner supports physical damage only")

    base_power = move.get("basePower")
    move_type = move.get("type")
    if not isinstance(base_power, int) or not isinstance(move_type, str):
        raise ReplayError("move lacks ordinary base power/type")

    attack_multiplier, final_multiplier = ITEM_DAMAGE_MODELS[item]
    attack = _showdown_modify(
        _neutral_stat(attacker["baseStats"]["atk"], observation.attacker_level),
        attack_multiplier,
    )
    defense = _neutral_stat(target["baseStats"]["def"], observation.target_level)

    base_damage = (
        math.floor(
            math.floor(
                math.floor(2 * observation.attacker_level / 5 + 2)
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
        # Historical Showdown randomizes before STAB, type and final modifiers.
        damage = math.floor(base_damage * random_percent / 100)
        damage = _showdown_modify(damage, stab)
        damage = math.floor(damage * effectiveness)
        damage = _showdown_modify(damage, final_multiplier)
        rolls.append(damage)
    return tuple(rolls)


def _turn_is_complete(events: Sequence[ProtocolEvent], turn: int) -> bool:
    return any(
        event.kind == "turn"
        and event.fields
        and event.fields[0].isdigit()
        and int(event.fields[0]) > turn
        for event in events
    )


def _life_orb_recoil_observed(
    events: Sequence[ProtocolEvent],
    *,
    species: str,
    turn: int,
) -> bool:
    active: dict[str, str] = {}
    for event in events:
        if event.turn > turn:
            break

        fields = event.fields
        if event.kind in {"switch", "drag"} and len(fields) >= 2:
            active[_slot_from_actor(fields[0])] = _species_from_details(fields[1])[0]
            continue

        if event.turn != turn or event.kind != "-damage" or len(fields) < 3:
            continue
        slot = _slot_from_actor(fields[0])
        if active.get(slot) != species:
            continue
        if any("item: Life Orb" in field for field in fields[2:]):
            return True
    return False


def condition_generator_prior_on_public_history(
    sample: Mapping[str, Any],
    events: Sequence[ProtocolEvent],
) -> dict[str, Any]:
    """Apply authoritative public observations preceding the target damage event."""
    raw_counts = sample.get("item_counts")
    if not isinstance(raw_counts, dict):
        raise ReplayError("world sample has no item counts")
    counts = dict(raw_counts)

    damaging_turn_18 = [
        observation
        for observation in extract_damage_observations(events)
        if observation.turn == 18
        and observation.attacker_species == "Infernape"
        and observation.move == "Close Combat"
    ]
    if len(damaging_turn_18) != 1:
        raise ReplayError(
            "expected exactly one turn-18 Infernape Close Combat damage observation"
        )
    if not _turn_is_complete(events, 18):
        raise ReplayError("turn 18 is incomplete; cannot certify Life Orb recoil absence")

    recoil_observed = _life_orb_recoil_observed(
        events,
        species="Infernape",
        turn=18,
    )
    if "Life Orb" in counts:
        if recoil_observed:
            counts = {"Life Orb": counts["Life Orb"]}
        else:
            del counts["Life Orb"]

    if not counts:
        raise ReplayError("public history eliminated every sampled item world")

    return {
        "item_counts": counts,
        "updates": [
            {
                "turn": 18,
                "kind": "life-orb-recoil",
                "observed": recoil_observed,
                "authority": "complete-public-turn",
                "remaining_items": sorted(counts),
            }
        ],
    }


def infer_item_posterior(
    sample: Mapping[str, Any],
    observation: DamageObservation,
    *,
    item_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Condition empirical generator item mass on exact public damage."""
    species = sample.get("species")
    if (
        not isinstance(species, str)
        or _to_id(species) != _to_id(observation.attacker_species)
    ):
        raise ReplayError("world sample species does not match damage attacker")

    if item_counts is None:
        raw_item_counts = sample.get("item_counts")
        if not isinstance(raw_item_counts, dict):
            raise ReplayError("world sample has no item counts")
        item_counts = raw_item_counts

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


def _target_replay_observation(
    events: Sequence[ProtocolEvent],
) -> DamageObservation:
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
            "expected exactly one turn-19 Infernape Close Combat observation, "
            f"got {len(matches)}"
        )

    observation = matches[0]
    if observation.target_tera_type != "Flying":
        raise ReplayError(
            f"expected Tera Flying Kingambit, got {observation.target_tera_type!r}"
        )
    return observation


def _public_prefix_sha256(
    events: Sequence[ProtocolEvent],
    *,
    through_index: int,
) -> str:
    semantic_prefix = [
        {
            "turn": event.turn,
            "kind": event.kind,
            "fields": list(event.fields),
        }
        for event in events
        if event.index <= through_index
        and event.kind not in NON_SEMANTIC_PROTOCOL_KINDS
    ]
    return _sha256(semantic_prefix)


def build_replay_belief(
    replay: ReplayEvidence,
    events: Sequence[ProtocolEvent],
    sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a deterministic belief certificate from acquired public evidence."""
    if sample.get("showdown_commit") != SHOWDOWN_COMMIT:
        raise ReplayError("world sample is not bound to the expected Showdown revision")
    if _normalized_generator_context(sample.get("generator_context")) != EXPECTED_GENERATOR_CONTEXT:
        raise ReplayError("world sample uses an unexpected generator context")
    if sample.get("observed_moves") != ["closecombat"]:
        raise ReplayError(
            "world sample must use exactly the moves public before the turn-19 update"
        )

    observation = _target_replay_observation(events)
    history_condition = condition_generator_prior_on_public_history(sample, events)
    inference = infer_item_posterior(
        sample,
        observation,
        item_counts=history_condition["item_counts"],
    )

    raw_item_counts = sample["item_counts"]
    raw_total = sum(raw_item_counts.values())
    generator_prior = {
        item: count / raw_total
        for item, count in sorted(raw_item_counts.items())
    }

    evidence = {
        "replay_id": replay.replay_id,
        "format": replay.format,
        "showdown_commit": SHOWDOWN_COMMIT,
        "mechanics_source": {
            "poke_env": POKE_ENV_VERSION,
            "showdown_damage_semantics": SHOWDOWN_COMMIT,
        },
        "public_prefix_sha256": _public_prefix_sha256(
            events,
            through_index=observation.event_index,
        ),
        "sample": {
            "species": sample["species"],
            "observed_moves": sample["observed_moves"],
            "showdown_commit": sample["showdown_commit"],
            "seed_family": sample.get("seed_family"),
            "generator_context": sample["generator_context"],
            "rounds": sample["rounds"],
            "matched": sample["matched"],
            "item_counts": sample["item_counts"],
            "sample_sha256": _sha256(sample),
        },
        "public_history_updates": history_condition["updates"],
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
        "generator_prior": generator_prior,
        "prior": inference["prior"],
        "compatible_worlds": inference["compatible"],
        "posterior": inference["posterior"],
    }

    return {
        "schema": "azelficoast.replay-belief",
        "schema_version": 1,
        **evidence,
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default=REPLAY_URL)
    parser.add_argument(
        "--world-sample",
        type=Path,
        required=True,
        help="Showdown-generated world sample to condition on replay evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = reconstruct_replay_belief(args.world_sample, args.replay)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
