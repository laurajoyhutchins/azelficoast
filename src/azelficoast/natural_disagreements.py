"""Mine real decision fixtures for hidden-item strategy-fusion candidates."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from poke_env.data import GenData

from azelficoast.corpus import DecisionFixture, load_corpus

RANDBATS_EV = 85
RANDBATS_IV = 31

SPEED_ABILITIES = {
    "chlorophyll",
    "protosynthesis",
    "quarkdrive",
    "quickfeet",
    "sandrush",
    "slushrush",
    "speedboost",
    "surgesurfer",
    "swiftswim",
    "unburden",
}

ORDER_ABILITIES = SPEED_ABILITIES | {
    "galewings",
    "myceliummight",
    "prankster",
    "quickdraw",
    "stall",
    "triage",
}

ORDER_ITEMS = {
    "fullincense",
    "laggingtail",
    "quickclaw",
}

PROTECT_MOVES = frozenset(
    {
        "banefulbunker",
        "burningbulwark",
        "detect",
        "kingsshield",
        "obstruct",
        "protect",
        "silktrap",
        "spikyshield",
    }
)

TYPE_IMMUNITIES: dict[str, frozenset[str]] = {
    "normal": frozenset({"ghost"}),
    "fighting": frozenset({"ghost"}),
    "poison": frozenset({"steel"}),
    "ground": frozenset({"flying"}),
    "ghost": frozenset({"normal"}),
    "electric": frozenset({"ground"}),
    "psychic": frozenset({"dark"}),
    "dragon": frozenset({"fairy"}),
}


class NaturalDisagreementError(ValueError):
    """Raised when frozen evidence cannot be interpreted safely."""


class UnsupportedWorldSample(NaturalDisagreementError):
    """Raised when public evidence is outside the generator-derived world model."""


@dataclass(frozen=True)
class PublicEvent:
    index: int
    turn: int
    kind: str
    fields: tuple[str, ...]


def _to_id(value: str | None) -> str:
    if value is None:
        return ""
    return "".join(character for character in value.lower() if character.isalnum())


def _item_is_hidden(value: Any) -> bool:
    return value is None or _to_id(str(value)) == _to_id(GenData.UNKNOWN_ITEM)


def _slot_side(actor: str) -> str | None:
    slot = actor.split(":", 1)[0].strip()
    if slot.startswith("p1"):
        return "p1"
    if slot.startswith("p2"):
        return "p2"
    return None


def _species_from_details(details: str) -> str:
    return details.split(",", 1)[0].strip()


def public_events(fixture: DecisionFixture) -> tuple[PublicEvent, ...]:
    events: list[PublicEvent] = []
    turn = 0
    index = 0
    for batch in fixture.protocol_prefix:
        for message in batch:
            if len(message) < 2 or message[0] != "":
                continue
            kind = message[1]
            fields = tuple(message[2:])
            if kind == "turn" and fields and fields[0].isdigit():
                turn = int(fields[0])
            events.append(PublicEvent(index=index, turn=turn, kind=kind, fields=fields))
            index += 1
    return tuple(events)


def _opponent_side(
    fixture: DecisionFixture,
    events: Sequence[PublicEvent],
) -> str | None:
    opponent = str(fixture.state.get("opponent") or "")
    for event in events:
        if event.kind == "player" and len(event.fields) >= 2:
            side, username = event.fields[0], event.fields[1]
            if username.casefold() == opponent.casefold() and side in {"p1", "p2"}:
                return side

    opponent_active = fixture.state.get("opponent_active")
    if not isinstance(opponent_active, Mapping):
        return None
    species = opponent_active.get("species")
    if not isinstance(species, str):
        return None

    active: dict[str, str] = {}
    for event in events:
        if event.kind in {"switch", "drag"} and len(event.fields) >= 2:
            side = _slot_side(event.fields[0])
            if side is not None:
                active[side] = _species_from_details(event.fields[1])
    matches = [
        side
        for side, active_species in active.items()
        if _to_id(active_species) == _to_id(species)
    ]
    return matches[0] if len(matches) == 1 else None


def _active_history(
    events: Sequence[PublicEvent],
    side: str,
) -> dict[str, Any]:
    active_species: str | None = None
    lead_species: str | None = None
    revealed_moves: dict[str, set[str]] = {}
    move_events: list[dict[str, Any]] = []

    for event in events:
        if event.kind in {"switch", "drag"} and len(event.fields) >= 2:
            event_side = _slot_side(event.fields[0])
            if event_side != side:
                continue
            active_species = _species_from_details(event.fields[1])
            if lead_species is None:
                lead_species = active_species
            revealed_moves.setdefault(_to_id(active_species), set())
            continue

        if event.kind != "move" or len(event.fields) < 2:
            continue
        event_side = _slot_side(event.fields[0])
        if event_side != side or active_species is None:
            continue
        move = event.fields[1]
        revealed_moves.setdefault(_to_id(active_species), set()).add(move)
        move_events.append(
            {
                "event_index": event.index,
                "turn": event.turn,
                "species": active_species,
                "move": move,
                "target": event.fields[2] if len(event.fields) >= 3 else None,
            }
        )

    return {
        "active_species": active_species,
        "lead_species": lead_species,
        "revealed_moves": revealed_moves,
        "move_events": move_events,
    }


def _move_caused_damage(
    events: Sequence[PublicEvent],
    *,
    move_event: Mapping[str, Any],
) -> bool:
    target = move_event.get("target")
    target_side = _slot_side(str(target)) if target is not None else None
    if target_side is None:
        return False

    start = int(move_event["event_index"])
    turn = int(move_event["turn"])
    for event in events:
        if event.index <= start:
            continue
        if event.turn != turn or event.kind in {"move", "turn"}:
            if event.turn != turn or event.kind == "move":
                break
        if event.kind == "-damage" and event.fields:
            if _slot_side(event.fields[0]) == target_side:
                return True
    return False


def _life_orb_recoil_observed(
    events: Sequence[PublicEvent],
    *,
    side: str,
    turn: int,
    after_event_index: int,
) -> bool:
    for event in events:
        if event.index <= after_event_index:
            continue
        if event.turn != turn:
            if event.turn > turn:
                break
            continue
        if event.kind == "move":
            break
        if event.kind != "-damage" or len(event.fields) < 3:
            continue
        if _slot_side(event.fields[0]) != side:
            continue
        if any("item: Life Orb" in field for field in event.fields[2:]):
            return True
    return False


def _has_condition(
    conditions: Mapping[str, Any] | None,
    name: str,
) -> bool:
    return isinstance(conditions, Mapping) and any(
        name in str(key).upper() for key in conditions
    )


def _plain_speed_context(fixture: DecisionFixture) -> bool:
    fields = fixture.state.get("fields")
    return not _has_condition(
        fields if isinstance(fields, Mapping) else None,
        "TRICK_ROOM",
    )


def _apply_speed_stage(speed: int, view: Mapping[str, Any]) -> int:
    boosts = view.get("boosts")
    stage = 0
    if isinstance(boosts, Mapping):
        raw = boosts.get("spe", 0)
        if isinstance(raw, (int, float)):
            stage = int(raw)
    stage = max(-6, min(6, stage))
    if stage >= 0:
        speed = speed * (2 + stage) // 2
    else:
        speed = speed * 2 // (2 - stage)
    return max(1, speed)


def _apply_public_speed_modifiers(
    speed: int,
    view: Mapping[str, Any],
    side_conditions: Mapping[str, Any] | None,
) -> int:
    speed = _apply_speed_stage(speed, view)
    status = _to_id(str(view.get("status") or ""))
    if status in {"par", "paralysis"}:
        speed = max(1, speed // 2)
    if _has_condition(side_conditions, "TAILWIND"):
        speed *= 2
    return speed


def _neutral_speed(species: str, level: int) -> int:
    data = GenData.from_gen(9)
    entry = data.pokedex.get(_to_id(species))
    if not isinstance(entry, Mapping):
        raise NaturalDisagreementError(f"unknown species {species!r}")
    base_speed = int(entry["baseStats"]["spe"])
    return int((2 * base_speed + RANDBATS_IV + RANDBATS_EV // 4) * level / 100) + 5


def _effective_own_speed(
    active: Mapping[str, Any],
    side_conditions: Mapping[str, Any] | None,
) -> int:
    stats = active.get("stats")
    if not isinstance(stats, Mapping) or not isinstance(stats.get("spe"), int):
        raise NaturalDisagreementError("active state lacks exact speed")
    speed = int(stats["spe"])
    if _to_id(str(active.get("item") or "")) == "choicescarf":
        speed = speed * 3 // 2
    return _apply_public_speed_modifiers(speed, active, side_conditions)


def _move_type(move_id: str) -> str | None:
    move = GenData.from_gen(9).moves.get(_to_id(move_id))
    if not isinstance(move, Mapping):
        return None
    move_type = move.get("type")
    return move_type.lower() if isinstance(move_type, str) else None


def _move_order_metadata(move_id: str) -> tuple[int, str] | None:
    move = GenData.from_gen(9).moves.get(_to_id(move_id))
    if not isinstance(move, Mapping):
        return None
    priority = move.get("priority", 0)
    category = move.get("category")
    if not isinstance(priority, int) or category not in {"Physical", "Special"}:
        return None
    return priority, str(category)


def _speed_context_mutated_after(
    events: Sequence[PublicEvent],
    *,
    start_index: int,
    sides: frozenset[str],
) -> bool:
    for event in events:
        if event.index <= start_index:
            continue

        event_side = _slot_side(event.fields[0]) if event.fields else None
        if event.kind in {"switch", "drag"} and event_side in sides:
            return True

        if event.kind in {"-boost", "-unboost", "-setboost"}:
            if (
                event_side in sides
                and len(event.fields) >= 2
                and _to_id(event.fields[1]) == "spe"
            ):
                return True
        if event.kind in {
            "-clearboost",
            "-clearallboost",
            "-copyboost",
            "-swapboost",
            "-invertboost",
        } and (event_side in sides or event.kind == "-clearallboost"):
            return True

        if event.kind in {"-status", "-curestatus"}:
            if (
                event_side in sides
                and len(event.fields) >= 2
                and _to_id(event.fields[1]) in {"par", "paralysis"}
            ):
                return True

        if event.kind in {"-item", "-enditem"} and event_side in sides:
            return True

        lowered = " ".join(str(field).lower() for field in event.fields)
        if event.kind in {"-sidestart", "-sideend"} and "tailwind" in lowered:
            return True
        if event.kind in {"-fieldstart", "-fieldend"} and "trick room" in lowered:
            return True
    return False


def _prior_choice_item_order_evidence(
    fixture: DecisionFixture,
    events: Sequence[PublicEvent],
    *,
    opponent_side: str,
    opponent_history: Mapping[str, Any],
    last_move: Mapping[str, Any],
    pair: frozenset[str],
    own_speed: int | None,
    base_speed: int,
    scarf_speed: int,
    variants: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if own_speed is None:
        return None

    active = fixture.state.get("active")
    if not isinstance(active, Mapping):
        return None
    if _to_id(str(active.get("ability") or "")) in ORDER_ABILITIES:
        return None
    if _to_id(str(active.get("item") or "")) in ORDER_ITEMS:
        return None
    if any(
        _to_id(str(variant.get("ability") or "")) in ORDER_ABILITIES
        for variant in variants
    ):
        return None

    own_side = "p1" if opponent_side == "p2" else "p2"
    own_history = _active_history(events, own_side)
    if _to_id(str(own_history["active_species"])) != _to_id(
        str(active.get("species") or "")
    ):
        return None

    turn = int(last_move["turn"])
    opponent_species = _to_id(str(opponent_history["active_species"]))
    own_species = _to_id(str(own_history["active_species"]))
    opponent_moves = [
        event
        for event in opponent_history["move_events"]
        if int(event["turn"]) == turn
        and _to_id(str(event["species"])) == opponent_species
    ]
    own_moves = [
        event
        for event in own_history["move_events"]
        if int(event["turn"]) == turn
        and _to_id(str(event["species"])) == own_species
    ]
    if len(opponent_moves) != 1 or len(own_moves) != 1:
        return None

    opponent_move = opponent_moves[0]
    own_move = own_moves[0]
    opponent_meta = _move_order_metadata(str(opponent_move["move"]))
    own_meta = _move_order_metadata(str(own_move["move"]))
    if opponent_meta is None or own_meta is None:
        return None
    if opponent_meta[0] != own_meta[0]:
        return None

    first_index = min(
        int(opponent_move["event_index"]),
        int(own_move["event_index"]),
    )
    if _speed_context_mutated_after(
        events,
        start_index=first_index,
        sides=frozenset({own_side, opponent_side}),
    ):
        return None

    own_before = int(own_move["event_index"]) < int(opponent_move["event_index"])
    allowed: list[str] = []
    for item in sorted(pair):
        opponent_speed = scarf_speed if item == "Choice Scarf" else base_speed
        if opponent_speed == own_speed:
            allowed.append(item)
            continue
        if own_before == (own_speed > opponent_speed):
            allowed.append(item)

    return {
        "turn": turn,
        "own_move": own_move["move"],
        "opponent_move": opponent_move["move"],
        "own_before_opponent": own_before,
        "own_speed": own_speed,
        "opponent_base_speed": base_speed,
        "opponent_scarf_speed": scarf_speed,
        "allowed_items": allowed,
    }


def _protect_blocks_locked_move(
    fixture: DecisionFixture,
    move_id: str,
) -> bool:
    move = GenData.from_gen(9).moves.get(_to_id(move_id))
    if not isinstance(move, Mapping):
        return False
    flags = move.get("flags")
    if not isinstance(flags, Mapping) or not flags.get("protect"):
        return False

    opponent = fixture.state.get("opponent_active")
    ability = ""
    if isinstance(opponent, Mapping):
        raw_ability = opponent.get("ability")
        if isinstance(raw_ability, str):
            ability = _to_id(raw_ability)

    if ability == "unseenfist" and bool(flags.get("contact")):
        return False
    return True


def _type_immune(view: Mapping[str, Any], move_type: str) -> bool:
    immune_types = TYPE_IMMUNITIES.get(move_type)
    if not immune_types:
        return False
    raw_types = view.get("types")
    if not isinstance(raw_types, Sequence) or isinstance(raw_types, (str, bytes)):
        return False
    return bool({_to_id(str(type_)) for type_ in raw_types} & immune_types)


def _switch_view(
    fixture: DecisionFixture,
    species: str,
) -> Mapping[str, Any] | None:
    team = fixture.state.get("team")
    if not isinstance(team, Mapping):
        return None
    matches = [
        view
        for view in team.values()
        if isinstance(view, Mapping)
        and _to_id(str(view.get("species") or "")) == _to_id(species)
    ]
    return matches[0] if len(matches) == 1 else None


def _persistent_protect_actions(
    fixture: DecisionFixture,
    *,
    locked_move: str | None = None,
) -> list[dict[str, Any]]:
    if locked_move is not None and not _protect_blocks_locked_move(
        fixture,
        locked_move,
    ):
        return []

    persistent: list[dict[str, Any]] = []
    for action in fixture.legal_actions:
        if not action.startswith("/choose move "):
            continue
        move = action.removeprefix("/choose move ").split(" ", 1)[0]
        if _to_id(move) not in PROTECT_MOVES:
            continue
        persistent.append(
            {
                "action": action,
                "kind": "protect",
                "observation": "blocked-no-item-reveal",
            }
        )
    return persistent


def _persistent_immunity_switches(
    fixture: DecisionFixture,
    *,
    move_type: str,
) -> list[dict[str, Any]]:
    raw_switches = fixture.state.get("available_switches")
    if not isinstance(raw_switches, Sequence) or isinstance(raw_switches, (str, bytes)):
        return []

    own_conditions = fixture.state.get("side_conditions")
    conditions = own_conditions if isinstance(own_conditions, Mapping) else None
    persistent: list[dict[str, Any]] = []

    for species in raw_switches:
        if not isinstance(species, str):
            continue
        view = _switch_view(fixture, species)
        if view is None or view.get("fainted") is True:
            continue
        if view.get("transformed") is True:
            continue
        if _to_id(str(view.get("ability") or "")) in SPEED_ABILITIES:
            continue
        if not _type_immune(view, move_type):
            continue

        speed = _effective_own_speed(view, conditions)

        action = next(
            (
                legal
                for legal in fixture.legal_actions
                if legal.startswith("/choose switch ")
                and _to_id(legal.removeprefix("/choose switch ")) == _to_id(species)
            ),
            None,
        )
        if action is None:
            continue
        persistent.append(
            {
                "action": action,
                "species": view.get("species"),
                "speed": speed,
                "observation": f"immune:{move_type}",
            }
        )

    return sorted(
        persistent,
        key=lambda item: (str(item["species"]), str(item["action"])),
    )


def _sample_worlds(
    *,
    showdown_root: Path,
    species: str,
    observed_moves: Sequence[str],
    rounds: int,
    is_lead: bool,
    public_level: int,
) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "scripts" / "sample_showdown_worlds.cjs"
    try:
        completed = subprocess.run(
            [
                "node",
                str(script),
                str(showdown_root),
                species,
                ",".join(observed_moves),
                str(rounds),
                "true" if is_lead else "false",
                "0",
                str(public_level),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        if error.returncode == 2:
            detail = (error.stderr or "").strip() or "unsupported generator evidence"
            raise UnsupportedWorldSample(detail) from error
        raise
    payload = json.loads(completed.stdout)
    if not isinstance(payload, dict):
        raise NaturalDisagreementError("world sampler did not return an object")
    return payload


def _condition_item_counts(
    *,
    sample: Mapping[str, Any],
    events: Sequence[PublicEvent],
    opponent_side: str,
    last_move: Mapping[str, Any],
) -> dict[str, int]:
    raw = sample.get("item_counts")
    if not isinstance(raw, Mapping):
        raise NaturalDisagreementError("world sample lacks item counts")
    counts = {str(item): int(count) for item, count in raw.items()}

    if "Life Orb" in counts and _move_caused_damage(events, move_event=last_move):
        recoil = _life_orb_recoil_observed(
            events,
            side=opponent_side,
            turn=int(last_move["turn"]),
            after_event_index=int(last_move["event_index"]),
        )
        if not recoil:
            del counts["Life Orb"]

    return counts


def _choice_pair_for_move(move_id: str) -> frozenset[str] | None:
    data = GenData.from_gen(9)
    move = data.moves.get(_to_id(move_id))
    if not isinstance(move, Mapping):
        return None
    category = move.get("category")
    if category == "Physical":
        return frozenset({"Choice Band", "Choice Scarf"})
    if category == "Special":
        return frozenset({"Choice Specs", "Choice Scarf"})
    return None


def mine_candidates(
    fixtures: Iterable[DecisionFixture],
    *,
    showdown_root: Path,
    rounds: int = 2048,
    persistent_only: bool = False,
) -> dict[str, Any]:
    cache: dict[tuple[str, tuple[str, ...], bool], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    excluded_fixtures: list[dict[str, str]] = []
    current_fixture_id: str | None = None

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1
        if current_fixture_id is None:
            raise NaturalDisagreementError("exclusion lacks fixture identity")
        excluded_fixtures.append(
            {"fixture_id": current_fixture_id, "reason": reason}
        )

    fixture_count = 0
    for fixture in fixtures:
        fixture_count += 1
        current_fixture_id = fixture.fixture_id
        legal_actions = fixture.legal_actions
        if not _plain_speed_context(fixture):
            skip("non-plain-speed-context")
            continue

        active = fixture.state.get("active")
        opponent_active = fixture.state.get("opponent_active")
        if not isinstance(active, Mapping) or not isinstance(opponent_active, Mapping):
            skip("missing-active-state")
            continue
        if active.get("transformed") is True:
            skip("active-transformed")
            continue
        if not _item_is_hidden(opponent_active.get("item")):
            skip("opponent-item-known")
            continue

        opponent_species = opponent_active.get("species")
        opponent_level = opponent_active.get("level")
        if not isinstance(opponent_species, str) or not isinstance(opponent_level, int):
            skip("opponent-level-unknown")
            continue

        events = public_events(fixture)
        side = _opponent_side(fixture, events)
        if side is None:
            skip("opponent-side-unresolved")
            continue
        history = _active_history(events, side)
        if _to_id(str(history["active_species"])) != _to_id(opponent_species):
            skip("active-history-mismatch")
            continue

        revealed = sorted(
            history["revealed_moves"].get(_to_id(opponent_species), set())
        )
        if not revealed:
            skip("no-revealed-opponent-move")
            continue

        current_moves = [
            item
            for item in history["move_events"]
            if _to_id(str(item["species"])) == _to_id(opponent_species)
        ]
        if not current_moves:
            skip("no-current-move-event")
            continue
        last_move = current_moves[-1]

        pair = _choice_pair_for_move(str(last_move["move"]))
        if pair is None:
            skip("last-move-not-fixed-damage-category")
            continue

        own_conditions = fixture.state.get("side_conditions")
        opponent_conditions = fixture.state.get("opponent_side_conditions")
        base_speed = _apply_public_speed_modifiers(
            _neutral_speed(opponent_species, opponent_level),
            opponent_active,
            opponent_conditions if isinstance(opponent_conditions, Mapping) else None,
        )
        scarf_speed = _apply_public_speed_modifiers(
            _neutral_speed(opponent_species, opponent_level) * 3 // 2,
            opponent_active,
            opponent_conditions if isinstance(opponent_conditions, Mapping) else None,
        )
        low, high = sorted((base_speed, scarf_speed))

        own_speed = None
        current_speed_fork = False
        if _to_id(str(active.get("ability") or "")) not in SPEED_ABILITIES:
            own_speed = _effective_own_speed(
                active,
                own_conditions if isinstance(own_conditions, Mapping) else None,
            )
            current_speed_fork = low < own_speed < high

        move_type = _move_type(str(last_move["move"]))
        persistent_switches = (
            _persistent_immunity_switches(
                fixture,
                move_type=move_type,
            )
            if move_type is not None
            else []
        )
        persistent_protect_actions = _persistent_protect_actions(
            fixture,
            locked_move=str(last_move["move"]),
        )
        has_persistent_branch = bool(
            persistent_switches or persistent_protect_actions
        )
        if persistent_only and not has_persistent_branch:
            skip("no-persistent-information-branch")
            continue
        if (
            not current_speed_fork
            and not has_persistent_branch
        ):
            skip("no-speed-or-persistent-information-fork")
            continue

        is_lead = _to_id(str(history["lead_species"])) == _to_id(opponent_species)
        key = (opponent_species, tuple(revealed), is_lead)
        if key not in cache:
            try:
                cache[key] = _sample_worlds(
                    showdown_root=showdown_root,
                    species=opponent_species,
                    observed_moves=revealed,
                    rounds=rounds,
                    is_lead=is_lead,
                    public_level=opponent_level,
                )
            except UnsupportedWorldSample:
                skip("world-sample-unsupported")
                continue
        sample = cache[key]
        if sample.get("showdown_commit") is None:
            raise NaturalDisagreementError("world sample lacks Showdown revision")

        variants = sample.get("variants")
        if not isinstance(variants, list):
            raise NaturalDisagreementError("world sample lacks variants")
        if any(
            isinstance(variant, Mapping)
            and _to_id(str(variant.get("ability") or "")) in SPEED_ABILITIES
            for variant in variants
        ):
            skip("opponent-speed-ability")
            continue

        counts = _condition_item_counts(
            sample=sample,
            events=events,
            opponent_side=side,
            last_move=last_move,
        )

        order_evidence = _prior_choice_item_order_evidence(
            fixture,
            events,
            opponent_side=side,
            opponent_history=history,
            last_move=last_move,
            pair=pair,
            own_speed=own_speed,
            base_speed=base_speed,
            scarf_speed=scarf_speed,
            variants=[
                variant
                for variant in variants
                if isinstance(variant, Mapping)
            ],
        )
        if order_evidence is not None:
            allowed_items = set(order_evidence["allowed_items"])
            counts = {
                item: count
                for item, count in counts.items()
                if item in allowed_items
            }

        if frozenset(counts) != pair:
            if order_evidence is not None:
                skip("choice-world-eliminated-by-prior-speed-order")
            else:
                skip("not-choice-pair-after-public-evidence")
            continue

        total = sum(counts.values())
        candidate = {
            "fixture_id": fixture.fixture_id,
            "turn": fixture.state.get("turn"),
            "player": fixture.state.get("player"),
            "opponent": fixture.state.get("opponent"),
            "active_species": active.get("species"),
            "active_speed": own_speed,
            "current_speed_fork": current_speed_fork,
            "persistent_switches": persistent_switches,
            "persistent_protect_actions": persistent_protect_actions,
            "opponent_species": opponent_species,
            "generator_species": sample["species"],
            "opponent_level": opponent_level,
            "opponent_base_speed": base_speed,
            "opponent_scarf_speed": scarf_speed,
            "revealed_moves": revealed,
            "locked_move": last_move["move"],
            "is_lead": is_lead,
            "item_counts": counts,
            "item_weights": {
                item: count / total for item, count in sorted(counts.items())
            },
            "prior_speed_order_evidence": order_evidence,
            "legal_actions": list(legal_actions),
            "protect_legal": "/choose move protect" in legal_actions,
            "control_actions": [
                control.get("chosen_action") for control in fixture.control_decisions
            ],
            "showdown_commit": sample["showdown_commit"],
            "sample_rounds": sample["rounds"],
        }
        candidates.append(candidate)

    candidates.sort(key=lambda item: item["fixture_id"])
    return {
        "schema": "azelficoast.natural-fusion-candidates",
        "schema_version": 1,
        "fixture_count": fixture_count,
        "persistent_only": persistent_only,
        "sampled_world_queries": len(cache),
        "candidate_count": len(candidates),
        "persistent_candidate_count": sum(
            bool(candidate["persistent_switches"])
            or bool(candidate["persistent_protect_actions"])
            for candidate in candidates
        ),
        "candidates": candidates,
        "excluded_fixtures": sorted(
            excluded_fixtures,
            key=lambda row: row["fixture_id"],
        ),
        "skipped": dict(sorted(skipped.items())),
    }


def mine_corpus(
    corpus_path: str | Path,
    *,
    showdown_root: str | Path,
    rounds: int = 2048,
    persistent_only: bool = False,
) -> dict[str, Any]:
    return mine_candidates(
        load_corpus(corpus_path),
        showdown_root=Path(showdown_root),
        rounds=rounds,
        persistent_only=persistent_only,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2048)
    parser.add_argument(
        "--persistent-only",
        action="store_true",
        help="mine only branches that preserve hidden worlds into the next decision",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    result = mine_corpus(
        args.corpus,
        showdown_root=args.showdown_root,
        rounds=args.rounds,
        persistent_only=args.persistent_only,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
