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


class NaturalDisagreementError(ValueError):
    """Raised when frozen evidence cannot be interpreted safely."""


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
    state = fixture.state
    fields = state.get("fields")
    if _has_condition(fields if isinstance(fields, Mapping) else None, "TRICK_ROOM"):
        return False

    active = state.get("active")
    if not isinstance(active, Mapping):
        return False
    return _to_id(str(active.get("ability") or "")) not in SPEED_ABILITIES


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


def _generator_species(species: str) -> str:
    """Map an observed battle forme to the species key used by randbats generation."""
    data = GenData.from_gen(9)
    entry = data.pokedex.get(_to_id(species))
    if not isinstance(entry, Mapping):
        raise NaturalDisagreementError(f"unknown species {species!r}")
    battle_only = entry.get("battleOnly")
    if isinstance(battle_only, str):
        return battle_only
    base_species = entry.get("baseSpecies")
    if isinstance(base_species, str):
        return base_species
    name = entry.get("name")
    return str(name) if isinstance(name, str) else species


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


def _sample_worlds(
    *,
    showdown_root: Path,
    species: str,
    observed_moves: Sequence[str],
    rounds: int,
    is_lead: bool,
) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "scripts" / "sample_showdown_worlds.cjs"
    completed = subprocess.run(
        [
            "node",
            str(script),
            str(showdown_root),
            species,
            ",".join(observed_moves),
            str(rounds),
            "true" if is_lead else "false",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
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
) -> dict[str, Any]:
    cache: dict[tuple[str, tuple[str, ...], bool], dict[str, Any]] = {}
    candidates: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    fixture_count = 0
    for fixture in fixtures:
        fixture_count += 1
        legal_actions = fixture.legal_actions
        if not _plain_speed_context(fixture):
            skip("non-plain-speed-context")
            continue

        active = fixture.state.get("active")
        opponent_active = fixture.state.get("opponent_active")
        if not isinstance(active, Mapping) or not isinstance(opponent_active, Mapping):
            skip("missing-active-state")
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
        own_speed = _effective_own_speed(
            active,
            own_conditions if isinstance(own_conditions, Mapping) else None,
        )
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
        if not (low < own_speed < high):
            skip("no-speed-order-fork")
            continue

        is_lead = _to_id(str(history["lead_species"])) == _to_id(opponent_species)
        generator_species = _generator_species(opponent_species)
        key = (generator_species, tuple(revealed), is_lead)
        if key not in cache:
            cache[key] = _sample_worlds(
                showdown_root=showdown_root,
                species=generator_species,
                observed_moves=revealed,
                rounds=rounds,
                is_lead=is_lead,
            )
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
        if frozenset(counts) != pair:
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
            "opponent_species": opponent_species,
            "generator_species": generator_species,
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
        "sampled_world_queries": len(cache),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "skipped": dict(sorted(skipped.items())),
    }


def mine_corpus(
    corpus_path: str | Path,
    *,
    showdown_root: str | Path,
    rounds: int = 2048,
) -> dict[str, Any]:
    return mine_candidates(
        load_corpus(corpus_path),
        showdown_root=Path(showdown_root),
        rounds=rounds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    result = mine_corpus(
        args.corpus,
        showdown_root=args.showdown_root,
        rounds=args.rounds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
