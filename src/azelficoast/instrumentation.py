"""Structured traces for battle observations and decisions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder

TRACE_SCHEMA = "azelficoast.decision-trace"
TRACE_SCHEMA_VERSION = 1


def _name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    return name if isinstance(name, str) else str(value)


def _named_mapping(mapping: Mapping[Any, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {
        _name(key) or "None": value
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0]))
    }


def pokemon_view(pokemon: Pokemon | None) -> dict[str, Any] | None:
    """Return only state exposed by the current poke-env Pokemon object."""
    if pokemon is None:
        return None
    return {
        "species": pokemon.species,
        "active": pokemon.active,
        "fainted": pokemon.fainted,
        "hp_fraction": pokemon.current_hp_fraction,
        "status": _name(pokemon.status),
        "item": pokemon.item,
        "ability": pokemon.ability,
        "moves": sorted(pokemon.moves),
        "boosts": dict(sorted(pokemon.boosts.items())),
        "types": [_name(type_) for type_ in pokemon.types],
        "tera_type": _name(pokemon.tera_type),
    }


def battle_view(battle: AbstractBattle) -> dict[str, Any]:
    """Snapshot the information state available to the player at this instant."""
    valid_orders = getattr(battle, "valid_orders", ())
    return {
        "battle_tag": battle.battle_tag,
        "turn": battle.turn,
        "player": battle.player_username,
        "opponent": battle.opponent_username,
        "active": pokemon_view(battle.active_pokemon),
        "opponent_active": pokemon_view(battle.opponent_active_pokemon),
        "team": {
            key: pokemon_view(pokemon)
            for key, pokemon in sorted(battle.team.items())
        },
        "opponent_team": {
            key: pokemon_view(pokemon)
            for key, pokemon in sorted(battle.opponent_team.items())
        },
        "weather": _named_mapping(battle.weather),
        "fields": _named_mapping(battle.fields),
        "side_conditions": _named_mapping(battle.side_conditions),
        "opponent_side_conditions": _named_mapping(battle.opponent_side_conditions),
        "available_moves": [move.id for move in getattr(battle, "available_moves", ())],
        "available_switches": [
            pokemon.species for pokemon in getattr(battle, "available_switches", ())
        ],
        "legal_actions": [order.message for order in valid_orders],
        "force_switch": getattr(battle, "force_switch", False),
        "trapped": getattr(battle, "trapped", False),
        "can_tera": getattr(battle, "can_tera", False),
    }


class DecisionTraceWriter:
    """Append-only JSONL evidence for one player's observable battle history."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._event_index = 0
        self._decision_index: dict[str, int] = {}
        self._protocol_index: dict[str, int] = {}

    def _append(self, record: dict[str, Any]) -> None:
        record = {
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "event_index": self._event_index,
            "observed_at": datetime.now(UTC).isoformat(),
            **record,
        }
        self._event_index += 1
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    def record_protocol_batch(self, split_messages: Sequence[Sequence[str]]) -> None:
        """Record the exact Showdown protocol observations delivered to poke-env."""
        if not split_messages:
            return
        room = split_messages[0][0].lstrip(">")
        index = self._protocol_index.get(room, 0)
        self._protocol_index[room] = index + 1
        self._append(
            {
                "kind": "protocol",
                "room": room,
                "protocol_index": index,
                "messages": [list(message) for message in split_messages[1:]],
            }
        )

    def record_decision(self, battle: AbstractBattle, order: BattleOrder) -> None:
        battle_tag = battle.battle_tag
        index = self._decision_index.get(battle_tag, 0)
        self._decision_index[battle_tag] = index + 1
        self._append(
            {
                "kind": "decision",
                "battle_tag": battle_tag,
                "decision_index": index,
                "state": battle_view(battle),
                "chosen_action": order.message,
            }
        )

    def record_terminal(self, battle: AbstractBattle) -> None:
        won = battle.won
        lost = battle.lost
        self._append(
            {
                "kind": "terminal",
                "battle_tag": battle.battle_tag,
                "won": won,
                "lost": lost,
                "tied": battle.finished and won is None and lost is None,
                "final_state": battle_view(battle),
            }
        )
