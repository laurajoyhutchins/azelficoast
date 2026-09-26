"""Additional competitive sparring opponents for the self-improvement league."""

from __future__ import annotations

from typing import Any, cast

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder


_DIRTY_ANTI_SETUP = frozenset(
    {"clearsmog", "encore", "haze", "roar", "taunt", "topsyturvy", "whirlwind"}
)
_DIRTY_DENIAL = frozenset(
    {"disable", "encore", "knockoff", "switcheroo", "taunt", "torment", "trick"}
)
_DIRTY_CHIP = frozenset(
    {"firespin", "infestation", "leechseed", "magmastorm", "saltcure", "sandtomb", "whirlpool"}
)
_DIRTY_STATUS = frozenset(
    {"glare", "nuzzle", "spore", "stunspore", "thunderwave", "toxic", "toxicthread", "willowisp", "yawn"}
)
_DIRTY_STALL = frozenset(
    {"banefulbunker", "burningbulwark", "detect", "kingsshield", "protect", "silktrap", "substitute"}
)
_DIRTY_HAZARDS = frozenset({"spikes", "stealthrock", "stickyweb", "toxicspikes"})


def _available_move_map(battle: AbstractBattle) -> dict[str, Any]:
    return {
        str(move.id): move
        for move in getattr(battle, "available_moves", ())
        if getattr(move, "id", None)
    }


def _first_available(move_map: dict[str, Any], candidates: frozenset[str]) -> Any | None:
    for move_id in sorted(candidates):
        move = move_map.get(move_id)
        if move is not None:
            return move
    return None


def _positive_boost_total(pokemon: Any) -> int:
    boosts = getattr(pokemon, "boosts", None)
    if not isinstance(boosts, dict):
        return 0
    return sum(max(0, int(value)) for value in boosts.values())


def _priority_finisher(move_map: dict[str, Any]) -> Any | None:
    candidates = [
        move
        for move in move_map.values()
        if int(getattr(move, "priority", 0) or 0) > 0
        and float(getattr(move, "base_power", 0) or 0) > 0
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda move: (
            float(getattr(move, "base_power", 0) or 0),
            str(getattr(move, "id", "")),
        ),
    )


def dirty_tricks_move(battle: AbstractBattle) -> Any | None:
    """Choose a legal disruptive move or delegate to the established heuristic player."""

    active = getattr(battle, "active_pokemon", None)
    opponent = getattr(battle, "opponent_active_pokemon", None)
    if active is None or opponent is None:
        return None

    move_map = _available_move_map(battle)
    if not move_map:
        return None

    if _positive_boost_total(opponent) >= 2:
        move = _first_available(move_map, _DIRTY_ANTI_SETUP)
        if move is not None:
            return move

    opponent_hp = float(getattr(opponent, "current_hp_fraction", 1.0) or 0.0)
    if opponent_hp <= 0.35:
        move = _priority_finisher(move_map)
        if move is not None:
            return move

    if getattr(opponent, "status", None) is None:
        move = _first_available(move_map, _DIRTY_STATUS)
        if move is not None:
            return move

    for candidates in (_DIRTY_DENIAL, _DIRTY_CHIP):
        move = _first_available(move_map, candidates)
        if move is not None:
            return move

    if getattr(opponent, "status", None) is not None:
        move = _first_available(move_map, _DIRTY_STALL)
        if move is not None:
            return move

    move = _first_available(move_map, _DIRTY_HAZARDS)
    if move is not None:
        return move

    return None


class DirtyTricksPlayer(SimpleHeuristicsPlayer):
    """Disruptive legal play with normal heuristic switching/Tera fallback."""

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        move = dirty_tricks_move(battle)
        if move is not None:
            return cast(BattleOrder, self.create_order(move))
        return cast(BattleOrder, super().choose_move(battle))
