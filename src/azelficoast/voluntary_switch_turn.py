"""Bounded voluntary switching composed with the certified whole-attack transition.

The switch is resolved before the modeled opposing ordinary damaging move. The outgoing
Pokémon becomes bench state, the selected teammate becomes active, and the opponent's
attack is evaluated against that incoming Pokémon.

This deliberately excludes entry hazards, switch-in/out abilities and items, trapping,
pursuit-like historical mechanics, pivot moves, forced switches, and end-turn effects.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from azelficoast.core.projection import compile_projection_ids
from azelficoast.gen9_attack import (
    AttackTransitionContext,
    attack_transition,
    attack_transition_dependency_key,
    attack_transition_dependency_signature,
)

VOLUNTARY_SWITCH_DEPENDENCY_SCHEMA_VERSION = 1


class VoluntarySwitchError(ValueError):
    """Raised when state is outside the bounded voluntary-switch model."""


@dataclass(frozen=True)
class VoluntarySwitchContext:
    outgoing_slot: int
    incoming_slot: int
    outgoing_hp: int
    outgoing_max_hp: int
    incoming_max_hp: int
    opponent_attack: AttackTransitionContext

    def __post_init__(self) -> None:
        if self.outgoing_slot < 0 or self.incoming_slot < 0:
            raise VoluntarySwitchError("team slots must be non-negative")
        if self.outgoing_slot == self.incoming_slot:
            raise VoluntarySwitchError("voluntary switch must change active slot")
        if not 1 <= self.outgoing_hp <= self.outgoing_max_hp:
            raise VoluntarySwitchError("outgoing HP must be within max HP")
        if self.outgoing_max_hp >= 1024:
            raise VoluntarySwitchError("outgoing max HP is outside the bounded domain")
        if not 1 <= self.opponent_attack.defender_hp <= self.incoming_max_hp:
            raise VoluntarySwitchError("incoming HP must be within max HP")
        if self.incoming_max_hp >= 1024:
            raise VoluntarySwitchError("incoming max HP is outside the bounded domain")


@dataclass(frozen=True)
class VoluntarySwitchOutcome:
    active_slot: int
    bench_slot: int
    active_hp: int
    active_max_hp: int
    bench_hp: int
    bench_max_hp: int
    opponent_hp: int
    opponent_move_pp: int
    hit: bool
    active_fainted: bool
    opponent_fainted: bool

    @property
    def successor_key(self) -> tuple[int, ...]:
        return (
            self.active_slot,
            self.bench_slot,
            self.active_hp,
            self.active_max_hp,
            self.bench_hp,
            self.bench_max_hp,
            self.opponent_hp,
            self.opponent_move_pp,
            int(self.hit),
            int(self.active_fainted),
            int(self.opponent_fainted),
        )


@dataclass(frozen=True)
class VoluntarySwitchWorld:
    context_index: int
    accuracy_roll: int
    damage_roll: int
    untouched_bench_signature: int


@dataclass(frozen=True)
class VoluntarySwitchProjection:
    class_ids: np.ndarray
    representative_indices: np.ndarray
    effect_signature: str

    @property
    def class_count(self) -> int:
        return int(len(self.representative_indices))


def voluntary_switch_turn(
    context: VoluntarySwitchContext,
    *,
    accuracy_roll: int,
    damage_roll: int,
) -> VoluntarySwitchOutcome:
    """Switch first, then execute the opposing attack against the incoming active."""

    attack = attack_transition(
        context.opponent_attack,
        accuracy_roll=accuracy_roll,
        damage_roll=damage_roll,
    )
    return VoluntarySwitchOutcome(
        active_slot=context.incoming_slot,
        bench_slot=context.outgoing_slot,
        active_hp=attack.defender_hp,
        active_max_hp=context.incoming_max_hp,
        bench_hp=context.outgoing_hp,
        bench_max_hp=context.outgoing_max_hp,
        opponent_hp=attack.attacker_hp,
        opponent_move_pp=attack.move_pp,
        hit=attack.hit,
        active_fainted=attack.defender_fainted,
        opponent_fainted=attack.attacker_fainted,
    )


def voluntary_switch_dependency_key(
    context: VoluntarySwitchContext,
    *,
    accuracy_roll: int,
    damage_roll: int,
) -> tuple[int, ...]:
    """Return the exact execution key for switch-before-attack mechanics.

    The selected incoming slot is always part of the key because two mechanically
    identical teammates are still different successor states. Untouched bench state is
    intentionally absent: this bounded effect neither reads nor mutates it.
    """

    return (
        context.outgoing_slot,
        context.incoming_slot,
        context.outgoing_hp,
        context.outgoing_max_hp,
        context.incoming_max_hp,
        *attack_transition_dependency_key(
            context.opponent_attack,
            accuracy_roll,
            damage_roll,
        ),
    )


def voluntary_switch_dependency_document() -> dict[str, object]:
    return {
        "schema": "azelficoast.voluntary-switch-dependencies",
        "schema_version": VOLUNTARY_SWITCH_DEPENDENCY_SCHEMA_VERSION,
        "binds": {
            "opponent_attack": attack_transition_dependency_signature(),
        },
        "ordering": {
            "first": "replace active identity with selected living bench teammate",
            "second": "execute modeled ordinary opposing damaging move",
        },
        "always_reads": [
            "p1.outgoing.slot",
            "p1.outgoing.hp",
            "p1.outgoing.max_hp",
            "p1.incoming.slot",
            "p1.incoming.hp",
            "p1.incoming.max_hp",
            "opponent_attack.always_dependencies",
        ],
        "conditional_reads": {
            "attack_miss": [
                "move.accuracy",
                "rng.accuracy_roll",
            ],
            "attack_hit": [
                "move.accuracy",
                "rng.accuracy_roll",
                "damage_numeric.dependencies",
                "rng.damage_roll",
            ],
        },
        "unread": [
            "untouched bench teammate state",
            "outgoing defensive stats after switch selection",
        ],
        "writes": [
            "p1.active.slot",
            "p1.outgoing bench hp",
            "p1.incoming active hp",
            "opponent.hp",
            "opponent.move.pp",
            "faint flags",
        ],
        "claim": (
            "The selected teammate becomes active before the modeled opponent attack. "
            "Attack dependencies therefore bind to the incoming defender, while "
            "untouched bench state remains outside this effect's execution quotient."
        ),
        "non_claims": [
            "Entry hazards are outside this bounded model.",
            "Switch-in and switch-out abilities and items are outside this bounded model.",
            "Trapping, forced switches, pivot moves, and end-turn effects are excluded.",
        ],
    }


def voluntary_switch_dependency_signature() -> str:
    encoded = json.dumps(
        voluntary_switch_dependency_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def compile_voluntary_switch_projection(
    contexts: Sequence[VoluntarySwitchContext],
    worlds: Sequence[VoluntarySwitchWorld],
) -> VoluntarySwitchProjection:
    if not contexts:
        raise VoluntarySwitchError("at least one switch context is required")
    if not worlds:
        raise VoluntarySwitchError("at least one switch world is required")

    def key_at(index: int) -> tuple[int, ...]:
        world = worlds[index]
        if not 0 <= world.context_index < len(contexts):
            raise VoluntarySwitchError("world references an unavailable switch context")
        return voluntary_switch_dependency_key(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
        )

    class_ids, representatives = compile_projection_ids(len(worlds), key_at)
    return VoluntarySwitchProjection(
        class_ids=class_ids,
        representative_indices=representatives,
        effect_signature=voluntary_switch_dependency_signature(),
    )
