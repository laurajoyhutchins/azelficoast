"""Bounded ordered attack transition with one observable secondary branch.

This extends the whole-attack kernel with action ordering against one no-op opponent
move and a Shadow Ball-shaped secondary SpD drop. It is deliberately not yet a
general two-action turn scheduler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from azelficoast.research.mechanics.gen9_attack import (
    COMPILED_ATTACK_CONTEXT_WIDTH,
    AttackTransitionContext,
    attack_transition_dependency_key,
    attack_transition_dependency_signature,
    attack_transition_numeric,
    compile_attack_context,
)

ATTACKER_PRIORITY_COLUMN = COMPILED_ATTACK_CONTEXT_WIDTH
OPPONENT_PRIORITY_COLUMN = COMPILED_ATTACK_CONTEXT_WIDTH + 1
ATTACKER_SPEED_COLUMN = COMPILED_ATTACK_CONTEXT_WIDTH + 2
OPPONENT_SPEED_COLUMN = COMPILED_ATTACK_CONTEXT_WIDTH + 3
SECONDARY_CHANCE_COLUMN = COMPILED_ATTACK_CONTEXT_WIDTH + 4
DEFENDER_SPD_STAGE_COLUMN = COMPILED_ATTACK_CONTEXT_WIDTH + 5
COMPILED_ORDERED_ATTACK_CONTEXT_WIDTH = COMPILED_ATTACK_CONTEXT_WIDTH + 6

PACK_HP_BASE = 1024
PACK_PP_BASE = PACK_HP_BASE * PACK_HP_BASE
PACK_STAGE_BASE = PACK_PP_BASE * 64
PACK_ORDER_BASE = PACK_STAGE_BASE * 13

ORDERED_ATTACK_DEPENDENCY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OrderedAttackContext:
    attack: AttackTransitionContext
    attacker_priority: int
    opponent_priority: int
    attacker_speed: int
    opponent_speed: int
    secondary_chance: int
    defender_spd_stage: int

    def __post_init__(self) -> None:
        if not -7 <= self.attacker_priority <= 7:
            raise ValueError("attacker priority is outside the bounded transition domain")
        if not -7 <= self.opponent_priority <= 7:
            raise ValueError("opponent priority is outside the bounded transition domain")
        if self.attacker_speed <= 0 or self.opponent_speed <= 0:
            raise ValueError("ordered transition speeds must be positive")
        if not 0 <= self.secondary_chance <= 100:
            raise ValueError("secondary chance must be in [0, 100]")
        if not -6 <= self.defender_spd_stage <= 6:
            raise ValueError("defender SpD stage must be in [-6, 6]")
        if self.attack.move_pp >= 64:
            raise ValueError("ordered transition PP must fit its packed state")


@dataclass(frozen=True)
class OrderedAttackTransition:
    defender_hp: int
    attacker_hp: int
    move_pp: int
    defender_spd_stage: int
    attacker_acted_first: bool

    @property
    def defender_fainted(self) -> bool:
        return self.defender_hp == 0

    @property
    def attacker_fainted(self) -> bool:
        return self.attacker_hp == 0

    @property
    def packed(self) -> int:
        return (
            self.defender_hp
            + self.attacker_hp * PACK_HP_BASE
            + self.move_pp * PACK_PP_BASE
            + (self.defender_spd_stage + 6) * PACK_STAGE_BASE
            + int(self.attacker_acted_first) * PACK_ORDER_BASE
        )


def compile_ordered_attack_context(context: OrderedAttackContext) -> tuple[int, ...]:
    return (
        *compile_attack_context(context.attack),
        context.attacker_priority,
        context.opponent_priority,
        context.attacker_speed,
        context.opponent_speed,
        context.secondary_chance,
        context.defender_spd_stage,
    )


def _attacker_acts_first_numeric(params: tuple[int, ...], order_tie_roll: int) -> int:
    attacker_priority = params[23]
    opponent_priority = params[24]
    if attacker_priority > opponent_priority:
        return 1
    if attacker_priority < opponent_priority:
        return 0

    attacker_speed = params[25]
    opponent_speed = params[26]
    if attacker_speed > opponent_speed:
        return 1
    if attacker_speed < opponent_speed:
        return 0

    if order_tie_roll == 0:
        return 1
    return 0


def ordered_attack_transition_numeric(
    params: tuple[int, ...],
    order_tie_roll: int,
    accuracy_roll: int,
    damage_roll: int,
    secondary_roll: int,
) -> int:
    """Execute ordering plus the bounded whole attack and one SpD-drop secondary."""
    packed_attack = attack_transition_numeric(params, accuracy_roll, damage_roll)

    attack_flags = packed_attack // 134217728
    attack_remainder = packed_attack - attack_flags * 134217728
    move_pp = attack_remainder // 1048576
    attack_remainder -= move_pp * 1048576
    attacker_hp = attack_remainder // 1024
    defender_hp = attack_remainder - attacker_hp * 1024
    hit = attack_flags - (attack_flags // 2) * 2

    defender_spd_stage = params[28]
    if hit != 0:
        if defender_hp > 0:
            secondary_chance = params[27]
            if secondary_chance > 0:
                if secondary_roll < secondary_chance:
                    if defender_spd_stage > -6:
                        defender_spd_stage -= 1

    acted_first = _attacker_acts_first_numeric(params, order_tie_roll)
    return (
        defender_hp
        + attacker_hp * 1024
        + move_pp * 1048576
        + (defender_spd_stage + 6) * 67108864
        + acted_first * 872415232
    )


def ordered_attack_transition(
    context: OrderedAttackContext,
    *,
    order_tie_roll: int,
    accuracy_roll: int,
    damage_roll: int,
    secondary_roll: int,
) -> OrderedAttackTransition:
    if order_tie_roll not in (0, 1):
        raise ValueError("order tie roll must be 0 or 1")
    if not 0 <= secondary_roll < 100:
        raise ValueError("secondary roll must be in [0, 99]")
    packed = ordered_attack_transition_numeric(
        compile_ordered_attack_context(context),
        order_tie_roll,
        accuracy_roll,
        damage_roll,
        secondary_roll,
    )
    return unpack_ordered_attack_transition(packed)


def unpack_ordered_attack_transition(packed: int) -> OrderedAttackTransition:
    if packed < 0:
        raise ValueError("packed ordered transition must be non-negative")
    acted_first, remainder = divmod(packed, PACK_ORDER_BASE)
    stage_encoded, remainder = divmod(remainder, PACK_STAGE_BASE)
    move_pp, remainder = divmod(remainder, PACK_PP_BASE)
    attacker_hp, defender_hp = divmod(remainder, PACK_HP_BASE)
    return OrderedAttackTransition(
        defender_hp=defender_hp,
        attacker_hp=attacker_hp,
        move_pp=move_pp,
        defender_spd_stage=stage_encoded - 6,
        attacker_acted_first=bool(acted_first),
    )


def _attacker_acts_first(context: OrderedAttackContext, order_tie_roll: int) -> int:
    return _attacker_acts_first_numeric(
        compile_ordered_attack_context(context),
        order_tie_roll,
    )


def ordered_attack_dependency_key(
    context: OrderedAttackContext,
    *,
    order_tie_roll: int,
    accuracy_roll: int,
    damage_roll: int,
    secondary_roll: int,
    include_order_tie_roll: bool = True,
    include_secondary_roll: bool = True,
) -> tuple[int, ...]:
    """Return the control-flow-sensitive dependency key for the whole ordered attack."""
    effective_order_roll = order_tie_roll if include_order_tie_roll else 0
    acted_first = _attacker_acts_first(context, effective_order_roll)
    attack_key = attack_transition_dependency_key(
        context.attack,
        accuracy_roll,
        damage_roll,
    )

    attack_result = ordered_attack_transition(
        context,
        order_tie_roll=order_tie_roll,
        accuracy_roll=accuracy_roll,
        damage_roll=damage_roll,
        secondary_roll=secondary_roll,
    )

    stage_key = context.defender_spd_stage
    if include_secondary_roll:
        stage_key = attack_result.defender_spd_stage

    return (
        acted_first,
        *attack_key,
        stage_key,
    )


def ordered_attack_dependency_document() -> dict[str, object]:
    return {
        "schema": "azelficoast.ordered-attack-dependencies",
        "schema_version": ORDERED_ATTACK_DEPENDENCY_SCHEMA_VERSION,
        "order": {
            "reads": [
                "attacker.move.priority",
                "opponent.move.priority",
                "attacker.speed",
                "opponent.speed",
            ],
            "conditional_random": "rng.order_tie_roll",
            "writes": ["attacker_acted_first"],
        },
        "attack": {
            "dependency_signature": attack_transition_dependency_signature(),
        },
        "secondary": {
            "condition": "hit AND defender survives AND secondary_chance > 0",
            "reads": [
                "move.secondary_chance",
                "defender.spd_stage",
                "rng.secondary_roll",
            ],
            "writes": ["defender.spd_stage"],
            "effect": "spd -1 bounded at -6",
        },
        "packed_state": {
            "hp_base": PACK_HP_BASE,
            "pp_base": PACK_PP_BASE,
            "stage_base": PACK_STAGE_BASE,
            "order_base": PACK_ORDER_BASE,
        },
    }


def ordered_attack_dependency_signature() -> str:
    encoded = json.dumps(
        ordered_attack_dependency_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
