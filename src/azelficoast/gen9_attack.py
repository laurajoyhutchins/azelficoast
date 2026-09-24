"""A bounded whole-attack transition built on the exact Gen 9 damage kernel.

The transition covers one ordinary damaging move from the point where the attacker
has been selected through accuracy, damage, HP application, faint flags, PP use,
and Life Orb recoil. It intentionally excludes secondaries, multihit sequencing,
contact hooks, protection, immunities, drain/recoil moves, self-stat drops, and
other move-specific effects until separate Showdown-backed treatments cover them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from azelficoast.gen9_damage import (
    COMPILED_CONTEXT_WIDTH,
    DAMAGE_DEPENDENCY_COLUMNS,
    DamageContext,
    damage_dependency_tuple,
    damage_numeric,
    compile_numeric_context,
)

ATTACK_ACCURACY_COLUMN = COMPILED_CONTEXT_WIDTH
ATTACKER_HP_COLUMN = COMPILED_CONTEXT_WIDTH + 1
ATTACKER_MAX_HP_COLUMN = COMPILED_CONTEXT_WIDTH + 2
DEFENDER_HP_COLUMN = COMPILED_CONTEXT_WIDTH + 3
MOVE_PP_COLUMN = COMPILED_CONTEXT_WIDTH + 4
COMPILED_ATTACK_CONTEXT_WIDTH = COMPILED_CONTEXT_WIDTH + 5

PACK_HP_BASE = 1024
PACK_PP_BASE = PACK_HP_BASE * PACK_HP_BASE
PACK_FLAGS_BASE = PACK_PP_BASE * 128

FLAG_HIT = 1
FLAG_DEFENDER_FAINTED = 2
FLAG_ATTACKER_FAINTED = 4

# This is metadata, not a name-version suffix. Changing the transition dependency
# program requires changing this schema version and therefore the advertised hash.
ATTACK_TRANSITION_DEPENDENCY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AttackTransitionContext:
    damage: DamageContext
    accuracy: int
    attacker_hp: int
    attacker_max_hp: int
    defender_hp: int
    move_pp: int

    def __post_init__(self) -> None:
        if not 1 <= self.accuracy <= 100:
            raise ValueError("attack accuracy must be in [1, 100]")
        if not 1 <= self.attacker_max_hp < PACK_HP_BASE:
            raise ValueError("attacker max HP is outside the bounded packed transition domain")
        if not 1 <= self.attacker_hp <= self.attacker_max_hp:
            raise ValueError("attacker HP must be positive and no greater than max HP")
        if not 1 <= self.defender_hp < PACK_HP_BASE:
            raise ValueError("defender HP is outside the bounded packed transition domain")
        if not 1 <= self.move_pp < 128:
            raise ValueError("move PP is outside the bounded packed transition domain")


@dataclass(frozen=True)
class AttackTransition:
    defender_hp: int
    attacker_hp: int
    move_pp: int
    hit: bool
    defender_fainted: bool
    attacker_fainted: bool

    @property
    def packed(self) -> int:
        flags = (
            int(self.hit) * FLAG_HIT
            + int(self.defender_fainted) * FLAG_DEFENDER_FAINTED
            + int(self.attacker_fainted) * FLAG_ATTACKER_FAINTED
        )
        return (
            self.defender_hp
            + self.attacker_hp * PACK_HP_BASE
            + self.move_pp * PACK_PP_BASE
            + flags * PACK_FLAGS_BASE
        )


def compile_attack_context(context: AttackTransitionContext) -> tuple[int, ...]:
    return (
        *compile_numeric_context(context.damage),
        context.accuracy,
        context.attacker_hp,
        context.attacker_max_hp,
        context.defender_hp,
        context.move_pp,
    )


def attack_transition_numeric(
    params: tuple[int, ...],
    accuracy_roll: int,
    damage_roll: int,
) -> int:
    """Execute one bounded whole-attack transition and return a packed post-state."""
    accuracy = params[18]
    attacker_hp = params[19]
    attacker_max_hp = params[20]
    defender_hp = params[21]
    move_pp = params[22]

    pp_after = move_pp - 1
    if pp_after < 0:
        pp_after = 0

    hit = 0
    defender_after = defender_hp
    attacker_after = attacker_hp

    if accuracy_roll < accuracy:
        hit = 1
        dealt = damage_numeric(params, damage_roll)
        defender_after = defender_hp - dealt
        if defender_after < 0:
            defender_after = 0

        # In the admitted transition slice, final modifier 5324 uniquely denotes
        # Life Orb. Its post-move recoil is floor(baseMaxHP / 10), clamped to 1.
        if params[16] == 5324:
            recoil = attacker_max_hp // 10
            if recoil < 1:
                recoil = 1
            attacker_after = attacker_hp - recoil
            if attacker_after < 0:
                attacker_after = 0

    flags = hit
    if defender_after == 0:
        flags += 2
    if attacker_after == 0:
        flags += 4

    return (
        defender_after
        + attacker_after * 1024
        + pp_after * 1048576
        + flags * 134217728
    )


def attack_transition(
    context: AttackTransitionContext,
    accuracy_roll: int,
    damage_roll: int,
) -> AttackTransition:
    if not 0 <= accuracy_roll < 100:
        raise ValueError("accuracy roll must be in [0, 99]")
    if not 0 <= damage_roll < 16:
        raise ValueError("damage roll must be in [0, 15]")
    return unpack_attack_transition(
        attack_transition_numeric(
            compile_attack_context(context),
            accuracy_roll,
            damage_roll,
        )
    )


def unpack_attack_transition(packed: int) -> AttackTransition:
    if packed < 0:
        raise ValueError("packed attack transition must be non-negative")
    flags, remainder = divmod(packed, PACK_FLAGS_BASE)
    move_pp, remainder = divmod(remainder, PACK_PP_BASE)
    attacker_hp, defender_hp = divmod(remainder, PACK_HP_BASE)
    return AttackTransition(
        defender_hp=defender_hp,
        attacker_hp=attacker_hp,
        move_pp=move_pp,
        hit=bool(flags & FLAG_HIT),
        defender_fainted=bool(flags & FLAG_DEFENDER_FAINTED),
        attacker_fainted=bool(flags & FLAG_ATTACKER_FAINTED),
    )


def attack_transition_dependency_key(
    context: AttackTransitionContext,
    accuracy_roll: int,
    damage_roll: int,
    *,
    include_attack_modifier: bool = True,
) -> tuple[int, ...]:
    """Return the minimal control-flow-sensitive key for this transition.

    Misses intentionally do not depend on damage mechanics or the damage RNG.
    Hits do. This routine and attack_transition_dependency_signature() are the
    class-native execution contract for the whole attack transition.
    """
    hit = int(accuracy_roll < context.accuracy)
    state = (
        hit,
        context.attacker_hp,
        context.attacker_max_hp,
        context.defender_hp,
        context.move_pp,
    )
    if not hit:
        return state
    return (
        *state,
        *damage_dependency_tuple(
            context.damage,
            include_attack_modifier=include_attack_modifier,
        ),
        damage_roll,
    )


def attack_transition_dependency_document() -> dict[str, object]:
    return {
        "schema": "azelficoast.attack-transition-dependencies",
        "schema_version": ATTACK_TRANSITION_DEPENDENCY_SCHEMA_VERSION,
        "control": {
            "reads": ["move.accuracy", "rng.accuracy_roll"],
            "branches": ["miss", "hit"],
        },
        "always_state_reads": [
            "attacker.hp",
            "attacker.max_hp",
            "defender.hp",
            "move.pp",
        ],
        "hit_branch_reads": [
            "damage_numeric.dependencies",
            "rng.damage_roll",
        ],
        "hit_branch_post_effects": [
            "defender.hp",
            "defender.fainted",
            "life_orb_recoil",
            "attacker.hp",
            "attacker.fainted",
        ],
        "always_writes": [
            "move.pp",
            "attacker.hp",
            "defender.hp",
            "attacker.fainted",
            "defender.fainted",
            "hit",
        ],
        "damage_dependency_columns": list(DAMAGE_DEPENDENCY_COLUMNS),
        "packed_state": {
            "hp_base": PACK_HP_BASE,
            "pp_base": PACK_PP_BASE,
            "flags_base": PACK_FLAGS_BASE,
        },
    }


def attack_transition_dependency_signature() -> str:
    encoded = json.dumps(
        attack_transition_dependency_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
