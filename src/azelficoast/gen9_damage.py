"""A narrow Gen 9 damage kernel with Pokémon Showdown-compatible integer rounding.

This module intentionally implements only the mechanics exercised by the first real
damage experiment: ordinary single-target damage, ordinary non-Stellar Terastallization,
Choice Band/Specs attack modification, burn, Life Orb, STAB, type-effectiveness
exponents, and an explicit fixed-point defender-stat modifier validated against Beads
of Ruin. Critical hits, weather, spread damage, variable base power, Stellar Tera,
and other exceptional mechanics remain outside scope.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

Category = Literal["Physical", "Special"]

ITEM_NONE = ""
ITEM_CHOICE_BAND = "Choice Band"
ITEM_CHOICE_SPECS = "Choice Specs"
ITEM_LIFE_ORB = "Life Orb"

MOD_ONE = 4096
MOD_HALF = 2048
MOD_THREE_QUARTERS = 3072
MOD_ONE_POINT_FIVE = 6144
MOD_TWO = 8192
MOD_LIFE_ORB = 5324


@dataclass(frozen=True)
class DamageContext:
    attacker_level: int
    defender_level: int
    base_power: int
    category: Category
    move_id: str
    move_type: str
    attacker_types: tuple[str, ...]
    tera_type: str | None
    attacker_base_stat: int
    attacker_iv: int
    attacker_ev: int
    attacker_nature_percent: int
    defender_base_stat: int
    defender_iv: int
    defender_ev: int
    defender_nature_percent: int
    attacker_item: str
    type_mod: int
    burned: bool = False
    defender_stat_modifier: int = MOD_ONE

    def without_item(self) -> DamageContext:
        return replace(self, attacker_item=ITEM_NONE)


class DamageKernelError(ValueError):
    """Raised when a context uses mechanics outside this narrow kernel."""


def showdown_modify(value: int, numerator: int, denominator: int = 1) -> int:
    """Match Battle.modify for non-negative bounded Gen 9 damage/stat values."""
    if value < 0 or numerator < 0 or denominator <= 0:
        raise DamageKernelError("modifier inputs must be non-negative with positive denominator")
    modifier = numerator * 4096 // denominator
    return (value * modifier + 2047) // 4096


def showdown_modify_fixed(value: int, modifier: int) -> int:
    if value < 0 or modifier < 0:
        raise DamageKernelError("fixed modifier inputs must be non-negative")
    return (value * modifier + 2047) // 4096


def ordinary_stat(
    base: int,
    iv: int,
    ev: int,
    level: int,
    nature_percent: int,
) -> int:
    """Calculate a non-HP battle stat for the bounded values used by this experiment."""
    if not 1 <= level <= 100:
        raise DamageKernelError(f"unsupported level {level}")
    if not 0 <= iv <= 31:
        raise DamageKernelError(f"unsupported IV {iv}")
    if not 0 <= ev <= 252:
        raise DamageKernelError(f"unsupported EV {ev}")
    if nature_percent not in (90, 100, 110):
        raise DamageKernelError(f"unsupported nature modifier {nature_percent}")

    stat = ((2 * base + iv + ev // 4) * level) // 100 + 5
    return stat * nature_percent // 100


def attack_modifier(context: DamageContext) -> int:
    if context.attacker_item == ITEM_CHOICE_SPECS and context.category == "Special":
        return MOD_ONE_POINT_FIVE
    if context.attacker_item == ITEM_CHOICE_BAND and context.category == "Physical":
        return MOD_ONE_POINT_FIVE
    return MOD_ONE


def stab_modifier(context: DamageContext) -> int:
    if context.tera_type == "Stellar":
        raise DamageKernelError("Stellar Terastallization is outside this kernel")

    base_stab = context.move_type in context.attacker_types
    if context.tera_type:
        if context.tera_type == context.move_type:
            return MOD_TWO if base_stab else MOD_ONE_POINT_FIVE
        return MOD_ONE_POINT_FIVE if base_stab else MOD_ONE
    return MOD_ONE_POINT_FIVE if base_stab else MOD_ONE


def burn_modifier(context: DamageContext) -> int:
    if context.burned and context.category == "Physical" and context.move_id != "facade":
        return MOD_HALF
    return MOD_ONE


def final_damage_modifier(context: DamageContext) -> int:
    if context.attacker_item == ITEM_LIFE_ORB:
        return MOD_LIFE_ORB
    return MOD_ONE


def resolved_attack(context: DamageContext) -> int:
    stat = ordinary_stat(
        context.attacker_base_stat,
        context.attacker_iv,
        context.attacker_ev,
        context.attacker_level,
        context.attacker_nature_percent,
    )
    return showdown_modify_fixed(stat, attack_modifier(context))


def resolved_defense(context: DamageContext) -> int:
    stat = ordinary_stat(
        context.defender_base_stat,
        context.defender_iv,
        context.defender_ev,
        context.defender_level,
        context.defender_nature_percent,
    )
    return showdown_modify_fixed(stat, context.defender_stat_modifier)


def apply_type_effectiveness(damage: int, type_mod: int) -> int:
    if not -6 <= type_mod <= 6:
        raise DamageKernelError(f"unsupported type modifier exponent {type_mod}")
    if type_mod >= 0:
        return damage * (1 << type_mod)
    return damage // (1 << (-type_mod))


def damage(context: DamageContext, roll: int) -> int:
    """Return exact damage for one Showdown randomizer roll in [0, 15]."""
    if not 0 <= roll <= 15:
        raise DamageKernelError(f"damage roll must be in [0, 15], got {roll}")
    if context.base_power <= 0:
        raise DamageKernelError("fixed positive base power is required")

    attack = resolved_attack(context)
    defense = resolved_defense(context)
    if defense <= 0:
        raise DamageKernelError("defense must be positive")

    level_term = (2 * context.attacker_level) // 5 + 2
    base_damage = ((level_term * context.base_power * attack) // defense) // 50
    base_damage += 2

    # Battle.randomizer(baseDamage): floor(floor(baseDamage * (100-roll)) / 100).
    base_damage = base_damage * (100 - roll) // 100

    base_damage = showdown_modify_fixed(base_damage, stab_modifier(context))
    base_damage = apply_type_effectiveness(base_damage, context.type_mod)
    base_damage = showdown_modify_fixed(base_damage, burn_modifier(context))
    base_damage = showdown_modify_fixed(base_damage, final_damage_modifier(context))

    # Gen 9 minimum damage check occurs after final damage modifiers.
    return max(1, base_damage)


def compile_numeric_context(context: DamageContext) -> tuple[int, ...]:
    """Lower semantic context into the integer columns consumed by the JAX kernel."""
    category_code = 1 if context.category == "Physical" else 2
    return (
        context.attacker_level,
        context.defender_level,
        context.base_power,
        context.attacker_base_stat,
        context.attacker_iv,
        context.attacker_ev,
        context.attacker_nature_percent,
        context.defender_base_stat,
        context.defender_iv,
        context.defender_ev,
        context.defender_nature_percent,
        context.defender_stat_modifier,
        attack_modifier(context),
        stab_modifier(context),
        context.type_mod,
        burn_modifier(context),
        final_damage_modifier(context),
        category_code,
    )
