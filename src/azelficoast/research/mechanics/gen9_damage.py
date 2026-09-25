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

# Stable columns emitted by compile_numeric_context(). Keeping these beside the
# compiler lets non-JAX machinery consume dependency signatures without importing
# the optional accelerator backend.
COMPILED_ATTACK_MOD_COLUMN = 12
COMPILED_CATEGORY_COLUMN = 17
COMPILED_CONTEXT_WIDTH = 18
DAMAGE_DEPENDENCY_COLUMNS = tuple(
    column
    for column in range(COMPILED_CONTEXT_WIDTH)
    if column != COMPILED_CATEGORY_COLUMN
)


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


def _kernel_modify(value: int, modifier: int) -> int:
    return (value * modifier + 2047) // 4096


def showdown_modify_fixed(value: int, modifier: int) -> int:
    if value < 0 or modifier < 0:
        raise DamageKernelError("fixed modifier inputs must be non-negative")
    return _kernel_modify(value, modifier)


def _kernel_ordinary_stat(
    base: int,
    iv: int,
    ev: int,
    level: int,
    nature_percent: int,
) -> int:
    stat = ((2 * base + iv + ev // 4) * level) // 100 + 5
    return stat * nature_percent // 100


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

    return _kernel_ordinary_stat(base, iv, ev, level, nature_percent)


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


def _kernel_type_effectiveness(damage: int, type_mod: int) -> int:
    factor = 1
    if type_mod >= 0:
        for _index in range(type_mod):
            factor *= 2
        return damage * factor
    for _index in range(-type_mod):
        factor *= 2
    return damage // factor


def apply_type_effectiveness(damage: int, type_mod: int) -> int:
    if not -6 <= type_mod <= 6:
        raise DamageKernelError(f"unsupported type modifier exponent {type_mod}")
    return _kernel_type_effectiveness(damage, type_mod)


def damage_numeric(params: tuple[int, ...], roll: int) -> int:
    """Execute the compiled numeric damage context using plain Python integer semantics."""
    attacker_level = params[0]
    attack = _kernel_ordinary_stat(
        params[3],
        params[4],
        params[5],
        attacker_level,
        params[6],
    )
    attack = _kernel_modify(attack, params[12])
    defense = _kernel_ordinary_stat(
        params[7],
        params[8],
        params[9],
        params[1],
        params[10],
    )
    defense = _kernel_modify(defense, params[11])

    result = (((2 * attacker_level) // 5 + 2) * params[2] * attack) // defense
    result = result // 50 + 2
    result = (result * (100 - roll)) // 100
    result = _kernel_modify(result, params[13])
    result = _kernel_type_effectiveness(result, params[14])
    result = _kernel_modify(result, params[15])
    result = _kernel_modify(result, params[16])
    if result < 1:
        result = 1
    return result


def damage(context: DamageContext, roll: int) -> int:
    """Return exact damage for one Showdown randomizer roll in [0, 15]."""
    if not 0 <= roll <= 15:
        raise DamageKernelError(f"damage roll must be in [0, 15], got {roll}")
    if context.base_power <= 0:
        raise DamageKernelError("fixed positive base power is required")

    if resolved_defense(context) <= 0:
        raise DamageKernelError("defense must be positive")

    # The numeric kernel is intentionally ordinary Python. The native compiler
    # extracts this same function and its three helpers from this source file.
    return damage_numeric(compile_numeric_context(context), roll)


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


def damage_dependency_tuple(
    context: DamageContext,
    *,
    include_attack_modifier: bool = True,
) -> tuple[int, ...]:
    """Return the exact numeric damage inputs read by damage_numeric()."""
    compiled = compile_numeric_context(context)
    return tuple(
        compiled[column]
        for column in DAMAGE_DEPENDENCY_COLUMNS
        if include_attack_modifier or column != COMPILED_ATTACK_MOD_COLUMN
    )
