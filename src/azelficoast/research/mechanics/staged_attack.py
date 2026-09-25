"""Exact attacker-stage damage and whole-attack helpers.

This module deliberately layers stat stages over the existing certified damage and
attack contexts without changing their compiled layouts. Showdown applies a stat stage
to the stored stat before ordinary item/stat modifiers such as Choice Band.
"""

from __future__ import annotations

from azelficoast.research.mechanics.gen9_attack import AttackTransition, AttackTransitionContext
from azelficoast.research.mechanics.gen9_damage import (
    ITEM_LIFE_ORB,
    DamageContext,
    DamageKernelError,
    apply_type_effectiveness,
    attack_modifier,
    burn_modifier,
    damage_dependency_tuple,
    final_damage_modifier,
    ordinary_stat,
    resolved_defense,
    showdown_modify_fixed,
    stab_modifier,
)


def staged_stat(stat: int, stage: int) -> int:
    """Apply Showdown's ordinary Gen 9 stat-stage transform."""
    if stat < 0:
        raise DamageKernelError("staged stat must be non-negative")
    if not -6 <= stage <= 6:
        raise DamageKernelError(f"stat stage must be in [-6, 6], got {stage}")
    if stage >= 0:
        return (stat * (2 + stage)) // 2
    return (stat * 2) // (2 - stage)


def damage_with_attack_stage(
    context: DamageContext,
    stage: int,
    roll: int,
) -> int:
    """Return bounded damage after applying one attacker stat stage."""
    if not 0 <= roll <= 15:
        raise DamageKernelError(f"damage roll must be in [0, 15], got {roll}")
    if context.base_power <= 0:
        raise DamageKernelError("fixed positive base power is required")

    defense = resolved_defense(context)
    if defense <= 0:
        raise DamageKernelError("defense must be positive")

    attack = ordinary_stat(
        context.attacker_base_stat,
        context.attacker_iv,
        context.attacker_ev,
        context.attacker_level,
        context.attacker_nature_percent,
    )
    attack = staged_stat(attack, stage)
    attack = showdown_modify_fixed(attack, attack_modifier(context))

    result = (((2 * context.attacker_level) // 5 + 2) * context.base_power * attack) // defense
    result = result // 50 + 2
    result = (result * (100 - roll)) // 100
    result = showdown_modify_fixed(result, stab_modifier(context))
    result = apply_type_effectiveness(result, context.type_mod)
    result = showdown_modify_fixed(result, burn_modifier(context))
    result = showdown_modify_fixed(result, final_damage_modifier(context))
    return max(1, result)


def staged_attack_transition(
    context: AttackTransitionContext,
    attacker_stage: int,
    accuracy_roll: int,
    damage_roll: int,
) -> AttackTransition:
    """Execute the existing bounded attack semantics with an explicit stat stage."""
    if not -6 <= attacker_stage <= 6:
        raise ValueError("attacker stage must be in [-6, 6]")
    if not 0 <= accuracy_roll < 100:
        raise ValueError("accuracy roll must be in [0, 99]")
    if not 0 <= damage_roll < 16:
        raise ValueError("damage roll must be in [0, 15]")

    pp_after = max(0, context.move_pp - 1)
    if accuracy_roll >= context.accuracy:
        return AttackTransition(
            defender_hp=context.defender_hp,
            attacker_hp=context.attacker_hp,
            move_pp=pp_after,
            hit=False,
            defender_fainted=False,
            attacker_fainted=False,
        )

    dealt = damage_with_attack_stage(context.damage, attacker_stage, damage_roll)
    defender_after = max(0, context.defender_hp - dealt)
    attacker_after = context.attacker_hp
    if context.damage.attacker_item == ITEM_LIFE_ORB:
        recoil = max(1, context.attacker_max_hp // 10)
        attacker_after = max(0, attacker_after - recoil)

    return AttackTransition(
        defender_hp=defender_after,
        attacker_hp=attacker_after,
        move_pp=pp_after,
        hit=True,
        defender_fainted=defender_after == 0,
        attacker_fainted=attacker_after == 0,
    )


def staged_attack_dependency_key(
    context: AttackTransitionContext,
    attacker_stage: int,
    accuracy_roll: int,
    damage_roll: int,
) -> tuple[int, ...]:
    """Return the exact control-flow-sensitive key for a staged attack."""
    if not -6 <= attacker_stage <= 6:
        raise ValueError("attacker stage must be in [-6, 6]")

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
        attacker_stage,
        *damage_dependency_tuple(context.damage),
        damage_roll,
    )
