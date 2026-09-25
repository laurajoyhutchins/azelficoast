"""Bounded two-action turn with causal interaction between both actions.

The transition executes either a damaging p1 move or first-use Protect against a real
p2 damaging move. Damaging actions still respect priority/speed order, Moonblast-shaped
SpA drops, and KO cancellation. Protect is the bounded ordinary first-use case: it has
priority +4, succeeds deterministically, consumes PP, and blocks the modeled opposing
attack before accuracy/damage dependencies are read.

The slice still excludes repeated-Protect probability, Feint/Unseen Fist-style bypass,
switching, redirection, multihit moves, contact hooks, status effects, and residual or
end-turn processing.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from azelficoast.gen9_attack import (
    COMPILED_ATTACK_CONTEXT_WIDTH,
    AttackTransitionContext,
    attack_transition,
    attack_transition_dependency_key,
    attack_transition_dependency_signature,
    compile_attack_context,
)
from azelficoast.gen9_damage import (
    ITEM_LIFE_ORB,
    _kernel_modify,
    _kernel_ordinary_stat,
    _kernel_type_effectiveness,
    compile_numeric_context,
    damage_dependency_tuple,
)
from azelficoast.gen9_ordered_attack import ordered_attack_dependency_signature

P1_ATTACK_OFFSET = 0
P2_ATTACK_OFFSET = COMPILED_ATTACK_CONTEXT_WIDTH
P1_PRIORITY_COLUMN = COMPILED_ATTACK_CONTEXT_WIDTH * 2
P2_PRIORITY_COLUMN = P1_PRIORITY_COLUMN + 1
P1_SPEED_COLUMN = P1_PRIORITY_COLUMN + 2
P2_SPEED_COLUMN = P1_PRIORITY_COLUMN + 3
P1_SPA_DROP_CHANCE_COLUMN = P1_PRIORITY_COLUMN + 4
P2_SPA_STAGE_COLUMN = P1_PRIORITY_COLUMN + 5
P1_ACTION_KIND_COLUMN = P1_PRIORITY_COLUMN + 6
COMPILED_TWO_ATTACK_TURN_CONTEXT_WIDTH = P1_PRIORITY_COLUMN + 7

P1_ACTION_ATTACK = 0
P1_ACTION_PROTECT = 1

PACK_HP_BASE = 512
PACK_PP_BASE = 8
PACK_P1_PP_BASE = PACK_HP_BASE * PACK_HP_BASE
PACK_P2_PP_BASE = PACK_P1_PP_BASE * PACK_PP_BASE
PACK_STAGE_BASE = PACK_P2_PP_BASE * PACK_PP_BASE
PACK_FLAGS_BASE = PACK_STAGE_BASE * 13

FLAG_P1_ACTED = 1
FLAG_P2_ACTED = 2

TWO_ATTACK_TURN_DEPENDENCY_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class TwoAttackTurnContext:
    p1_attack: AttackTransitionContext
    p2_attack: AttackTransitionContext
    p1_priority: int
    p2_priority: int
    p1_speed: int
    p2_speed: int
    p1_spa_drop_chance: int
    p2_spa_stage: int
    p1_action_kind: int = P1_ACTION_ATTACK

    def __post_init__(self) -> None:
        if self.p1_attack.attacker_hp != self.p2_attack.defender_hp:
            raise ValueError("p1 HP must agree across both attack contexts")
        if self.p1_attack.defender_hp != self.p2_attack.attacker_hp:
            raise ValueError("p2 HP must agree across both attack contexts")
        if self.p1_attack.attacker_max_hp >= PACK_HP_BASE:
            raise ValueError("p1 max HP is outside the packed two-attack domain")
        if self.p2_attack.attacker_max_hp >= PACK_HP_BASE:
            raise ValueError("p2 max HP is outside the packed two-attack domain")
        if self.p1_attack.move_pp >= PACK_PP_BASE:
            raise ValueError("p1 PP must fit the bounded two-attack packed state")
        if self.p2_attack.move_pp >= PACK_PP_BASE:
            raise ValueError("p2 PP must fit the bounded two-attack packed state")
        if not -7 <= self.p1_priority <= 7 or not -7 <= self.p2_priority <= 7:
            raise ValueError("move priority is outside the bounded turn domain")
        if self.p1_speed <= 0 or self.p2_speed <= 0:
            raise ValueError("turn speeds must be positive")
        if not 0 <= self.p1_spa_drop_chance <= 100:
            raise ValueError("p1 SpA-drop chance must be in [0, 100]")
        if not -6 <= self.p2_spa_stage <= 6:
            raise ValueError("p2 SpA stage must be in [-6, 6]")
        if self.p1_action_kind not in (P1_ACTION_ATTACK, P1_ACTION_PROTECT):
            raise ValueError("p1 action kind is outside the bounded turn domain")
        if self.p1_action_kind == P1_ACTION_PROTECT and self.p1_priority != 4:
            raise ValueError("bounded Protect must use priority +4")
        if self.p2_attack.damage.category != "Special":
            raise ValueError("the bounded p2 attack must be Special")
        if self.p1_attack.damage.attacker_item == ITEM_LIFE_ORB:
            raise ValueError("p1 Life Orb recoil is outside the bounded two-attack turn")
        if self.p2_attack.damage.attacker_item == ITEM_LIFE_ORB:
            raise ValueError("p2 Life Orb recoil is outside the bounded two-attack turn")


@dataclass(frozen=True)
class TwoAttackTurn:
    p1_hp: int
    p2_hp: int
    p1_pp: int
    p2_pp: int
    p2_spa_stage: int
    p1_acted: bool
    p2_acted: bool

    @property
    def p1_fainted(self) -> bool:
        return self.p1_hp == 0

    @property
    def p2_fainted(self) -> bool:
        return self.p2_hp == 0

    @property
    def packed(self) -> int:
        flags = int(self.p1_acted) * FLAG_P1_ACTED + int(self.p2_acted) * FLAG_P2_ACTED
        return (
            self.p1_hp
            + self.p2_hp * PACK_HP_BASE
            + self.p1_pp * PACK_P1_PP_BASE
            + self.p2_pp * PACK_P2_PP_BASE
            + (self.p2_spa_stage + 6) * PACK_STAGE_BASE
            + flags * PACK_FLAGS_BASE
        )


def compile_two_attack_turn_context(context: TwoAttackTurnContext) -> tuple[int, ...]:
    return (
        *compile_attack_context(context.p1_attack),
        *compile_attack_context(context.p2_attack),
        context.p1_priority,
        context.p2_priority,
        context.p1_speed,
        context.p2_speed,
        context.p1_spa_drop_chance,
        context.p2_spa_stage,
        context.p1_action_kind,
    )


def _turn_stage_stat_numeric(stat: int, stage: int) -> int:
    if stage >= 0:
        return (stat * (2 + stage)) // 2
    return (stat * 2) // (2 - stage)


def _turn_damage_numeric(
    params: tuple[int, ...],
    offset: int,
    stage: int,
    roll: int,
) -> int:
    attacker_level = params[offset + 0]
    attack = _kernel_ordinary_stat(
        params[offset + 3],
        params[offset + 4],
        params[offset + 5],
        attacker_level,
        params[offset + 6],
    )
    attack = _turn_stage_stat_numeric(attack, stage)
    attack = _kernel_modify(attack, params[offset + 12])
    defense = _kernel_ordinary_stat(
        params[offset + 7],
        params[offset + 8],
        params[offset + 9],
        params[offset + 1],
        params[offset + 10],
    )
    defense = _kernel_modify(defense, params[offset + 11])

    result = (((2 * attacker_level) // 5 + 2) * params[offset + 2] * attack) // defense
    result = result // 50 + 2
    result = (result * (100 - roll)) // 100
    result = _kernel_modify(result, params[offset + 13])
    result = _kernel_type_effectiveness(result, params[offset + 14])
    result = _kernel_modify(result, params[offset + 15])
    result = _kernel_modify(result, params[offset + 16])
    if result < 1:
        result = 1
    return result


def _turn_p1_first_numeric(params: tuple[int, ...], order_tie_roll: int) -> int:
    p1_priority = params[46]
    p2_priority = params[47]
    if p1_priority > p2_priority:
        return 1
    if p1_priority < p2_priority:
        return 0

    p1_speed = params[48]
    p2_speed = params[49]
    if p1_speed > p2_speed:
        return 1
    if p1_speed < p2_speed:
        return 0

    if order_tie_roll == 0:
        return 1
    return 0


def two_attack_turn_numeric(
    params: tuple[int, ...],
    order_tie_roll: int,
    p1_accuracy_roll: int,
    p1_damage_roll: int,
    p1_secondary_roll: int,
    p2_accuracy_roll: int,
    p2_damage_roll: int,
) -> int:
    """Execute the bounded two-action turn and return one exact packed post-state."""
    p1_hp = params[19]
    p2_hp = params[21]
    p1_pp = params[22]
    p2_pp = params[45]
    p2_stage = params[51]
    p1_action_kind = params[52]

    p1_acted = 0
    p2_acted = 0
    p1_hit = 0
    p1_damage = 0
    p2_damage = 0
    p1_first = _turn_p1_first_numeric(params, order_tie_roll)

    if p1_first != 0:
        p1_acted = 1
        p1_pp -= 1
        if p1_pp < 0:
            p1_pp = 0

        if p1_action_kind == 1:
            if p2_hp > 0:
                p2_acted = 1
                p2_pp -= 1
                if p2_pp < 0:
                    p2_pp = 0
        else:
            if p1_accuracy_roll < params[18]:
                p1_hit = 1
                p1_damage = _turn_damage_numeric(params, 0, 0, p1_damage_roll)
                p2_hp -= p1_damage
                if p2_hp < 0:
                    p2_hp = 0

            if p1_hit != 0:
                if p2_hp > 0:
                    if p1_secondary_roll < params[50]:
                        if p2_stage > -6:
                            p2_stage -= 1

            if p2_hp > 0:
                p2_acted = 1
                p2_pp -= 1
                if p2_pp < 0:
                    p2_pp = 0
                if p2_accuracy_roll < params[41]:
                    p2_damage = _turn_damage_numeric(
                        params,
                        23,
                        p2_stage,
                        p2_damage_roll,
                    )
                    p1_hp -= p2_damage
                    if p1_hp < 0:
                        p1_hp = 0
    else:
        p2_acted = 1
        p2_pp -= 1
        if p2_pp < 0:
            p2_pp = 0
        if p2_accuracy_roll < params[41]:
            p2_damage = _turn_damage_numeric(
                params,
                23,
                p2_stage,
                p2_damage_roll,
            )
            p1_hp -= p2_damage
            if p1_hp < 0:
                p1_hp = 0

        if p1_hp > 0:
            p1_acted = 1
            p1_pp -= 1
            if p1_pp < 0:
                p1_pp = 0

            if p1_action_kind != 1:
                p1_hit = 0
                if p1_accuracy_roll < params[18]:
                    p1_hit = 1
                    p1_damage = _turn_damage_numeric(
                        params,
                        0,
                        0,
                        p1_damage_roll,
                    )
                    p2_hp -= p1_damage
                    if p2_hp < 0:
                        p2_hp = 0

                if p1_hit != 0:
                    if p2_hp > 0:
                        if p1_secondary_roll < params[50]:
                            if p2_stage > -6:
                                p2_stage -= 1

    flags = p1_acted + p2_acted * 2
    return (
        p1_hp
        + p2_hp * 512
        + p1_pp * 262144
        + p2_pp * 2097152
        + (p2_stage + 6) * 16777216
        + flags * 218103808
    )


def unpack_two_attack_turn(packed: int) -> TwoAttackTurn:
    if packed < 0:
        raise ValueError("packed two-attack turn must be non-negative")
    flags, remainder = divmod(packed, PACK_FLAGS_BASE)
    stage_encoded, remainder = divmod(remainder, PACK_STAGE_BASE)
    p2_pp, remainder = divmod(remainder, PACK_P2_PP_BASE)
    p1_pp, remainder = divmod(remainder, PACK_P1_PP_BASE)
    p2_hp, p1_hp = divmod(remainder, PACK_HP_BASE)
    return TwoAttackTurn(
        p1_hp=p1_hp,
        p2_hp=p2_hp,
        p1_pp=p1_pp,
        p2_pp=p2_pp,
        p2_spa_stage=stage_encoded - 6,
        p1_acted=bool(flags & FLAG_P1_ACTED),
        p2_acted=bool(flags & FLAG_P2_ACTED),
    )


def two_attack_turn(
    context: TwoAttackTurnContext,
    *,
    order_tie_roll: int,
    p1_accuracy_roll: int,
    p1_damage_roll: int,
    p1_secondary_roll: int,
    p2_accuracy_roll: int,
    p2_damage_roll: int,
) -> TwoAttackTurn:
    if order_tie_roll not in (0, 1):
        raise ValueError("order tie roll must be 0 or 1")
    for name, value in (
        ("p1 accuracy", p1_accuracy_roll),
        ("p1 secondary", p1_secondary_roll),
        ("p2 accuracy", p2_accuracy_roll),
    ):
        if not 0 <= value < 100:
            raise ValueError(f"{name} roll must be in [0, 99]")
    for name, value in (("p1 damage", p1_damage_roll), ("p2 damage", p2_damage_roll)):
        if not 0 <= value < 16:
            raise ValueError(f"{name} roll must be in [0, 15]")
    return unpack_two_attack_turn(
        two_attack_turn_numeric(
            compile_two_attack_turn_context(context),
            order_tie_roll,
            p1_accuracy_roll,
            p1_damage_roll,
            p1_secondary_roll,
            p2_accuracy_roll,
            p2_damage_roll,
        )
    )


def _staged_damage(context: AttackTransitionContext, stage: int, roll: int) -> int:
    return _turn_damage_numeric(compile_numeric_context(context.damage), 0, stage, roll)


def _staged_attack_dependency_key(
    attack: AttackTransitionContext,
    stage: int,
    accuracy_roll: int,
    damage_roll: int,
) -> tuple[int, ...]:
    hit = int(accuracy_roll < attack.accuracy)
    state = (
        hit,
        attack.attacker_hp,
        attack.attacker_max_hp,
        attack.defender_hp,
        attack.move_pp,
        stage,
    )
    if not hit:
        return state
    return (*state, *damage_dependency_tuple(attack.damage), damage_roll)


def two_attack_turn_dependency_key(
    context: TwoAttackTurnContext,
    *,
    order_tie_roll: int,
    p1_accuracy_roll: int,
    p1_damage_roll: int,
    p1_secondary_roll: int,
    p2_accuracy_roll: int,
    p2_damage_roll: int,
    include_order_tie: bool = True,
    include_secondary: bool = True,
) -> tuple[int, ...]:
    """Return the dynamic dependency partition key for the complete bounded turn."""
    effective_order = order_tie_roll if include_order_tie else 0
    effective_secondary = p1_secondary_roll if include_secondary else 99
    p1_first = _turn_p1_first_numeric(
        compile_two_attack_turn_context(context),
        effective_order,
    )

    if context.p1_action_kind == P1_ACTION_PROTECT:
        if p1_first:
            return (
                P1_ACTION_PROTECT,
                1,
                context.p1_attack.attacker_hp,
                context.p1_attack.move_pp,
                context.p2_attack.attacker_hp,
                context.p2_attack.move_pp,
                context.p2_spa_stage,
            )

        p2_key = _staged_attack_dependency_key(
            context.p2_attack,
            context.p2_spa_stage,
            p2_accuracy_roll,
            p2_damage_roll,
        )
        p1_hp_after = context.p1_attack.attacker_hp
        if p2_accuracy_roll < context.p2_attack.accuracy:
            p1_hp_after -= _staged_damage(
                context.p2_attack,
                context.p2_spa_stage,
                p2_damage_roll,
            )
            if p1_hp_after < 0:
                p1_hp_after = 0
        return (
            P1_ACTION_PROTECT,
            0,
            *p2_key,
            int(p1_hp_after > 0),
            context.p1_attack.move_pp if p1_hp_after > 0 else 0,
        )

    if p1_first:
        p1_key = attack_transition_dependency_key(
            context.p1_attack,
            p1_accuracy_roll,
            p1_damage_roll,
        )
        p1_result = attack_transition(
            context.p1_attack,
            p1_accuracy_roll,
            p1_damage_roll,
        )
        if p1_result.defender_fainted:
            return (P1_ACTION_ATTACK, 1, context.p2_spa_stage, *p1_key, 0)

        stage_after = context.p2_spa_stage
        secondary_applied = int(
            p1_result.hit
            and effective_secondary < context.p1_spa_drop_chance
            and stage_after > -6
        )
        if secondary_applied:
            stage_after -= 1

        p2_key = _staged_attack_dependency_key(
            context.p2_attack,
            stage_after,
            p2_accuracy_roll,
            p2_damage_roll,
        )
        return (P1_ACTION_ATTACK, 1, *p1_key, secondary_applied, stage_after, *p2_key)

    p2_key = _staged_attack_dependency_key(
        context.p2_attack,
        context.p2_spa_stage,
        p2_accuracy_roll,
        p2_damage_roll,
    )
    p1_hp_after = context.p1_attack.attacker_hp
    if p2_accuracy_roll < context.p2_attack.accuracy:
        p1_hp_after -= _staged_damage(
            context.p2_attack,
            context.p2_spa_stage,
            p2_damage_roll,
        )
        if p1_hp_after < 0:
            p1_hp_after = 0
    if p1_hp_after == 0:
        return (P1_ACTION_ATTACK, 0, *p2_key, 0)

    p1_key = attack_transition_dependency_key(
        context.p1_attack,
        p1_accuracy_roll,
        p1_damage_roll,
    )
    p1_result = attack_transition(
        context.p1_attack,
        p1_accuracy_roll,
        p1_damage_roll,
    )
    stage_after = context.p2_spa_stage
    secondary_applied = int(
        p1_result.hit
        and not p1_result.defender_fainted
        and effective_secondary < context.p1_spa_drop_chance
        and stage_after > -6
    )
    if secondary_applied:
        stage_after -= 1
    return (P1_ACTION_ATTACK, 0, *p2_key, *p1_key, secondary_applied, stage_after)


def two_attack_turn_dependency_document() -> dict[str, object]:
    return {
        "schema": "azelficoast.two-attack-turn-dependencies",
        "schema_version": TWO_ATTACK_TURN_DEPENDENCY_SCHEMA_VERSION,
        "binds": {
            "whole_attack": attack_transition_dependency_signature(),
            "ordered_attack": ordered_attack_dependency_signature(),
        },
        "order": {
            "reads": [
                "p1.move.priority",
                "p2.move.priority",
                "p1.speed",
                "p2.speed",
            ],
            "conditional_random": "rng.final_speed_tie_outcome",
        },
        "p1_action": {
            "kind": {
                "values": ["attack", "first-use-protect"],
                "read": "p1.action.kind",
            },
            "attack_reads": ["whole_attack.dependencies"],
            "protect": {
                "priority": 4,
                "success": "deterministic first-use bounded case",
                "effect": (
                    "modeled opposing attack consumes PP but is blocked before "
                    "accuracy, damage RNG, damage stats, item modifiers, or HP writes"
                ),
            },
            "secondary": {
                "condition": "p1 hit AND p2 survives AND roll < chance",
                "reads": [
                    "p1.move.spa_drop_chance",
                    "p2.spa_stage",
                    "rng.p1_secondary_roll",
                ],
                "writes": ["p2.spa_stage"],
            },
        },
        "p2_action": {
            "condition": "p2 is alive when its queued action is reached",
            "reads": [
                "whole_attack.dependencies",
                "p2.spa_stage_at_execution",
            ],
        },
        "protection_boundary": {
            "covered": "ordinary opposing damaging move blocked by first-use Protect",
            "excluded": [
                "consecutive Protect probability",
                "Protect-bypassing moves and abilities",
            ],
        },
        "cancellation": {
            "condition": "first actor faints the queued second actor",
            "effect": "second action is not executed and its PP/RNG dependencies are irrelevant",
        },
        "packed_state": {
            "hp_base": PACK_HP_BASE,
            "pp_base": PACK_PP_BASE,
            "stage_base": PACK_STAGE_BASE,
            "flags_base": PACK_FLAGS_BASE,
        },
    }


def two_attack_turn_dependency_signature() -> str:
    encoded = json.dumps(
        two_attack_turn_dependency_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()
