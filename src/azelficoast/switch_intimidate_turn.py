"""Intimidate composed into switch -> hazards -> queued physical attack.

The bounded ordering is:

1. selected teammate becomes active;
2. Stealth Rock and Spikes resolve;
3. if the incoming Pokémon fainted, its ability never starts;
4. otherwise Intimidate may publicly activate and lower the opponent's Attack stage;
5. the queued ordinary physical attack resolves using that updated stage.

The slice excludes Substitute and all effects that block, redirect, invert, copy, or
otherwise modify Intimidate's boost attempt. It also excludes non-physical queued moves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from azelficoast.core.projection import compile_projection_ids
from azelficoast.staged_attack import (
    staged_attack_dependency_key,
    staged_attack_transition,
)
from azelficoast.switch_hazard_turn import (
    SwitchHazardContext,
    resolve_entry_hazards,
    switch_hazard_dependency_signature,
    switch_hazard_turn,
)

SWITCH_INTIMIDATE_DEPENDENCY_SCHEMA_VERSION = 1


class SwitchIntimidateError(ValueError):
    """Raised when state is outside the bounded Intimidate switch model."""


@dataclass(frozen=True)
class SwitchIntimidateContext:
    switch_hazards: SwitchHazardContext
    incoming_has_intimidate: bool
    opponent_attack_stage: int

    def __post_init__(self) -> None:
        if not -6 <= self.opponent_attack_stage <= 6:
            raise SwitchIntimidateError("opponent Attack stage must be in [-6, 6]")
        if self.switch_hazards.switch.opponent_attack.damage.category != "Physical":
            raise SwitchIntimidateError(
                "bounded Intimidate switch requires a physical queued attack"
            )


@dataclass(frozen=True)
class SwitchIntimidateOutcome:
    active_slot: int
    bench_slot: int
    active_hp: int
    active_max_hp: int
    bench_hp: int
    bench_max_hp: int
    opponent_hp: int
    opponent_move_pp: int
    hazard_damage: int
    hazard_fainted: bool
    intimidate_activated: bool
    intimidate_changed_stage: bool
    opponent_attack_stage: int
    attack_executed: bool
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
            self.hazard_damage,
            int(self.hazard_fainted),
            int(self.intimidate_activated),
            int(self.intimidate_changed_stage),
            self.opponent_attack_stage,
            int(self.attack_executed),
            int(self.hit),
            int(self.active_fainted),
            int(self.opponent_fainted),
        )


@dataclass(frozen=True)
class SwitchIntimidateWorld:
    context_index: int
    accuracy_roll: int
    damage_roll: int
    untouched_bench_signature: int


@dataclass(frozen=True)
class SwitchIntimidateProjection:
    class_ids: np.ndarray
    representative_indices: np.ndarray
    effect_signature: str

    @property
    def class_count(self) -> int:
        return int(len(self.representative_indices))


def _stage_after_switch(context: SwitchIntimidateContext) -> tuple[bool, bool, int]:
    if not context.incoming_has_intimidate:
        return False, False, context.opponent_attack_stage

    before = context.opponent_attack_stage
    after = max(-6, before - 1)
    return True, after != before, after


def _post_hazard_attack_context(
    context: SwitchIntimidateContext,
    incoming_hp: int,
):
    attack = context.switch_hazards.switch.opponent_attack
    return replace(attack, defender_hp=incoming_hp)


def switch_intimidate_turn(
    context: SwitchIntimidateContext,
    *,
    accuracy_roll: int,
    damage_roll: int,
) -> SwitchIntimidateOutcome:
    hazards = resolve_entry_hazards(context.switch_hazards)
    switch = context.switch_hazards.switch

    if hazards.fainted:
        result = switch_hazard_turn(
            context.switch_hazards,
            accuracy_roll=accuracy_roll,
            damage_roll=damage_roll,
        )
        return SwitchIntimidateOutcome(
            active_slot=result.active_slot,
            bench_slot=result.bench_slot,
            active_hp=result.active_hp,
            active_max_hp=result.active_max_hp,
            bench_hp=result.bench_hp,
            bench_max_hp=result.bench_max_hp,
            opponent_hp=result.opponent_hp,
            opponent_move_pp=result.opponent_move_pp,
            hazard_damage=result.hazard_damage,
            hazard_fainted=True,
            intimidate_activated=False,
            intimidate_changed_stage=False,
            opponent_attack_stage=context.opponent_attack_stage,
            attack_executed=result.attack_executed,
            hit=result.hit,
            active_fainted=result.active_fainted,
            opponent_fainted=result.opponent_fainted,
        )

    activated, changed, stage_after = _stage_after_switch(context)
    attack = staged_attack_transition(
        _post_hazard_attack_context(context, hazards.incoming_hp_after),
        stage_after,
        accuracy_roll,
        damage_roll,
    )
    return SwitchIntimidateOutcome(
        active_slot=switch.incoming_slot,
        bench_slot=switch.outgoing_slot,
        active_hp=attack.defender_hp,
        active_max_hp=switch.incoming_max_hp,
        bench_hp=switch.outgoing_hp,
        bench_max_hp=switch.outgoing_max_hp,
        opponent_hp=attack.attacker_hp,
        opponent_move_pp=attack.move_pp,
        hazard_damage=hazards.total_damage,
        hazard_fainted=False,
        intimidate_activated=activated,
        intimidate_changed_stage=changed,
        opponent_attack_stage=stage_after,
        attack_executed=True,
        hit=attack.hit,
        active_fainted=attack.defender_fainted,
        opponent_fainted=attack.attacker_fainted,
    )


def switch_intimidate_dependency_key(
    context: SwitchIntimidateContext,
    *,
    accuracy_roll: int,
    damage_roll: int,
) -> tuple[int, ...]:
    hazards = resolve_entry_hazards(context.switch_hazards)
    switch = context.switch_hazards.switch

    if hazards.fainted:
        attack = switch.opponent_attack
        return (
            1,
            switch.outgoing_slot,
            switch.incoming_slot,
            switch.outgoing_hp,
            switch.outgoing_max_hp,
            switch.incoming_max_hp,
            hazards.total_damage,
            context.opponent_attack_stage,
            attack.attacker_hp,
            attack.move_pp,
        )

    activated, changed, stage_after = _stage_after_switch(context)
    attack = _post_hazard_attack_context(context, hazards.incoming_hp_after)
    return (
        0,
        switch.outgoing_slot,
        switch.incoming_slot,
        switch.outgoing_hp,
        switch.outgoing_max_hp,
        switch.incoming_max_hp,
        hazards.total_damage,
        int(activated),
        int(changed),
        stage_after,
        *staged_attack_dependency_key(
            attack,
            stage_after,
            accuracy_roll,
            damage_roll,
        ),
    )


def switch_intimidate_dependency_document() -> dict[str, object]:
    return {
        "schema": "azelficoast.switch-intimidate-dependencies",
        "schema_version": SWITCH_INTIMIDATE_DEPENDENCY_SCHEMA_VERSION,
        "binds": {
            "switch_hazards": switch_hazard_dependency_signature(),
        },
        "ordering": [
            "replace active identity",
            "resolve Stealth Rock and grounded Spikes",
            "process hazard faint before later SwitchIn handlers",
            "publicly activate Intimidate on a surviving Intimidate user",
            "lower opponent Attack stage by one, clamped at -6",
            "execute queued physical attack at resulting Attack stage",
        ],
        "dynamic_reads": {
            "hazard_faint": [
                "switch and effective hazard state",
                "opponent current Attack stage preserved",
                "queued move PP state",
            ],
            "survive_without_intimidate": [
                "switch and effective hazard state",
                "opponent current Attack stage",
                "staged physical attack dependencies",
            ],
            "survive_with_intimidate": [
                "incoming ability identity as a public observation",
                "opponent current Attack stage",
                "resulting Attack stage",
                "staged physical attack dependencies",
            ],
        },
        "control_flow_pruning": {
            "hazard_faint": (
                "Intimidate identity, accuracy RNG, damage RNG, damage stats, "
                "and Choice Band modifier are not read by the current-turn effect"
            ),
            "attack_miss": (
                "resulting Attack stage remains public state, but staged damage "
                "and damage RNG are not read"
            ),
        },
        "non_claims": [
            "Substitute interactions are excluded.",
            "Inner Focus, Oblivious, Own Tempo, Scrappy, Clear Body, White Smoke, Full Metal Body, Mirror Armor, and other boost-interception effects are excluded.",
            "Neutralizing Gas, Mold Breaker interactions, Trace, Skill Swap, and ability suppression are excluded.",
            "Special queued attacks are excluded from this treatment.",
            "Other switch-in abilities and items remain outside this slice.",
        ],
    }


def switch_intimidate_dependency_signature() -> str:
    encoded = json.dumps(
        switch_intimidate_dependency_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def compile_switch_intimidate_projection(
    contexts: Sequence[SwitchIntimidateContext],
    worlds: Sequence[SwitchIntimidateWorld],
) -> SwitchIntimidateProjection:
    if not contexts:
        raise SwitchIntimidateError("at least one Intimidate switch context is required")
    if not worlds:
        raise SwitchIntimidateError("at least one Intimidate switch world is required")

    def key_at(index: int) -> tuple[int, ...]:
        world = worlds[index]
        if not 0 <= world.context_index < len(contexts):
            raise SwitchIntimidateError(
                "world references an unavailable Intimidate switch context"
            )
        return switch_intimidate_dependency_key(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
        )

    class_ids, representatives = compile_projection_ids(len(worlds), key_at)
    return SwitchIntimidateProjection(
        class_ids=class_ids,
        representative_indices=representatives,
        effect_signature=switch_intimidate_dependency_signature(),
    )
