"""Entry hazards composed with certified voluntary switch-before-attack mechanics.

The bounded ordering is:

1. selected teammate becomes active;
2. Stealth Rock and Spikes damage the incoming Pokémon;
3. if the incoming Pokémon survives, the certified opposing attack executes;
4. if hazards faint it, the queued move action consumes PP but has no live target.

Heavy-Duty Boots bypass both modeled hazards. Spikes additionally require groundedness.
Magic Guard, item suppression, switch-in/out abilities, Toxic Spikes, Sticky Web, forced
switches, pivot moves, and residual/end-turn mechanics remain outside this slice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from azelficoast.core.projection import compile_projection_ids
from azelficoast.voluntary_switch_turn import (
    VoluntarySwitchContext,
    voluntary_switch_dependency_key,
    voluntary_switch_dependency_signature,
    voluntary_switch_turn,
)

SWITCH_HAZARD_DEPENDENCY_SCHEMA_VERSION = 1


class SwitchHazardError(ValueError):
    """Raised when state is outside the bounded entry-hazard model."""


@dataclass(frozen=True)
class SwitchHazardContext:
    switch: VoluntarySwitchContext
    stealth_rock: bool
    spikes_layers: int
    incoming_has_heavy_duty_boots: bool
    incoming_grounded: bool
    stealth_rock_type_mod: int

    def __post_init__(self) -> None:
        if not 0 <= self.spikes_layers <= 3:
            raise SwitchHazardError("Spikes layers must be in [0, 3]")
        if not -6 <= self.stealth_rock_type_mod <= 6:
            raise SwitchHazardError("Stealth Rock type modifier must be in [-6, 6]")


@dataclass(frozen=True)
class EntryHazardResolution:
    incoming_hp_before: int
    incoming_hp_after: int
    stealth_rock_damage: int
    spikes_damage: int

    @property
    def total_damage(self) -> int:
        return self.incoming_hp_before - self.incoming_hp_after

    @property
    def fainted(self) -> bool:
        return self.incoming_hp_after == 0


@dataclass(frozen=True)
class SwitchHazardOutcome:
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
            int(self.attack_executed),
            int(self.hit),
            int(self.active_fainted),
            int(self.opponent_fainted),
        )


@dataclass(frozen=True)
class SwitchHazardWorld:
    context_index: int
    accuracy_roll: int
    damage_roll: int
    untouched_bench_signature: int


@dataclass(frozen=True)
class SwitchHazardProjection:
    class_ids: np.ndarray
    representative_indices: np.ndarray
    effect_signature: str

    @property
    def class_count(self) -> int:
        return int(len(self.representative_indices))


def _fractional_hazard_damage(max_hp: int, numerator: int, denominator: int) -> int:
    if numerator <= 0 or denominator <= 0:
        raise SwitchHazardError("hazard damage fraction must be positive")
    return max(1, (max_hp * numerator) // denominator)


def stealth_rock_damage(max_hp: int, type_mod: int) -> int:
    if not 1 <= max_hp < 1024:
        raise SwitchHazardError("hazard max HP is outside the bounded domain")
    if not -6 <= type_mod <= 6:
        raise SwitchHazardError("Stealth Rock type modifier must be in [-6, 6]")
    if type_mod >= 0:
        return _fractional_hazard_damage(max_hp, 1 << type_mod, 8)
    return _fractional_hazard_damage(max_hp, 1, 8 * (1 << (-type_mod)))


def spikes_damage(max_hp: int, layers: int) -> int:
    if not 1 <= max_hp < 1024:
        raise SwitchHazardError("hazard max HP is outside the bounded domain")
    if layers not in (1, 2, 3):
        raise SwitchHazardError("Spikes damage requires 1, 2, or 3 layers")
    numerator = (3, 4, 6)[layers - 1]
    return _fractional_hazard_damage(max_hp, numerator, 24)


def resolve_entry_hazards(context: SwitchHazardContext) -> EntryHazardResolution:
    switch = context.switch
    hp_before = switch.opponent_attack.defender_hp
    hp = hp_before
    rock_damage = 0
    spike_damage = 0

    if context.incoming_has_heavy_duty_boots:
        return EntryHazardResolution(
            incoming_hp_before=hp_before,
            incoming_hp_after=hp_before,
            stealth_rock_damage=0,
            spikes_damage=0,
        )

    if context.stealth_rock:
        raw = stealth_rock_damage(
            switch.incoming_max_hp,
            context.stealth_rock_type_mod,
        )
        rock_damage = min(hp, raw)
        hp -= rock_damage

    if hp > 0 and context.spikes_layers and context.incoming_grounded:
        raw = spikes_damage(switch.incoming_max_hp, context.spikes_layers)
        spike_damage = min(hp, raw)
        hp -= spike_damage

    return EntryHazardResolution(
        incoming_hp_before=hp_before,
        incoming_hp_after=hp,
        stealth_rock_damage=rock_damage,
        spikes_damage=spike_damage,
    )


def _switch_after_hazards(
    context: SwitchHazardContext,
    hazards: EntryHazardResolution,
) -> VoluntarySwitchContext:
    return replace(
        context.switch,
        opponent_attack=replace(
            context.switch.opponent_attack,
            defender_hp=hazards.incoming_hp_after,
        ),
    )


def switch_hazard_turn(
    context: SwitchHazardContext,
    *,
    accuracy_roll: int,
    damage_roll: int,
) -> SwitchHazardOutcome:
    hazards = resolve_entry_hazards(context)
    switch = context.switch

    if hazards.fainted:
        attack = switch.opponent_attack
        return SwitchHazardOutcome(
            active_slot=switch.incoming_slot,
            bench_slot=switch.outgoing_slot,
            active_hp=0,
            active_max_hp=switch.incoming_max_hp,
            bench_hp=switch.outgoing_hp,
            bench_max_hp=switch.outgoing_max_hp,
            opponent_hp=attack.attacker_hp,
            opponent_move_pp=max(0, attack.move_pp - 1),
            hazard_damage=hazards.total_damage,
            hazard_fainted=True,
            attack_executed=True,
            hit=False,
            active_fainted=True,
            opponent_fainted=False,
        )

    result = voluntary_switch_turn(
        _switch_after_hazards(context, hazards),
        accuracy_roll=accuracy_roll,
        damage_roll=damage_roll,
    )
    return SwitchHazardOutcome(
        active_slot=result.active_slot,
        bench_slot=result.bench_slot,
        active_hp=result.active_hp,
        active_max_hp=result.active_max_hp,
        bench_hp=result.bench_hp,
        bench_max_hp=result.bench_max_hp,
        opponent_hp=result.opponent_hp,
        opponent_move_pp=result.opponent_move_pp,
        hazard_damage=hazards.total_damage,
        hazard_fainted=False,
        attack_executed=True,
        hit=result.hit,
        active_fainted=result.active_fainted,
        opponent_fainted=result.opponent_fainted,
    )


def switch_hazard_dependency_key(
    context: SwitchHazardContext,
    *,
    accuracy_roll: int,
    damage_roll: int,
) -> tuple[int, ...]:
    """Return the control-flow-sensitive execution key for switch + hazards + attack."""

    hazards = resolve_entry_hazards(context)
    switch = context.switch

    if hazards.fainted:
        attack = switch.opponent_attack
        return (
            switch.outgoing_slot,
            switch.incoming_slot,
            switch.outgoing_hp,
            switch.outgoing_max_hp,
            switch.incoming_max_hp,
            1,
            hazards.total_damage,
            attack.attacker_hp,
            attack.move_pp,
        )

    adjusted = _switch_after_hazards(context, hazards)
    return (
        0,
        hazards.total_damage,
        *voluntary_switch_dependency_key(
            adjusted,
            accuracy_roll=accuracy_roll,
            damage_roll=damage_roll,
        ),
    )


def switch_hazard_dependency_document() -> dict[str, object]:
    return {
        "schema": "azelficoast.switch-entry-hazard-dependencies",
        "schema_version": SWITCH_HAZARD_DEPENDENCY_SCHEMA_VERSION,
        "binds": {
            "switch_before_attack": voluntary_switch_dependency_signature(),
        },
        "ordering": [
            "replace active identity",
            "apply Stealth Rock",
            "apply grounded Spikes if still alive",
            "consume queued opponent move PP without target-dependent execution on hazard faint",
            "otherwise execute certified opponent attack",
        ],
        "hazards": {
            "heavy_duty_boots": "bypasses Stealth Rock and Spikes",
            "stealth_rock": "floor(max_hp * 2^type_mod / 8), minimum 1",
            "spikes": {
                "requires_grounded": True,
                "layers": {
                    "1": "floor(max_hp * 3 / 24), minimum 1",
                    "2": "floor(max_hp * 4 / 24), minimum 1",
                    "3": "floor(max_hp * 6 / 24), minimum 1",
                },
            },
        },
        "dynamic_reads": {
            "hazard_faint": [
                "switch identity/state",
                "effective hazard damage",
                "opponent hp and move pp; queued move consumes PP without target-dependent reads",
            ],
            "survive": [
                "switch identity/state",
                "effective hazard damage",
                "voluntary-switch attack dependencies at post-hazard hp",
            ],
        },
        "factored_external_state": [
            "untouched teammate state",
            "raw hazard representation when it yields identical current-turn effects",
        ],
        "non_claims": [
            "Magic Guard and item-suppression effects are excluded.",
            "Toxic Spikes and Sticky Web are excluded.",
            "Switch-in and switch-out abilities and items other than Heavy-Duty Boots are excluded.",
            "Forced switches, pivot moves, trapping, and residual/end-turn effects are excluded.",
        ],
    }


def switch_hazard_dependency_signature() -> str:
    encoded = json.dumps(
        switch_hazard_dependency_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def compile_switch_hazard_projection(
    contexts: Sequence[SwitchHazardContext],
    worlds: Sequence[SwitchHazardWorld],
) -> SwitchHazardProjection:
    if not contexts:
        raise SwitchHazardError("at least one switch-hazard context is required")
    if not worlds:
        raise SwitchHazardError("at least one switch-hazard world is required")

    def key_at(index: int) -> tuple[int, ...]:
        world = worlds[index]
        if not 0 <= world.context_index < len(contexts):
            raise SwitchHazardError("world references an unavailable switch-hazard context")
        return switch_hazard_dependency_key(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
        )

    class_ids, representatives = compile_projection_ids(len(worlds), key_at)
    return SwitchHazardProjection(
        class_ids=class_ids,
        representative_indices=representatives,
        effect_signature=switch_hazard_dependency_signature(),
    )
