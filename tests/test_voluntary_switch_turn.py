from __future__ import annotations

from dataclasses import replace

import pytest

from azelficoast.research.mechanics.gen9_attack import AttackTransitionContext
from azelficoast.research.mechanics.gen9_damage import DamageContext, ITEM_CHOICE_SPECS
from azelficoast.research.mechanics.voluntary_switch_turn import (
    VoluntarySwitchContext,
    VoluntarySwitchError,
    VoluntarySwitchWorld,
    compile_voluntary_switch_projection,
    voluntary_switch_dependency_key,
    voluntary_switch_turn,
)


def _damage(
    *,
    item: str = "",
    defender_base_stat: int = 95,
    type_mod: int = 0,
) -> DamageContext:
    return DamageContext(
        attacker_level=100,
        defender_level=100,
        base_power=110,
        category="Special",
        move_id="hydropump",
        move_type="Water",
        attacker_types=("Psychic",),
        tera_type=None,
        attacker_base_stat=100,
        attacker_iv=31,
        attacker_ev=252,
        attacker_nature_percent=110,
        defender_base_stat=defender_base_stat,
        defender_iv=31,
        defender_ev=252,
        defender_nature_percent=110,
        attacker_item=item,
        type_mod=type_mod,
    )


def _context(
    *,
    incoming_slot: int = 1,
    item: str = "",
) -> VoluntarySwitchContext:
    if incoming_slot == 1:
        incoming_hp = 400
        incoming_max_hp = 500
        defender_base_stat = 95
        type_mod = -1
    else:
        incoming_hp = 600
        incoming_max_hp = 700
        defender_base_stat = 135
        type_mod = 0

    return VoluntarySwitchContext(
        outgoing_slot=0,
        incoming_slot=incoming_slot,
        outgoing_hp=200,
        outgoing_max_hp=341,
        incoming_max_hp=incoming_max_hp,
        opponent_attack=AttackTransitionContext(
            damage=_damage(
                item=item,
                defender_base_stat=defender_base_stat,
                type_mod=type_mod,
            ),
            accuracy=80,
            attacker_hp=341,
            attacker_max_hp=341,
            defender_hp=incoming_hp,
            move_pp=8,
        ),
    )


def test_switch_replaces_active_before_opponent_attack() -> None:
    outcome = voluntary_switch_turn(
        _context(incoming_slot=1),
        accuracy_roll=0,
        damage_roll=7,
    )

    assert outcome.active_slot == 1
    assert outcome.bench_slot == 0
    assert outcome.bench_hp == 200
    assert outcome.active_hp < 400
    assert outcome.opponent_move_pp == 7
    assert outcome.hit is True


def test_miss_drops_opponent_item_and_damage_rng_dependencies() -> None:
    plain = _context(item="")
    specs = _context(item=ITEM_CHOICE_SPECS)

    plain_key = voluntary_switch_dependency_key(
        plain,
        accuracy_roll=99,
        damage_roll=0,
    )
    specs_key = voluntary_switch_dependency_key(
        specs,
        accuracy_roll=99,
        damage_roll=15,
    )
    assert plain_key == specs_key

    plain_outcome = voluntary_switch_turn(
        plain,
        accuracy_roll=99,
        damage_roll=0,
    )
    specs_outcome = voluntary_switch_turn(
        specs,
        accuracy_roll=99,
        damage_roll=15,
    )
    assert plain_outcome.successor_key == specs_outcome.successor_key


def test_hit_restores_opponent_item_and_damage_rng_dependencies() -> None:
    plain = _context(item="")
    specs = _context(item=ITEM_CHOICE_SPECS)

    plain_key = voluntary_switch_dependency_key(
        plain,
        accuracy_roll=0,
        damage_roll=0,
    )
    specs_key = voluntary_switch_dependency_key(
        specs,
        accuracy_roll=0,
        damage_roll=15,
    )
    assert plain_key != specs_key

    plain_outcome = voluntary_switch_turn(
        plain,
        accuracy_roll=0,
        damage_roll=0,
    )
    specs_outcome = voluntary_switch_turn(
        specs,
        accuracy_roll=0,
        damage_roll=15,
    )
    assert specs_outcome.active_hp < plain_outcome.active_hp


def test_selected_incoming_identity_never_collapses() -> None:
    first = _context(incoming_slot=1)
    same_mechanics_other_slot = replace(first, incoming_slot=2)

    first_key = voluntary_switch_dependency_key(
        first,
        accuracy_roll=99,
        damage_roll=15,
    )
    second_key = voluntary_switch_dependency_key(
        same_mechanics_other_slot,
        accuracy_roll=99,
        damage_roll=15,
    )

    assert first_key != second_key


def test_projection_drops_untouched_bench_factor_and_preserves_successors() -> None:
    contexts = (
        _context(incoming_slot=1, item=""),
        _context(incoming_slot=1, item=ITEM_CHOICE_SPECS),
        _context(incoming_slot=2, item=""),
        _context(incoming_slot=2, item=ITEM_CHOICE_SPECS),
    )
    worlds = [
        VoluntarySwitchWorld(
            context_index=context_index,
            accuracy_roll=accuracy_roll,
            damage_roll=damage_roll,
            untouched_bench_signature=untouched,
        )
        for context_index in range(len(contexts))
        for accuracy_roll in (0, 99)
        for damage_roll in range(16)
        for untouched in range(16)
    ]

    projection = compile_voluntary_switch_projection(contexts, worlds)
    direct = [
        voluntary_switch_turn(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
        ).successor_key
        for world in worlds
    ]
    representative_outputs = [
        direct[int(index)]
        for index in projection.representative_indices
    ]
    expanded = [
        representative_outputs[int(class_id)]
        for class_id in projection.class_ids
    ]

    miss_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if world.accuracy_roll == 99
    }
    hit_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if world.accuracy_roll == 0
    }

    assert len(worlds) == 2048
    assert projection.class_count == 66
    assert len(miss_ids) == 2
    assert len(hit_ids) == 64
    assert miss_ids.isdisjoint(hit_ids)
    assert direct == expanded


def test_same_slot_and_invalid_hp_fail_closed() -> None:
    with pytest.raises(VoluntarySwitchError, match="change active slot"):
        replace(_context(), incoming_slot=0)

    with pytest.raises(VoluntarySwitchError, match="outgoing HP"):
        replace(_context(), outgoing_hp=0)
