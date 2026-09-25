from __future__ import annotations

from azelficoast.gen9_attack import AttackTransitionContext
from azelficoast.gen9_damage import DamageContext, ITEM_CHOICE_SPECS
from azelficoast.switch_hazard_turn import (
    SwitchHazardContext,
    SwitchHazardError,
    SwitchHazardWorld,
    compile_switch_hazard_projection,
    resolve_entry_hazards,
    spikes_damage,
    stealth_rock_damage,
    switch_hazard_dependency_key,
    switch_hazard_turn,
)
from azelficoast.voluntary_switch_turn import VoluntarySwitchContext


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


def _switch(
    *,
    incoming_slot: int,
    incoming_hp: int,
    incoming_max_hp: int,
    opponent_item: str,
    defender_base_stat: int = 95,
    attack_type_mod: int = 0,
) -> VoluntarySwitchContext:
    return VoluntarySwitchContext(
        outgoing_slot=0,
        incoming_slot=incoming_slot,
        outgoing_hp=200,
        outgoing_max_hp=341,
        incoming_max_hp=incoming_max_hp,
        opponent_attack=AttackTransitionContext(
            damage=_damage(
                item=opponent_item,
                defender_base_stat=defender_base_stat,
                type_mod=attack_type_mod,
            ),
            accuracy=80,
            attacker_hp=341,
            attacker_max_hp=341,
            defender_hp=incoming_hp,
            move_pp=8,
        ),
    )


def _context(
    *,
    incoming_slot: int = 1,
    incoming_hp: int = 400,
    incoming_max_hp: int = 500,
    opponent_item: str = "",
    boots: bool = False,
    grounded: bool = True,
    rock_type_mod: int = 0,
) -> SwitchHazardContext:
    return SwitchHazardContext(
        switch=_switch(
            incoming_slot=incoming_slot,
            incoming_hp=incoming_hp,
            incoming_max_hp=incoming_max_hp,
            opponent_item=opponent_item,
            defender_base_stat=95 if incoming_slot == 1 else 135,
            attack_type_mod=-1 if incoming_slot == 1 else 0,
        ),
        stealth_rock=True,
        spikes_layers=1,
        incoming_has_heavy_duty_boots=boots,
        incoming_grounded=grounded,
        stealth_rock_type_mod=rock_type_mod,
    )


def test_showdown_hazard_damage_rounding() -> None:
    assert stealth_rock_damage(500, 0) == 62
    assert stealth_rock_damage(401, 2) == 200
    assert stealth_rock_damage(401, -1) == 25
    assert stealth_rock_damage(1, -6) == 1

    assert spikes_damage(500, 1) == 62
    assert spikes_damage(500, 2) == 83
    assert spikes_damage(500, 3) == 125


def test_boots_bypass_both_hazards() -> None:
    hazards = resolve_entry_hazards(_context(boots=True))

    assert hazards.incoming_hp_before == 400
    assert hazards.incoming_hp_after == 400
    assert hazards.stealth_rock_damage == 0
    assert hazards.spikes_damage == 0


def test_airborne_target_ignores_spikes_but_not_stealth_rock() -> None:
    hazards = resolve_entry_hazards(
        _context(
            incoming_slot=2,
            incoming_hp=400,
            incoming_max_hp=400,
            grounded=False,
            rock_type_mod=2,
        )
    )

    assert hazards.stealth_rock_damage == 200
    assert hazards.spikes_damage == 0
    assert hazards.incoming_hp_after == 200


def test_hazard_faint_suppresses_queued_opponent_attack() -> None:
    context = _context(incoming_hp=50, incoming_max_hp=500)

    low = switch_hazard_turn(
        context,
        accuracy_roll=0,
        damage_roll=0,
    )
    hostile = switch_hazard_turn(
        _context(
            incoming_hp=50,
            incoming_max_hp=500,
            opponent_item=ITEM_CHOICE_SPECS,
        ),
        accuracy_roll=99,
        damage_roll=15,
    )

    assert low.hazard_fainted is True
    assert low.attack_executed is True
    assert low.active_hp == 0
    assert low.hazard_damage == 50
    assert low.opponent_move_pp == 7
    assert low.opponent_hp == 341
    assert low.successor_key == hostile.successor_key


def test_hazard_faint_dependency_key_drops_opponent_attack_item_and_rng() -> None:
    plain = _context(incoming_hp=50, incoming_max_hp=500)
    specs = _context(
        incoming_hp=50,
        incoming_max_hp=500,
        opponent_item=ITEM_CHOICE_SPECS,
    )

    plain_key = switch_hazard_dependency_key(
        plain,
        accuracy_roll=0,
        damage_roll=0,
    )
    specs_key = switch_hazard_dependency_key(
        specs,
        accuracy_roll=99,
        damage_roll=15,
    )

    assert plain_key == specs_key


def test_survivor_still_uses_post_hazard_attack_dependencies() -> None:
    plain = _context(opponent_item="")
    specs = _context(opponent_item=ITEM_CHOICE_SPECS)

    miss_plain = switch_hazard_dependency_key(
        plain,
        accuracy_roll=99,
        damage_roll=0,
    )
    miss_specs = switch_hazard_dependency_key(
        specs,
        accuracy_roll=99,
        damage_roll=15,
    )
    hit_plain = switch_hazard_dependency_key(
        plain,
        accuracy_roll=0,
        damage_roll=0,
    )
    hit_specs = switch_hazard_dependency_key(
        specs,
        accuracy_roll=0,
        damage_roll=15,
    )

    assert miss_plain == miss_specs
    assert hit_plain != hit_specs

    outcome = switch_hazard_turn(
        plain,
        accuracy_roll=99,
        damage_roll=15,
    )
    assert outcome.hazard_damage == 124
    assert outcome.active_hp == 276
    assert outcome.attack_executed is True
    assert outcome.hit is False
    assert outcome.opponent_move_pp == 7


def test_projection_collapses_external_bench_and_hazard_ko_attack_factors() -> None:
    contexts = (
        _context(opponent_item=""),
        _context(opponent_item=ITEM_CHOICE_SPECS),
        _context(boots=True, opponent_item=""),
        _context(boots=True, opponent_item=ITEM_CHOICE_SPECS),
        _context(
            incoming_slot=2,
            incoming_hp=400,
            incoming_max_hp=400,
            grounded=False,
            rock_type_mod=2,
            opponent_item="",
        ),
        _context(
            incoming_slot=2,
            incoming_hp=400,
            incoming_max_hp=400,
            grounded=False,
            rock_type_mod=2,
            opponent_item=ITEM_CHOICE_SPECS,
        ),
        _context(
            incoming_hp=50,
            incoming_max_hp=500,
            opponent_item="",
        ),
        _context(
            incoming_hp=50,
            incoming_max_hp=500,
            opponent_item=ITEM_CHOICE_SPECS,
        ),
    )
    worlds = [
        SwitchHazardWorld(
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

    projection = compile_switch_hazard_projection(contexts, worlds)
    direct = [
        switch_hazard_turn(
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

    hazard_ko_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if contexts[world.context_index].switch.opponent_attack.defender_hp == 50
    }
    survivor_miss_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if contexts[world.context_index].switch.opponent_attack.defender_hp != 50
        and world.accuracy_roll == 99
    }
    survivor_hit_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if contexts[world.context_index].switch.opponent_attack.defender_hp != 50
        and world.accuracy_roll == 0
    }

    assert len(worlds) == 4096
    assert projection.class_count == 100
    assert len(hazard_ko_ids) == 1
    assert len(survivor_miss_ids) == 3
    assert len(survivor_hit_ids) == 96
    assert direct == expanded


def test_invalid_spikes_layers_fail_closed() -> None:
    try:
        SwitchHazardContext(
            switch=_switch(
                incoming_slot=1,
                incoming_hp=400,
                incoming_max_hp=500,
                opponent_item="",
            ),
            stealth_rock=True,
            spikes_layers=4,
            incoming_has_heavy_duty_boots=False,
            incoming_grounded=True,
            stealth_rock_type_mod=0,
        )
    except SwitchHazardError:
        return
    raise AssertionError("invalid Spikes layers were accepted")
