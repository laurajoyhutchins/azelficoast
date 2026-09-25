from __future__ import annotations

import pytest

from azelficoast.research.mechanics.gen9_attack import AttackTransitionContext
from azelficoast.research.mechanics.gen9_damage import DamageContext, ITEM_CHOICE_BAND, damage
from azelficoast.research.mechanics.staged_attack import (
    staged_attack_dependency_key,
    staged_attack_transition,
    staged_stat,
)
from azelficoast.research.mechanics.switch_hazard_turn import SwitchHazardContext
from azelficoast.research.mechanics.switch_intimidate_turn import (
    SwitchIntimidateContext,
    SwitchIntimidateError,
    SwitchIntimidateWorld,
    compile_switch_intimidate_projection,
    switch_intimidate_dependency_key,
    switch_intimidate_turn,
)
from azelficoast.research.mechanics.voluntary_switch_turn import VoluntarySwitchContext


def _damage(*, item: str = "", category: str = "Physical") -> DamageContext:
    return DamageContext(
        attacker_level=100,
        defender_level=100,
        base_power=95,
        category=category,  # type: ignore[arg-type]
        move_id="highhorsepower" if category == "Physical" else "hydropump",
        move_type="Ground" if category == "Physical" else "Water",
        attacker_types=("Psychic",),
        tera_type=None,
        attacker_base_stat=100,
        attacker_iv=31,
        attacker_ev=252,
        attacker_nature_percent=110,
        defender_base_stat=80,
        defender_iv=31,
        defender_ev=252,
        defender_nature_percent=110,
        attacker_item=item,
        type_mod=1 if category == "Physical" else 0,
    )


def _switch(
    *,
    incoming_hp: int = 384,
    item: str = "",
    category: str = "Physical",
) -> VoluntarySwitchContext:
    return VoluntarySwitchContext(
        outgoing_slot=0,
        incoming_slot=1,
        outgoing_hp=200,
        outgoing_max_hp=341,
        incoming_max_hp=384,
        opponent_attack=AttackTransitionContext(
            damage=_damage(item=item, category=category),
            accuracy=95,
            attacker_hp=341,
            attacker_max_hp=341,
            defender_hp=incoming_hp,
            move_pp=8,
        ),
    )


def _context(
    *,
    incoming_hp: int = 384,
    item: str = "",
    intimidate: bool = True,
    stage: int = 0,
    boots: bool = False,
    category: str = "Physical",
) -> SwitchIntimidateContext:
    hazards = SwitchHazardContext(
        switch=_switch(
            incoming_hp=incoming_hp,
            item=item,
            category=category,
        ),
        stealth_rock=True,
        spikes_layers=1,
        incoming_has_heavy_duty_boots=boots,
        incoming_grounded=True,
        stealth_rock_type_mod=1,
    )
    return SwitchIntimidateContext(
        switch_hazards=hazards,
        incoming_has_intimidate=intimidate,
        opponent_attack_stage=stage,
    )


def test_staged_stat_matches_showdown_flooring() -> None:
    assert staged_stat(300, 1) == 450
    assert staged_stat(301, -1) == 200
    assert staged_stat(301, -2) == 150
    assert staged_stat(301, -6) == 75


def test_staged_attack_matches_neutral_kernel_and_drops_stage_on_miss() -> None:
    attack = _switch(item=ITEM_CHOICE_BAND).opponent_attack

    neutral = staged_attack_transition(attack, 0, 0, 7)
    lowered = staged_attack_transition(attack, -1, 0, 7)
    miss_neutral = staged_attack_transition(attack, 0, 99, 0)
    miss_lowered = staged_attack_transition(attack, -1, 99, 15)

    assert neutral.defender_hp == attack.defender_hp - damage(attack.damage, 7)
    assert lowered.defender_hp > neutral.defender_hp
    assert miss_neutral == miss_lowered
    assert staged_attack_dependency_key(attack, 0, 99, 0) == (
        staged_attack_dependency_key(attack, -1, 99, 15)
    )
    assert staged_attack_dependency_key(attack, 0, 0, 7) != (
        staged_attack_dependency_key(attack, -1, 0, 7)
    )


def test_surviving_intimidate_changes_public_stage_before_attack() -> None:
    no_ability = switch_intimidate_turn(
        _context(intimidate=False),
        accuracy_roll=0,
        damage_roll=7,
    )
    intimidated = switch_intimidate_turn(
        _context(intimidate=True),
        accuracy_roll=0,
        damage_roll=7,
    )

    assert no_ability.hazard_fainted is False
    assert no_ability.intimidate_activated is False
    assert no_ability.opponent_attack_stage == 0

    assert intimidated.intimidate_activated is True
    assert intimidated.intimidate_changed_stage is True
    assert intimidated.opponent_attack_stage == -1
    assert intimidated.active_hp > no_ability.active_hp


def test_intimidate_is_observable_even_when_stage_is_already_minimum() -> None:
    result = switch_intimidate_turn(
        _context(stage=-6),
        accuracy_roll=99,
        damage_roll=15,
    )

    assert result.intimidate_activated is True
    assert result.intimidate_changed_stage is False
    assert result.opponent_attack_stage == -6
    assert result.hit is False


def test_hazard_ko_skips_intimidate_and_target_dependent_attack_reads() -> None:
    plain = _context(
        incoming_hp=20,
        item="",
        intimidate=False,
        stage=0,
    )
    hidden_intimidate_band = _context(
        incoming_hp=20,
        item=ITEM_CHOICE_BAND,
        intimidate=True,
        stage=0,
    )

    first = switch_intimidate_turn(
        plain,
        accuracy_roll=0,
        damage_roll=0,
    )
    second = switch_intimidate_turn(
        hidden_intimidate_band,
        accuracy_roll=99,
        damage_roll=15,
    )

    assert first.successor_key == second.successor_key
    assert first.hazard_fainted is True
    assert first.intimidate_activated is False
    assert first.opponent_attack_stage == 0
    assert first.opponent_move_pp == 7

    assert switch_intimidate_dependency_key(
        plain,
        accuracy_roll=0,
        damage_roll=0,
    ) == switch_intimidate_dependency_key(
        hidden_intimidate_band,
        accuracy_roll=99,
        damage_roll=15,
    )


def test_boots_can_keep_intimidate_user_alive_long_enough_to_activate() -> None:
    without_boots = switch_intimidate_turn(
        _context(incoming_hp=20, intimidate=True, boots=False),
        accuracy_roll=99,
        damage_roll=15,
    )
    with_boots = switch_intimidate_turn(
        _context(incoming_hp=20, intimidate=True, boots=True),
        accuracy_roll=99,
        damage_roll=15,
    )

    assert without_boots.hazard_fainted is True
    assert without_boots.intimidate_activated is False
    assert without_boots.opponent_attack_stage == 0

    assert with_boots.hazard_fainted is False
    assert with_boots.intimidate_activated is True
    assert with_boots.opponent_attack_stage == -1


def test_surviving_miss_retains_ability_observation_and_stage_not_damage_inputs() -> None:
    plain = _context(item="", intimidate=True, stage=0)
    band = _context(item=ITEM_CHOICE_BAND, intimidate=True, stage=0)

    plain_result = switch_intimidate_turn(
        plain,
        accuracy_roll=99,
        damage_roll=0,
    )
    band_result = switch_intimidate_turn(
        band,
        accuracy_roll=99,
        damage_roll=15,
    )

    assert plain_result.successor_key == band_result.successor_key
    assert plain_result.intimidate_activated is True
    assert plain_result.opponent_attack_stage == -1
    assert switch_intimidate_dependency_key(
        plain,
        accuracy_roll=99,
        damage_roll=0,
    ) == switch_intimidate_dependency_key(
        band,
        accuracy_roll=99,
        damage_roll=15,
    )


def test_projection_has_exact_control_flow_partition() -> None:
    contexts = (
        _context(item="", intimidate=False, stage=0),
        _context(item=ITEM_CHOICE_BAND, intimidate=False, stage=0),
        _context(item="", intimidate=True, stage=0),
        _context(item=ITEM_CHOICE_BAND, intimidate=True, stage=0),
        _context(item="", intimidate=True, stage=1),
        _context(item=ITEM_CHOICE_BAND, intimidate=True, stage=1),
        _context(incoming_hp=20, item="", intimidate=False, stage=0),
        _context(
            incoming_hp=20,
            item=ITEM_CHOICE_BAND,
            intimidate=True,
            stage=0,
        ),
    )
    worlds = [
        SwitchIntimidateWorld(
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

    projection = compile_switch_intimidate_projection(contexts, worlds)
    direct = [
        switch_intimidate_turn(
            contexts[world.context_index],
            accuracy_roll=world.accuracy_roll,
            damage_roll=world.damage_roll,
        ).successor_key
        for world in worlds
    ]
    representatives = [
        direct[int(index)]
        for index in projection.representative_indices
    ]
    expanded = [
        representatives[int(class_id)]
        for class_id in projection.class_ids
    ]

    ko_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if contexts[world.context_index].switch_hazards.switch.opponent_attack.defender_hp
        == 20
    }
    miss_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if contexts[world.context_index].switch_hazards.switch.opponent_attack.defender_hp
        != 20
        and world.accuracy_roll == 99
    }
    hit_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if contexts[world.context_index].switch_hazards.switch.opponent_attack.defender_hp
        != 20
        and world.accuracy_roll == 0
    }

    assert len(worlds) == 4096
    assert projection.class_count == 100
    assert len(ko_ids) == 1
    assert len(miss_ids) == 3
    assert len(hit_ids) == 96
    assert direct == expanded


def test_special_queued_attack_fails_closed() -> None:
    with pytest.raises(SwitchIntimidateError, match="physical"):
        _context(category="Special")
