from __future__ import annotations

from dataclasses import replace

import pytest

from azelficoast.gen9_attack import AttackTransitionContext
from azelficoast.gen9_damage import DamageContext, ITEM_CHOICE_SPECS
from azelficoast.gen9_two_attack_turn import (
    P1_ACTION_PROTECT,
    TwoAttackTurnContext,
)
from azelficoast.stateful_protect_turn import (
    PROTECT_ROLL_DENOMINATOR,
    PROTECT_STALL_COUNTERS,
    StatefulProtectError,
    StatefulProtectWorld,
    compile_stateful_protect_projection,
    next_stall_counter,
    protect_success_threshold,
    protect_succeeds,
    showdown_stall_roll,
    stateful_protect_dependency_key,
    stateful_protect_turn,
)


def _damage(*, move_id: str, base_power: int, item: str = "") -> DamageContext:
    return DamageContext(
        attacker_level=100,
        defender_level=100,
        base_power=base_power,
        category="Special",
        move_id=move_id,
        move_type="Fairy" if move_id == "protect" else "Fighting",
        attacker_types=("Psychic",),
        tera_type=None,
        attacker_base_stat=100,
        attacker_iv=31,
        attacker_ev=252,
        attacker_nature_percent=100,
        defender_base_stat=100,
        defender_iv=31,
        defender_ev=0,
        defender_nature_percent=100,
        attacker_item=item,
        type_mod=0,
    )


def _context(*, p2_item: str = "", p1_hp: int = 300) -> TwoAttackTurnContext:
    return TwoAttackTurnContext(
        p1_attack=AttackTransitionContext(
            damage=_damage(move_id="protect", base_power=0),
            accuracy=100,
            attacker_hp=p1_hp,
            attacker_max_hp=341,
            defender_hp=300,
            move_pp=5,
        ),
        p2_attack=AttackTransitionContext(
            damage=_damage(
                move_id="aurasphere",
                base_power=80,
                item=p2_item,
            ),
            accuracy=100,
            attacker_hp=300,
            attacker_max_hp=341,
            defender_hp=p1_hp,
            move_pp=5,
        ),
        p1_priority=4,
        p2_priority=0,
        p1_speed=100,
        p2_speed=300,
        p1_spa_drop_chance=0,
        p2_spa_stage=0,
        p1_action_kind=P1_ACTION_PROTECT,
    )


def test_exact_729_point_lattice_matches_showdown_counter_probabilities() -> None:
    expected_successes = {
        1: 729,
        3: 243,
        9: 81,
        27: 27,
        81: 9,
        243: 3,
        729: 1,
    }

    assert tuple(expected_successes) == PROTECT_STALL_COUNTERS
    for counter, expected in expected_successes.items():
        successes = sum(
            protect_succeeds(counter, roll)
            for roll in range(PROTECT_ROLL_DENOMINATOR)
        )
        assert protect_success_threshold(counter) == expected
        assert successes == expected
        for roll in (0, expected - 1, expected, PROTECT_ROLL_DENOMINATOR - 1):
            if 0 <= roll < PROTECT_ROLL_DENOMINATOR:
                assert protect_succeeds(counter, roll) == (
                    showdown_stall_roll(counter, roll) == 0
                )


def test_counter_advances_on_success_and_resets_on_failure() -> None:
    assert next_stall_counter(1, succeeded=True) == 3
    assert next_stall_counter(3, succeeded=True) == 9
    assert next_stall_counter(243, succeeded=True) == 729
    assert next_stall_counter(729, succeeded=True) == 729
    assert next_stall_counter(729, succeeded=False) == 1


def test_success_drops_blocked_item_and_damage_dependencies() -> None:
    plain = _context(p2_item="")
    specs = _context(p2_item=ITEM_CHOICE_SPECS)

    plain_key = stateful_protect_dependency_key(
        plain,
        stall_counter=3,
        protect_roll=0,
        p2_accuracy_roll=0,
        p2_damage_roll=0,
    )
    specs_key = stateful_protect_dependency_key(
        specs,
        stall_counter=3,
        protect_roll=200,
        p2_accuracy_roll=99,
        p2_damage_roll=15,
    )

    assert plain_key == specs_key

    plain_outcome = stateful_protect_turn(
        plain,
        stall_counter=3,
        protect_roll=0,
        p2_accuracy_roll=0,
        p2_damage_roll=0,
    )
    specs_outcome = stateful_protect_turn(
        specs,
        stall_counter=3,
        protect_roll=200,
        p2_accuracy_roll=99,
        p2_damage_roll=15,
    )
    assert plain_outcome.successor_key == specs_outcome.successor_key
    assert plain_outcome.protect_succeeded is True
    assert plain_outcome.stall_counter_after == 9
    assert plain_outcome.turn.p1_hp == 300
    assert plain_outcome.turn.p1_pp == 4
    assert plain_outcome.turn.p2_pp == 4


def test_failure_restores_opponent_item_and_damage_dependencies() -> None:
    plain = _context(p2_item="")
    specs = _context(p2_item=ITEM_CHOICE_SPECS)

    plain_key = stateful_protect_dependency_key(
        plain,
        stall_counter=3,
        protect_roll=728,
        p2_accuracy_roll=0,
        p2_damage_roll=0,
    )
    specs_key = stateful_protect_dependency_key(
        specs,
        stall_counter=3,
        protect_roll=728,
        p2_accuracy_roll=0,
        p2_damage_roll=15,
    )
    assert plain_key != specs_key

    plain_outcome = stateful_protect_turn(
        plain,
        stall_counter=3,
        protect_roll=728,
        p2_accuracy_roll=0,
        p2_damage_roll=0,
    )
    specs_outcome = stateful_protect_turn(
        specs,
        stall_counter=3,
        protect_roll=728,
        p2_accuracy_roll=0,
        p2_damage_roll=15,
    )

    assert plain_outcome.protect_succeeded is False
    assert specs_outcome.protect_succeeded is False
    assert plain_outcome.stall_counter_after == 1
    assert specs_outcome.stall_counter_after == 1
    assert plain_outcome.turn.p1_hp < 300
    assert specs_outcome.turn.p1_hp < plain_outcome.turn.p1_hp
    assert plain_outcome.turn.p1_pp == 4
    assert plain_outcome.turn.p2_pp == 4


def test_projection_collapses_success_worlds_but_not_failed_damage_worlds() -> None:
    contexts = (_context(p2_item=""), _context(p2_item=ITEM_CHOICE_SPECS))
    worlds = [
        StatefulProtectWorld(
            context_index=context_index,
            stall_counter=3,
            protect_roll=protect_roll,
            p2_accuracy_roll=0,
            p2_damage_roll=damage_roll,
        )
        for context_index in range(2)
        for protect_roll in (0, 200, 728)
        for damage_roll in (0, 15)
    ]

    projection = compile_stateful_protect_projection(contexts, worlds)
    success_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if protect_succeeds(world.stall_counter, world.protect_roll)
    }
    failure_ids = {
        int(projection.class_ids[index])
        for index, world in enumerate(worlds)
        if not protect_succeeds(world.stall_counter, world.protect_roll)
    }

    assert len(success_ids) == 1
    assert len(failure_ids) > 1
    assert success_ids.isdisjoint(failure_ids)
    assert projection.class_count == 5


def test_invalid_history_and_out_of_boundary_priority_fail_closed() -> None:
    with pytest.raises(StatefulProtectError, match="stall counter"):
        protect_succeeds(2, 0)

    with pytest.raises(StatefulProtectError, match="roll"):
        protect_succeeds(3, 729)

    hostile = replace(_context(), p2_priority=4)
    with pytest.raises(StatefulProtectError, match="opposing attack to act later"):
        stateful_protect_turn(
            hostile,
            stall_counter=3,
            protect_roll=0,
            p2_accuracy_roll=0,
            p2_damage_roll=0,
        )
