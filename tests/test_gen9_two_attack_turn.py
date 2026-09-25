from __future__ import annotations

import ctypes
from dataclasses import replace
import shutil
from pathlib import Path

import pytest

import azelficoast.gen9_damage as gen9_damage
import azelficoast.gen9_two_attack_turn as gen9_two_attack_turn
from azelficoast.gen9_attack import AttackTransitionContext
from azelficoast.gen9_damage import DamageContext, ITEM_CHOICE_SPECS
from azelficoast.gen9_two_attack_turn import (
    P1_ACTION_PROTECT,
    TwoAttackTurnContext,
    compile_two_attack_turn_context,
    two_attack_turn,
    two_attack_turn_dependency_key,
    two_attack_turn_dependency_signature,
)
from azelficoast.two_attack_turn_belief import (
    build_two_attack_turn_support,
    compile_two_attack_turn_projection,
)
from azelficoast.two_attack_turn_compiler import build_two_attack_turn_library


def _damage(
    *,
    move_id: str,
    base_power: int,
    item: str = "",
) -> DamageContext:
    return DamageContext(
        attacker_level=100,
        defender_level=100,
        base_power=base_power,
        category="Special",
        move_id=move_id,
        move_type="Fairy" if move_id == "moonblast" else "Fighting",
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


def _context(
    *,
    p1_hp: int = 300,
    p2_hp: int = 300,
    p1_speed: int = 300,
    p2_speed: int = 200,
    p1_priority: int = 0,
    p2_priority: int = 0,
    p2_item: str = ITEM_CHOICE_SPECS,
    p1_protect: bool = False,
) -> TwoAttackTurnContext:
    return TwoAttackTurnContext(
        p1_attack=AttackTransitionContext(
            damage=_damage(move_id="moonblast", base_power=95),
            accuracy=100,
            attacker_hp=p1_hp,
            attacker_max_hp=341,
            defender_hp=p2_hp,
            move_pp=5,
        ),
        p2_attack=AttackTransitionContext(
            damage=_damage(move_id="aurasphere", base_power=80, item=p2_item),
            accuracy=100,
            attacker_hp=p2_hp,
            attacker_max_hp=341,
            defender_hp=p1_hp,
            move_pp=5,
        ),
        p1_priority=4 if p1_protect else p1_priority,
        p2_priority=p2_priority,
        p1_speed=p1_speed,
        p2_speed=p2_speed,
        p1_spa_drop_chance=30,
        p2_spa_stage=0,
        p1_action_kind=P1_ACTION_PROTECT if p1_protect else 0,
    )


def _run(
    context: TwoAttackTurnContext,
    *,
    order: int = 0,
    p1_damage: int = 7,
    secondary: int = 30,
    p2_damage: int = 7,
):
    return two_attack_turn(
        context,
        order_tie_roll=order,
        p1_accuracy_roll=0,
        p1_damage_roll=p1_damage,
        p1_secondary_roll=secondary,
        p2_accuracy_roll=0,
        p2_damage_roll=p2_damage,
    )


def test_first_actor_can_faint_and_cancel_queued_second_action() -> None:
    p1_ko = _run(_context(p2_hp=1, p1_speed=300, p2_speed=200))
    assert p1_ko.p1_acted is True
    assert p1_ko.p2_acted is False
    assert p1_ko.p2_hp == 0
    assert p1_ko.p1_pp == 4
    assert p1_ko.p2_pp == 5

    p2_ko = _run(_context(p1_hp=1, p1_speed=100, p2_speed=200))
    assert p2_ko.p2_acted is True
    assert p2_ko.p1_acted is False
    assert p2_ko.p1_hp == 0
    assert p2_ko.p2_pp == 4
    assert p2_ko.p1_pp == 5


def test_spa_drop_changes_later_damage_but_not_already_resolved_damage() -> None:
    p1_first = _context(p1_speed=300, p2_speed=200)
    dropped_before = _run(p1_first, secondary=0)
    not_dropped_before = _run(p1_first, secondary=30)
    assert dropped_before.p2_spa_stage == -1
    assert not_dropped_before.p2_spa_stage == 0
    assert dropped_before.p1_hp > not_dropped_before.p1_hp

    p2_first = _context(p1_speed=100, p2_speed=200)
    dropped_after = _run(p2_first, secondary=0)
    not_dropped_after = _run(p2_first, secondary=30)
    assert dropped_after.p2_spa_stage == -1
    assert not_dropped_after.p2_spa_stage == 0
    assert dropped_after.p1_hp == not_dropped_after.p1_hp


def test_first_use_protect_blocks_damage_and_consumes_both_pp() -> None:
    protected = _run(
        _context(
            p1_hp=1,
            p1_speed=100,
            p2_speed=300,
            p2_item=ITEM_CHOICE_SPECS,
            p1_protect=True,
        ),
        p2_damage=15,
    )

    assert protected.p1_hp == 1
    assert protected.p2_hp == 300
    assert protected.p1_pp == 4
    assert protected.p2_pp == 4
    assert protected.p1_acted is True
    assert protected.p2_acted is True


def test_protect_dependency_key_omits_blocked_attack_item_and_damage_rng() -> None:
    plain = _context(p2_item="", p1_protect=True)
    specs = _context(p2_item=ITEM_CHOICE_SPECS, p1_protect=True)

    plain_key = two_attack_turn_dependency_key(
        plain,
        order_tie_roll=0,
        p1_accuracy_roll=0,
        p1_damage_roll=0,
        p1_secondary_roll=0,
        p2_accuracy_roll=0,
        p2_damage_roll=0,
    )
    specs_key = two_attack_turn_dependency_key(
        specs,
        order_tie_roll=1,
        p1_accuracy_roll=99,
        p1_damage_roll=15,
        p1_secondary_roll=99,
        p2_accuracy_roll=99,
        p2_damage_roll=15,
    )

    assert plain_key == specs_key

    different_stage = replace(specs, p2_spa_stage=-1)
    stage_key = two_attack_turn_dependency_key(
        different_stage,
        order_tie_roll=1,
        p1_accuracy_roll=99,
        p1_damage_roll=15,
        p1_secondary_roll=99,
        p2_accuracy_roll=99,
        p2_damage_roll=15,
    )
    assert stage_key != specs_key


def test_priority_and_final_tie_outcome_change_causal_order() -> None:
    priority = _run(
        _context(p1_priority=1, p1_speed=100, p2_speed=300),
        secondary=0,
    )
    ordinary_slow = _run(
        _context(p1_priority=0, p1_speed=100, p2_speed=300),
        secondary=0,
    )
    assert priority.p1_hp > ordinary_slow.p1_hp

    tied = _context(p1_speed=200, p2_speed=200)
    p1_first = _run(tied, order=0, secondary=0)
    p2_first = _run(tied, order=1, secondary=0)
    assert p1_first.p1_hp > p2_first.p1_hp

    key_a = two_attack_turn_dependency_key(
        tied,
        order_tie_roll=0,
        p1_accuracy_roll=0,
        p1_damage_roll=7,
        p1_secondary_roll=0,
        p2_accuracy_roll=0,
        p2_damage_roll=7,
    )
    key_b = two_attack_turn_dependency_key(
        tied,
        order_tie_roll=1,
        p1_accuracy_roll=0,
        p1_damage_roll=7,
        p1_secondary_roll=0,
        p2_accuracy_roll=0,
        p2_damage_roll=7,
    )
    assert key_a != key_b


def test_projection_requires_order_and_secondary_dependencies() -> None:
    contexts = (
        _context(p1_speed=200, p2_speed=200, p2_item=""),
        _context(p1_speed=200, p2_speed=200, p2_item=ITEM_CHOICE_SPECS),
    )
    support = build_two_attack_turn_support(
        len(contexts),
        bench_variants=2,
        order_tie_rolls=2,
        p1_damage_rolls=4,
        p1_secondary_rolls=31,
        p2_damage_rolls=4,
    )
    good = compile_two_attack_turn_projection(support, contexts)
    missing_order = compile_two_attack_turn_projection(
        support,
        contexts,
        include_order_tie=False,
    )
    missing_secondary = compile_two_attack_turn_projection(
        support,
        contexts,
        include_secondary=False,
    )

    assert good.effect_signature == two_attack_turn_dependency_signature()
    assert missing_order.effect_signature is None
    assert missing_secondary.effect_signature is None
    assert good.class_count > missing_order.class_count
    assert good.class_count > missing_secondary.class_count


@pytest.mark.skipif(shutil.which("cc") is None, reason="system C compiler unavailable")
def test_native_compiler_matches_reference(tmp_path: Path) -> None:
    library_path = tmp_path / "two_attack_turn.so"
    build_two_attack_turn_library(
        Path(gen9_damage.__file__),
        Path(gen9_two_attack_turn.__file__),
        library_path,
    )

    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_int32)
    library.az_two_attack_turn_one.argtypes = [
        pointer,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    library.az_two_attack_turn_one.restype = ctypes.c_int32

    context = _context(p1_speed=200, p2_speed=200)
    params_values = compile_two_attack_turn_context(context)
    params = (ctypes.c_int32 * len(params_values))(*params_values)
    expected = _run(
        context,
        order=1,
        p1_damage=15,
        secondary=0,
        p2_damage=0,
    ).packed

    assert library.az_two_attack_turn_one(params, 1, 0, 15, 0, 0, 0) == expected
