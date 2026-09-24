from __future__ import annotations

import ctypes
import shutil
from pathlib import Path

import pytest

import azelficoast.gen9_attack as gen9_attack
import azelficoast.gen9_damage as gen9_damage
import azelficoast.gen9_ordered_attack as gen9_ordered_attack
from azelficoast.gen9_attack import AttackTransitionContext
from azelficoast.gen9_damage import DamageContext
from azelficoast.gen9_ordered_attack import (
    OrderedAttackContext,
    compile_ordered_attack_context,
    ordered_attack_dependency_key,
    ordered_attack_dependency_signature,
    ordered_attack_transition,
)
from azelficoast.ordered_attack_belief import (
    build_ordered_attack_support,
    compile_ordered_attack_projection,
)
from azelficoast.ordered_attack_compiler import build_ordered_attack_library


def _damage(item: str = "") -> DamageContext:
    return DamageContext(
        attacker_level=100,
        defender_level=100,
        base_power=80,
        category="Special",
        move_id="shadowball",
        move_type="Ghost",
        attacker_types=("Psychic",),
        tera_type=None,
        attacker_base_stat=100,
        attacker_iv=31,
        attacker_ev=252,
        attacker_nature_percent=100,
        defender_base_stat=95,
        defender_iv=31,
        defender_ev=252,
        defender_nature_percent=100,
        attacker_item=item,
        type_mod=0,
    )


def _context(
    item: str = "",
    *,
    attacker_priority: int = 0,
    opponent_priority: int = 0,
    attacker_speed: int = 200,
    opponent_speed: int = 200,
    secondary_chance: int = 20,
    stage: int = 0,
) -> OrderedAttackContext:
    return OrderedAttackContext(
        attack=AttackTransitionContext(
            damage=_damage(item),
            accuracy=100,
            attacker_hp=341,
            attacker_max_hp=341,
            defender_hp=400,
            move_pp=15,
        ),
        attacker_priority=attacker_priority,
        opponent_priority=opponent_priority,
        attacker_speed=attacker_speed,
        opponent_speed=opponent_speed,
        secondary_chance=secondary_chance,
        defender_spd_stage=stage,
    )


def test_order_prefers_priority_then_speed_then_tie_roll() -> None:
    priority = ordered_attack_transition(
        _context(attacker_priority=1, attacker_speed=100, opponent_speed=400),
        order_tie_roll=1, accuracy_roll=0, damage_roll=7, secondary_roll=99,
    )
    speed = ordered_attack_transition(
        _context(attacker_speed=300, opponent_speed=200),
        order_tie_roll=1, accuracy_roll=0, damage_roll=7, secondary_roll=99,
    )
    tie_first = ordered_attack_transition(
        _context(), order_tie_roll=0, accuracy_roll=0, damage_roll=7, secondary_roll=99,
    )
    tie_second = ordered_attack_transition(
        _context(), order_tie_roll=1, accuracy_roll=0, damage_roll=7, secondary_roll=99,
    )
    assert priority.attacker_acted_first is True
    assert speed.attacker_acted_first is True
    assert tie_first.attacker_acted_first is True
    assert tie_second.attacker_acted_first is False


def test_secondary_boundary_and_dependency_falsifiers() -> None:
    context = _context()
    applied = ordered_attack_transition(
        context, order_tie_roll=0, accuracy_roll=0, damage_roll=7, secondary_roll=19,
    )
    missed = ordered_attack_transition(
        context, order_tie_roll=0, accuracy_roll=0, damage_roll=7, secondary_roll=20,
    )
    assert applied.defender_spd_stage == -1
    assert missed.defender_spd_stage == 0

    tie_a = ordered_attack_dependency_key(
        context, order_tie_roll=0, accuracy_roll=0, damage_roll=7, secondary_roll=99,
    )
    tie_b = ordered_attack_dependency_key(
        context, order_tie_roll=1, accuracy_roll=0, damage_roll=7, secondary_roll=99,
    )
    assert tie_a != tie_b
    assert ordered_attack_dependency_signature().startswith("sha256:")


def test_class_native_projection_advertises_only_complete_signature() -> None:
    contexts = (_context(""), _context("Choice Specs"), _context("Life Orb"))
    support = build_ordered_attack_support(
        len(contexts),
        bench_variants=2,
        order_tie_rolls=2,
        accuracy_rolls=1,
        damage_rolls=16,
        secondary_rolls=21,
    )
    good = compile_ordered_attack_projection(support, contexts)
    no_order = compile_ordered_attack_projection(
        support, contexts, include_order_tie_roll=False,
    )
    no_secondary = compile_ordered_attack_projection(
        support, contexts, include_secondary_roll=False,
    )
    assert good.effect_signature == ordered_attack_dependency_signature()
    assert no_order.effect_signature is None
    assert no_secondary.effect_signature is None
    assert good.class_count > no_order.class_count
    assert good.class_count > no_secondary.class_count


@pytest.mark.skipif(shutil.which("cc") is None, reason="system C compiler unavailable")
def test_native_compiler_matches_reference(tmp_path: Path) -> None:
    library_path = tmp_path / "ordered_attack.so"
    build_ordered_attack_library(
        Path(gen9_damage.__file__),
        Path(gen9_attack.__file__),
        Path(gen9_ordered_attack.__file__),
        library_path,
    )
    library = ctypes.CDLL(str(library_path))
    pointer = ctypes.POINTER(ctypes.c_int32)
    library.az_ordered_attack_one.argtypes = [
        pointer,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
    ]
    library.az_ordered_attack_one.restype = ctypes.c_int32

    context = _context("Life Orb")
    numeric = compile_ordered_attack_context(context)
    params = (ctypes.c_int32 * len(numeric))(*numeric)
    expected = ordered_attack_transition(
        context,
        order_tie_roll=1,
        accuracy_roll=0,
        damage_roll=15,
        secondary_roll=19,
    ).packed
    assert library.az_ordered_attack_one(params, 1, 0, 15, 19) == expected
