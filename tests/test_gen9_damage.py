from __future__ import annotations

from dataclasses import replace

import pytest

from azelficoast.gen9_damage import (
    DamageContext,
    DamageKernelError,
    MOD_ONE_POINT_FIVE,
    MOD_THREE_QUARTERS,
    MOD_TWO,
    apply_type_effectiveness,
    attack_modifier,
    damage,
    ordinary_stat,
    showdown_modify,
    stab_modifier,
)


def _context(**overrides: object) -> DamageContext:
    values: dict[str, object] = {
        "attacker_level": 100,
        "defender_level": 100,
        "base_power": 80,
        "category": "Special",
        "move_id": "aurasphere",
        "move_type": "Fighting",
        "attacker_types": ("Fighting", "Steel"),
        "tera_type": None,
        "attacker_base_stat": 115,
        "attacker_iv": 31,
        "attacker_ev": 252,
        "attacker_nature_percent": 110,
        "defender_base_stat": 95,
        "defender_iv": 31,
        "defender_ev": 252,
        "defender_nature_percent": 110,
        "attacker_item": "",
        "type_mod": 1,
        "burned": False,
    }
    values.update(overrides)
    return DamageContext(**values)  # type: ignore[arg-type]


def test_showdown_modifier_rounding() -> None:
    assert showdown_modify(101, 3, 2) == 151
    assert showdown_modify(100, 5324, 4096) == 130


def test_ordinary_stat_matches_bounded_gen9_formula() -> None:
    assert ordinary_stat(115, 31, 252, 100, 110) == 361
    assert ordinary_stat(95, 31, 252, 100, 110) == 317


def test_stab_handles_ordinary_and_tera_cases() -> None:
    ordinary = _context()
    assert stab_modifier(ordinary) == MOD_ONE_POINT_FIVE

    tera_from_base = replace(ordinary, tera_type="Fighting")
    assert stab_modifier(tera_from_base) == MOD_TWO

    tera_new_type = replace(
        ordinary,
        attacker_types=("Psychic",),
        tera_type="Fighting",
    )
    assert stab_modifier(tera_new_type) == MOD_ONE_POINT_FIVE


def test_type_effectiveness_uses_repeated_integer_halving() -> None:
    assert apply_type_effectiveness(101, -1) == 50
    assert apply_type_effectiveness(101, -2) == 25
    assert apply_type_effectiveness(101, 2) == 404


def test_item_and_burn_modifiers_change_damage() -> None:
    baseline = _context()
    specs = replace(baseline, attacker_item="Choice Specs")
    life_orb = replace(baseline, attacker_item="Life Orb")
    assert attack_modifier(specs) == MOD_ONE_POINT_FIVE
    assert damage(specs, 7) > damage(baseline, 7)
    assert damage(life_orb, 7) > damage(baseline, 7)

    physical = replace(
        baseline,
        category="Physical",
        move_id="closecombat",
        base_power=120,
        attacker_base_stat=110,
        defender_base_stat=80,
        burned=True,
    )
    assert damage(physical, 7) < damage(replace(physical, burned=False), 7)


def test_defender_stat_modifier_changes_damage() -> None:
    baseline = _context(type_mod=0)
    beads = replace(
        baseline,
        defender_stat_modifier=MOD_THREE_QUARTERS,
    )
    assert damage(beads, 7) > damage(baseline, 7)


def test_stellar_is_explicitly_outside_scope() -> None:
    with pytest.raises(DamageKernelError, match="Stellar"):
        stab_modifier(replace(_context(), tera_type="Stellar"))


def test_attacker_and_defender_levels_are_independent() -> None:
    baseline = _context(attacker_level=78, defender_level=88)
    stronger_defender = _context(attacker_level=78, defender_level=100)
    stronger_attacker = _context(attacker_level=100, defender_level=88)

    assert damage(stronger_defender, 7) < damage(baseline, 7)
    assert damage(stronger_attacker, 7) > damage(baseline, 7)
