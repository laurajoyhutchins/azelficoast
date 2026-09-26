from __future__ import annotations

from dataclasses import replace

import pytest

from azelficoast.research.mechanics.gen9_attack import (
    ATTACK_TRANSITION_DEPENDENCY_SCHEMA_VERSION,
    AttackTransitionContext,
    attack_transition,
    attack_transition_dependency_document,
    attack_transition_dependency_key,
    attack_transition_dependency_signature,
)
from azelficoast.research.mechanics.gen9_damage import DamageContext


def _damage(item: str = "") -> DamageContext:
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
        defender_base_stat=95,
        defender_iv=31,
        defender_ev=252,
        defender_nature_percent=110,
        attacker_item=item,
        type_mod=-1,
    )


def _context(item: str = "", *, attacker_hp: int = 341, defender_hp: int = 464):
    return AttackTransitionContext(
        damage=_damage(item),
        accuracy=80,
        attacker_hp=attacker_hp,
        attacker_max_hp=341,
        defender_hp=defender_hp,
        move_pp=8,
    )


def test_attack_transition_resolves_hit_damage_pp_and_life_orb_recoil() -> None:
    result = attack_transition(_context("Life Orb"), accuracy_roll=79, damage_roll=7)

    assert result.hit is True
    assert result.defender_hp < 464
    assert result.attacker_hp == 307
    assert result.move_pp == 7
    assert result.defender_fainted is False
    assert result.attacker_fainted is False


def test_attack_transition_miss_consumes_pp_but_not_damage_or_recoil() -> None:
    low = attack_transition(_context("Life Orb"), accuracy_roll=80, damage_roll=0)
    high = attack_transition(_context("Life Orb"), accuracy_roll=99, damage_roll=15)

    assert low == high
    assert low.hit is False
    assert low.defender_hp == 464
    assert low.attacker_hp == 341
    assert low.move_pp == 7


def test_attack_transition_reports_defender_and_recoil_faints() -> None:
    defender_faint = attack_transition(
        _context("Choice Specs", defender_hp=20),
        accuracy_roll=0,
        damage_roll=15,
    )
    attacker_faint = attack_transition(
        _context("Life Orb", attacker_hp=20),
        accuracy_roll=0,
        damage_roll=15,
    )

    assert defender_faint.defender_hp == 0
    assert defender_faint.defender_fainted is True
    assert attacker_faint.attacker_hp == 0
    assert attacker_faint.attacker_fainted is True


def test_dependency_key_is_control_flow_sensitive() -> None:
    base = _context("")
    specs = replace(base, damage=replace(base.damage, attacker_item="Choice Specs"))

    miss_a = attack_transition_dependency_key(base, 80, 0)
    miss_b = attack_transition_dependency_key(specs, 99, 15)
    hit_a = attack_transition_dependency_key(base, 0, 7)
    hit_b = attack_transition_dependency_key(specs, 0, 7)

    assert miss_a == miss_b
    assert hit_a != hit_b


def test_whole_attack_advertises_one_stable_dependency_signature() -> None:
    document = attack_transition_dependency_document()
    signature = attack_transition_dependency_signature()

    assert document["schema_version"] == ATTACK_TRANSITION_DEPENDENCY_SCHEMA_VERSION
    assert document["control"]["branches"] == ["miss", "hit"]
    assert document["hit_branch_reads"] == [
        "damage_numeric.dependencies",
        "rng.damage_roll",
    ]
    assert signature.startswith("sha256:")
    assert len(signature) == len("sha256:") + 64



def test_attack_benchmark_admits_sparse_low_world_support() -> None:
    pytest.importorskip("jax")
    from azelficoast.research.experiments.attack_transition_experiment import _uniform_belief_for_worlds
    from azelficoast.research.mechanics.class_native_belief import build_factor_support

    support = build_factor_support(
        2,
        bench_variants=1,
        accuracy_rolls=10,
        rolls=4,
    )
    sparse = _uniform_belief_for_worlds(support, 20)

    assert support.class_count == 80
    assert sparse.logical_world_count == 20
    assert sparse.active_canonical_classes == 20
    active = sparse.weights.nonzero()[0]
    assert active[0] == 0
    assert active[-1] >= 72


def test_attack_benchmark_keeps_uniform_full_support_behavior() -> None:
    pytest.importorskip("jax")
    from azelficoast.research.experiments.attack_transition_experiment import _uniform_belief_for_worlds
    from azelficoast.research.mechanics.class_native_belief import build_factor_support

    support = build_factor_support(
        2,
        bench_variants=1,
        accuracy_rolls=10,
        rolls=4,
    )
    belief = _uniform_belief_for_worlds(support, 160)

    assert belief.logical_world_count == 160
    assert belief.active_canonical_classes == 80
    assert set(int(weight) for weight in belief.weights) == {2}
