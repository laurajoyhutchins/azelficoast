from __future__ import annotations

import pytest

from azelficoast.research.experiments.simulator_experiment import run_experiment
from azelficoast.research.mechanics.simulator_ir import (
    ITEM_CHOICE_SCARF,
    ITEM_CHOICE_SPECS,
    MOVE_MOONBLAST,
    SPECIAL_DAMAGE,
    DependencyViolation,
    EffectSpec,
    RandomField,
    RandomInput,
    StateField,
    World,
    collapse_worlds,
    field_mask,
    verify_dependency_corpus,
)


def _world(item: int, bench_signature: int = 0) -> World:
    return World.from_values(
        {
            StateField.OWN_HP: 180,
            StateField.OPPONENT_ITEM: item,
            StateField.OPPONENT_MOVE: MOVE_MOONBLAST,
            StateField.BENCH_SIGNATURE: bench_signature,
        }
    )


def test_dependency_experiment_collapses_irrelevant_hidden_worlds_exactly() -> None:
    result = run_experiment(bench_variants=32)

    assert result["passed"] is True
    assert result["world_count"] == 64
    assert result["protect"] == {
        "unique_transitions": 1,
        "reduction_factor": 64.0,
        "exact": True,
    }
    assert result["choice_special_damage"] == {
        "unique_transitions": 2,
        "reduction_factor": 32.0,
        "exact": True,
    }
    assert result["negative_control_detected"] is True
    assert result["random_dependency_verified"] is True
    assert result["random_negative_control_detected"] is True


def test_choice_item_is_a_required_damage_dependency() -> None:
    worlds = (
        _world(ITEM_CHOICE_SCARF),
        _world(ITEM_CHOICE_SPECS),
    )
    random = RandomInput.from_values({RandomField.DAMAGE_ROLL: 7})
    misdeclared = EffectSpec(
        name="missing-item-dependency",
        op=SPECIAL_DAMAGE.op,
        reads=field_mask(StateField.OWN_HP),
        writes=SPECIAL_DAMAGE.writes,
        random=SPECIAL_DAMAGE.random,
    )

    with pytest.raises(DependencyViolation, match="dependency is missing"):
        collapse_worlds(misdeclared, worlds, random)


def test_ir_rejects_effect_that_writes_outside_declared_mask() -> None:
    worlds = (_world(ITEM_CHOICE_SCARF),)
    random = RandomInput.from_values({RandomField.DAMAGE_ROLL: 7})
    misdeclared = EffectSpec(
        name="missing-write-declaration",
        op=SPECIAL_DAMAGE.op,
        reads=SPECIAL_DAMAGE.reads,
        writes=0,
        random=SPECIAL_DAMAGE.random,
    )

    with pytest.raises(DependencyViolation, match="undeclared fields"):
        collapse_worlds(misdeclared, worlds, random)


def test_damage_roll_is_a_required_random_dependency() -> None:
    worlds = (_world(ITEM_CHOICE_SCARF),)
    random_inputs = tuple(
        RandomInput.from_values({RandomField.DAMAGE_ROLL: roll}) for roll in range(16)
    )
    misdeclared = EffectSpec(
        name="missing-random-dependency",
        op=SPECIAL_DAMAGE.op,
        reads=SPECIAL_DAMAGE.reads,
        writes=SPECIAL_DAMAGE.writes,
        random=0,
    )

    with pytest.raises(DependencyViolation, match="dependency is missing"):
        verify_dependency_corpus(misdeclared, worlds, random_inputs)
