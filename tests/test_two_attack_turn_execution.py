from __future__ import annotations

import numpy as np
import pytest

from azelficoast.research.adaptive_execution import (
    ExecutionCostProfile,
    ExecutionPath,
)
from azelficoast.research.mechanics.gen9_attack import AttackTransitionContext
from azelficoast.research.mechanics.gen9_damage import DamageContext, ITEM_CHOICE_SPECS
from azelficoast.research.mechanics.gen9_two_attack_turn import (
    TwoAttackTurnContext,
    two_attack_turn_dependency_signature,
    two_attack_turn_numeric,
)
from azelficoast.research.mechanics.two_attack_turn_belief import (
    build_two_attack_turn_support,
    compile_two_attack_turn_projection,
    uniform_two_attack_turn_belief,
)
from azelficoast.research.mechanics.two_attack_turn_execution import (
    TwoAttackTurnExecutionError,
    execute_two_attack_turn_belief,
)


def _damage(*, move_id: str, base_power: int, item: str = "") -> DamageContext:
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


def _context(*, item: str = "", speed: int = 200) -> TwoAttackTurnContext:
    return TwoAttackTurnContext(
        p1_attack=AttackTransitionContext(
            damage=_damage(move_id="moonblast", base_power=95),
            accuracy=100,
            attacker_hp=300,
            attacker_max_hp=341,
            defender_hp=300,
            move_pp=5,
        ),
        p2_attack=AttackTransitionContext(
            damage=_damage(move_id="aurasphere", base_power=80, item=item),
            accuracy=100,
            attacker_hp=300,
            attacker_max_hp=341,
            defender_hp=300,
            move_pp=5,
        ),
        p1_priority=0,
        p2_priority=0,
        p1_speed=speed,
        p2_speed=200,
        p1_spa_drop_chance=30,
        p2_spa_stage=0,
    )


def _batch(
    params,
    order,
    p1_accuracy,
    p1_damage,
    p1_secondary,
    p2_accuracy,
    p2_damage,
):
    return np.asarray(
        [
            two_attack_turn_numeric(
                tuple(int(value) for value in params[index]),
                int(order[index]),
                int(p1_accuracy[index]),
                int(p1_damage[index]),
                int(p1_secondary[index]),
                int(p2_accuracy[index]),
                int(p2_damage[index]),
            )
            for index in range(len(params))
        ],
        dtype=np.int64,
    )


def _fixture():
    contexts = (
        _context(item="", speed=200),
        _context(item=ITEM_CHOICE_SPECS, speed=300),
    )
    support = build_two_attack_turn_support(
        len(contexts),
        bench_variants=4,
        order_tie_rolls=2,
        p1_damage_rolls=2,
        p1_secondary_rolls=31,
        p2_damage_rolls=2,
    )
    belief = uniform_two_attack_turn_belief(
        support,
        support.class_count * 2,
    )
    projection = compile_two_attack_turn_projection(support, contexts)
    return contexts, belief, projection


def test_preexecution_projection_matches_fully_expanded_world_histogram() -> None:
    contexts, belief, projection = _fixture()

    direct = execute_two_attack_turn_belief(
        belief,
        projection,
        contexts,
        batch_executor=_batch,
        backend="cpu",
        target_signature="sha256:test",
        force_path=ExecutionPath.DIRECT,
    )
    projected = execute_two_attack_turn_belief(
        belief,
        projection,
        contexts,
        batch_executor=_batch,
        backend="cpu",
        target_signature="sha256:test",
        force_path=ExecutionPath.PROJECTED,
    )

    assert projected.histogram() == direct.histogram()
    assert projected.total_weight == direct.total_weight == belief.logical_world_count
    assert projected.certificate["effect_signature"] == (
        two_attack_turn_dependency_signature()
    )
    assert projected.certificate["transition_evaluations"] == (
        projected.certificate["active_execution_classes"]
    )
    assert projected.certificate["transition_evaluations"] < (
        direct.certificate["transition_evaluations"]
    )
    assert projected.certificate["logical_reduction_fraction"] > 0.75


def test_unsigned_hostile_projection_is_rejected_before_execution() -> None:
    contexts, belief, _ = _fixture()
    hostile = compile_two_attack_turn_projection(
        belief.support,
        contexts,
        include_secondary=False,
    )
    calls = 0

    def never(*args):
        nonlocal calls
        calls += 1
        return _batch(*args)

    with pytest.raises(
        TwoAttackTurnExecutionError,
        match="not bound",
    ):
        execute_two_attack_turn_belief(
            belief,
            hostile,
            contexts,
            batch_executor=never,
            backend="cpu",
            target_signature="sha256:test",
            force_path=ExecutionPath.PROJECTED,
        )

    assert calls == 0


def _profile(*, projected_per_class: float) -> ExecutionCostProfile:
    return ExecutionCostProfile(
        backend="cpu",
        target_signature="sha256:test",
        effect_signature=two_attack_turn_dependency_signature(),
        direct_intercept_ms=0.0,
        direct_per_world_ms=1.0,
        direct_per_world_squared_ms=0.0,
        projected_intercept_ms=0.0,
        projected_per_canonical_class_ms=0.0,
        projected_per_execution_class_ms=projected_per_class,
        uncertainty_guard_ms=0.0,
        calibrated_max_logical_world_count=100_000,
        calibrated_max_canonical_classes=10_000,
        calibrated_max_projected_classes=10_000,
    )


def test_calibrated_dispatcher_executes_the_selected_exact_path() -> None:
    contexts, belief, projection = _fixture()

    projected = execute_two_attack_turn_belief(
        belief,
        projection,
        contexts,
        batch_executor=_batch,
        backend="cpu",
        target_signature="sha256:test",
        profile=_profile(projected_per_class=0.1),
    )
    direct = execute_two_attack_turn_belief(
        belief,
        projection,
        contexts,
        batch_executor=_batch,
        backend="cpu",
        target_signature="sha256:test",
        profile=_profile(projected_per_class=1_000.0),
    )

    assert projected.path is ExecutionPath.PROJECTED
    assert direct.path is ExecutionPath.DIRECT
    assert projected.decision is not None
    assert direct.decision is not None
    assert projected.histogram() == direct.histogram()
