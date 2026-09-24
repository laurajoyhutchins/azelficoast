from __future__ import annotations

from dataclasses import replace

import numpy as np

from azelficoast.class_native_belief import (
    build_factor_support,
    compile_bench_projection,
    compile_damage_projection,
    compile_full_projection,
    compile_protect_projection,
    filter_exact_damage_observation,
    materialize_projection_ids,
    project_belief,
    uniform_belief,
)
from azelficoast.gen9_damage import DamageContext, damage


def _contexts() -> tuple[DamageContext, ...]:
    base = DamageContext(
        attacker_level=82,
        defender_level=88,
        base_power=80,
        category="Special",
        move_id="aurasphere",
        move_type="Fighting",
        attacker_types=("Fighting", "Steel"),
        tera_type=None,
        attacker_base_stat=115,
        attacker_iv=31,
        attacker_ev=252,
        attacker_nature_percent=110,
        defender_base_stat=95,
        defender_iv=31,
        defender_ev=252,
        defender_nature_percent=110,
        attacker_item="",
        type_mod=1,
        burned=False,
    )
    return (
        base,
        replace(base, attacker_item="Choice Specs"),
    )


def test_projection_sequence_splits_only_on_requested_dependencies() -> None:
    contexts = _contexts()
    support = build_factor_support(len(contexts), bench_variants=3, rolls=16)
    belief = uniform_belief(support, 960)

    protect = compile_protect_projection(support)
    damage_projection = compile_damage_projection(support, contexts)
    bench = compile_bench_projection(support)
    full = compile_full_projection(support)

    assert project_belief(belief, protect).active_classes == 1
    assert project_belief(belief, damage_projection).active_classes == 32
    assert project_belief(belief, bench).active_classes == 3
    assert project_belief(belief, full).active_classes == 96
    assert project_belief(belief, protect).active_classes == 1

    for projection in (protect, damage_projection, bench, full):
        assert project_belief(belief, projection).logical_world_count == 960


def test_compressed_projection_matches_raw_expansion() -> None:
    contexts = _contexts()
    support = build_factor_support(len(contexts), bench_variants=4, rolls=16)
    belief = uniform_belief(support, 4096)
    projection = compile_damage_projection(support, contexts)

    expanded = materialize_projection_ids(belief, projection)
    raw = np.bincount(expanded, minlength=projection.class_count)
    compressed = project_belief(belief, projection).weights

    assert np.array_equal(raw.astype(np.int64), compressed)


def test_exact_damage_observation_updates_weights_without_particles() -> None:
    contexts = _contexts()
    support = build_factor_support(len(contexts), bench_variants=2, rolls=16)
    belief = uniform_belief(support, 1024)
    observed = damage(contexts[0], 7)

    posterior = filter_exact_damage_observation(belief, contexts, observed)

    assert 0 < posterior.logical_world_count < belief.logical_world_count
    raw_context = np.repeat(support.context_index, belief.weights)
    raw_roll = np.repeat(support.roll, belief.weights)
    expected = sum(
        damage(contexts[int(context_index)], int(roll)) == observed
        for context_index, roll in zip(raw_context, raw_roll, strict=True)
    )
    assert posterior.logical_world_count == expected


def test_missing_attack_modifier_merges_choice_specs_worlds() -> None:
    contexts = _contexts()
    support = build_factor_support(len(contexts), bench_variants=2, rolls=16)
    good = compile_damage_projection(support, contexts)
    bad = compile_damage_projection(
        support,
        contexts,
        include_attack_modifier=False,
    )

    assert good.class_count == 32
    assert bad.class_count == 16
