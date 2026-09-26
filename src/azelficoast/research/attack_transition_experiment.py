"""Showdown-backed whole-attack transition and adaptive-dispatch experiment."""

from __future__ import annotations

import ctypes
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Mapping, Sequence

import jax
import numpy as np

import azelficoast.research.mechanics.gen9_attack as gen9_attack
import azelficoast.research.mechanics.gen9_damage as gen9_damage
from azelficoast.research.adaptive_execution_experiment import (
    _evaluate,
    _execution_target_signature,
    _fit_absolute_baseline,
    _fit_cost_profile,
    _profile_record,
)
from azelficoast.research.mechanics.class_native_belief import (
    ClassNativeBelief,
    ProjectionMap,
    build_factor_support,
    compile_attack_projection,
    project_belief,
    uniform_belief,
)
from azelficoast.research.mechanics.gen9_attack import (
    COMPILED_ATTACK_CONTEXT_WIDTH,
    AttackTransitionContext,
    attack_transition,
    attack_transition_dependency_signature,
    attack_transition_numeric,
    compile_attack_context,
    unpack_attack_transition,
)
from azelficoast.research.mechanics.jax_gen9_attack import (
    attack_transition_batch,
    attack_transition_score,
    weighted_attack_transition_score,
)
from azelficoast.research.mechanics.native_damage_compiler import build_attack_library
from azelficoast.research.verification.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    ShowdownDamageCorpusError,
    _context as damage_context_from_mapping,
)

REPEATS = 7
TRAINING_WORLD_COUNTS = (
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    65536,
    131072,
    262144,
    524288,
)
# Development runs established only a crossover bracket: direct was faster at
# 6,144 worlds and class-native was faster at 8,192. These confirmation points
# were not present in either the training grid or the earlier exploratory holdout.
CONFIRMATION_WORLD_COUNTS = (
    6400,
    6656,
    6912,
    9216,
    11264,
    18432,
    40960,
    81920,
    327680,
)


class AttackTransitionExperimentError(ValueError):
    """Raised when hosted transition evidence is incomplete or inconsistent."""


def _attack_context(fixture: Mapping[str, object]) -> AttackTransitionContext:
    context = fixture.get("context")
    before = fixture.get("before")
    if not isinstance(context, Mapping) or not isinstance(before, Mapping):
        raise AttackTransitionExperimentError("fixture lacks attack context or pre-state")
    return AttackTransitionContext(
        damage=damage_context_from_mapping(context),
        accuracy=int(fixture["move_accuracy"]),
        attacker_hp=int(before["attacker_hp"]),
        attacker_max_hp=int(before["attacker_max_hp"]),
        defender_hp=int(before["defender_hp"]),
        move_pp=int(before["move_pp"]),
    )


def _expected_packed(fixture: Mapping[str, object]) -> int:
    after = fixture.get("after")
    if not isinstance(after, Mapping):
        raise AttackTransitionExperimentError("fixture lacks post-state")
    return gen9_attack.AttackTransition(
        defender_hp=int(after["defender_hp"]),
        attacker_hp=int(after["attacker_hp"]),
        move_pp=int(after["move_pp"]),
        hit=bool(fixture["hit"]),
        defender_fainted=bool(after["defender_fainted"]),
        attacker_fainted=bool(after["attacker_fainted"]),
    ).packed


def _load_document(path: Path) -> tuple[Mapping[str, object], ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ShowdownDamageCorpusError("attack fixture document must be an object")
    if document.get("schema") != "azelficoast.showdown-attack-transition-fixtures":
        raise AttackTransitionExperimentError("unexpected attack fixture schema")
    if document.get("schema_version") != 1:
        raise AttackTransitionExperimentError("unexpected attack fixture schema version")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise AttackTransitionExperimentError("attack fixtures use the wrong Showdown revision")
    raw = document.get("fixtures")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AttackTransitionExperimentError("attack fixture document lacks fixtures")
    fixtures = tuple(fixture for fixture in raw if isinstance(fixture, Mapping))
    if len(fixtures) != int(document.get("fixture_count") or 0):
        raise AttackTransitionExperimentError("attack fixture matrix is incomplete")
    return fixtures


class NativeAttack:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path))
        pointer = ctypes.POINTER(ctypes.c_int32)
        self.library.az_attack_batch.argtypes = [
            pointer,
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
        ]
        self.library.az_attack_batch.restype = None

    @staticmethod
    def _pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_int32):
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    def batch(
        self,
        params: np.ndarray,
        accuracy_rolls: np.ndarray,
        damage_rolls: np.ndarray,
    ) -> np.ndarray:
        params_i32 = np.ascontiguousarray(params, dtype=np.int32)
        accuracy_i32 = np.ascontiguousarray(accuracy_rolls, dtype=np.int32)
        damage_i32 = np.ascontiguousarray(damage_rolls, dtype=np.int32)
        if params_i32.ndim != 2 or params_i32.shape[1] != COMPILED_ATTACK_CONTEXT_WIDTH:
            raise ValueError("unexpected whole-attack context shape")
        if accuracy_i32.shape != damage_i32.shape or accuracy_i32.shape != (len(params_i32),):
            raise ValueError("attack RNG vectors do not match contexts")
        out = np.empty(len(params_i32), dtype=np.int32)
        self.library.az_attack_batch(
            self._pointer(params_i32),
            self._pointer(accuracy_i32),
            self._pointer(damage_i32),
            self._pointer(out),
            len(params_i32),
        )
        return out


def _fixture_correctness(
    fixtures: Sequence[Mapping[str, object]],
    native: NativeAttack,
) -> dict[str, object]:
    contexts = tuple(_attack_context(fixture) for fixture in fixtures)
    params = np.asarray([compile_attack_context(context) for context in contexts], dtype=np.int32)
    accuracy = np.asarray([int(fixture["accuracy_roll"]) for fixture in fixtures], dtype=np.int32)
    rolls = np.asarray([int(fixture["damage_roll"]) for fixture in fixtures], dtype=np.int32)
    expected = np.asarray([_expected_packed(fixture) for fixture in fixtures], dtype=np.int32)

    python = np.asarray(
        [
            attack_transition_numeric(
                tuple(int(value) for value in params[index]),
                int(accuracy[index]),
                int(rolls[index]),
            )
            for index in range(len(fixtures))
        ],
        dtype=np.int32,
    )
    native_actual = native.batch(params, accuracy, rolls)
    jax_actual = np.asarray(
        attack_transition_batch(
            jax.device_put(params),
            jax.device_put(accuracy),
            jax.device_put(rolls),
        ),
        dtype=np.int32,
    )

    miss_damage_rng_absent = all(
        (bool(fixture["hit"]) or 16 not in fixture["rng_requests"])
        for fixture in fixtures
    )
    hit_damage_rng_present = all(
        (not bool(fixture["hit"]) or 16 in fixture["rng_requests"])
        for fixture in fixtures
    )

    mismatches: list[dict[str, object]] = []
    for index, fixture in enumerate(fixtures):
        if int(python[index]) == int(expected[index]):
            continue
        expected_state = unpack_attack_transition(int(expected[index]))
        actual_state = unpack_attack_transition(int(python[index]))
        mismatches.append(
            {
                "item": fixture.get("item"),
                "state_variant": fixture.get("state_variant"),
                "bench_signature": fixture.get("bench_signature"),
                "accuracy_roll": int(fixture["accuracy_roll"]),
                "damage_roll": int(fixture["damage_roll"]),
                "rng_requests": list(fixture["rng_requests"]),
                "transition_log": list(fixture["transition_log"]),
                "expected_packed": int(expected[index]),
                "actual_packed": int(python[index]),
                "expected": {
                    "defender_hp": expected_state.defender_hp,
                    "attacker_hp": expected_state.attacker_hp,
                    "move_pp": expected_state.move_pp,
                    "hit": expected_state.hit,
                    "defender_fainted": expected_state.defender_fainted,
                    "attacker_fainted": expected_state.attacker_fainted,
                },
                "actual": {
                    "defender_hp": actual_state.defender_hp,
                    "attacker_hp": actual_state.attacker_hp,
                    "move_pp": actual_state.move_pp,
                    "hit": actual_state.hit,
                    "defender_fainted": actual_state.defender_fainted,
                    "attacker_fainted": actual_state.attacker_fainted,
                },
            }
        )
        if len(mismatches) >= 12:
            break

    return {
        "fixture_count": len(fixtures),
        "python_exact": bool(np.array_equal(python, expected)),
        "native_exact": bool(np.array_equal(native_actual, expected)),
        "jax_exact": bool(np.array_equal(jax_actual, expected)),
        "backends_equal": bool(
            np.array_equal(python, native_actual)
            and np.array_equal(python, jax_actual)
        ),
        "miss_damage_rng_absent": miss_damage_rng_absent,
        "hit_damage_rng_present": hit_damage_rng_present,
        "mismatch_examples": mismatches,
    }


def _benchmark_contexts(
    fixtures: Sequence[Mapping[str, object]],
) -> tuple[AttackTransitionContext, ...]:
    selected: dict[str, AttackTransitionContext] = {}
    for fixture in fixtures:
        if fixture.get("state_variant") != "normal" or int(fixture.get("bench_signature", -1)) != 0:
            continue
        item = str(fixture.get("item"))
        selected.setdefault(item, _attack_context(fixture))
    required = {"None", "Choice Specs", "Life Orb"}
    if set(selected) != required:
        raise AttackTransitionExperimentError("benchmark contexts lack item treatment matrix")
    return tuple(selected[item] for item in ("None", "Choice Specs", "Life Orb"))


def _projection_oracle(contexts: Sequence[AttackTransitionContext]) -> dict[str, object]:
    support = build_factor_support(
        len(contexts),
        bench_variants=4,
        accuracy_rolls=100,
        rolls=16,
    )
    good = compile_attack_projection(support, contexts)
    bad = compile_attack_projection(
        support,
        contexts,
        include_attack_modifier=False,
    )
    belief = uniform_belief(support, support.class_count)

    def score(projection: ProjectionMap) -> int:
        projected = project_belief(belief, projection)
        total = 0
        for class_id, representative in enumerate(projection.representative_indices):
            weight = int(projected.weights[class_id])
            if not weight:
                continue
            context = contexts[int(support.context_index[representative])]
            total += (
                attack_transition(
                    context,
                    int(support.accuracy_roll[representative]),
                    int(support.roll[representative]),
                ).packed
                * weight
            )
        return total

    direct = sum(
        attack_transition(
            contexts[int(context_index)],
            int(accuracy_roll),
            int(damage_roll),
        ).packed
        for context_index, accuracy_roll, damage_roll in zip(
            support.context_index,
            support.accuracy_roll,
            support.roll,
            strict=True,
        )
    )
    return {
        "canonical_classes": support.class_count,
        "execution_classes": good.class_count,
        "expected_execution_classes": 49,
        "effect_signature": good.effect_signature,
        "signature_matches_transition": (
            good.effect_signature == attack_transition_dependency_signature()
        ),
        "direct_equals_projected": direct == score(good),
        "missing_attack_modifier_changes_result": direct != score(bad),
        "hostile_execution_classes": bad.class_count,
    }


def _uniform_belief_for_worlds(
    support,
    world_count: int,
) -> ClassNativeBelief:
    if world_count <= 0:
        raise ValueError("logical world count must be positive")
    if world_count >= support.class_count:
        return uniform_belief(support, world_count)

    # Low-work crossover treatments intentionally contain fewer logical worlds
    # than the full canonical support. Activate a deterministic, evenly spaced
    # subset so the treatment still samples the complete support geometry rather
    # than taking a prefix concentrated in one context or RNG region.
    indices = (
        np.arange(world_count, dtype=np.int64) * support.class_count
    ) // world_count
    weights = np.zeros(support.class_count, dtype=np.int64)
    weights[indices] = 1
    return ClassNativeBelief(support=support, weights=weights)


def _materialize_direct(
    belief: ClassNativeBelief,
    contexts: Sequence[AttackTransitionContext],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = belief.support
    context_index = np.repeat(support.context_index, belief.weights)
    accuracy = np.repeat(support.accuracy_roll, belief.weights).astype(np.int32, copy=False)
    rolls = np.repeat(support.roll, belief.weights).astype(np.int32, copy=False)
    rows = np.asarray([compile_attack_context(context) for context in contexts], dtype=np.int32)
    return rows[context_index], accuracy, rolls


def _projected_inputs(
    belief: ClassNativeBelief,
    projection: ProjectionMap,
    contexts: Sequence[AttackTransitionContext],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    projected = project_belief(belief, projection)
    active = projected.weights > 0
    representatives = projection.representative_indices[active]
    rows = np.asarray([compile_attack_context(context) for context in contexts], dtype=np.int32)
    return (
        rows[belief.support.context_index[representatives]],
        belief.support.accuracy_roll[representatives].astype(np.int32, copy=False),
        belief.support.roll[representatives].astype(np.int32, copy=False),
        projected.weights[active].astype(np.int32, copy=False),
    )


def _median_and_mad(samples: Sequence[float]) -> tuple[float, float]:
    median = statistics.median(samples)
    mad = statistics.median(abs(value - median) for value in samples)
    return median, mad


def _benchmark_row(
    contexts: Sequence[AttackTransitionContext],
    support,
    projection: ProjectionMap,
    world_count: int,
) -> dict[str, object]:
    belief = _uniform_belief_for_worlds(support, world_count)
    projected = project_belief(belief, projection)

    direct_params, direct_accuracy, direct_rolls = _materialize_direct(belief, contexts)
    direct_params_device = jax.device_put(direct_params)
    direct_accuracy_device = jax.device_put(direct_accuracy)
    direct_rolls_device = jax.device_put(direct_rolls)

    def direct_call() -> int:
        value = attack_transition_score(
            direct_params_device,
            direct_accuracy_device,
            direct_rolls_device,
        )
        value.block_until_ready()
        return int(np.asarray(value))

    def projected_call() -> int:
        params, accuracy, rolls, weights = _projected_inputs(
            belief,
            projection,
            contexts,
        )
        value = weighted_attack_transition_score(
            jax.device_put(params),
            jax.device_put(accuracy),
            jax.device_put(rolls),
            jax.device_put(weights),
        )
        value.block_until_ready()
        return int(np.asarray(value))

    direct_warm = direct_call()
    projected_warm = projected_call()
    direct_samples: list[float] = []
    projected_samples: list[float] = []
    direct_score = direct_warm
    projected_score = projected_warm

    for repeat in range(REPEATS):
        calls = (
            (projected_call, projected_samples, "projected"),
            (direct_call, direct_samples, "direct"),
        )
        if repeat % 2:
            calls = tuple(reversed(calls))
        for call, samples, name in calls:
            start = time.perf_counter_ns()
            value = call()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
            if name == "direct":
                direct_score = value
            else:
                projected_score = value

    direct_ms, direct_mad = _median_and_mad(direct_samples)
    projected_ms, projected_mad = _median_and_mad(projected_samples)
    delta_samples = [
        projected_value - direct_value
        for projected_value, direct_value in zip(projected_samples, direct_samples, strict=True)
    ]
    delta_ms, delta_mad = _median_and_mad(delta_samples)

    return {
        "treatment": f"worlds-{world_count}",
        "logical_world_count": belief.logical_world_count,
        "active_canonical_classes": belief.active_canonical_classes,
        "active_projected_classes": projected.active_classes,
        "direct_median_ms": direct_ms,
        "direct_mad_ms": direct_mad,
        "projected_median_ms": projected_ms,
        "projected_mad_ms": projected_mad,
        "projected_minus_direct_median_ms": delta_ms,
        "projected_minus_direct_mad_ms": delta_mad,
        "oracle_path": "projected" if delta_ms < 0 else "direct",
        "score_equal": direct_score == projected_score == direct_warm == projected_warm,
    }


def run_experiment(fixtures_path: Path) -> dict[str, object]:
    fixtures = _load_document(fixtures_path)
    contexts = _benchmark_contexts(fixtures)
    effect_signature = attack_transition_dependency_signature()
    target_signature = _execution_target_signature()
    backend = jax.default_backend()

    with tempfile.TemporaryDirectory(prefix="azelficoast-attack-") as temp_dir:
        native_build = build_attack_library(
            Path(gen9_damage.__file__),
            Path(gen9_attack.__file__),
            Path(temp_dir) / "attack.so",
            context_width=COMPILED_ATTACK_CONTEXT_WIDTH,
        )
        correctness = _fixture_correctness(fixtures, NativeAttack(native_build.library))

    projection_evidence = _projection_oracle(contexts)
    support = build_factor_support(
        len(contexts),
        bench_variants=1,
        accuracy_rolls=100,
        rolls=16,
    )
    projection = compile_attack_projection(support, contexts)
    if projection.effect_signature != effect_signature:
        raise AttackTransitionExperimentError("projection advertised the wrong effect signature")

    training = [
        _benchmark_row(contexts, support, projection, world_count)
        for world_count in TRAINING_WORLD_COUNTS
    ]
    profile, model_selection = _fit_cost_profile(
        training,
        backend=backend,
        target_signature=target_signature,
        effect_signature=effect_signature,
    )
    baseline = _fit_absolute_baseline(training)
    held_out_rows = [
        _benchmark_row(contexts, support, projection, world_count)
        for world_count in CONFIRMATION_WORLD_COUNTS
    ]
    evaluation = _evaluate(
        held_out_rows,
        profile=profile,
        baseline=baseline,
    )
    candidate = evaluation["candidate"]

    disjoint = set(TRAINING_WORLD_COUNTS).isdisjoint(CONFIRMATION_WORLD_COUNTS)
    semantic_passed = (
        correctness["python_exact"]
        and correctness["native_exact"]
        and correctness["jax_exact"]
        and correctness["backends_equal"]
        and correctness["miss_damage_rng_absent"]
        and correctness["hit_damage_rng_present"]
        and projection_evidence["execution_classes"]
        == projection_evidence["expected_execution_classes"]
        and projection_evidence["signature_matches_transition"]
        and projection_evidence["direct_equals_projected"]
        and projection_evidence["missing_attack_modifier_changes_result"]
        and disjoint
        and all(bool(row["score_equal"]) for row in training)
        and all(bool(row["score_equal"]) for row in candidate["rows"])
    )
    performance_passed = (
        candidate["choice_accuracy"] >= 0.75
        and candidate["adaptive_over_oracle"] <= 1.10
        and candidate["worst_case_over_oracle"] <= 1.20
        and candidate["chosen_paths"] == ["direct", "projected"]
        and candidate["speedup_vs_always_direct"] > 1.0
        and candidate["speedup_vs_always_projected"] > 1.0
    )
    passed = semantic_passed and performance_passed

    return {
        "schema": "azelficoast.attack-transition-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "backend": backend,
        "jax_version": jax.__version__,
        "target_signature": target_signature,
        "effect_signature": effect_signature,
        "generated_c_bytes": len(native_build.c_source.encode("utf-8")),
        "correctness": correctness,
        "projection": projection_evidence,
        "training_and_held_out_disjoint": disjoint,
        "confirmation_selection_basis": {
            "development_crossover_bracket_worlds": [6144, 8192],
            "confirmation_counts_previously_unmeasured": True,
            "purpose": "require a fresh grid capable of falsifying both dispatcher paths",
        },
        "training": training,
        "profile": _profile_record(profile),
        "model_selection": model_selection,
        "held_out": evaluation,
        "semantic_passed": semantic_passed,
        "performance_passed": performance_passed,
        "passed": passed,
        "non_claims": [
            "the transition is one ordinary single-target attack, not a complete turn scheduler",
            "secondaries, multihit, protection, immunities, drain, contact hooks, self-stat drops, and exceptional move logic remain outside this transition",
            "the critical-hit branch is disabled by the Shell Armor Showdown fixture and is not modeled",
            "the first exploratory holdout sat entirely above the observed crossover; this confirmation grid was frozen afterward and uses new world counts",
            "hosted timing is execution-target-specific CPU evidence",
        ],
    }


if __name__ == "__main__":
    raise SystemExit(main())
