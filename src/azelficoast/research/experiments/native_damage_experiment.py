"""Hosted correctness and CPU benchmark for Azelficoast's owned native damage lowering."""

from __future__ import annotations

import ctypes
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import azelficoast.research.mechanics.gen9_damage as gen9_damage
from azelficoast.research.mechanics.gen9_damage import (
    COMPILED_CONTEXT_WIDTH,
    DamageContext,
    compile_numeric_context,
    damage,
    damage_numeric,
)
from azelficoast.research.mechanics.jax_gen9_damage import damage_batch as jax_damage_batch
from azelficoast.research.mechanics.jax_gen9_damage import damage_score, weighted_damage_score
from azelficoast.research.mechanics.native_damage_compiler import build_damage_library
from azelficoast.research.verification.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    ShowdownDamageCorpusError,
    analyze_file,
    contexts_from_document,
)

BATCH_SIZES = (1, 16, 192, 4096, 65536, 524288)
REPEATS = 9


def _load_contexts(path: Path) -> tuple[DamageContext, ...]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ShowdownDamageCorpusError("fixture document must be an object")
    if document.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise ShowdownDamageCorpusError(
            "fixture Showdown revision does not match the pinned oracle"
        )
    contexts = contexts_from_document(document)
    if not contexts:
        raise ShowdownDamageCorpusError("fixture document has no contexts")
    return contexts


def _world_arrays(
    contexts: Sequence[DamageContext],
    world_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    compiled = np.asarray(
        [compile_numeric_context(context) for context in contexts],
        dtype=np.int32,
    )
    class_count = len(contexts) * 16
    indices = np.arange(world_count, dtype=np.int64) % class_count
    params = compiled[(indices // 16).astype(np.intp)]
    rolls = (indices % 16).astype(np.int32)
    return params, rolls


def _weights_for_worlds(class_count: int, world_count: int) -> np.ndarray:
    quotient, remainder = divmod(world_count, class_count)
    weights = np.full(class_count, quotient, dtype=np.int32)
    weights[:remainder] += 1
    return weights


def _median_ms(call: Callable[[], int]) -> tuple[float, int]:
    samples: list[float] = []
    result = 0
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples), result


class NativeDamage:
    def __init__(self, library_path: Path) -> None:
        self.library = ctypes.CDLL(str(library_path))
        pointer = ctypes.POINTER(ctypes.c_int32)

        self.library.az_damage_batch.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
        ]
        self.library.az_damage_batch.restype = None

        self.library.az_damage_score.argtypes = [
            pointer,
            pointer,
            ctypes.c_int64,
        ]
        self.library.az_damage_score.restype = ctypes.c_int64

        self.library.az_weighted_damage_score.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_int64,
        ]
        self.library.az_weighted_damage_score.restype = ctypes.c_int64

    @staticmethod
    def _pointer(values: np.ndarray) -> ctypes.POINTER[ctypes.c_int32]:
        return values.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    @staticmethod
    def _inputs(
        params: np.ndarray,
        rolls: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        params_i32 = np.ascontiguousarray(params, dtype=np.int32)
        rolls_i32 = np.ascontiguousarray(rolls, dtype=np.int32)
        if params_i32.ndim != 2 or params_i32.shape[1] != COMPILED_CONTEXT_WIDTH:
            raise ValueError("unexpected numeric damage context shape")
        if rolls_i32.shape != (params_i32.shape[0],):
            raise ValueError("roll vector shape does not match damage contexts")
        return params_i32, rolls_i32

    def batch(self, params: np.ndarray, rolls: np.ndarray) -> np.ndarray:
        params_i32, rolls_i32 = self._inputs(params, rolls)
        out = np.empty(len(rolls_i32), dtype=np.int32)
        self.library.az_damage_batch(
            self._pointer(params_i32),
            self._pointer(rolls_i32),
            self._pointer(out),
            len(rolls_i32),
        )
        return out

    def score(self, params: np.ndarray, rolls: np.ndarray) -> int:
        params_i32, rolls_i32 = self._inputs(params, rolls)
        return int(
            self.library.az_damage_score(
                self._pointer(params_i32),
                self._pointer(rolls_i32),
                len(rolls_i32),
            )
        )

    def weighted_score(
        self,
        params: np.ndarray,
        rolls: np.ndarray,
        weights: np.ndarray,
    ) -> int:
        params_i32, rolls_i32 = self._inputs(params, rolls)
        weights_i32 = np.ascontiguousarray(weights, dtype=np.int32)
        if weights_i32.shape != rolls_i32.shape:
            raise ValueError("weight vector shape does not match damage contexts")
        return int(
            self.library.az_weighted_damage_score(
                self._pointer(params_i32),
                self._pointer(rolls_i32),
                self._pointer(weights_i32),
                len(rolls_i32),
            )
        )


def _correctness(
    contexts: Sequence[DamageContext],
    native: NativeDamage,
) -> dict[str, object]:
    class_count = len(contexts) * 16
    params, rolls = _world_arrays(contexts, class_count)
    expected = np.asarray(
        [damage(context, roll) for context in contexts for roll in range(16)],
        dtype=np.int32,
    )
    interpreted_numeric = np.asarray(
        [
            damage_numeric(tuple(int(value) for value in params[index]), int(rolls[index]))
            for index in range(class_count)
        ],
        dtype=np.int32,
    )
    native_actual = native.batch(params, rolls)
    jax_actual = np.asarray(
        jax_damage_batch(jnp.asarray(params), jnp.asarray(rolls)),
        dtype=np.int32,
    )
    return {
        "scenario_count": len(contexts),
        "roll_case_count": class_count,
        "interpreted_numeric_exact": bool(np.array_equal(interpreted_numeric, expected)),
        "native_exact": bool(np.array_equal(native_actual, expected)),
        "jax_exact": bool(np.array_equal(jax_actual, expected)),
        "all_equal": bool(
            np.array_equal(native_actual, jax_actual)
            and np.array_equal(native_actual, interpreted_numeric)
        ),
    }


def _benchmark_batch(
    contexts: Sequence[DamageContext],
    world_count: int,
    native: NativeDamage,
) -> dict[str, object]:
    params, rolls = _world_arrays(contexts, world_count)

    def native_batch_call() -> int:
        return int(np.sum(native.batch(params, rolls), dtype=np.int64))

    def native_score_call() -> int:
        return native.score(params, rolls)

    native_batch_ms, native_batch_score = _median_ms(native_batch_call)
    native_score_ms, native_score = _median_ms(native_score_call)

    params_device = jax.device_put(params)
    rolls_device = jax.device_put(rolls)
    warm = damage_score(params_device, rolls_device)
    warm.block_until_ready()

    def jax_call() -> int:
        value = damage_score(params_device, rolls_device)
        value.block_until_ready()
        return int(np.asarray(value))

    jax_ms, jax_score = _median_ms(jax_call)

    return {
        "world_count": world_count,
        "native_batch_median_ms": native_batch_ms,
        "native_score_median_ms": native_score_ms,
        "jax_score_median_ms": jax_ms,
        "native_score_vs_jax_ratio": native_score_ms / jax_ms,
        "score_equal": native_batch_score == native_score == jax_score,
    }


def _class_native_benchmark(
    contexts: Sequence[DamageContext],
    native: NativeDamage,
) -> dict[str, object]:
    logical_world_count = 524288
    class_count = len(contexts) * 16
    params, rolls = _world_arrays(contexts, class_count)
    weights = _weights_for_worlds(class_count, logical_world_count)

    def native_call() -> int:
        return native.weighted_score(params, rolls, weights)

    native_ms, native_score = _median_ms(native_call)

    params_device = jax.device_put(params)
    rolls_device = jax.device_put(rolls)
    weights_device = jax.device_put(weights)
    warm = weighted_damage_score(params_device, rolls_device, weights_device)
    warm.block_until_ready()

    def jax_call() -> int:
        value = weighted_damage_score(params_device, rolls_device, weights_device)
        value.block_until_ready()
        return int(np.asarray(value))

    jax_ms, jax_score = _median_ms(jax_call)
    return {
        "logical_world_count": logical_world_count,
        "execution_class_count": class_count,
        "native_weighted_median_ms": native_ms,
        "jax_weighted_median_ms": jax_ms,
        "native_vs_jax_ratio": native_ms / jax_ms,
        "score_equal": native_score == jax_score,
    }


def run_experiment(fixtures: Path) -> dict[str, object]:
    showdown_analysis = analyze_file(fixtures)
    contexts = _load_contexts(fixtures)
    source_path = Path(gen9_damage.__file__)

    with tempfile.TemporaryDirectory(prefix="azelficoast-native-") as temp_dir:
        start = time.perf_counter_ns()
        native_build = build_damage_library(
            source_path,
            Path(temp_dir) / "damage.so",
            context_width=COMPILED_CONTEXT_WIDTH,
        )
        build_ms = (time.perf_counter_ns() - start) / 1_000_000
        native = NativeDamage(native_build.library)

        correctness = _correctness(contexts, native)
        batches = [
            _benchmark_batch(contexts, world_count, native)
            for world_count in BATCH_SIZES
        ]
        class_native = _class_native_benchmark(contexts, native)

    passed = (
        showdown_analysis["passed"] is True
        and correctness["interpreted_numeric_exact"] is True
        and correctness["native_exact"] is True
        and correctness["jax_exact"] is True
        and correctness["all_equal"] is True
        and all(row["score_equal"] for row in batches)
        and class_native["score_equal"] is True
    )
    return {
        "schema": "azelficoast.native-damage-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "native_build_ms": build_ms,
        "generated_c_bytes": len(native_build.c_source.encode("utf-8")),
        "correctness": correctness,
        "batch_benchmarks": batches,
        "class_native_benchmark": class_native,
        "showdown_analysis": showdown_analysis,
        "passed": passed,
        "non_claims": [
            "the native compiler intentionally accepts only the current numeric damage subset",
            "hosted timings are CPU evidence and do not compare accelerator backends",
            "performance is observational and is not an acceptance threshold",
        ],
    }

