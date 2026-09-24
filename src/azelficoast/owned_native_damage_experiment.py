"""Compare Azelficoast's owned C99 lowering with POST and JAX controls."""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import azelficoast.gen9_damage as gen9_damage
from azelficoast.gen9_damage import (
    COMPILED_CONTEXT_WIDTH,
    DamageContext,
    damage,
    damage_numeric,
)
from azelficoast.jax_gen9_damage import damage_batch as jax_damage_batch
from azelficoast.jax_gen9_damage import damage_score, weighted_damage_score
from azelficoast.native_damage_compiler import build_damage_library
from azelficoast.post_gen9_damage_experiment import (
    BATCH_SIZES,
    _load_contexts,
    _load_native_kernel,
    _weights_for_worlds,
    _world_arrays,
)
from azelficoast.showdown_damage_corpus import PINNED_SHOWDOWN_COMMIT, analyze_file

REPEATS = 9


def _median_ms(call: Callable[[], int]) -> tuple[float, int]:
    samples: list[float] = []
    result = 0
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples), result


class OwnedNativeDamage:
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
    owned: OwnedNativeDamage,
    post_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
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
    owned_actual = owned.batch(params, rolls)
    post_actual = np.asarray(post_kernel(params, rolls), dtype=np.int32)
    jax_actual = np.asarray(
        jax_damage_batch(jnp.asarray(params), jnp.asarray(rolls)),
        dtype=np.int32,
    )
    return {
        "scenario_count": len(contexts),
        "roll_case_count": class_count,
        "interpreted_numeric_exact": bool(np.array_equal(interpreted_numeric, expected)),
        "owned_native_exact": bool(np.array_equal(owned_actual, expected)),
        "post_exact": bool(np.array_equal(post_actual, expected)),
        "jax_exact": bool(np.array_equal(jax_actual, expected)),
        "all_equal": bool(
            np.array_equal(owned_actual, post_actual)
            and np.array_equal(owned_actual, jax_actual)
            and np.array_equal(owned_actual, interpreted_numeric)
        ),
    }


def _benchmark_batch(
    contexts: Sequence[DamageContext],
    world_count: int,
    owned: OwnedNativeDamage,
    post_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, object]:
    params, rolls = _world_arrays(contexts, world_count)

    def owned_batch_call() -> int:
        return int(np.sum(owned.batch(params, rolls), dtype=np.int64))

    def owned_score_call() -> int:
        return owned.score(params, rolls)

    def post_call() -> int:
        return int(np.sum(np.asarray(post_kernel(params, rolls)), dtype=np.int64))

    owned_batch_ms, owned_batch_score = _median_ms(owned_batch_call)
    owned_score_ms, owned_score = _median_ms(owned_score_call)
    post_ms, post_score = _median_ms(post_call)

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
        "owned_batch_median_ms": owned_batch_ms,
        "owned_score_median_ms": owned_score_ms,
        "post_batch_median_ms": post_ms,
        "jax_score_median_ms": jax_ms,
        "owned_batch_vs_post_ratio": owned_batch_ms / post_ms,
        "owned_score_vs_jax_ratio": owned_score_ms / jax_ms,
        "score_equal": (
            owned_batch_score == owned_score == post_score == jax_score
        ),
    }


def _class_native_benchmark(
    contexts: Sequence[DamageContext],
    owned: OwnedNativeDamage,
    post_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, object]:
    logical_world_count = 524288
    class_count = len(contexts) * 16
    params, rolls = _world_arrays(contexts, class_count)
    weights = _weights_for_worlds(class_count, logical_world_count)

    def owned_call() -> int:
        return owned.weighted_score(params, rolls, weights)

    def post_call() -> int:
        values = np.asarray(post_kernel(params, rolls), dtype=np.int32)
        return int(np.sum(values.astype(np.int64) * weights.astype(np.int64)))

    owned_ms, owned_score = _median_ms(owned_call)
    post_ms, post_score = _median_ms(post_call)

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
        "owned_weighted_median_ms": owned_ms,
        "post_weighted_median_ms": post_ms,
        "jax_weighted_median_ms": jax_ms,
        "owned_vs_post_ratio": owned_ms / post_ms,
        "owned_vs_jax_ratio": owned_ms / jax_ms,
        "score_equal": owned_score == post_score == jax_score,
    }


def run_experiment(fixtures: Path, post_kernel_path: Path) -> dict[str, object]:
    showdown_analysis = analyze_file(fixtures)
    contexts = _load_contexts(fixtures)
    source_path = Path(gen9_damage.__file__)

    with tempfile.TemporaryDirectory(prefix="azelficoast-owned-native-") as temp_dir:
        temp = Path(temp_dir)

        own_start = time.perf_counter_ns()
        owned_build = build_damage_library(
            source_path,
            temp / "owned_damage.so",
            context_width=COMPILED_CONTEXT_WIDTH,
        )
        owned_build_ms = (time.perf_counter_ns() - own_start) / 1_000_000
        owned = OwnedNativeDamage(owned_build.library)

        post_module, post_build_ms = _load_native_kernel(post_kernel_path, temp)
        post_kernel = post_module.damage_batch

        correctness = _correctness(contexts, owned, post_kernel)
        batches = [
            _benchmark_batch(contexts, world_count, owned, post_kernel)
            for world_count in BATCH_SIZES
        ]
        class_native = _class_native_benchmark(contexts, owned, post_kernel)

    passed = (
        showdown_analysis["passed"] is True
        and correctness["interpreted_numeric_exact"] is True
        and correctness["owned_native_exact"] is True
        and correctness["post_exact"] is True
        and correctness["jax_exact"] is True
        and correctness["all_equal"] is True
        and all(row["score_equal"] for row in batches)
        and class_native["score_equal"] is True
    )
    return {
        "schema": "azelficoast.owned-native-damage-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "owned_build_ms": owned_build_ms,
        "post_build_ms": post_build_ms,
        "generated_c_bytes": len(owned_build.c_source.encode("utf-8")),
        "correctness": correctness,
        "batch_benchmarks": batches,
        "class_native_benchmark": class_native,
        "showdown_analysis": showdown_analysis,
        "passed": passed,
        "non_claims": [
            "the owned compiler intentionally accepts only the current numeric damage subset",
            "POST remains a comparison control in this treatment",
            "hosted timings are CPU evidence and do not compare accelerator backends",
            "performance is observational and is not an acceptance threshold",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument(
        "--post-kernel",
        type=Path,
        default=Path("experiments/post_gen9_damage_kernel.py"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_experiment(args.fixtures, args.post_kernel)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
