"""Compare a POST Python native damage kernel with scalar Python and JAX."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import statistics
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from postpyc.build import build_file

from azelficoast.gen9_damage import DamageContext, compile_numeric_context, damage
from azelficoast.jax_gen9_damage import damage_batch as jax_damage_batch
from azelficoast.jax_gen9_damage import damage_score, weighted_damage_score
from azelficoast.showdown_damage_corpus import (
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


def _median_ms(call: Callable[[], int], repeats: int = REPEATS) -> tuple[float, int]:
    samples: list[float] = []
    result = 0
    for _ in range(repeats):
        start = time.perf_counter_ns()
        result = call()
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return statistics.median(samples), result


def _timed_once(call: Callable[[], int]) -> tuple[float, int]:
    start = time.perf_counter_ns()
    result = call()
    return (time.perf_counter_ns() - start) / 1_000_000, result


def _load_native_kernel(kernel_path: Path, build_dir: Path) -> tuple[ModuleType, float]:
    module_name = "_azelficoast_post_damage"
    suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    output = build_dir / f"{module_name}{suffix}"

    start = time.perf_counter_ns()
    build_file(
        kernel_path,
        output=output,
        ext_module=True,
        module_name=module_name,
    )
    build_ms = (time.perf_counter_ns() - start) / 1_000_000

    spec = importlib.util.spec_from_file_location(module_name, output)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to create import specification for POST extension")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, build_ms


def _correctness(
    contexts: Sequence[DamageContext],
    post_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, object]:
    class_count = len(contexts) * 16
    params, rolls = _world_arrays(contexts, class_count)
    expected = np.asarray(
        [damage(context, roll) for context in contexts for roll in range(16)],
        dtype=np.int32,
    )
    post_actual = np.asarray(post_kernel(params, rolls), dtype=np.int32)
    jax_actual = np.asarray(
        jax_damage_batch(jnp.asarray(params), jnp.asarray(rolls)),
        dtype=np.int32,
    )
    return {
        "scenario_count": len(contexts),
        "roll_case_count": class_count,
        "post_exact": bool(np.array_equal(post_actual, expected)),
        "jax_exact": bool(np.array_equal(jax_actual, expected)),
        "post_jax_exact": bool(np.array_equal(post_actual, jax_actual)),
    }


def _benchmark_batch(
    contexts: Sequence[DamageContext],
    world_count: int,
    post_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, object]:
    params, rolls = _world_arrays(contexts, world_count)

    def post_call() -> int:
        values = np.asarray(post_kernel(params, rolls), dtype=np.int32)
        return int(np.sum(values, dtype=np.int64))

    post_first_ms, post_score = _timed_once(post_call)
    post_steady_ms, post_steady_score = _median_ms(post_call)

    params_device = jax.device_put(params)
    rolls_device = jax.device_put(rolls)

    def jax_call() -> int:
        value = damage_score(params_device, rolls_device)
        value.block_until_ready()
        return int(np.asarray(value))

    jax_first_ms, jax_score = _timed_once(jax_call)
    jax_steady_ms, jax_steady_score = _median_ms(jax_call)

    python_ms: float | None = None
    python_score: int | None = None
    if world_count <= 4096:
        class_count = len(contexts) * 16

        def python_call() -> int:
            return sum(
                damage(contexts[(index % class_count) // 16], index % 16)
                for index in range(world_count)
            )

        python_ms, python_score = _median_ms(python_call)

    return {
        "world_count": world_count,
        "post_first_ms": post_first_ms,
        "post_steady_median_ms": post_steady_ms,
        "jax_first_ms": jax_first_ms,
        "jax_steady_median_ms": jax_steady_ms,
        "python_scalar_median_ms": python_ms,
        "post_vs_jax_steady_ratio": post_steady_ms / jax_steady_ms,
        "score_equal": (
            post_score == post_steady_score == jax_score == jax_steady_score
            and (python_score is None or python_score == post_score)
        ),
    }


def _benchmark_class_native(
    contexts: Sequence[DamageContext],
    logical_world_count: int,
    post_kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> dict[str, object]:
    class_count = len(contexts) * 16
    params, rolls = _world_arrays(contexts, class_count)
    weights = _weights_for_worlds(class_count, logical_world_count)

    def post_call() -> int:
        values = np.asarray(post_kernel(params, rolls), dtype=np.int32)
        return int(np.sum(values.astype(np.int64) * weights.astype(np.int64)))

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
        "post_steady_median_ms": post_ms,
        "jax_steady_median_ms": jax_ms,
        "post_vs_jax_steady_ratio": post_ms / jax_ms,
        "score_equal": post_score == jax_score,
    }


def run_experiment(fixtures: Path, kernel: Path) -> dict[str, object]:
    showdown_analysis = analyze_file(fixtures)
    contexts = _load_contexts(fixtures)
    with tempfile.TemporaryDirectory(prefix="azelficoast-post-") as temp_dir:
        module, build_ms = _load_native_kernel(kernel, Path(temp_dir))
        post_kernel = module.damage_batch
        correctness = _correctness(contexts, post_kernel)
        batches = [
            _benchmark_batch(contexts, world_count, post_kernel)
            for world_count in BATCH_SIZES
        ]
        class_native = _benchmark_class_native(contexts, 524288, post_kernel)

    passed = (
        showdown_analysis["passed"] is True
        and correctness["post_exact"] is True
        and correctness["jax_exact"] is True
        and correctness["post_jax_exact"] is True
        and all(row["score_equal"] for row in batches)
        and class_native["score_equal"] is True
    )
    return {
        "schema": "azelficoast.post-gen9-damage-experiment",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "postpyc_version": importlib.metadata.version("postpyc"),
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "post_build_ms": build_ms,
        "showdown_analysis": showdown_analysis,
        "correctness": correctness,
        "batch_benchmarks": batches,
        "class_native_benchmark": class_native,
        "passed": passed,
        "non_claims": [
            "POST Python 0.3.0 is alpha research machinery, not a production dependency",
            "the benchmark covers only the existing bounded Gen 9 damage subset",
            "hosted evidence is CPU evidence and does not compare accelerator backends",
            "steady-state timings exclude POST build time and JAX first-call compilation",
            "performance observations do not select a backend without reproduced evidence",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument(
        "--kernel",
        type=Path,
        default=Path("experiments/post_gen9_damage_kernel.py"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_experiment(args.fixtures, args.kernel)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
