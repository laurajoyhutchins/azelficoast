"""Compare serial and independent parallel Showdown generator outputs byte-semantically.

This is a non-certifying performance experiment. It does not publish a population
manifest, change scientific selection, or alter the authoritative hosted miner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT
from azelficoast.research.posterior_population_selection import _source_binding
from azelficoast.research.studies import natural_disagreements
from smoke_posterior_study import (
    CONTRACT_PATH,
    SMOKE_PROFILES,
    SmokeError,
    _load_fixture_window,
    _load_object,
    _source_files,
    smoke_limits,
)


class ParallelParityError(SmokeError):
    """An exact-seed parallel sample differs from its serial reference."""


def _key(kwargs: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        kwargs["species"],
        tuple(kwargs["observed_moves"]),
        kwargs["is_lead"],
        kwargs["public_level"],
        kwargs["public_ability"],
    )


def _discover_sampler_requests(
    fixtures: Sequence[Any],
    *,
    showdown_root: Path,
    rounds: int,
    offset: int,
    count: int,
) -> list[dict[str, Any]]:
    """Discover the same public-precondition sampler inputs without generating worlds."""
    original = natural_disagreements._sample_worlds
    unique: dict[tuple[object, ...], dict[str, Any]] = {}

    def inspect(**kwargs: Any) -> dict[str, Any]:
        unique.setdefault(_key(kwargs), kwargs)
        raise natural_disagreements.UnsupportedWorldSample(
            "smoke-only discover sampler requests"
        )

    try:
        natural_disagreements._sample_worlds = inspect
        natural_disagreements.mine_candidates(
            fixtures, showdown_root=showdown_root, rounds=rounds
        )
    finally:
        natural_disagreements._sample_worlds = original
    requests = list(unique.values())[offset : offset + count]
    if len(requests) != count:
        raise ParallelParityError(
            f"expected {count} unique diagnostic keys at offset {offset}; "
            f"found {len(requests)}"
        )
    return requests


def _sample_digest(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Each worker invokes the original Showdown CLI in a separate Node process."""
    try:
        result = natural_disagreements._sample_worlds(**kwargs)
    except natural_disagreements.UnsupportedWorldSample as error:
        return {
            "status": "unsupported",
            "reason": str(error),
        }
    if result.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise ParallelParityError("world sample has unexpected Showdown commit")
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "status": "sampled",
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "matched": result.get("matched"),
    }


def verify_equal(serial: Sequence[Mapping[str, Any]], parallel: Sequence[Mapping[str, Any]]) -> None:
    if len(serial) != len(parallel):
        raise ParallelParityError("parallel sample count differs from serial sample count")
    for index, (expected, actual) in enumerate(zip(serial, parallel, strict=True)):
        if expected != actual:
            raise ParallelParityError(f"parallel sampler drift at key index {index}")


def run_parity(
    *,
    source_root: Path,
    showdown_root: Path,
    workers: int,
) -> dict[str, Any]:
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ParallelParityError("workers must be in 1..8")
    fixture_limit, rounds, sample_keys, screen_rounds, offset = SMOKE_PROFILES["mid-seed-1"]
    smoke_limits(fixture_limit, rounds, sample_keys, screen_rounds, offset)
    started = time.monotonic()
    contract = _load_object(CONTRACT_PATH)
    corpus, metadata = _source_files(source_root, contract)
    binding = _source_binding(contract, _load_object(metadata), corpus)
    actual_sha = subprocess.run(
        ("git", "-C", str(showdown_root), "rev-parse", "HEAD"),
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if actual_sha != PINNED_SHOWDOWN_COMMIT:
        raise ParallelParityError("Showdown authority drift before parity test")
    fixtures, total = _load_fixture_window(
        corpus, limit=fixture_limit, expected_total=int(binding["state_count"])
    )
    requests = _discover_sampler_requests(
        fixtures, showdown_root=showdown_root, rounds=rounds,
        offset=offset, count=sample_keys,
    )
    setup_seconds = time.monotonic() - started

    started = time.monotonic()
    serial = [_sample_digest(request) for request in requests]
    serial_seconds = time.monotonic() - started

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        parallel = list(pool.map(_sample_digest, requests))
    parallel_seconds = time.monotonic() - started

    verify_equal(serial, parallel)
    return {
        "schema": "azelficoast.posterior-generator-parity-smoke/v1",
        "certified": False,
        "scientific_population_admitted": False,
        "execution_plan_issued": False,
        "parity": True,
        "source_digest": binding["digest"],
        "generation_head_sha": binding["generation_head_sha"],
        "showdown_commit": actual_sha,
        "total_source_fixtures": total,
        "fixture_window": fixture_limit,
        "key_offset": offset,
        "sample_keys": len(requests),
        "seed_rounds": rounds,
        "parallel_workers": workers,
        "negative_samples": sum(item["status"] == "unsupported" for item in serial),
        "reference_digests": serial,
        "timing_seconds": {
            "source_and_request_discovery": round(setup_seconds, 3),
            "serial": round(serial_seconds, 3),
            "parallel": round(parallel_seconds, 3),
            "sampling_speedup": round(serial_seconds / parallel_seconds, 3),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/tmp/posterior-smoke-source"))
    parser.add_argument("--showdown-root", type=Path, default=Path("/tmp/pokemon-showdown"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/posterior-smoke/parallel-parity.json"))
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        record = run_parity(
            source_root=args.source_root,
            showdown_root=args.showdown_root,
            workers=4,
        )
        code = 0
    except Exception as error:
        record = {
            "schema": "azelficoast.posterior-generator-parity-smoke/v1",
            "certified": False,
            "scientific_population_admitted": False,
            "execution_plan_issued": False,
            "parity": False,
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }
        code = 1
    args.output.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, sort_keys=True, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
