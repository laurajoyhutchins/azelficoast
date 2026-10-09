"""Bounded, non-certifying smoke of the frozen posterior population source.

This intentionally cannot create a selected cohort or a scientific certificate.
The ordinary hosted preparation/selection remains the only admission path.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT
from azelficoast.live.corpus import (
    CORPUS_SCHEMA,
    CORPUS_SCHEMA_VERSION,
    DecisionFixture,
    _fixture_id,
)
from azelficoast.research.hosted.common import REPOSITORY_ROOT
from azelficoast.research.posterior_population_selection import _source_binding
from azelficoast.research.studies import natural_disagreements


CONTRACT_PATH = (
    REPOSITORY_ROOT / "experiments" / "data"
    / "posterior-stratified-population-contract.json"
)


SMOKE_PROFILES = {
    "quick": (128, 128, 2, 64, 0),
    "full-seed": (1024, 2048, 16, 512, 0),
    "mid-seed-1": (1024, 2048, 16, 512, 16),
    "mid-seed-2": (1024, 2048, 16, 512, 32),
}


class SmokeError(RuntimeError):
    """A diagnostic boundary was not exercised or its authority is invalid."""


def smoke_limits(
    fixtures: int, rounds: int, sample_keys: int, screen_rounds: int,
    key_offset: int = 0,
) -> None:
    for label, value, maximum in (
        ("fixtures", fixtures, 4096),
        ("rounds", rounds, 2048),
        ("sample_keys", sample_keys, 32),
        ("screen_rounds", screen_rounds, 512),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise SmokeError(f"{label} must be within 1..{maximum}")
    if isinstance(key_offset, bool) or not isinstance(key_offset, int) or not 0 <= key_offset <= 4096:
        raise SmokeError("key_offset must be within 0..4096")


def _load_object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise SmokeError(f"{path} must contain a JSON object")
    return result


def _source_files(root: Path, contract: Mapping[str, Any]) -> tuple[Path, Path]:
    source = contract["population"]["source_artifact"]
    if not isinstance(source, Mapping):
        raise SmokeError("source artifact contract must be an object")
    corpus = root / "corpus.jsonl"
    metadata = root / "source-metadata.json"
    if not (corpus.exists() and metadata.exists()):
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            (
                "gh", "run", "download",
                str(source["workflow_run_id"]), "--repo",
                "laurajoyhutchins/azelficoast",
                "--name", str(source["artifact_name"]), "--dir", str(root),
            ),
            check=True,
            timeout=120,
        )
    if not corpus.is_file() or not metadata.is_file():
        raise SmokeError("canonical frozen source download is incomplete")
    return corpus, metadata



def _load_fixture_window(
    corpus: Path, *, limit: int, expected_total: int
) -> tuple[list[DecisionFixture], int]:
    """Read only the smoke window from the already SHA-verified frozen corpus.

    _source_binding validates the SHA of the *entire* original corpus first.
    The remaining original records are intentionally not deserialized.
    """
    window: list[DecisionFixture] = []
    seen: set[str] = set()
    with corpus.open(encoding="utf-8") as stream:
        manifest = json.loads(stream.readline())
        if not isinstance(manifest, dict) or (
            manifest.get("schema") != CORPUS_SCHEMA
            or manifest.get("schema_version") != CORPUS_SCHEMA_VERSION
            or manifest.get("kind") != "manifest"
        ):
            raise SmokeError("frozen source corpus has an invalid manifest")
        total = manifest.get("fixture_count")
        if isinstance(total, bool) or not isinstance(total, int) or total != expected_total:
            raise SmokeError("frozen source fixture count differs from source binding")
        for line_number in range(2, min(limit, total) + 2):
            line = stream.readline()
            if not line:
                raise SmokeError("source corpus ended before bounded smoke window")
            record = json.loads(line)
            if not isinstance(record, dict) or (
                record.get("schema") != CORPUS_SCHEMA
                or record.get("schema_version") != CORPUS_SCHEMA_VERSION
                or record.get("kind") != "fixture"
            ):
                raise SmokeError(f"invalid source fixture at line {line_number}")
            fixture_id = record.get("fixture_id")
            state = record.get("state")
            prefix = record.get("protocol_prefix")
            controls = record.get("control_decisions")
            if (
                not isinstance(fixture_id, str)
                or not isinstance(state, dict)
                or not isinstance(prefix, list)
                or not isinstance(controls, list)
                or fixture_id != _fixture_id(state, prefix)
                or fixture_id in seen
            ):
                raise SmokeError(f"invalid fixture identity at line {line_number}")
            seen.add(fixture_id)
            window.append(
                DecisionFixture(
                    fixture_id=fixture_id,
                    state=state,
                    protocol_prefix=tuple(
                        tuple(tuple(str(value) for value in message) for message in batch)
                        for batch in prefix
                    ),
                    control_decisions=tuple(controls),
                )
            )
    return window, total


def _bounded_mine(
    fixtures: Sequence[DecisionFixture],
    showdown_root: Path,
    *,
    rounds: int,
    max_keys: int,
    key_offset: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    real_sampler = natural_disagreements._sample_worlds
    observations: list[dict[str, Any]] = []
    cached: dict[tuple[object, ...], dict[str, Any] | Exception] = {}
    prior_keys: set[tuple[object, ...]] = set()
    truncated = 0

    def bounded_sampler(**kwargs: Any) -> dict[str, Any]:
        nonlocal truncated
        key = (
            kwargs["species"],
            tuple(kwargs["observed_moves"]),
            kwargs["is_lead"],
            kwargs["public_level"],
            kwargs["public_ability"],
        )
        if key in prior_keys:
            raise natural_disagreements.UnsupportedWorldSample("smoke-only earlier key")
        if len(prior_keys) < key_offset:
            prior_keys.add(key)
            raise natural_disagreements.UnsupportedWorldSample("smoke-only earlier key")
        previous = cached.get(key)
        if isinstance(previous, Exception):
            raise natural_disagreements.UnsupportedWorldSample(str(previous))
        if isinstance(previous, dict):
            return previous
        if len(cached) >= max_keys:
            truncated += 1
            raise natural_disagreements.UnsupportedWorldSample(
                "smoke-only unique sampler key cap"
            )

        begun = time.monotonic()
        try:
            sample = real_sampler(**kwargs)
            if sample.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
                raise SmokeError("sampler did not use the pinned Showdown revision")
        except natural_disagreements.UnsupportedWorldSample as error:
            cached[key] = error
            observations.append(
                {
                    "species": str(kwargs["species"]),
                    "status": "unsupported",
                    "duration_seconds": round(time.monotonic() - begun, 3),
                    "reason": str(error)[:240],
                }
            )
            raise
        else:
            cached[key] = sample
            observations.append(
                {
                    "species": str(kwargs["species"]),
                    "status": "sampled",
                    "matched": sample.get("matched"),
                    "duration_seconds": round(time.monotonic() - begun, 3),
                }
            )
            return sample

    try:
        natural_disagreements._sample_worlds = bounded_sampler
        candidates = natural_disagreements.mine_candidates(
            fixtures, showdown_root=showdown_root, rounds=rounds
        )
    finally:
        natural_disagreements._sample_worlds = real_sampler
    return candidates, observations, truncated



def _verified_mechanics_summary(
    document: object, *, candidate_count: int, rounds: int, showdown_commit: str
) -> dict[str, int]:
    if not isinstance(document, Mapping) or (
        document.get("schema") != "azelficoast.public-belief-speed-fork-mechanics"
        or document.get("schema_version") != 1
        or document.get("showdown_commit") != showdown_commit
        or document.get("rounds") != rounds
    ):
        raise SmokeError("mechanics smoke response drifted from pinned contract")
    count = document.get("case_count")
    strict = document.get("strict_execution_fork_count")
    cases = document.get("cases")
    if (
        isinstance(count, bool) or not isinstance(count, int)
        or isinstance(strict, bool) or not isinstance(strict, int)
        or count != candidate_count or not isinstance(cases, list)
        or len(cases) != count or strict < 0 or strict > count
    ):
        raise SmokeError("mechanics case counts do not match bounded candidate inputs")
    return {
        "case_count": count,
        "strict_execution_fork_count": strict,
    }


def smoke(
    *,
    artifact_root: Path,
    showdown_root: Path,
    output: Path,
    fixture_limit: int,
    rounds: int,
    sample_keys: int,
    screen_rounds: int,
    key_offset: int,
) -> dict[str, Any]:
    smoke_limits(fixture_limit, rounds, sample_keys, screen_rounds, key_offset)
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    contract = _load_object(CONTRACT_PATH)
    corpus, source_metadata = _source_files(artifact_root, contract)
    binding = _source_binding(contract, _load_object(source_metadata), corpus)
    verify_source_seconds = time.monotonic() - start

    pinned = subprocess.run(
        ("git", "-C", str(showdown_root), "rev-parse", "HEAD"),
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if pinned != PINNED_SHOWDOWN_COMMIT:
        raise SmokeError("Showdown checkout differs from pinned mechanics authority")

    started = time.monotonic()
    chosen, total_count = _load_fixture_window(
        corpus,
        limit=fixture_limit,
        expected_total=int(binding["state_count"]),
    )
    if not chosen:
        raise SmokeError("source corpus has no fixtures")
    load_seconds = time.monotonic() - started

    started = time.monotonic()
    candidates, sampler_observations, truncated = _bounded_mine(
        chosen, showdown_root, rounds=rounds, max_keys=sample_keys,
        key_offset=key_offset
    )
    mining_seconds = time.monotonic() - started
    if not sampler_observations:
        raise SmokeError(
            "smoke window did not reach any genuine generator sampling; "
            "increase fixture_limit, not scientific admission thresholds"
        )
    (output / "candidates-smoke.json").write_text(
        json.dumps(candidates, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    real_candidates = candidates.get("candidates")
    if not isinstance(real_candidates, list):
        raise SmokeError("candidate miner returned no candidate list")
    mechanics_seconds: float | None = None
    mechanics_status = "not-run-no-candidates"
    mechanics_summary: dict[str, int] | None = None
    if real_candidates:
        candidate_ids = {candidate["fixture_id"] for candidate in real_candidates}
        smoke_fixtures = output / "candidate-fixtures-smoke.jsonl"
        with smoke_fixtures.open("w", encoding="utf-8") as handle:
            for fixture in chosen:
                if fixture.fixture_id in candidate_ids:
                    handle.write(json.dumps(fixture.as_record(), sort_keys=True) + "\n")
        started = time.monotonic()
        screen = subprocess.run(
            (
                "node",
                str(REPOSITORY_ROOT / "showdown" / "research" / "belief"
                    / "screen_public_belief_speed_forks.cjs"),
                str(showdown_root), str(output / "candidates-smoke.json"),
                str(smoke_fixtures), str(screen_rounds),
            ),
            check=True, capture_output=True, text=True, timeout=120,
            cwd=REPOSITORY_ROOT,
        )
        mechanics = json.loads(screen.stdout)
        mechanics_summary = _verified_mechanics_summary(
            mechanics,
            candidate_count=len(real_candidates),
            rounds=screen_rounds,
            showdown_commit=pinned,
        )
        (output / "mechanics-smoke.json").write_text(
            json.dumps(mechanics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mechanics_seconds = time.monotonic() - started
        mechanics_status = "executed"

    return {
        "schema": "azelficoast.posterior-population-smoke/v1",
        "purpose": "performance-diagnostic-only",
        "certified": False,
        "scientific_population_admitted": False,
        "execution_plan_issued": False,
        "complete_study_result": None,
        "source": {
            "workflow_run_id": binding["workflow_run_id"],
            "corpus_digest": binding["digest"],
            "generation_head_sha": binding["generation_head_sha"],
            "showdown_commit": pinned,
        },
        "scope": {
            "fixture_limit": fixture_limit,
            "fixture_count_checked": len(chosen),
            "full_source_fixture_count": total_count,
            "generator_rounds": rounds,
            "sampler_key_cap": sample_keys,
            "sampler_key_offset": key_offset,
            "mechanics_rounds": screen_rounds,
        },
        "observations": {
            "sampler_keys_attempted": len(sampler_observations),
            "sampling": sampler_observations,
            "sampler_keys_not_attempted_under_smoke_cap": truncated,
            "candidate_count_under_smoke_bounds": candidates["candidate_count"],
            "mechanics_status": mechanics_status,
            "mechanics_summary": mechanics_summary,
        },
        "timing_seconds": {
            "download_and_source_binding": round(verify_source_seconds, 3),
            "corpus_load": round(load_seconds, 3),
            "candidate_mining": round(mining_seconds, 3),
            "mechanics_screen": (
                None if mechanics_seconds is None else round(mechanics_seconds, 3)
            ),
            "total": round(time.monotonic() - start, 3),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("/tmp/posterior-smoke-source"))
    parser.add_argument("--showdown-root", type=Path, default=Path("/tmp/pokemon-showdown"))
    parser.add_argument("--output", type=Path, default=Path("/tmp/posterior-smoke"))
    parser.add_argument("--profile", choices=tuple(SMOKE_PROFILES), default="quick")
    args = parser.parse_args(argv)
    fixture_limit, rounds, sample_keys, screen_rounds, key_offset = SMOKE_PROFILES[args.profile]
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        result = smoke(
            artifact_root=args.artifact_root,
            showdown_root=args.showdown_root,
            output=args.output,
            fixture_limit=fixture_limit,
            rounds=rounds,
            sample_keys=sample_keys,
            screen_rounds=screen_rounds,
            key_offset=key_offset,
        )
    except Exception as error:
        result = {
            "schema": "azelficoast.posterior-population-smoke/v1",
            "certified": False,
            "scientific_population_admitted": False,
            "execution_plan_issued": False,
            "status": "smoke-failed",
            "error_type": type(error).__name__,
            "error": str(error)[:500],
        }
        code = 1
    else:
        result["status"] = "smoke-executed"
        code = 0
    (args.output / "smoke-receipt.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
