"""Bind a newly generated source corpus without changing issue #69 gates."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import argparse
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT
from azelficoast.live.corpus import load_corpus
from azelficoast.research.posterior_population_contract import validate_contract


class PopulationAcquisitionError(ValueError):
    """Raised when acquisition provenance or its receipt is incomplete."""


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
SOURCE_BATTLE_BUDGET = 3072


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PopulationAcquisitionError(f"{field} must be a positive integer")
    return value


def _hex_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PopulationAcquisitionError(f"{field} must be 64 lowercase hex characters")
    return value


def _head_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA1.fullmatch(value) is None:
        raise PopulationAcquisitionError(f"{field} must be a 40-character commit SHA")
    return value


def bind_source_artifact(
    contract: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    corpus_path: str | Path,
    decisions_path: str | Path,
) -> dict[str, Any]:
    """Return a validated temporary contract bound to one exact source artifact."""
    checked = validate_contract(contract)
    if source_metadata.get("schema") != "azelficoast.posterior-population-source/v1":
        raise PopulationAcquisitionError("unexpected source metadata schema")

    run_id = _positive_int(source_metadata.get("workflow_run_id"), field="workflow_run_id")
    if _positive_int(artifact.get("workflow_run_id"), field="artifact workflow_run_id") != run_id:
        raise PopulationAcquisitionError("source metadata and artifact run IDs differ")
    artifact_id = _positive_int(artifact.get("id"), field="artifact ID")
    artifact_name = artifact.get("name")
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        raise PopulationAcquisitionError("artifact name is missing")
    if "matches" in artifact and artifact.get("matches") != 1:
        raise PopulationAcquisitionError("artifact identity is ambiguous")

    generation_head = _head_sha(
        source_metadata.get("generation_head_sha"), field="generation_head_sha"
    )
    metadata_head = _head_sha(source_metadata.get("head_sha"), field="head_sha")
    if metadata_head != generation_head:
        raise PopulationAcquisitionError("source metadata generation head fields differ")
    workflow_head = _head_sha(
        artifact.get("workflow_head_sha"), field="artifact workflow_head_sha"
    )
    if workflow_head != generation_head:
        raise PopulationAcquisitionError("artifact workflow head differs from generation head")
    showdown_commit = source_metadata.get("showdown_commit")
    if showdown_commit != PINNED_SHOWDOWN_COMMIT:
        raise PopulationAcquisitionError("source was not generated with pinned Showdown")

    battle_count = _positive_int(source_metadata.get("battle_count"), field="battle_count")
    if battle_count != SOURCE_BATTLE_BUDGET:
        raise PopulationAcquisitionError(
            f"source battle count must equal the preregistered {SOURCE_BATTLE_BUDGET}"
        )
    state_count = _positive_int(
        source_metadata.get("decision_state_count"), field="decision_state_count"
    )
    decisions_digest = _hex_digest(
        source_metadata.get("decisions_sha256"), field="decisions_sha256"
    )
    corpus_digest = _hex_digest(source_metadata.get("corpus_sha256"), field="corpus_sha256")
    if _sha256_file(decisions_path) != decisions_digest:
        raise PopulationAcquisitionError("source decisions file digest differs from metadata")
    if _sha256_file(corpus_path) != corpus_digest:
        raise PopulationAcquisitionError("source corpus file digest differs from metadata")
    try:
        fixture_count = len(load_corpus(corpus_path))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PopulationAcquisitionError(f"source corpus is invalid: {error}") from error
    if fixture_count != state_count:
        raise PopulationAcquisitionError(
            "source corpus fixture count differs from decision_state_count"
        )
    artifact_digest = artifact.get("digest")
    if (
        not isinstance(artifact_digest, str)
        or not artifact_digest.startswith("sha256:")
        or _SHA256.fullmatch(artifact_digest[7:]) is None
    ):
        raise PopulationAcquisitionError("artifact digest must be sha256:<64 lowercase hex>")

    rebound = copy.deepcopy(checked)
    population = rebound.get("population")
    assert isinstance(population, dict)
    population["source_artifact"] = {
        "workflow_run_id": run_id,
        "artifact_id": artifact_id,
        "artifact_name": artifact_name,
        "artifact_digest": artifact_digest,
        "workflow_head_sha": workflow_head,
        "generation_head_sha": generation_head,
        "battle_count": battle_count,
        "decision_state_count": state_count,
        "decisions_sha256": decisions_digest,
        "corpus_sha256": corpus_digest,
    }
    return validate_contract(rebound)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_metadata(
    *,
    results_path: str | Path,
    decisions_path: str | Path,
    corpus_path: str | Path,
    workflow_run_id: int,
    generation_head_sha: str,
) -> dict[str, Any]:
    """Verify a completed generation and bind its immutable source digests."""
    run_id = _positive_int(workflow_run_id, field="workflow_run_id")
    head_sha = _head_sha(generation_head_sha, field="generation_head_sha")
    result_battles: set[str] = set()
    for line_number, line in enumerate(Path(results_path).read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PopulationAcquisitionError(
                f"results line {line_number} is invalid JSON"
            ) from error
        battle_tag = record.get("battle_tag") if isinstance(record, dict) else None
        if not isinstance(battle_tag, str) or not battle_tag:
            raise PopulationAcquisitionError(f"results line {line_number} lacks battle_tag")
        if battle_tag in result_battles:
            raise PopulationAcquisitionError("results contain duplicate battle tags")
        result_battles.add(battle_tag)
    if len(result_battles) != SOURCE_BATTLE_BUDGET:
        raise PopulationAcquisitionError(
            f"source generation completed {len(result_battles)} battles; "
            f"requires exactly {SOURCE_BATTLE_BUDGET}"
        )

    decision_count = 0
    seen_decisions: set[tuple[str, int]] = set()
    for line_number, line in enumerate(
        Path(decisions_path).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise PopulationAcquisitionError(
                f"decisions line {line_number} is invalid JSON"
            ) from error
        if not isinstance(record, dict):
            raise PopulationAcquisitionError(f"decisions line {line_number} is not an object")
        if record.get("kind") != "decision":
            continue
        battle_tag = record.get("battle_tag")
        decision_index = record.get("decision_index")
        if (
            not isinstance(battle_tag, str)
            or not battle_tag
            or not isinstance(decision_index, int)
            or isinstance(decision_index, bool)
            or decision_index < 0
        ):
            raise PopulationAcquisitionError(
                f"decisions line {line_number} has invalid decision identity"
            )
        identity = (battle_tag, decision_index)
        if battle_tag not in result_battles:
            raise PopulationAcquisitionError(
                f"decision line {line_number} references a battle absent from results battle tags"
            )
        if identity in seen_decisions:
            raise PopulationAcquisitionError("decisions contain duplicate decision identities")
        seen_decisions.add(identity)
        decision_count += 1
    if decision_count < 1:
        raise PopulationAcquisitionError("source generation contains no decision states")
    try:
        fixtures = load_corpus(corpus_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PopulationAcquisitionError(f"generated source corpus is invalid: {error}") from error
    fixture_count = len(fixtures)
    if fixture_count < 1:
        raise PopulationAcquisitionError("source corpus contains no decision fixtures")
    corpus_decisions: set[tuple[str, int]] = set()
    for fixture in fixtures:
        for control in fixture.control_decisions:
            battle_tag = control.get("battle_tag")
            decision_index = control.get("decision_index")
            if (
                not isinstance(battle_tag, str)
                or not battle_tag
                or not isinstance(decision_index, int)
                or isinstance(decision_index, bool)
                or decision_index < 0
            ):
                raise PopulationAcquisitionError(
                    "source corpus contains an invalid decision identity"
                )
            identity = (battle_tag, decision_index)
            if identity in corpus_decisions:
                raise PopulationAcquisitionError(
                    "source corpus contains duplicate decision identities"
                )
            corpus_decisions.add(identity)
    if corpus_decisions != seen_decisions:
        raise PopulationAcquisitionError(
            "source corpus decision identities differ from decision traces"
        )

    return {
        "schema": "azelficoast.posterior-population-source/v1",
        "workflow_run_id": run_id,
        "head_sha": head_sha,
        "generation_head_sha": head_sha,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "battle_count": SOURCE_BATTLE_BUDGET,
        "decision_state_count": fixture_count,
        "decisions_sha256": _sha256_file(decisions_path),
        "corpus_sha256": _sha256_file(corpus_path),
    }


def generate_source(
    *,
    output_root: str | Path,
    workflow_run_id: int,
    generation_head_sha: str,
    showdown_root: str | Path,
) -> dict[str, Any]:
    """Generate and freeze the preregistered local source before hosted treatments."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "acquisition-status.json"
    results_path = root / "results.jsonl"
    decisions_path = root / "decisions.jsonl"
    corpus_path = root / "corpus.jsonl"
    _write_json(
        status_path,
        {
            "status": "started",
            "workflow_run_id": workflow_run_id,
            "head_sha": generation_head_sha,
        },
    )
    process: subprocess.Popen[bytes] | None = None
    try:
        from azelficoast.research.hosted.common import start_showdown_server

        process = start_showdown_server()
        base = [
            sys.executable,
            "-m",
            "azelficoast.live.harness",
            "--results",
            str(results_path),
            "--decisions",
            str(decisions_path),
            "--replays",
            str(root / "replays"),
            "--showdown-root",
            str(showdown_root),
        ]
        subprocess.run(
            [*base, "local", "--battles", str(SOURCE_BATTLE_BUDGET), "--concurrency", "4"],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "azelficoast.live.harness",
                "corpus",
                "build",
                str(decisions_path),
                "--output",
                str(corpus_path),
            ],
            check=True,
        )
        metadata = build_source_metadata(
            results_path=results_path,
            decisions_path=decisions_path,
            corpus_path=corpus_path,
            workflow_run_id=workflow_run_id,
            generation_head_sha=generation_head_sha,
        )
        _write_json(root / "source-metadata.json", metadata)
        _write_json(
            status_path,
            {"status": "complete", "workflow_run_id": workflow_run_id, "head_sha": generation_head_sha},
        )
        return metadata
    except Exception as error:
        _write_json(
            status_path,
            {
                "status": "negative",
                "workflow_run_id": workflow_run_id,
                "head_sha": generation_head_sha,
                "failure": f"{type(error).__name__}: {error}"[-2000:],
            },
        )
        raise PopulationAcquisitionError(f"source generation failed: {error}") from error
    finally:
        if process is not None:
            process.terminate()
            process.wait(timeout=30)


def validate_source(
    *,
    contract_path: str | Path,
    source_root: str | Path,
    showdown_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Run the existing mining, pinned mechanics, and selection admission path."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    selected_dir = root / "selected"
    selected_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "receipt.json"
    contract = _load_json_object(contract_path)
    population = contract.get("population")
    if not isinstance(population, Mapping):
        raise PopulationAcquisitionError("study contract population is missing")
    source = Path(source_root)
    try:
        bound_contract = bind_source_artifact(
            contract,
            _load_json_object(source / "source-metadata.json"),
            _load_json_object(source / "artifact-binding.json"),
            corpus_path=source / "corpus.jsonl",
            decisions_path=source / "decisions.jsonl",
        )
        overlay_path = root / "contract-overlay.json"
        _write_json(overlay_path, bound_contract)

        generator_rounds = int(bound_contract["population"]["generator_rounds"])
        miner = subprocess.run(
            [
                sys.executable,
                "-m",
                "azelficoast.research.natural_disagreements",
                str(source / "corpus.jsonl"),
                "--showdown-root",
                str(showdown_root),
                "--rounds",
                str(generator_rounds),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        (root / "candidate-miner.log").write_text(
            miner.stderr[-8000:], encoding="utf-8"
        )
        if miner.returncode != 0:
            raise PopulationAcquisitionError("candidate mining failed")
        candidates_path = root / "candidates.json"
        candidates_path.write_text(miner.stdout, encoding="utf-8")
        candidates = _load_json_object(candidates_path)
        candidate_fixtures_path = root / "candidate-fixtures.jsonl"
        extract_candidate_fixtures(
            corpus_path=source / "corpus.jsonl",
            candidates_document=candidates,
            candidate_fixtures_path=candidate_fixtures_path,
            exclusions_path=root / "candidate-exclusions.json",
        )
        if not candidates.get("candidates"):
            raise PopulationAcquisitionError("candidate miner produced no candidates")

        mechanics_rounds = int(bound_contract["population"]["mechanics_screen_rounds"])
        mechanics = subprocess.run(
            [
                "node",
                "showdown/research/belief/screen_public_belief_speed_forks.cjs",
                str(showdown_root),
                str(candidates_path),
                str(candidate_fixtures_path),
                str(mechanics_rounds),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        (root / "mechanics-screen.log").write_text(
            mechanics.stderr[-8000:], encoding="utf-8"
        )
        if mechanics.returncode != 0:
            raise PopulationAcquisitionError("pinned Showdown mechanics screen failed")
        mechanics_path = root / "mechanics.json"
        mechanics_path.write_text(mechanics.stdout, encoding="utf-8")

        selection = subprocess.run(
            [
                sys.executable,
                "-m",
                "azelficoast.research.posterior_population_selection",
                str(overlay_path),
                str(source / "source-metadata.json"),
                str(candidates_path),
                str(mechanics_path),
                str(source / "corpus.jsonl"),
                "--selected-dir",
                str(selected_dir),
                "--manifest",
                str(root / "manifest.json"),
                "--execution-plan",
                str(root / "execution-plan.json"),
                "--matched-plan",
                str(root / "matched-plan.json"),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        (root / "selection.log").write_text(
            (selection.stdout + selection.stderr)[-12000:], encoding="utf-8"
        )
        receipt = build_acquisition_receipt(
            selected_fixture_ids=selected_fixture_ids(selected_dir),
            minimum_selected_states=int(bound_contract["population"]["minimum_selected_states"]),
            max_selected_states=int(bound_contract["population"]["max_selected_states"]),
            selector_exit_code=selection.returncode,
        )
        receipt["failure_reason"] = None if selection.returncode == 0 else "selector rejected population"
        _write_json(receipt_path, receipt)
        if selection.returncode != 0:
            raise PopulationAcquisitionError(
                f"population selector failed with status {selection.returncode}"
            )
        if receipt["status"] != "acquisition-positive":
            raise PopulationAcquisitionError("population selector did not admit 100 states")
        return receipt
    except Exception as error:
        if not receipt_path.is_file():
            try:
                receipt = build_acquisition_receipt(
                    selected_fixture_ids=selected_fixture_ids(selected_dir),
                    minimum_selected_states=int(population["minimum_selected_states"]),
                    max_selected_states=int(population["max_selected_states"]),
                    selector_exit_code=1,
                )
            except Exception:
                receipt = {
                    "schema": "azelficoast.posterior-population-acquisition-receipt/v1",
                    "status": "negative",
                    "selected_fixture_ids": [],
                    "selected_count": 0,
                    "required_minimum": int(population["minimum_selected_states"]),
                    "maximum_selected_states": int(population["max_selected_states"]),
                    "selector_exit_code": 1,
                    "study_positive": False,
                }
            receipt["failure_reason"] = f"{type(error).__name__}: {error}"[-2000:]
            _write_json(receipt_path, receipt)
        raise PopulationAcquisitionError(f"population validation failed: {error}") from error


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(value), sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata = subparsers.add_parser("source-metadata")
    metadata.add_argument("--results", required=True, type=Path)
    metadata.add_argument("--decisions", required=True, type=Path)
    metadata.add_argument("--corpus", required=True, type=Path)
    metadata.add_argument("--run-id", required=True, type=int)
    metadata.add_argument("--head-sha", required=True)
    metadata.add_argument("--output", required=True, type=Path)

    generate = subparsers.add_parser("generate-source")
    generate.add_argument("--output-root", required=True, type=Path)
    generate.add_argument("--run-id", required=True, type=int)
    generate.add_argument("--head-sha", required=True)
    generate.add_argument("--showdown-root", required=True, type=Path)

    validate = subparsers.add_parser("validate-source")
    validate.add_argument("--contract", required=True, type=Path)
    validate.add_argument("--source-root", required=True, type=Path)
    validate.add_argument("--showdown-root", required=True, type=Path)
    validate.add_argument("--output-root", required=True, type=Path)

    binding = subparsers.add_parser("bind-source")
    binding.add_argument("--contract", required=True, type=Path)
    binding.add_argument("--source-metadata", required=True, type=Path)
    binding.add_argument("--artifact", required=True, type=Path)
    binding.add_argument("--corpus", required=True, type=Path)
    binding.add_argument("--decisions", required=True, type=Path)
    binding.add_argument("--output", required=True, type=Path)

    receipt = subparsers.add_parser("receipt")
    receipt.add_argument("--contract", required=True, type=Path)
    receipt.add_argument("--selected-dir", required=True, type=Path)
    receipt.add_argument("--selector-exit-code", required=True, type=int)
    receipt.add_argument("--failure-reason", default="")
    receipt.add_argument("--output", required=True, type=Path)

    args = parser.parse_args(argv)
    if args.command == "source-metadata":
        try:
            value = build_source_metadata(
                results_path=args.results,
                decisions_path=args.decisions,
                corpus_path=args.corpus,
                workflow_run_id=args.run_id,
                generation_head_sha=args.head_sha,
            )
        except PopulationAcquisitionError as error:
            parser.error(str(error))
        _write_json(args.output, value)
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "generate-source":
        try:
            value = generate_source(
                output_root=args.output_root,
                workflow_run_id=args.run_id,
                generation_head_sha=args.head_sha,
                showdown_root=args.showdown_root,
            )
        except PopulationAcquisitionError as error:
            parser.error(str(error))
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "validate-source":
        try:
            value = validate_source(
                contract_path=args.contract,
                source_root=args.source_root,
                showdown_root=args.showdown_root,
                output_root=args.output_root,
            )
        except PopulationAcquisitionError as error:
            parser.error(str(error))
        print(json.dumps(value, sort_keys=True))
        return 0
    if args.command == "bind-source":
        try:
            bound = bind_source_artifact(
                _load_json_object(args.contract),
                _load_json_object(args.source_metadata),
                _load_json_object(args.artifact),
                corpus_path=args.corpus,
                decisions_path=args.decisions,
            )
        except PopulationAcquisitionError as error:
            parser.error(str(error))
        _write_json(args.output, bound)
        return 0
    if args.command == "receipt":
        contract = _load_json_object(args.contract)
        population = contract.get("population")
        if not isinstance(population, Mapping):
            parser.error("contract population is missing")
        try:
            value = build_acquisition_receipt(
                selected_fixture_ids=selected_fixture_ids(args.selected_dir),
                minimum_selected_states=int(population["minimum_selected_states"]),
                max_selected_states=int(population["max_selected_states"]),
                selector_exit_code=args.selector_exit_code,
            )
        except (KeyError, TypeError, ValueError) as error:
            parser.error(str(error))
        value["failure_reason"] = args.failure_reason or None
        _write_json(args.output, value)
        print(json.dumps(value, sort_keys=True))
        return 0
    raise AssertionError(f"unsupported acquisition command: {args.command}")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PopulationAcquisitionError(f"{path} must contain a JSON object")
    return value


def selected_fixture_ids(selected_dir: str | Path) -> list[str]:
    """Read selected state files while ignoring the admission-ledger sidecar."""
    identifiers: list[str] = []
    for path in sorted(Path(selected_dir).glob("*.json")):
        if path.name == "admission-ledger.json":
            continue
        match = re.fullmatch(r"[0-9]+-([0-9a-f]{64})\.json", path.name)
        if match is None:
            raise PopulationAcquisitionError(f"selected file has invalid name: {path.name}")
        identifiers.append(match.group(1))
    return identifiers


def extract_candidate_fixtures(
    *,
    corpus_path: str | Path,
    candidates_document: Mapping[str, Any],
    candidate_fixtures_path: str | Path,
    exclusions_path: str | Path,
) -> dict[str, Any]:
    """Extract exact candidate rows and prove candidates plus exclusions cover source."""
    candidate_rows = candidates_document.get("candidates")
    excluded_rows = candidates_document.get("excluded_fixtures")
    if not isinstance(candidate_rows, list) or not isinstance(excluded_rows, list):
        raise PopulationAcquisitionError("candidate miner omitted candidates or exclusions")

    candidate_ids = [
        row.get("fixture_id") if isinstance(row, Mapping) else None for row in candidate_rows
    ]
    excluded_ids = [
        row.get("fixture_id") if isinstance(row, Mapping) else None for row in excluded_rows
    ]
    if any(not isinstance(value, str) or not value for value in candidate_ids + excluded_ids):
        raise PopulationAcquisitionError("candidate/exclusion ledger contains an invalid fixture ID")
    if len(set(candidate_ids)) != len(candidate_ids) or len(set(excluded_ids)) != len(excluded_ids):
        raise PopulationAcquisitionError("candidate/exclusion ledger contains duplicate fixture IDs")

    source_fixtures = load_corpus(corpus_path)
    source_ids = {fixture.fixture_id for fixture in source_fixtures}
    candidate_set = {str(value) for value in candidate_ids}
    excluded_set = {str(value) for value in excluded_ids}
    if (
        candidate_set & excluded_set
        or candidate_set | excluded_set != source_ids
        or len(candidate_ids) + len(excluded_ids) != len(source_fixtures)
    ):
        raise PopulationAcquisitionError(
            "candidate/exclusion ledger does not cover source fixtures exactly"
        )

    wanted = candidate_set
    source_lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
    found: set[str] = set()
    destination = Path(candidate_fixtures_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as output:
        for line in source_lines[1:]:
            if not line:
                continue
            row = json.loads(line)
            fixture_id = row.get("fixture_id") if isinstance(row, dict) else None
            if fixture_id in wanted:
                output.write(line + "\n")
                found.add(str(fixture_id))
    if found != wanted:
        raise PopulationAcquisitionError(
            f"candidate fixture extraction is incomplete: {len(found)}/{len(wanted)}"
        )

    ledger = {
        "schema": "azelficoast.posterior-population-source-exclusions/v1",
        "source_fixture_count": len(source_fixtures),
        "candidate_fixture_ids": sorted(candidate_set),
        "excluded_fixtures": [dict(row) for row in excluded_rows],
    }
    _write_json(exclusions_path, ledger)
    return ledger


def build_acquisition_receipt(
    *,
    selected_fixture_ids: Sequence[str],
    minimum_selected_states: int,
    max_selected_states: int,
    selector_exit_code: int = 0,
) -> dict[str, Any]:
    """Classify selection only; an acquisition receipt is never a study result."""
    minimum = _positive_int(minimum_selected_states, field="minimum_selected_states")
    maximum = _positive_int(max_selected_states, field="max_selected_states")
    if not isinstance(selector_exit_code, int) or isinstance(selector_exit_code, bool):
        raise PopulationAcquisitionError("selector_exit_code must be an integer")
    if maximum < minimum:
        raise PopulationAcquisitionError("maximum selected states is below the required minimum")
    identifiers = list(selected_fixture_ids)
    if any(not isinstance(value, str) or not value for value in identifiers):
        raise PopulationAcquisitionError("selected fixture IDs must be non-empty strings")
    if len(set(identifiers)) != len(identifiers):
        raise PopulationAcquisitionError("selected fixture IDs must be unique")
    selected_count = len(identifiers)
    if selected_count > maximum:
        raise PopulationAcquisitionError("selected count exceeds contract maximum")
    return {
        "schema": "azelficoast.posterior-population-acquisition-receipt/v1",
        "status": (
            "acquisition-positive"
            if selected_count >= minimum and selector_exit_code == 0
            else "negative"
        ),
        "selected_fixture_ids": identifiers,
        "selected_count": selected_count,
        "required_minimum": minimum,
        "maximum_selected_states": maximum,
        "selector_exit_code": selector_exit_code,
        "study_positive": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
