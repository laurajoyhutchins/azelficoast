"""Regression coverage for schema-owned research orchestration contracts."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

import pytest
from jsonschema import Draft202012Validator

from azelficoast.research.experiment_contracts import (
    CONTRACT_PATH as CANDIDATE_PATH,
    CandidateExperimentError,
    load_experiments,
)
from azelficoast.research.hosted.contracts import (
    CONTRACT_PATH as HOSTED_PATH,
    HostedResearchContractError,
    load_studies,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "experiments/contracts/schemas"


def _altered_contract(
    tmp_path: Path,
    source: Path,
    mutate: Callable[[dict[str, Any]], None],
) -> Path:
    data = copy.deepcopy(json.loads(source.read_text(encoding="utf-8")))
    mutate(data)
    output = tmp_path / source.name
    output.write_text(json.dumps(data), encoding="utf-8")
    return output


@pytest.mark.parametrize("name", ["candidate", "hosted"])
def test_committed_contract_schemas_and_documents(name: str) -> None:
    schema = json.loads((SCHEMAS / f"{name}.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    document_path = CANDIDATE_PATH if name == "candidate" else HOSTED_PATH
    document = json.loads(document_path.read_text(encoding="utf-8"))
    Draft202012Validator(schema).validate(document)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("showdown", "true"),
        ("paths", [""]),
        ("artifact_name", ""),
        ("modules", [{"module": "example", "output": "x"}]),
        ("checks", [{"output": "x", "path": [True], "operator": "eq", "expected": True}]),
        ("extra", "unreviewed"),
    ],
)
def test_candidate_schema_rejects_invalid_shapes(
    tmp_path: Path, field: str, bad: Any,
) -> None:
    def mutate(data: dict[str, Any]) -> None:
        experiment = next(iter(data["experiments"].values()))
        experiment[field] = bad

    path = _altered_contract(tmp_path, CANDIDATE_PATH, mutate)
    with pytest.raises(CandidateExperimentError, match="JSON Schema"):
        load_experiments(path)


def test_candidate_schema_rejects_wrong_operator_operand(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        checks = next(iter(data["experiments"].values()))["checks"]
        checks[1]["operator"] = "ge"
        checks[1]["expected"] = True

    with pytest.raises(CandidateExperimentError, match="JSON Schema"):
        load_experiments(_altered_contract(tmp_path, CANDIDATE_PATH, mutate))


def test_candidate_semantic_uniqueness_survives_schema_migration(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        values = list(data["experiments"].values())
        values[1]["artifact_name"] = values[0]["artifact_name"]

    with pytest.raises(CandidateExperimentError, match="candidate artifact names must be unique"):
        load_experiments(_altered_contract(tmp_path, CANDIDATE_PATH, mutate))


def test_candidate_nonfinite_json_rejected(tmp_path: Path) -> None:
    path = tmp_path / "candidate.json"
    content = CANDIDATE_PATH.read_text(encoding="utf-8")
    path.write_text(content.replace('"simulator": true', '"simulator": NaN', 1), encoding="utf-8")
    with pytest.raises(CandidateExperimentError, match="nonfinite"):
        load_experiments(path)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("evidence", 1),
        ("paths", [False]),
        ("run", {"kind": "command"}),
        ("extra", "unsafe"),
    ],
)
def test_hosted_schema_rejects_invalid_studies(
    tmp_path: Path, field: str, bad: Any,
) -> None:
    def mutate(data: dict[str, Any]) -> None:
        study = next(iter(data["studies"].values()))
        study[field] = bad

    with pytest.raises(HostedResearchContractError, match="JSON Schema"):
        load_studies(_altered_contract(tmp_path, HOSTED_PATH, mutate))


def test_hosted_schema_rejects_unknown_artifact_disposition(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        study = next(iter(data["studies"].values()))
        study["run"]["artifact"]["if_no_files"] = "ignore"

    with pytest.raises(HostedResearchContractError, match="JSON Schema"):
        load_studies(_altered_contract(tmp_path, HOSTED_PATH, mutate))


def test_hosted_schema_rejects_empty_units(tmp_path: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        study = next(iter(data["studies"].values()))
        study["run"]["units"] = []

    with pytest.raises(HostedResearchContractError, match="JSON Schema"):
        load_studies(_altered_contract(tmp_path, HOSTED_PATH, mutate))


def test_schema_change_selects_contract_consumers() -> None:
    from azelficoast.research.experiment_contracts import EXPERIMENTS, experiments_for_paths
    from azelficoast.research.hosted.contracts import STUDIES, studies_for_paths

    assert experiments_for_paths(("experiments/contracts/schemas/candidate.schema.json",)) == EXPERIMENTS
    assert studies_for_paths(("experiments/contracts/schemas/hosted.schema.json",)) == STUDIES
    assert experiments_for_paths(("src/azelficoast/research/json_schema_contracts.py",)) == EXPERIMENTS
    assert studies_for_paths(("src/azelficoast/research/json_schema_contracts.py",)) == STUDIES
