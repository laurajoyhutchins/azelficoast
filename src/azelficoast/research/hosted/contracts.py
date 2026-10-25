from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from azelficoast.research.json_schema_contracts import ContractShapeError, load_contract_document


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPOSITORY_ROOT / "experiments" / "contracts" / "hosted.json"

_SHARED_PYTHON_PATHS = {
    ".github/actions/setup-python-environment/action.yml",
    ".python-version",
    ".github/workflows/research.yml",
    "experiments/contracts/hosted.json",
    "experiments/contracts/schemas/hosted.schema.json",
    "src/azelficoast/research/json_schema_contracts.py",
    "src/azelficoast/research/hosted/**",
    "pyproject.toml",
    "uv.lock",
}
_SHARED_SHOWDOWN_PATHS = {
    ".github/actions/setup-showdown/action.yml",
    "showdown/revision.json",
}
_SHARED_EVIDENCE_PATHS = {
    ".github/actions/setup-research-evidence/action.yml",
    "src/azelficoast/research/evidence.py",
    "experiments/evidence/canonical-evidence.json",
    "experiments/evidence/canonical-evidence.tar.gz",
}


class HostedResearchContractError(RuntimeError):
    """Raised when hosted research scheduling metadata is malformed."""


@dataclass(frozen=True, slots=True)
class Artifact:
    name: str
    paths: tuple[str, ...]
    if_no_files: str


@dataclass(frozen=True, slots=True)
class PrepareContract:
    operation: str
    artifact: Artifact


@dataclass(frozen=True, slots=True)
class RunContract:
    kind: str
    operation: str
    units: tuple[str, ...]
    artifact: Artifact


@dataclass(frozen=True, slots=True)
class AggregateContract:
    operation: str
    download_pattern: str
    artifact: Artifact
    prepare: str | None = None
    evidence: bool = False
    showdown: bool = False


@dataclass(frozen=True, slots=True)
class HostedStudy:
    name: str
    paths: tuple[str, ...]
    showdown: bool
    evidence: bool
    run: RunContract
    prepare: PrepareContract | None = None
    aggregate: AggregateContract | None = None


def _artifact(record: dict[str, Any]) -> Artifact:
    return Artifact(
        name=record["name"],
        paths=tuple(record["paths"]),
        if_no_files=record["if_no_files"],
    )


def _prepare_contract(record: dict[str, Any]) -> PrepareContract:
    return PrepareContract(
        operation=record["operation"], artifact=_artifact(record["artifact"]),
    )


def _run_contract(record: dict[str, Any]) -> RunContract:
    return RunContract(
        kind=record["kind"], operation=record["operation"],
        units=tuple(record["units"]), artifact=_artifact(record["artifact"]),
    )


def _aggregate_contract(record: dict[str, Any]) -> AggregateContract:
    return AggregateContract(
        operation=record["operation"],
        download_pattern=record["download_pattern"],
        artifact=_artifact(record["artifact"]),
        prepare=record.get("prepare"),
        evidence=record.get("evidence", False),
        showdown=record.get("showdown", False),
    )


def load_studies(path: Path = CONTRACT_PATH) -> tuple[HostedStudy, ...]:
    try:
        document = load_contract_document(
            path, REPOSITORY_ROOT / "experiments/contracts/schemas/hosted.schema.json",
        )
    except ContractShapeError as error:
        raise HostedResearchContractError(str(error)) from error
    studies: list[HostedStudy] = []
    for name, record in document["studies"].items():
        prepare = record.get("prepare")
        aggregate = record.get("aggregate")
        studies.append(
            HostedStudy(
                name=name,
                paths=tuple(record["paths"]),
                showdown=record["showdown"],
                evidence=record["evidence"],
                run=_run_contract(record["run"]),
                prepare=None if prepare is None else _prepare_contract(prepare),
                aggregate=None if aggregate is None else _aggregate_contract(aggregate),
            )
        )
    return tuple(studies)

STUDIES = load_studies()
STUDIES_BY_NAME = {study.name: study for study in STUDIES}


def studies_for_paths(paths: Iterable[str]) -> tuple[HostedStudy, ...]:
    changed = set(paths)
    selected: list[HostedStudy] = []
    for study in STUDIES:
        patterns = set(study.paths) | _SHARED_PYTHON_PATHS
        if study.showdown:
            patterns |= _SHARED_SHOWDOWN_PATHS
        if study.evidence:
            patterns |= _SHARED_EVIDENCE_PATHS
        if any(
            fnmatch.fnmatch(path, pattern)
            for path in changed
            for pattern in patterns
        ):
            selected.append(study)
    return tuple(selected)


def _artifact_name(template: str, unit: str) -> str:
    return template.replace("{unit}", unit)


def prepare_matrix(studies: Sequence[HostedStudy]) -> dict[str, object]:
    include: list[dict[str, object]] = []
    for study in studies:
        prepare = study.prepare
        if prepare is None:
            continue
        include.append(
            {
                "study": study.name,
                "showdown": study.showdown,
                "evidence": study.evidence,
                "artifact_name": prepare.artifact.name,
                "artifact_path": "\n".join(prepare.artifact.paths),
                "if_no_files": prepare.artifact.if_no_files,
            }
        )
    return {"include": include}


def run_matrix(studies: Sequence[HostedStudy]) -> dict[str, object]:
    include: list[dict[str, object]] = []
    for study in studies:
        for unit in study.run.units:
            include.append(
                {
                    "study": study.name,
                    "unit": unit,
                    "showdown": study.showdown,
                    "evidence": study.evidence,
                    "artifact_name": _artifact_name(
                        study.run.artifact.name,
                        unit,
                    ),
                    "artifact_path": "\n".join(study.run.artifact.paths),
                    "if_no_files": study.run.artifact.if_no_files,
                    "prepare_artifact_name": (
                        study.prepare.artifact.name
                        if study.prepare is not None
                        else ""
                    ),
                }
            )
    return {"include": include}


def aggregate_matrix(studies: Sequence[HostedStudy]) -> dict[str, object]:
    include: list[dict[str, object]] = []
    for study in studies:
        aggregate = study.aggregate
        if aggregate is None:
            continue
        include.append(
            {
                "study": study.name,
                "showdown": aggregate.showdown,
                "evidence": aggregate.evidence,
                "download_pattern": aggregate.download_pattern,
                "artifact_name": aggregate.artifact.name,
                "artifact_path": "\n".join(aggregate.artifact.paths),
                "if_no_files": aggregate.artifact.if_no_files,
                "prepare_artifact_name": (
                    study.prepare.artifact.name
                    if study.prepare is not None
                    else ""
                ),
            }
        )
    return {"include": include}


def changed_paths(base: str, head: str) -> tuple[str, ...]:
    if not base or not head:
        raise HostedResearchContractError(
            "base and head SHAs are required for hosted research planning"
        )
    process = subprocess.run(
        ("git", "diff", "--name-only", base, head),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return tuple(line for line in process.stdout.splitlines() if line)


def select_studies(
    *,
    base: str = "",
    head: str = "",
    requested: str = "",
) -> tuple[HostedStudy, ...]:
    if requested:
        if requested == "all":
            return STUDIES
        study = STUDIES_BY_NAME.get(requested)
        if study is None:
            raise HostedResearchContractError(
                f"unknown hosted research study {requested!r}"
            )
        return (study,)
    return studies_for_paths(changed_paths(base, head))
