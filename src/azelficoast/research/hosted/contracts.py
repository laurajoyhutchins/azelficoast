from __future__ import annotations

import fnmatch
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPOSITORY_ROOT / "experiments" / "hosted-research-contracts.json"

_SHARED_PYTHON_PATHS = {
    ".github/actions/setup-python-environment/action.yml",
    ".github/workflows/research.yml",
    "experiments/hosted-research-contracts.json",
    "src/azelficoast/research/hosted/**",
    "pyproject.toml",
    "uv.lock",
}
_SHARED_SHOWDOWN_PATHS = {
    ".github/actions/setup-showdown/action.yml",
    "experiments/showdown-revision.txt",
}
_SHARED_EVIDENCE_PATHS = {
    ".github/actions/setup-research-evidence/action.yml",
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
    aggregate: AggregateContract | None = None


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise HostedResearchContractError(f"{label} must be an object")
    return value


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise HostedResearchContractError(f"{label} must be a non-empty string list")
    return tuple(value)


def _artifact(value: object, *, label: str) -> Artifact:
    record = _object(value, label=label)
    name = record.get("name")
    if_no_files = record.get("if_no_files")
    if not isinstance(name, str) or not name:
        raise HostedResearchContractError(f"{label}.name must be non-empty")
    if if_no_files not in {"warn", "error"}:
        raise HostedResearchContractError(
            f"{label}.if_no_files must be warn or error"
        )
    return Artifact(
        name=name,
        paths=_strings(record.get("paths"), label=f"{label}.paths"),
        if_no_files=if_no_files,
    )


def _run_contract(value: object, *, label: str) -> RunContract:
    record = _object(value, label=label)
    kind = record.get("kind")
    operation = record.get("operation")
    if not isinstance(kind, str) or not kind:
        raise HostedResearchContractError(f"{label}.kind must be non-empty")
    if not isinstance(operation, str) or not operation:
        raise HostedResearchContractError(f"{label}.operation must be non-empty")
    return RunContract(
        kind=kind,
        operation=operation,
        units=_strings(record.get("units"), label=f"{label}.units"),
        artifact=_artifact(record.get("artifact"), label=f"{label}.artifact"),
    )


def _aggregate_contract(value: object, *, label: str) -> AggregateContract:
    record = _object(value, label=label)
    operation = record.get("operation")
    pattern = record.get("download_pattern")
    prepare = record.get("prepare")
    if not isinstance(operation, str) or not operation:
        raise HostedResearchContractError(f"{label}.operation must be non-empty")
    if not isinstance(pattern, str) or not pattern:
        raise HostedResearchContractError(
            f"{label}.download_pattern must be non-empty"
        )
    if prepare is not None and (not isinstance(prepare, str) or not prepare):
        raise HostedResearchContractError(f"{label}.prepare must be non-empty")
    evidence = record.get("evidence", False)
    showdown = record.get("showdown", False)
    if not isinstance(evidence, bool) or not isinstance(showdown, bool):
        raise HostedResearchContractError(
            f"{label} capability flags must be booleans"
        )
    return AggregateContract(
        operation=operation,
        download_pattern=pattern,
        artifact=_artifact(record.get("artifact"), label=f"{label}.artifact"),
        prepare=prepare,
        evidence=evidence,
        showdown=showdown,
    )


def load_studies(path: Path = CONTRACT_PATH) -> tuple[HostedStudy, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HostedResearchContractError(
            f"cannot read hosted research contracts: {error}"
        ) from error
    root = _object(document, label="contract document")
    if root.get("schema") != "azelficoast.hosted-research-contracts":
        raise HostedResearchContractError("unsupported hosted research schema")
    studies_record = _object(root.get("studies"), label="studies")
    studies: list[HostedStudy] = []
    for name, raw in studies_record.items():
        if not isinstance(name, str) or not name:
            raise HostedResearchContractError("study names must be non-empty")
        record = _object(raw, label=name)
        showdown = record.get("showdown")
        evidence = record.get("evidence")
        if not isinstance(showdown, bool) or not isinstance(evidence, bool):
            raise HostedResearchContractError(
                f"{name} capability flags must be booleans"
            )
        aggregate_raw = record.get("aggregate")
        studies.append(
            HostedStudy(
                name=name,
                paths=_strings(record.get("paths"), label=f"{name}.paths"),
                showdown=showdown,
                evidence=evidence,
                run=_run_contract(record.get("run"), label=f"{name}.run"),
                aggregate=(
                    None
                    if aggregate_raw is None
                    else _aggregate_contract(
                        aggregate_raw,
                        label=f"{name}.aggregate",
                    )
                ),
            )
        )
    if not studies:
        raise HostedResearchContractError("at least one hosted study is required")
    names = [study.name for study in studies]
    if len(names) != len(set(names)):
        raise HostedResearchContractError("hosted study names must be unique")
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
