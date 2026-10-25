from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from azelficoast.research.json_schema_contracts import ContractShapeError, load_contract_document


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = REPOSITORY_ROOT / "experiments" / "contracts" / "candidate.json"


class CandidateExperimentError(RuntimeError):
    """Raised when candidate research evidence cannot be admitted."""


@dataclass(frozen=True, slots=True)
class Generator:
    script: str
    output: str


@dataclass(frozen=True, slots=True)
class ModuleRun:
    module: str
    output: str
    input: str | None
    entrypoint: str


@dataclass(frozen=True, slots=True)
class Check:
    output: str
    path: tuple[str | int, ...]
    operator: str
    expected: object


@dataclass(frozen=True, slots=True)
class CandidateExperiment:
    name: str
    paths: tuple[str, ...]
    artifact_name: str
    simulator: bool
    showdown: bool
    generators: tuple[Generator, ...]
    tests: tuple[str, ...]
    modules: tuple[ModuleRun, ...]
    checks: tuple[Check, ...]


def _generator(record: dict[str, Any]) -> Generator:
    return Generator(script=record["script"], output=record["output"])


def _module(record: dict[str, Any]) -> ModuleRun:
    return ModuleRun(
        module=record["module"], output=record["output"],
        input=record.get("input"), entrypoint=record["entrypoint"],
    )


def _check(record: dict[str, Any]) -> Check:
    return Check(
        output=record["output"], path=tuple(record["path"]),
        operator=record["operator"], expected=record["expected"],
    )


def _experiment(name: str, record: dict[str, Any]) -> CandidateExperiment:
    return CandidateExperiment(
        name=name,
        paths=tuple(record["paths"]),
        artifact_name=record["artifact_name"],
        simulator=record["simulator"],
        showdown=record["showdown"],
        generators=tuple(_generator(item) for item in record["generators"]),
        tests=tuple(record["tests"]),
        modules=tuple(_module(item) for item in record["modules"]),
        checks=tuple(_check(item) for item in record["checks"]),
    )


def load_experiments(path: Path = CONTRACT_PATH) -> tuple[CandidateExperiment, ...]:
    try:
        document = load_contract_document(
            path, REPOSITORY_ROOT / "experiments/contracts/schemas/candidate.schema.json",
        )
    except ContractShapeError as error:
        raise CandidateExperimentError(str(error)) from error

    experiments = tuple(
        _experiment(name, record) for name, record in document["experiments"].items()
    )
    artifact_names = [experiment.artifact_name for experiment in experiments]
    if len(artifact_names) != len(set(artifact_names)):
        raise CandidateExperimentError("candidate artifact names must be unique")
    return experiments

EXPERIMENTS = load_experiments()
CONTRACTS_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}

_SHARED_PYTHON_PATHS = {
    ".github/actions/setup-python-environment/action.yml",
    ".python-version",
    ".github/workflows/research.yml",
    "pyproject.toml",
    "uv.lock",
    "experiments/contracts/candidate.json",
    "experiments/contracts/schemas/candidate.schema.json",
    "src/azelficoast/research/json_schema_contracts.py",
    "src/azelficoast/research/ci.py",
    "src/azelficoast/research/experiment_contracts.py",
}
_SHARED_SHOWDOWN_PATHS = {
    ".github/actions/setup-showdown/action.yml",
    "showdown/revision.json",
}


def experiments_for_paths(
    paths: Iterable[str],
) -> tuple[CandidateExperiment, ...]:
    changed = set(paths)
    selected: list[CandidateExperiment] = []
    for experiment in EXPERIMENTS:
        patterns = set(experiment.paths) | _SHARED_PYTHON_PATHS
        if experiment.showdown:
            patterns |= _SHARED_SHOWDOWN_PATHS
        if experiment.simulator:
            patterns |= {"pyproject.toml", "uv.lock"}
        if any(
            fnmatch.fnmatch(path, pattern)
            for path in changed
            for pattern in patterns
        ):
            selected.append(experiment)
    return tuple(selected)


def matrix_for(
    experiments: Sequence[CandidateExperiment],
) -> dict[str, object]:
    return {
        "include": [
            {
                "experiment": experiment.name,
                "artifact_name": experiment.artifact_name,
                "simulator": experiment.simulator,
                "showdown": experiment.showdown,
            }
            for experiment in experiments
        ]
    }
