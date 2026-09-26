from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_PATH = REPOSITORY_ROOT / "experiments" / "candidate-research-contracts.json"


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


def _object(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CandidateExperimentError(f"{label} must be an object")
    return value


def _strings(
    value: object,
    *,
    label: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise CandidateExperimentError(f"{label} must be a string list")
    if not allow_empty and not value:
        raise CandidateExperimentError(f"{label} must not be empty")
    return tuple(value)


def _generator(value: object, *, label: str) -> Generator:
    record = _object(value, label=label)
    script = record.get("script")
    output = record.get("output")
    if not isinstance(script, str) or not script:
        raise CandidateExperimentError(f"{label}.script must be non-empty")
    if not isinstance(output, str) or not output:
        raise CandidateExperimentError(f"{label}.output must be non-empty")
    return Generator(script=script, output=output)


def _module(value: object, *, label: str) -> ModuleRun:
    record = _object(value, label=label)
    module = record.get("module")
    output = record.get("output")
    input_path = record.get("input")
    entrypoint = record.get("entrypoint")
    if not isinstance(module, str) or not module:
        raise CandidateExperimentError(f"{label}.module must be non-empty")
    if not isinstance(output, str) or not output:
        raise CandidateExperimentError(f"{label}.output must be non-empty")
    if input_path is not None and (
        not isinstance(input_path, str) or not input_path
    ):
        raise CandidateExperimentError(f"{label}.input must be null or non-empty")
    if not isinstance(entrypoint, str) or not entrypoint:
        raise CandidateExperimentError(f"{label}.entrypoint must be non-empty")
    return ModuleRun(
        module=module,
        output=output,
        input=input_path,
        entrypoint=entrypoint,
    )


def _check(value: object, *, label: str) -> Check:
    record = _object(value, label=label)
    output = record.get("output")
    path = record.get("path")
    operator = record.get("operator")
    expected = record.get("expected")
    if not isinstance(output, str) or not output:
        raise CandidateExperimentError(f"{label}.output must be non-empty")
    if (
        not isinstance(path, list)
        or not path
        or any(
            isinstance(component, bool)
            or not isinstance(component, (str, int))
            or isinstance(component, str)
            and not component
            for component in path
        )
    ):
        raise CandidateExperimentError(
            f"{label}.path must contain non-empty strings or integers"
        )
    if operator not in {"eq", "ge", "prefix"}:
        raise CandidateExperimentError(f"{label}.operator is unsupported")
    if operator == "ge" and (
        isinstance(expected, bool) or not isinstance(expected, (int, float))
    ):
        raise CandidateExperimentError(f"{label}.expected must be numeric for ge")
    if operator == "prefix" and not isinstance(expected, str):
        raise CandidateExperimentError(
            f"{label}.expected must be a string for prefix"
        )
    return Check(
        output=output,
        path=tuple(path),
        operator=operator,
        expected=expected,
    )


def _experiment(name: str, value: object) -> CandidateExperiment:
    record = _object(value, label=name)
    artifact_name = record.get("artifact_name")
    simulator = record.get("simulator")
    showdown = record.get("showdown")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise CandidateExperimentError(f"{name}.artifact_name must be non-empty")
    if not isinstance(simulator, bool) or not isinstance(showdown, bool):
        raise CandidateExperimentError(f"{name} capability flags must be booleans")

    raw_generators = record.get("generators")
    raw_modules = record.get("modules")
    raw_checks = record.get("checks")
    if not isinstance(raw_generators, list):
        raise CandidateExperimentError(f"{name}.generators must be a list")
    if not isinstance(raw_modules, list):
        raise CandidateExperimentError(f"{name}.modules must be a list")
    if not isinstance(raw_checks, list):
        raise CandidateExperimentError(f"{name}.checks must be a list")

    return CandidateExperiment(
        name=name,
        paths=_strings(record.get("paths"), label=f"{name}.paths", allow_empty=False),
        artifact_name=artifact_name,
        simulator=simulator,
        showdown=showdown,
        generators=tuple(
            _generator(item, label=f"{name}.generators[{index}]")
            for index, item in enumerate(raw_generators)
        ),
        tests=_strings(record.get("tests"), label=f"{name}.tests"),
        modules=tuple(
            _module(item, label=f"{name}.modules[{index}]")
            for index, item in enumerate(raw_modules)
        ),
        checks=tuple(
            _check(item, label=f"{name}.checks[{index}]")
            for index, item in enumerate(raw_checks)
        ),
    )


def load_experiments(path: Path = CONTRACT_PATH) -> tuple[CandidateExperiment, ...]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateExperimentError(
            f"cannot read candidate research contracts: {error}"
        ) from error

    root = _object(document, label="contract document")
    if root.get("schema") != "azelficoast.candidate-research-contracts":
        raise CandidateExperimentError("unsupported candidate research schema")
    records = _object(root.get("experiments"), label="experiments")

    experiments = tuple(
        _experiment(name, value)
        for name, value in records.items()
        if isinstance(name, str) and name
    )
    if len(experiments) != len(records):
        raise CandidateExperimentError("experiment names must be non-empty strings")
    if not experiments:
        raise CandidateExperimentError("at least one candidate experiment is required")

    names = [experiment.name for experiment in experiments]
    artifacts = [experiment.artifact_name for experiment in experiments]
    if len(names) != len(set(names)):
        raise CandidateExperimentError("candidate experiment names must be unique")
    if len(artifacts) != len(set(artifacts)):
        raise CandidateExperimentError("candidate artifact names must be unique")
    return experiments


EXPERIMENTS = load_experiments()
CONTRACTS_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}

_SHARED_PYTHON_PATHS = {
    ".github/actions/setup-python-environment/action.yml",
    ".github/workflows/research.yml",
    "experiments/candidate-research-contracts.json",
    "src/azelficoast/research/ci.py",
    "src/azelficoast/research/experiment_contracts.py",
}
_SHARED_SHOWDOWN_PATHS = {
    ".github/actions/setup-showdown/action.yml",
    "experiments/showdown-revision.txt",
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
