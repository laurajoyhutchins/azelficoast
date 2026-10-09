from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

from azelficoast.research.experiment_contracts import (
    CONTRACTS_BY_NAME,
    EXPERIMENTS,
    CandidateExperiment,
    CandidateExperimentError,
    Check,
    ModuleRun,
    experiments_for_paths,
    matrix_for,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SHOWDOWN_ROOT = Path("/tmp/pokemon-showdown")
DEFAULT_OUTPUT_ROOT = Path("/tmp/azelficoast-research")


def showdown_revision() -> str:
    from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT

    return PINNED_SHOWDOWN_COMMIT


SHOWDOWN_REVISION = showdown_revision()


def candidate_certificate(
    *,
    head_sha: str,
    matrix: Mapping[str, object],
    selected_count: int,
    exact_result: str,
    accelerator_static_result: str,
) -> dict[str, object]:
    """Validate exact-head evidence and build its canonical certificate."""
    entries = matrix.get("include")
    if not isinstance(entries, list) or any(
        not isinstance(entry, Mapping) or not isinstance(entry.get("experiment"), str)
        for entry in entries
    ):
        raise CandidateExperimentError(
            "candidate matrix must contain named experiment entries"
        )
    if selected_count != len(entries) or selected_count < 0:
        raise CandidateExperimentError(
            "candidate selected count does not match its matrix"
        )
    if len(head_sha) != 40 or any(
        character not in "0123456789abcdef" for character in head_sha
    ):
        raise CandidateExperimentError(
            "candidate head SHA must be a full lowercase Git SHA"
        )
    if accelerator_static_result != "success":
        raise CandidateExperimentError(
            f"accelerator static check failed: {accelerator_static_result}"
        )
    allowed_exact_results = {"success"} if selected_count else {"success", "skipped"}
    if exact_result not in allowed_exact_results:
        raise CandidateExperimentError(
            f"candidate exact evidence failed: {exact_result}"
        )
    return {
        "schema": "azelficoast.candidate-research-certificate",
        "git_sha": head_sha,
        "selected_experiments": [entry["experiment"] for entry in entries],
        "passed": True,
    }


def _changed_paths(base: str, head: str) -> tuple[str, ...]:
    if not base or not head:
        raise CandidateExperimentError(
            "base and head SHAs are required for PR planning"
        )
    process = subprocess.run(
        ("git", "diff", "--name-only", base, head),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return tuple(line for line in process.stdout.splitlines() if line)


def _run(
    command: Sequence[str],
    *,
    stdout: Path | None = None,
) -> None:
    print("+", " ".join(command), flush=True)
    if stdout is None:
        subprocess.run(command, check=True)
        return
    try:
        with stdout.open("w", encoding="utf-8") as handle:
            subprocess.run(command, check=True, text=True, stdout=handle)
    finally:
        if stdout.is_file():
            print(stdout.read_text(encoding="utf-8"), end="", flush=True)


def _execute_module(module: ModuleRun, work: Path) -> None:
    entrypoint = getattr(importlib.import_module(module.module), module.entrypoint, None)
    if not callable(entrypoint):
        raise CandidateExperimentError(
            f"{module.module} lacks callable entrypoint {module.entrypoint!r}"
        )

    if module.entrypoint == "analyze_document":
        if module.input is None:
            raise CandidateExperimentError(
                f"{module.module}:{module.entrypoint} requires an input artifact"
            )
        argument = json.loads((work / module.input).read_text(encoding="utf-8"))
        if not isinstance(argument, Mapping):
            raise CandidateExperimentError(
                f"{module.input} must contain a JSON object"
            )
        result = entrypoint(argument)
    elif module.input is None:
        result = entrypoint()
    else:
        result = entrypoint(work / module.input)

    if not isinstance(result, Mapping):
        raise CandidateExperimentError(
            f"{module.module}:{module.entrypoint} returned a non-object result"
        )

    output = work / module.output
    output.write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output.read_text(encoding="utf-8"), end="", flush=True)


def _read_path(record: object, path: Sequence[str | int]) -> object:
    value = record
    for component in path:
        if isinstance(component, int):
            if not isinstance(value, list):
                raise CandidateExperimentError(f"expected list at {path!r}")
            value = value[component]
        else:
            if not isinstance(value, Mapping) or component not in value:
                raise CandidateExperimentError(
                    f"missing {component!r} at {path!r}"
                )
            value = value[component]
    return value


def _check(check: Check, work: Path) -> None:
    document = json.loads((work / check.output).read_text(encoding="utf-8"))
    actual = _read_path(document, check.path)
    if check.operator == "eq":
        passed = actual == check.expected
    elif check.operator == "ge":
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or isinstance(check.expected, bool)
            or not isinstance(check.expected, (int, float))
        ):
            raise CandidateExperimentError(
                f"{check.output}:{'.'.join(map(str, check.path))} "
                "requires numeric values for ge"
            )
        passed = actual >= check.expected
    elif check.operator == "prefix":
        passed = isinstance(actual, str) and actual.startswith(str(check.expected))
    else:
        raise CandidateExperimentError(
            f"unknown check operator {check.operator!r}"
        )
    if not passed:
        raise CandidateExperimentError(
            f"{check.output}:{'.'.join(map(str, check.path))} "
            f"failed {check.operator} {check.expected!r}; got {actual!r}"
        )


def _git_sha() -> str:
    configured = os.environ.get("AZELFICOAST_GIT_SHA", "").strip()
    if configured:
        return configured
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _receipt(experiment: CandidateExperiment, work: Path) -> None:
    outputs = [
        {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(work.iterdir())
        if path.is_file() and path.name != "receipt.json"
    ]
    module_outcomes = []
    for module in experiment.modules:
        document = json.loads((work / module.output).read_text(encoding="utf-8"))
        if isinstance(document, Mapping) and isinstance(document.get("passed"), bool):
            module_outcomes.append(
                {
                    "path": module.output,
                    "passed": document["passed"],
                }
            )

    (work / "receipt.json").write_text(
        json.dumps(
            {
                "schema": "azelficoast.candidate-research-receipt",
                "experiment": experiment.name,
                "git_sha": _git_sha(),
                "showdown_revision": (
                    SHOWDOWN_REVISION if experiment.showdown else None
                ),
                "outputs": outputs,
                "module_outcomes": module_outcomes,
                "certified": True,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def run_experiment(
    name: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    experiment = CONTRACTS_BY_NAME.get(name)
    if experiment is None:
        raise CandidateExperimentError(
            f"unknown candidate experiment {name!r}"
        )
    if experiment.showdown:
        marker = SHOWDOWN_ROOT / ".azelficoast-showdown-sha"
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != SHOWDOWN_REVISION
        ):
            raise CandidateExperimentError(
                "verified pinned Showdown is not available"
            )

    work = output_root / experiment.name
    work.mkdir(parents=True, exist_ok=True)

    for generator in experiment.generators:
        _run(
            ("node", generator.script, str(SHOWDOWN_ROOT)),
            stdout=work / generator.output,
        )

    if experiment.tests:
        _run((sys.executable, "-m", "pytest", *experiment.tests))

    for module in experiment.modules:
        _execute_module(module, work)

    for check in experiment.checks:
        _check(check, work)

    _receipt(experiment, work)
    return work


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and execute exact-head candidate research contracts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--base", default="")
    plan.add_argument("--head", default="")
    plan.add_argument("--requested", default="")

    run = commands.add_parser("run")
    run.add_argument("experiment", choices=tuple(CONTRACTS_BY_NAME))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    commands.add_parser("certify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        if args.requested:
            if args.requested == "all":
                selected = EXPERIMENTS
            else:
                requested = CONTRACTS_BY_NAME.get(args.requested)
                if requested is None:
                    raise CandidateExperimentError(
                        f"unknown requested experiment {args.requested!r}"
                    )
                selected = (requested,)
        else:
            selected = experiments_for_paths(
                _changed_paths(args.base, args.head)
            )
        print(json.dumps(matrix_for(selected), separators=(",", ":")))
        return 0

    if args.command == "certify":
        try:
            certificate = candidate_certificate(
                head_sha=os.environ["HEAD_SHA"],
                matrix=json.loads(os.environ["MATRIX"]),
                selected_count=int(os.environ["SELECTED_COUNT"]),
                exact_result=os.environ["EXACT_RESULT"],
                accelerator_static_result=os.environ["ACCELERATOR_STATIC_RESULT"],
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise CandidateExperimentError(str(error)) from error
        destination = Path(
            os.environ.get(
                "CERTIFICATE_PATH",
                "/tmp/candidate-research-certificate/certificate.json",
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return 0

    work = run_experiment(args.experiment, Path(args.output_root))
    print(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
