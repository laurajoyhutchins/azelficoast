from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SHOWDOWN_REVISION_PATH = Path("experiments/showdown-revision.txt")
SHOWDOWN_ROOT = Path("/tmp/pokemon-showdown")
DEFAULT_OUTPUT_ROOT = Path("/tmp/azelficoast-research")


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
    input: str | None = None


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
    simulator: bool = True
    showdown: bool = True
    allow_nonzero_module_results: bool = False
    generators: tuple[Generator, ...] = ()
    tests: tuple[str, ...] = ()
    modules: tuple[ModuleRun, ...] = ()
    checks: tuple[Check, ...] = ()


def _spec(
    name: str,
    *,
    paths: tuple[str, ...],
    artifact: str,
    generator: tuple[str, str] | None = None,
    tests: tuple[str, ...] = (),
    modules: tuple[tuple[str, str, str | None], ...],
    checks: tuple[Check, ...] = (),
    simulator: bool = True,
    showdown: bool = True,
    allow_nonzero_module_results: bool = False,
) -> CandidateExperiment:
    return CandidateExperiment(
        name=name,
        paths=paths,
        artifact_name=artifact,
        simulator=simulator,
        showdown=showdown,
        allow_nonzero_module_results=allow_nonzero_module_results,
        generators=() if generator is None else (Generator(*generator),),
        tests=tests,
        modules=tuple(ModuleRun(*module) for module in modules),
        checks=checks,
    )


EXPERIMENTS: tuple[CandidateExperiment, ...] = (
    _spec(
        "adaptive-execution",
        paths=(
            "src/azelficoast/research/adaptive_execution.py",
            "src/azelficoast/research/adaptive_execution_experiment.py",
            "src/azelficoast/research/mechanics/class_native_belief.py",
            "tests/test_adaptive_execution.py",
            "README.md",
        ),
        artifact="adaptive-cost-model-evidence",
        generator=(
            "scripts/generate_showdown_damage_fixtures.cjs",
            "showdown-gen9-damage-fixtures.json",
        ),
        tests=("tests/test_adaptive_execution.py",),
        modules=(
            (
                "azelficoast.research.adaptive_execution_experiment",
                "adaptive-execution-experiment.json",
                "showdown-gen9-damage-fixtures.json",
            ),
        ),
        checks=(
            Check(
                "adaptive-execution-experiment.json",
                ("profile", "calibrated_max_logical_world_count"),
                "ge",
                524288,
            ),
            Check(
                "adaptive-execution-experiment.json",
                ("target_signature",),
                "prefix",
                "sha256:",
            ),
        ),
    ),
    _spec(
        "attack-transition",
        paths=(
            "src/azelficoast/research/mechanics/gen9_attack.py",
            "src/azelficoast/research/mechanics/gen9_damage.py",
            "src/azelficoast/research/mechanics/jax_gen9_attack.py",
            "src/azelficoast/research/mechanics/class_native_belief.py",
            "src/azelficoast/research/mechanics/native_damage_compiler.py",
            "src/azelficoast/research/attack_transition_experiment.py",
            "scripts/generate_showdown_attack_fixtures.cjs",
            "tests/test_gen9_attack.py",
            "tests/test_class_native_belief.py",
            "tests/test_native_damage_compiler.py",
            "README.md",
        ),
        artifact="attack-transition-evidence",
        allow_nonzero_module_results=True,
        generator=(
            "scripts/generate_showdown_attack_fixtures.cjs",
            "showdown-attack-fixtures.json",
        ),
        tests=(
            "tests/test_gen9_attack.py",
            "tests/test_class_native_belief.py",
            "tests/test_native_damage_compiler.py",
        ),
        modules=(
            (
                "azelficoast.research.attack_transition_experiment",
                "attack-transition-experiment.json",
                "showdown-attack-fixtures.json",
            ),
        ),
        checks=(
            Check(
                "attack-transition-experiment.json",
                ("semantic_passed",),
                "eq",
                True,
            ),
            Check(
                "attack-transition-experiment.json",
                ("projection", "execution_classes"),
                "eq",
                49,
            ),
            Check(
                "attack-transition-experiment.json",
                ("effect_signature",),
                "prefix",
                "sha256:",
            ),
        ),
    ),
    _spec(
        "class-native-belief",
        paths=(
            "src/azelficoast/research/mechanics/class_native_belief.py",
            "src/azelficoast/research/class_native_belief_experiment.py",
            "tests/test_class_native_belief.py",
            "README.md",
        ),
        artifact="class-native-belief-evidence",
        generator=(
            "scripts/generate_showdown_damage_fixtures.cjs",
            "showdown-gen9-damage-fixtures.json",
        ),
        tests=("tests/test_class_native_belief.py",),
        modules=(
            (
                "azelficoast.research.class_native_belief_experiment",
                "class-native-belief-experiment.json",
                "showdown-gen9-damage-fixtures.json",
            ),
        ),
        checks=(
            Check(
                "class-native-belief-experiment.json",
                ("benchmarks", -1, "logical_world_count"),
                "eq",
                524288,
            ),
        ),
    ),
    _spec(
        "gen9-damage",
        paths=(
            "src/azelficoast/research/mechanics/gen9_damage.py",
            "src/azelficoast/research/verification/showdown_damage_corpus.py",
            "src/azelficoast/research/mechanics/jax_gen9_damage.py",
            "src/azelficoast/research/jax_gen9_damage_experiment.py",
            "scripts/generate_showdown_damage_fixtures.cjs",
            "tests/test_gen9_damage.py",
            "tests/test_showdown_damage_corpus.py",
            "tests/test_jax_gen9_damage.py",
        ),
        artifact="gen9-damage-kernel-evidence",
        generator=(
            "scripts/generate_showdown_damage_fixtures.cjs",
            "showdown-gen9-damage-fixtures.json",
        ),
        tests=(
            "tests/test_gen9_damage.py",
            "tests/test_showdown_damage_corpus.py",
            "tests/test_jax_gen9_damage.py",
        ),
        modules=(
            (
                "azelficoast.research.verification.showdown_damage_corpus",
                "showdown-gen9-damage-analysis.json",
                "showdown-gen9-damage-fixtures.json",
            ),
            (
                "azelficoast.research.jax_gen9_damage_experiment",
                "gen9-damage-jax-experiment.json",
                "showdown-gen9-damage-fixtures.json",
            ),
        ),
        checks=(
            Check(
                "showdown-gen9-damage-analysis.json",
                ("scenario_count",),
                "eq",
                12,
            ),
            Check(
                "showdown-gen9-damage-analysis.json",
                ("roll_case_count",),
                "eq",
                192,
            ),
            Check(
                "showdown-gen9-damage-analysis.json",
                ("exact_case_count",),
                "eq",
                192,
            ),
            Check(
                "gen9-damage-jax-experiment.json",
                ("correctness", "roll_case_count"),
                "eq",
                192,
            ),
        ),
    ),
    _spec(
        "native-damage",
        paths=(
            "src/azelficoast/research/mechanics/gen9_damage.py",
            "src/azelficoast/research/mechanics/native_damage_compiler.py",
            "src/azelficoast/research/native_damage_experiment.py",
            "src/azelficoast/research/mechanics/jax_gen9_damage.py",
            "src/azelficoast/research/verification/showdown_damage_corpus.py",
            "scripts/generate_showdown_damage_fixtures.cjs",
            "tests/test_native_damage_compiler.py",
        ),
        artifact="native-damage-evidence",
        generator=(
            "scripts/generate_showdown_damage_fixtures.cjs",
            "showdown-gen9-damage-fixtures.json",
        ),
        tests=("tests/test_native_damage_compiler.py", "tests/test_gen9_damage.py"),
        modules=(
            (
                "azelficoast.research.verification.showdown_damage_corpus",
                "showdown-gen9-damage-analysis.json",
                "showdown-gen9-damage-fixtures.json",
            ),
            (
                "azelficoast.research.native_damage_experiment",
                "native-damage-experiment.json",
                "showdown-gen9-damage-fixtures.json",
            ),
        ),
        checks=(
            Check(
                "native-damage-experiment.json",
                ("correctness", "roll_case_count"),
                "eq",
                192,
            ),
        ),
    ),
    _spec(
        "jax-simulator",
        paths=(
            "src/azelficoast/research/mechanics/simulator_ir.py",
            "src/azelficoast/research/mechanics/jax_simulator.py",
            "src/azelficoast/research/jax_simulator_experiment.py",
            "tests/test_jax_simulator.py",
        ),
        artifact="jax-simulator-evidence",
        tests=("tests/test_jax_simulator.py",),
        modules=(
            (
                "azelficoast.research.jax_simulator_experiment",
                "jax-simulator-experiment.json",
                None,
            ),
        ),
        checks=(
            Check(
                "jax-simulator-experiment.json",
                ("benchmarks", -1, "world_count"),
                "eq",
                524288,
            ),
        ),
        showdown=False,
    ),
    _spec(
        "showdown-dependency",
        paths=(
            "src/azelficoast/research/mechanics/simulator_ir.py",
            "src/azelficoast/research/verification/showdown_transition_corpus.py",
            "scripts/generate_showdown_transition_fixtures.cjs",
            "tests/test_showdown_transition_corpus.py",
        ),
        artifact="showdown-transition-dependency-evidence",
        generator=(
            "scripts/generate_showdown_transition_fixtures.cjs",
            "showdown-transition-fixtures.json",
        ),
        modules=(
            (
                "azelficoast.research.verification.showdown_transition_corpus",
                "showdown-transition-analysis.json",
                "showdown-transition-fixtures.json",
            ),
        ),
        checks=(
            Check(
                "showdown-transition-analysis.json",
                ("fixture_count",),
                "eq",
                256,
            ),
            Check(
                "showdown-transition-analysis.json",
                ("protect", "dependency_classes"),
                "eq",
                8,
            ),
            Check(
                "showdown-transition-analysis.json",
                ("damage", "dependency_classes"),
                "eq",
                16,
            ),
        ),
        simulator=False,
    ),
    _spec(
        "ordered-attack",
        paths=(
            "src/azelficoast/research/mechanics/gen9_ordered_attack.py",
            "src/azelficoast/research/mechanics/jax_gen9_ordered_attack.py",
            "src/azelficoast/research/mechanics/ordered_attack_belief.py",
            "src/azelficoast/research/mechanics/ordered_attack_compiler.py",
            "src/azelficoast/research/ordered_attack_experiment.py",
            "scripts/generate_showdown_ordered_attack_fixtures.cjs",
            "tests/test_gen9_ordered_attack.py",
        ),
        artifact="ordered-attack-evidence",
        generator=(
            "scripts/generate_showdown_ordered_attack_fixtures.cjs",
            "showdown-ordered-attack-fixtures.json",
        ),
        tests=("tests/test_gen9_ordered_attack.py",),
        modules=(
            (
                "azelficoast.research.ordered_attack_experiment",
                "ordered-attack-experiment.json",
                "showdown-ordered-attack-fixtures.json",
            ),
        ),
    ),
    _spec(
        "two-attack-turn",
        paths=(
            "src/azelficoast/research/mechanics/gen9_two_attack_turn.py",
            "src/azelficoast/research/mechanics/jax_gen9_two_attack_turn.py",
            "src/azelficoast/research/mechanics/two_attack_turn_belief.py",
            "src/azelficoast/research/mechanics/two_attack_turn_compiler.py",
            "src/azelficoast/research/two_attack_turn_experiment.py",
            "scripts/generate_showdown_two_attack_turn_fixtures.cjs",
            "tests/test_gen9_two_attack_turn.py",
        ),
        artifact="two-attack-turn-evidence",
        generator=(
            "scripts/generate_showdown_two_attack_turn_fixtures.cjs",
            "showdown-two-attack-turn-fixtures.json",
        ),
        tests=("tests/test_gen9_two_attack_turn.py",),
        modules=(
            (
                "azelficoast.research.two_attack_turn_experiment",
                "two-attack-turn-experiment.json",
                "showdown-two-attack-turn-fixtures.json",
            ),
        ),
    ),
    _spec(
        "stateful-protect",
        paths=(
            "src/azelficoast/research/mechanics/stateful_protect_turn.py",
            "src/azelficoast/research/stateful_protect_experiment.py",
            "scripts/generate_showdown_stateful_protect_fixtures.cjs",
            "tests/test_stateful_protect_turn.py",
        ),
        artifact="stateful-protect-evidence",
        generator=(
            "scripts/generate_showdown_stateful_protect_fixtures.cjs",
            "stateful-protect-fixtures.json",
        ),
        tests=("tests/test_stateful_protect_turn.py",),
        modules=(
            (
                "azelficoast.research.stateful_protect_experiment",
                "stateful-protect-result.json",
                "stateful-protect-fixtures.json",
            ),
        ),
    ),
    _spec(
        "switch-entry-hazard",
        paths=(
            "src/azelficoast/research/mechanics/switch_hazard_turn.py",
            "src/azelficoast/research/switch_hazard_experiment.py",
            "scripts/generate_showdown_switch_hazard_fixtures.cjs",
            "tests/test_switch_hazard_turn.py",
        ),
        artifact="switch-entry-hazard-evidence",
        generator=(
            "scripts/generate_showdown_switch_hazard_fixtures.cjs",
            "switch-hazard-fixtures.json",
        ),
        tests=("tests/test_switch_hazard_turn.py",),
        modules=(
            (
                "azelficoast.research.switch_hazard_experiment",
                "switch-hazard-result.json",
                "switch-hazard-fixtures.json",
            ),
        ),
    ),
    _spec(
        "switch-intimidate",
        paths=(
            "src/azelficoast/research/mechanics/gen9_damage.py",
            "src/azelficoast/research/mechanics/gen9_attack.py",
            "src/azelficoast/research/mechanics/voluntary_switch_turn.py",
            "src/azelficoast/research/mechanics/switch_hazard_turn.py",
            "src/azelficoast/research/mechanics/staged_attack.py",
            "src/azelficoast/research/mechanics/switch_intimidate_turn.py",
            "src/azelficoast/research/switch_intimidate_experiment.py",
            "scripts/generate_showdown_switch_intimidate_fixtures.cjs",
            "tests/test_switch_intimidate_turn.py",
        ),
        artifact="switch-intimidate-evidence",
        generator=(
            "scripts/generate_showdown_switch_intimidate_fixtures.cjs",
            "switch-intimidate-fixtures.json",
        ),
        tests=("tests/test_switch_intimidate_turn.py",),
        modules=(
            (
                "azelficoast.research.switch_intimidate_experiment",
                "switch-intimidate-result.json",
                "switch-intimidate-fixtures.json",
            ),
        ),
    ),
    _spec(
        "voluntary-switch",
        paths=(
            "src/azelficoast/research/mechanics/voluntary_switch_turn.py",
            "src/azelficoast/research/voluntary_switch_experiment.py",
            "scripts/generate_showdown_voluntary_switch_fixtures.cjs",
            "tests/test_voluntary_switch_turn.py",
        ),
        artifact="voluntary-switch-evidence",
        generator=(
            "scripts/generate_showdown_voluntary_switch_fixtures.cjs",
            "voluntary-switch-fixtures.json",
        ),
        tests=("tests/test_voluntary_switch_turn.py",),
        modules=(
            (
                "azelficoast.research.voluntary_switch_experiment",
                "voluntary-switch-result.json",
                "voluntary-switch-fixtures.json",
            ),
        ),
    ),
)

_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}
_SHARED_PYTHON_PATHS = {
    ".github/actions/setup-python-environment/action.yml",
    ".github/workflows/candidate-research.yml",
    "src/azelficoast/research/ci.py",
}
_SHARED_SHOWDOWN_PATHS = {
    ".github/actions/setup-showdown/action.yml",
    "experiments/showdown-revision.txt",
}


def experiments_for_paths(paths: Iterable[str]) -> tuple[CandidateExperiment, ...]:
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


def matrix_for(experiments: Sequence[CandidateExperiment]) -> dict[str, object]:
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
    check: bool = True,
) -> int:
    print("+", " ".join(command), flush=True)
    if stdout is None:
        return subprocess.run(command, check=check).returncode
    try:
        with stdout.open("w", encoding="utf-8") as handle:
            process = subprocess.run(
                command,
                check=check,
                text=True,
                stdout=handle,
            )
    finally:
        if stdout.is_file():
            print(stdout.read_text(encoding="utf-8"), end="", flush=True)
    return process.returncode


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
        passed = float(actual) >= float(check.expected)
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


def showdown_revision() -> str:
    revision = SHOWDOWN_REVISION_PATH.read_text(encoding="utf-8").strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise CandidateExperimentError("invalid authoritative Pokémon Showdown revision")
    return revision


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
                    showdown_revision() if experiment.showdown else None
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
    experiment = _BY_NAME.get(name)
    if experiment is None:
        raise CandidateExperimentError(
            f"unknown candidate experiment {name!r}"
        )
    if experiment.showdown:
        marker = SHOWDOWN_ROOT / ".azelficoast-showdown-sha"
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip()
            != showdown_revision()
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
        command = [sys.executable, "-m", module.module]
        if module.input is not None:
            command.append(str(work / module.input))
        _run(
            command,
            stdout=work / module.output,
            check=not experiment.allow_nonzero_module_results,
        )

    for check in experiment.checks:
        _check(check, work)

    _receipt(experiment, work)
    return work


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan and execute exact-head candidate research."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    plan = commands.add_parser("plan")
    plan.add_argument("--base", default="")
    plan.add_argument("--head", default="")
    plan.add_argument("--requested", default="")

    run = commands.add_parser("run")
    run.add_argument("experiment", choices=tuple(_BY_NAME))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))

    certify = commands.add_parser("certify")
    certify.add_argument("--matrix", required=True)
    certify.add_argument("--selected-count", required=True, type=int)
    certify.add_argument("--exact-result", required=True)
    certify.add_argument("--head", required=True)
    certify.add_argument("--output", type=Path, required=True)
    return parser


def certify_candidate(
    *,
    matrix_json: str,
    selected_count: int,
    exact_result: str,
    head_sha: str,
    output: Path,
) -> dict[str, object]:
    if selected_count != 0 and exact_result != "success":
        raise CandidateExperimentError(
            f"candidate research failed: exact jobs={exact_result}"
        )
    matrix = json.loads(matrix_json)
    if not isinstance(matrix, Mapping):
        raise CandidateExperimentError("candidate matrix must be an object")
    include = matrix.get("include")
    if not isinstance(include, list):
        raise CandidateExperimentError("candidate matrix must contain include list")
    if len(include) != selected_count:
        raise CandidateExperimentError(
            f"candidate matrix count mismatch: {len(include)} != {selected_count}"
        )
    selected: list[str] = []
    for entry in include:
        if not isinstance(entry, Mapping):
            raise CandidateExperimentError("candidate matrix entry must be an object")
        experiment = entry.get("experiment")
        if not isinstance(experiment, str) or experiment not in _BY_NAME:
            raise CandidateExperimentError(
                f"candidate matrix references unknown experiment {experiment!r}"
            )
        selected.append(experiment)
    if len(selected) != len(set(selected)):
        raise CandidateExperimentError("candidate matrix contains duplicate experiments")
    if len(head_sha) != 40 or any(
        character not in "0123456789abcdef" for character in head_sha
    ):
        raise CandidateExperimentError("candidate certificate requires an exact git SHA")

    certificate = {
        "schema": "azelficoast.candidate-research-certificate",
        "git_sha": head_sha,
        "selected_experiments": selected,
        "passed": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(certificate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return certificate


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan":
        if args.requested:
            if args.requested == "all":
                selected = EXPERIMENTS
            else:
                requested = _BY_NAME.get(args.requested)
                if requested is None:
                    raise CandidateExperimentError(
                        f"unknown requested experiment {args.requested!r}"
                    )
                selected = (requested,)
        else:
            selected = experiments_for_paths(
                _changed_paths(args.base, args.head)
            )
        print(
            json.dumps(
                matrix_for(selected),
                separators=(",", ":"),
            )
        )
        return 0

    if args.command == "certify":
        certificate = certify_candidate(
            matrix_json=args.matrix,
            selected_count=args.selected_count,
            exact_result=args.exact_result,
            head_sha=args.head,
            output=args.output,
        )
        print(json.dumps(certificate, sort_keys=True))
        return 0

    work = run_experiment(args.experiment, Path(args.output_root))
    print(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
