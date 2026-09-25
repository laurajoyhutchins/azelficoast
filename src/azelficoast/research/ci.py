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


SHOWDOWN_REVISION = "a5df8274e85b0889bf2a9b3422a08b39732374fc"
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
) -> CandidateExperiment:
    return CandidateExperiment(
        name=name,
        paths=paths,
        artifact_name=artifact,
        simulator=simulator,
        showdown=showdown,
        generators=() if generator is None else (Generator(*generator),),
        tests=tests,
        modules=tuple(ModuleRun(*module) for module in modules),
        checks=checks,
    )


EXPERIMENTS: tuple[CandidateExperiment, ...] = (
    _spec(
        "adaptive-execution",
        paths=(
            "src/azelficoast/adaptive_execution.py",
            "src/azelficoast/adaptive_execution_experiment.py",
            "src/azelficoast/class_native_belief.py",
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
                "azelficoast.adaptive_execution_experiment",
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
            "src/azelficoast/gen9_attack.py",
            "src/azelficoast/gen9_damage.py",
            "src/azelficoast/jax_gen9_attack.py",
            "src/azelficoast/class_native_belief.py",
            "src/azelficoast/native_damage_compiler.py",
            "src/azelficoast/attack_transition_experiment.py",
            "scripts/generate_showdown_attack_fixtures.cjs",
            "tests/test_gen9_attack.py",
            "tests/test_class_native_belief.py",
            "tests/test_native_damage_compiler.py",
            "README.md",
        ),
        artifact="attack-transition-evidence",
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
                "azelficoast.attack_transition_experiment",
                "attack-transition-experiment.json",
                "showdown-attack-fixtures.json",
            ),
        ),
        checks=(
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
            "src/azelficoast/class_native_belief.py",
            "src/azelficoast/class_native_belief_experiment.py",
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
                "azelficoast.class_native_belief_experiment",
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
            "src/azelficoast/gen9_damage.py",
            "src/azelficoast/showdown_damage_corpus.py",
            "src/azelficoast/jax_gen9_damage.py",
            "src/azelficoast/jax_gen9_damage_experiment.py",
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
                "azelficoast.showdown_damage_corpus",
                "showdown-gen9-damage-analysis.json",
                "showdown-gen9-damage-fixtures.json",
            ),
            (
                "azelficoast.jax_gen9_damage_experiment",
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
            "src/azelficoast/gen9_damage.py",
            "src/azelficoast/native_damage_compiler.py",
            "src/azelficoast/native_damage_experiment.py",
            "src/azelficoast/jax_gen9_damage.py",
            "src/azelficoast/showdown_damage_corpus.py",
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
                "azelficoast.showdown_damage_corpus",
                "showdown-gen9-damage-analysis.json",
                "showdown-gen9-damage-fixtures.json",
            ),
            (
                "azelficoast.native_damage_experiment",
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
            "src/azelficoast/simulator_ir.py",
            "src/azelficoast/jax_simulator.py",
            "src/azelficoast/jax_simulator_experiment.py",
            "tests/test_jax_simulator.py",
        ),
        artifact="jax-simulator-evidence",
        tests=("tests/test_jax_simulator.py",),
        modules=(
            (
                "azelficoast.jax_simulator_experiment",
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
            "src/azelficoast/simulator_ir.py",
            "src/azelficoast/showdown_transition_corpus.py",
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
                "azelficoast.showdown_transition_corpus",
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
            "src/azelficoast/gen9_ordered_attack.py",
            "src/azelficoast/jax_gen9_ordered_attack.py",
            "src/azelficoast/ordered_attack_belief.py",
            "src/azelficoast/ordered_attack_compiler.py",
            "src/azelficoast/ordered_attack_experiment.py",
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
                "azelficoast.ordered_attack_experiment",
                "ordered-attack-experiment.json",
                "showdown-ordered-attack-fixtures.json",
            ),
        ),
    ),
    _spec(
        "two-attack-turn",
        paths=(
            "src/azelficoast/gen9_two_attack_turn.py",
            "src/azelficoast/jax_gen9_two_attack_turn.py",
            "src/azelficoast/two_attack_turn_belief.py",
            "src/azelficoast/two_attack_turn_compiler.py",
            "src/azelficoast/two_attack_turn_experiment.py",
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
                "azelficoast.two_attack_turn_experiment",
                "two-attack-turn-experiment.json",
                "showdown-two-attack-turn-fixtures.json",
            ),
        ),
    ),
    _spec(
        "stateful-protect",
        paths=(
            "src/azelficoast/stateful_protect_turn.py",
            "src/azelficoast/stateful_protect_experiment.py",
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
                "azelficoast.stateful_protect_experiment",
                "stateful-protect-result.json",
                "stateful-protect-fixtures.json",
            ),
        ),
    ),
    _spec(
        "switch-entry-hazard",
        paths=(
            "src/azelficoast/switch_hazard_turn.py",
            "src/azelficoast/switch_hazard_experiment.py",
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
                "azelficoast.switch_hazard_experiment",
                "switch-hazard-result.json",
                "switch-hazard-fixtures.json",
            ),
        ),
    ),
    _spec(
        "switch-intimidate",
        paths=(
            "src/azelficoast/gen9_damage.py",
            "src/azelficoast/gen9_attack.py",
            "src/azelficoast/voluntary_switch_turn.py",
            "src/azelficoast/switch_hazard_turn.py",
            "src/azelficoast/staged_attack.py",
            "src/azelficoast/switch_intimidate_turn.py",
            "src/azelficoast/switch_intimidate_experiment.py",
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
                "azelficoast.switch_intimidate_experiment",
                "switch-intimidate-result.json",
                "switch-intimidate-fixtures.json",
            ),
        ),
    ),
    _spec(
        "voluntary-switch",
        paths=(
            "src/azelficoast/voluntary_switch_turn.py",
            "src/azelficoast/voluntary_switch_experiment.py",
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
                "azelficoast.voluntary_switch_experiment",
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
_SHARED_SHOWDOWN_PATHS = {".github/actions/setup-showdown/action.yml"}


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


def _run(command: Sequence[str], *, stdout: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    if stdout is None:
        subprocess.run(command, check=True)
        return
    with stdout.open("w", encoding="utf-8") as handle:
        subprocess.run(command, check=True, text=True, stdout=handle)
    print(stdout.read_text(encoding="utf-8"), end="")


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
                "passed": True,
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
            != SHOWDOWN_REVISION
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
        _run(command, stdout=work / module.output)

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
    return parser


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

    work = run_experiment(args.experiment, Path(args.output_root))
    print(work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
