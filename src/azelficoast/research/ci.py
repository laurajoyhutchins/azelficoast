from __future__ import annotations

import argparse
import fnmatch
import importlib
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
    modules: tuple[tuple[str, str, str | None, str], ...],
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
            "src/azelficoast/research/experiments/adaptive_execution_experiment.py",
            "src/azelficoast/research/mechanics/class_native_belief.py",
            "tests/test_adaptive_execution.py",
            "README.md",
        ),
        artifact="adaptive-cost-model-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_damage_fixtures.cjs",
            "showdown-gen9-damage-fixtures.json",
        ),
        tests=("tests/test_adaptive_execution.py",),
        modules=(
            (
                "azelficoast.research.experiments.adaptive_execution_experiment",
                "adaptive-execution-experiment.json",
                "showdown-gen9-damage-fixtures.json",
                "run_experiment",
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
            "src/azelficoast/research/experiments/attack_transition_experiment.py",
            "showdown/verification/fixtures/generate_showdown_attack_fixtures.cjs",
            "tests/test_gen9_attack.py",
            "tests/test_class_native_belief.py",
            "tests/test_native_damage_compiler.py",
            "README.md",
        ),
        artifact="attack-transition-evidence",
        allow_nonzero_module_results=True,
        generator=(
            "showdown/verification/fixtures/generate_showdown_attack_fixtures.cjs",
            "showdown-attack-fixtures.json",
        ),
        tests=(
            "tests/test_gen9_attack.py",
            "tests/test_class_native_belief.py",
            "tests/test_native_damage_compiler.py",
        ),
        modules=(
            (
                "azelficoast.research.experiments.attack_transition_experiment",
                "attack-transition-experiment.json",
                "showdown-attack-fixtures.json",
                "run_experiment",
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
            "src/azelficoast/research/experiments/class_native_belief_experiment.py",
            "tests/test_class_native_belief.py",
            "README.md",
        ),
        artifact="class-native-belief-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_damage_fixtures.cjs",
            "showdown-gen9-damage-fixtures.json",
        ),
        tests=("tests/test_class_native_belief.py",),
        modules=(
            (
                "azelficoast.research.experiments.class_native_belief_experiment",
                "class-native-belief-experiment.json",
                "showdown-gen9-damage-fixtures.json",
                "run_experiment",
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
            "src/azelficoast/research/experiments/jax_gen9_damage_experiment.py",
            "showdown/verification/fixtures/generate_showdown_damage_fixtures.cjs",
            "tests/test_gen9_damage.py",
            "tests/test_showdown_damage_corpus.py",
            "tests/test_jax_gen9_damage.py",
        ),
        artifact="gen9-damage-kernel-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_damage_fixtures.cjs",
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
                "analyze_file",
            ),
            (
                "azelficoast.research.experiments.jax_gen9_damage_experiment",
                "gen9-damage-jax-experiment.json",
                "showdown-gen9-damage-fixtures.json",
                "run_experiment",
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
            "src/azelficoast/research/experiments/native_damage_experiment.py",
            "src/azelficoast/research/mechanics/jax_gen9_damage.py",
            "src/azelficoast/research/verification/showdown_damage_corpus.py",
            "showdown/verification/fixtures/generate_showdown_damage_fixtures.cjs",
            "tests/test_native_damage_compiler.py",
        ),
        artifact="native-damage-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_damage_fixtures.cjs",
            "showdown-gen9-damage-fixtures.json",
        ),
        tests=("tests/test_native_damage_compiler.py", "tests/test_gen9_damage.py"),
        modules=(
            (
                "azelficoast.research.verification.showdown_damage_corpus",
                "showdown-gen9-damage-analysis.json",
                "showdown-gen9-damage-fixtures.json",
                "analyze_file",
            ),
            (
                "azelficoast.research.experiments.native_damage_experiment",
                "native-damage-experiment.json",
                "showdown-gen9-damage-fixtures.json",
                "run_experiment",
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
            "src/azelficoast/research/experiments/jax_simulator_experiment.py",
            "tests/test_jax_simulator.py",
        ),
        artifact="jax-simulator-evidence",
        tests=("tests/test_jax_simulator.py",),
        modules=(
            (
                "azelficoast.research.experiments.jax_simulator_experiment",
                "jax-simulator-experiment.json",
                None,
                "run_experiment",
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
        "compiled-search-topology",
        paths=(
            "src/azelficoast/core/compiled_search.py",
            "src/azelficoast/belief/compiled_search.py",
            "src/azelficoast/belief/packed_evaluator.py",
            "src/azelficoast/research/experiments/compiled_search_experiment.py",
            "tests/test_compiled_search.py",
            "tests/test_compiled_packed_search.py",
            "docs/architecture/compiled-data-plane.md",
            "docs/search-architecture.md",
        ),
        artifact="compiled-search-topology-evidence",
        tests=(
            "tests/test_compiled_search.py",
            "tests/test_compiled_packed_search.py",
        ),
        modules=(
            (
                "azelficoast.research.experiments.compiled_search_experiment",
                "compiled-search-topology-experiment.json",
                None,
                "run_experiment",
            ),
        ),
        checks=(
            Check(
                "compiled-search-topology-experiment.json",
                ("passed",),
                "eq",
                True,
            ),
            Check(
                "compiled-search-topology-experiment.json",
                ("problem", "world_count"),
                "ge",
                512,
            ),
            Check(
                "compiled-search-topology-experiment.json",
                ("compiled_shape", "edge_count"),
                "ge",
                4096,
            ),
        ),
        showdown=False,
    ),
    _spec(
        "showdown-dependency",
        paths=(
            "src/azelficoast/research/mechanics/simulator_ir.py",
            "src/azelficoast/research/verification/showdown_transition_corpus.py",
            "showdown/verification/fixtures/generate_showdown_transition_fixtures.cjs",
            "tests/test_showdown_transition_corpus.py",
        ),
        artifact="showdown-transition-dependency-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_transition_fixtures.cjs",
            "showdown-transition-fixtures.json",
        ),
        modules=(
            (
                "azelficoast.research.verification.showdown_transition_corpus",
                "showdown-transition-analysis.json",
                "showdown-transition-fixtures.json",
                "analyze_file",
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
            "src/azelficoast/research/experiments/ordered_attack_experiment.py",
            "showdown/verification/fixtures/generate_showdown_ordered_attack_fixtures.cjs",
            "tests/test_gen9_ordered_attack.py",
        ),
        artifact="ordered-attack-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_ordered_attack_fixtures.cjs",
            "showdown-ordered-attack-fixtures.json",
        ),
        tests=("tests/test_gen9_ordered_attack.py",),
        modules=(
            (
                "azelficoast.research.experiments.ordered_attack_experiment",
                "ordered-attack-experiment.json",
                "showdown-ordered-attack-fixtures.json",
                "run_experiment",
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
            "src/azelficoast/research/experiments/two_attack_turn_experiment.py",
            "showdown/verification/fixtures/generate_showdown_two_attack_turn_fixtures.cjs",
            "tests/test_gen9_two_attack_turn.py",
        ),
        artifact="two-attack-turn-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_two_attack_turn_fixtures.cjs",
            "showdown-two-attack-turn-fixtures.json",
        ),
        tests=("tests/test_gen9_two_attack_turn.py",),
        modules=(
            (
                "azelficoast.research.experiments.two_attack_turn_experiment",
                "two-attack-turn-experiment.json",
                "showdown-two-attack-turn-fixtures.json",
                "run_experiment",
            ),
        ),
        allow_nonzero_module_results=True,
    ),
    _spec(
        "stateful-protect",
        paths=(
            "src/azelficoast/research/mechanics/stateful_protect_turn.py",
            "src/azelficoast/research/experiments/stateful_protect_experiment.py",
            "showdown/verification/fixtures/generate_showdown_stateful_protect_fixtures.cjs",
            "tests/test_stateful_protect_turn.py",
        ),
        artifact="stateful-protect-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_stateful_protect_fixtures.cjs",
            "stateful-protect-fixtures.json",
        ),
        tests=("tests/test_stateful_protect_turn.py",),
        modules=(
            (
                "azelficoast.research.experiments.stateful_protect_experiment",
                "stateful-protect-result.json",
                "stateful-protect-fixtures.json",
                "analyze_document",
            ),
        ),
    ),
    _spec(
        "switch-entry-hazard",
        paths=(
            "src/azelficoast/research/mechanics/switch_hazard_turn.py",
            "src/azelficoast/research/experiments/switch_hazard_experiment.py",
            "showdown/verification/fixtures/generate_showdown_switch_hazard_fixtures.cjs",
            "tests/test_switch_hazard_turn.py",
        ),
        artifact="switch-entry-hazard-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_switch_hazard_fixtures.cjs",
            "switch-hazard-fixtures.json",
        ),
        tests=("tests/test_switch_hazard_turn.py",),
        modules=(
            (
                "azelficoast.research.experiments.switch_hazard_experiment",
                "switch-hazard-result.json",
                "switch-hazard-fixtures.json",
                "analyze_document",
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
            "src/azelficoast/research/experiments/switch_intimidate_experiment.py",
            "showdown/verification/fixtures/generate_showdown_switch_intimidate_fixtures.cjs",
            "tests/test_switch_intimidate_turn.py",
        ),
        artifact="switch-intimidate-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_switch_intimidate_fixtures.cjs",
            "switch-intimidate-fixtures.json",
        ),
        tests=("tests/test_switch_intimidate_turn.py",),
        modules=(
            (
                "azelficoast.research.experiments.switch_intimidate_experiment",
                "switch-intimidate-result.json",
                "switch-intimidate-fixtures.json",
                "analyze_document",
            ),
        ),
    ),
    _spec(
        "voluntary-switch",
        paths=(
            "src/azelficoast/research/mechanics/voluntary_switch_turn.py",
            "src/azelficoast/research/experiments/voluntary_switch_experiment.py",
            "showdown/verification/fixtures/generate_showdown_voluntary_switch_fixtures.cjs",
            "tests/test_voluntary_switch_turn.py",
        ),
        artifact="voluntary-switch-evidence",
        generator=(
            "showdown/verification/fixtures/generate_showdown_voluntary_switch_fixtures.cjs",
            "voluntary-switch-fixtures.json",
        ),
        tests=("tests/test_voluntary_switch_turn.py",),
        modules=(
            (
                "azelficoast.research.experiments.voluntary_switch_experiment",
                "voluntary-switch-result.json",
                "voluntary-switch-fixtures.json",
                "analyze_document",
            ),
        ),
    ),
    _spec(
        "candidate-research-contract",
        paths=(
            "src/azelficoast/research/ci.py",
            "tests/test_research_ci.py",
        ),
        artifact="candidate-research-contract-evidence",
        tests=("tests/test_research_ci.py",),
        modules=(),
        simulator=False,
        showdown=False,
    ),
    _spec(
        "sql-policy-parameter-sweep",
        paths=(
            "src/azelficoast/core/sql.py",
            "src/azelficoast/queries/risk_adjusted.sql",
            "src/azelficoast/research/policy_sweep.py",
            "src/azelficoast/research/plans/risk_adjusted_policy_sweep.json",
            "src/azelficoast/research/fixtures/policy_action_statistics.json",
            "tests/test_core_sql.py",
            "tests/test_policy_sweep.py",
            "docs/sql-writing.md",
        ),
        artifact="sql-policy-parameter-sweep-evidence",
        tests=("tests/test_policy_sweep.py",),
        modules=(
            (
                "azelficoast.research.policy_sweep",
                "policy-parameter-sweep.json",
                None,
                "run_default_plan",
            ),
        ),
        checks=(
            Check(
                "policy-parameter-sweep.json",
                ("passed",),
                "eq",
                True,
            ),
            Check(
                "policy-parameter-sweep.json",
                ("grid", "point_count"),
                "eq",
                5,
            ),
            Check(
                "policy-parameter-sweep.json",
                ("matched_evidence", "result_count"),
                "eq",
                20,
            ),
            Check(
                "policy-parameter-sweep.json",
                ("fixture_summaries", 0, "policy_changes_across_grid"),
                "eq",
                True,
            ),
        ),
        simulator=False,
        showdown=False,
    ),
)

_BY_NAME = {experiment.name: experiment for experiment in EXPERIMENTS}
_SHARED_PYTHON_PATHS = {
    ".github/actions/setup-python-environment/action.yml",
    ".github/workflows/candidate-research.yml",
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


def _execute_module(module: ModuleRun, work: Path) -> int:
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
            raise CandidateExperimentError(f"{module.input} must contain a JSON object")
        result = entrypoint(argument)
    elif module.input is None:
        result = entrypoint()
    else:
        result = entrypoint(work / module.input)

    if not isinstance(result, Mapping):
        raise CandidateExperimentError(
            f"{module.module}:{module.entrypoint} returned a non-object result"
        )
    passed = result.get("passed")
    if passed is not None and not isinstance(passed, bool):
        raise CandidateExperimentError(
            f"{module.module}:{module.entrypoint} returned non-boolean passed"
        )

    output = work / module.output
    output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print(output.read_text(encoding="utf-8"), end="", flush=True)
    return 1 if passed is False else 0

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
        returncode = _execute_module(module, work)
        if returncode != 0 and not experiment.allow_nonzero_module_results:
            raise CandidateExperimentError(
                f"{experiment.name} module {module.module} failed"
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
