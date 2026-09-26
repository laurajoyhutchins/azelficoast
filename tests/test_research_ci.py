from __future__ import annotations

import json
from pathlib import Path
import sys
import types

from azelficoast.research.ci import (
    EXPERIMENTS,
    ModuleRun,
    _execute_module,
    experiments_for_paths,
    matrix_for,
)


def _names(paths: tuple[str, ...]) -> set[str]:
    return {experiment.name for experiment in experiments_for_paths(paths)}


def test_shared_python_setup_selects_every_collapsed_experiment() -> None:
    assert _names((".github/actions/setup-python-environment/action.yml",)) == {
        experiment.name for experiment in EXPERIMENTS
    }


def test_candidate_runner_change_selects_only_its_contract() -> None:
    assert _names(("src/azelficoast/research/ci.py",)) == {
        "candidate-research-contract"
    }


def test_showdown_setup_selects_only_showdown_consumers() -> None:
    selected = experiments_for_paths((".github/actions/setup-showdown/action.yml",))
    assert selected
    assert all(experiment.showdown for experiment in selected)
    assert "jax-simulator" not in {experiment.name for experiment in selected}


def test_damage_change_selects_every_dependent_candidate_experiment() -> None:
    selected = _names(("src/azelficoast/research/mechanics/gen9_damage.py",))
    assert {
        "attack-transition",
        "gen9-damage",
        "native-damage",
        "switch-intimidate",
    } <= selected
    assert "jax-simulator" not in selected


def test_unrelated_population_change_spends_no_collapsed_candidate_evidence() -> None:
    assert experiments_for_paths(("src/azelficoast/research/population.py",)) == ()


def test_candidate_matrix_carries_execution_requirements() -> None:
    selected = experiments_for_paths(("src/azelficoast/research/mechanics/jax_simulator.py",))
    assert matrix_for(selected) == {
        "include": [
            {
                "experiment": "jax-simulator",
                "artifact_name": "jax-simulator-evidence",
                "simulator": True,
                "showdown": False,
            }
        ]
    }


def test_candidate_experiment_names_and_artifacts_are_unique() -> None:
    names = [experiment.name for experiment in EXPERIMENTS]
    artifacts = [experiment.artifact_name for experiment in EXPERIMENTS]
    assert len(names) == len(set(names))
    assert len(artifacts) == len(set(artifacts))

def test_host_performance_does_not_override_semantic_candidate_certification() -> None:
    attack = next(
        experiment for experiment in EXPERIMENTS if experiment.name == "attack-transition"
    )

    assert attack.allow_nonzero_module_results is True
    assert any(
        check.path == ("semantic_passed",)
        and check.operator == "eq"
        and check.expected is True
        for check in attack.checks
    )
    assert all(
        not experiment.allow_nonzero_module_results
        for experiment in EXPERIMENTS
        if experiment.name != "attack-transition"
    )



def test_compiled_search_changes_select_jax_candidate_evidence() -> None:
    selected = experiments_for_paths(
        ("src/azelficoast/core/compiled_search.py",)
    )
    assert [experiment.name for experiment in selected] == [
        "compiled-search-topology"
    ]
    assert selected[0].simulator is True
    assert selected[0].showdown is False


ROOT = Path(__file__).resolve().parents[1]


def test_registered_callable_experiments_have_no_local_cli_authority() -> None:
    modules = [
        module
        for experiment in EXPERIMENTS
        for module in experiment.modules
        if module.entrypoint is not None
    ]
    assert len(modules) == 12

    for module in modules:
        path = ROOT / "src" / Path(*module.module.split(".")).with_suffix(".py")
        source = path.read_text(encoding="utf-8")
        assert "argparse.ArgumentParser" not in source
        assert 'if __name__ == "__main__":' not in source


def test_callable_module_contract_executes_path_entrypoint(tmp_path, monkeypatch) -> None:
    module_name = "azelficoast_test_path_experiment"
    imported = types.ModuleType(module_name)
    input_path = tmp_path / "fixture.json"
    input_path.write_text("{}\n", encoding="utf-8")

    def run_experiment(path: Path) -> dict[str, object]:
        assert path == input_path
        return {"schema": "test", "passed": True}

    imported.run_experiment = run_experiment
    monkeypatch.setitem(sys.modules, module_name, imported)

    contract = ModuleRun(
        module=module_name,
        output="result.json",
        input=input_path.name,
        entrypoint="run_experiment",
        input_kind="path",
    )
    assert _execute_module(contract, tmp_path) == 0
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8")) == {
        "passed": True,
        "schema": "test",
    }


def test_callable_module_contract_loads_json_document(tmp_path, monkeypatch) -> None:
    module_name = "azelficoast_test_document_experiment"
    imported = types.ModuleType(module_name)
    input_path = tmp_path / "fixture.json"
    input_path.write_text('{"fixture": 7}\n', encoding="utf-8")

    def analyze_document(document: object) -> dict[str, object]:
        assert document == {"fixture": 7}
        return {"schema": "test", "passed": False}

    imported.analyze_document = analyze_document
    monkeypatch.setitem(sys.modules, module_name, imported)

    contract = ModuleRun(
        module=module_name,
        output="result.json",
        input=input_path.name,
        entrypoint="analyze_document",
        input_kind="json",
    )
    assert _execute_module(contract, tmp_path) == 1
    assert json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))[
        "passed"
    ] is False
