from __future__ import annotations

from azelficoast.research.ci import EXPERIMENTS, experiments_for_paths, matrix_for


def _names(paths: tuple[str, ...]) -> set[str]:
    return {experiment.name for experiment in experiments_for_paths(paths)}


def test_shared_python_setup_selects_every_collapsed_experiment() -> None:
    assert _names((".github/actions/setup-python-environment/action.yml",)) == {
        experiment.name for experiment in EXPERIMENTS
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
