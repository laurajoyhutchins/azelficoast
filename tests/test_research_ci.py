from __future__ import annotations

from azelficoast.research.ci import (
    EXPERIMENTS,
    candidate_certificate,
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


def test_candidate_certificate_is_owned_by_python_and_fails_closed() -> None:
    matrix = {"include": [{"experiment": "contract", "artifact_name": "contract-evidence"}]}
    certificate = candidate_certificate(
        head_sha="a" * 40,
        matrix=matrix,
        selected_count=1,
        exact_result="success",
        accelerator_static_result="success",
    )
    assert certificate == {
        "schema": "azelficoast.candidate-research-certificate",
        "git_sha": "a" * 40,
        "selected_experiments": ["contract"],
        "passed": True,
    }

    for statuses in (("failure", "success"), ("success", "failure"), ("success", "skipped")):
        try:
            candidate_certificate(
                head_sha="a" * 40,
                matrix=matrix,
                selected_count=1,
                exact_result=statuses[0],
                accelerator_static_result=statuses[1],
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted failed certificate statuses: {statuses}")


def test_repository_showdown_revision_selects_showdown_consumers() -> None:
    selected = experiments_for_paths(("experiments/showdown-revision.txt",))
    assert selected
    assert all(experiment.showdown for experiment in selected)
