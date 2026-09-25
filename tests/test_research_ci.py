from __future__ import annotations

import json
from pathlib import Path

import pytest

from azelficoast.research.ci import (
    CandidateExperimentError,
    EXPERIMENTS,
    certify_candidate,
    experiments_for_paths,
    matrix_for,
    showdown_revision,
)


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


def test_showdown_revision_selects_only_showdown_consumers() -> None:
    selected = experiments_for_paths(("experiments/showdown-revision.txt",))
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


def test_candidate_certificate_is_derived_from_exact_matrix(tmp_path) -> None:
    matrix = matrix_for((EXPERIMENTS[0], EXPERIMENTS[1]))
    output = tmp_path / "certificate.json"
    certificate = certify_candidate(
        matrix_json=json.dumps(matrix),
        selected_count=2,
        exact_result="success",
        head_sha="a" * 40,
        output=output,
    )

    assert certificate == json.loads(output.read_text(encoding="utf-8"))
    assert certificate["git_sha"] == "a" * 40
    assert certificate["selected_experiments"] == [
        EXPERIMENTS[0].name,
        EXPERIMENTS[1].name,
    ]
    assert certificate["passed"] is True


def test_candidate_certificate_fails_closed_on_missing_exact_evidence(tmp_path) -> None:
    matrix = matrix_for((EXPERIMENTS[0],))
    with pytest.raises(CandidateExperimentError):
        certify_candidate(
            matrix_json=json.dumps(matrix),
            selected_count=1,
            exact_result="failure",
            head_sha="b" * 40,
            output=tmp_path / "certificate.json",
        )


def test_candidate_research_uses_repository_showdown_authority() -> None:
    assert showdown_revision() == Path("experiments/showdown-revision.txt").read_text(
        encoding="utf-8"
    ).strip()
