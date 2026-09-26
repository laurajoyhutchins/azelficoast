from __future__ import annotations

from pathlib import Path

from azelficoast.research.hosted import belief
from azelficoast.research.hosted.contracts import (
    STUDIES,
    STUDIES_BY_NAME,
    aggregate_matrix,
    run_matrix,
    studies_for_paths,
)


EXPECTED_STUDIES = {
    "conditional-team-prior",
    "decision-relevance-quotient",
    "exhausted-bench-real-belief",
    "factored-hidden-bench-prior",
    "joint-random-battle-posterior",
    "live-belief-coverage",
    "natural-depth-regret",
    "natural-population-strategy-fusion",
    "natural-status-move-public-belief",
    "opponent-team-completion-support",
    "policy-boundary-refinement-experiment",
    "protect-action-survival-experiment",
    "protect-continuation-experiment",
    "protect-speed-fork-mechanics",
    "public-belief-exact-corpus",
    "real-belief-decision-trace",
    "real-belief-survival-witness",
    "replay-world-experiment",
    "status-move-hidden-world-prior",
}


def _names(paths: tuple[str, ...]) -> set[str]:
    return {study.name for study in studies_for_paths(paths)}


def test_all_hosted_research_is_contract_registered() -> None:
    assert {study.name for study in STUDIES} == EXPECTED_STUDIES


def test_hosted_runner_change_selects_every_study() -> None:
    assert _names(("src/azelficoast/research/hosted/common.py",)) == EXPECTED_STUDIES
    assert _names(("experiments/hosted-research-contracts.json",)) == EXPECTED_STUDIES
    assert _names((".github/workflows/research.yml",)) == EXPECTED_STUDIES


def test_shared_capabilities_select_only_consumers() -> None:
    showdown = studies_for_paths((".github/actions/setup-showdown/action.yml",))
    evidence = studies_for_paths(
        (".github/actions/setup-research-evidence/action.yml",)
    )
    evidence_authority = studies_for_paths(
        ("src/azelficoast/research/evidence.py",)
    )

    assert showdown
    assert evidence
    assert evidence_authority == evidence
    assert all(study.showdown for study in showdown)
    assert all(study.evidence for study in evidence)
    assert "replay-world-experiment" not in {study.name for study in showdown}
    assert "protect-action-survival-experiment" not in {
        study.name for study in evidence
    }


def test_witness_membership_is_contract_data_not_workflow_data() -> None:
    exhausted = STUDIES_BY_NAME["exhausted-bench-real-belief"].run.units
    public = STUDIES_BY_NAME["public-belief-exact-corpus"].run.units

    assert set(exhausted) == set(belief.EXHAUSTED_FIXTURES)
    assert set(public) == set(belief.PUBLIC_BELIEF_FIXTURES)


def test_sharded_studies_compile_to_explicit_execution_units() -> None:
    depth = run_matrix((STUDIES_BY_NAME["natural-depth-regret"],))["include"]
    population = run_matrix(
        (STUDIES_BY_NAME["natural-population-strategy-fusion"],)
    )["include"]

    assert isinstance(depth, list)
    assert isinstance(population, list)
    assert [entry["unit"] for entry in depth] == [str(index) for index in range(6)]
    assert [entry["unit"] for entry in population] == [
        str(index) for index in range(8)
    ]
    assert all(entry["showdown"] is True for entry in depth)
    assert all(entry["evidence"] is True for entry in population)


def test_only_population_studies_require_aggregation() -> None:
    matrix = aggregate_matrix(STUDIES)["include"]
    assert isinstance(matrix, list)
    assert {entry["study"] for entry in matrix} == {
        "natural-depth-regret",
        "natural-population-strategy-fusion",
    }


def test_compiled_artifact_names_are_unique() -> None:
    matrix = run_matrix(STUDIES)["include"]
    assert isinstance(matrix, list)
    names = [entry["artifact_name"] for entry in matrix]
    assert len(names) == len(set(names))


def test_hosted_cli_exposes_only_generic_execution_commands() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "azelficoast"
        / "research"
        / "hosted"
        / "__main__.py"
    ).read_text(encoding="utf-8")

    for command in (
        'commands.add_parser("matrix")',
        'commands.add_parser("exhausted-bench")',
        'commands.add_parser("public-belief-exact")',
        'commands.add_parser("natural-population-shard")',
        'commands.add_parser("natural-depth-shard")',
    ):
        assert command not in source
    for command in ("plan", "run-unit", "prepare-aggregate", "aggregate"):
        assert f'commands.add_parser("{command}")' in source
