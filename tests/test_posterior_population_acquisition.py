from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import azelficoast.research.posterior_population_acquisition as acquisition
from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT
from azelficoast.live.corpus import DecisionFixture, _fixture_id, write_corpus
from azelficoast.research.posterior_population_acquisition import (
    PopulationAcquisitionError,
    build_source_metadata,
    bind_source_artifact,
    build_acquisition_receipt,
    selected_fixture_ids,
    extract_candidate_fixtures,
)
from azelficoast.research.posterior_population_selection import (
    PosteriorPopulationSelectionError,
    freeze_issue_69_population,
)


CONTRACT_PATH = (
    Path(__file__).parents[1]
    / "experiments/data/posterior-stratified-population-contract.json"
)


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _source(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    corpus_path = tmp_path / "corpus.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    fixture_id = _fixture_id({}, ())
    write_corpus(
        [
            DecisionFixture(
                fixture_id,
                {},
                (),
                ({"battle_tag": "battle-1", "decision_index": 0},),
            )
        ],
        corpus_path,
    )
    decisions_path.write_bytes(b'{"decision":"frozen"}\n')
    return (
        {
            "schema": "azelficoast.posterior-population-source/v1",
            "workflow_run_id": 40001,
            "head_sha": "a" * 40,
            "generation_head_sha": "a" * 40,
            "showdown_commit": PINNED_SHOWDOWN_COMMIT,
            "battle_count": 3072,
            "decision_state_count": 1,
            "decisions_sha256": hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
            "corpus_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
        },
        corpus_path,
        decisions_path,
    )


def _artifact() -> dict[str, Any]:
    return {
        "id": 50001,
        "name": "posterior-population-source-40001",
        "digest": "sha256:" + "e" * 64,
        "workflow_run_id": 40001,
        "workflow_head_sha": "a" * 40,
    }


def test_source_generation_uses_single_concurrent_battle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = Mock()
    monkeypatch.setattr(
        "azelficoast.research.hosted.common.start_showdown_server",
        lambda: server,
    )
    commands: list[list[str]] = []

    def fail_generation(command: list[str], *, check: bool) -> None:
        commands.append(command)
        raise acquisition.subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(acquisition.subprocess, "run", fail_generation)

    with pytest.raises(PopulationAcquisitionError, match="source generation failed"):
        acquisition.generate_source(
            output_root=tmp_path / "source",
            workflow_run_id=40001,
            generation_head_sha="a" * 40,
            showdown_root=tmp_path / "showdown",
        )

    assert len(commands) == 1
    assert commands[0][commands[0].index("--concurrency") + 1] == "1"
    server.terminate.assert_called_once_with()
    server.wait.assert_called_once_with(timeout=30)


def test_source_metadata_requires_exact_battle_budget_and_records_file_digests(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    results_path.write_text(
        "".join(
            json.dumps({"battle_tag": f"battle-{index}"}) + "\n"
            for index in range(3072)
        ),
        encoding="utf-8",
    )
    decisions_path.write_text(
        json.dumps(
            {
                "schema": "azelficoast.decision-trace",
                "schema_version": 1,
                "kind": "decision",
                "battle_tag": "battle-1",
                "decision_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fixture_id = _fixture_id({}, ())
    write_corpus(
        [
            DecisionFixture(
                fixture_id,
                {},
                (),
                ({"battle_tag": "battle-1", "decision_index": 0},),
            )
        ],
        corpus_path,
    )

    metadata = build_source_metadata(
        results_path=results_path,
        decisions_path=decisions_path,
        corpus_path=corpus_path,
        workflow_run_id=40001,
        generation_head_sha="a" * 40,
    )

    assert metadata["battle_count"] == 3072
    assert metadata["decision_state_count"] == 1
    assert metadata["corpus_sha256"] == hashlib.sha256(corpus_path.read_bytes()).hexdigest()
    assert metadata["decisions_sha256"] == hashlib.sha256(decisions_path.read_bytes()).hexdigest()
    assert metadata["showdown_commit"] == PINNED_SHOWDOWN_COMMIT


def test_source_metadata_rejects_decision_traces_from_another_source(
    tmp_path: Path,
) -> None:
    results_path = tmp_path / "results.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    results_path.write_text(
        "".join(
            json.dumps({"battle_tag": f"battle-{index}"}) + "\n"
            for index in range(3072)
        ),
        encoding="utf-8",
    )
    decision = {
        "kind": "decision",
        "battle_tag": "unrelated-battle",
        "decision_index": 0,
    }
    decisions_path.write_text(json.dumps(decision) + "\n", encoding="utf-8")
    fixture_id = _fixture_id({}, ())
    write_corpus(
        [
            DecisionFixture(
                fixture_id,
                {},
                (),
                ({"battle_tag": "unrelated-battle", "decision_index": 0},),
            )
        ],
        corpus_path,
    )

    with pytest.raises(PopulationAcquisitionError, match="results battle tags"):
        build_source_metadata(
            results_path=results_path,
            decisions_path=decisions_path,
            corpus_path=corpus_path,
            workflow_run_id=40001,
            generation_head_sha="a" * 40,
        )


def test_source_metadata_rejects_short_battle_run(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    decisions_path = tmp_path / "decisions.jsonl"
    corpus_path = tmp_path / "corpus.jsonl"
    results_path.write_text('{"battle_tag":"only-one"}\n', encoding="utf-8")
    decisions_path.write_text("", encoding="utf-8")
    corpus_path.write_text("", encoding="utf-8")

    with pytest.raises(PopulationAcquisitionError, match="3072"):
        build_source_metadata(
            results_path=results_path,
            decisions_path=decisions_path,
            corpus_path=corpus_path,
            workflow_run_id=40001,
            generation_head_sha="a" * 40,
        )


def test_source_binding_changes_only_source_artifact_identity(tmp_path: Path) -> None:
    contract = _contract()
    source, corpus_path, decisions_path = _source(tmp_path)
    original = copy.deepcopy(contract)

    bound = bind_source_artifact(
        contract,
        source,
        _artifact(),
        corpus_path=corpus_path,
        decisions_path=decisions_path,
    )

    assert bound["population"]["source_artifact"] == {
        "workflow_run_id": 40001,
        "artifact_id": 50001,
        "artifact_name": "posterior-population-source-40001",
        "artifact_digest": "sha256:" + "e" * 64,
        "workflow_head_sha": "a" * 40,
        "generation_head_sha": "a" * 40,
        "battle_count": 3072,
        "decision_state_count": 1,
        "decisions_sha256": source["decisions_sha256"],
        "corpus_sha256": source["corpus_sha256"],
    }
    original["population"]["source_artifact"] = bound["population"]["source_artifact"]
    assert bound == original
    assert bound["population"]["minimum_selected_states"] == 100
    assert bound["population"]["max_selected_states"] == 128
    assert bound["population"]["generator_rounds"] == 2048
    assert bound["population"]["mechanics_screen_rounds"] == 512
    assert bound["population"]["forbidden_selection_fields"] == contract["population"][
        "forbidden_selection_fields"
    ]


@pytest.mark.parametrize(
    ("source_key", "source_value", "artifact_key", "artifact_value"),
    [
        ("workflow_run_id", 40002, "workflow_run_id", 40001),
        ("generation_head_sha", "f" * 40, "workflow_head_sha", "a" * 40),
        ("battle_count", 3071, "workflow_run_id", 40001),
        ("decisions_sha256", "f" * 64, "workflow_run_id", 40001),
        ("corpus_sha256", "f" * 64, "workflow_run_id", 40001),
    ],
)
def test_source_binding_rejects_mismatched_provenance(
    tmp_path: Path,
    source_key: str,
    source_value: Any,
    artifact_key: str,
    artifact_value: Any,
) -> None:
    source, corpus_path, decisions_path = _source(tmp_path)
    artifact = _artifact()
    source[source_key] = source_value
    artifact[artifact_key] = artifact_value

    with pytest.raises(PopulationAcquisitionError):
        bind_source_artifact(
            _contract(),
            source,
            artifact,
            corpus_path=corpus_path,
            decisions_path=decisions_path,
        )


def test_source_binding_rejects_missing_digest_or_ambiguous_artifact(tmp_path: Path) -> None:
    source, corpus_path, decisions_path = _source(tmp_path)
    artifact = _artifact()
    del source["corpus_sha256"]
    with pytest.raises(PopulationAcquisitionError):
        bind_source_artifact(
            _contract(), source, artifact, corpus_path=corpus_path, decisions_path=decisions_path
        )

    source, corpus_path, decisions_path = _source(tmp_path)
    artifact["matches"] = 2
    with pytest.raises(PopulationAcquisitionError):
        bind_source_artifact(
            _contract(), source, artifact, corpus_path=corpus_path, decisions_path=decisions_path
        )


def test_source_binding_rejects_changed_source_files(tmp_path: Path) -> None:
    source, corpus_path, decisions_path = _source(tmp_path)
    corpus_path.write_bytes(b'{"state":"changed"}\n')

    with pytest.raises(PopulationAcquisitionError, match="digest"):
        bind_source_artifact(
            _contract(), source, _artifact(), corpus_path=corpus_path, decisions_path=decisions_path
        )


def test_shortfall_keeps_complete_admission_ledger_before_selector_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, corpus_path, decisions_path = _source(tmp_path)
    contract = _contract()
    artifact = contract["population"]["source_artifact"]
    artifact.update(
        {
            "workflow_run_id": source["workflow_run_id"],
            "artifact_id": 50001,
            "artifact_name": "posterior-population-source-40001",
            "artifact_digest": _artifact()["digest"],
            "workflow_head_sha": source["generation_head_sha"],
            "generation_head_sha": source["generation_head_sha"],
            "battle_count": source["battle_count"],
            "decision_state_count": source["decision_state_count"],
            "decisions_sha256": source["decisions_sha256"],
            "corpus_sha256": source["corpus_sha256"],
        }
    )
    selected_rows = [
        {"fixture_id": f"fixture-{index}", "filename": f"{index:03d}-fixture-{index}.json"}
        for index in range(51)
    ]
    manifest = {
        "selected_count": 51,
        "selected": selected_rows,
        "eligible_fixture_ids": [row["fixture_id"] for row in selected_rows],
        "ineligible_fixtures": [{"fixture_id": "rejected", "reason": "screen"}],
        "source_exclusions": [{"fixture_id": "outside", "reason": "no-world-query"}],
        "source_decision_state_count": 1,
        "bounded_candidate_count": 52,
    }
    monkeypatch.setattr(
        "azelficoast.research.posterior_population_selection.freeze_population",
        lambda **_: manifest,
    )

    with pytest.raises(PosteriorPopulationSelectionError, match="produced 51 states"):
        freeze_issue_69_population(
            contract=contract,
            source_metadata=source,
            candidates={},
            mechanics={},
            corpus_path=corpus_path,
            selected_dir=tmp_path / "selected",
        )

    ledger = json.loads(
        (tmp_path / "selected" / "admission-ledger.json").read_text(encoding="utf-8")
    )
    assert ledger["selected_count"] == 51
    assert len(ledger["selected_fixture_ids"]) == 51
    assert ledger["candidate_rejections"] == manifest["ineligible_fixtures"]
    assert ledger["source_exclusions"] == manifest["source_exclusions"]


def test_shortfall_receipt_is_negative_and_records_minimum() -> None:
    receipt = build_acquisition_receipt(
        selected_fixture_ids=[f"fixture-{index}" for index in range(51)],
        minimum_selected_states=100,
        max_selected_states=128,
    )

    assert receipt["status"] == "negative"
    assert receipt["selected_count"] == 51
    assert receipt["required_minimum"] == 100
    assert receipt["study_positive"] is False


def test_hundred_distinct_states_is_acquisition_positive_not_study_positive() -> None:
    receipt = build_acquisition_receipt(
        selected_fixture_ids=[f"fixture-{index}" for index in range(100)],
        minimum_selected_states=100,
        max_selected_states=128,
    )

    assert receipt["status"] == "acquisition-positive"
    assert receipt["selected_count"] == 100
    assert receipt["study_positive"] is False


def test_selector_failure_cannot_yield_acquisition_positive_receipt() -> None:
    receipt = build_acquisition_receipt(
        selected_fixture_ids=[f"fixture-{index}" for index in range(100)],
        minimum_selected_states=100,
        max_selected_states=128,
        selector_exit_code=1,
    )

    assert receipt["status"] == "negative"
    assert receipt["selector_exit_code"] == 1
    assert receipt["study_positive"] is False


def test_receipt_rejects_duplicate_ids_and_selector_cap_violation() -> None:
    with pytest.raises(PopulationAcquisitionError, match="unique"):
        build_acquisition_receipt(
            selected_fixture_ids=["same", "same"],
            minimum_selected_states=100,
            max_selected_states=128,
        )

    with pytest.raises(PopulationAcquisitionError, match="maximum"):
        build_acquisition_receipt(
            selected_fixture_ids=[f"fixture-{index}" for index in range(129)],
            minimum_selected_states=100,
            max_selected_states=128,
        )


def test_receipt_ignores_admission_ledger_sidecar(tmp_path: Path) -> None:
    selected_dir = tmp_path / "selected"
    selected_dir.mkdir()
    fixture_id = "a" * 64
    (selected_dir / f"001-{fixture_id}.json").write_text("{}\n", encoding="utf-8")
    (selected_dir / "admission-ledger.json").write_text("{}\n", encoding="utf-8")

    assert selected_fixture_ids(selected_dir) == [fixture_id]


def test_candidate_extraction_accounts_for_every_source_fixture(tmp_path: Path) -> None:
    states = [{"fixture": "candidate"}, {"fixture": "excluded"}]
    fixtures = [
        DecisionFixture(_fixture_id(state, ()), state, (), ()) for state in states
    ]
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus(fixtures, corpus_path)
    candidate_id, excluded_id = [fixture.fixture_id for fixture in fixtures]
    candidate_document = {
        "candidates": [{"fixture_id": candidate_id}],
        "excluded_fixtures": [{"fixture_id": excluded_id, "reason": "no-world-query"}],
    }
    candidates_path = tmp_path / "candidate-fixtures.jsonl"
    exclusions_path = tmp_path / "source-exclusions.json"

    ledger = extract_candidate_fixtures(
        corpus_path=corpus_path,
        candidates_document=candidate_document,
        candidate_fixtures_path=candidates_path,
        exclusions_path=exclusions_path,
    )

    assert json.loads(candidates_path.read_text(encoding="utf-8"))["fixture_id"] == candidate_id
    assert ledger["excluded_fixtures"] == candidate_document["excluded_fixtures"]


def test_candidate_extraction_rejects_incomplete_source_coverage(tmp_path: Path) -> None:
    fixture_id = _fixture_id({}, ())
    corpus_path = tmp_path / "corpus.jsonl"
    write_corpus([DecisionFixture(fixture_id, {}, (), ())], corpus_path)

    with pytest.raises(PopulationAcquisitionError, match="cover source fixtures"):
        extract_candidate_fixtures(
            corpus_path=corpus_path,
            candidates_document={"candidates": [], "excluded_fixtures": []},
            candidate_fixtures_path=tmp_path / "candidate-fixtures.jsonl",
            exclusions_path=tmp_path / "source-exclusions.json",
        )
