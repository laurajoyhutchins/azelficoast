from __future__ import annotations

from pathlib import Path

from azelficoast.research import training_service


def _identity() -> dict[str, object]:
    return {
        "schema": "azelficoast.belief-evaluator",
        "schema_version": 1,
        "checkpoint_digest": "sha256:" + ("a" * 64),
    }


def test_cold_start_bootstraps_then_runs_automatic_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    state = tmp_path / "state"
    promotion = state / "evaluators" / "current.json"

    def fake_invoke(argv):
        command = tuple(str(value) for value in argv)
        calls.append(command)
        if "import-public" in command:
            page = sum("import-public" in call for call in calls)
            return {
                "admitted_count": 12 if page == 1 else 0,
                "decision_count": 120,
                "trace": str(tmp_path / "decisions.jsonl"),
                "next_before": 10,
            }
        if "bootstrap-public" in command:
            promotion.parent.mkdir(parents=True, exist_ok=True)
            promotion.write_text("{}\n", encoding="utf-8")
            return {"promoted": True}
        if "auto" in command:
            return {"generation_count": 1}
        raise AssertionError(command)

    monkeypatch.setattr(training_service, "_invoke", fake_invoke)
    monkeypatch.setattr(training_service, "_current_identity", lambda _: _identity())

    summary = training_service.run_training_iteration(
        showdown_root=tmp_path / "showdown",
        state_root=state,
        work_root=tmp_path / "work",
        run_id="123",
    )

    assert summary["status"] == "trained"
    assert summary["current_evaluator"] == _identity()
    assert any("import-public" in command for command in calls)
    assert any("bootstrap-public" in command for command in calls)
    assert any("auto" in command for command in calls)
    assert (state / "state.json").is_file()
    assert (tmp_path / "work" / "run-summary.json").is_file()


def test_existing_promotion_skips_public_bootstrap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []
    state = tmp_path / "state"
    promotion = state / "evaluators" / "current.json"
    promotion.parent.mkdir(parents=True)
    promotion.write_text("{}\n", encoding="utf-8")

    def fake_invoke(argv):
        command = tuple(str(value) for value in argv)
        calls.append(command)
        assert "auto" in command
        return {"generation_count": 1}

    monkeypatch.setattr(training_service, "_invoke", fake_invoke)
    monkeypatch.setattr(training_service, "_current_identity", lambda _: _identity())

    summary = training_service.run_training_iteration(
        showdown_root=tmp_path / "showdown",
        state_root=state,
        work_root=tmp_path / "work",
        run_id="456",
    )

    assert summary["status"] == "trained"
    assert len(calls) == 1
    assert "auto" in calls[0]
    assert "import-public" not in calls[0]
    assert "bootstrap-public" not in calls[0]


def test_cold_start_fails_closed_without_enough_revision_matched_replays(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_invoke(argv):
        command = tuple(str(value) for value in argv)
        calls.append(command)
        assert "import-public" in command
        page = len(calls)
        return {
            "admitted_count": 0,
            "decision_count": 0,
            "trace": str(tmp_path / f"decisions-{page}.jsonl"),
            "next_before": 100 - page,
        }

    monkeypatch.setattr(training_service, "_invoke", fake_invoke)

    summary = training_service.run_training_iteration(
        showdown_root=tmp_path / "showdown",
        state_root=tmp_path / "state",
        work_root=tmp_path / "work",
        run_id="789",
    )

    assert summary["status"] == "bootstrap-not-ready"
    assert summary["current_evaluator"] is None
    assert len(calls) == training_service.BOOTSTRAP_MAX_PAGES
    assert all("import-public" in command for command in calls)
