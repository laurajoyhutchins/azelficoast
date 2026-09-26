from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import pytest

from azelficoast.belief import public_pretraining
from azelficoast.belief.public_pretraining import (
    PinnedShowdownPublicPosteriorSource,
    PublicPretrainingError,
    build_public_pretraining_records,
    run_public_pretraining,
)
from azelficoast.live.corpus import DecisionFixture
from azelficoast.live.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION


class FakePosteriorSource:
    showdown_commit = "showdown-test-commit"

    def posterior(self, fixture: DecisionFixture) -> Mapping[str, Any]:
        return {
            "schema": "test.posterior",
            "schema_version": 1,
            "source_fixture_id": fixture.fixture_id,
            "showdown_commit": self.showdown_commit,
            "legal_actions": list(fixture.legal_actions),
            "treatment": "generator_faithful",
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "worlds": [
                {
                    "world_id": "world-1",
                    "weight": 1.0,
                    "features": {"opponent": {"item": "unknown"}},
                }
            ],
        }


def _record(run_id: str, event_index: int, kind: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run_id,
        "event_index": event_index,
        "observed_at": "2026-01-01T00:00:00+00:00",
        "kind": kind,
        **extra,
    }


def _write_public_trace(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    replay_id = "gen9randombattle-123"
    battle_tag = f"battle-{replay_id}"
    for side, won, action in (
        ("p1", True, "/choose move earthquake"),
        ("p2", False, "/choose move protect"),
    ):
        run_id = f"public-replay:{replay_id}:{side}"
        rows.extend(
            [
                _record(
                    run_id,
                    0,
                    "protocol",
                    room=battle_tag,
                    protocol_index=0,
                    messages=[["", "turn", "1"]],
                ),
                _record(
                    run_id,
                    1,
                    "decision",
                    battle_tag=battle_tag,
                    decision_index=0,
                    state={
                        "battle_tag": battle_tag,
                        "legal_actions": [
                            "/choose move earthquake",
                            "/choose move protect",
                        ],
                    },
                    chosen_action=action,
                    decision_metadata={
                        "selected_policy": "recorded-human",
                        "training_policy_authority": False,
                        "source_replay_id": replay_id,
                        "source_side": side,
                        "source_rating": 1700,
                        "source_showdown_version": FakePosteriorSource.showdown_commit,
                    },
                ),
                _record(
                    run_id,
                    2,
                    "terminal",
                    battle_tag=battle_tag,
                    won=won,
                    lost=not won,
                    tied=False,
                    final_state={},
                ),
            ]
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_public_pretraining_keeps_both_replay_perspectives_in_one_split(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "public.jsonl"
    _write_public_trace(trace)

    rows, summary = build_public_pretraining_records(
        [trace],
        posterior_source=FakePosteriorSource(),
    )

    assert len(rows) == 2
    assert len({row["battle_id"] for row in rows}) == 1
    assert len({row["split_group_id"] for row in rows}) == 1
    assert len({row["split"] for row in rows}) == 1
    assert {row["targets"]["value"]["eventual_battle_outcome"] for row in rows} == {
        -1.0,
        1.0,
    }
    assert {
        row["provenance"]["policy_target"]["kind"] for row in rows
    } == {"public-human-imitation"}
    assert all(
        row["provenance"]["policy_target"]["scientific_search_teacher"] is False
        for row in rows
    )
    assert summary["policy_target_source"] == "public-human-imitation"


def test_public_pretraining_excludes_controls_from_another_showdown_revision(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "public.jsonl"
    _write_public_trace(trace)
    text = trace.read_text(encoding="utf-8").replace(
        FakePosteriorSource.showdown_commit,
        "other-showdown-commit",
    )
    trace.write_text(text, encoding="utf-8")

    with pytest.raises(PublicPretrainingError, match="no public human decisions"):
        build_public_pretraining_records(
            [trace],
            posterior_source=FakePosteriorSource(),
        )


def test_public_posterior_source_reuses_generator_population_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    revision = "a" * 40
    showdown = tmp_path / "showdown"
    (showdown / "dist" / "sim").mkdir(parents=True)
    (showdown / "dist" / "sim" / "battle.js").write_text("", encoding="utf-8")
    fixture = DecisionFixture(
        fixture_id="fixture-cache",
        state={"legal_actions": ["/choose move protect"]},
        protocol_prefix=(),
        control_decisions=(),
    )
    monkeypatch.setattr(
        public_pretraining,
        "build_probe_source",
        lambda _fixture: (
            {
                "schema": "azelficoast.real-belief-source-fixture",
                "schema_version": 1,
                "fixture_id": fixture.fixture_id,
                "showdown_commit": revision,
                "fixture": fixture.as_record(),
                "own_active_tera_type": "Steel",
            },
            "admitted",
        ),
    )

    node_calls: list[list[str]] = []

    def fake_run(command, **_kwargs):
        if command[0] == "git":
            return subprocess.CompletedProcess(command, 0, stdout=revision + "\n", stderr="")
        node_calls.append(list(command))
        event = "miss" if len(node_calls) == 1 else "hit"
        posterior = {
            "schema": "azelficoast.live-belief-posterior",
            "schema_version": 1,
            "source_fixture_id": fixture.fixture_id,
            "showdown_commit": revision,
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "treatment": "generator_faithful",
            "legal_actions": list(fixture.legal_actions),
            "worlds": [{"world_id": "same", "weight": 1.0, "hidden": {}}],
        }
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(posterior),
            stderr=f"azelficoast-generator-cache:{event}:cache-key\n",
        )

    monkeypatch.setattr(public_pretraining.subprocess, "run", fake_run)
    source = PinnedShowdownPublicPosteriorSource(showdown)
    cache_root = source.generator_cache_root
    try:
        first = source.posterior(fixture)
        second = source.posterior(fixture)
        assert first == second
        assert source.cache_stats() == {
            "posterior_probe_count": 2,
            "generator_population_cache_hit_count": 1,
            "generator_population_cache_miss_count": 1,
            "generator_population_count": 1,
        }
        assert all("--generator-cache-dir" in call for call in node_calls)
        assert all(str(cache_root) in call for call in node_calls)
    finally:
        source.close()

    assert not cache_root.exists()


def test_pinned_showdown_posterior_probe_declares_generator_faithful_treatment() -> None:
    script = (
        Path(public_pretraining.__file__).resolve().parents[3]
        / "showdown"
        / "runtime"
        / "probe_real_belief_trace.cjs"
    )
    source = script.read_text(encoding="utf-8")
    assert source.count('treatment: "generator_faithful"') == 1
    assert "function runProbe(argv, sourceDocument = null)" in source


def test_public_pretraining_refuses_to_replace_existing_promoted_evaluator(
    tmp_path: Path,
) -> None:
    promotion = tmp_path / "current.json"
    promotion.write_text("{}\n", encoding="utf-8")

    with pytest.raises(PublicPretrainingError, match="cold-start"):
        run_public_pretraining(
            [tmp_path / "unused.jsonl"],
            showdown_root=tmp_path / "showdown",
            dataset_path=tmp_path / "training.jsonl",
            models_dir=tmp_path / "models",
            receipts_dir=tmp_path / "receipts",
            promotion_file=promotion,
        )
