from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from azelficoast.belief.public_pretraining import (
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
