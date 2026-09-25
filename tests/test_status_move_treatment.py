from __future__ import annotations

import hashlib
import json

import pytest

from azelficoast.status_move_treatment import (
    StatusMoveTreatmentError,
    freeze_source,
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _trace(tmp_path, *, current_move=True):
    path = tmp_path / "decisions.jsonl"
    state = {
        "battle_tag": "battle-test",
        "turn": 7,
        "player": "Azelficoast",
        "opponent": "Rival",
        "active": {"species": "Manaphy"},
        "opponent_active": {
            "species": "Articuno",
            "level": 86,
            "item": None,
        },
        "team": {"p1: Manaphy": {"species": "Manaphy"}},
        "legal_actions": ["/choose move surf"],
    }
    messages = [
        ["", "player", "p1", "Azelficoast"],
        ["", "player", "p2", "Rival"],
        ["", "switch", "p2a: Swalot", "Swalot, L90", "100/100"],
        ["", "move", "p2a: Swalot", "Sludge Bomb", "p1a: Manaphy"],
        ["", "switch", "p2a: Articuno", "Articuno, L86", "87/100"],
    ]
    if current_move:
        messages.append(["", "move", "p2a: Articuno", "Roost", "p2a: Articuno"])

    records = [
        {
            "run_id": "run",
            "event_index": 1,
            "kind": "protocol",
            "room": "battle-test",
            "messages": messages,
        },
        {
            "run_id": "run",
            "event_index": 2,
            "kind": "decision",
            "battle_tag": "battle-test",
            "decision_index": 7,
            "chosen_action": "/choose move surf",
            "state": state,
        },
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    frozen_state = dict(state)
    frozen_state.pop("battle_tag")
    fixture_id = hashlib.sha256(
        _canonical({"state": frozen_state, "protocol_prefix": [messages]}).encode()
    ).hexdigest()
    return path, fixture_id


def _spec(fixture_id: str) -> dict[str, object]:
    return {
        "schema": "azelficoast.natural-status-move-treatment",
        "schema_version": 1,
        "source_artifact": {"artifact_id": 1},
        "selection": {
            "trace_run_id": "run",
            "battle_tag": "battle-test",
            "event_index": 2,
            "decision_index": 7,
            "turn": 7,
            "own_species": "manaphy",
            "opponent_species": "articuno",
            "observed_opponent_move": "Roost",
            "opponent_response_move": "Roost",
            "opponent_bench_species": "Swalot",
            "own_active_tera_type": "Grass",
            "plausible_items": ["Heavy-Duty Boots"],
            "expected_fixture_id": fixture_id,
        },
        "showdown_commit": "pinned",
    }


def test_freeze_source_binds_exact_current_active_move(tmp_path) -> None:
    trace, fixture_id = _trace(tmp_path)

    source = freeze_source(_spec(fixture_id), trace)

    assert source["fixture_id"] == fixture_id
    assert source["opponent_response_move"] == "Roost"
    assert source["plausible_items"] == ["Heavy-Duty Boots"]
    assert "battle_tag" not in source["fixture"]["state"]


def test_freeze_source_rejects_move_from_previous_active(tmp_path) -> None:
    trace, fixture_id = _trace(tmp_path, current_move=False)

    with pytest.raises(
        StatusMoveTreatmentError,
        match="current opponent response evidence changed",
    ):
        freeze_source(_spec(fixture_id), trace)
