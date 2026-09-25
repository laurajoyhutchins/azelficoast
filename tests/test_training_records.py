from __future__ import annotations

import json

from azelficoast.corpus import build_fixtures
from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION
from azelficoast.training_records import build_training_records


def _record(*, event: int, kind: str, **extra: object) -> dict[str, object]:
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": "run",
        "event_index": event,
        "observed_at": "2026-09-25T00:00:00+00:00",
        "kind": kind,
        **extra,
    }


def _state(turn: int) -> dict[str, object]:
    return {
        "battle_tag": "battle-one",
        "turn": turn,
        "player": "azelficoast",
        "opponent": "opponent",
        "active": {"species": "Azelf"},
        "opponent_active": {"species": "Rotom-Wash", "item": None},
        "team": {},
        "opponent_team": {},
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "available_moves": ["psychic", "thunderbolt"],
        "available_switches": [],
        "legal_actions": ["/choose move psychic", "/choose move thunderbolt"],
        "force_switch": False,
        "trapped": False,
        "can_tera": False,
    }


def _search_metadata(
    *,
    action: str = "/choose move psychic",
    psychic: float = 0.7,
    thunderbolt: float = 0.2,
) -> dict[str, object]:
    values = {
        "/choose move psychic": psychic,
        "/choose move thunderbolt": thunderbolt,
    }
    return {
        "selected_policy": "public-belief",
        "belief": {
            "status": "selected",
            "reason": "bounded-public-belief",
            "action": action,
            "diagnostics": {
                "showdown_commit": "pinned",
                "public_belief_value": values[action],
                "public_belief_root_values": values,
                "public_belief_search_horizons": 2,
            },
        },
    }


def _write_trace(path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_training_records_keep_all_decisions_from_one_battle_in_one_split(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                event=0,
                kind="protocol",
                room="battle-one",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                event=1,
                kind="decision",
                battle_tag="battle-one",
                decision_index=0,
                state=_state(4),
                chosen_action="/choose move psychic",
                decision_metadata=_search_metadata(),
            ),
            _record(
                event=2,
                kind="protocol",
                room="battle-one",
                protocol_index=1,
                messages=[["", "turn", "5"]],
            ),
            _record(
                event=3,
                kind="decision",
                battle_tag="battle-one",
                decision_index=1,
                state=_state(5),
                chosen_action="/choose move psychic",
                decision_metadata=_search_metadata(psychic=0.6, thunderbolt=0.4),
            ),
            _record(
                event=4,
                kind="terminal",
                battle_tag="battle-one",
                won=True,
                lost=False,
                tied=False,
                final_state={},
            ),
        ],
    )

    records, summary = build_training_records([trace])

    assert len(records) == 2
    assert len({record["battle_id"] for record in records}) == 1
    assert len({record["split"] for record in records}) == 1
    assert summary["battle_count"] == 1
    assert all(
        record["targets"]["value"]["eventual_battle_outcome"] == 1.0
        for record in records
    )
    assert records[0]["targets"]["value"]["public_belief_search_return"] == 0.7
    assert records[0]["targets"]["policy"]["action_probabilities"] == {
        "/choose move psychic": 1.0,
        "/choose move thunderbolt": 0.0,
    }


def test_stronger_search_annotation_relabels_policy_without_using_behavior_action(
    tmp_path,
) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                event=0,
                kind="protocol",
                room="battle-one",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                event=1,
                kind="decision",
                battle_tag="battle-one",
                decision_index=0,
                state=_state(4),
                chosen_action="/choose move thunderbolt",
                decision_metadata=_search_metadata(
                    action="/choose move thunderbolt",
                    psychic=0.1,
                    thunderbolt=0.5,
                ),
            ),
            _record(
                event=2,
                kind="terminal",
                battle_tag="battle-one",
                won=False,
                lost=True,
                tied=False,
                final_state={},
            ),
        ],
    )
    [fixture] = build_fixtures([trace])
    annotation = tmp_path / "search.json"
    annotation.write_text(
        json.dumps(
            {
                "schema": "azelficoast.public-belief-search-target",
                "schema_version": 1,
                "fixture_id": fixture.fixture_id,
                "chosen_action": "/choose move psychic",
                "value": 0.9,
                "root_values": {
                    "/choose move psychic": 0.9,
                    "/choose move thunderbolt": 0.1,
                },
                "search_depth": 5,
            }
        ),
        encoding="utf-8",
    )

    records, summary = build_training_records(
        [trace],
        search_annotation_paths=[annotation],
    )

    [record] = records
    assert record["provenance"]["behavior_action"] == "/choose move thunderbolt"
    assert record["provenance"]["behavior_matches_policy_target"] is False
    assert record["targets"]["policy"]["selected_action"] == "/choose move psychic"
    assert record["targets"]["value"] == {
        "eventual_battle_outcome": -1.0,
        "public_belief_search_return": 0.9,
    }
    assert summary["search_target_sources"] == {
        "public-belief-search-annotation": 1
    }
