from __future__ import annotations

import json

import pytest

from azelficoast.corpus import (
    CorpusConflictError,
    FirstLegalPolicy,
    RecordedModePolicy,
    build_fixtures,
    evaluate_fixtures,
    load_corpus,
    write_corpus,
)
from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION


def _record(
    *,
    run: str,
    event: int,
    kind: str,
    **extra: object,
) -> dict[str, object]:
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": run,
        "event_index": event,
        "observed_at": "2026-09-24T00:00:00+00:00",
        "kind": kind,
        **extra,
    }


def _state() -> dict[str, object]:
    return {
        "battle_tag": "battle-gen9randombattle-test",
        "turn": 4,
        "player": "azelficoast",
        "opponent": "opponent",
        "active": {"species": "Azelf", "item": "lifeorb"},
        "opponent_active": {"species": "Corviknight", "item": None},
        "team": {},
        "opponent_team": {},
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "available_moves": ["thunderbolt", "psychic"],
        "available_switches": [],
        "legal_actions": [
            "/choose move thunderbolt",
            "/choose move psychic",
        ],
        "force_switch": False,
        "trapped": False,
        "can_tera": False,
    }


def _write_trace(path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_fixture_freezes_only_protocol_observed_before_decision(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                run="r",
                event=0,
                kind="protocol",
                room="battle-gen9randombattle-test",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                run="r",
                event=1,
                kind="decision",
                battle_tag="battle-gen9randombattle-test",
                decision_index=0,
                state=_state(),
                chosen_action="/choose move thunderbolt",
            ),
            _record(
                run="r",
                event=2,
                kind="protocol",
                room="battle-gen9randombattle-test",
                protocol_index=1,
                messages=[["", "move", "p2a: Corviknight", "Roost"]],
            ),
        ],
    )

    [fixture] = build_fixtures([trace])

    assert fixture.protocol_prefix == ((("", "turn", "4"),),)
    fields = [
        field
        for batch in fixture.protocol_prefix
        for message in batch
        for field in message
    ]
    assert "Roost" not in fields
    assert "battle_tag" not in fixture.state


def test_control_action_is_evidence_not_fixture_identity(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    common_protocol = dict(
        kind="protocol",
        room="battle-gen9randombattle-test",
        protocol_index=0,
        messages=[["", "turn", "4"]],
    )
    _write_trace(
        trace,
        [
            _record(run="a", event=0, **common_protocol),
            _record(
                run="a",
                event=1,
                kind="decision",
                battle_tag="battle-gen9randombattle-test",
                decision_index=0,
                state=_state(),
                chosen_action="/choose move thunderbolt",
            ),
            _record(run="b", event=0, **common_protocol),
            _record(
                run="b",
                event=1,
                kind="decision",
                battle_tag="battle-gen9randombattle-test",
                decision_index=0,
                state=_state(),
                chosen_action="/choose move psychic",
            ),
        ],
    )

    [fixture] = build_fixtures([trace])

    assert {c["chosen_action"] for c in fixture.control_decisions} == {
        "/choose move thunderbolt",
        "/choose move psychic",
    }


def test_transport_timestamps_do_not_change_fixture_identity(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    records = []
    for run, timestamp in (("a", "100"), ("b", "200")):
        records.extend(
            [
                _record(
                    run=run,
                    event=0,
                    kind="protocol",
                    room="battle-gen9randombattle-test",
                    protocol_index=0,
                    messages=[
                        ["", "t:", timestamp],
                        ["", "turn", "4"],
                    ],
                ),
                _record(
                    run=run,
                    event=1,
                    kind="decision",
                    battle_tag="battle-gen9randombattle-test",
                    decision_index=0,
                    state=_state(),
                    chosen_action="/choose move thunderbolt",
                ),
            ]
        )
    _write_trace(trace, records)

    [fixture] = build_fixtures([trace])

    assert len(fixture.control_decisions) == 2
    fields = [
        field
        for batch in fixture.protocol_prefix
        for message in batch
        for field in message
    ]
    assert "100" not in fields
    assert "200" not in fields


def test_public_history_is_part_of_fixture_identity(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    records = []
    for run, move in (("a", "Roost"), ("b", "U-turn")):
        records.extend(
            [
                _record(
                    run=run,
                    event=0,
                    kind="protocol",
                    room="battle-gen9randombattle-test",
                    protocol_index=0,
                    messages=[["", "move", "p2a: Corviknight", move]],
                ),
                _record(
                    run=run,
                    event=1,
                    kind="decision",
                    battle_tag="battle-gen9randombattle-test",
                    decision_index=0,
                    state=_state(),
                    chosen_action="/choose move thunderbolt",
                ),
            ]
        )
    _write_trace(trace, records)

    fixtures = build_fixtures([trace])

    assert len(fixtures) == 2
    assert fixtures[0].fixture_id != fixtures[1].fixture_id


def test_corpus_is_deterministic_and_refuses_different_overwrite(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                run="r",
                event=0,
                kind="protocol",
                room="battle-gen9randombattle-test",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                run="r",
                event=1,
                kind="decision",
                battle_tag="battle-gen9randombattle-test",
                decision_index=0,
                state=_state(),
                chosen_action="/choose move thunderbolt",
            ),
        ],
    )
    fixtures = build_fixtures([trace])
    corpus = tmp_path / "corpus.jsonl"

    first_digest = write_corpus(fixtures, corpus)
    second_digest = write_corpus(fixtures, corpus)

    assert first_digest == second_digest
    assert load_corpus(corpus)[0].fixture_id == fixtures[0].fixture_id

    changed = list(fixtures)
    changed[0] = type(fixtures[0])(
        fixture_id=fixtures[0].fixture_id,
        state={**fixtures[0].state, "turn": 5},
        protocol_prefix=fixtures[0].protocol_prefix,
        control_decisions=fixtures[0].control_decisions,
    )
    with pytest.raises(CorpusConflictError):
        write_corpus(changed, corpus)


def test_builtin_policies_share_one_frozen_fixture_interface(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _write_trace(
        trace,
        [
            _record(
                run="r",
                event=0,
                kind="protocol",
                room="battle-gen9randombattle-test",
                protocol_index=0,
                messages=[["", "turn", "4"]],
            ),
            _record(
                run="r",
                event=1,
                kind="decision",
                battle_tag="battle-gen9randombattle-test",
                decision_index=0,
                state=_state(),
                chosen_action="/choose move psychic",
            ),
        ],
    )
    fixtures = build_fixtures([trace])

    first_summary, first_rows = evaluate_fixtures(fixtures, FirstLegalPolicy())
    control_summary, control_rows = evaluate_fixtures(fixtures, RecordedModePolicy())

    assert first_rows[0]["chosen_action"] == "/choose move thunderbolt"
    assert first_summary["control_action_agreement"] == 0
    assert control_rows[0]["chosen_action"] == "/choose move psychic"
    assert control_summary["control_action_agreement"] == 1
    assert first_summary["illegal_action_count"] == 0
    assert control_summary["illegal_action_count"] == 0


def test_filtered_fixture_build_matches_full_fixture_semantics(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    protect_state = _state()
    protect_state["legal_actions"] = [
        "/choose move protect",
        "/choose move psychic",
    ]
    reject_state = _state()
    reject_state["turn"] = 5

    _write_trace(
        trace,
        [
            _record(
                run="r",
                event=0,
                kind="protocol",
                room="battle-gen9randombattle-test",
                protocol_index=0,
                messages=[["", "move", "p2a: Corviknight", "Roost"]],
            ),
            _record(
                run="r",
                event=1,
                kind="decision",
                battle_tag="battle-gen9randombattle-test",
                decision_index=0,
                state=protect_state,
                chosen_action="/choose move protect",
            ),
            _record(
                run="r",
                event=2,
                kind="protocol",
                room="battle-gen9randombattle-test",
                protocol_index=1,
                messages=[["", "turn", "5"]],
            ),
            _record(
                run="r",
                event=3,
                kind="decision",
                battle_tag="battle-gen9randombattle-test",
                decision_index=1,
                state=reject_state,
                chosen_action="/choose move thunderbolt",
            ),
        ],
    )

    full = build_fixtures([trace])
    filtered = build_fixtures(
        [trace],
        decision_predicate=lambda state: any(
            str(action).startswith("/choose move protect")
            for action in state.get("legal_actions", ())
        ),
    )

    expected = [
        fixture
        for fixture in full
        if any(
            action.startswith("/choose move protect")
            for action in fixture.legal_actions
        )
    ]
    assert filtered == expected
