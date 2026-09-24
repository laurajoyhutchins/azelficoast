from __future__ import annotations

import json
from types import SimpleNamespace

from azelficoast.instrumentation import (
    TRACE_SCHEMA,
    TRACE_SCHEMA_VERSION,
    DecisionTraceWriter,
    battle_view,
)


def _pokemon(species: str, *, opponent: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        level=80,
        active=True,
        fainted=False,
        current_hp=150,
        max_hp=200,
        current_hp_fraction=0.75,
        base_stats={"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        stats={"hp": 200, "atk": 120, "def": 130, "spa": 110, "spd": 120, "spe": 90},
        status=None,
        item="unknown_item" if opponent else "leftovers",
        ability=None if opponent else "pressure",
        moves={"protect": object()} if opponent else {"recover": object()},
        boosts={"atk": 0, "def": 1},
        types=[SimpleNamespace(name="NORMAL")],
        tera_type=None,
    )


def _battle() -> SimpleNamespace:
    active = _pokemon("Snorlax")
    opponent = _pokemon("Ditto", opponent=True)
    return SimpleNamespace(
        battle_tag="battle-gen9randombattle-test",
        turn=7,
        player_username="azelficoast",
        opponent_username="opponent",
        active_pokemon=active,
        opponent_active_pokemon=opponent,
        team={"p1a: Snorlax": active},
        opponent_team={"p2a: Ditto": opponent},
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        available_moves=[SimpleNamespace(id="recover")],
        available_switches=[],
        valid_orders=[SimpleNamespace(message="/choose move recover")],
        force_switch=False,
        trapped=False,
        can_tera=True,
        won=True,
        lost=False,
        finished=True,
    )


def test_battle_view_contains_decision_information_without_inventing_hidden_state() -> None:
    view = battle_view(_battle())

    assert view["turn"] == 7
    assert view["legal_actions"] == ["/choose move recover"]
    assert view["active"]["item"] == "leftovers"
    assert view["active"]["level"] == 80
    assert view["active"]["transformed"] is False
    assert view["active"]["current_hp"] == 150
    assert view["active"]["max_hp"] == 200
    assert view["active"]["stats"]["spe"] == 90
    assert view["opponent_active"]["item"] is None
    assert _pokemon("Ditto", opponent=True).item == "unknown_item"
    assert view["opponent_active"]["ability"] is None
    assert view["opponent_active"]["moves"] == ["protect"]


def test_trace_writer_records_protocol_decision_and_terminal_events(tmp_path) -> None:
    trace = tmp_path / "decisions.jsonl"
    writer = DecisionTraceWriter(trace, run_id="test-run")
    battle = _battle()

    writer.record_protocol_batch(
        [[">battle-gen9randombattle-test"], ["", "move", "p2a: Ditto", "Protect"]]
    )
    writer.record_decision(battle, SimpleNamespace(message="/choose move recover"))
    writer.record_terminal(battle)

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [record["kind"] for record in records] == ["protocol", "decision", "terminal"]
    assert [record["event_index"] for record in records] == [0, 1, 2]
    assert all(record["run_id"] == "test-run" for record in records)
    assert all(record["schema"] == TRACE_SCHEMA for record in records)
    assert all(record["schema_version"] == TRACE_SCHEMA_VERSION for record in records)
    assert records[0]["protocol_index"] == 0
    assert records[1]["decision_index"] == 0
    assert records[1]["chosen_action"] == "/choose move recover"
    assert records[2]["tied"] is False
    assert all("observed_at" in record for record in records)


def test_distinct_writers_get_distinct_run_ids_when_appending_to_one_file(tmp_path) -> None:
    trace = tmp_path / "decisions.jsonl"
    first = DecisionTraceWriter(trace)
    second = DecisionTraceWriter(trace)

    first.record_protocol_batch([[">battle-one"], ["", "turn", "1"]])
    second.record_protocol_batch([[">battle-two"], ["", "turn", "1"]])

    records = [json.loads(line) for line in trace.read_text().splitlines()]
    assert records[0]["run_id"] != records[1]["run_id"]
    assert records[0]["event_index"] == 0
    assert records[1]["event_index"] == 0


def test_terminal_tie_is_derived_from_finished_without_calling_tied_method(tmp_path) -> None:
    battle = _battle()
    battle.won = None
    battle.lost = None
    trace = tmp_path / "decisions.jsonl"

    DecisionTraceWriter(trace).record_terminal(battle)

    record = json.loads(trace.read_text())
    assert record["tied"] is True
