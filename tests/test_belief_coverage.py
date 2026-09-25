from __future__ import annotations

import json

from azelficoast.belief.coverage import summarize_traces
from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION


def _record(
    *,
    event: int,
    kind: str,
    run: str = "coverage-run",
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


def _state(*, item=None, last_move="U-turn") -> dict[str, object]:
    return {
        "battle_tag": "battle-coverage",
        "turn": 8,
        "player": "Azelficoast",
        "opponent": "Rival",
        "active": {
            "species": "Tinkaton",
            "tera_type": "Steel",
        },
        "opponent_active": {
            "species": "Zapdos-Galar",
            "level": 77,
            "item": item,
        },
        "team": {"p1: Tinkaton": {"species": "Tinkaton"}},
        "opponent_team": {"p2: Zapdos-Galar": {"species": "Zapdos-Galar"}},
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "available_moves": ["protect"],
        "available_switches": [],
        "legal_actions": ["/choose move protect"],
        "force_switch": False,
        "trapped": False,
        "can_tera": True,
        "_test_last_move": last_move,
    }


def _write_trace(
    path,
    *,
    item=None,
    selected_policy="public-belief",
    run="coverage-run",
) -> None:
    state = _state(item=item)
    records = [
        _record(
            run=run,
            event=0,
            kind="protocol",
            room="battle-coverage",
            protocol_index=0,
            messages=[
                ["", "player", "p1", "Azelficoast"],
                ["", "player", "p2", "Rival"],
                ["", "move", "p2a: Zapdos-Galar", state["_test_last_move"]],
            ],
        ),
        _record(
            run=run,
            event=1,
            kind="decision",
            battle_tag="battle-coverage",
            decision_index=0,
            state={key: value for key, value in state.items() if key != "_test_last_move"},
            chosen_action="/choose move protect",
            decision_metadata={
                "selected_policy": selected_policy,
                "belief": {
                    "status": "selected" if selected_policy == "public-belief" else "fallback",
                    "reason": (
                        "bounded-public-belief"
                        if selected_policy == "public-belief"
                        else "opponent-item-known"
                    ),
                    "action": (
                        "/choose move protect"
                        if selected_policy == "public-belief"
                        else None
                    ),
                    "diagnostics": {},
                },
            },
        ),
    ]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def test_coverage_reports_static_admission_and_observed_routing(tmp_path) -> None:
    admitted = tmp_path / "admitted.jsonl"
    rejected = tmp_path / "rejected.jsonl"
    _write_trace(admitted, run="admitted-run")
    _write_trace(
        rejected,
        item="leftovers",
        selected_policy="simple-heuristics",
        run="rejected-run",
    )

    report = summarize_traces([admitted, rejected])

    assert report["observed_routing"]["decision_count"] == 2
    assert report["observed_routing"]["routed_decision_count"] == 2
    assert report["observed_routing"]["public_belief_selection_rate"] == 0.5
    assert report["observed_routing"]["selected_policy_counts"] == [
        {"policy": "public-belief", "count": 1},
        {"policy": "simple-heuristics", "count": 1},
    ]

    admission = report["static_admission"]
    assert admission["decision_count"] == 2
    assert admission["admitted_decision_count"] == 1
    assert admission["fallback_decision_count"] == 1
    assert admission["admission_rate"] == 0.5
    assert admission["fallback_reason_counts"] == [
        {"reason": "opponent-item-known", "count": 1}
    ]


def test_coverage_handles_legacy_trace_without_decision_metadata(tmp_path) -> None:
    trace = tmp_path / "legacy.jsonl"
    state = _state()
    records = [
        _record(
            event=0,
            kind="protocol",
            room="battle-coverage",
            protocol_index=0,
            messages=[
                ["", "player", "p1", "Azelficoast"],
                ["", "player", "p2", "Rival"],
                ["", "move", "p2a: Zapdos-Galar", "U-turn"],
            ],
        ),
        _record(
            event=1,
            kind="decision",
            battle_tag="battle-coverage",
            decision_index=0,
            state={key: value for key, value in state.items() if key != "_test_last_move"},
            chosen_action="/choose move protect",
        ),
    ]
    trace.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )

    report = summarize_traces([trace])

    assert report["observed_routing"]["decision_count"] == 1
    assert report["observed_routing"]["routed_decision_count"] == 0
    assert report["observed_routing"]["public_belief_selection_rate"] is None
    assert report["static_admission"]["admitted_decision_count"] == 1
