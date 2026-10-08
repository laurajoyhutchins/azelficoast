from __future__ import annotations

import json
from pathlib import Path

from azelficoast.research.source_progress import TraceProgress, observe_generation


def test_progress_tails_complete_records_once_and_omits_outcomes(tmp_path: Path) -> None:
    trace = tmp_path / "decisions.jsonl"
    progress = TraceProgress(trace)
    assert progress.poll(0)["decisions"] == 0
    decision = json.dumps(
        {
            "kind": "decision",
            "chosen_action": "SECRET",
            "decision_metadata": {"belief": {"reason": "belief-search-timeout"}},
        }
    ).encode()
    trace.write_bytes(decision[:30])
    assert progress.poll(60)["decisions"] == 0
    with trace.open("ab") as handle:
        handle.write(decision[30:] + b"\n")
        handle.write(b'{"kind":"terminal","battle_tag":"one","won":true}\n')
        handle.write(b'{"kind":"terminal","battle_tag":"one","won":true}\n')
    before = trace.read_bytes()
    result = progress.poll(120)
    assert result["decisions"] == 1
    assert result["finished_battles"] == 1
    assert result["policy_reasons"] == {"belief-search-timeout": 1}
    assert result["seconds_since_trace_activity"] == 0
    again = progress.poll(180)
    assert again["decisions"] == 1
    assert again["finished_battles"] == 1
    assert again["seconds_since_trace_activity"] == 60
    assert "won" not in json.dumps(result)
    assert "SECRET" not in json.dumps(result)
    assert trace.read_bytes() == before


def test_observer_finishes_and_writes_final_counts_on_generation_error(tmp_path: Path) -> None:
    trace = tmp_path / "decisions.jsonl"
    output = tmp_path / "progress.jsonl"
    try:
        with observe_generation(trace, output, interval_seconds=60):
            trace.write_text('{"kind":"terminal","battle_tag":"one"}\n')
            raise ValueError("generation failed")
    except ValueError:
        pass
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[-1]["finished_battles"] == 1


def test_observer_output_failure_does_not_interrupt_generation(tmp_path: Path) -> None:
    with observe_generation(tmp_path / "trace", tmp_path, interval_seconds=60):
        pass
