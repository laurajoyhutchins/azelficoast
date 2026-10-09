from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELTA_MODULE = ROOT / "showdown" / "runtime" / "transition_successor_delta.cjs"


def _node_delta(before: object, after: object, current: object) -> dict[str, object]:
    script = r"""
const {applySuccessorDelta, successorDelta} = require(process.argv[1]);
const before = JSON.parse(process.argv[2]);
const after = JSON.parse(process.argv[3]);
const current = JSON.parse(process.argv[4]);
const delta = successorDelta(before, after);
process.stdout.write(JSON.stringify({
  delta,
  hydrated: applySuccessorDelta(current, delta),
  before,
  current,
}));
"""
    completed = subprocess.run(
        [
            "node",
            "-e",
            script,
            str(DELTA_MODULE),
            json.dumps(before, sort_keys=True),
            json.dumps(after, sort_keys=True),
            json.dumps(current, sort_keys=True),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def test_successor_delta_carries_unmodified_current_root_leaves() -> None:
    before = {
        "turn": 8,
        "p1": [
            {"species": "rotomwash", "hp": 100, "status": None},
            {"species": "dragonite", "hp": 120, "status": None},
        ],
        "weather": None,
    }
    after = {
        "turn": 9,
        "p1": [
            {"species": "rotomwash", "hp": 63, "status": None},
            {"species": "dragonite", "hp": 120, "status": None},
        ],
        "weather": None,
    }
    current = {
        "turn": 8,
        "p1": [
            {"species": "rotomwash", "hp": 100, "status": None},
            {"species": "dragonite", "hp": 77, "status": "brn"},
        ],
        "weather": None,
    }

    result = _node_delta(before, after, current)

    assert result["hydrated"] == {
        "turn": 9,
        "p1": [
            {"species": "rotomwash", "hp": 63, "status": None},
            {"species": "dragonite", "hp": 77, "status": "brn"},
        ],
        "weather": None,
    }
    assert result["before"] == before
    assert result["current"] == current


def test_successor_delta_replaces_structurally_changed_arrays() -> None:
    before = {"volatiles": ["protect"], "winner": None}
    after = {"volatiles": ["protect", "substitute"], "winner": "p1"}
    current = {"volatiles": ["protect"], "winner": None, "unrelated": 3}

    result = _node_delta(before, after, current)

    assert result["hydrated"] == {
        "volatiles": ["protect", "substitute"],
        "winner": "p1",
        "unrelated": 3,
    }
