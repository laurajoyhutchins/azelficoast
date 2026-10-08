"""Read-only exact-head canary for training run 37713372447 (artifact 11522812587)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

from azelficoast.core.mechanics import MechanicsExecutionRequest, VerifiedTransitionProgramSet
from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT
from azelficoast.live.belief import build_probe_source, live_fixture
from azelficoast.research.decision_contracts import MechanicsIdentity, parse_belief_artifact

FAILED_ACTION = "/choose switch Mesprit"
ACTIONS = [FAILED_ACTION, "/choose move leafstorm"]


def load_failure(root: Path):
    traces = list(root.rglob("decisions.jsonl"))
    trace = next((p for p in traces if p.stat().st_size > 0), None)
    if trace is None:
        raise ValueError("failure artifact has no decision trace")
    records = [json.loads(row) for row in trace.read_text().splitlines() if row]
    decisions = [
        row for row in records
        if row.get("kind") == "decision"
        and row.get("state", {}).get("legal_actions") == ACTIONS
        and row.get("battle_tag") == "battle-gen9randombattle-22"
        and row.get("event_index") == 387
        and row.get("run_id") == "0e6eea9379234edf9cfde9b23723d081"
    ]
    if len(decisions) != 1:
        raise ValueError(f"expected one Mesprit switch state, found {len(decisions)}")
    decision = decisions[0]
    history = [
        row["messages"] for row in records
        if row.get("kind") == "protocol"
        and row.get("room") == decision["battle_tag"]
        and row.get("run_id") == decision["run_id"]
        and row["event_index"] < decision["event_index"]
    ]
    if not history:
        raise ValueError("failing decision lacks protocol history")
    return live_fixture(decision["state"], history), decision


def probe(showdown: Path, source: dict, mode: str) -> dict:
    script = Path(__file__).resolve().parents[1] / "showdown/runtime/probe_real_belief_trace.cjs"
    with tempfile.TemporaryDirectory(prefix="mesprit-canary-") as temp:
        path = Path(temp) / "source.json"
        path.write_text(json.dumps(source, sort_keys=True))
        process = subprocess.run(
            ["node", str(script), str(showdown), str(path), mode],
            check=False, capture_output=True, text=True, timeout=240,
        )
    if process.returncode:
        raise ValueError(
            f"pinned Showdown probe {mode} failed ({process.returncode}): "
            + process.stderr[-3000:]
        )
    document = json.loads(process.stdout)
    if not isinstance(document, dict):
        raise ValueError("Showdown probe returned a non-object")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--showdown-root", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()

    marker = args.showdown_root / ".azelficoast-showdown-sha"
    if not marker.is_file() or marker.read_text().strip() != PINNED_SHOWDOWN_COMMIT:
        raise ValueError("verified pinned Showdown is unavailable")
    fixture, decision = load_failure(args.artifact_dir)
    if list(fixture.legal_actions) != ACTIONS:
        raise ValueError("original Mesprit action set changed")
    source, reason = build_probe_source(fixture)
    if source is None or reason != "admitted":
        raise ValueError(f"recorded gameplay state was rejected: {reason}")
    posterior = probe(args.showdown_root, source, "--posterior-only")
    program = probe(args.showdown_root, source, "--transition-program-only")
    if program.get("source_fixture_id") != fixture.fixture_id:
        raise ValueError("transition program fixture identity drifted")
    if program.get("legal_actions") != ACTIONS:
        raise ValueError("transition program omitted Mesprit switch")
    if program.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise ValueError("transition program used another Showdown revision")

    belief, index = parse_belief_artifact(posterior)
    mechanics = VerifiedTransitionProgramSet.from_artifact(
        artifact=program,
        identity=MechanicsIdentity.from_showdown_commit(PINNED_SHOWDOWN_COMMIT),
        fixture_id=fixture.fixture_id,
        legal_actions=tuple(ACTIONS),
        belief=belief,
        transport_index=index,
    )
    admitted = mechanics.execute(
        MechanicsExecutionRequest(
            mechanics_identity=mechanics.identity,
            action=FAILED_ACTION,
        )
    ).program.to_record()
    if admitted.get("partition_method") != "dynamic-read-refinement":
        raise ValueError("Mesprit switch was not admitted by the existing method")
    if not admitted.get("classes"):
        raise ValueError("Mesprit switch has no admitted execution class")

    head_sha = os.environ.get("GITHUB_SHA", "")
    checked_out_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if len(head_sha) != 40 or head_sha != checked_out_sha:
        raise ValueError(
            f"canary source identity mismatch: declared={head_sha} checkout={checked_out_sha}"
        )
    receipt = {
        "schema": "azelficoast.mesprit-switch-canary/v1",
        "passed": True,
        "source_training_run_id": "37713372447",
        "source_artifact_id": 11522812587,
        "source_battle_tag": decision["battle_tag"],
        "fixture_id": fixture.fixture_id,
        "candidate_sha": head_sha,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "action": FAILED_ACTION,
        "world_count": len(mechanics.world_ids),
        "class_count": len(admitted["classes"]),
        "transition_program_digest": mechanics.transition_program_digest,
        "mechanics_evidence_digest": mechanics.semantic_evidence_digest,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
