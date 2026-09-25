from __future__ import annotations

import json

from azelficoast.belief.posterior_truth import (
    POSTERIOR_TRUTH_SCHEMA,
    POSTERIOR_TRUTH_SCHEMA_VERSION,
    load_posterior_truth,
    score_control_posterior_truth,
    summarize_posterior_truth,
)


def _truth(path, *, world_id: str = "world-b"):
    document = {
        "schema": POSTERIOR_TRUTH_SCHEMA,
        "schema_version": POSTERIOR_TRUTH_SCHEMA_VERSION,
        "replay_id": "replay",
        "showdown_commit": "pinned",
        "truth_scope": "post-hoc",
        "sides": {
            "p1": [
                {
                    "decision_index": 0,
                    "opponent_hidden_world_id": world_id,
                    "opponent_hidden": {"secret": "not-for-policy"},
                }
            ],
            "p2": [],
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _control():
    return {
        "decision_index": 0,
        "source": {
            "kind": "public-showdown-replay",
            "replay_id": "replay",
            "side": "p1",
        },
    }


def _posterior():
    return {
        "worlds": [
            {"world_id": "world-a", "weight": 0.25, "hidden": {}},
            {"world_id": "world-b", "weight": 0.75, "hidden": {}},
        ]
    }


def test_truth_loader_discards_hidden_payload_from_runtime_index(tmp_path) -> None:
    path = tmp_path / "truth.json"
    _truth(path)

    truth = load_posterior_truth([path])

    assert truth[("replay", "p1", 0)] == {
        "replay_id": "replay",
        "side": "p1",
        "decision_index": 0,
        "opponent_hidden_world_id": "world-b",
    }


def test_truth_scores_only_matching_public_replay_controls(tmp_path) -> None:
    path = tmp_path / "truth.json"
    _truth(path)
    truth = load_posterior_truth([path])

    score = score_control_posterior_truth(_control(), _posterior(), truth=truth)

    assert score is not None
    assert score["realized_state_in_support"] is True
    assert score["assigned_mass"] == 0.75


def test_truth_summary_fails_gate_on_any_realized_world_omission(tmp_path) -> None:
    path = tmp_path / "truth.json"
    _truth(path, world_id="missing")
    truth = load_posterior_truth([path])

    score = score_control_posterior_truth(_control(), _posterior(), truth=truth)
    assert score is not None
    summary = summarize_posterior_truth([score])

    assert summary["status"] == "evaluated"
    assert summary["omitted_realized_state_count"] == 1
    assert summary["gate_passed"] is False


def test_absent_truth_does_not_make_a_validity_claim() -> None:
    summary = summarize_posterior_truth([])

    assert summary["status"] == "unavailable"
    assert summary["evaluated_decision_count"] == 0
    assert summary["gate_passed"] is True
