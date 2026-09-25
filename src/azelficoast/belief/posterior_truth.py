"""Post-hoc posterior truth scoring kept outside decision/search inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.belief.validity import aggregate_realized_support, score_realized_support

POSTERIOR_TRUTH_SCHEMA = "azelficoast.public-replay-hidden-truth"
POSTERIOR_TRUTH_SCHEMA_VERSION = 1


class PosteriorTruthError(ValueError):
    """Raised when post-hoc hidden truth evidence is malformed or ambiguous."""


def load_posterior_truth(
    paths: Sequence[str | Path],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Load replay truth sidecars keyed by replay, observing side, and decision."""

    truth: dict[tuple[str, str, int], dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PosteriorTruthError(f"cannot read posterior truth {path}: {error}") from error
        if not isinstance(document, Mapping):
            raise PosteriorTruthError(f"{path}: posterior truth must be an object")
        if (
            document.get("schema") != POSTERIOR_TRUTH_SCHEMA
            or document.get("schema_version") != POSTERIOR_TRUTH_SCHEMA_VERSION
        ):
            raise PosteriorTruthError(f"{path}: unexpected posterior truth schema")
        replay_id = document.get("replay_id")
        sides = document.get("sides")
        if not isinstance(replay_id, str) or not replay_id:
            raise PosteriorTruthError(f"{path}: truth lacks replay identity")
        if not isinstance(sides, Mapping):
            raise PosteriorTruthError(f"{path}: truth lacks side evidence")

        for side in ("p1", "p2"):
            rows = sides.get(side)
            if not isinstance(rows, list):
                raise PosteriorTruthError(f"{path}: truth lacks {side} decision rows")
            for row in rows:
                if not isinstance(row, Mapping):
                    raise PosteriorTruthError(f"{path}: {side} truth row is not an object")
                index = row.get("decision_index")
                world_id = row.get("opponent_hidden_world_id")
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    or not isinstance(world_id, str)
                    or not world_id
                ):
                    raise PosteriorTruthError(f"{path}: malformed {side} truth row")
                key = (replay_id, side, index)
                record = {
                    "replay_id": replay_id,
                    "side": side,
                    "decision_index": index,
                    "opponent_hidden_world_id": world_id,
                }
                prior = truth.get(key)
                if prior is not None and prior != record:
                    raise PosteriorTruthError(f"conflicting posterior truth for {key!r}")
                truth[key] = record
    return truth


def score_control_posterior_truth(
    control: Mapping[str, Any],
    posterior: Mapping[str, Any],
    *,
    truth: Mapping[tuple[str, str, int], Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Score a posterior only after it has been constructed from public evidence."""

    source = control.get("source")
    decision_index = control.get("decision_index")
    if not isinstance(source, Mapping):
        return None
    if source.get("kind") != "public-showdown-replay":
        return None
    replay_id = source.get("replay_id")
    side = source.get("side")
    if (
        not isinstance(replay_id, str)
        or not isinstance(side, str)
        or side not in {"p1", "p2"}
        or not isinstance(decision_index, int)
        or isinstance(decision_index, bool)
    ):
        raise PosteriorTruthError("public replay control has malformed truth identity")
    record = truth.get((replay_id, side, decision_index))
    if record is None:
        return None
    scored = score_realized_support(
        posterior,
        realized_world_id=str(record["opponent_hidden_world_id"]),
    )
    return {
        "replay_id": replay_id,
        "side": side,
        "decision_index": decision_index,
        **scored,
    }


def summarize_posterior_truth(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate truth scoring for a teacher run without inventing missing evidence."""

    if not rows:
        return {
            "status": "unavailable",
            "evaluated_decision_count": 0,
            "gate_passed": True,
            "gate_rule": "no truth evidence does not assert posterior validity",
        }
    aggregate = aggregate_realized_support(rows)
    omitted = int(aggregate["omitted_realized_state_count"])
    return {
        "status": "evaluated",
        "evaluated_decision_count": len(rows),
        **aggregate,
        "gate_passed": omitted == 0,
        "gate_rule": "every truth-evaluable teacher posterior must contain the realized hidden world",
    }
