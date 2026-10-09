"""Paired shallow-versus-deeper public-belief experiment endpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

TRACE_SCHEMA = "azelficoast.real-belief-decision-trace"
TRACE_SCHEMA_VERSION = 3
PAIR_SCHEMA = "azelficoast.public-belief-depth-pair"
PAIR_SCHEMA_VERSION = 1


class DepthStudyError(ValueError):
    """Raised when paired depth evidence violates the experimental contract."""


def _root_values(trace: Mapping[str, Any], policy: str) -> dict[str, float]:
    result = trace.get(policy)
    if not isinstance(result, Mapping):
        raise DepthStudyError(f"trace lacks {policy} result")
    raw = result.get("root_values")
    if not isinstance(raw, Mapping) or not raw:
        raise DepthStudyError(f"trace lacks {policy} root values")
    values: dict[str, float] = {}
    for action, value in raw.items():
        if not isinstance(value, (int, float)):
            raise DepthStudyError(f"{policy} root value is not numeric")
        values[str(action)] = float(value)
    return values


def _endpoints(trace: Mapping[str, Any]) -> dict[str, Any]:
    determinization = _root_values(trace, "determinization")
    public = _root_values(trace, "public_belief")
    if set(determinization) != set(public):
        raise DepthStudyError("search methods did not evaluate identical root actions")

    det_result = trace["determinization"]
    public_result = trace["public_belief"]
    det_action = det_result.get("chosen_action")
    public_action = public_result.get("chosen_action")
    if not isinstance(det_action, str) or det_action not in public:
        raise DepthStudyError("determinization chose a non-evaluated root action")
    if not isinstance(public_action, str) or public_action not in public:
        raise DepthStudyError("public-belief search chose a non-evaluated root action")

    gaps = {
        action: determinization[action] - public[action]
        for action in determinization
    }
    max_bias = max(gaps.values())
    public_best = max(public.values())
    regret = public_best - public[det_action]
    if regret < -1e-10:
        raise DepthStudyError("computed determinization regret is negative")

    return {
        "max_strategy_fusion_value_advantage": max_bias,
        "determinization_public_regret": max(0.0, regret),
        "policy_disagreement": det_action != public_action,
        "determinization_action": det_action,
        "public_belief_action": public_action,
        "determinization_root_values": determinization,
        "public_belief_root_values": public,
    }


def summarize_pair(
    shallow: Mapping[str, Any],
    deeper: Mapping[str, Any],
) -> dict[str, Any]:
    for label, trace in (("shallow", shallow), ("deeper", deeper)):
        if (
            trace.get("schema") != TRACE_SCHEMA
            or trace.get("schema_version") != TRACE_SCHEMA_VERSION
        ):
            raise DepthStudyError(f"{label} trace has an unexpected schema")
        if trace.get("experiment_valid") is not True:
            raise DepthStudyError(f"{label} trace is not valid")

    if shallow.get("source_fixture_id") != deeper.get("source_fixture_id"):
        raise DepthStudyError("paired traces reference different fixtures")
    if shallow.get("showdown_commit") != deeper.get("showdown_commit"):
        raise DepthStudyError("paired traces use different Showdown revisions")
    if shallow.get("continuation_decision_horizons", 1) != 1:
        raise DepthStudyError("shallow trace must use one continuation decision horizon")
    if deeper.get("continuation_decision_horizons") != 2:
        raise DepthStudyError("deeper trace must use exactly two continuation decision horizons")

    shallow_endpoints = _endpoints(shallow)
    deeper_endpoints = _endpoints(deeper)
    shallow_actions = set(shallow_endpoints["determinization_root_values"])
    deeper_actions = set(deeper_endpoints["determinization_root_values"])
    if shallow_actions != deeper_actions:
        raise DepthStudyError("paired traces do not evaluate identical root actions")

    return {
        "schema": PAIR_SCHEMA,
        "schema_version": PAIR_SCHEMA_VERSION,
        "source_fixture_id": shallow.get("source_fixture_id"),
        "showdown_commit": shallow.get("showdown_commit"),
        "shallow": shallow_endpoints,
        "deeper": deeper_endpoints,
        "paired_change": {
            "max_strategy_fusion_value_advantage": (
                deeper_endpoints["max_strategy_fusion_value_advantage"]
                - shallow_endpoints["max_strategy_fusion_value_advantage"]
            ),
            "determinization_public_regret": (
                deeper_endpoints["determinization_public_regret"]
                - shallow_endpoints["determinization_public_regret"]
            ),
        },
    }


def _load(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise DepthStudyError(f"{path}: expected a JSON object")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shallow", type=Path)
    parser.add_argument("deeper", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    result = summarize_pair(_load(args.shallow), _load(args.deeper))
    encoded = json.dumps(result, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
