"""Measure live posterior reconstruction, exact-search readiness, and observed routing."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from azelficoast.corpus import DecisionFixture, build_fixtures
from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION
from azelficoast.live.belief import build_probe_source, opponent_move_from_protocol

REPORT_SCHEMA = "azelficoast.live-belief-coverage"
REPORT_SCHEMA_VERSION = 1
ADMITTED = "admitted"


class BeliefCoverageError(ValueError):
    """Raised when trace evidence cannot be summarized safely."""


def _decision_weight(fixture: DecisionFixture) -> int:
    return max(1, len(fixture.control_decisions))


def _ranked(counter: Counter[str]) -> list[dict[str, Any]]:
    return [
        {"reason": reason, "count": count}
        for reason, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]


def _species_ranked(
    counts: Mapping[str, Counter[str]],
) -> dict[str, list[dict[str, Any]]]:
    return {
        reason: [
            {"opponent_species": species, "count": count}
            for species, count in sorted(
                species_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ]
        for reason, species_counts in sorted(counts.items())
    }


def summarize_fixtures(fixtures: Iterable[DecisionFixture]) -> dict[str, Any]:
    """Estimate current live admission coverage over frozen decision fixtures.

    Counts are weighted by the number of recorded control decisions represented by
    each immutable fixture. This preserves repeated-decision frequency after corpus
    deduplication.
    """

    fixture_list = list(fixtures)
    reason_decisions: Counter[str] = Counter()
    reason_fixtures: Counter[str] = Counter()
    opponent_species_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    opponent_move_by_reason: dict[str, Counter[str]] = defaultdict(Counter)
    decision_count = 0

    for fixture in fixture_list:
        source, reason = build_probe_source(fixture)
        category = ADMITTED if source is not None else reason
        weight = _decision_weight(fixture)
        decision_count += weight
        reason_decisions[category] += weight
        reason_fixtures[category] += 1

        opponent = fixture.state.get("opponent_active")
        species = (
            str(opponent.get("species") or "<unknown>")
            if isinstance(opponent, Mapping)
            else "<unknown>"
        )
        opponent_species_by_reason[category][species] += weight
        move = opponent_move_from_protocol(fixture) or "<unresolved>"
        opponent_move_by_reason[category][move] += weight

    admitted = reason_decisions[ADMITTED]
    fallback = decision_count - admitted
    fallback_reasons = Counter(reason_decisions)
    fallback_reasons.pop(ADMITTED, None)

    exact_search_ready = 0
    exact_search_blockers: Counter[str] = Counter()
    for fixture in fixture_list:
        source, reason = build_probe_source(fixture)
        weight = _decision_weight(fixture)
        if source is None:
            exact_search_blockers[f"reconstruction:{reason}"] += weight
        elif isinstance(source.get("opponent_response_move"), str):
            exact_search_ready += weight
        else:
            exact_search_blockers["opponent-model-unavailable"] += weight

    return {
        "decision_count": decision_count,
        "unique_fixture_count": len(fixture_list),
        "admitted_decision_count": admitted,
        "fallback_decision_count": fallback,
        "admission_rate": (admitted / decision_count) if decision_count else None,
        "exact_search_ready_decision_count": exact_search_ready,
        "exact_search_blocked_decision_count": decision_count - exact_search_ready,
        "exact_search_ready_rate": (
            exact_search_ready / decision_count if decision_count else None
        ),
        "exact_search_blocker_counts": _ranked(exact_search_blockers),
        "decision_reason_counts": _ranked(reason_decisions),
        "fixture_reason_counts": _ranked(reason_fixtures),
        "fallback_reason_counts": _ranked(fallback_reasons),
        "opponent_species_by_reason": _species_ranked(opponent_species_by_reason),
        "opponent_move_by_reason": {
            reason: [
                {"opponent_move": move, "count": count}
                for move, count in sorted(
                    move_counts.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ]
            for reason, move_counts in sorted(opponent_move_by_reason.items())
        },
    }


def _observed_routing(trace_paths: Sequence[str | Path]) -> dict[str, Any]:
    selected: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    routed = 0
    decisions = 0

    for raw_path in trace_paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise BeliefCoverageError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if record.get("schema") != TRACE_SCHEMA:
                    raise BeliefCoverageError(
                        f"{path}:{line_number}: unexpected trace schema"
                    )
                if record.get("schema_version") != TRACE_SCHEMA_VERSION:
                    raise BeliefCoverageError(
                        f"{path}:{line_number}: unsupported trace schema version"
                    )
                if record.get("kind") != "decision":
                    continue

                decisions += 1
                metadata = record.get("decision_metadata")
                if not isinstance(metadata, Mapping):
                    continue
                policy = metadata.get("selected_policy")
                belief = metadata.get("belief")
                if isinstance(policy, str):
                    selected[policy] += 1
                    routed += 1
                if isinstance(belief, Mapping) and isinstance(belief.get("reason"), str):
                    reasons[str(belief["reason"])] += 1

    return {
        "decision_count": decisions,
        "routed_decision_count": routed,
        "selected_policy_counts": [
            {"policy": policy, "count": count}
            for policy, count in sorted(
                selected.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ],
        "belief_reason_counts": _ranked(reasons),
        "public_belief_selection_rate": (
            selected["public-belief"] / routed if routed else None
        ),
    }


def summarize_traces(trace_paths: Sequence[str | Path]) -> dict[str, Any]:
    if not trace_paths:
        raise BeliefCoverageError("at least one decision trace is required")
    fixtures = build_fixtures(trace_paths)
    return {
        "schema": REPORT_SCHEMA,
        "schema_version": REPORT_SCHEMA_VERSION,
        "trace_count": len(trace_paths),
        "observed_routing": _observed_routing(trace_paths),
        "static_admission": summarize_fixtures(fixtures),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure bounded public-belief routing and admission coverage."
    )
    parser.add_argument("traces", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(summarize_traces(args.traces), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
