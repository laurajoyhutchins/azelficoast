"""Generator-conditioned hidden-item prior experiment for status-move observations."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SELECTION_SCHEMA = "azelficoast.status-move-prior-selection"
SELECTION_SCHEMA_VERSION = 1
SAMPLE_SCHEMA = "azelficoast.showdown-world-sample"
RESULT_SCHEMA = "azelficoast.status-move-prior-results"
RESULT_SCHEMA_VERSION = 1


class StatusMovePriorError(ValueError):
    """Raised when status-move prior evidence is malformed or inconsistent."""


def _weights(sample: Mapping[str, Any]) -> dict[str, float]:
    raw = sample.get("item_weights")
    if not isinstance(raw, Mapping) or not raw:
        raise StatusMovePriorError("world sample lacks a nonempty item-weight map")
    weights = {str(item): float(weight) for item, weight in raw.items()}
    if any(weight <= 0 for weight in weights.values()):
        raise StatusMovePriorError("item weights must be positive")
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise StatusMovePriorError("item weights must sum to one")
    return weights


def compare_item_samples(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    minimum_matches: int,
    maximum_total_variation: float,
) -> dict[str, Any]:
    """Compare two disjoint deterministic generator samples for one treatment."""

    for sample in (left, right):
        if sample.get("schema") != SAMPLE_SCHEMA or sample.get("schema_version") != 1:
            raise StatusMovePriorError("unexpected world-sample schema")
    if left.get("requested_species") != right.get("requested_species"):
        raise StatusMovePriorError("sample species differ")
    if left.get("observed_moves") != right.get("observed_moves"):
        raise StatusMovePriorError("sample observed moves differ")
    if left.get("showdown_commit") != right.get("showdown_commit"):
        raise StatusMovePriorError("sample Showdown revisions differ")

    left_weights = _weights(left)
    right_weights = _weights(right)
    support = sorted(set(left_weights) | set(right_weights))
    total_variation = 0.5 * sum(
        abs(left_weights.get(item, 0.0) - right_weights.get(item, 0.0))
        for item in support
    )
    left_support = sorted(left_weights)
    right_support = sorted(right_weights)
    support_stable = left_support == right_support
    left_matched = int(left.get("matched", 0))
    right_matched = int(right.get("matched", 0))

    return {
        "requested_species": left.get("requested_species"),
        "observed_moves": list(left.get("observed_moves", ())),
        "showdown_commit": left.get("showdown_commit"),
        "left": {
            "seed_offset": left.get("seed_offset"),
            "rounds": left.get("rounds"),
            "matched": left_matched,
            "item_weights": left_weights,
        },
        "right": {
            "seed_offset": right.get("seed_offset"),
            "rounds": right.get("rounds"),
            "matched": right_matched,
            "item_weights": right_weights,
        },
        "item_support": support,
        "item_support_size": len(support),
        "support_stable": support_stable,
        "total_variation": total_variation,
        "sufficient_matches": min(left_matched, right_matched) >= minimum_matches,
        "passed": (
            support_stable
            and min(left_matched, right_matched) >= minimum_matches
            and total_variation <= maximum_total_variation
        ),
    }


def _sample(
    *,
    showdown_root: Path,
    species: str,
    observed_moves: Sequence[str],
    rounds: int,
    is_lead: bool,
    seed_offset: int,
) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "scripts" / "sample_showdown_worlds.cjs"
    completed = subprocess.run(
        [
            "node",
            str(script),
            str(showdown_root),
            species,
            ",".join(observed_moves),
            str(rounds),
            "true" if is_lead else "false",
            str(seed_offset),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(completed.stdout)
    if not isinstance(document, dict):
        raise StatusMovePriorError("world sampler returned a non-object")
    return document


def run_experiment(
    selection: Mapping[str, Any],
    *,
    showdown_root: str | Path,
) -> dict[str, Any]:
    if (
        selection.get("schema") != SELECTION_SCHEMA
        or selection.get("schema_version") != SELECTION_SCHEMA_VERSION
    ):
        raise StatusMovePriorError("unexpected selection schema")

    confirmation = selection.get("confirmation")
    treatments = selection.get("treatments")
    if not isinstance(confirmation, Mapping) or not isinstance(treatments, list):
        raise StatusMovePriorError("selection lacks confirmation or treatments")

    rounds = int(confirmation["rounds_per_family"])
    offsets = confirmation.get("seed_offsets")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(offset, int) for offset in offsets)
    ):
        raise StatusMovePriorError("confirmation requires exactly two integer seed offsets")
    minimum_matches = int(confirmation["minimum_matches_per_family"])
    maximum_total_variation = float(confirmation["maximum_total_variation"])

    cases: list[dict[str, Any]] = []
    root = Path(showdown_root)
    for treatment in treatments:
        if not isinstance(treatment, Mapping):
            raise StatusMovePriorError("treatment must be an object")
        species = str(treatment["species"])
        observed_moves = [str(move) for move in treatment["observed_moves"]]
        is_lead = bool(treatment["is_lead"])
        samples = [
            _sample(
                showdown_root=root,
                species=species,
                observed_moves=observed_moves,
                rounds=rounds,
                is_lead=is_lead,
                seed_offset=offset,
            )
            for offset in offsets
        ]
        comparison = compare_item_samples(
            samples[0],
            samples[1],
            minimum_matches=minimum_matches,
            maximum_total_variation=maximum_total_variation,
        )
        cases.append(
            {
                "move": treatment["move"],
                "species": species,
                "observed_moves": observed_moves,
                "is_lead": is_lead,
                "discovery_decision_count": treatment["discovery_decision_count"],
                **comparison,
            }
        )

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "source": selection.get("source"),
        "hypothesis": selection.get("hypothesis"),
        "confirmation": dict(confirmation),
        "case_count": len(cases),
        "passed_case_count": sum(bool(case["passed"]) for case in cases),
        "passed": bool(cases) and all(bool(case["passed"]) for case in cases),
        "cases": cases,
        "non_claims": selection.get("non_claims", []),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection", type=Path)
    parser.add_argument("--showdown-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    result = run_experiment(selection, showdown_root=args.showdown_root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
