#!/usr/bin/env python3
"""Falsifiable policy-boundary refinement experiment over exact public-belief traces.

The treatment leaves public observation cells unresolved under the theorem-safe
material-utility interval [-1, 6] and evaluates cells only until one root action
is certified. A seeded random-refinement control uses the same bounds and cells.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

UTILITY_MIN = -1.0
UTILITY_MAX = 6.0
RANDOM_SEEDS = 256
EPS = 1e-12


def _load_case(path: Path) -> tuple[Mapping[str, Any], dict[str, list[tuple[str, float, float]]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    actions: dict[str, list[tuple[str, float, float]]] = {}

    for action_record in document["actions"]:
        action = str(action_record["action"])
        masses = {
            str(row["observation_hash"]): float(row["probability"])
            for row in action_record["successor_beliefs"]
        }
        values = {
            str(row["observation_hash"]): float(row["value"])
            for row in action_record["public_belief_continuations"]
        }
        if masses.keys() != values.keys():
            raise ValueError(f"{action}: successor and continuation observation sets differ")

        cells = [(key, masses[key], values[key]) for key in masses]
        if abs(sum(probability for _, probability, _ in cells) - 1.0) > 1e-9:
            raise ValueError(f"{action}: successor probability mass does not sum to one")
        if any(
            value < UTILITY_MIN - EPS or value > UTILITY_MAX + EPS
            for _, _, value in cells
        ):
            raise ValueError(f"{action}: continuation utility lies outside safe bounds")
        reconstructed = sum(probability * value for _, probability, value in cells)
        expected = float(document["public_belief"]["root_values"][action])
        if abs(reconstructed - expected) > 1e-9:
            raise ValueError(
                f"{action}: reconstructed public value {reconstructed} != {expected}"
            )
        actions[action] = cells

    return document, actions


def _exact_best(
    actions: Mapping[str, Sequence[tuple[str, float, float]]],
) -> tuple[str, dict[str, float]]:
    values = {
        action: sum(probability * value for _, probability, value in cells)
        for action, cells in actions.items()
    }
    best_value = max(values.values())
    best = min(
        action
        for action, value in values.items()
        if abs(value - best_value) <= EPS
    )
    return best, values


def _initial_state(
    actions: Mapping[str, Sequence[tuple[str, float, float]]],
) -> dict[str, list[list[Any]]]:
    return {
        action: [[key, probability, value, False] for key, probability, value in cells]
        for action, cells in actions.items()
    }


def _bounds(state: Mapping[str, Sequence[Sequence[Any]]]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for action, cells in state.items():
        exact = sum(
            float(probability) * float(value)
            for _, probability, value, evaluated in cells
            if evaluated
        )
        unresolved_mass = sum(
            float(probability)
            for _, probability, _, evaluated in cells
            if not evaluated
        )
        bounds[action] = (
            exact + unresolved_mass * UTILITY_MIN,
            exact + unresolved_mass * UTILITY_MAX,
        )
    return bounds


def _certified(bounds: Mapping[str, tuple[float, float]]) -> str | None:
    """Certify the exact lexicographic policy from conservative value intervals."""

    for action in sorted(bounds):
        lower = bounds[action][0]
        certified = True
        for other, (_, upper) in bounds.items():
            if other == action:
                continue
            if other < action:
                if not lower > upper + EPS:
                    certified = False
                    break
            elif not lower >= upper - EPS:
                certified = False
                break
        if certified:
            return action
    return None

def _largest_unresolved(
    cells: Sequence[Sequence[Any]],
) -> tuple[float, int] | None:
    candidates = [
        (float(probability), index)
        for index, (_, probability, _, evaluated) in enumerate(cells)
        if not evaluated
    ]
    return max(candidates, default=None)


def _policy_run(
    actions: Mapping[str, Sequence[tuple[str, float, float]]],
) -> tuple[str, int]:
    state = _initial_state(actions)
    evaluations = 0

    while True:
        bounds = _bounds(state)
        certified = _certified(bounds)
        if certified is not None:
            return certified, evaluations

        incumbent = min(state, key=lambda action: (-bounds[action][0], action))
        challengers = [action for action in state if action != incumbent]
        challenger = (
            min(challengers, key=lambda action: (-bounds[action][1], action))
            if challengers
            else incumbent
        )

        candidates: list[tuple[float, float, str, int]] = []
        for action in dict.fromkeys((incumbent, challenger)):
            unresolved = _largest_unresolved(state[action])
            if unresolved is None:
                continue
            probability, index = unresolved
            interval_width = bounds[action][1] - bounds[action][0]
            candidates.append((interval_width, probability, action, index))

        if not candidates:
            raise RuntimeError("refinement reached an uncertified dead-end")

        _, _, action, index = max(candidates)
        state[action][index][3] = True
        evaluations += 1


def _random_run(
    actions: Mapping[str, Sequence[tuple[str, float, float]]],
    seed: int,
) -> tuple[str, int]:
    rng = random.Random(seed)
    state = _initial_state(actions)
    evaluations = 0

    while True:
        bounds = _bounds(state)
        certified = _certified(bounds)
        if certified is not None:
            return certified, evaluations

        best_lower = max(lower for lower, _ in bounds.values())
        choices: list[tuple[str, int]] = []
        for action, cells in state.items():
            if bounds[action][1] + 1e-12 < best_lower:
                continue
            choices.extend(
                (action, index)
                for index, (_, _, _, evaluated) in enumerate(cells)
                if not evaluated
            )

        if not choices:
            raise RuntimeError("random refinement reached an uncertified dead-end")

        action, index = rng.choice(choices)
        state[action][index][3] = True
        evaluations += 1


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def run(paths: Sequence[Path]) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    for index, path in enumerate(paths, start=1):
        document, actions = _load_case(path)
        exact_action, exact_values = _exact_best(actions)
        if exact_action != document["public_belief"]["chosen_action"]:
            raise ValueError("reconstructed exact action disagrees with recorded trace")

        total_cells = sum(len(cells) for cells in actions.values())
        policy_action, policy_evaluations = _policy_run(actions)

        random_costs: list[int] = []
        random_errors = 0
        for seed in range(RANDOM_SEEDS):
            action, evaluations = _random_run(actions, seed)
            random_costs.append(evaluations)
            random_errors += int(action != exact_action)

        ranked_values = sorted(
            exact_values.items(),
            key=lambda row: (-row[1], row[0]),
        )
        cases.append(
            {
                "case": f"case{index}",
                "source_fixture_id": document["source_fixture_id"],
                "world_count": int(document["world_count"]),
                "legal_action_count": len(actions),
                "observation_cell_count": total_cells,
                "exact_action": exact_action,
                "exact_gap": ranked_values[0][1] - ranked_values[1][1],
                "policy_action": policy_action,
                "policy_evaluations": policy_evaluations,
                "policy_fraction": policy_evaluations / total_cells,
                "policy_error": policy_action != exact_action,
                "random_seed_count": RANDOM_SEEDS,
                "random_median_evaluations": statistics.median(random_costs),
                "random_median_fraction": statistics.median(random_costs) / total_cells,
                "random_p90_fraction": _nearest_rank(
                    [cost / total_cells for cost in random_costs],
                    0.90,
                ),
                "random_errors": random_errors,
            }
        )

    policy_fractions = [float(case["policy_fraction"]) for case in cases]
    random_fractions = [float(case["random_median_fraction"]) for case in cases]
    result: dict[str, Any] = {
        "schema": "azelficoast.policy-boundary-refinement-experiment",
        "schema_version": 2,
        "utility_bounds": [UTILITY_MIN, UTILITY_MAX],
        "case_count": len(cases),
        "cases": cases,
        "certification_errors": sum(bool(case["policy_error"]) for case in cases),
        "median_fraction": statistics.median(policy_fractions),
        "p90_fraction": _nearest_rank(policy_fractions, 0.90),
        "acceptance": {
            "zero_certification_errors": True,
            "median_fraction_at_most": 0.50,
            "p90_fraction_at_most": 0.80,
        },
    }
    result["headline_pass"] = (
        result["certification_errors"] == 0
        and result["median_fraction"] <= 0.50
        and result["p90_fraction"] <= 0.80
    )
    at_or_below_target = sum(fraction <= 0.50 for fraction in policy_fractions)
    result["post_hoc_diagnostics"] = {
        "cases_at_or_below_median_target": at_or_below_target,
        "median_random_fraction": statistics.median(random_fractions),
        "median_absolute_fraction_improvement_vs_random": statistics.median(
            random_fraction - policy_fraction
            for policy_fraction, random_fraction in zip(
                policy_fractions, random_fractions, strict=True
            )
        ),
        "median_relative_reduction_vs_random": statistics.median(
            1.0 - policy_fraction / random_fraction
            for policy_fraction, random_fraction in zip(
                policy_fractions, random_fractions, strict=True
            )
        ),
        "exploratory_one_sided_sign_test_p_for_median_at_most_half": sum(
            math.comb(len(policy_fractions), count)
            for count in range(at_or_below_target + 1)
        ) / (2 ** len(policy_fractions)),
    }
    result["non_claims"] = [
        f"This is a {len(cases)}-natural-trace extension, not the originally proposed 96-decision ecological corpus.",
        "The treatment begins after exact root transition mechanics have produced public observation cells and their probability masses.",
        "The measured work is continuation-cell evaluation, not total battle-search wall time.",
        "The sign-test diagnostic is post hoc and the nine traces are not claimed to be an IID random sample.",
        "No refinement heuristic or acceptance threshold was retuned after observing the original three-case result.",
    ]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("decision_traces", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(run(args.decision_traces), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
