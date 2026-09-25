"""Executable contract for matched determinization/information-set comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

PLAN_SCHEMA = "azelficoast.matched-search-comparison-plan"
PLAN_SCHEMA_VERSION = 1
PACKET_SCHEMA = "azelficoast.matched-search-comparison-packet"
PACKET_SCHEMA_VERSION = 1
RESULT_SCHEMA = "azelficoast.matched-search-comparison-result"
RESULT_SCHEMA_VERSION = 1

METHODS = ("determinization", "information_set")
POSTERIOR_TREATMENTS = ("oracle", "generator_faithful", "practical")
OPPONENT_MODELS = ("fixed_observed_response", "two_sided_information_sets")
BUDGET_UNITS = ("transition_evaluations",)


class MatchedComparisonError(ValueError):
    """Raised when an experiment attempts an unmatched comparison."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedComparisonError(f"{path}: expected a JSON object")
    return value


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        raise MatchedComparisonError("unexpected matched-comparison plan schema")

    treatments = plan.get("posterior_treatments")
    if not isinstance(treatments, list) or not treatments:
        raise MatchedComparisonError("plan must preregister posterior treatments")
    normalized_treatments = [str(value) for value in treatments]
    if len(set(normalized_treatments)) != len(normalized_treatments):
        raise MatchedComparisonError("posterior treatments must be unique")
    unknown_treatments = set(normalized_treatments) - set(POSTERIOR_TREATMENTS)
    if unknown_treatments:
        raise MatchedComparisonError(
            f"unknown posterior treatments: {sorted(unknown_treatments)!r}"
        )

    budget = plan.get("compute_budget")
    if not isinstance(budget, Mapping):
        raise MatchedComparisonError("plan must declare a compute budget")
    unit = budget.get("unit")
    limit = budget.get("per_method_limit")
    if unit not in BUDGET_UNITS:
        raise MatchedComparisonError(
            f"compute budget unit must be one of {BUDGET_UNITS!r}"
        )
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise MatchedComparisonError("per-method compute limit must be a positive integer")

    opponent_model = plan.get("opponent_model")
    if opponent_model not in OPPONENT_MODELS:
        raise MatchedComparisonError(
            f"opponent model must be one of {OPPONENT_MODELS!r}"
        )

    depths = plan.get("depths")
    if not isinstance(depths, list) or not depths:
        raise MatchedComparisonError("plan must preregister at least one search depth")
    normalized_depths = [int(depth) for depth in depths]
    if any(depth <= 0 for depth in normalized_depths):
        raise MatchedComparisonError("search depths must be positive")
    if sorted(set(normalized_depths)) != normalized_depths:
        raise MatchedComparisonError("search depths must be unique and ascending")

    predictors = plan.get("confirmatory_predictors")
    if not isinstance(predictors, list) or not 2 <= len(predictors) <= 3:
        raise MatchedComparisonError(
            "plan must freeze two or three confirmatory predictors"
        )
    if len(set(map(str, predictors))) != len(predictors):
        raise MatchedComparisonError("confirmatory predictors must be unique")

    if plan.get("cluster_unit") != "battle_tag":
        raise MatchedComparisonError("population inference must cluster by battle_tag")

    showdown_commit = plan.get("showdown_commit")
    if not isinstance(showdown_commit, str) or not showdown_commit:
        raise MatchedComparisonError("plan must pin a Showdown commit")

    return dict(plan)


def freeze_packet(
    *,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    posterior: Mapping[str, Any],
    posterior_treatment: str,
    depth: int,
) -> dict[str, Any]:
    checked_plan = validate_plan(plan)
    if posterior_treatment not in checked_plan["posterior_treatments"]:
        raise MatchedComparisonError(
            f"posterior treatment {posterior_treatment!r} was not preregistered"
        )
    if int(depth) not in checked_plan["depths"]:
        raise MatchedComparisonError(f"depth {depth} was not preregistered")

    fixture_id = state.get("fixture_id")
    battle_tag = state.get("battle_tag")
    public_state = state.get("public_state")
    legal_actions = state.get("legal_actions")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise MatchedComparisonError("state must identify its fixture")
    if not isinstance(battle_tag, str) or not battle_tag:
        raise MatchedComparisonError("state must identify its battle")
    if not isinstance(public_state, Mapping):
        raise MatchedComparisonError("comparison state must contain public_state only")
    if (
        not isinstance(legal_actions, list)
        or not legal_actions
        or not all(isinstance(action, str) and action for action in legal_actions)
    ):
        raise MatchedComparisonError("state must contain non-empty legal actions")
    if len(set(legal_actions)) != len(legal_actions):
        raise MatchedComparisonError("legal actions must be unique")

    if posterior.get("treatment") != posterior_treatment:
        raise MatchedComparisonError("posterior artifact treatment does not match packet")
    if posterior.get("conditioned_on_public_history") is not True:
        raise MatchedComparisonError("posterior must be conditioned on public history")
    if posterior.get("realized_hidden_state_revealed") is not False:
        raise MatchedComparisonError(
            "posterior may not reveal the realized hidden state"
        )
    worlds = posterior.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise MatchedComparisonError("posterior must contain hidden-world support")

    state_digest = _sha256(
        {
            "fixture_id": fixture_id,
            "battle_tag": battle_tag,
            "public_state": public_state,
            "legal_actions": legal_actions,
        }
    )
    posterior_digest = _sha256(posterior)
    input_digest = _sha256(
        {
            "state_digest": state_digest,
            "posterior_digest": posterior_digest,
            "showdown_commit": checked_plan["showdown_commit"],
            "depth": int(depth),
        }
    )
    budget = {
        "unit": checked_plan["compute_budget"]["unit"],
        "authorized": int(checked_plan["compute_budget"]["per_method_limit"]),
    }

    packet = {
        "schema": PACKET_SCHEMA,
        "schema_version": PACKET_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "battle_tag": battle_tag,
        "posterior_treatment": posterior_treatment,
        "opponent_model": checked_plan["opponent_model"],
        "showdown_commit": checked_plan["showdown_commit"],
        "depth": int(depth),
        "state_digest": state_digest,
        "posterior_digest": posterior_digest,
        "input_digest": input_digest,
        "legal_actions": list(legal_actions),
        "compute_budget": budget,
        "work": [
            {
                "method": method,
                "input_digest": input_digest,
                "compute_budget": dict(budget),
            }
            for method in METHODS
        ],
    }
    packet["packet_digest"] = _sha256(packet)
    return packet


def _validate_packet(packet: Mapping[str, Any]) -> None:
    if (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("schema_version") != PACKET_SCHEMA_VERSION
    ):
        raise MatchedComparisonError("unexpected comparison packet schema")
    work = packet.get("work")
    if not isinstance(work, list) or len(work) != len(METHODS):
        raise MatchedComparisonError("comparison packet must contain both methods")
    methods = [row.get("method") for row in work if isinstance(row, Mapping)]
    if tuple(methods) != METHODS:
        raise MatchedComparisonError("comparison methods are missing or reordered")
    expected_digest = packet.get("input_digest")
    expected_budget = packet.get("compute_budget")
    for row in work:
        if not isinstance(row, Mapping):
            raise MatchedComparisonError("comparison work item must be an object")
        if row.get("input_digest") != expected_digest:
            raise MatchedComparisonError("comparison methods do not share one input")
        if row.get("compute_budget") != expected_budget:
            raise MatchedComparisonError("comparison methods do not share one budget")


def settle_packet(
    *,
    packet: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_packet(packet)
    if len(receipts) != len(METHODS):
        raise MatchedComparisonError("exactly two method receipts are required")
    by_method = {str(receipt.get("method")): receipt for receipt in receipts}
    if set(by_method) != set(METHODS) or len(by_method) != len(METHODS):
        raise MatchedComparisonError("receipts must contain each comparison method once")

    legal_actions = list(packet["legal_actions"])
    root_values: dict[str, dict[str, float]] = {}
    consumed: dict[str, int] = {}
    chosen: dict[str, str] = {}

    for method in METHODS:
        receipt = by_method[method]
        if receipt.get("input_digest") != packet["input_digest"]:
            raise MatchedComparisonError(f"{method}: receipt used another frozen input")
        budget = receipt.get("compute_budget")
        if not isinstance(budget, Mapping) or budget != packet["compute_budget"]:
            raise MatchedComparisonError(f"{method}: receipt used another authorized budget")
        used = receipt.get("consumed")
        if not isinstance(used, int) or isinstance(used, bool) or used < 0:
            raise MatchedComparisonError(f"{method}: consumed budget must be an integer")
        if used > int(packet["compute_budget"]["authorized"]):
            raise MatchedComparisonError(f"{method}: exceeded the authorized budget")
        consumed[method] = used

        values = receipt.get("root_values")
        if not isinstance(values, Mapping) or set(map(str, values)) != set(legal_actions):
            raise MatchedComparisonError(
                f"{method}: root values do not cover the frozen legal actions"
            )
        normalized = {str(action): float(value) for action, value in values.items()}
        root_values[method] = normalized
        best_value = max(normalized.values())
        expected_action = min(
            action for action, value in normalized.items() if value == best_value
        )
        action = receipt.get("chosen_action")
        if action != expected_action:
            raise MatchedComparisonError(
                f"{method}: chosen action is inconsistent with its root values"
            )
        chosen[method] = str(action)

    det_values = root_values["determinization"]
    info_values = root_values["information_set"]
    gaps = {
        action: det_values[action] - info_values[action] for action in legal_actions
    }
    max_gap = max(gaps.values())
    max_gap_action = min(action for action, gap in gaps.items() if gap == max_gap)
    info_best = max(info_values.values())
    regret = max(0.0, info_best - info_values[chosen["determinization"]])

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "packet_digest": packet["packet_digest"],
        "fixture_id": packet["fixture_id"],
        "battle_tag": packet["battle_tag"],
        "posterior_treatment": packet["posterior_treatment"],
        "opponent_model": packet["opponent_model"],
        "depth": packet["depth"],
        "matched_input": True,
        "matched_authorized_compute": True,
        "compute_budget": dict(packet["compute_budget"]),
        "compute_consumed": consumed,
        "max_determinization_value_optimism": max_gap,
        "max_optimism_action": max_gap_action,
        "information_set_regret_of_determinization_action": regret,
        "policy_disagreement": (
            chosen["determinization"] != chosen["information_set"]
        ),
        "chosen_actions": chosen,
        "root_values": root_values,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("plan", type=Path)
    freeze.add_argument("state", type=Path)
    freeze.add_argument("posterior", type=Path)
    freeze.add_argument("--posterior-treatment", required=True, choices=POSTERIOR_TREATMENTS)
    freeze.add_argument("--depth", required=True, type=int)
    freeze.add_argument("--output", required=True, type=Path)

    settle = commands.add_parser("settle")
    settle.add_argument("packet", type=Path)
    settle.add_argument("receipts", nargs=2, type=Path)
    settle.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        packet = freeze_packet(
            plan=_load_object(args.plan),
            state=_load_object(args.state),
            posterior=_load_object(args.posterior),
            posterior_treatment=args.posterior_treatment,
            depth=args.depth,
        )
        _write_json(args.output, packet)
        print(json.dumps(packet, sort_keys=True))
        return 0
    if args.command == "settle":
        result = settle_packet(
            packet=_load_object(args.packet),
            receipts=[_load_object(path) for path in args.receipts],
        )
        _write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(f"unexpected command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
