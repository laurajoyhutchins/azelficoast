"""Independent budgeted execution of matched search methods over a frozen oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.matched_comparison import (
    METHODS,
    PACKET_SCHEMA,
    PACKET_SCHEMA_VERSION,
    _sha256,
)
from azelficoast.real_belief_trace import (
    SCHEMA as ORACLE_SCHEMA,
    SCHEMA_VERSION as ORACLE_SCHEMA_VERSION,
    BeliefTraceError,
    _canonical,
    _choose,
    _outcomes,
    _weighted_continuation_choice,
)

RECEIPT_SCHEMA = "azelficoast.matched-search-receipt"
RECEIPT_SCHEMA_VERSION = 1


class MatchedSearchExecutionError(ValueError):
    """Raised when a frozen matched-search work item cannot execute faithfully."""


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedSearchExecutionError(f"{path}: expected a JSON object")
    return value


def _oracle_digest(oracle: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            oracle,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _validated_inputs(
    *,
    packet: Mapping[str, Any],
    posterior: Mapping[str, Any],
    oracle: Mapping[str, Any],
    method: str,
) -> tuple[list[dict[str, Any]], list[str], dict[tuple[str, str], Mapping[str, Any]]]:
    if (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("schema_version") != PACKET_SCHEMA_VERSION
    ):
        raise MatchedSearchExecutionError("unexpected matched-search packet schema")
    if method not in METHODS:
        raise MatchedSearchExecutionError(f"unknown search method {method!r}")
    if int(packet.get("depth", 0)) != 1:
        raise MatchedSearchExecutionError(
            "oracle receipt executor currently supports depth 1 only"
        )
    if packet.get("posterior_digest") != _sha256(posterior):
        raise MatchedSearchExecutionError("posterior artifact does not match frozen packet")
    if posterior.get("treatment") != packet.get("posterior_treatment"):
        raise MatchedSearchExecutionError("posterior treatment drifted after freezing")

    if (
        oracle.get("schema") != ORACLE_SCHEMA
        or oracle.get("schema_version") != ORACLE_SCHEMA_VERSION
    ):
        raise MatchedSearchExecutionError("unexpected transition-oracle schema")
    if oracle.get("source_fixture_id") != packet.get("fixture_id"):
        raise MatchedSearchExecutionError("transition oracle belongs to another fixture")
    if oracle.get("showdown_commit") != packet.get("showdown_commit"):
        raise MatchedSearchExecutionError("transition oracle used another Showdown revision")

    legal_actions = packet.get("legal_actions")
    if not isinstance(legal_actions, list) or oracle.get("legal_actions") != legal_actions:
        raise MatchedSearchExecutionError(
            "transition oracle legal actions differ from frozen packet"
        )

    raw_worlds = oracle.get("worlds")
    posterior_worlds = posterior.get("worlds")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise MatchedSearchExecutionError("transition oracle has no worlds")
    if not isinstance(posterior_worlds, list) or not posterior_worlds:
        raise MatchedSearchExecutionError("posterior has no worlds")

    worlds = [dict(world) for world in raw_worlds if isinstance(world, Mapping)]
    if len(worlds) != len(raw_worlds):
        raise MatchedSearchExecutionError("transition-oracle world must be an object")
    oracle_by_id = {str(world.get("world_id")): world for world in worlds}
    posterior_by_id = {
        str(world.get("world_id")): world
        for world in posterior_worlds
        if isinstance(world, Mapping)
    }
    if len(oracle_by_id) != len(worlds) or len(posterior_by_id) != len(posterior_worlds):
        raise MatchedSearchExecutionError("world ids must be unique")
    if set(oracle_by_id) != set(posterior_by_id):
        raise MatchedSearchExecutionError(
            "posterior and transition oracle have different hidden-world support"
        )

    oracle_total = sum(float(world.get("weight", 0)) for world in worlds)
    posterior_total = sum(
        float(world.get("weight", 0)) for world in posterior_by_id.values()
    )
    if oracle_total <= 0 or posterior_total <= 0:
        raise MatchedSearchExecutionError("hidden-world posterior has no positive mass")

    for world_id, oracle_world in oracle_by_id.items():
        posterior_world = posterior_by_id[world_id]
        hidden = posterior_world.get("hidden")
        if not isinstance(hidden, Mapping):
            raise MatchedSearchExecutionError(
                f"{world_id}: posterior must retain correlated hidden state"
            )
        if _canonical(hidden) != _canonical(oracle_world.get("hidden")):
            raise MatchedSearchExecutionError(
                f"{world_id}: posterior hidden state differs from transition oracle"
            )
        oracle_mass = float(oracle_world.get("weight", 0)) / oracle_total
        posterior_mass = float(posterior_world.get("weight", 0)) / posterior_total
        if abs(oracle_mass - posterior_mass) > 1e-12:
            raise MatchedSearchExecutionError(
                f"{world_id}: posterior weight differs from transition oracle"
            )
        oracle_world["weight"] = posterior_mass

    raw_transitions = oracle.get("transitions")
    if not isinstance(raw_transitions, list):
        raise MatchedSearchExecutionError("transition oracle transitions must be a list")
    transitions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for transition in raw_transitions:
        if not isinstance(transition, Mapping):
            raise MatchedSearchExecutionError("transition must be an object")
        world_id = str(transition.get("world_id"))
        action = str(transition.get("action"))
        key = (world_id, action)
        if key in transitions:
            raise MatchedSearchExecutionError(
                f"duplicate transition for {world_id} {action}"
            )
        try:
            outcomes = _outcomes(transition)
        except BeliefTraceError as error:
            raise MatchedSearchExecutionError(str(error)) from error
        if any("continuation_transitions" in outcome for outcome in outcomes):
            raise MatchedSearchExecutionError(
                "depth-1 receipt cannot consume deeper continuation evidence"
            )
        transitions[key] = transition

    expected = {
        (world_id, action)
        for world_id in oracle_by_id
        for action in legal_actions
    }
    missing = expected - set(transitions)
    extra = set(transitions) - expected
    if missing or extra:
        raise MatchedSearchExecutionError(
            "transition matrix differs from frozen world/action product"
        )

    return worlds, list(legal_actions), transitions


def _determinization_values(
    worlds: Sequence[Mapping[str, Any]],
    legal_actions: Sequence[str],
    transitions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for action in legal_actions:
        total = 0.0
        for world in worlds:
            world_id = str(world["world_id"])
            prior = float(world["weight"])
            by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for outcome_index, outcome in enumerate(
                _outcomes(transitions[(world_id, action)])
            ):
                by_observation[_canonical(outcome.get("observation"))].append(
                    {
                        "outcome_index": outcome_index,
                        "chance": float(outcome["probability"]),
                        "outcome": outcome,
                    }
                )

            world_value = 0.0
            for members in by_observation.values():
                chance = sum(float(member["chance"]) for member in members)
                _, value = _weighted_continuation_choice(
                    members,
                    weight_key="chance",
                )
                world_value += chance * value
            total += prior * world_value
        values[action] = total
    return values


def _information_set_values(
    worlds: Sequence[Mapping[str, Any]],
    legal_actions: Sequence[str],
    transitions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for action in legal_actions:
        by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for world in worlds:
            world_id = str(world["world_id"])
            prior = float(world["weight"])
            for outcome_index, outcome in enumerate(
                _outcomes(transitions[(world_id, action)])
            ):
                chance = float(outcome["probability"])
                by_observation[_canonical(outcome.get("observation"))].append(
                    {
                        "world_id": world_id,
                        "outcome_index": outcome_index,
                        "mass": prior * chance,
                        "outcome": outcome,
                    }
                )

        total = 0.0
        for members in by_observation.values():
            mass = sum(float(member["mass"]) for member in members)
            _, value = _weighted_continuation_choice(
                members,
                weight_key="mass",
            )
            total += mass * value
        values[action] = total
    return values


def execute_method(
    *,
    packet: Mapping[str, Any],
    posterior: Mapping[str, Any],
    oracle: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    worlds, legal_actions, transitions = _validated_inputs(
        packet=packet,
        posterior=posterior,
        oracle=oracle,
        method=method,
    )
    budget = packet.get("compute_budget")
    if not isinstance(budget, Mapping) or budget.get("unit") != "transition_evaluations":
        raise MatchedSearchExecutionError(
            "receipt executor requires transition_evaluations budget"
        )

    required = len(worlds) * len(legal_actions)
    authorized = int(budget.get("authorized", 0))
    if required > authorized:
        raise MatchedSearchExecutionError(
            f"{method}: requires {required} transitions but budget authorizes {authorized}"
        )

    if method == "determinization":
        root_values = _determinization_values(worlds, legal_actions, transitions)
    else:
        root_values = _information_set_values(worlds, legal_actions, transitions)

    chosen_action, _ = _choose(root_values)
    return {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "method": method,
        "input_digest": packet["input_digest"],
        "packet_digest": packet["packet_digest"],
        "posterior_digest": packet["posterior_digest"],
        "transition_oracle_digest": _oracle_digest(oracle),
        "showdown_commit": packet["showdown_commit"],
        "compute_budget": dict(budget),
        "consumed": required,
        "budget_unit_definition": (
            "one transition_evaluation per frozen hidden-world/root-action "
            "transition record consumed"
        ),
        "chosen_action": chosen_action,
        "root_values": root_values,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("posterior", type=Path)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    result = execute_method(
        packet=_load_object(args.packet),
        posterior=_load_object(args.posterior),
        oracle=_load_object(args.oracle),
        method=args.method,
    )
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
