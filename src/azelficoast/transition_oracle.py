"""Single trusted structural contract for bounded transition oracles.

This module intentionally owns only structural validity shared by oracle consumers:
schema identity, materialized worlds, root actions, the complete transition matrix,
chance normalization, and dependency-candidate validity. Policy semantics remain in
their respective analyzers.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

ORACLE_SCHEMA = "azelficoast.real-belief-transition-oracle"
ORACLE_SCHEMA_VERSION = 1


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def transition_outcomes(
    transition: Mapping[str, Any],
    *,
    error_type: type[ValueError],
    empty_message: str = "every root transition must contain chance outcomes",
) -> list[Mapping[str, Any]]:
    raw = transition.get("outcomes")
    if not isinstance(raw, list) or not raw:
        raise error_type(empty_message)

    total = 0.0
    outcomes: list[Mapping[str, Any]] = []
    for outcome in raw:
        if not isinstance(outcome, Mapping):
            raise error_type("transition outcome must be an object")
        probability = outcome.get("probability")
        if not isinstance(probability, (int, float)) or probability <= 0:
            raise error_type("transition outcome probabilities must be positive")
        total += float(probability)
        outcomes.append(outcome)

    if abs(total - 1.0) > 1e-9:
        raise error_type(
            f"transition outcome probabilities sum to {total}, not 1"
        )
    return outcomes


def validate_oracle_core(
    document: Mapping[str, Any],
    *,
    error_type: type[ValueError],
    outcome_empty_message: str = "every root transition must contain chance outcomes",
) -> tuple[
    list[dict[str, Any]],
    list[str],
    dict[tuple[str, str], Mapping[str, Any]],
    list[str],
]:
    if (
        document.get("schema") != ORACLE_SCHEMA
        or document.get("schema_version") != ORACLE_SCHEMA_VERSION
    ):
        raise error_type("unsupported transition oracle schema")

    raw_worlds = document.get("worlds")
    raw_actions = document.get("legal_actions")
    raw_transitions = document.get("transitions")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise error_type("oracle must contain at least one hidden world")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise error_type("oracle must contain root legal actions")
    if not isinstance(raw_transitions, list):
        raise error_type("oracle transitions must be a list")

    worlds = [dict(world) for world in raw_worlds]
    world_ids = [str(world.get("world_id")) for world in worlds]
    if len(set(world_ids)) != len(world_ids):
        raise error_type("world ids must be unique")
    for world in worlds:
        if not isinstance(world.get("hidden"), Mapping):
            raise error_type("every world must contain hidden state")
        weight = world.get("weight")
        if not isinstance(weight, (int, float)) or float(weight) <= 0:
            raise error_type("world weights must be positive")

    actions = [str(action) for action in raw_actions]
    if len(set(actions)) != len(actions):
        raise error_type("root actions must be unique")

    world_id_set = set(world_ids)
    action_set = set(actions)
    transitions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for transition in raw_transitions:
        if not isinstance(transition, Mapping):
            raise error_type("transition must be an object")
        world_id = str(transition.get("world_id"))
        action = str(transition.get("action"))
        key = (world_id, action)
        if world_id not in world_id_set:
            raise error_type(f"transition references unknown world {world_id}")
        if action not in action_set:
            raise error_type(f"transition references non-root action {action}")
        if key in transitions:
            raise error_type(f"duplicate transition for {world_id} {action}")
        transition_outcomes(
            transition,
            error_type=error_type,
            empty_message=outcome_empty_message,
        )
        transitions[key] = transition

    expected = {(world_id, action) for world_id in world_ids for action in actions}
    missing = expected - set(transitions)
    if missing:
        raise error_type(
            f"oracle omitted root transitions: {sorted(missing)[:3]!r}"
        )

    hidden_fields = sorted(
        {
            str(field)
            for world in worlds
            for field in world["hidden"]
        }
    )
    raw_candidates = document.get("dependency_candidates")
    if raw_candidates is None:
        candidates = hidden_fields
    elif isinstance(raw_candidates, list) and all(
        isinstance(field, str) for field in raw_candidates
    ):
        candidates = list(dict.fromkeys(raw_candidates))
    else:
        raise error_type(
            "dependency_candidates must be an array of semantic field paths"
        )

    unknown = [field for field in candidates if field not in hidden_fields]
    if unknown:
        raise error_type(
            f"dependency candidates reference unknown fields: {unknown!r}"
        )

    return worlds, actions, transitions, candidates
