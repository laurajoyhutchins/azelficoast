"""Exact quotienting of hidden worlds by bounded decision semantics.

The quotient is domain-neutral: it depends only on finite hidden-world support and the
observable transition/value surface encoded by the supplied oracle. It does not import
Pokémon, Showdown, poke-env, or any Azelficoast live battle adapter.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

from azelficoast.core.transition import (
    ORACLE_SCHEMA,
    ORACLE_SCHEMA_VERSION,
    validate_transition_oracle,
)

CERTIFICATE_SCHEMA = "azelficoast.core.decision-relevance-certificate"
CERTIFICATE_SCHEMA_VERSION = 1


class DecisionRelevanceError(ValueError):
    """Raised when an exact decision-relevance quotient cannot be certified."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _normalized_outcomes(transition: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = transition.get("outcomes")
    if not isinstance(raw, list) or not raw:
        raise DecisionRelevanceError("every transition must contain chance outcomes")

    total = 0.0
    normalized: list[dict[str, Any]] = []
    for outcome in raw:
        if not isinstance(outcome, Mapping):
            raise DecisionRelevanceError("transition outcome must be an object")
        probability = outcome.get("probability")
        if (
            not isinstance(probability, (int, float))
            or isinstance(probability, bool)
            or float(probability) <= 0
        ):
            raise DecisionRelevanceError(
                "transition outcome probabilities must be positive"
            )
        total += float(probability)

        continuation = outcome.get("continuations")
        terminal = outcome.get("terminal_utility")
        if isinstance(continuation, Mapping) and continuation:
            value_surface: dict[str, Any] = {
                "continuations": {
                    str(action): float(value)
                    for action, value in sorted(continuation.items())
                }
            }
        elif isinstance(terminal, (int, float)) and not isinstance(terminal, bool):
            value_surface = {"terminal_utility": float(terminal)}
        else:
            raise DecisionRelevanceError(
                "outcome has neither continuations nor terminal utility"
            )

        normalized.append(
            {
                "probability": float(probability),
                "observation": outcome.get("observation"),
                "successor": outcome.get("successor"),
                **value_surface,
            }
        )

    if abs(total - 1.0) > 1e-9:
        raise DecisionRelevanceError(
            f"transition outcome probabilities sum to {total}, not 1"
        )
    return sorted(normalized, key=_canonical)


def _world_semantics(
    world_id: str,
    actions: Sequence[str],
    transitions: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        action: _normalized_outcomes(transitions[(world_id, action)])
        for action in actions
    }


def decision_relevance_quotient(
    document: Mapping[str, Any],
    *,
    expected_schema: str | None = ORACLE_SCHEMA,
    expected_schema_version: int | None = ORACLE_SCHEMA_VERSION,
    certificate_schema: str = CERTIFICATE_SCHEMA,
    certificate_schema_version: int = CERTIFICATE_SCHEMA_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an exact certificate and semantically equivalent quotient oracle."""

    worlds, actions, transitions, candidates = validate_transition_oracle(
        document,
        error_type=DecisionRelevanceError,
        outcome_empty_message="every transition must contain chance outcomes",
        expected_schema=expected_schema,
        expected_schema_version=expected_schema_version,
    )
    semantics = {
        str(world["world_id"]): _world_semantics(
            str(world["world_id"]),
            actions,
            transitions,
        )
        for world in worlds
    }
    semantic_hashes = {
        world_id: _sha256(surface)
        for world_id, surface in semantics.items()
    }

    decision_fields: tuple[str, ...] | None = None
    for size in range(len(candidates) + 1):
        for candidate in itertools.combinations(candidates, size):
            keyed_semantics: dict[tuple[str, ...], str] = {}
            sufficient = True
            for world in worlds:
                world_id = str(world["world_id"])
                hidden = world["hidden"]
                key = tuple(_canonical(hidden.get(field)) for field in candidate)
                semantic_hash = semantic_hashes[world_id]
                previous = keyed_semantics.setdefault(key, semantic_hash)
                if previous != semantic_hash:
                    sufficient = False
                    break
            if sufficient:
                decision_fields = candidate
                break
        if decision_fields is not None:
            break

    if decision_fields is None:
        raise DecisionRelevanceError(
            "dependency candidates cannot explain bounded decision semantics"
        )

    classes: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for world in worlds:
        hidden = world["hidden"]
        key = tuple(_canonical(hidden.get(field)) for field in decision_fields)
        classes[key].append(world)

    quotient_worlds: list[dict[str, Any]] = []
    quotient_transitions: list[dict[str, Any]] = []
    certificate_classes: list[dict[str, Any]] = []

    for key, members in sorted(classes.items(), key=lambda row: row[0]):
        member_ids = sorted(str(world["world_id"]) for world in members)
        representative_id = member_ids[0]
        semantic_hash = semantic_hashes[representative_id]
        if any(semantic_hashes[member_id] != semantic_hash for member_id in member_ids):
            raise DecisionRelevanceError(
                "decision-relevance class contains non-equivalent worlds"
            )

        representative = next(
            world
            for world in members
            if str(world["world_id"]) == representative_id
        )
        class_id = "decision-class-" + _sha256(
            {
                "fields": decision_fields,
                "key": key,
                "semantic_hash": semantic_hash,
            }
        )[:24]
        class_weight = sum(float(world["weight"]) for world in members)
        class_hidden = {
            field: copy.deepcopy(representative["hidden"].get(field))
            for field in decision_fields
        }

        quotient_worlds.append(
            {
                "world_id": class_id,
                "weight": class_weight,
                "hidden": class_hidden,
                "provenance": {
                    "decision_relevance_members": member_ids,
                    "representative_world_id": representative_id,
                    "semantic_hash": semantic_hash,
                },
            }
        )
        for action in actions:
            transition = copy.deepcopy(transitions[(representative_id, action)])
            transition["world_id"] = class_id
            for outcome in transition.get("outcomes", []):
                if isinstance(outcome, dict):
                    outcome.pop("transition_reads", None)
            quotient_transitions.append(transition)

        certificate_classes.append(
            {
                "class_id": class_id,
                "weight": class_weight,
                "member_world_ids": member_ids,
                "semantic_hash": semantic_hash,
            }
        )

    worlds_in = len(worlds)
    classes_out = len(quotient_worlds)
    certificate = {
        "schema": certificate_schema,
        "schema_version": certificate_schema_version,
        "source_fixture_id": document.get("source_fixture_id"),
        "decision_fields": list(decision_fields),
        "worlds_in": worlds_in,
        "classes_out": classes_out,
        "world_reduction": worlds_in - classes_out,
        "reduction_fraction": 1.0 - (classes_out / worlds_in),
        "belief_branching_required": classes_out > 1,
        "partition_key_hash": _sha256(
            {
                "fields": decision_fields,
                "classes": [
                    sorted(row["member_world_ids"])
                    for row in certificate_classes
                ],
            }
        ),
        "classes": certificate_classes,
        "claim": (
            "Within the supplied bounded oracle, worlds in each class have identical "
            "root chance distributions, public observations, successor states, and "
            "continuation or terminal utility surfaces for every legal root action."
        ),
        "non_claim": (
            "The certificate does not establish irrelevance beyond the supplied "
            "oracle horizon or opponent-response model."
        ),
    }

    quotient = copy.deepcopy(dict(document))
    quotient["worlds"] = quotient_worlds
    quotient["transitions"] = quotient_transitions
    quotient["dependency_candidates"] = list(decision_fields)
    quotient["decision_relevance_certificate"] = copy.deepcopy(certificate)

    return certificate, quotient
