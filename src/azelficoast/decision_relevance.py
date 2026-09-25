"""Exact quotienting of hidden worlds by bounded decision semantics.

A world may differ in hidden facts that survive public evidence yet remain irrelevant to
the current bounded decision. This module derives the smallest hidden-field projection
that determines the complete root transition and continuation evidence, then merges only
worlds whose bounded decision semantics are identical.

The quotient is exact for the supplied oracle. It is not a learned approximation and it
does not claim that omitted hidden fields are irrelevant outside the oracle's horizon.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from collections import defaultdict
from typing import Any, Mapping, Sequence

from azelficoast.real_belief_trace import (
    SCHEMA as ORACLE_SCHEMA,
    SCHEMA_VERSION as ORACLE_SCHEMA_VERSION,
    analyze_oracle,
)

CERTIFICATE_SCHEMA = "azelficoast.decision-relevance-certificate"
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
        if not isinstance(probability, (int, float)) or probability <= 0:
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
        elif isinstance(terminal, (int, float)):
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

    # Chance-sample order is not semantic. Canonicalize the distribution so two
    # worlds that differ only in outcome enumeration can still be merged.
    return sorted(normalized, key=_canonical)


def _validated_oracle(
    document: Mapping[str, Any],
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
        raise DecisionRelevanceError("unsupported transition oracle schema")

    raw_worlds = document.get("worlds")
    raw_actions = document.get("legal_actions")
    raw_transitions = document.get("transitions")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise DecisionRelevanceError("oracle must contain at least one hidden world")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise DecisionRelevanceError("oracle must contain root legal actions")
    if not isinstance(raw_transitions, list):
        raise DecisionRelevanceError("oracle transitions must be a list")

    worlds = [dict(world) for world in raw_worlds]
    world_ids = [str(world.get("world_id")) for world in worlds]
    if len(set(world_ids)) != len(world_ids):
        raise DecisionRelevanceError("world ids must be unique")
    for world in worlds:
        if not isinstance(world.get("hidden"), Mapping):
            raise DecisionRelevanceError("every world must contain hidden state")
        weight = world.get("weight")
        if not isinstance(weight, (int, float)) or float(weight) <= 0:
            raise DecisionRelevanceError("world weights must be positive")

    actions = [str(action) for action in raw_actions]
    if len(set(actions)) != len(actions):
        raise DecisionRelevanceError("root actions must be unique")

    transitions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for transition in raw_transitions:
        if not isinstance(transition, Mapping):
            raise DecisionRelevanceError("transition must be an object")
        world_id = str(transition.get("world_id"))
        action = str(transition.get("action"))
        key = (world_id, action)
        if world_id not in set(world_ids):
            raise DecisionRelevanceError(
                f"transition references unknown world {world_id}"
            )
        if action not in set(actions):
            raise DecisionRelevanceError(
                f"transition references non-root action {action}"
            )
        if key in transitions:
            raise DecisionRelevanceError(
                f"duplicate transition for {world_id} {action}"
            )
        _normalized_outcomes(transition)
        transitions[key] = transition

    expected = {(world_id, action) for world_id in world_ids for action in actions}
    missing = expected - set(transitions)
    if missing:
        raise DecisionRelevanceError(
            f"oracle omitted root transitions: {sorted(missing)[:3]!r}"
        )

    raw_candidates = document.get("dependency_candidates")
    hidden_fields = sorted(
        {
            str(field)
            for world in worlds
            for field in world["hidden"]
        }
    )
    if raw_candidates is None:
        candidates = hidden_fields
    elif isinstance(raw_candidates, list) and all(
        isinstance(field, str) for field in raw_candidates
    ):
        candidates = list(dict.fromkeys(raw_candidates))
    else:
        raise DecisionRelevanceError(
            "dependency_candidates must be an array of semantic field paths"
        )

    unknown = [field for field in candidates if field not in hidden_fields]
    if unknown:
        raise DecisionRelevanceError(
            f"dependency candidates reference unknown fields: {unknown!r}"
        )

    return worlds, actions, transitions, candidates


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
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an exact certificate and a semantically equivalent quotient oracle."""

    worlds, actions, transitions, candidates = _validated_oracle(document)
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
        key = tuple(
            _canonical(hidden.get(field))
            for field in decision_fields
        )
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
            world for world in members
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
        "schema": CERTIFICATE_SCHEMA,
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
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


def analyze_quotiented_oracle(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze the exact quotient and return its trace plus the certificate."""

    certificate, quotient = decision_relevance_quotient(document)
    trace = analyze_oracle(quotient)
    trace["source_world_count"] = certificate["worlds_in"]
    trace["decision_relevance"] = {
        key: copy.deepcopy(value)
        for key, value in certificate.items()
        if key != "classes"
    }
    return trace, certificate



def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Certify and analyze the exact decision-relevance quotient."
    )
    parser.add_argument("oracle", type=Path)
    args = parser.parse_args(argv)

    document = json.loads(args.oracle.read_text(encoding="utf-8"))
    trace, certificate = analyze_quotiented_oracle(document)
    print(
        json.dumps(
            {
                "schema": "azelficoast.decision-relevance-analysis",
                "schema_version": 1,
                "certificate": certificate,
                "quotient_trace": trace,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
