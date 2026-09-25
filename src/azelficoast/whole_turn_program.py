"""Compile exact whole-turn transition programs from a pinned Showdown oracle.

The compiler deliberately ignores continuation values. A transition program describes
only one complete root turn: chance probability, public observation, and successor
state. Search and value functions consume that surface later.

The default compiler derives semantic classes from the complete finite-support oracle,
so analysis measures behavior rather than implementation reads. An explicit dynamic-read
strategy is available for validating representative-refinement machinery; lazy Showdown
production emits that strategy directly and is verified against the semantic oracle.
"""

from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from typing import Any, Mapping, Sequence

from azelficoast.transition_oracle import (
    canonical_json,
    sha256_json,
    transition_outcomes,
    validate_oracle_core,
)

PROGRAM_SET_SCHEMA = "azelficoast.whole-turn-transition-program-set"
PROGRAM_SET_SCHEMA_VERSION = 1
EXECUTION_SCHEMA = "azelficoast.weighted-whole-turn-outcomes"
EXECUTION_SCHEMA_VERSION = 1
VERIFICATION_SCHEMA = "azelficoast.whole-turn-program-verification"
VERIFICATION_SCHEMA_VERSION = 1


class WholeTurnProgramError(ValueError):
    """Raised when a whole-turn program cannot be compiled or applied exactly."""


def _immediate_distribution(
    transition: Mapping[str, Any],
) -> list[dict[str, Any]]:
    outcomes = [
        {
            "probability": float(outcome["probability"]),
            "observation": copy.deepcopy(outcome.get("observation")),
            "successor": copy.deepcopy(outcome.get("successor")),
        }
        for outcome in transition_outcomes(
            transition,
            error_type=WholeTurnProgramError,
        )
    ]
    return sorted(outcomes, key=canonical_json)


def _minimal_dependency_fields(
    worlds: Sequence[Mapping[str, Any]],
    semantic_hashes: Mapping[str, str],
    candidates: Sequence[str],
) -> tuple[str, ...]:
    fields = tuple(dict.fromkeys(str(field) for field in candidates))
    for size in range(len(fields) + 1):
        for candidate in itertools.combinations(fields, size):
            classes: dict[tuple[str, ...], str] = {}
            sufficient = True
            for world in worlds:
                world_id = str(world["world_id"])
                hidden = world["hidden"]
                key = tuple(canonical_json(hidden.get(field)) for field in candidate)
                semantic_hash = semantic_hashes[world_id]
                previous = classes.setdefault(key, semantic_hash)
                if previous != semantic_hash:
                    sufficient = False
                    break
            if sufficient:
                return candidate
    raise WholeTurnProgramError(
        "dependency candidates cannot explain whole-turn transition semantics"
    )


def _dynamic_transition_reads(
    transition: Mapping[str, Any],
    candidates: Sequence[str],
) -> tuple[str, ...] | None:
    allowed = set(candidates)
    reads: set[str] = set()
    for outcome in transition_outcomes(
        transition,
        error_type=WholeTurnProgramError,
    ):
        raw = outcome.get("transition_reads")
        if raw is None:
            return None
        if not isinstance(raw, list) or not all(isinstance(field, str) for field in raw):
            raise WholeTurnProgramError(
                "transition_reads must be an array of semantic hidden-field paths"
            )
        unknown = set(raw) - allowed
        if unknown:
            raise WholeTurnProgramError(
                f"transition_reads reference unknown dependency candidates: {sorted(unknown)!r}"
            )
        reads.update(raw)
    return tuple(sorted(reads))


def _read_refined_classes(
    worlds: Sequence[Mapping[str, Any]],
    *,
    action: str,
    transitions: Mapping[tuple[str, str], Mapping[str, Any]],
    semantic_hashes: Mapping[str, str],
    candidates: Sequence[str],
) -> tuple[tuple[str, ...], list[dict[str, Any]]] | None:
    world_by_id = {str(world["world_id"]): world for world in worlds}
    read_fields: dict[str, tuple[str, ...]] = {}
    for world_id in world_by_id:
        fields = _dynamic_transition_reads(
            transitions[(world_id, action)],
            candidates,
        )
        if fields is None:
            return None
        read_fields[world_id] = fields

    pending: list[list[str]] = [sorted(world_by_id)]
    classes: list[dict[str, Any]] = []
    while pending:
        member_ids = pending.pop()
        representative_id = member_ids[0]
        fields = read_fields[representative_id]

        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for world_id in member_ids:
            hidden = world_by_id[world_id]["hidden"]
            key = tuple(canonical_json(hidden.get(field)) for field in fields)
            groups[key].append(world_id)

        if len(groups) > 1:
            for _, group in sorted(groups.items(), reverse=True):
                pending.append(sorted(group))
            continue

        semantic_hash = semantic_hashes[representative_id]
        divergent = [
            world_id
            for world_id in member_ids
            if semantic_hashes[world_id] != semantic_hash
        ]
        if divergent:
            raise WholeTurnProgramError(
                f"{action}: instrumented transition reads are incomplete; "
                f"representative {representative_id} cannot distinguish "
                f"{divergent[0]}"
            )

        hidden = world_by_id[representative_id]["hidden"]
        key = tuple(canonical_json(hidden.get(field)) for field in fields)
        classes.append(
            {
                "read_fields": list(fields),
                "projection_key": list(key),
                "representative_world_id": representative_id,
                "member_world_ids": sorted(member_ids),
                "semantic_hash": semantic_hash,
            }
        )

    classes.sort(key=lambda row: row["representative_world_id"])
    union_fields = tuple(
        sorted(
            {
                field
                for row in classes
                for field in row["read_fields"]
            }
        )
    )
    return union_fields, classes


def compile_whole_turn_programs(
    oracle: Mapping[str, Any],
    *,
    partition_strategy: str = "semantic",
) -> dict[str, Any]:
    """Compile one exact finite-support transition program per legal root action.

    The semantic strategy is authoritative for analysis: classes are derived from
    complete immediate transition behavior. dynamic_reads is an explicit validation
    mode for representative refinement and is never inferred merely from trace presence.
    """

    if partition_strategy not in {"semantic", "dynamic_reads"}:
        raise WholeTurnProgramError(
            f"unsupported whole-turn partition strategy: {partition_strategy!r}"
        )

    worlds, actions, transitions, candidates = validate_oracle_core(
        oracle,
        error_type=WholeTurnProgramError,
    )
    programs: list[dict[str, Any]] = []

    for action in actions:
        immediate = {
            str(world["world_id"]): _immediate_distribution(
                transitions[(str(world["world_id"]), action)]
            )
            for world in worlds
        }
        semantic_hashes = {
            world_id: sha256_json(distribution)
            for world_id, distribution in immediate.items()
        }
        refined = (
            _read_refined_classes(
                worlds,
                action=action,
                transitions=transitions,
                semantic_hashes=semantic_hashes,
                candidates=candidates,
            )
            if partition_strategy == "dynamic_reads"
            else None
        )
        if partition_strategy == "dynamic_reads":
            if refined is None:
                raise WholeTurnProgramError(
                    f"{action}: dynamic-read partition requested but oracle has no "
                    "complete transition_reads evidence"
                )
            partition_method = "dynamic-read-refinement"
            dependency_fields, class_rows = refined
        else:
            partition_method = "finite-support-minimal-semantics"
            dependency_fields = _minimal_dependency_fields(
                worlds,
                semantic_hashes,
                candidates,
            )
            grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
            for world in worlds:
                hidden = world["hidden"]
                key = tuple(
                    canonical_json(hidden.get(field))
                    for field in dependency_fields
                )
                grouped[key].append(str(world["world_id"]))
            class_rows = [
                {
                    "read_fields": list(dependency_fields),
                    "projection_key": list(key),
                    "representative_world_id": sorted(member_ids)[0],
                    "member_world_ids": sorted(member_ids),
                    "semantic_hash": semantic_hashes[sorted(member_ids)[0]],
                }
                for key, member_ids in sorted(grouped.items(), key=lambda row: row[0])
            ]

        classes: list[dict[str, Any]] = []
        for row in class_rows:
            member_ids = list(row["member_world_ids"])
            representative_id = str(row["representative_world_id"])
            semantic_hash = str(row["semantic_hash"])
            if any(
                semantic_hashes[member_id] != semantic_hash
                for member_id in member_ids
            ):
                raise WholeTurnProgramError(
                    "whole-turn execution class contains non-equivalent worlds"
                )

            class_id = "transition-class-" + sha256_json(
                {
                    "action": action,
                    "read_fields": row["read_fields"],
                    "key": row["projection_key"],
                    "semantic_hash": semantic_hash,
                }
            )[:24]
            classes.append(
                {
                    "class_id": class_id,
                    "read_fields": list(row["read_fields"]),
                    "projection_key": list(row["projection_key"]),
                    "representative_world_id": representative_id,
                    "member_world_ids": member_ids,
                    "semantic_hash": semantic_hash,
                    "outcomes": copy.deepcopy(immediate[representative_id]),
                }
            )

        partition_key_hash = sha256_json(
            {
                "action": action,
                "partition_method": partition_method,
                "fields": dependency_fields,
                "classes": [
                    {
                        "class_id": row["class_id"],
                        "members": row["member_world_ids"],
                        "semantic_hash": row["semantic_hash"],
                    }
                    for row in classes
                ],
            }
        )
        effect_signature = "sha256:" + sha256_json(
            {
                "showdown_commit": oracle.get("showdown_commit"),
                "source_fixture_id": oracle.get("source_fixture_id"),
                "action": action,
                "dependency_fields": dependency_fields,
                "partition_key_hash": partition_key_hash,
            }
        )
        programs.append(
            {
                "action": action,
                "effect_signature": effect_signature,
                "dependency_fields": list(dependency_fields),
                "partition_method": partition_method,
                "representative_world_count": len(classes),
                "worlds_in": len(worlds),
                "classes_out": len(classes),
                "world_reduction": len(worlds) - len(classes),
                "reduction_fraction": 1.0 - len(classes) / len(worlds),
                "partition_key_hash": partition_key_hash,
                "classes": classes,
            }
        )

    return {
        "schema": PROGRAM_SET_SCHEMA,
        "schema_version": PROGRAM_SET_SCHEMA_VERSION,
        "source_fixture_id": oracle.get("source_fixture_id"),
        "showdown_commit": oracle.get("showdown_commit"),
        "world_ids": sorted(str(world["world_id"]) for world in worlds),
        "legal_actions": list(actions),
        "dependency_candidates": list(candidates),
        "programs": programs,
        "claim": (
            "For the supplied finite hidden-world support, every class member has "
            "the same complete immediate chance distribution, public observation, "
            "and successor state as its representative for the named root action."
        ),
        "non_claim": (
            "The compiled fields are not claimed sufficient for unseen hidden worlds, "
            "other actions, later turns, continuation values, or another Showdown revision."
        ),
    }


def verify_whole_turn_program_set(
    program_set: Mapping[str, Any],
    oracle: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify a representative-produced program against a complete direct oracle."""

    if (
        program_set.get("schema") != PROGRAM_SET_SCHEMA
        or program_set.get("schema_version") != PROGRAM_SET_SCHEMA_VERSION
    ):
        raise WholeTurnProgramError("unsupported whole-turn transition program schema")

    worlds, actions, transitions, candidates = validate_oracle_core(
        oracle,
        error_type=WholeTurnProgramError,
    )
    world_ids = sorted(str(world["world_id"]) for world in worlds)
    if program_set.get("source_fixture_id") != oracle.get("source_fixture_id"):
        raise WholeTurnProgramError("program and oracle belong to different fixtures")
    if program_set.get("showdown_commit") != oracle.get("showdown_commit"):
        raise WholeTurnProgramError("program and oracle use different Showdown revisions")
    if program_set.get("world_ids") != world_ids:
        raise WholeTurnProgramError("program and oracle have different hidden-world support")
    if program_set.get("legal_actions") != actions:
        raise WholeTurnProgramError("program and oracle have different legal actions")
    if program_set.get("dependency_candidates") != candidates:
        raise WholeTurnProgramError("program and oracle have different dependency candidates")

    programs = program_set.get("programs")
    if not isinstance(programs, list) or len(programs) != len(actions):
        raise WholeTurnProgramError("program set does not contain one program per action")

    verified_classes = 0
    representative_world_executions = 0
    for action in actions:
        program = program_for_action(program_set, action)
        raw_classes = program.get("classes")
        if not isinstance(raw_classes, list) or not raw_classes:
            raise WholeTurnProgramError(f"{action}: program has no execution classes")

        covered: set[str] = set()
        for row in raw_classes:
            if not isinstance(row, Mapping):
                raise WholeTurnProgramError(f"{action}: execution class must be an object")
            representative_id = str(row.get("representative_world_id"))
            raw_members = row.get("member_world_ids")
            raw_outcomes = row.get("outcomes")
            if not isinstance(raw_members, list) or not raw_members:
                raise WholeTurnProgramError(f"{action}: execution class has no members")
            if not isinstance(raw_outcomes, list) or not raw_outcomes:
                raise WholeTurnProgramError(f"{action}: execution class has no outcomes")
            members = [str(world_id) for world_id in raw_members]
            if representative_id not in members:
                raise WholeTurnProgramError(
                    f"{action}: class representative is not a class member"
                )
            overlap = covered.intersection(members)
            if overlap:
                raise WholeTurnProgramError(
                    f"{action}: execution classes overlap at {sorted(overlap)[0]}"
                )
            covered.update(members)

            expected = _immediate_distribution(
                transitions[(representative_id, action)]
            )
            supplied = sorted(
                [
                    {
                        "probability": float(outcome["probability"]),
                        "observation": copy.deepcopy(outcome.get("observation")),
                        "successor": copy.deepcopy(outcome.get("successor")),
                    }
                    for outcome in raw_outcomes
                    if isinstance(outcome, Mapping)
                ],
                key=canonical_json,
            )
            if len(supplied) != len(raw_outcomes) or supplied != expected:
                raise WholeTurnProgramError(
                    f"{action}: representative outcomes differ from direct oracle"
                )

            expected_hash = sha256_json(expected)
            if row.get("semantic_hash") != expected_hash:
                raise WholeTurnProgramError(
                    f"{action}: representative semantic hash differs from direct oracle"
                )
            for member_id in members:
                if member_id not in world_ids:
                    raise WholeTurnProgramError(
                        f"{action}: class references unknown world {member_id}"
                    )
                member = _immediate_distribution(transitions[(member_id, action)])
                if member != expected:
                    raise WholeTurnProgramError(
                        f"{action}: class merges semantically different world {member_id}"
                    )
            verified_classes += 1

        if covered != set(world_ids):
            raise WholeTurnProgramError(
                f"{action}: execution classes do not cover hidden-world support"
            )
        if int(program.get("classes_out", -1)) != len(raw_classes):
            raise WholeTurnProgramError(f"{action}: classes_out does not match program")
        if int(program.get("worlds_in", -1)) != len(world_ids):
            raise WholeTurnProgramError(f"{action}: worlds_in does not match oracle")
        representative_world_executions += len(raw_classes)

    exhaustive_world_action_product = len(world_ids) * len(actions)
    if representative_world_executions > exhaustive_world_action_product:
        raise WholeTurnProgramError(
            "representative execution count exceeds exhaustive world/action product"
        )
    return {
        "schema": VERIFICATION_SCHEMA,
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "source_fixture_id": oracle.get("source_fixture_id"),
        "showdown_commit": oracle.get("showdown_commit"),
        "world_count": len(world_ids),
        "action_count": len(actions),
        "verified_class_count": verified_classes,
        "representative_world_executions": representative_world_executions,
        "exhaustive_world_action_product": exhaustive_world_action_product,
        "saved_world_action_evaluations": (
            exhaustive_world_action_product - representative_world_executions
        ),
        "reduction_fraction": (
            1.0
            - representative_world_executions / exhaustive_world_action_product
        ),
        "program_digest": sha256_json(program_set),
        "oracle_digest": sha256_json(oracle),
        "claim": (
            "Every supplied execution class exactly matches the complete immediate "
            "transition distribution of every member in the direct oracle."
        ),
        "non_claim": (
            "Verification is limited to this finite hidden-world support, action set, "
            "chance-sample family, and pinned Showdown revision."
        ),
    }


def program_for_action(
    program_set: Mapping[str, Any],
    action: str,
) -> Mapping[str, Any]:
    raw_programs = program_set.get("programs")
    if not isinstance(raw_programs, list):
        raise WholeTurnProgramError("transition program set has no programs")
    matches = [
        program
        for program in raw_programs
        if isinstance(program, Mapping) and program.get("action") == action
    ]
    if len(matches) != 1:
        raise WholeTurnProgramError(
            f"expected one whole-turn program for action {action!r}"
        )
    return matches[0]


def execute_whole_turn_program(
    program_set: Mapping[str, Any],
    *,
    action: str,
    posterior: Mapping[str, float],
) -> dict[str, Any]:
    """Apply one compiled program to belief mass without re-executing class members."""

    if (
        program_set.get("schema") != PROGRAM_SET_SCHEMA
        or program_set.get("schema_version") != PROGRAM_SET_SCHEMA_VERSION
    ):
        raise WholeTurnProgramError("unsupported whole-turn transition program schema")

    world_ids = program_set.get("world_ids")
    if not isinstance(world_ids, list) or not all(
        isinstance(world_id, str) for world_id in world_ids
    ):
        raise WholeTurnProgramError("transition program world support is malformed")
    if set(posterior) != set(world_ids):
        raise WholeTurnProgramError(
            "posterior support differs from compiled whole-turn program"
        )

    normalized: dict[str, float] = {}
    total = 0.0
    for world_id in world_ids:
        weight = posterior[world_id]
        if not isinstance(weight, (int, float)) or float(weight) <= 0:
            raise WholeTurnProgramError("posterior weights must be positive")
        normalized[world_id] = float(weight)
        total += float(weight)
    if total <= 0:
        raise WholeTurnProgramError("posterior has no positive mass")
    normalized = {world_id: weight / total for world_id, weight in normalized.items()}

    program = program_for_action(program_set, action)
    raw_classes = program.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise WholeTurnProgramError("whole-turn program has no execution classes")

    edges: list[dict[str, Any]] = []
    covered: set[str] = set()
    for execution_class in raw_classes:
        if not isinstance(execution_class, Mapping):
            raise WholeTurnProgramError("whole-turn execution class must be an object")
        member_ids = execution_class.get("member_world_ids")
        outcomes = execution_class.get("outcomes")
        if not isinstance(member_ids, list) or not member_ids:
            raise WholeTurnProgramError("whole-turn execution class has no members")
        if not isinstance(outcomes, list) or not outcomes:
            raise WholeTurnProgramError("whole-turn execution class has no outcomes")
        members = [str(world_id) for world_id in member_ids]
        if covered.intersection(members):
            raise WholeTurnProgramError("whole-turn execution classes overlap")
        covered.update(members)
        class_mass = sum(normalized[world_id] for world_id in members)

        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                raise WholeTurnProgramError("whole-turn outcome must be an object")
            chance = float(outcome.get("probability", 0))
            if chance <= 0:
                raise WholeTurnProgramError("whole-turn outcome probability must be positive")
            edges.append(
                {
                    "class_id": execution_class.get("class_id"),
                    "representative_world_id": execution_class.get(
                        "representative_world_id"
                    ),
                    "member_world_ids": members,
                    "mass": class_mass * chance,
                    "observation": copy.deepcopy(outcome.get("observation")),
                    "successor": copy.deepcopy(outcome.get("successor")),
                }
            )

    if covered != set(world_ids):
        raise WholeTurnProgramError(
            "whole-turn execution classes do not cover compiled world support"
        )
    mass = sum(float(edge["mass"]) for edge in edges)
    if abs(mass - 1.0) > 1e-9:
        raise WholeTurnProgramError(
            f"weighted whole-turn outcomes sum to {mass}, not 1"
        )

    return {
        "schema": EXECUTION_SCHEMA,
        "schema_version": EXECUTION_SCHEMA_VERSION,
        "source_fixture_id": program_set.get("source_fixture_id"),
        "showdown_commit": program_set.get("showdown_commit"),
        "action": action,
        "effect_signature": program.get("effect_signature"),
        "logical_world_count": len(world_ids),
        "execution_class_count": len(raw_classes),
        "transition_evaluations": len(raw_classes),
        "edges": edges,
    }
