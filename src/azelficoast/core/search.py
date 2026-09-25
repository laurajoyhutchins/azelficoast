"""Exact depth-one search over finite partial-information transition programs."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from azelficoast.core.contracts import BeliefEvaluator
from azelficoast.core.program import program_for_action
from azelficoast.core.transition import sha256_json

SEARCH_METHODS = ("determinization", "information_set")
SEARCH_SCHEMA = "azelficoast.partial-information-search"
SEARCH_SCHEMA_VERSION = 1


class PartialInformationSearchError(ValueError):
    """Raised when a transition program cannot support exact belief search."""


@dataclass
class _EvaluatorMeter:
    evaluator: BeliefEvaluator
    calls: int = 0

    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        if not legal_actions:
            raise PartialInformationSearchError(
                "successor information set has no legal actions"
            )
        try:
            value = float(
                self.evaluator.value(
                    public_state=public_state,
                    posterior_worlds=posterior_worlds,
                    legal_actions=legal_actions,
                )
            )
        except PartialInformationSearchError:
            raise
        except Exception as error:
            raise PartialInformationSearchError(
                f"evaluator failed at successor leaf: {error}"
            ) from error
        self.calls += 1
        if not math.isfinite(value):
            raise PartialInformationSearchError(
                "evaluator returned a non-finite successor value"
            )
        return value


def _normalized_inputs(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    expected_program_schema: str | None,
    expected_program_schema_version: int | None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, float]]:
    if (
        expected_program_schema is not None
        and program_set.get("schema") != expected_program_schema
    ):
        raise PartialInformationSearchError("unexpected transition-program schema")
    if (
        expected_program_schema_version is not None
        and program_set.get("schema_version") != expected_program_schema_version
    ):
        raise PartialInformationSearchError("unexpected transition-program schema")

    raw_world_ids = program_set.get("world_ids")
    raw_actions = program_set.get("legal_actions")
    raw_worlds = posterior.get("worlds")
    if (
        not isinstance(raw_world_ids, list)
        or not raw_world_ids
        or not all(isinstance(world_id, str) and world_id for world_id in raw_world_ids)
    ):
        raise PartialInformationSearchError(
            "transition program has invalid hidden-world support"
        )
    if (
        not isinstance(raw_actions, list)
        or not raw_actions
        or not all(isinstance(action, str) and action for action in raw_actions)
    ):
        raise PartialInformationSearchError(
            "transition program has no legal root actions"
        )
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise PartialInformationSearchError("posterior has no hidden-world support")

    world_ids = list(raw_world_ids)
    actions = list(raw_actions)
    if len(set(world_ids)) != len(world_ids):
        raise PartialInformationSearchError(
            "transition-program world ids must be unique"
        )
    if len(set(actions)) != len(actions):
        raise PartialInformationSearchError(
            "transition-program actions must be unique"
        )

    worlds_by_id: dict[str, dict[str, Any]] = {}
    raw_weights: dict[str, float] = {}
    for raw_world in raw_worlds:
        if not isinstance(raw_world, Mapping):
            raise PartialInformationSearchError("posterior world must be an object")
        world_id = str(raw_world.get("world_id"))
        if world_id in worlds_by_id:
            raise PartialInformationSearchError("posterior world ids must be unique")
        hidden = raw_world.get("hidden")
        if not isinstance(hidden, Mapping):
            raise PartialInformationSearchError(
                f"{world_id}: posterior must retain correlated hidden state"
            )
        weight = raw_world.get("weight")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise PartialInformationSearchError(
                f"{world_id}: posterior weight must be positive and finite"
            )
        worlds_by_id[world_id] = copy.deepcopy(dict(raw_world))
        raw_weights[world_id] = float(weight)

    if set(world_ids) != set(worlds_by_id):
        raise PartialInformationSearchError(
            "posterior support differs from transition-program support"
        )

    total = sum(raw_weights.values())
    weights = {world_id: raw_weights[world_id] / total for world_id in world_ids}
    for world_id, weight in weights.items():
        worlds_by_id[world_id]["weight"] = weight

    return actions, worlds_by_id, weights


def _validated_classes(
    *,
    program_set: Mapping[str, Any],
    action: str,
    world_ids: set[str],
) -> list[Mapping[str, Any]]:
    program = program_for_action(
        program_set,
        action,
        error_type=PartialInformationSearchError,
    )
    raw_classes = program.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise PartialInformationSearchError(
            f"{action}: transition program has no classes"
        )

    covered: set[str] = set()
    classes: list[Mapping[str, Any]] = []
    for row in raw_classes:
        if not isinstance(row, Mapping):
            raise PartialInformationSearchError(
                f"{action}: execution class is not an object"
            )
        raw_members = row.get("member_world_ids")
        raw_outcomes = row.get("outcomes")
        if (
            not isinstance(raw_members, list)
            or not raw_members
            or not all(isinstance(world_id, str) for world_id in raw_members)
        ):
            raise PartialInformationSearchError(
                f"{action}: execution class has invalid members"
            )
        if not isinstance(raw_outcomes, list) or not raw_outcomes:
            raise PartialInformationSearchError(
                f"{action}: execution class has no outcomes"
            )

        members = list(raw_members)
        overlap = covered.intersection(members)
        if overlap:
            raise PartialInformationSearchError(
                f"{action}: execution classes overlap at {sorted(overlap)[0]}"
            )
        unknown = set(members) - world_ids
        if unknown:
            raise PartialInformationSearchError(
                f"{action}: execution class references unknown world {sorted(unknown)[0]}"
            )
        covered.update(members)

        probability = 0.0
        for outcome in raw_outcomes:
            if not isinstance(outcome, Mapping):
                raise PartialInformationSearchError(
                    f"{action}: transition outcome is not an object"
                )
            chance = outcome.get("probability")
            if (
                not isinstance(chance, (int, float))
                or isinstance(chance, bool)
                or not math.isfinite(float(chance))
                or float(chance) <= 0
            ):
                raise PartialInformationSearchError(
                    f"{action}: transition probabilities must be positive and finite"
                )
            probability += float(chance)
            if not isinstance(outcome.get("successor"), Mapping):
                raise PartialInformationSearchError(
                    f"{action}: successor state must be public structured data"
                )
            legal = outcome.get("legal_actions")
            if (
                not isinstance(legal, list)
                or not legal
                or not all(isinstance(choice, str) and choice for choice in legal)
            ):
                raise PartialInformationSearchError(
                    f"{action}: successor outcome has no legal-action surface"
                )
        if abs(probability - 1.0) > 1e-9:
            raise PartialInformationSearchError(
                f"{action}: class outcome probabilities sum to {probability}, not 1"
            )
        classes.append(row)

    if covered != world_ids:
        raise PartialInformationSearchError(
            f"{action}: transition classes do not cover posterior support"
        )
    if int(program.get("classes_out", -1)) != len(classes):
        raise PartialInformationSearchError(
            f"{action}: classes_out does not match execution classes"
        )
    return classes


def _leaf_value(
    members: Sequence[Mapping[str, Any]],
    *,
    weight_key: str,
    worlds_by_id: Mapping[str, Mapping[str, Any]],
    evaluator: _EvaluatorMeter,
) -> float:
    if not members:
        raise PartialInformationSearchError(
            "cannot evaluate an empty successor information set"
        )
    total = sum(float(member[weight_key]) for member in members)
    if total <= 0:
        raise PartialInformationSearchError(
            "successor information set has no probability mass"
        )

    successor_by_digest = {
        repr(member["outcome"]["successor"]): member["outcome"]["successor"]
        for member in members
    }
    if len(successor_by_digest) != 1:
        raise PartialInformationSearchError(
            "one public observation mapped to multiple successor public states"
        )
    successor = next(iter(successor_by_digest.values()))

    legal_sets = [set(member["outcome"]["legal_actions"]) for member in members]
    common_legal = set(legal_sets[0])
    for legal in legal_sets[1:]:
        common_legal &= legal
    if not common_legal:
        raise PartialInformationSearchError(
            "successor information set has no common legal action"
        )

    world_mass: dict[str, float] = defaultdict(float)
    for member in members:
        world_id = str(member["world_id"])
        if world_id not in worlds_by_id:
            raise PartialInformationSearchError(
                f"successor references unknown hidden world {world_id}"
            )
        world_mass[world_id] += float(member[weight_key])

    posterior_worlds: list[dict[str, Any]] = []
    for world_id in sorted(world_mass):
        row = copy.deepcopy(dict(worlds_by_id[world_id]))
        row["weight"] = world_mass[world_id] / total
        posterior_worlds.append(row)

    return evaluator.value(
        public_state=successor,
        posterior_worlds=posterior_worlds,
        legal_actions=sorted(common_legal),
    )


def _determinization_values(
    *,
    program_set: Mapping[str, Any],
    actions: Sequence[str],
    worlds_by_id: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    evaluator: _EvaluatorMeter,
) -> tuple[dict[str, float], int]:
    values: dict[str, float] = {}
    transition_evaluations = 0
    world_ids = set(worlds_by_id)

    for action in actions:
        classes = _validated_classes(
            program_set=program_set,
            action=action,
            world_ids=world_ids,
        )
        transition_evaluations += len(classes)
        class_by_world = {
            world_id: row
            for row in classes
            for world_id in row["member_world_ids"]
        }

        total = 0.0
        for world_id in sorted(world_ids):
            row = class_by_world[world_id]
            by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for outcome_index, outcome in enumerate(row["outcomes"]):
                by_observation[repr(outcome.get("observation"))].append(
                    {
                        "world_id": world_id,
                        "outcome_index": outcome_index,
                        "chance": float(outcome["probability"]),
                        "outcome": outcome,
                    }
                )

            world_value = 0.0
            for members in by_observation.values():
                chance = sum(float(member["chance"]) for member in members)
                world_value += chance * _leaf_value(
                    members,
                    weight_key="chance",
                    worlds_by_id=worlds_by_id,
                    evaluator=evaluator,
                )
            total += weights[world_id] * world_value
        values[action] = total

    return values, transition_evaluations


def _information_set_values(
    *,
    program_set: Mapping[str, Any],
    actions: Sequence[str],
    worlds_by_id: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    evaluator: _EvaluatorMeter,
) -> tuple[dict[str, float], int]:
    values: dict[str, float] = {}
    transition_evaluations = 0
    world_ids = set(worlds_by_id)

    for action in actions:
        classes = _validated_classes(
            program_set=program_set,
            action=action,
            world_ids=world_ids,
        )
        transition_evaluations += len(classes)
        by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in classes:
            members = list(row["member_world_ids"])
            for outcome_index, outcome in enumerate(row["outcomes"]):
                chance = float(outcome["probability"])
                observation = repr(outcome.get("observation"))
                for world_id in members:
                    by_observation[observation].append(
                        {
                            "world_id": world_id,
                            "outcome_index": outcome_index,
                            "mass": weights[world_id] * chance,
                            "outcome": outcome,
                        }
                    )

        total = 0.0
        for members in by_observation.values():
            mass = sum(float(member["mass"]) for member in members)
            total += mass * _leaf_value(
                members,
                weight_key="mass",
                worlds_by_id=worlds_by_id,
                evaluator=evaluator,
            )
        values[action] = total

    return values, transition_evaluations


def search_transition_program(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    evaluator: BeliefEvaluator,
    expected_program_schema: str | None = None,
    expected_program_schema_version: int | None = None,
    result_schema: str = SEARCH_SCHEMA,
    result_schema_version: int = SEARCH_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Evaluate a finite transition program under one information-flow discipline."""

    if method not in SEARCH_METHODS:
        raise PartialInformationSearchError(f"unknown search method {method!r}")
    actions, worlds_by_id, weights = _normalized_inputs(
        program_set=program_set,
        posterior=posterior,
        expected_program_schema=expected_program_schema,
        expected_program_schema_version=expected_program_schema_version,
    )

    meter = _EvaluatorMeter(evaluator)
    if method == "determinization":
        root_values, transition_evaluations = _determinization_values(
            program_set=program_set,
            actions=actions,
            worlds_by_id=worlds_by_id,
            weights=weights,
            evaluator=meter,
        )
    else:
        root_values, transition_evaluations = _information_set_values(
            program_set=program_set,
            actions=actions,
            worlds_by_id=worlds_by_id,
            weights=weights,
            evaluator=meter,
        )

    best = max(root_values.values())
    chosen_action = min(
        action for action, value in root_values.items() if value == best
    )
    return {
        "schema": result_schema,
        "schema_version": result_schema_version,
        "method": method,
        "transition_program_digest": sha256_json(program_set),
        "transition_evaluations": transition_evaluations,
        "evaluator_calls": meter.calls,
        "chosen_action": chosen_action,
        "root_values": root_values,
    }
