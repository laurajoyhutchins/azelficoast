"""Depth-one learned search directly over verified whole-turn TransitionPrograms."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

from azelficoast.belief.evaluator import (
    BeliefEvaluatorInput,
    BeliefEvaluatorSpec,
    BeliefPrediction,
    build_evaluator_input_for_contract,
)
from azelficoast.core.mechanics import (
    MechanicsExecutionRequest,
    MechanicsExecutor,
    MechanicsContractError,
)
from azelficoast.research.contracts import (
    BeliefInput,
    BeliefTransportIndex,
    PublicSuccessorState,
    ResearchContractError,
)
from azelficoast.core.program import program_for_action
from azelficoast.core.search import SEARCH_METHODS, SEARCH_SCHEMA, SEARCH_SCHEMA_VERSION
from azelficoast.core.transition import canonical_json

METHODS = SEARCH_METHODS


class SearchEvaluator(Protocol):
    spec: BeliefEvaluatorSpec

    def predict(self, inputs: BeliefEvaluatorInput) -> BeliefPrediction:
        """Predict from validated public and posterior features."""


class TransitionProgramSearchError(ValueError):
    """Raised when a TransitionProgram cannot support exact learned search."""


BudgetCheck = Callable[[], None]


def _check_budget(check_budget: BudgetCheck | None) -> None:
    if check_budget is not None:
        check_budget()


@dataclass
class _EvaluatorMeter:
    evaluator: SearchEvaluator
    calls: int = 0

    def value(
        self,
        *,
        public_state: PublicSuccessorState,
        posterior: BeliefInput,
        legal_actions: Sequence[str],
    ) -> float:
        if not legal_actions:
            raise TransitionProgramSearchError(
                "successor information set has no legal actions"
            )
        try:
            inputs = build_evaluator_input_for_contract(
                public_state=public_state,
                belief=posterior,
                legal_actions=legal_actions,
                spec=self.evaluator.spec,
            )
            prediction = self.evaluator.predict(inputs)
        except Exception as error:
            raise TransitionProgramSearchError(
                f"learned evaluator failed at successor leaf: {error}"
            ) from error
        self.calls += 1
        value = float(prediction.value)
        if not math.isfinite(value):
            raise TransitionProgramSearchError(
                "learned evaluator returned a non-finite successor value"
            )
        return value


def _normalized_inputs(
    *,
    mechanics: MechanicsExecutor,
    belief: BeliefInput,
    transport_index: BeliefTransportIndex,
    check_budget: BudgetCheck | None = None,
) -> tuple[
    list[str],
    dict[str, dict[str, Any]],
    dict[str, float],
    dict[str, str],
    dict[str, Any],
]:
    _check_budget(check_budget)
    world_ids = list(mechanics.world_ids)
    actions = list(mechanics.legal_actions)
    if len(set(world_ids)) != len(world_ids):
        raise TransitionProgramSearchError("transition-program world ids must be unique")
    if len(set(actions)) != len(actions):
        raise TransitionProgramSearchError("transition-program actions must be unique")

    semantic_worlds = {world.semantic_identity: world for world in belief.model_worlds}
    worlds_by_id: dict[str, dict[str, Any]] = {}
    raw_weights: dict[str, float] = {}
    semantic_by_transport: dict[str, str] = {}
    for world_id in world_ids:
        _check_budget(check_budget)
        semantic_identity = transport_index.semantic_identity_for(world_id)
        weight = transport_index.weight_for(world_id)
        semantic_world = semantic_worlds.get(str(semantic_identity))
        if semantic_world is None or weight is None:
            raise TransitionProgramSearchError(
                "posterior support differs from transition-program support"
            )
        row = semantic_world.to_record()
        row["world_id"] = world_id
        worlds_by_id[world_id] = row
        raw_weights[world_id] = weight
        semantic_by_transport[world_id] = semantic_world.semantic_identity

    if set(world_ids) != set(worlds_by_id):
        raise TransitionProgramSearchError(
            "posterior support differs from transition-program support"
        )

    total = math.fsum(raw_weights.values())
    if not math.isfinite(total) or total <= 0:
        raise TransitionProgramSearchError("posterior has no finite positive mass")
    weights = {world_id: raw_weights[world_id] / total for world_id in world_ids}
    for world_id, weight in weights.items():
        worlds_by_id[world_id]["weight"] = weight

    program_set = mechanics.program_set.to_record()
    raw_programs: list[dict[str, object]] = []
    try:
        for action in actions:
            _check_budget(check_budget)
            execution = mechanics.execute(
                MechanicsExecutionRequest(
                    mechanics_identity=mechanics.identity,
                    action=action,
                )
            )
            raw_programs.append(execution.program.to_record())
    except MechanicsContractError as error:
        raise TransitionProgramSearchError(str(error)) from error
    program_set["programs"] = raw_programs
    return actions, worlds_by_id, weights, semantic_by_transport, program_set


def _validated_classes(
    *,
    program_set: Mapping[str, Any],
    action: str,
    world_ids: set[str],
    check_budget: BudgetCheck | None = None,
) -> list[Mapping[str, Any]]:
    _check_budget(check_budget)
    program = program_for_action(program_set, action, error_type=TransitionProgramSearchError)
    raw_classes = program.get("classes")
    if not isinstance(raw_classes, list) or not raw_classes:
        raise TransitionProgramSearchError(f"{action}: transition program has no classes")

    covered: set[str] = set()
    classes: list[Mapping[str, Any]] = []
    for row in raw_classes:
        _check_budget(check_budget)
        if not isinstance(row, Mapping):
            raise TransitionProgramSearchError(f"{action}: execution class is not an object")
        raw_members = row.get("member_world_ids")
        raw_outcomes = row.get("outcomes")
        if (
            not isinstance(raw_members, list)
            or not raw_members
            or not all(isinstance(world_id, str) for world_id in raw_members)
        ):
            raise TransitionProgramSearchError(f"{action}: execution class has invalid members")
        if not isinstance(raw_outcomes, list) or not raw_outcomes:
            raise TransitionProgramSearchError(f"{action}: execution class has no outcomes")

        members = list(raw_members)
        overlap = covered.intersection(members)
        if overlap:
            raise TransitionProgramSearchError(
                f"{action}: execution classes overlap at {sorted(overlap)[0]}"
            )
        unknown = set(members) - world_ids
        if unknown:
            raise TransitionProgramSearchError(
                f"{action}: execution class references unknown world {sorted(unknown)[0]}"
            )
        covered.update(members)

        probability = 0.0
        for outcome in raw_outcomes:
            _check_budget(check_budget)
            if not isinstance(outcome, Mapping):
                raise TransitionProgramSearchError(
                    f"{action}: transition outcome is not an object"
                )
            chance = outcome.get("probability")
            if (
                not isinstance(chance, (int, float))
                or isinstance(chance, bool)
                or not math.isfinite(float(chance))
                or float(chance) <= 0
            ):
                raise TransitionProgramSearchError(
                    f"{action}: transition probabilities must be positive and finite"
                )
            probability += float(chance)
            if not isinstance(outcome.get("successor"), Mapping):
                raise TransitionProgramSearchError(
                    f"{action}: successor state must be public structured data"
                )
            legal = outcome.get("legal_actions")
            if (
                not isinstance(legal, list)
                or not legal
                or not all(isinstance(choice, str) and choice for choice in legal)
            ):
                raise TransitionProgramSearchError(
                    f"{action}: successor outcome has no legal-action surface"
                )
        if abs(probability - 1.0) > 1e-9:
            raise TransitionProgramSearchError(
                f"{action}: class outcome probabilities sum to {probability}, not 1"
            )
        classes.append(row)

    if covered != world_ids:
        raise TransitionProgramSearchError(
            f"{action}: transition classes do not cover posterior support"
        )
    if int(program.get("classes_out", -1)) != len(classes):
        raise TransitionProgramSearchError(
            f"{action}: classes_out does not match execution classes"
        )
    return classes


def _leaf_value(
    members: Sequence[Mapping[str, Any]],
    *,
    weight_key: str,
    worlds_by_id: Mapping[str, Mapping[str, Any]],
    semantic_by_transport: Mapping[str, str],
    belief: BeliefInput,
    evaluator: _EvaluatorMeter,
    check_budget: BudgetCheck | None = None,
) -> float:
    _check_budget(check_budget)
    if not members:
        raise TransitionProgramSearchError(
            "cannot evaluate an empty successor information set"
        )
    total = sum(float(member[weight_key]) for member in members)
    if total <= 0:
        raise TransitionProgramSearchError(
            "successor information set has no probability mass"
        )

    successor_by_digest = {
        canonical_json(member["outcome"]["successor"]): member["outcome"]["successor"]
        for member in members
    }
    if len(successor_by_digest) != 1:
        raise TransitionProgramSearchError(
            "one public observation mapped to multiple successor public states"
        )
    successor = next(iter(successor_by_digest.values()))

    legal_sets = [set(member["outcome"]["legal_actions"]) for member in members]
    common_legal = set(legal_sets[0])
    for legal in legal_sets[1:]:
        common_legal &= legal
    if not common_legal:
        raise TransitionProgramSearchError(
            "successor information set has no common legal action"
        )

    semantic_mass: dict[str, float] = defaultdict(float)
    for member in members:
        world_id = str(member["world_id"])
        if world_id not in worlds_by_id:
            raise TransitionProgramSearchError(
                f"successor references unknown hidden world {world_id}"
            )
        semantic_identity = semantic_by_transport.get(world_id)
        if semantic_identity is None:
            raise TransitionProgramSearchError("successor references unknown semantic world")
        semantic_mass[semantic_identity] += float(member[weight_key])

    try:
        public_successor = PublicSuccessorState.from_record(successor)
        posterior = belief.reweighted(semantic_mass)
    except ResearchContractError as error:
        raise TransitionProgramSearchError(str(error)) from error

    _check_budget(check_budget)
    return evaluator.value(
        public_state=public_successor,
        posterior=posterior,
        legal_actions=sorted(common_legal),
    )


def _determinization_values(
    *,
    program_set: Mapping[str, Any],
    actions: Sequence[str],
    worlds_by_id: Mapping[str, Mapping[str, Any]],
    weights: Mapping[str, float],
    semantic_by_transport: Mapping[str, str],
    belief: BeliefInput,
    evaluator: _EvaluatorMeter,
    check_budget: BudgetCheck | None = None,
) -> tuple[dict[str, float], int]:
    values: dict[str, float] = {}
    transition_evaluations = 0
    world_ids = set(worlds_by_id)

    for action in actions:
        _check_budget(check_budget)
        classes = _validated_classes(
            program_set=program_set,
            action=action,
            world_ids=world_ids,
            check_budget=check_budget,
        )
        transition_evaluations += len(classes)
        class_by_world = {
            world_id: row
            for row in classes
            for world_id in row["member_world_ids"]
        }

        total = 0.0
        for world_id in sorted(world_ids):
            _check_budget(check_budget)
            row = class_by_world[world_id]
            by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for outcome_index, outcome in enumerate(row["outcomes"]):
                _check_budget(check_budget)
                by_observation[canonical_json(outcome.get("observation"))].append(
                    {
                        "world_id": world_id,
                        "outcome_index": outcome_index,
                        "chance": float(outcome["probability"]),
                        "outcome": outcome,
                    }
                )

            world_value = 0.0
            for members in by_observation.values():
                _check_budget(check_budget)
                chance = sum(float(member["chance"]) for member in members)
                world_value += chance * _leaf_value(
                    members,
                    weight_key="chance",
                    worlds_by_id=worlds_by_id,
                    semantic_by_transport=semantic_by_transport,
                    belief=belief,
                    evaluator=evaluator,
                    check_budget=check_budget,
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
    semantic_by_transport: Mapping[str, str],
    belief: BeliefInput,
    evaluator: _EvaluatorMeter,
    check_budget: BudgetCheck | None = None,
) -> tuple[dict[str, float], int]:
    values: dict[str, float] = {}
    transition_evaluations = 0
    world_ids = set(worlds_by_id)

    for action in actions:
        _check_budget(check_budget)
        classes = _validated_classes(
            program_set=program_set,
            action=action,
            world_ids=world_ids,
            check_budget=check_budget,
        )
        transition_evaluations += len(classes)
        by_observation: dict[str, list[dict[str, Any]]] = defaultdict(list)

        for row in classes:
            _check_budget(check_budget)
            members = list(row["member_world_ids"])
            for outcome_index, outcome in enumerate(row["outcomes"]):
                _check_budget(check_budget)
                chance = float(outcome["probability"])
                observation = canonical_json(outcome.get("observation"))
                for world_id in members:
                    _check_budget(check_budget)
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
            _check_budget(check_budget)
            mass = sum(float(member["mass"]) for member in members)
            total += mass * _leaf_value(
                members,
                weight_key="mass",
                worlds_by_id=worlds_by_id,
                semantic_by_transport=semantic_by_transport,
                belief=belief,
                evaluator=evaluator,
                check_budget=check_budget,
            )
        values[action] = total

    return values, transition_evaluations


def search_transition_program(
    *,
    mechanics: MechanicsExecutor,
    belief: BeliefInput,
    transport_index: BeliefTransportIndex,
    method: str,
    evaluator: SearchEvaluator,
    check_budget: BudgetCheck | None = None,
) -> dict[str, Any]:
    """Evaluate a verified mechanics program without consuming an exhaustive oracle."""

    _check_budget(check_budget)
    if method not in METHODS:
        raise TransitionProgramSearchError(f"unknown search method {method!r}")
    actions, worlds_by_id, weights, semantic_by_transport, program_set = _normalized_inputs(
        mechanics=mechanics,
        belief=belief,
        transport_index=transport_index,
        check_budget=check_budget,
    )

    meter = _EvaluatorMeter(evaluator)
    if method == "determinization":
        root_values, transition_evaluations = _determinization_values(
            program_set=program_set,
            actions=actions,
            worlds_by_id=worlds_by_id,
            weights=weights,
            semantic_by_transport=semantic_by_transport,
            belief=belief,
            evaluator=meter,
            check_budget=check_budget,
        )
    else:
        root_values, transition_evaluations = _information_set_values(
            program_set=program_set,
            actions=actions,
            worlds_by_id=worlds_by_id,
            weights=weights,
            semantic_by_transport=semantic_by_transport,
            belief=belief,
            evaluator=meter,
            check_budget=check_budget,
        )

    _check_budget(check_budget)
    best = max(root_values.values())
    chosen_action = min(
        action for action, value in root_values.items() if value == best
    )
    return {
        "schema": SEARCH_SCHEMA,
        "schema_version": SEARCH_SCHEMA_VERSION,
        "method": method,
        "transition_program_digest": mechanics.transition_program_digest,
        "mechanics_evidence_digest": mechanics.semantic_evidence_digest,
        "transition_evaluations": transition_evaluations,
        "evaluator_calls": meter.calls,
        "chosen_action": chosen_action,
        "root_values": root_values,
    }
