"""Information-set analysis over a pinned external mechanics oracle.

The mechanics oracle owns battle execution. This module owns only hidden-world
partition diagnostics, public-observation belief updates, and the policy constraint
that one continuation must serve every world in the same information set.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from azelficoast.core.decision_relevance import decision_relevance_quotient
from azelficoast.core.program import program_for_action
from azelficoast.core.transition import (
    ORACLE_SCHEMA,
    ORACLE_SCHEMA_VERSION,
    canonical_json,
    transition_outcomes,
    validate_transition_oracle,
)
from azelficoast.core.whole_turn_program import compile_whole_turn_programs

SCHEMA = ORACLE_SCHEMA
SCHEMA_VERSION = ORACLE_SCHEMA_VERSION
_canonical = canonical_json
RESULT_SCHEMA = "azelficoast.real-belief-decision-trace"
RESULT_SCHEMA_VERSION = 3


class BeliefTraceError(ValueError):
    """Raised when transition evidence is incomplete or internally inconsistent."""


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _choose(values: Mapping[str, float]) -> tuple[str, float]:
    if not values:
        raise BeliefTraceError("policy has no legal action")
    best_value = max(values.values())
    best_action = min(action for action, value in values.items() if value == best_value)
    return best_action, best_value


def _outcomes(transition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return transition_outcomes(
        transition,
        error_type=BeliefTraceError,
    )


def _leaf_continuation_values(outcome: Mapping[str, Any]) -> dict[str, float]:
    raw = outcome.get("continuations")
    if isinstance(raw, Mapping) and raw:
        values: dict[str, float] = {}
        for action, value in raw.items():
            if not isinstance(value, (int, float)):
                raise BeliefTraceError("continuation utility must be numeric")
            values[str(action)] = float(value)
        return values
    terminal = outcome.get("terminal_utility")
    if isinstance(terminal, (int, float)):
        return {"<terminal>": float(terminal)}
    raise BeliefTraceError("leaf outcome has neither continuations nor terminal utility")


def _continuation_actions(outcome: Mapping[str, Any]) -> set[str]:
    shallow = outcome.get("continuations")
    deep = outcome.get("continuation_transitions")
    terminal = outcome.get("terminal_utility")

    if deep is not None:
        if shallow is not None or terminal is not None:
            raise BeliefTraceError(
                "outcome cannot mix continuation_transitions with leaf utility"
            )
        if not isinstance(deep, Mapping) or not deep:
            raise BeliefTraceError("continuation_transitions must be a non-empty object")
        actions = {str(action) for action in deep}
        for action in actions:
            _nested_outcomes(outcome, action)
        return actions

    return set(_leaf_continuation_values(outcome))


def _nested_outcomes(
    outcome: Mapping[str, Any],
    action: str,
) -> list[Mapping[str, Any]]:
    raw = outcome.get("continuation_transitions")
    if not isinstance(raw, Mapping):
        raise BeliefTraceError("outcome does not contain deeper continuation transitions")
    branch = raw.get(action)
    if not isinstance(branch, list) or not branch:
        raise BeliefTraceError(
            f"deeper continuation {action!r} must contain chance outcomes"
        )

    total = 0.0
    nested: list[Mapping[str, Any]] = []
    for next_outcome in branch:
        if not isinstance(next_outcome, Mapping):
            raise BeliefTraceError("deeper continuation outcome must be an object")
        if next_outcome.get("continuation_transitions") is not None:
            raise BeliefTraceError(
                "deeper continuation exceeds the supported extra decision horizon"
            )
        probability = next_outcome.get("probability")
        if not isinstance(probability, (int, float)) or probability <= 0:
            raise BeliefTraceError(
                "deeper continuation probabilities must be positive"
            )
        _leaf_continuation_values(next_outcome)
        total += float(probability)
        nested.append(next_outcome)
    if abs(total - 1.0) > 1e-9:
        raise BeliefTraceError(
            f"deeper continuation probabilities sum to {total}, not 1"
        )
    return nested


def _weighted_leaf_choice(
    members: Sequence[Mapping[str, Any]],
    *,
    weight_key: str,
) -> tuple[str, float]:
    maps = [_leaf_continuation_values(member["outcome"]) for member in members]
    common = set(maps[0])
    for mapping in maps[1:]:
        common &= set(mapping)
    if not common:
        raise BeliefTraceError(
            "deeper successor information set has no common legal continuation"
        )

    total_weight = sum(float(member[weight_key]) for member in members)
    if total_weight <= 0:
        raise BeliefTraceError("deeper successor information set has no probability mass")

    values = {
        action: sum(
            float(member[weight_key])
            * _leaf_continuation_values(member["outcome"])[action]
            for member in members
        )
        / total_weight
        for action in sorted(common)
    }
    return _choose(values)


def _weighted_deeper_value(
    members: Sequence[Mapping[str, Any]],
    *,
    weight_key: str,
    action: str,
) -> float:
    total_weight = sum(float(member[weight_key]) for member in members)
    if total_weight <= 0:
        raise BeliefTraceError("successor information set has no probability mass")

    observation_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in members:
        member_weight = float(member[weight_key])
        for next_outcome in _nested_outcomes(member["outcome"], action):
            probability = float(next_outcome["probability"])
            observation_key = _canonical(next_outcome.get("observation"))
            observation_members[observation_key].append(
                {
                    "weight": member_weight * probability,
                    "outcome": next_outcome,
                }
            )

    expected = 0.0
    for nested_members in observation_members.values():
        group_weight = sum(float(member["weight"]) for member in nested_members)
        _, value = _weighted_leaf_choice(nested_members, weight_key="weight")
        expected += group_weight * value
    return expected / total_weight


def _weighted_continuation_choice(
    members: Sequence[Mapping[str, Any]],
    *,
    weight_key: str,
) -> tuple[str, float]:
    """Choose once per public information set, at one or two continuation horizons."""
    if not members:
        raise BeliefTraceError("cannot choose a continuation for an empty information set")

    action_sets = [_continuation_actions(member["outcome"]) for member in members]
    common = set(action_sets[0])
    for actions in action_sets[1:]:
        common &= actions
    if not common:
        raise BeliefTraceError(
            "successor information set has no common legal continuation"
        )

    modes = {
        member["outcome"].get("continuation_transitions") is not None
        for member in members
    }
    if len(modes) != 1:
        raise BeliefTraceError(
            "successor information set mixes shallow and deeper continuation evidence"
        )
    deeper = next(iter(modes))

    if deeper:
        values = {
            action: _weighted_deeper_value(
                members,
                weight_key=weight_key,
                action=action,
            )
            for action in sorted(common)
        }
    else:
        total_weight = sum(float(member[weight_key]) for member in members)
        if total_weight <= 0:
            raise BeliefTraceError("successor information set has no probability mass")
        values = {
            action: sum(
                float(member[weight_key])
                * _leaf_continuation_values(member["outcome"])[action]
                for member in members
            )
            / total_weight
            for action in sorted(common)
        }
    return _choose(values)


def analyze_oracle(document: Mapping[str, Any]) -> dict[str, Any]:
    worlds, legal_actions, transitions, dependency_candidates = (
        validate_transition_oracle(
            document,
            error_type=BeliefTraceError,
        )
    )
    world_by_id = {str(world["world_id"]): world for world in worlds}
    total_world_weight = sum(float(world["weight"]) for world in worlds)

    raw_factors = document.get("factored_hidden", {})
    if not isinstance(raw_factors, Mapping):
        raise BeliefTraceError("factored_hidden must be an object")

    concrete_hidden_fields = {key for world in worlds for key in world["hidden"]}
    factored_hidden: dict[str, dict[str, Any]] = {}
    for raw_field, raw_factor in raw_factors.items():
        field = str(raw_field)
        if field in concrete_hidden_fields:
            raise BeliefTraceError(
                f"factored hidden field {field!r} is also materialized in worlds"
            )
        if not isinstance(raw_factor, Mapping):
            raise BeliefTraceError(f"factored hidden field {field!r} must be an object")
        distribution = raw_factor.get("distribution")
        if not isinstance(distribution, list) or not distribution:
            raise BeliefTraceError(
                f"factored hidden field {field!r} must have a non-empty distribution"
            )

        seen_values: set[str] = set()
        total_factor_weight = 0.0
        normalized_distribution: list[dict[str, Any]] = []
        for entry in distribution:
            if not isinstance(entry, Mapping) or "value" not in entry:
                raise BeliefTraceError(
                    f"factored hidden field {field!r} has a malformed distribution entry"
                )
            weight = entry.get("weight")
            if not isinstance(weight, (int, float)) or float(weight) <= 0:
                raise BeliefTraceError(
                    f"factored hidden field {field!r} weights must be positive"
                )
            value_key = _canonical(entry["value"])
            if value_key in seen_values:
                raise BeliefTraceError(
                    f"factored hidden field {field!r} has duplicate values"
                )
            seen_values.add(value_key)
            total_factor_weight += float(weight)
            normalized_distribution.append(
                {"value": entry["value"], "weight": float(weight)}
            )
        if abs(total_factor_weight - 1.0) > 1e-9:
            raise BeliefTraceError(
                f"factored hidden field {field!r} weights sum to "
                f"{total_factor_weight}, not 1"
            )

        unread_actions = raw_factor.get("unread_actions")
        if not isinstance(unread_actions, list) or not all(
            isinstance(action, str) for action in unread_actions
        ):
            raise BeliefTraceError(
                f"factored hidden field {field!r} must declare unread_actions"
            )
        unread = list(dict.fromkeys(unread_actions))
        evidence = raw_factor.get("evidence")
        if evidence is not None and not isinstance(evidence, Mapping):
            raise BeliefTraceError(
                f"factored hidden field {field!r} evidence must be an object"
            )
        factored_hidden[field] = {
            "distribution": normalized_distribution,
            "unread_actions": unread,
            "evidence": dict(evidence or {}),
        }

    for field, factor in factored_hidden.items():
        unread = set(factor["unread_actions"])
        unknown_actions = unread - set(legal_actions)
        if unknown_actions:
            raise BeliefTraceError(
                f"factored hidden field {field!r} names unknown actions: "
                f"{sorted(unknown_actions)!r}"
            )
        requiring_expansion = [
            action for action in legal_actions if action not in unread
        ]
        if requiring_expansion:
            raise BeliefTraceError(
                f"factored hidden field {field!r} is read by actions "
                f"{requiring_expansion!r}; materialize that factor before analysis"
            )
    raw_declared = document.get("declared_reads", {})
    if not isinstance(raw_declared, Mapping):
        raise BeliefTraceError("declared_reads must be an action-to-fields object")

    for field, factor in factored_hidden.items():
        unread = set(factor["unread_actions"])
        for action in unread:
            for world_id in world_by_id:
                for outcome in _outcomes(transitions[(world_id, action)]):
                    raw_reads = outcome.get("hidden_reads")
                    if not isinstance(raw_reads, list) or not all(
                        isinstance(read, str) for read in raw_reads
                    ):
                        raise BeliefTraceError(
                            f"{action}: factored hidden field {field!r} requires "
                            "an explicit per-outcome hidden_reads witness"
                        )
                    if field in raw_reads:
                        raise BeliefTraceError(
                            f"{action}: hidden_reads proves factored field "
                            f"{field!r} was read"
                        )

    whole_turn_program_set = compile_whole_turn_programs(document)

    action_reports: list[dict[str, Any]] = []
    determinization_values: dict[str, float] = {}
    public_values: dict[str, float] = {}

    for action in legal_actions:
        action_transitions = {
            world_id: transitions[(world_id, action)] for world_id in world_by_id
        }
        transition_program = program_for_action(whole_turn_program_set, action)
        dependency_fields = tuple(
            str(field) for field in transition_program["dependency_fields"]
        )

        declared_for_action = raw_declared.get(action, dependency_candidates)
        if not isinstance(declared_for_action, list) or not all(
            isinstance(field, str) for field in declared_for_action
        ):
            raise BeliefTraceError(f"{action}: declared read set is malformed")
        declared = list(dict.fromkeys(declared_for_action))
        missing_declared = [field for field in dependency_fields if field not in declared]

        # Observation classes contain joint hidden-world/chance mass. Chance is
        # evidence, not a hidden fact the determinization baseline knows in advance.
        observation_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
        det_value = 0.0
        det_choices: list[dict[str, Any]] = []
        for world in worlds:
            world_id = str(world["world_id"])
            prior = float(world["weight"]) / total_world_weight
            transition = action_transitions[world_id]
            world_observation_members: dict[str, list[dict[str, Any]]] = defaultdict(list)

            for outcome_index, outcome in enumerate(_outcomes(transition)):
                chance = float(outcome["probability"])
                observation_key = _canonical(outcome.get("observation"))
                member = {
                    "world_id": world_id,
                    "chance": chance,
                    "mass": prior * chance,
                    "outcome_index": outcome_index,
                    "outcome": outcome,
                }
                observation_members[observation_key].append(member)
                world_observation_members[observation_key].append(member)

            world_expected = 0.0
            for observation_key, members in sorted(world_observation_members.items()):
                group_chance = sum(float(member["chance"]) for member in members)
                choice, value = _weighted_continuation_choice(
                    members,
                    weight_key="chance",
                )
                world_expected += group_chance * value
                det_choices.append(
                    {
                        "world_id": world_id,
                        "outcome_indices": [
                            int(member["outcome_index"]) for member in members
                        ],
                        "observation_hash": hashlib.sha256(
                            observation_key.encode()
                        ).hexdigest(),
                        "choice": choice,
                        "value": value,
                    }
                )
            det_value += prior * world_expected
        determinization_values[action] = det_value

        public_total = 0.0
        successor_beliefs: list[dict[str, Any]] = []
        public_choices: list[dict[str, Any]] = []
        for observation_key, members in sorted(observation_members.items()):
            group_mass = sum(float(member["mass"]) for member in members)
            if group_mass <= 0:
                raise BeliefTraceError("public observation class has no probability mass")

            by_world_mass: dict[str, float] = defaultdict(float)
            by_world_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for member in members:
                world_id = str(member["world_id"])
                by_world_mass[world_id] += float(member["mass"])
                by_world_members[world_id].append(member)

            choice, value = _weighted_continuation_choice(
                members,
                weight_key="mass",
            )
            public_total += group_mass * value

            observation_hash = hashlib.sha256(observation_key.encode()).hexdigest()
            successor_beliefs.append(
                {
                    "observation_hash": observation_hash,
                    "probability": group_mass,
                    "members": [
                        {
                            "world_id": world_id,
                            "posterior": mass / group_mass,
                        }
                        for world_id, mass in sorted(by_world_mass.items())
                    ],
                }
            )
            world_aware_choices = sorted(
                {
                    _weighted_continuation_choice(
                        world_members,
                        weight_key="mass",
                    )[0]
                    for world_members in by_world_members.values()
                }
            )
            public_choices.append(
                {
                    "observation_hash": observation_hash,
                    "choice": choice,
                    "value": value,
                    "world_ids": sorted(by_world_mass),
                    "world_aware_choices": world_aware_choices,
                    "world_aware_choice_count": len(world_aware_choices),
                    "strategy_fusion_possible": len(world_aware_choices) > 1,
                }
            )
        public_values[action] = public_total

        strategy_fusion_observation_count = sum(
            choice["strategy_fusion_possible"] for choice in public_choices
        )
        max_world_aware_choices_per_observation = max(
            choice["world_aware_choice_count"] for choice in public_choices
        )

        if missing_declared:
            raise BeliefTraceError(
                f"{action}: empirical dependency fields missing from declaration: {missing_declared!r}"
            )

        action_reports.append(
            {
                "action": action,
                "dependency_signature": {
                    "declared_reads": declared,
                    "empirically_required_reads": list(dependency_fields),
                    "partition_key_hash": transition_program["partition_key_hash"],
                    "effect_signature": transition_program["effect_signature"],
                    "worlds_in": transition_program["worlds_in"],
                    "classes_out": transition_program["classes_out"],
                    "observable_classes_out": len(observation_members),
                    "marginalized_hidden_factors": sorted(factored_hidden),
                },
                "successor_beliefs": successor_beliefs,
                "determinization_continuations": det_choices,
                "public_belief_continuations": public_choices,
                "strategy_fusion_observation_count": strategy_fusion_observation_count,
                "max_world_aware_choices_per_observation": (
                    max_world_aware_choices_per_observation
                ),
            }
        )

    determinization_action, determinization_value = _choose(determinization_values)
    public_action, public_value = _choose(public_values)
    policy_disagreement = determinization_action != public_action
    strategy_fusion_observation_count = sum(
        int(report["strategy_fusion_observation_count"])
        for report in action_reports
    )
    max_world_aware_choices_per_observation = max(
        int(report["max_world_aware_choices_per_observation"])
        for report in action_reports
    )

    continuation_decision_horizons = max(
        (
            2
            if outcome.get("continuation_transitions") is not None
            else 1
            if outcome.get("continuations") is not None
            else 0
        )
        for transition in transitions.values()
        for outcome in _outcomes(transition)
    )

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "source_fixture_id": document.get("source_fixture_id"),
        "showdown_commit": document.get("showdown_commit"),
        "world_count": len(worlds),
        "materialized_world_count": len(worlds),
        "latent_world_count": len(worlds)
        * math.prod(
            len(factor["distribution"]) for factor in factored_hidden.values()
        ),
        "factoring_ratio": math.prod(
            len(factor["distribution"]) for factor in factored_hidden.values()
        ),
        "factored_hidden": {
            field: {
                "support_count": len(factor["distribution"]),
                "distribution_sha256": _sha256(factor["distribution"]),
                "unread_actions": factor["unread_actions"],
                "evidence": factor["evidence"],
            }
            for field, factor in sorted(factored_hidden.items())
        },
        "legal_action_count": len(legal_actions),
        "continuation_decision_horizons": continuation_decision_horizons,
        "actions": action_reports,
        "determinization": {
            "chosen_action": determinization_action,
            "value": determinization_value,
            "root_values": determinization_values,
        },
        "public_belief": {
            "chosen_action": public_action,
            "value": public_value,
            "root_values": public_values,
        },
        "policy_disagreement": policy_disagreement,
        "strategy_fusion_observation_count": strategy_fusion_observation_count,
        "max_world_aware_choices_per_observation": (
            max_world_aware_choices_per_observation
        ),
        "hypothesis_supported": policy_disagreement,
        "experiment_valid": True,
    }


def analyze_quotiented_oracle(
    document: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Analyze the exact decision-relevance quotient with the battle trace analyzer."""

    certificate, quotient = decision_relevance_quotient(document)
    trace = analyze_oracle(quotient)
    trace["source_world_count"] = certificate["worlds_in"]
    trace["decision_relevance"] = {
        key: copy.deepcopy(value)
        for key, value in certificate.items()
        if key != "classes"
    }
    return trace, certificate


class _JsonStreamReader:
    """Incremental JSON reader that materializes selected top-level arrays only."""

    def __init__(self, source: TextIO, chunk_size: int) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._source = source
        self._chunk_size = chunk_size
        self._decoder = json.JSONDecoder()
        self._buffer = ""
        self._position = 0

    def _fill(self) -> bool:
        remaining = self._buffer[self._position :]
        chunk = self._source.read(self._chunk_size)
        self._buffer = remaining + chunk
        self._position = 0
        return bool(chunk)

    def _skip_whitespace(self) -> None:
        while True:
            while (
                self._position < len(self._buffer)
                and self._buffer[self._position] in " \t\r\n"
            ):
                self._position += 1
            if self._position < len(self._buffer) or not self._fill():
                return

    def _peek(self) -> str | None:
        self._skip_whitespace()
        if self._position >= len(self._buffer):
            return None
        return self._buffer[self._position]

    def _consume(self, expected: str) -> None:
        actual = self._peek()
        if actual != expected:
            raise BeliefTraceError(
                f"malformed oracle JSON: expected {expected!r}, got {actual!r}"
            )
        self._position += 1

    def _read_value(self) -> Any:
        self._skip_whitespace()
        while True:
            try:
                value, end = self._decoder.raw_decode(self._buffer, self._position)
            except json.JSONDecodeError as error:
                if self._fill():
                    continue
                raise BeliefTraceError("malformed or truncated oracle JSON") from error

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                extension_characters = "0123456789.eE+-"
                if end == len(self._buffer):
                    if self._fill():
                        continue
                elif self._buffer[end] in extension_characters:
                    if self._fill():
                        continue

            self._position = end
            self._buffer = self._buffer[self._position :]
            self._position = 0
            return value

    def _read_array(self) -> list[Any]:
        self._consume("[")
        values: list[Any] = []
        if self._peek() == "]":
            self._consume("]")
            return values
        while True:
            values.append(self._read_value())
            separator = self._peek()
            if separator == "]":
                self._consume("]")
                return values
            if separator != ",":
                raise BeliefTraceError("malformed oracle JSON array separator")
            self._consume(",")

    def read_document(self) -> dict[str, Any]:
        self._consume("{")
        document: dict[str, Any] = {}
        if self._peek() == "}":
            self._consume("}")
            return document
        while True:
            key = self._read_value()
            if not isinstance(key, str):
                raise BeliefTraceError("oracle JSON object keys must be strings")
            self._consume(":")
            if key in {"worlds", "transitions"} and self._peek() == "[":
                document[key] = self._read_array()
            else:
                document[key] = self._read_value()

            separator = self._peek()
            if separator == "}":
                self._consume("}")
                break
            if separator != ",":
                raise BeliefTraceError("malformed oracle JSON object separator")
            self._consume(",")

        if self._peek() is not None:
            raise BeliefTraceError("unexpected data after oracle JSON document")
        return document


def _load_oracle_document(
    path: str | Path,
    *,
    chunk_size: int = 64 * 1024,
) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as source:
        return _JsonStreamReader(source, chunk_size).read_document()


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument(
        "--quotient",
        action="store_true",
        help="analyze the exact decision-relevance quotient before tracing",
    )
    args = parser.parse_args(argv)
    document = _load_oracle_document(args.oracle)
    if args.quotient:
        trace, certificate = analyze_quotiented_oracle(document)
        result = {
            "schema": "azelficoast.real-belief-decision-relevance-analysis",
            "schema_version": 1,
            "certificate": certificate,
            "quotient_trace": trace,
        }
    else:
        result = analyze_oracle(document)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
