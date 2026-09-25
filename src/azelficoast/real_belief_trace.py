"""Information-set analysis over a pinned external mechanics oracle.

The mechanics oracle owns battle execution. This module owns only hidden-world
partition diagnostics, public-observation belief updates, and the policy constraint
that one continuation must serve every world in the same information set.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

SCHEMA = "azelficoast.real-belief-transition-oracle"
SCHEMA_VERSION = 1
RESULT_SCHEMA = "azelficoast.real-belief-decision-trace"
RESULT_SCHEMA_VERSION = 3


class BeliefTraceError(ValueError):
    """Raised when transition evidence is incomplete or internally inconsistent."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _minimal_dependency_fields(
    worlds: Sequence[Mapping[str, Any]],
    immediate_outcomes_by_world: Mapping[str, Any],
    candidates: Sequence[str],
) -> tuple[str, ...]:
    fields = tuple(dict.fromkeys(str(field) for field in candidates))
    hidden_fields = {key for world in worlds for key in world["hidden"]}
    unknown = [field for field in fields if field not in hidden_fields]
    if unknown:
        raise BeliefTraceError(f"dependency candidates reference unknown fields: {unknown!r}")

    def sufficient(candidate: tuple[str, ...]) -> bool:
        classes: dict[tuple[Any, ...], str] = {}
        for world in worlds:
            world_id = str(world["world_id"])
            hidden = world["hidden"]
            key = tuple(_canonical(hidden.get(field)) for field in candidate)
            outcome = _canonical(immediate_outcomes_by_world[world_id])
            previous = classes.setdefault(key, outcome)
            if previous != outcome:
                return False
        return True

    for size in range(len(fields) + 1):
        for candidate in itertools.combinations(fields, size):
            if sufficient(candidate):
                return candidate
    raise BeliefTraceError("no hidden-state dependency signature explains outcomes")


def _choose(values: Mapping[str, float]) -> tuple[str, float]:
    if not values:
        raise BeliefTraceError("policy has no legal action")
    best_value = max(values.values())
    best_action = min(action for action, value in values.items() if value == best_value)
    return best_action, best_value


def _outcomes(transition: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = transition.get("outcomes")
    if not isinstance(raw, list) or not raw:
        raise BeliefTraceError("every root transition must contain chance outcomes")
    total = 0.0
    outcomes: list[Mapping[str, Any]] = []
    for outcome in raw:
        if not isinstance(outcome, Mapping):
            raise BeliefTraceError("transition outcome must be an object")
        probability = outcome.get("probability")
        if not isinstance(probability, (int, float)) or probability <= 0:
            raise BeliefTraceError("transition outcome probabilities must be positive")
        total += float(probability)
        outcomes.append(outcome)
    if abs(total - 1.0) > 1e-9:
        raise BeliefTraceError(f"transition outcome probabilities sum to {total}, not 1")
    return outcomes


def _immediate_distribution(transition: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return mechanics-only outcomes for hidden-state partition validation.

    Continuation utilities are deliberately excluded. A Protect transition can be
    mechanically independent of a hidden item even when later decisions are not.
    """
    return [
        {
            "probability": float(outcome["probability"]),
            "observation": outcome.get("observation"),
            "successor": outcome.get("successor"),
        }
        for outcome in _outcomes(transition)
    ]


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
    if document.get("schema") != SCHEMA or document.get("schema_version") != SCHEMA_VERSION:
        raise BeliefTraceError("unsupported transition oracle schema")

    raw_worlds = document.get("worlds")
    raw_actions = document.get("legal_actions")
    raw_transitions = document.get("transitions")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise BeliefTraceError("oracle must contain at least one hidden world")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise BeliefTraceError("oracle must contain root legal actions")
    if not isinstance(raw_transitions, list):
        raise BeliefTraceError("oracle transitions must be a list")

    worlds = [dict(world) for world in raw_worlds]
    world_by_id = {str(world["world_id"]): world for world in worlds}
    if len(world_by_id) != len(worlds):
        raise BeliefTraceError("world ids must be unique")
    if any(float(world.get("weight", 0)) <= 0 for world in worlds):
        raise BeliefTraceError("world weights must be positive")
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

    legal_actions = [str(action) for action in raw_actions]
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
    raw_candidates = document.get("dependency_candidates")
    if raw_candidates is None:
        dependency_candidates = sorted({key for world in worlds for key in world["hidden"]})
    elif isinstance(raw_candidates, list) and all(isinstance(field, str) for field in raw_candidates):
        dependency_candidates = list(dict.fromkeys(raw_candidates))
    else:
        raise BeliefTraceError("dependency_candidates must be an array of semantic field paths")

    raw_declared = document.get("declared_reads", {})
    if not isinstance(raw_declared, Mapping):
        raise BeliefTraceError("declared_reads must be an action-to-fields object")

    transitions: dict[tuple[str, str], Mapping[str, Any]] = {}
    for transition in raw_transitions:
        if not isinstance(transition, Mapping):
            raise BeliefTraceError("transition must be an object")
        world_id = str(transition.get("world_id"))
        action = str(transition.get("action"))
        key = (world_id, action)
        if world_id not in world_by_id:
            raise BeliefTraceError(f"transition references unknown world {world_id}")
        if action not in legal_actions:
            raise BeliefTraceError(f"transition references non-root action {action}")
        if key in transitions:
            raise BeliefTraceError(f"duplicate transition for {world_id} {action}")
        _outcomes(transition)
        transitions[key] = transition

    expected = {(world_id, action) for world_id in world_by_id for action in legal_actions}
    missing = expected - set(transitions)
    if missing:
        raise BeliefTraceError(f"oracle omitted root transitions: {sorted(missing)[:3]!r}")

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

    action_reports: list[dict[str, Any]] = []
    determinization_values: dict[str, float] = {}
    public_values: dict[str, float] = {}

    for action in legal_actions:
        action_transitions = {
            world_id: transitions[(world_id, action)] for world_id in world_by_id
        }
        immediate_by_world = {
            world_id: _immediate_distribution(transition)
            for world_id, transition in action_transitions.items()
        }
        dependency_fields = _minimal_dependency_fields(
            worlds, immediate_by_world, dependency_candidates
        )

        declared_for_action = raw_declared.get(action, dependency_candidates)
        if not isinstance(declared_for_action, list) or not all(
            isinstance(field, str) for field in declared_for_action
        ):
            raise BeliefTraceError(f"{action}: declared read set is malformed")
        declared = list(dict.fromkeys(declared_for_action))
        missing_declared = [field for field in dependency_fields if field not in declared]
        if missing_declared:
            raise BeliefTraceError(
                f"{action}: empirical dependency fields missing from declaration: {missing_declared!r}"
            )

        dependency_classes: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for world in worlds:
            hidden = world["hidden"]
            key = tuple(_canonical(hidden.get(field)) for field in dependency_fields)
            dependency_classes[key].append(str(world["world_id"]))

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

        action_reports.append(
            {
                "action": action,
                "dependency_signature": {
                    "declared_reads": declared,
                    "empirically_required_reads": list(dependency_fields),
                    "partition_key_hash": _sha256(
                        {
                            "fields": dependency_fields,
                            "classes": sorted(sorted(ids) for ids in dependency_classes.values()),
                        }
                    ),
                    "worlds_in": len(worlds),
                    "classes_out": len(dependency_classes),
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


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    args = parser.parse_args(argv)
    document = json.loads(args.oracle.read_text(encoding="utf-8"))
    result = analyze_oracle(document)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
