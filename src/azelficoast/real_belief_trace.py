"""Information-set analysis over a pinned external mechanics oracle.

The mechanics oracle owns battle execution. This module owns only hidden-world
partition diagnostics, public-observation belief updates, and the policy constraint
that one continuation must serve every world in the same information set.
"""

from __future__ import annotations

import hashlib
import itertools
import json
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


def _continuation_values(outcome: Mapping[str, Any]) -> dict[str, float]:
    raw = outcome.get("continuations")
    if isinstance(raw, Mapping) and raw:
        return {str(action): float(value) for action, value in raw.items()}
    terminal = outcome.get("terminal_utility")
    if isinstance(terminal, (int, float)):
        return {"<terminal>": float(terminal)}
    raise BeliefTraceError("outcome has neither continuations nor terminal utility")


def _weighted_continuation_choice(
    members: Sequence[Mapping[str, Any]],
    *,
    weight_key: str,
) -> tuple[str, float]:
    """Choose once for an information set, averaging chance inside that set."""
    if not members:
        raise BeliefTraceError("cannot choose a continuation for an empty information set")

    continuation_maps = [
        _continuation_values(member["outcome"]) for member in members
    ]
    common = set(continuation_maps[0])
    for mapping in continuation_maps[1:]:
        common &= set(mapping)
    if not common:
        raise BeliefTraceError(
            "successor information set has no common legal continuation"
        )

    total_weight = sum(float(member[weight_key]) for member in members)
    if total_weight <= 0:
        raise BeliefTraceError("successor information set has no probability mass")

    values = {
        continuation: sum(
            float(member[weight_key])
            * _continuation_values(member["outcome"])[continuation]
            for member in members
        )
        / total_weight
        for continuation in sorted(common)
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

    legal_actions = [str(action) for action in raw_actions]
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

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "source_fixture_id": document.get("source_fixture_id"),
        "showdown_commit": document.get("showdown_commit"),
        "world_count": len(worlds),
        "legal_action_count": len(legal_actions),
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
