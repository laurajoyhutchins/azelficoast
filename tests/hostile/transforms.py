"""Independent adversarial transformations for JSON-like test artifacts."""

from __future__ import annotations

import copy
import json
import math
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any


TRANSPORT_FIELDS = frozenset({"world_id", "id", "transport_id"})


def _posterior_copy(posterior: Mapping[str, Any]) -> dict[str, Any]:
    copied = copy.deepcopy(dict(posterior))
    worlds = copied.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise ValueError("posterior must contain a non-empty worlds list")
    if any(not isinstance(world, Mapping) for world in worlds):
        raise ValueError("posterior worlds must be objects")
    return copied


def rename_world_ids(
    posterior: Mapping[str, Any], identifiers: Sequence[str | int]
) -> dict[str, Any]:
    result = _posterior_copy(posterior)
    worlds = result["worlds"]
    if len(identifiers) != len(worlds):
        raise ValueError("replacement identifiers must cover every world")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("replacement world identifiers must be unique")
    for world, identifier in zip(worlds, identifiers, strict=True):
        if "world_id" in world:
            world["world_id"] = identifier
        elif "id" in world:
            world["id"] = identifier
        else:
            world["world_id"] = identifier
    return result


def permute_support(posterior: Mapping[str, Any], order: Sequence[int]) -> dict[str, Any]:
    result = _posterior_copy(posterior)
    if sorted(order) != list(range(len(result["worlds"]))):
        raise ValueError("support order must be a permutation of all world indexes")
    result["worlds"] = [result["worlds"][index] for index in order]
    return result


def split_world(
    posterior: Mapping[str, Any],
    index: int,
    weights: Sequence[float],
    *,
    identifiers: Sequence[str | int] | None = None,
) -> dict[str, Any]:
    result = _posterior_copy(posterior)
    world = result["worlds"][index]
    original_weight = world.get("weight")
    if not isinstance(original_weight, (int, float)) or isinstance(original_weight, bool):
        raise ValueError("world weight must be numeric")
    pieces = [float(value) for value in weights]
    if len(pieces) < 2 or any(not math.isfinite(value) or value <= 0 for value in pieces):
        raise ValueError("split weights must contain at least two positive finite values")
    if not math.isclose(math.fsum(pieces), float(original_weight), rel_tol=1e-12, abs_tol=1e-15):
        raise ValueError("split weights must preserve the original world mass")
    if identifiers is None:
        base = world.get("world_id", world.get("id", f"world-{index}"))
        identifiers = [f"{base}/split-{part}" for part in range(len(pieces))]
    if len(identifiers) != len(pieces) or len(set(identifiers)) != len(identifiers):
        raise ValueError("split identifiers must be unique and cover every split world")

    copies: list[dict[str, Any]] = []
    for identifier, weight in zip(identifiers, pieces, strict=True):
        clone = copy.deepcopy(dict(world))
        clone["weight"] = weight
        if "world_id" in clone or "id" not in clone:
            clone["world_id"] = identifier
            clone.pop("id", None)
        else:
            clone["id"] = identifier
        copies.append(clone)
    result["worlds"] = result["worlds"][:index] + copies + result["worlds"][index + 1 :]
    return result


def _semantic_world_key(world: Mapping[str, Any]) -> str:
    content = {
        key: value for key, value in world.items() if key not in TRANSPORT_FIELDS | {"weight"}
    }
    return json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def merge_equivalent_worlds(posterior: Mapping[str, Any]) -> dict[str, Any]:
    result = _posterior_copy(posterior)
    groups: OrderedDict[str, list[Mapping[str, Any]]] = OrderedDict()
    for world in result["worlds"]:
        groups.setdefault(_semantic_world_key(world), []).append(world)

    merged: list[dict[str, Any]] = []
    for worlds in groups.values():
        representative = copy.deepcopy(dict(worlds[0]))
        weights = [world.get("weight") for world in worlds]
        if any(
            not isinstance(weight, (int, float)) or isinstance(weight, bool) for weight in weights
        ):
            raise ValueError("world weights must be numeric")
        representative["weight"] = math.fsum(float(weight) for weight in weights)
        merged.append(representative)
    result["worlds"] = merged
    return result


def permute_actions(actions: Sequence[str], order: Sequence[int]) -> list[str]:
    if sorted(order) != list(range(len(actions))):
        raise ValueError("action order must be a permutation of all action indexes")
    return [actions[index] for index in order]


def permute_oracle_support(oracle: Mapping[str, Any], order: Sequence[int]) -> dict[str, Any]:
    """Reorder hidden worlds and their direct transitions without changing content."""
    result = copy.deepcopy(dict(oracle))
    worlds = result.get("worlds")
    transitions = result.get("transitions")
    actions = result.get("legal_actions")
    if not isinstance(worlds, list) or not isinstance(transitions, list):
        raise ValueError("transition oracle needs worlds and transitions")
    if not isinstance(actions, list):
        raise ValueError("transition oracle needs legal actions")
    if sorted(order) != list(range(len(worlds))):
        raise ValueError("oracle support order must be a permutation of all world indexes")
    ordered_worlds = [worlds[index] for index in order]
    transition_by_key = {(str(row["world_id"]), str(row["action"])): row for row in transitions}
    result["worlds"] = ordered_worlds
    result["transitions"] = [
        transition_by_key[(str(world["world_id"]), str(action))]
        for world in ordered_worlds
        for action in actions
    ]
    return result


def rename_game_world_ids(game: Mapping[str, Any], identifiers: Sequence[str]) -> dict[str, Any]:
    """Rename synthetic game worlds and every action table keyed by them."""
    result = copy.deepcopy(dict(game))
    worlds = result.get("worlds")
    actions = result.get("actions")
    if not isinstance(worlds, list) or not isinstance(actions, dict):
        raise ValueError("synthetic game needs worlds and actions")
    if len(identifiers) != len(worlds) or len(set(identifiers)) != len(identifiers):
        raise ValueError("replacement names must uniquely cover every game world")
    renaming = {
        str(world["name"]): identifier
        for world, identifier in zip(worlds, identifiers, strict=True)
    }
    for world, identifier in zip(worlds, identifiers, strict=True):
        world["name"] = identifier
    for action in actions.values():
        for field in ("terminal_payoffs", "observations"):
            values = action.get(field)
            if isinstance(values, dict):
                action[field] = {renaming[str(key)]: value for key, value in values.items()}
        continuations = action.get("continuations")
        if isinstance(continuations, dict):
            action["continuations"] = {
                name: {renaming[str(key)]: value for key, value in values.items()}
                for name, values in continuations.items()
            }
    return result


def permute_game_support(game: Mapping[str, Any], order: Sequence[int]) -> dict[str, Any]:
    result = copy.deepcopy(dict(game))
    worlds = result.get("worlds")
    if not isinstance(worlds, list) or sorted(order) != list(range(len(worlds))):
        raise ValueError("game support order must be a permutation of all world indexes")
    result["worlds"] = [worlds[index] for index in order]
    return result


def split_game_world(
    game: Mapping[str, Any],
    index: int,
    weights: Sequence[float],
    *,
    identifiers: Sequence[str],
) -> dict[str, Any]:
    result = copy.deepcopy(dict(game))
    worlds = result.get("worlds")
    actions = result.get("actions")
    if not isinstance(worlds, list) or not isinstance(actions, dict):
        raise ValueError("synthetic game needs worlds and actions")
    original = worlds[index]
    total = math.fsum(float(weight) for weight in weights)
    if len(weights) < 2 or not math.isclose(total, float(original["weight"]), abs_tol=1e-15):
        raise ValueError("split weights must preserve the original mass")
    if len(identifiers) != len(weights) or len(set(identifiers)) != len(identifiers):
        raise ValueError("split identifiers must uniquely cover all pieces")
    old_name = str(original["name"])
    pieces = [
        {**copy.deepcopy(original), "name": name, "weight": float(weight)}
        for name, weight in zip(identifiers, weights, strict=True)
    ]
    result["worlds"] = worlds[:index] + pieces + worlds[index + 1 :]
    for action in actions.values():
        for field in ("terminal_payoffs", "observations"):
            values = action.get(field)
            if isinstance(values, dict) and old_name in values:
                value = values.pop(old_name)
                for name in identifiers:
                    values[name] = copy.deepcopy(value)
        continuations = action.get("continuations")
        if isinstance(continuations, dict):
            for values in continuations.values():
                if old_name in values:
                    value = values.pop(old_name)
                    for name in identifiers:
                        values[name] = copy.deepcopy(value)
    return result


def merge_equivalent_game_worlds(game: Mapping[str, Any]) -> dict[str, Any]:
    """Merge worlds with identical per-action observations and payoffs."""
    result = copy.deepcopy(dict(game))
    worlds = result.get("worlds")
    actions = result.get("actions")
    if not isinstance(worlds, list) or not isinstance(actions, dict):
        raise ValueError("synthetic game needs worlds and actions")
    signatures: dict[str, list[dict[str, Any]]] = {}
    for world in worlds:
        name = str(world["name"])
        outcomes = {}
        for action_name, action in actions.items():
            outcomes[action_name] = {}
            for field in ("terminal_payoffs", "observations"):
                values = action.get(field)
                if isinstance(values, dict) and name in values:
                    outcomes[action_name][field] = values[name]
            continuations = action.get("continuations")
            if isinstance(continuations, dict):
                outcomes[action_name]["continuations"] = {
                    label: values[name] for label, values in continuations.items()
                }
        signature = json.dumps(outcomes, sort_keys=True, separators=(",", ":"))
        signatures.setdefault(signature, []).append(world)

    merged_worlds = []
    for members in signatures.values():
        representative = members[0]
        member_names = {str(world["name"]) for world in members}
        merged_worlds.append(
            {
                **representative,
                "weight": math.fsum(float(world["weight"]) for world in members),
            }
        )
        for action in actions.values():
            for field in ("terminal_payoffs", "observations"):
                values = action.get(field)
                if isinstance(values, dict):
                    for name in member_names - {str(representative["name"])}:
                        values.pop(name, None)
            continuations = action.get("continuations")
            if isinstance(continuations, dict):
                for values in continuations.values():
                    for name in member_names - {str(representative["name"])}:
                        values.pop(name, None)
    result["worlds"] = merged_worlds
    return result


def perturb_hidden_field(
    posterior: Mapping[str, Any],
    world_index: int,
    path: Sequence[str],
    value: Any,
) -> dict[str, Any]:
    if not path:
        raise ValueError("hidden field path must be non-empty")
    result = _posterior_copy(posterior)
    target: dict[str, Any] = result["worlds"][world_index]
    for component in path[:-1]:
        child = target.setdefault(component, {})
        if not isinstance(child, dict):
            raise ValueError("hidden field path crosses a non-object value")
        target = child
    target[path[-1]] = copy.deepcopy(value)
    return result


def change_budget(document: Mapping[str, Any], value: int) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    budget = result.get("compute_budget")
    if not isinstance(budget, dict):
        raise ValueError("artifact has no compute_budget object")
    if "per_method_limit" in budget:
        budget["per_method_limit"] = value
    elif "authorized" in budget:
        budget["authorized"] = value
    else:
        raise ValueError("compute_budget has no recognized limit")
    return result


def change_revision(
    document: Mapping[str, Any], value: str, *, field: str = "showdown_commit"
) -> dict[str, Any]:
    result = copy.deepcopy(dict(document))
    if field not in result:
        raise ValueError(f"artifact has no {field!r} revision field")
    result[field] = value
    return result


def inject_future_information(
    document: Mapping[str, Any], path: Sequence[str], value: Any
) -> dict[str, Any]:
    if not path:
        raise ValueError("future information path must be non-empty")
    result = copy.deepcopy(dict(document))
    target: dict[str, Any] = result
    for component in path[:-1]:
        child = target.setdefault(component, {})
        if not isinstance(child, dict):
            raise ValueError("future information path crosses a non-object value")
        target = child
    target[path[-1]] = copy.deepcopy(value)
    return result
