"""Minimal hidden-world solvers for determinization versus public belief."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from azelficoast.live.corpus import CorpusError, DecisionFixture

GAME_KEY = "imperfect_information_game"


class ImperfectInformationError(CorpusError):
    """Raised when a frozen fixture does not define a valid hidden-world game."""


@dataclass(frozen=True)
class HiddenWorld:
    name: str
    weight: float


@dataclass(frozen=True)
class RootAction:
    name: str
    terminal_payoffs: Mapping[str, float] | None
    observations: Mapping[str, str] | None
    continuations: Mapping[str, Mapping[str, float]] | None

    @property
    def terminal(self) -> bool:
        return self.terminal_payoffs is not None


@dataclass(frozen=True)
class HiddenWorldGame:
    worlds: tuple[HiddenWorld, ...]
    actions: tuple[RootAction, ...]


@dataclass(frozen=True)
class ActionEvaluation:
    value: float
    continuation_by_observation: Mapping[str, str]


@dataclass(frozen=True)
class SearchSolution:
    action: str
    value: float
    action_values: Mapping[str, float]
    continuation_by_action: Mapping[str, Mapping[str, str]]


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImperfectInformationError(f"{label} must be an object")
    return value


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ImperfectInformationError(f"{label} must be numeric")
    return float(value)


def parse_game(fixture: DecisionFixture) -> HiddenWorldGame:
    raw_game = _require_mapping(fixture.state.get(GAME_KEY), GAME_KEY)
    raw_worlds = raw_game.get("worlds")
    raw_actions = _require_mapping(raw_game.get("actions"), f"{GAME_KEY}.actions")

    if not isinstance(raw_worlds, Sequence) or isinstance(raw_worlds, (str, bytes)):
        raise ImperfectInformationError(f"{GAME_KEY}.worlds must be an array")
    if not raw_worlds:
        raise ImperfectInformationError("hidden-world game must contain at least one world")

    worlds: list[HiddenWorld] = []
    world_names: set[str] = set()
    for index, raw_world in enumerate(raw_worlds):
        world_record = _require_mapping(raw_world, f"{GAME_KEY}.worlds[{index}]")
        name = world_record.get("name")
        if not isinstance(name, str) or not name:
            raise ImperfectInformationError(f"world {index} must have a non-empty name")
        if name in world_names:
            raise ImperfectInformationError(f"duplicate hidden world {name!r}")
        weight = _require_number(world_record.get("weight"), f"world {name!r} weight")
        if weight <= 0:
            raise ImperfectInformationError(f"world {name!r} weight must be positive")
        world_names.add(name)
        worlds.append(HiddenWorld(name=name, weight=weight))

    total_weight = sum(world.weight for world in worlds)
    normalized_worlds = tuple(
        HiddenWorld(name=world.name, weight=world.weight / total_weight)
        for world in worlds
    )

    legal_actions = set(fixture.legal_actions)
    actions: list[RootAction] = []
    for action_name in sorted(raw_actions):
        if action_name not in legal_actions:
            raise ImperfectInformationError(
                f"game action {action_name!r} is not legal in fixture {fixture.fixture_id}"
            )
        raw_action = _require_mapping(
            raw_actions[action_name],
            f"{GAME_KEY}.actions[{action_name!r}]",
        )
        has_terminal = "terminal_payoffs" in raw_action
        has_continuations = "continuations" in raw_action or "observations" in raw_action
        if has_terminal == has_continuations:
            raise ImperfectInformationError(
                f"action {action_name!r} must be exactly terminal or continuing"
            )

        if has_terminal:
            raw_payoffs = _require_mapping(
                raw_action.get("terminal_payoffs"),
                f"action {action_name!r} terminal_payoffs",
            )
            payoffs = {
                world.name: _require_number(
                    raw_payoffs.get(world.name),
                    f"action {action_name!r} payoff in {world.name!r}",
                )
                for world in normalized_worlds
            }
            actions.append(
                RootAction(
                    name=action_name,
                    terminal_payoffs=payoffs,
                    observations=None,
                    continuations=None,
                )
            )
            continue

        raw_observations = _require_mapping(
            raw_action.get("observations"),
            f"action {action_name!r} observations",
        )
        observations: dict[str, str] = {}
        for world in normalized_worlds:
            observation = raw_observations.get(world.name)
            if not isinstance(observation, str) or not observation:
                raise ImperfectInformationError(
                    f"action {action_name!r} needs an observation for {world.name!r}"
                )
            observations[world.name] = observation

        raw_continuations = _require_mapping(
            raw_action.get("continuations"),
            f"action {action_name!r} continuations",
        )
        if not raw_continuations:
            raise ImperfectInformationError(
                f"action {action_name!r} needs at least one continuation"
            )
        continuations: dict[str, dict[str, float]] = {}
        for continuation_name in sorted(raw_continuations):
            raw_payoffs = _require_mapping(
                raw_continuations[continuation_name],
                f"continuation {continuation_name!r}",
            )
            continuations[continuation_name] = {
                world.name: _require_number(
                    raw_payoffs.get(world.name),
                    f"continuation {continuation_name!r} payoff in {world.name!r}",
                )
                for world in normalized_worlds
            }

        actions.append(
            RootAction(
                name=action_name,
                terminal_payoffs=None,
                observations=observations,
                continuations=continuations,
            )
        )

    if not actions:
        raise ImperfectInformationError("hidden-world game must contain at least one action")

    return HiddenWorldGame(worlds=normalized_worlds, actions=tuple(actions))


def _terminal_value(game: HiddenWorldGame, action: RootAction) -> float:
    assert action.terminal_payoffs is not None
    return sum(
        world.weight * action.terminal_payoffs[world.name]
        for world in game.worlds
    )


def _pick_best(values: Mapping[str, float]) -> tuple[str, float]:
    if not values:
        raise ImperfectInformationError("cannot choose from an empty value map")
    best_value = max(values.values())
    best_name = min(name for name, value in values.items() if value == best_value)
    return best_name, best_value


class DeterminizationPolicy:
    """Perfect-information world sampling with strategy fusion at continuation nodes."""

    name = "determinization"

    def solve(self, fixture: DecisionFixture) -> SearchSolution:
        game = parse_game(fixture)
        action_values: dict[str, float] = {}
        continuation_by_action: dict[str, Mapping[str, str]] = {}

        for action in game.actions:
            if action.terminal:
                action_values[action.name] = _terminal_value(game, action)
                continuation_by_action[action.name] = {}
                continue

            assert action.continuations is not None
            world_choices: dict[str, str] = {}
            value = 0.0
            for world in game.worlds:
                continuation_values = {
                    continuation_name: payoffs[world.name]
                    for continuation_name, payoffs in action.continuations.items()
                }
                chosen, world_value = _pick_best(continuation_values)
                world_choices[world.name] = chosen
                value += world.weight * world_value

            action_values[action.name] = value
            continuation_by_action[action.name] = world_choices

        chosen_action, value = _pick_best(action_values)
        return SearchSolution(
            action=chosen_action,
            value=value,
            action_values=action_values,
            continuation_by_action=continuation_by_action,
        )

    def choose(self, fixture: DecisionFixture) -> str:
        return self.solve(fixture).action


class PublicBeliefPolicy:
    """Exact depth-two solver that preserves public information sets."""

    name = "public-belief"

    def solve(self, fixture: DecisionFixture) -> SearchSolution:
        game = parse_game(fixture)
        action_values: dict[str, float] = {}
        continuation_by_action: dict[str, Mapping[str, str]] = {}

        for action in game.actions:
            if action.terminal:
                action_values[action.name] = _terminal_value(game, action)
                continuation_by_action[action.name] = {}
                continue

            assert action.observations is not None
            assert action.continuations is not None

            worlds_by_observation: dict[str, list[HiddenWorld]] = {}
            for world in game.worlds:
                observation = action.observations[world.name]
                worlds_by_observation.setdefault(observation, []).append(world)

            observation_choices: dict[str, str] = {}
            value = 0.0
            for observation in sorted(worlds_by_observation):
                worlds = worlds_by_observation[observation]
                continuation_values = {
                    continuation_name: sum(
                        world.weight * payoffs[world.name] for world in worlds
                    )
                    for continuation_name, payoffs in action.continuations.items()
                }
                chosen, observation_value = _pick_best(continuation_values)
                observation_choices[observation] = chosen
                value += observation_value

            action_values[action.name] = value
            continuation_by_action[action.name] = observation_choices

        chosen_action, value = _pick_best(action_values)
        return SearchSolution(
            action=chosen_action,
            value=value,
            action_values=action_values,
            continuation_by_action=continuation_by_action,
        )

    def choose(self, fixture: DecisionFixture) -> str:
        return self.solve(fixture).action


determinization = DeterminizationPolicy()
public_belief = PublicBeliefPolicy()


def _reference_fixture(*, reveal_without_cost: bool) -> DecisionFixture:
    actions = {
        "/choose move safe": {
            "terminal_payoffs": {
                "red": 0.25,
                "blue": 0.25,
            }
        },
        "/choose move guess": {
            "observations": {
                "red": "red" if reveal_without_cost else "unrevealed",
                "blue": "blue" if reveal_without_cost else "unrevealed",
            },
            "continuations": {
                "guess-red": {"red": 1.0, "blue": -1.0},
                "guess-blue": {"red": -1.0, "blue": 1.0},
            },
        },
        "/choose move scout": {
            "observations": {
                "red": "red",
                "blue": "blue",
            },
            "continuations": {
                "guess-red": {"red": 0.8, "blue": -1.2},
                "guess-blue": {"red": -1.2, "blue": 0.8},
            },
        },
    }
    state = {
        "legal_actions": sorted(actions),
        GAME_KEY: {
            "worlds": [
                {"name": "red", "weight": 0.5},
                {"name": "blue", "weight": 0.5},
            ],
            "actions": actions,
        },
    }
    return DecisionFixture(
        fixture_id="reference-revealed" if reveal_without_cost else "reference-hidden",
        state=state,
        protocol_prefix=(),
        control_decisions=(),
    )


def run_reference_experiment() -> dict[str, Any]:
    hidden = _reference_fixture(reveal_without_cost=False)
    revealed = _reference_fixture(reveal_without_cost=True)

    hidden_determinization = determinization.solve(hidden)
    hidden_belief = public_belief.solve(hidden)
    revealed_determinization = determinization.solve(revealed)
    revealed_belief = public_belief.solve(revealed)

    treatment_passed = (
        hidden_determinization.action == "/choose move guess"
        and hidden_belief.action == "/choose move scout"
        and hidden_determinization.action_values["/choose move guess"] == 1.0
        and hidden_belief.action_values["/choose move guess"] == 0.0
    )
    negative_control_passed = (
        revealed_determinization.action == "/choose move guess"
        and revealed_belief.action == "/choose move guess"
        and revealed_determinization.action_values == revealed_belief.action_values
    )

    return {
        "schema": "azelficoast.imperfect-information-experiment",
        "schema_version": 1,
        "hypothesis": (
            "perfect-information determinization overvalues continuations when "
            "indistinguishable hidden worlds require one shared future action"
        ),
        "treatment": {
            "determinization": {
                "action": hidden_determinization.action,
                "value": hidden_determinization.value,
                "action_values": dict(hidden_determinization.action_values),
                "continuations": dict(hidden_determinization.continuation_by_action),
            },
            "public_belief": {
                "action": hidden_belief.action,
                "value": hidden_belief.value,
                "action_values": dict(hidden_belief.action_values),
                "continuations": dict(hidden_belief.continuation_by_action),
            },
            "passed": treatment_passed,
        },
        "negative_control": {
            "determinization": {
                "action": revealed_determinization.action,
                "value": revealed_determinization.value,
                "action_values": dict(revealed_determinization.action_values),
            },
            "public_belief": {
                "action": revealed_belief.action,
                "value": revealed_belief.value,
                "action_values": dict(revealed_belief.action_values),
            },
            "passed": negative_control_passed,
        },
        "passed": treatment_passed and negative_control_passed,
    }


def main() -> int:
    result = run_reference_experiment()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
