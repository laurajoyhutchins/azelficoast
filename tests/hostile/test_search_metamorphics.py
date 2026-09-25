from __future__ import annotations

import copy
import json
import math
from typing import Any, Mapping, Sequence

from azelficoast.core.search import search_transition_program
from azelficoast.corpus import build_fixtures
from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION
from hostile.fixtures import hostile_case
from hostile.transforms import (
    merge_equivalent_game_worlds,
    permute_actions,
    permute_game_support,
    rename_game_world_ids,
    split_game_world,
)


class WeightedPayoffEvaluator:
    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        del legal_actions
        action = public_state["root_action"]
        return math.fsum(
            float(world["weight"]) * float(world["hidden"]["values"][action])
            for world in posterior_worlds
        )


class SupportSizeEvaluator:
    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        del public_state, legal_actions
        return 1.0 if len(posterior_worlds) == 1 else 0.0


def _weighted_game() -> dict[str, Any]:
    names = ("w0", "w1")
    worlds = [
        {"name": names[0], "weight": 0.25, "hidden": {"values": {"A": 1.0, "B": 0.0}}},
        {"name": names[1], "weight": 0.75, "hidden": {"values": {"A": 0.0, "B": 1.0}}},
    ]
    return {
        "worlds": worlds,
        "actions": {
            "A": {"terminal_payoffs": {"w0": 1.0, "w1": 0.0}},
            "B": {"terminal_payoffs": {"w0": 0.0, "w1": 1.0}},
        },
    }


def _search_inputs(game: Mapping[str, Any], actions: Sequence[str] = ("A", "B")):
    worlds = game["worlds"]
    world_ids = [world["name"] for world in worlds]
    posterior = {
        "worlds": [
            {
                "world_id": world["name"],
                "weight": world["weight"],
                "hidden": copy.deepcopy(world["hidden"]),
            }
            for world in worlds
        ]
    }
    programs = []
    for action in actions:
        programs.append(
            {
                "action": action,
                "classes_out": 1,
                "classes": [
                    {
                        "member_world_ids": list(world_ids),
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"signal": "same"},
                                "successor": {"root_action": action},
                                "legal_actions": ["continue"],
                            }
                        ],
                    }
                ],
            }
        )
    program_set = {
        "schema": "example.transition-program-set",
        "schema_version": 1,
        "world_ids": list(world_ids),
        "legal_actions": list(actions),
        "programs": programs,
    }
    return program_set, posterior


def _solve(program_set, posterior, method="information_set"):
    return search_transition_program(
        program_set=program_set,
        posterior=posterior,
        method=method,
        evaluator=WeightedPayoffEvaluator(),
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )


@hostile_case(
    mutation="rename and reorder hidden worlds, permute actions, then split and merge an atom",
    expected="both methods return hand-computed values A=.25 and B=.75 with action association intact",
    threat="search averages support rows or couples values to incidental list positions",
    layer="generic determinization and information-set search",
)
def test_weighted_search_is_invariant_to_support_and_action_representations() -> None:
    game = _weighted_game()
    split = split_game_world(game, 0, (0.1, 0.15), identifiers=("w0-a", "w0-b"))
    variants = [
        permute_game_support(game, (1, 0)),
        rename_game_world_ids(game, ("opaque-92841", "opaque-negative-seven")),
        split,
        merge_equivalent_game_worlds(split),
    ]
    expected = {"A": 0.25, "B": 0.75}
    for variant in variants:
        actions = permute_actions(("A", "B"), (1, 0))
        program_set, posterior = _search_inputs(variant, actions)
        for method in ("determinization", "information_set"):
            result = _solve(program_set, posterior, method)
            assert result["root_values"] == expected
            assert result["chosen_action"] == "B"


def _information_game(reveal: bool):
    posterior = {
        "worlds": [
            {"world_id": "good", "weight": 0.5, "hidden": {"kind": "good"}},
            {"world_id": "bad", "weight": 0.5, "hidden": {"kind": "bad"}},
        ]
    }
    programs = [
        {
            "action": "hide",
            "classes_out": 1,
            "classes": [
                {
                    "member_world_ids": ["good", "bad"],
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"signal": "seen"} if reveal else {"signal": "none"},
                            "successor": {"stage": "done"},
                            "legal_actions": ["continue"],
                        }
                    ],
                }
            ],
        },
        {
            "action": "probe",
            "classes_out": 2,
            "classes": [
                {
                    "member_world_ids": [world_id],
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"signal": world_id},
                            "successor": {"stage": "done"},
                            "legal_actions": ["continue"],
                        }
                    ],
                }
                for world_id in ("good", "bad")
            ],
        },
    ]
    return {
        "schema": "example.transition-program-set",
        "schema_version": 1,
        "world_ids": ["good", "bad"],
        "legal_actions": ["hide", "probe"],
        "programs": programs,
    }, posterior


@hostile_case(
    mutation="compare a continuation before versus after the public observation separates worlds",
    expected="unresolved worlds share one continuation; revealed worlds may condition on the observation",
    threat="strategy fusion leaks unavailable information or suppresses legitimate public information",
    layer="information-set search coupling",
)
def test_information_set_coupling_canary_changes_only_after_reveal() -> None:
    evaluator = SupportSizeEvaluator()
    unresolved_program, posterior = _information_game(reveal=False)
    revealed_program, _ = _information_game(reveal=True)

    unresolved = search_transition_program(
        program_set=unresolved_program,
        posterior=posterior,
        method="information_set",
        evaluator=evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    revealed = search_transition_program(
        program_set=revealed_program,
        posterior=posterior,
        method="information_set",
        evaluator=evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )

    assert unresolved["root_values"] == {"hide": 0.0, "probe": 1.0}
    assert revealed["root_values"] == {"hide": 0.0, "probe": 1.0}
    # Determinization can condition on the world even before observation.
    determinized = search_transition_program(
        program_set=unresolved_program,
        posterior=posterior,
        method="determinization",
        evaluator=evaluator,
        expected_program_schema="example.transition-program-set",
        expected_program_schema_version=1,
    )
    assert determinized["root_values"]["hide"] == 1.0


@hostile_case(
    mutation="repeat deterministic search and permute program/action support order",
    expected="values and selected semantic action remain exact across runs and encodings",
    threat="undeclared randomness or positional action coupling changes the experiment output",
    layer="generic search result identity",
)
def test_search_is_deterministic_and_action_order_invariant() -> None:
    program_set, posterior = _search_inputs(_weighted_game())
    baseline = _solve(program_set, posterior)
    assert _solve(program_set, posterior) == baseline

    reordered_actions = ("B", "A")
    reordered, reordered_posterior = _search_inputs(_weighted_game(), reordered_actions)
    reordered["programs"].reverse()
    actual = _solve(reordered, reordered_posterior)
    assert actual["root_values"] == baseline["root_values"]
    assert actual["chosen_action"] == baseline["chosen_action"]


@hostile_case(
    mutation="change only the protocol observation recorded after the historical decision boundary",
    expected="the reconstructed historical decision fixture and protocol prefix remain unchanged",
    threat="future battle evidence leaks backward into the decision information set",
    layer="trace-to-decision fixture construction",
)
def test_future_protocol_observation_cannot_change_a_past_decision_fixture(tmp_path) -> None:
    state = {"battle_tag": "battle-1", "turn": 4, "legal_actions": ["wait", "reveal"]}

    def record(event, kind, **extra):
        return {
            "schema": TRACE_SCHEMA,
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": "future-info-canary",
            "event_index": event,
            "observed_at": "2026-09-25T00:00:00+00:00",
            "kind": kind,
            **extra,
        }

    common = [
        record(
            0,
            "protocol",
            room="battle-1",
            protocol_index=0,
            messages=[["", "turn", "4"]],
        ),
        record(
            1,
            "decision",
            battle_tag="battle-1",
            decision_index=0,
            state=state,
            chosen_action="wait",
        ),
    ]
    variants = (
        record(
            2, "protocol", room="battle-1", protocol_index=1, messages=[["", "move", "Protect"]]
        ),
        record(
            2,
            "protocol",
            room="battle-1",
            protocol_index=1,
            messages=[["", "move", "Item revealed"]],
        ),
    )
    paths = [tmp_path / "future-a.jsonl", tmp_path / "future-b.jsonl"]
    for path, future in zip(paths, variants, strict=True):
        path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in [*common, future]),
            encoding="utf-8",
        )
    [before_a] = build_fixtures([paths[0]])
    [before_b] = build_fixtures([paths[1]])

    assert before_a.fixture_id == before_b.fixture_id
    assert before_a.state == before_b.state
    assert before_a.protocol_prefix == before_b.protocol_prefix
