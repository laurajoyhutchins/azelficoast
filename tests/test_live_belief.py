from __future__ import annotations

import json
from types import SimpleNamespace

from azelficoast.corpus import DecisionFixture
from azelficoast.live_belief import (
    LiveDecisionResult,
    build_probe_source,
    live_fixture,
    public_belief_result,
)
from azelficoast.player import AzelficoastPlayer


def _state(
    *,
    opponent_item=None,
    tera_type="Steel",
    opponent_species="Zapdos-Galar",
) -> dict[str, object]:
    return {
        "battle_tag": "battle-live",
        "player": "Azelficoast",
        "opponent": "Rival",
        "active": {
            "species": "Tinkaton",
            "tera_type": tera_type,
        },
        "opponent_active": {
            "species": opponent_species,
            "level": 77,
            "item": opponent_item,
        },
        "team": {"p1: Tinkaton": {"species": "Tinkaton"}},
        "legal_actions": [
            "/choose move protect",
            "/choose move gigatonhammer",
        ],
    }


def _protocol(*, own_side: str = "p1") -> tuple[tuple[tuple[str, ...], ...], ...]:
    opponent_side = "p2" if own_side == "p1" else "p1"
    return (
        (
            ("", "player", own_side, "Azelficoast"),
            ("", "player", opponent_side, "Rival"),
            ("", "move", f"{opponent_side}a: Zapdos-Galar", "U-turn"),
        ),
    )


def test_live_fixture_uses_information_state_identity_and_drops_battle_tag() -> None:
    first = live_fixture(_state(), _protocol())
    second = live_fixture(_state(), _protocol())

    assert first.fixture_id == second.fixture_id
    assert "battle_tag" not in first.state
    assert first.protocol_prefix == _protocol()


def test_probe_source_admits_hidden_choice_from_either_showdown_side() -> None:
    p1_fixture = live_fixture(_state(), _protocol(own_side="p1"))
    p2_fixture = live_fixture(_state(), _protocol(own_side="p2"))

    p1_source, p1_reason = build_probe_source(p1_fixture)
    p2_source, p2_reason = build_probe_source(p2_fixture)

    assert p1_reason == p2_reason == "admitted"
    assert p1_source is not None
    assert p2_source is not None
    assert p1_source["plausible_items"] == ["Choice Band", "Choice Scarf"]
    assert p2_source["opponent_response_move"] == "U-turn"


def test_probe_source_rejects_known_opponent_item() -> None:
    fixture = live_fixture(_state(opponent_item="leftovers"), _protocol())
    source, reason = build_probe_source(fixture)

    assert source is None
    assert reason == "opponent-item-known"


def test_probe_source_does_not_reuse_previous_active_move_after_switch() -> None:
    protocol = (
        (
            ("", "player", "p1", "Azelficoast"),
            ("", "player", "p2", "Rival"),
            ("", "switch", "p2a: Zapdos-Galar", "Zapdos-Galar, L77", "100/100"),
            ("", "move", "p2a: Zapdos-Galar", "U-turn", "p1a: Tinkaton"),
            ("", "switch", "p2a: Gouging Fire", "Gouging Fire, L77", "100/100"),
        ),
    )
    fixture = live_fixture(_state(opponent_species="Gouging Fire"), protocol)

    source, reason = build_probe_source(fixture)

    assert source is None
    assert reason == "opponent-side-or-last-move-unresolved"


def test_probe_source_can_reuse_current_species_move_after_switching_back() -> None:
    protocol = (
        (
            ("", "player", "p1", "Azelficoast"),
            ("", "player", "p2", "Rival"),
            ("", "switch", "p2a: Zapdos-Galar", "Zapdos-Galar, L77", "100/100"),
            ("", "move", "p2a: Zapdos-Galar", "U-turn", "p1a: Tinkaton"),
            ("", "switch", "p2a: Gouging Fire", "Gouging Fire, L77", "100/100"),
            ("", "switch", "p2a: Zapdos-Galar", "Zapdos-Galar, L77", "100/100"),
        ),
    )
    fixture = live_fixture(_state(), protocol)

    source, reason = build_probe_source(fixture)

    assert reason == "admitted"
    assert source is not None
    assert source["opponent_response_move"] == "U-turn"


def test_probe_source_recovers_current_species_tera_from_prior_public_request() -> None:
    request = json.dumps(
        {
            "active": [{"canTerastallize": "Steel"}],
            "side": {
                "pokemon": [
                    {
                        "details": "Tinkaton, L82",
                        "active": True,
                    }
                ]
            },
        },
        sort_keys=True,
    )
    protocol = (
        (
            ("", "player", "p1", "Azelficoast"),
            ("", "player", "p2", "Rival"),
            ("", "request", request),
            ("", "move", "p2a: Zapdos-Galar", "U-turn"),
        ),
    )
    fixture = live_fixture(_state(tera_type=None), protocol)

    source, reason = build_probe_source(fixture)

    assert reason == "admitted"
    assert source is not None
    assert source["own_active_tera_type"] == "Steel"


def test_probe_source_does_not_reuse_another_species_tera_type() -> None:
    request = json.dumps(
        {
            "active": [{"canTerastallize": "Water"}],
            "side": {
                "pokemon": [
                    {
                        "details": "Lapras, L82",
                        "active": True,
                    }
                ]
            },
        },
        sort_keys=True,
    )
    protocol = (
        (
            ("", "player", "p1", "Azelficoast"),
            ("", "player", "p2", "Rival"),
            ("", "request", request),
            ("", "move", "p2a: Zapdos-Galar", "U-turn"),
        ),
    )
    fixture = live_fixture(_state(tera_type=None), protocol)

    source, reason = build_probe_source(fixture)

    assert source is None
    assert reason == "own-active-tera-type-unavailable"


def _strategy_fusion_oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": f"{item}-{noise}",
            "weight": 0.25,
            "hidden": {
                "opponent.active.item": item,
                "noise": noise,
            },
        }
        for item in ("Choice Band", "Choice Scarf")
        for noise in (1, 2)
    ]
    transitions = []
    for world in worlds:
        item = world["hidden"]["opponent.active.item"]
        risky = {"hit": 2.0, "wait": -2.0}
        if item == "Choice Scarf":
            risky = {"hit": -2.0, "wait": 2.0}
        transitions.extend(
            [
                {
                    "world_id": world["world_id"],
                    "action": "risky",
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"same": True},
                            "successor": {"same": True},
                            "continuations": risky,
                        }
                    ],
                },
                {
                    "world_id": world["world_id"],
                    "action": "safe",
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"safe": True},
                            "successor": {"safe": True},
                            "terminal_utility": 1.0,
                        }
                    ],
                },
            ]
        )
    return {
        "schema": "azelficoast.real-belief-transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "live",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["risky", "safe"],
        "dependency_candidates": ["opponent.active.item", "noise"],
        "declared_reads": {
            "risky": ["opponent.active.item"],
            "safe": ["opponent.active.item"],
        },
        "transitions": transitions,
    }


def test_public_belief_result_selects_shared_information_set_action() -> None:
    result = public_belief_result(_strategy_fusion_oracle(), ["risky", "safe"])

    assert result.status == "selected"
    assert result.action == "safe"
    assert result.diagnostics["determinization_action"] == "risky"
    assert result.diagnostics["policy_disagreement"] is True
    assert result.diagnostics["source_world_count"] == 4
    assert result.diagnostics["decision_class_count"] == 2
    assert result.diagnostics["decision_relevant_hidden_fields"] == [
        "opponent.active.item"
    ]
    assert result.diagnostics["decision_world_reduction"] == 2
    assert result.diagnostics["decision_reduction_fraction"] == 0.5
    assert result.diagnostics["belief_branching_required"] is True


def _pokemon(species: str, *, opponent: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        level=80,
        transformed=False,
        active=True,
        fainted=False,
        current_hp=100,
        max_hp=100,
        current_hp_fraction=1.0,
        base_stats={"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        stats={"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100},
        status=None,
        item="unknown_item" if opponent else "leftovers",
        ability=None if opponent else "moldbreaker",
        moves={"uturn": object()} if opponent else {"protect": object()},
        boosts={"atk": 0},
        types=[SimpleNamespace(name="STEEL")],
        tera_type=None,
    )


def _battle() -> SimpleNamespace:
    active = _pokemon("Tinkaton")
    opponent = _pokemon("Zapdos-Galar", opponent=True)
    order = SimpleNamespace(message="/choose move protect")
    return SimpleNamespace(
        battle_tag="battle-live",
        turn=8,
        player_username="Azelficoast",
        opponent_username="Rival",
        active_pokemon=active,
        opponent_active_pokemon=opponent,
        team={"p1a: Tinkaton": active},
        opponent_team={"p2a: Zapdos-Galar": opponent},
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        available_moves=[SimpleNamespace(id="protect")],
        available_switches=[],
        valid_orders=[order],
        force_switch=False,
        trapped=False,
        can_tera=True,
        _last_request={"active": [{"canTerastallize": "Steel"}]},
    )


def test_player_uses_live_belief_action_when_it_maps_to_valid_order() -> None:
    class Policy:
        def choose(self, fixture: DecisionFixture) -> LiveDecisionResult:
            assert fixture.legal_actions == ("/choose move protect",)
            return LiveDecisionResult(
                action="/choose move protect",
                status="selected",
                reason="test",
            )

    player = object.__new__(AzelficoastPlayer)
    player._belief_policy = Policy()
    player._decision_trace = None
    player._protocol_history = {
        "battle-live": [
            [
                ["", "player", "p1", "Azelficoast"],
                ["", "player", "p2", "Rival"],
                ["", "move", "p2a: Zapdos-Galar", "U-turn"],
            ]
        ]
    }

    order = player.choose_move(_battle())

    assert order.message == "/choose move protect"
