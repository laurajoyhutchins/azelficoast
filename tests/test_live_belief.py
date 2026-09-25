from __future__ import annotations

import json
from types import SimpleNamespace

from azelficoast.belief.evaluator import BeliefEvaluatorSpec, BeliefPrediction
from azelficoast.live.corpus import DecisionFixture
from azelficoast.live.belief import (
    LiveDecisionResult,
    PinnedShowdownBeliefPolicy,
    build_probe_source,
    live_fixture,
    public_belief_result,
    selective_belief_result,
    transition_program_belief_result,
)
from azelficoast.live.player import AzelficoastPlayer
from azelficoast.search.selective import PolicyMarginSearchGate
from azelficoast.core.whole_turn_program import compile_whole_turn_programs


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


def test_probe_source_reconstructs_general_posterior_from_either_showdown_side() -> None:
    p1_fixture = live_fixture(_state(), _protocol(own_side="p1"))
    p2_fixture = live_fixture(_state(), _protocol(own_side="p2"))

    p1_source, p1_reason = build_probe_source(p1_fixture)
    p2_source, p2_reason = build_probe_source(p2_fixture)

    assert p1_reason == p2_reason == "admitted"
    assert p1_source is not None
    assert p2_source is not None
    assert "plausible_items" not in p1_source
    assert p2_source["opponent_response_move"] == "U-turn"
    assert p2_source["opponent_policy"] == {
        "kind": "strategy-mixture",
        "weighting": "equal-active-strategies",
        "strategies": [
            {"kind": "simple-heuristics"},
            {"kind": "dirty-tricks"},
            {"kind": "max-damage"},
            {"kind": "uniform-legal-moves"},
            {"kind": "repeat-observed-move", "move": "U-turn"},
        ],
        "voluntary_switches": True,
    }


def test_probe_source_uses_known_opponent_item_as_public_evidence() -> None:
    fixture = live_fixture(_state(opponent_item="leftovers"), _protocol())
    source, reason = build_probe_source(fixture)

    assert reason == "admitted"
    assert source is not None
    assert source["known_opponent_item"] == "leftovers"


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

    assert reason == "admitted"
    assert source is not None
    assert "opponent_response_move" not in source
    assert source["opponent_policy"] == {
        "kind": "strategy-mixture",
        "weighting": "equal-active-strategies",
        "strategies": [
            {"kind": "simple-heuristics"},
            {"kind": "dirty-tricks"},
            {"kind": "max-damage"},
            {"kind": "uniform-legal-moves"},
        ],
        "voluntary_switches": True,
    }


def test_probe_source_allows_status_move_as_bounded_response() -> None:
    protocol = (
        (
            ("", "player", "p1", "Azelficoast"),
            ("", "player", "p2", "Rival"),
            ("", "move", "p2a: Zapdos-Galar", "Bulk Up"),
        ),
    )
    fixture = live_fixture(_state(), protocol)

    source, reason = build_probe_source(fixture)

    assert reason == "admitted"
    assert source is not None
    assert source["opponent_response_move"] == "Bulk Up"
    assert source["opponent_policy"] == {
        "kind": "strategy-mixture",
        "weighting": "equal-active-strategies",
        "strategies": [
            {"kind": "simple-heuristics"},
            {"kind": "dirty-tricks"},
            {"kind": "max-damage"},
            {"kind": "uniform-legal-moves"},
            {"kind": "repeat-observed-move", "move": "Bulk Up"},
        ],
        "voluntary_switches": True,
    }
    assert "plausible_items" not in source


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
        "schema": "azelficoast.core.transition-oracle",
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




class _FakeEvaluator:
    spec = BeliefEvaluatorSpec(
        public_width=8,
        world_width=8,
        action_width=8,
        hidden_width=8,
        world_hidden_width=8,
    )
    identity = {
        "checkpoint_digest": "sha256:" + "a" * 64,
        "observability": "public_belief_only",
    }

    def __init__(self, margin: float) -> None:
        self.margin = margin

    def predict(self, inputs) -> BeliefPrediction:
        assert inputs.legal_actions == ("risky", "safe")
        return BeliefPrediction(
            value=0.25,
            legal_actions=inputs.legal_actions,
            probabilities=((1.0 + self.margin) / 2.0, (1.0 - self.margin) / 2.0),
            selected_action="risky",
            policy_margin=self.margin,
            policy_entropy_bits=0.8,
        )


class _PosteriorSpreadEvaluator:
    spec = BeliefEvaluatorSpec(
        public_width=8,
        world_width=8,
        action_width=8,
        hidden_width=8,
        world_hidden_width=8,
    )
    identity = {
        "checkpoint_digest": "sha256:" + "c" * 64,
        "observability": "public_belief_only",
    }

    def predict(self, inputs) -> BeliefPrediction:
        value = 1.0 - max(inputs.world_weights)
        probability = 1.0 / len(inputs.legal_actions)
        return BeliefPrediction(
            value=value,
            legal_actions=inputs.legal_actions,
            probabilities=tuple(probability for _ in inputs.legal_actions),
            selected_action=min(inputs.legal_actions),
            policy_margin=0.0,
            policy_entropy_bits=0.0,
        )


def _program_search_oracle(
    *,
    fixture_id: str = "live",
    showdown_commit: str = "a5df8274e85b0889bf2a9b3422a08b39732374fc",
    legal_actions: tuple[str, str] = ("risky", "safe"),
) -> dict[str, object]:
    reveal_action, preserve_action = legal_actions
    worlds = [
        {
            "world_id": f"{item}-{noise}",
            "weight": 0.25,
            "hidden": {"opponent.active.item": item, "noise": noise},
        }
        for item in ("Band", "Scarf")
        for noise in (1, 2)
    ]
    transitions: list[dict[str, object]] = []
    for world in worlds:
        item = str(world["hidden"]["opponent.active.item"])
        transitions.extend(
            [
                {
                    "world_id": world["world_id"],
                    "action": reveal_action,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"revealed": item},
                            "successor": {"turn": 9, "revealed": item},
                            "continuations": {"continue": 0.0},
                        }
                    ],
                },
                {
                    "world_id": world["world_id"],
                    "action": preserve_action,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"same": True},
                            "successor": {"turn": 9, "same": True},
                            "continuations": {"continue": 0.0},
                        }
                    ],
                },
            ]
        )
    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": fixture_id,
        "showdown_commit": showdown_commit,
        "worlds": worlds,
        "legal_actions": list(legal_actions),
        "dependency_candidates": ["opponent.active.item", "noise"],
        "declared_reads": {
            reveal_action: ["opponent.active.item"],
            preserve_action: [],
        },
        "transitions": transitions,
    }


def _selective_fixture() -> DecisionFixture:
    return DecisionFixture(
        fixture_id="live",
        state={"turn": 8, "legal_actions": ["risky", "safe"]},
        protocol_prefix=(),
        control_decisions=(),
    )


def test_transition_program_belief_search_uses_successor_beliefs() -> None:
    fixture = _selective_fixture()
    oracle = _program_search_oracle()
    worlds = oracle["worlds"]
    assert isinstance(worlds, list)
    posterior = {
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": worlds,
    }
    program = compile_whole_turn_programs(oracle)
    program["producer"] = {
        "root_snapshot_builds": 2,
        "saved_root_snapshot_builds": 3,
        "transition_execution_cache_hits": 4,
        "transition_execution_cache_misses": 5,
        "fresh_showdown_turn_executions": 6,
        "reused_showdown_turn_executions": 7,
    }

    result = transition_program_belief_result(
        fixture=fixture,
        posterior=posterior,
        transition_program=program,
        evaluator=_PosteriorSpreadEvaluator(),
    )

    assert result.status == "selected"
    assert result.action == "safe"
    assert result.reason == "transition-program-public-belief"
    assert result.diagnostics["transition_evaluations"] == 3
    assert result.diagnostics["root_snapshot_builds"] == 2
    assert result.diagnostics["saved_root_snapshot_builds"] == 3
    assert result.diagnostics["transition_execution_cache_hits"] == 4
    assert result.diagnostics["transition_execution_cache_misses"] == 5
    assert result.diagnostics["fresh_showdown_turn_executions"] == 6
    assert result.diagnostics["reused_showdown_turn_executions"] == 7
    assert result.diagnostics["evaluator_calls"] == 3
    assert set(result.diagnostics["public_belief_root_values"]) == {"risky", "safe"}


def test_selective_policy_uses_high_margin_learned_action_without_exact_search() -> None:
    result = selective_belief_result(
        fixture=_selective_fixture(),
        oracle=_strategy_fusion_oracle(),
        evaluator=_FakeEvaluator(0.80),
        search_gate=PolicyMarginSearchGate(search_if_margin_at_most=0.20),
    )

    assert result.action == "risky"
    assert result.reason == "learned-public-belief"
    assert result.diagnostics["learned_route"] == "direct-policy"


def test_selective_policy_searches_low_margin_position() -> None:
    result = selective_belief_result(
        fixture=_selective_fixture(),
        oracle=_strategy_fusion_oracle(),
        evaluator=_FakeEvaluator(0.10),
        search_gate=PolicyMarginSearchGate(search_if_margin_at_most=0.20),
    )

    assert result.action == "safe"
    assert result.reason == "bounded-public-belief"
    assert result.diagnostics["learned_route"] == "exact-public-belief-search"


def test_selective_policy_falls_back_to_exact_search_on_evaluator_error() -> None:
    class BrokenEvaluator(_FakeEvaluator):
        def predict(self, inputs):
            raise RuntimeError("bad checkpoint runtime")

    result = selective_belief_result(
        fixture=_selective_fixture(),
        oracle=_strategy_fusion_oracle(),
        evaluator=BrokenEvaluator(0.80),
        search_gate=PolicyMarginSearchGate(search_if_margin_at_most=0.20),
    )

    assert result.action == "safe"
    assert result.reason == "bounded-public-belief"
    assert result.diagnostics["learned_route"] == "search-after-evaluator-error"
    assert result.diagnostics["learned_evaluator_error"]["type"] == "RuntimeError"

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
    assert set(result.diagnostics["public_belief_root_values"]) == {"risky", "safe"}
    assert result.diagnostics["public_belief_search_horizons"] == 1


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



def test_live_high_margin_route_skips_transition_oracle_probe() -> None:
    class LiveEvaluator:
        spec = BeliefEvaluatorSpec(
            public_width=8,
            world_width=8,
            action_width=8,
            hidden_width=8,
            world_hidden_width=8,
        )
        identity = {
            "checkpoint_digest": "sha256:" + "b" * 64,
            "observability": "public_belief_only",
        }

        def predict(self, inputs) -> BeliefPrediction:
            return BeliefPrediction(
                value=0.4,
                legal_actions=inputs.legal_actions,
                probabilities=(0.95, 0.05),
                selected_action=inputs.legal_actions[0],
                policy_margin=0.90,
                policy_entropy_bits=0.29,
            )

    fixture = live_fixture(_state(), _protocol())
    policy = object.__new__(PinnedShowdownBeliefPolicy)
    policy._configuration_error = None
    policy.learned_evaluator = LiveEvaluator()
    policy.search_gate = PolicyMarginSearchGate(search_if_margin_at_most=0.20)
    policy._probe_posterior = lambda source: {
        "schema": "azelficoast.live-belief-posterior",
        "schema_version": 1,
        "source_fixture_id": fixture.fixture_id,
        "showdown_commit": "a5df8274e85b0889bf2a9b3422a08b39732374fc",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "legal_actions": list(fixture.legal_actions),
        "worlds": [
            {"world_id": "a", "weight": 0.5, "hidden": {"item": "band"}},
            {"world_id": "b", "weight": 0.5, "hidden": {"item": "scarf"}},
        ],
    }

    def unexpected_full_probe(source):
        raise AssertionError("high-margin learned route expanded the transition oracle")

    policy._probe = unexpected_full_probe

    result = policy.choose(fixture)

    assert result.action == fixture.legal_actions[0]
    assert result.reason == "learned-public-belief"
    assert result.diagnostics["learned_route"] == "direct-policy"



def test_live_high_margin_route_does_not_require_opponent_response_model() -> None:
    class LiveEvaluator:
        spec = BeliefEvaluatorSpec(
            public_width=8,
            world_width=8,
            action_width=8,
            hidden_width=8,
            world_hidden_width=8,
        )
        identity = {
            "checkpoint_digest": "sha256:" + "d" * 64,
            "observability": "public_belief_only",
        }

        def predict(self, inputs) -> BeliefPrediction:
            return BeliefPrediction(
                value=0.4,
                legal_actions=inputs.legal_actions,
                probabilities=(0.95, 0.05),
                selected_action=inputs.legal_actions[0],
                policy_margin=0.90,
                policy_entropy_bits=0.29,
            )

    protocol = (
        (
            ("", "player", "p1", "Azelficoast"),
            ("", "player", "p2", "Rival"),
        ),
    )
    fixture = live_fixture(_state(), protocol)
    policy = object.__new__(PinnedShowdownBeliefPolicy)
    policy._configuration_error = None
    policy.learned_evaluator = LiveEvaluator()
    policy.search_gate = PolicyMarginSearchGate(search_if_margin_at_most=0.20)
    policy._probe_posterior = lambda source: {
        "schema": "azelficoast.live-belief-posterior",
        "schema_version": 1,
        "source_fixture_id": fixture.fixture_id,
        "showdown_commit": "a5df8274e85b0889bf2a9b3422a08b39732374fc",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "legal_actions": list(fixture.legal_actions),
        "worlds": [
            {"world_id": "a", "weight": 0.5, "hidden": {"item": "band"}},
            {"world_id": "b", "weight": 0.5, "hidden": {"item": "scarf"}},
        ],
    }

    def unexpected_exact_probe(source):
        raise AssertionError("learned-only decision requested an opponent transaction")

    policy._probe_transition_program = unexpected_exact_probe
    policy._probe = unexpected_exact_probe

    result = policy.choose(fixture)

    assert result.action == fixture.legal_actions[0]
    assert result.reason == "learned-public-belief"


def test_live_low_margin_route_uses_strategy_mixture_without_move_history() -> None:
    protocol = (
        (
            ("", "player", "p1", "Azelficoast"),
            ("", "player", "p2", "Rival"),
        ),
    )
    fixture = live_fixture(_state(), protocol)
    actions = tuple(fixture.legal_actions)
    oracle = _program_search_oracle(
        fixture_id=fixture.fixture_id,
        showdown_commit="a5df8274e85b0889bf2a9b3422a08b39732374fc",
        legal_actions=(actions[0], actions[1]),
    )
    program = compile_whole_turn_programs(oracle)
    worlds = oracle["worlds"]
    assert isinstance(worlds, list)

    class LiveEvaluator(_PosteriorSpreadEvaluator):
        def predict(self, inputs) -> BeliefPrediction:
            if inputs.legal_actions == actions:
                return BeliefPrediction(
                    value=0.0,
                    legal_actions=inputs.legal_actions,
                    probabilities=(0.55, 0.45),
                    selected_action=actions[0],
                    policy_margin=0.10,
                    policy_entropy_bits=0.99,
                )
            return super().predict(inputs)

    policy = object.__new__(PinnedShowdownBeliefPolicy)
    policy._configuration_error = None
    policy.learned_evaluator = LiveEvaluator()
    policy.search_gate = PolicyMarginSearchGate(search_if_margin_at_most=0.20)
    policy._probe_posterior = lambda source: {
        "schema": "azelficoast.live-belief-posterior",
        "schema_version": 1,
        "source_fixture_id": fixture.fixture_id,
        "showdown_commit": "a5df8274e85b0889bf2a9b3422a08b39732374fc",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "legal_actions": list(actions),
        "worlds": worlds,
    }
    seen_policy: dict[str, object] = {}

    def program_probe(source):
        seen_policy.update(source["opponent_policy"])
        return program

    policy._probe_transition_program = program_probe
    policy._probe = lambda source: (_ for _ in ()).throw(
        AssertionError("TransitionProgram live search expanded the exhaustive oracle")
    )

    result = policy.choose(fixture)

    assert result.action == actions[1]
    assert result.reason == "transition-program-public-belief"
    assert seen_policy == {
        "kind": "strategy-mixture",
        "weighting": "equal-active-strategies",
        "strategies": [
            {"kind": "simple-heuristics"},
            {"kind": "dirty-tricks"},
            {"kind": "max-damage"},
            {"kind": "uniform-legal-moves"},
        ],
        "voluntary_switches": True,
    }


def test_live_low_margin_route_uses_transition_program_without_full_oracle() -> None:
    fixture = live_fixture(_state(), _protocol())
    actions = tuple(fixture.legal_actions)
    assert len(actions) == 2
    oracle = _program_search_oracle(
        fixture_id=fixture.fixture_id,
        showdown_commit="a5df8274e85b0889bf2a9b3422a08b39732374fc",
        legal_actions=(actions[0], actions[1]),
    )
    program = compile_whole_turn_programs(oracle)
    worlds = oracle["worlds"]
    assert isinstance(worlds, list)
    posterior = {
        "schema": "azelficoast.live-belief-posterior",
        "schema_version": 1,
        "source_fixture_id": fixture.fixture_id,
        "showdown_commit": "a5df8274e85b0889bf2a9b3422a08b39732374fc",
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "legal_actions": list(actions),
        "worlds": worlds,
    }

    class LiveEvaluator(_PosteriorSpreadEvaluator):
        def predict(self, inputs) -> BeliefPrediction:
            if inputs.legal_actions == actions:
                return BeliefPrediction(
                    value=0.0,
                    legal_actions=inputs.legal_actions,
                    probabilities=(0.55, 0.45),
                    selected_action=actions[0],
                    policy_margin=0.10,
                    policy_entropy_bits=0.99,
                )
            return super().predict(inputs)

    policy = object.__new__(PinnedShowdownBeliefPolicy)
    policy._configuration_error = None
    policy.learned_evaluator = LiveEvaluator()
    policy.search_gate = PolicyMarginSearchGate(search_if_margin_at_most=0.20)
    policy._probe_posterior = lambda source: posterior
    policy._probe_transition_program = lambda source: program

    def unexpected_full_probe(source):
        raise AssertionError("TransitionProgram live search expanded the exhaustive oracle")

    policy._probe = unexpected_full_probe

    result = policy.choose(fixture)

    assert result.action == actions[1]
    assert result.reason == "transition-program-public-belief"
    assert result.diagnostics["learned_route"] == "transition-program-search"
    assert result.diagnostics["transition_evaluations"] == 3
