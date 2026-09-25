"""The stable player seam used by the battle harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder

from azelficoast.belief.evaluator import BeliefEvaluatorRuntime
from azelficoast.live.instrumentation import DecisionTraceWriter, battle_view
from azelficoast.live.belief import (
    LiveDecisionResult,
    PinnedShowdownBeliefPolicy,
    live_fixture,
)
from azelficoast.search.selective import PolicyMarginSearchGate


class AzelficoastPlayer(SimpleHeuristicsPlayer):
    """Live player with learned public-belief policy, exact search, and fallback.

    A pinned, built Pokémon Showdown checkout provides posterior reconstruction and
    complete-turn mechanics. Exact search uses an explicit bounded opponent strategy
    mixture: a poke-env-style simple heuristic, a dirty-tricks disruption policy,
    max-damage play, a uniform exploration floor, and observed-move persistence when
    available. The components are mixed
    equally rather than pretending one handcrafted rule is a calibrated human model.
    The bounded prior may switch to a publicly established surviving bench Pokemon and
    may spend Tera when pinned Showdown exposes it as legal; it never invents an
    unrevealed switch target. Unsupported boundaries fall back to poke-env's simple
    heuristics rather than inventing semantics.
    """

    def __init__(
        self,
        *args: Any,
        decision_log: str | Path | None = None,
        trace_source: Mapping[str, Any] | None = None,
        showdown_root: str | Path | None = None,
        belief_timeout_seconds: float = 20.0,
        evaluator_checkpoint: str | Path | None = None,
        search_policy_margin: float = 1.0,
        belief_policy: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if belief_policy is not None and (
            showdown_root is not None or evaluator_checkpoint is not None
        ):
            raise ValueError(
                "provide belief_policy or configured Showdown/evaluator machinery, not both"
            )
        if evaluator_checkpoint is not None and showdown_root is None:
            raise ValueError("evaluator_checkpoint requires showdown_root")
        self._decision_trace = (
            DecisionTraceWriter(decision_log, source=trace_source)
            if decision_log is not None
            else None
        )
        self._protocol_history: dict[str, list[list[list[str]]]] = {}
        self._belief_policy = belief_policy
        if self._belief_policy is None and showdown_root is not None:
            learned_evaluator = None
            search_gate = None
            if evaluator_checkpoint is not None:
                learned_evaluator = BeliefEvaluatorRuntime.from_checkpoint(
                    evaluator_checkpoint
                )
                search_gate = PolicyMarginSearchGate(
                    search_if_margin_at_most=search_policy_margin
                )
            self._belief_policy = PinnedShowdownBeliefPolicy(
                showdown_root,
                timeout_seconds=belief_timeout_seconds,
                learned_evaluator=learned_evaluator,
                search_gate=search_gate,
            )
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        # poke-env funnels the exact Showdown observations through this callback.
        # The dependency is pinned so this private seam is version-fenced.
        if split_messages:
            room = split_messages[0][0].lstrip(">")
            if room:
                self._protocol_history.setdefault(room, []).append(
                    [list(message) for message in split_messages[1:]]
                )
        if self._decision_trace is not None:
            self._decision_trace.record_protocol_batch(split_messages)
        await super()._handle_battle_message(split_messages)

    @staticmethod
    def _order_for_action(
        battle: AbstractBattle,
        action: str,
    ) -> BattleOrder | None:
        for order in getattr(battle, "valid_orders", ()):
            if order.message == action:
                return order
        return None

    def _belief_decision(self, battle: AbstractBattle) -> LiveDecisionResult:
        if self._belief_policy is None:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="belief-search-disabled",
            )

        fixture = live_fixture(
            battle_view(battle),
            self._protocol_history.get(battle.battle_tag, ()),
        )
        try:
            result = self._belief_policy.choose(fixture)
        except Exception as error:
            # A research policy must never turn an otherwise legal live decision
            # into a forfeit. Preserve the exception as evidence and fall back.
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="belief-policy-exception",
                diagnostics={
                    "type": type(error).__name__,
                    "error": str(error)[-1000:],
                },
            )

        if result.action is not None and result.action not in fixture.legal_actions:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="belief-policy-returned-nonlegal-action",
                diagnostics={"action": result.action},
            )
        return result

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        belief = self._belief_decision(battle)
        order = (
            self._order_for_action(battle, belief.action)
            if belief.action is not None
            else None
        )
        selected_policy = "public-belief"

        if order is None:
            if belief.action is not None:
                belief = LiveDecisionResult(
                    action=None,
                    status="fallback",
                    reason="belief-action-order-mismatch",
                    diagnostics={
                        **dict(belief.diagnostics),
                        "requested_action": belief.action,
                    },
                )
            order = super().choose_move(battle)
            selected_policy = "simple-heuristics"

        if self._decision_trace is not None:
            self._decision_trace.record_decision(
                battle,
                order,
                decision_metadata={
                    "selected_policy": selected_policy,
                    "belief": belief.as_trace(),
                },
            )
        return order

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if self._decision_trace is not None:
            self._decision_trace.record_terminal(battle)
        self._protocol_history.pop(battle.battle_tag, None)
        super()._battle_finished_callback(battle)
