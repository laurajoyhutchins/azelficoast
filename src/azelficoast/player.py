"""The stable player seam used by the battle harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player import SimpleHeuristicsPlayer
from poke_env.player.battle_order import BattleOrder

from azelficoast.instrumentation import DecisionTraceWriter


class AzelficoastPlayer(SimpleHeuristicsPlayer):
    """Temporary baseline behind the Azelficoast player interface.

    The harness depends on this class rather than a particular model or search
    implementation. Replacing the decision machinery should not require
    changing local, ladder, or challenge orchestration.
    """

    def __init__(
        self,
        *args: Any,
        decision_log: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        self._decision_trace = (
            DecisionTraceWriter(decision_log) if decision_log is not None else None
        )
        super().__init__(*args, **kwargs)

    async def _handle_battle_message(self, split_messages: list[list[str]]) -> None:
        # poke-env funnels the exact Showdown observations through this callback.
        # The dependency is pinned so this private seam is version-fenced.
        if self._decision_trace is not None:
            self._decision_trace.record_protocol_batch(split_messages)
        await super()._handle_battle_message(split_messages)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        order = super().choose_move(battle)
        if self._decision_trace is not None:
            self._decision_trace.record_decision(battle, order)
        return order

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        if self._decision_trace is not None:
            self._decision_trace.record_terminal(battle)
        super()._battle_finished_callback(battle)
