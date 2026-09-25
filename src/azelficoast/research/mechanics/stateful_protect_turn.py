"""Stateful repeated-Protect semantics over the certified compiled turn.

Pokémon Showdown represents consecutive stalling-move history with a counter. The first
Protect is deterministic. After success the counter is 3; each further consecutive
success multiplies it by three up to 729. A repeated Protect succeeds with probability
1 / counter, and failure clears the stall state.

This module keeps that history state outside the fixed-width damage kernel. It resolves
the tiny Protect state machine first, then reuses the existing two-action transition for
the protected or failed-Protect mechanics branch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np

from azelficoast.core.projection import compile_projection_ids
from azelficoast.research.mechanics.gen9_two_attack_turn import (
    P1_ACTION_ATTACK,
    P1_ACTION_PROTECT,
    TwoAttackTurn,
    TwoAttackTurnContext,
    two_attack_turn,
    two_attack_turn_dependency_key,
    two_attack_turn_dependency_signature,
)

PROTECT_ROLL_DENOMINATOR = 729
PROTECT_STALL_COUNTERS = (1, 3, 9, 27, 81, 243, 729)
STATEFUL_PROTECT_SCHEMA_VERSION = 1


class StatefulProtectError(ValueError):
    """Raised when state is outside the bounded repeated-Protect model."""


@dataclass(frozen=True)
class StatefulProtectOutcome:
    turn: TwoAttackTurn
    stall_counter_before: int
    stall_counter_after: int
    protect_succeeded: bool

    @property
    def successor_key(self) -> tuple[int, int]:
        return (self.turn.packed, self.stall_counter_after)


@dataclass(frozen=True)
class ResolvedProtectBranch:
    mechanics_context: TwoAttackTurnContext
    stall_counter_before: int
    stall_counter_after: int
    protect_succeeded: bool


@dataclass(frozen=True)
class StatefulProtectWorld:
    context_index: int
    stall_counter: int
    protect_roll: int
    p2_accuracy_roll: int
    p2_damage_roll: int


@dataclass(frozen=True)
class StatefulProtectProjection:
    class_ids: np.ndarray
    representative_indices: np.ndarray
    effect_signature: str

    @property
    def class_count(self) -> int:
        return int(len(self.representative_indices))


def _validate_counter(counter: int) -> None:
    if counter not in PROTECT_STALL_COUNTERS:
        raise StatefulProtectError(
            f"Protect stall counter must be one of {PROTECT_STALL_COUNTERS!r}"
        )


def _validate_context(context: TwoAttackTurnContext) -> None:
    if context.p1_action_kind != P1_ACTION_PROTECT:
        raise StatefulProtectError("stateful Protect requires a Protect p1 action")
    if context.p1_priority != 4:
        raise StatefulProtectError("stateful Protect requires priority +4")
    if context.p2_priority >= context.p1_priority:
        raise StatefulProtectError(
            "bounded stateful Protect requires the modeled opposing attack to act later"
        )


def protect_success_threshold(counter: int) -> int:
    """Return successful values on the exact 729-point canonical RNG lattice."""

    _validate_counter(counter)
    return PROTECT_ROLL_DENOMINATOR // counter


def protect_succeeds(counter: int, protect_roll: int) -> bool:
    _validate_counter(counter)
    if not 0 <= protect_roll < PROTECT_ROLL_DENOMINATOR:
        raise StatefulProtectError(
            f"Protect roll must be in [0, {PROTECT_ROLL_DENOMINATOR - 1}]"
        )
    return protect_roll < protect_success_threshold(counter)


def showdown_stall_roll(counter: int, protect_roll: int) -> int:
    """Map the canonical 729-point roll onto Showdown's random(counter) draw."""

    protect_succeeds(counter, protect_roll)
    return (protect_roll * counter) // PROTECT_ROLL_DENOMINATOR


def next_stall_counter(counter: int, *, succeeded: bool) -> int:
    _validate_counter(counter)
    if not succeeded:
        return 1
    if counter == 1:
        return 3
    return min(counter * 3, 729)


def _failed_protect_context(context: TwoAttackTurnContext) -> TwoAttackTurnContext:
    """Encode failed Protect as a priority-4 no-hit action in the certified kernel."""

    _validate_context(context)
    return replace(
        context,
        p1_action_kind=P1_ACTION_ATTACK,
        p1_spa_drop_chance=0,
        p1_attack=replace(context.p1_attack, accuracy=1),
    )


def resolve_stateful_protect(
    context: TwoAttackTurnContext,
    *,
    stall_counter: int,
    protect_roll: int,
) -> ResolvedProtectBranch:
    """Resolve history-dependent Protect success before mechanics execution."""

    _validate_context(context)
    succeeded = protect_succeeds(stall_counter, protect_roll)
    return ResolvedProtectBranch(
        mechanics_context=context if succeeded else _failed_protect_context(context),
        stall_counter_before=stall_counter,
        stall_counter_after=next_stall_counter(
            stall_counter,
            succeeded=succeeded,
        ),
        protect_succeeded=succeeded,
    )


def stateful_protect_turn(
    context: TwoAttackTurnContext,
    *,
    stall_counter: int,
    protect_roll: int,
    p2_accuracy_roll: int,
    p2_damage_roll: int,
) -> StatefulProtectOutcome:
    """Execute one Protect turn with exact consecutive-use state."""

    resolved = resolve_stateful_protect(
        context,
        stall_counter=stall_counter,
        protect_roll=protect_roll,
    )

    turn = two_attack_turn(
        resolved.mechanics_context,
        order_tie_roll=0,
        p1_accuracy_roll=99,
        p1_damage_roll=0,
        p1_secondary_roll=99,
        p2_accuracy_roll=p2_accuracy_roll,
        p2_damage_roll=p2_damage_roll,
    )
    return StatefulProtectOutcome(
        turn=turn,
        stall_counter_before=resolved.stall_counter_before,
        stall_counter_after=resolved.stall_counter_after,
        protect_succeeded=resolved.protect_succeeded,
    )


def stateful_protect_dependency_key(
    context: TwoAttackTurnContext,
    *,
    stall_counter: int,
    protect_roll: int,
    p2_accuracy_roll: int,
    p2_damage_roll: int,
) -> tuple[int, ...]:
    """Return the exact dynamic dependency key for one stateful Protect world."""

    resolved = resolve_stateful_protect(
        context,
        stall_counter=stall_counter,
        protect_roll=protect_roll,
    )
    mechanics_key = two_attack_turn_dependency_key(
        resolved.mechanics_context,
        order_tie_roll=0,
        p1_accuracy_roll=99,
        p1_damage_roll=0,
        p1_secondary_roll=99,
        p2_accuracy_roll=p2_accuracy_roll,
        p2_damage_roll=p2_damage_roll,
    )
    return (
        stall_counter,
        int(resolved.protect_succeeded),
        resolved.stall_counter_after,
        *mechanics_key,
    )


def stateful_protect_dependency_document() -> dict[str, object]:
    return {
        "schema": "azelficoast.stateful-protect-dependencies",
        "schema_version": STATEFUL_PROTECT_SCHEMA_VERSION,
        "binds": {
            "two_action_turn": two_attack_turn_dependency_signature(),
        },
        "history": {
            "counter_values": list(PROTECT_STALL_COUNTERS),
            "first_use_counter": 1,
            "on_first_success": 3,
            "on_repeated_success": "min(counter * 3, 729)",
            "on_failure": 1,
        },
        "random": {
            "canonical_roll_denominator": PROTECT_ROLL_DENOMINATOR,
            "success_condition": "roll < 729 / counter",
            "showdown_equivalence": "floor(roll * counter / 729) == 0",
        },
        "dynamic_reads": {
            "success": [
                "p1.protect_stall_counter",
                "rng.protect_stall_roll",
                "protected two-action successor state",
            ],
            "failure": [
                "p1.protect_stall_counter",
                "rng.protect_stall_roll",
                "unprotected two-action dependencies",
            ],
        },
        "claim": (
            "Repeated-Protect history and success randomness are resolved before the "
            "certified two-action mechanics transition. Successful protection drops "
            "blocked opponent damage dependencies; failed protection restores them."
        ),
        "non_claims": [
            "Protect-bypassing moves and abilities are outside this bounded model.",
            "Other stalling moves are not yet represented as action kinds.",
            "Residual and end-turn mechanics remain outside the two-action transition.",
        ],
    }


def stateful_protect_dependency_signature() -> str:
    encoded = json.dumps(
        stateful_protect_dependency_document(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def compile_stateful_protect_projection(
    contexts: Sequence[TwoAttackTurnContext],
    worlds: Sequence[StatefulProtectWorld],
) -> StatefulProtectProjection:
    if not contexts:
        raise StatefulProtectError("at least one Protect context is required")
    if not worlds:
        raise StatefulProtectError("at least one Protect world is required")

    def key_at(index: int) -> tuple[int, ...]:
        world = worlds[index]
        if not 0 <= world.context_index < len(contexts):
            raise StatefulProtectError("world references an unavailable Protect context")
        return stateful_protect_dependency_key(
            contexts[world.context_index],
            stall_counter=world.stall_counter,
            protect_roll=world.protect_roll,
            p2_accuracy_roll=world.p2_accuracy_roll,
            p2_damage_roll=world.p2_damage_roll,
        )

    class_ids, representatives = compile_projection_ids(len(worlds), key_at)
    return StatefulProtectProjection(
        class_ids=class_ids,
        representative_indices=representatives,
        effect_signature=stateful_protect_dependency_signature(),
    )
