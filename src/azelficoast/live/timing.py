"""Deadline-aware timing for live Pokémon Showdown decisions."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Sequence

DEFAULT_LIVE_CLOCK_RESERVE_SECONDS = 5.0
DEFAULT_LIVE_FALLBACK_BUDGET_SECONDS = 20.0
DEFAULT_LIVE_OPERATION_TIMEOUT_SECONDS = 20.0

_TIMER_MESSAGE = re.compile(
    r"^Time left: (?P<turn>\d+) sec this turn \| "
    r"(?P<total>\d+) sec total"
    r"(?: \| (?P<grace>\d+) sec grace)?$"
)


class DecisionDeadlineExceeded(TimeoutError):
    """Raised before work starts when the live decision budget is exhausted."""


@dataclass(frozen=True, slots=True)
class BattleClockObservation:
    """One authoritative timer observation emitted by Pokémon Showdown."""

    turn_seconds_left: float
    total_seconds_left: float
    grace_seconds_left: float
    observed_at_monotonic: float

    def remaining_seconds(self, *, now: float | None = None) -> float:
        observed_remaining = min(
            self.turn_seconds_left,
            self.total_seconds_left + self.grace_seconds_left,
        )
        current = time.monotonic() if now is None else now
        elapsed = max(0.0, current - self.observed_at_monotonic)
        return max(0.0, observed_remaining - elapsed)


@dataclass(frozen=True, slots=True)
class LiveDecisionBudget:
    """The wall-clock budget granted to one live decision."""

    source: str
    usable_seconds: float
    safety_reserve_seconds: float
    clock_turn_seconds_left: float | None = None
    clock_total_seconds_left: float | None = None
    clock_grace_seconds_left: float | None = None
    clock_age_seconds: float | None = None

    def as_trace(self, *, elapsed_seconds: float | None = None) -> dict[str, object]:
        record: dict[str, object] = {
            "source": self.source,
            "usable_seconds": self.usable_seconds,
            "safety_reserve_seconds": self.safety_reserve_seconds,
        }
        if self.clock_turn_seconds_left is not None:
            record["clock_turn_seconds_left"] = self.clock_turn_seconds_left
        if self.clock_total_seconds_left is not None:
            record["clock_total_seconds_left"] = self.clock_total_seconds_left
        if self.clock_grace_seconds_left is not None:
            record["clock_grace_seconds_left"] = self.clock_grace_seconds_left
        if self.clock_age_seconds is not None:
            record["clock_age_seconds"] = self.clock_age_seconds
        if elapsed_seconds is not None:
            record["elapsed_seconds"] = max(0.0, elapsed_seconds)
        return record


@dataclass(frozen=True, slots=True)
class LiveTimingPolicy:
    """Configuration for translating the battle clock into execution budgets."""

    safety_reserve_seconds: float = DEFAULT_LIVE_CLOCK_RESERVE_SECONDS
    fallback_decision_budget_seconds: float = DEFAULT_LIVE_FALLBACK_BUDGET_SECONDS
    operation_timeout_seconds: float = DEFAULT_LIVE_OPERATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if self.safety_reserve_seconds < 0:
            raise ValueError("live timing safety reserve must be non-negative")
        if self.fallback_decision_budget_seconds <= 0:
            raise ValueError("live fallback decision budget must be positive")
        if self.operation_timeout_seconds <= 0:
            raise ValueError("live operation timeout must be positive")

    def decision_budget(
        self,
        observation: BattleClockObservation | None,
        *,
        now: float | None = None,
    ) -> LiveDecisionBudget:
        if observation is None:
            return LiveDecisionBudget(
                source="configured-fallback",
                usable_seconds=self.fallback_decision_budget_seconds,
                safety_reserve_seconds=0.0,
            )

        current = time.monotonic() if now is None else now
        clock_age = max(0.0, current - observation.observed_at_monotonic)
        remaining = observation.remaining_seconds(now=current)
        usable = max(0.0, remaining - self.safety_reserve_seconds)
        return LiveDecisionBudget(
            source="showdown-clock",
            usable_seconds=usable,
            safety_reserve_seconds=self.safety_reserve_seconds,
            clock_turn_seconds_left=observation.turn_seconds_left,
            clock_total_seconds_left=observation.total_seconds_left,
            clock_grace_seconds_left=observation.grace_seconds_left,
            clock_age_seconds=clock_age,
        )


@dataclass(frozen=True, slots=True)
class DecisionDeadline:
    """One absolute monotonic deadline shared by every operation in a decision."""

    expires_at_monotonic: float

    @classmethod
    def after(
        cls,
        seconds: float,
        *,
        now: float | None = None,
    ) -> "DecisionDeadline":
        if seconds <= 0:
            raise DecisionDeadlineExceeded("live decision budget is exhausted")
        current = time.monotonic() if now is None else now
        return cls(expires_at_monotonic=current + seconds)

    def remaining_seconds(self, *, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        return max(0.0, self.expires_at_monotonic - current)

    def check(self, *, now: float | None = None) -> None:
        if self.remaining_seconds(now=now) <= 0:
            raise DecisionDeadlineExceeded("live decision deadline expired")

    def operation_timeout(
        self,
        operation_ceiling_seconds: float,
        *,
        now: float | None = None,
    ) -> tuple[float, bool]:
        if operation_ceiling_seconds <= 0:
            raise ValueError("operation timeout ceiling must be positive")
        remaining = self.remaining_seconds(now=now)
        if remaining <= 0:
            raise DecisionDeadlineExceeded("live decision deadline expired")
        deadline_limited = remaining < operation_ceiling_seconds
        return min(remaining, operation_ceiling_seconds), deadline_limited


class BattleClockTracker:
    """Track Showdown's player-specific timer messages by battle room."""

    def __init__(self, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic = monotonic
        self._by_room: dict[str, BattleClockObservation] = {}

    @staticmethod
    def _message_kind_and_payload(message: Sequence[str]) -> tuple[str, str] | None:
        if len(message) >= 3 and message[0] == "":
            return str(message[1]), str(message[2])
        if len(message) >= 2:
            return str(message[0]), str(message[1])
        return None

    def observe_protocol(self, split_messages: Sequence[Sequence[str]]) -> None:
        if not split_messages:
            return
        room = str(split_messages[0][0]).lstrip(">")
        if not room:
            return

        for message in split_messages[1:]:
            parsed = self._message_kind_and_payload(message)
            if parsed is None:
                continue
            kind, payload = parsed
            if kind == "inactiveoff":
                self._by_room.pop(room, None)
                continue
            if kind != "inactive":
                continue
            match = _TIMER_MESSAGE.fullmatch(payload)
            if match is None:
                continue
            self._by_room[room] = BattleClockObservation(
                turn_seconds_left=float(match.group("turn")),
                total_seconds_left=float(match.group("total")),
                grace_seconds_left=float(match.group("grace") or 0),
                observed_at_monotonic=self._monotonic(),
            )

    def observation(self, room: str) -> BattleClockObservation | None:
        return self._by_room.get(room)

    def clear(self, room: str) -> None:
        self._by_room.pop(room, None)
