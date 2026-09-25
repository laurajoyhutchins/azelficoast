from __future__ import annotations

import pytest

from azelficoast.live.timing import (
    BattleClockObservation,
    BattleClockTracker,
    DecisionDeadline,
    DecisionDeadlineExceeded,
    LiveTimingPolicy,
)


def test_tracker_parses_authoritative_showdown_timer_message() -> None:
    now = [100.0]
    tracker = BattleClockTracker(monotonic=lambda: now[0])

    tracker.observe_protocol(
        [
            [">battle-test"],
            ["", "inactive", "Time left: 150 sec this turn | 150 sec total | 60 sec grace"],
        ]
    )

    observation = tracker.observation("battle-test")
    assert observation is not None
    assert observation.turn_seconds_left == 150.0
    assert observation.total_seconds_left == 150.0
    assert observation.grace_seconds_left == 60.0
    assert observation.remaining_seconds(now=105.0) == 145.0


def test_clock_budget_uses_tighter_turn_or_total_limit_minus_reserve() -> None:
    policy = LiveTimingPolicy(
        safety_reserve_seconds=5.0,
        fallback_decision_budget_seconds=20.0,
        operation_timeout_seconds=15.0,
    )
    observation = BattleClockObservation(
        turn_seconds_left=40.0,
        total_seconds_left=25.0,
        grace_seconds_left=0.0,
        observed_at_monotonic=100.0,
    )

    budget = policy.decision_budget(observation, now=103.0)

    assert budget.source == "showdown-clock"
    assert budget.usable_seconds == 17.0
    assert budget.clock_age_seconds == 3.0
    assert budget.safety_reserve_seconds == 5.0


def test_no_clock_uses_explicit_fallback_budget_without_fake_server_reserve() -> None:
    policy = LiveTimingPolicy(
        safety_reserve_seconds=5.0,
        fallback_decision_budget_seconds=12.0,
        operation_timeout_seconds=3.0,
    )

    budget = policy.decision_budget(None)

    assert budget.source == "configured-fallback"
    assert budget.usable_seconds == 12.0
    assert budget.safety_reserve_seconds == 0.0


def test_inactiveoff_clears_stale_clock_observation() -> None:
    tracker = BattleClockTracker(monotonic=lambda: 100.0)
    tracker.observe_protocol(
        [
            [">battle-test"],
            ["", "inactive", "Time left: 30 sec this turn | 90 sec total"],
        ]
    )
    assert tracker.observation("battle-test") is not None

    tracker.observe_protocol(
        [[">battle-test"], ["", "inactiveoff", "Battle timer is now OFF."]]
    )

    assert tracker.observation("battle-test") is None


def test_deadline_caps_each_operation_without_resetting_total_budget() -> None:
    deadline = DecisionDeadline.after(12.0, now=100.0)

    timeout, deadline_limited = deadline.operation_timeout(20.0, now=103.0)

    assert timeout == 9.0
    assert deadline_limited is True
    assert deadline.remaining_seconds(now=108.0) == 4.0


def test_operation_ceiling_still_protects_against_hung_subprocesses() -> None:
    deadline = DecisionDeadline.after(30.0, now=100.0)

    timeout, deadline_limited = deadline.operation_timeout(7.0, now=101.0)

    assert timeout == 7.0
    assert deadline_limited is False


def test_expired_deadline_fails_before_launching_more_work() -> None:
    deadline = DecisionDeadline.after(1.0, now=100.0)

    with pytest.raises(DecisionDeadlineExceeded):
        deadline.operation_timeout(20.0, now=101.0)
