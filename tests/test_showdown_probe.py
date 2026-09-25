from __future__ import annotations

from typing import Any, Mapping

from azelficoast.core.program import PROGRAM_SET_SCHEMA, PROGRAM_SET_SCHEMA_VERSION
from azelficoast.live.belief import PinnedShowdownBeliefPolicy
from azelficoast.live.showdown_probe import probe_session_key


def test_probe_session_key_is_canonical_and_content_addressed() -> None:
    first = {
        "fixture": {"turn": 7, "weather": "rain"},
        "legal_actions": ["move hydro", "switch ferrothorn"],
    }
    reordered = {
        "legal_actions": ["move hydro", "switch ferrothorn"],
        "fixture": {"weather": "rain", "turn": 7},
    }
    changed = {
        **first,
        "fixture": {"turn": 8, "weather": "rain"},
    }

    assert probe_session_key(first) == probe_session_key(reordered)
    assert probe_session_key(first) != probe_session_key(changed)


def test_live_policy_routes_posterior_and_program_through_one_runtime() -> None:
    source = {"fixture_id": "fixture-a", "fixture": {"turn": 4}}
    posterior = {
        "schema": "azelficoast.live-belief-posterior",
        "schema_version": 1,
    }
    program = {
        "schema": PROGRAM_SET_SCHEMA,
        "schema_version": PROGRAM_SET_SCHEMA_VERSION,
    }

    class Runtime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Mapping[str, Any], float]] = []

        def posterior(
            self,
            received: Mapping[str, Any],
            *,
            timeout_seconds: float,
        ) -> Mapping[str, Any]:
            self.calls.append(("posterior", received, timeout_seconds))
            return posterior

        def transition_program(
            self,
            received: Mapping[str, Any],
            *,
            timeout_seconds: float,
        ) -> Mapping[str, Any]:
            self.calls.append(("transition_program", received, timeout_seconds))
            return program

        def release(
            self,
            received: Mapping[str, Any],
            *,
            timeout_seconds: float,
        ) -> None:
            self.calls.append(("release", received, timeout_seconds))

    runtime = Runtime()
    policy = object.__new__(PinnedShowdownBeliefPolicy)
    policy.timeout_seconds = 7.5
    policy._probe_runtime = runtime
    policy._probe_document = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("persistent probe unexpectedly used one-shot subprocess")
    )

    assert policy._probe_posterior(source) == posterior
    assert policy._probe_transition_program(source) == program
    policy._release_probe_session(source)

    assert [name for name, _, _ in runtime.calls] == [
        "posterior",
        "transition_program",
        "release",
    ]
    assert runtime.calls[0][1] is source
    assert runtime.calls[1][1] is source
    assert runtime.calls[0][2] == 7.5
    assert runtime.calls[1][2] == 7.5
    assert runtime.calls[2][2] == 1.0
