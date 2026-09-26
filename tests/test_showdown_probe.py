from __future__ import annotations

from pathlib import Path\nfrom typing import Any, Mapping

from azelficoast.core.program import PROGRAM_SET_SCHEMA, PROGRAM_SET_SCHEMA_VERSION
from azelficoast.live.belief import PinnedShowdownBeliefPolicy
from azelficoast.live.execution_planning import (
    CACHE_MODE_PROJECTION,
    CACHE_MODE_PROJECTION_FIRST,
    TransitionRouteHistory,
)
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
            cache_mode: str = "projection",
        ) -> Mapping[str, Any]:
            self.calls.append(
                (f"transition_program:{cache_mode}", received, timeout_seconds)
            )
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
        "transition_program:projection",
        "release",
    ]
    assert runtime.calls[0][1] is source
    assert runtime.calls[1][1] is source
    assert runtime.calls[0][2] == 7.5
    assert runtime.calls[1][2] == 7.5
    assert runtime.calls[2][2] == 1.0



def test_live_policy_can_route_projection_cache_before_exact_cache() -> None:
    source = {"fixture_id": "fixture-b", "fixture": {"turn": 6}}
    program = {
        "schema": PROGRAM_SET_SCHEMA,
        "schema_version": PROGRAM_SET_SCHEMA_VERSION,
        "producer": {
            "transition_cache_mode": CACHE_MODE_PROJECTION_FIRST,
            "physical_cost_observations": {
                "exact_cache_hit_count": 0,
                "exact_cache_hit_total_ms": 0.0,
                "exact_cache_miss_count": 0,
                "exact_cache_miss_total_ms": 0.0,
                "projection_cache_hit_count": 1,
                "projection_cache_hit_total_ms": 0.1,
                "projection_cache_miss_count": 0,
                "projection_cache_miss_total_ms": 0.0,
                "fresh_execution_count": 0,
                "fresh_execution_total_ms": 0.0,
                "route_execution_count": 1,
                "route_total_ms": 0.1,
            },
        },
    }

    class Runtime:
        def __init__(self) -> None:
            self.cache_modes: list[str] = []

        def transition_program(
            self,
            received: Mapping[str, Any],
            *,
            timeout_seconds: float,
            cache_mode: str = CACHE_MODE_PROJECTION,
        ) -> Mapping[str, Any]:
            del received, timeout_seconds
            self.cache_modes.append(cache_mode)
            return program

    history = TransitionRouteHistory(
        exact_first_route_units=10,
        exact_first_route_total_ms=10.0,
        projection_first_route_units=10,
        projection_first_route_total_ms=1.0,
        exact_hits=1,
        exact_misses=9,
        exact_hit_total_ms=0.1,
        exact_miss_total_ms=0.9,
        projection_hits=9,
        projection_misses=1,
        projection_hit_total_ms=0.9,
        projection_miss_total_ms=0.1,
        fresh_executions=1,
        fresh_execution_total_ms=2.0,
        observations=2,
    )
    runtime = Runtime()
    policy = object.__new__(PinnedShowdownBeliefPolicy)
    policy.timeout_seconds = 7.5
    policy._probe_runtime = runtime
    policy._transition_route_history = history
    policy._last_transition_route_plan = {}

    assert policy._probe_transition_program(source) == program
    assert runtime.cache_modes == [CACHE_MODE_PROJECTION_FIRST]
    assert policy._last_transition_route_plan["cache_probe_order"] == [
        "projected-delta",
        "exact-cache",
    ]


def test_node_worker_uses_explicit_probe_session_api() -> None:
    root = Path(__file__).resolve().parents[1]
    worker = (
        root / "showdown" / "runtime" / "probe_real_belief_worker.cjs"
    ).read_text(encoding="utf-8")
    probe = (
        root / "showdown" / "runtime" / "probe_real_belief_trace.cjs"
    ).read_text(encoding="utf-8")

    assert 'const {runProbe} = require("./probe_real_belief_trace.cjs");' in worker
    assert "compileTransitionProgram: result.compileTransitionProgram" in worker
    assert 'require("node:vm")' not in worker
    assert "vm.Script" not in worker
    assert "function runProbe(argv, sourceDocument = null)" in probe
    assert "module.exports = {ProbeError, runProbe};" in probe
