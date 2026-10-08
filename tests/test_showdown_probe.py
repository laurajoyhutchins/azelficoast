from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import copy
import json
import os
import shutil
import subprocess
import time

import pytest

from azelficoast.core.program import PROGRAM_SET_SCHEMA, PROGRAM_SET_SCHEMA_VERSION
from azelficoast.live.belief import PinnedShowdownBeliefPolicy
from azelficoast.live.execution_planning import (
    CACHE_MODE_PROJECTION,
    CACHE_MODE_PROJECTION_FIRST,
    TransitionRouteHistory,
)
from azelficoast.live.showdown_probe import probe_session_key
from azelficoast.live.showdown_probe import PersistentShowdownProbe, ShowdownProbeRuntimeError
from azelficoast.live.belief import live_fixture, public_belief_result


def test_full_oracle_reuses_runtime_with_same_source_and_deadline() -> None:
    source = {"fixture_id": "fixture-a"}
    document = {"source_fixture_id": "fixture-a"}

    class Runtime:
        def oracle(
            self, received: Mapping[str, Any], *, timeout_seconds: float
        ) -> Mapping[str, Any]:
            assert received is source
            assert timeout_seconds == 20.0
            return document

    policy = object.__new__(PinnedShowdownBeliefPolicy)
    policy.timeout_seconds = 20.0
    policy._probe_runtime = Runtime()
    policy._probe_document = lambda *args, **kwargs: pytest.fail("one-shot oracle used")
    assert policy._probe(source) is document


def test_oracle_client_binds_request_and_rejects_missing_document() -> None:
    source = {"fixture_id": "fixture-a"}
    runtime = PersistentShowdownProbe("/unused")
    calls: list[dict[str, Any]] = []

    def request(**kwargs: Any) -> Mapping[str, Any]:
        calls.append(kwargs)
        return {"document": None}

    runtime._request = request  # type: ignore[method-assign]
    with pytest.raises(ShowdownProbeRuntimeError, match="oracle.*document"):
        runtime.oracle(source, timeout_seconds=20.0)
    assert calls == [
        {
            "op": "oracle",
            "session": probe_session_key(source),
            "source": source,
            "timeout_seconds": 20.0,
        }
    ]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node required")
def test_oracle_timeout_and_identity_drift_restart_worker(tmp_path: Path) -> None:
    script = tmp_path / "worker.cjs"
    script.write_text("""
const input = require("node:readline").createInterface({input: process.stdin});
input.on("line", line => {
  const request = JSON.parse(line);
  if (request.source.hang) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5000);
  const id = request.id + (request.source.drift ? 1 : 0);
  process.stdout.write(JSON.stringify({id, ok: true, document: request.source}) + "\\n");
});
""")
    runtime = PersistentShowdownProbe(tmp_path)
    runtime._worker_script = lambda: script  # type: ignore[method-assign]
    try:
        assert runtime.oracle({"fixture_id": "a"}, timeout_seconds=5) == {"fixture_id": "a"}
        assert runtime._generator_cache is not None
        old_cache = Path(runtime._generator_cache.name)
        assert old_cache.is_dir()
        with pytest.raises(subprocess.TimeoutExpired):
            runtime.oracle({"hang": True}, timeout_seconds=0.05)
        assert runtime._process is None
        assert not old_cache.exists()
        assert runtime.oracle({"fixture_id": "b"}, timeout_seconds=5) == {"fixture_id": "b"}
        with pytest.raises(ShowdownProbeRuntimeError, match="identity mismatch"):
            runtime.oracle({"drift": True}, timeout_seconds=5)
        assert runtime._process is None
        assert runtime.oracle({"fixture_id": "a"}, timeout_seconds=5) == {"fixture_id": "a"}
        assert runtime._generator_cache is not None
        final_cache = Path(runtime._generator_cache.name)
    finally:
        runtime.close()
    assert not final_cache.exists()


@pytest.mark.skipif(
    not os.getenv("AZELFICOAST_TEST_SHOWDOWN_ROOT"), reason="pinned Showdown build required"
)
def test_full_oracle_worker_equals_fresh_across_source_changes_and_error() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = PinnedShowdownBeliefPolicy(os.environ["AZELFICOAST_TEST_SHOWDOWN_ROOT"])
    source_a = json.loads(
        (root / "experiments/data/real-belief-source-tinkaton-zapdosgalar-lowhp.json").read_text()
    )
    # Bounded parity query, not population evidence; generator/chance budgets
    # are production defaults and the complete returned oracle must match.
    source_a["state"]["legal_actions"] = ["/choose move gigatonhammer"]
    source_a["fixture_id"] = live_fixture(source_a["state"], ()).fixture_id
    source_b = copy.deepcopy(source_a)
    source_b["state"]["opponent_active"]["current_hp"] = 5
    source_b["state"]["opponent_active"]["hp_fraction"] = 0.05
    source_b["fixture_id"] = live_fixture(source_b["state"], ()).fixture_id
    try:
        fresh = []
        for source in (source_a, source_b):
            started = time.monotonic()
            fresh.append(policy._probe_document(source))
            print(f"fresh oracle seconds: {time.monotonic() - started:.3f}")
        for index, source in enumerate((source_a, source_b, source_a)):
            expected = fresh[index % 2]
            started = time.monotonic()
            actual = policy._probe(source)
            print(f"worker oracle seconds: {time.monotonic() - started:.3f}")
            assert actual == expected
            assert public_belief_result(
                actual, source["state"]["legal_actions"]
            ) == public_belief_result(expected, source["state"]["legal_actions"])
        with pytest.raises(ShowdownProbeRuntimeError):
            policy._probe({**source_a, "showdown_commit": "0" * 40})
        assert policy._probe(source_a) == fresh[0]
    finally:
        policy.close()


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
            self.calls.append((f"transition_program:{cache_mode}", received, timeout_seconds))
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
    worker = (root / "showdown" / "runtime" / "probe_real_belief_worker.cjs").read_text(
        encoding="utf-8"
    )
    probe = (root / "showdown" / "runtime" / "probe_real_belief_trace.cjs").read_text(
        encoding="utf-8"
    )

    assert 'const {runProbe} = require("./probe_real_belief_trace.cjs");' in worker
    assert "compileTransitionProgram: result.compileTransitionProgram" in worker
    assert 'require("node:vm")' not in worker
    assert "vm.Script" not in worker
    assert "function runProbe(argv, sourceDocument = null)" in probe
    assert "module.exports = {ProbeError, runProbe};" in probe
