from __future__ import annotations

import json
from pathlib import Path

import pytest

from azelficoast.research.studies.policy_sweep import (
    DEFAULT_PLAN,
    FIXTURE_SCHEMA,
    FIXTURE_SCHEMA_VERSION,
    PLAN_SCHEMA,
    PLAN_SCHEMA_VERSION,
    PolicySweepError,
    run_plan,
    run_policy_sweep,
)


def _fixtures() -> dict[str, object]:
    return {
        "schema": FIXTURE_SCHEMA,
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "fixtures": [
            {
                "fixture_id": "fixture-a",
                "action_statistics": [
                    {
                        "action_id": "aggressive",
                        "posterior_mass": 1.0,
                        "expected_value": 0.8,
                        "worst_value": -0.4,
                        "best_value": 1.0,
                    },
                    {
                        "action_id": "stable",
                        "posterior_mass": 1.0,
                        "expected_value": 0.65,
                        "worst_value": 0.4,
                        "best_value": 0.8,
                    },
                ],
            }
        ],
    }


def _plan(values: list[float]) -> dict[str, object]:
    return {
        "schema": PLAN_SCHEMA,
        "schema_version": PLAN_SCHEMA_VERSION,
        "policy_resource": "risk_adjusted.sql",
        "fixture_resource": "../fixtures/policy_action_statistics.json",
        "parameter_grid": {"risk_aversion": values},
        "require_all_actions": True,
    }


def test_default_policy_sweep_produces_complete_matched_grid() -> None:
    result = run_plan(DEFAULT_PLAN)

    assert result["passed"] is True
    assert result["grid"]["parameter_names"] == ["risk_aversion"]
    assert result["grid"]["point_count"] == 5
    assert result["fixtures"]["fixture_count"] == 4
    assert result["matched_evidence"] == {
        "matrix_complete": True,
        "same_policy_semantics_across_grid": True,
        "same_fixture_identity_across_grid": True,
        "result_count": 20,
        "expected_result_count": 20,
    }

    summaries = {
        row["fixture_id"]: row for row in result["fixture_summaries"]
    }
    assert summaries["upside-versus-floor"]["chosen_actions"] == [
        "aggressive",
        "stable",
        "stable",
        "stable",
        "stable",
    ]
    assert summaries["late-risk-switch"]["chosen_actions"] == [
        "burst",
        "burst",
        "steady",
        "steady",
        "steady",
    ]
    assert summaries["deterministic-tie-break"]["chosen_actions"] == [
        "alpha",
        "alpha",
        "alpha",
        "beta",
        "beta",
    ]
    assert summaries["robust-dominance"]["chosen_actions"] == [
        "alpha",
        "alpha",
        "alpha",
        "alpha",
        "alpha",
    ]


def test_policy_sweep_binds_one_semantic_policy_to_distinct_grid_points() -> None:
    result = run_policy_sweep(
        plan=_plan([0.25, 0.75]),
        fixtures=_fixtures(),
    )

    points = result["grid"]["points"]
    assert len(points) == 2
    assert points[0]["parameter_bindings_identity"] != (
        points[1]["parameter_bindings_identity"]
    )
    assert points[0]["bound_semantic_identity"] != (
        points[1]["bound_semantic_identity"]
    )

    semantic_ids = {
        row["policy_semantic_identity"] for row in result["results"]
    }
    assert semantic_ids == {result["policy"]["semantic_identity"]}
    fixture_digests = {
        row["fixture_digest"] for row in result["results"]
    }
    assert len(fixture_digests) == 1


def test_parameter_grid_is_the_only_source_of_sweep_values() -> None:
    low_only = run_policy_sweep(
        plan=_plan([0.0]),
        fixtures=_fixtures(),
    )
    high_only = run_policy_sweep(
        plan=_plan([1.0]),
        fixtures=_fixtures(),
    )

    assert low_only["grid"]["point_count"] == 1
    assert high_only["grid"]["point_count"] == 1
    assert low_only["results"][0]["chosen_action"] == "aggressive"
    assert high_only["results"][0]["chosen_action"] == "stable"
    assert low_only["plan_identity"] != high_only["plan_identity"]


def test_duplicate_typed_parameter_bindings_fail_closed() -> None:
    with pytest.raises(PolicySweepError, match="duplicate typed bindings"):
        run_policy_sweep(
            plan=_plan([0.25, 0.25]),
            fixtures=_fixtures(),
        )


def test_duplicate_fixture_ids_fail_closed() -> None:
    fixtures = _fixtures()
    duplicate = dict(fixtures["fixtures"][0])
    fixtures["fixtures"] = [fixtures["fixtures"][0], duplicate]

    with pytest.raises(PolicySweepError, match="fixture IDs must be unique"):
        run_policy_sweep(
            plan=_plan([0.25]),
            fixtures=fixtures,
        )


def test_nonfinite_fixture_values_fail_closed() -> None:
    fixtures = _fixtures()
    fixtures["fixtures"][0]["action_statistics"][0]["worst_value"] = float("nan")

    with pytest.raises(PolicySweepError, match="value must be finite"):
        run_policy_sweep(
            plan=_plan([0.25]),
            fixtures=fixtures,
        )


def test_run_plan_binds_plan_to_named_fixture_resource(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixtures.json"
    fixture_path.write_text(
        json.dumps(_fixtures(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    plan = _plan([0.25])
    plan["fixture_resource"] = "fixtures.json"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(plan, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    result = run_plan(plan_path)

    assert result["grid"]["point_count"] == 1
    assert result["fixtures"]["fixture_count"] == 1
