"""Matched parameter sweeps for source-derived SQL policies.

The experiment plan owns the policy source, parameter grid, and fixture corpus. Python is
the generic executor and evidence checker; it does not encode coefficient sweeps.
"""

from __future__ import annotations

import itertools
import json
import math
import sqlite3
from importlib.resources import files
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.core.sql import PreparedSQLQuery, prepare_policy_query
from azelficoast.research.contracts import stable_digest

PLAN_SCHEMA = "azelficoast.sql-policy-parameter-sweep-plan"
PLAN_SCHEMA_VERSION = 1
FIXTURE_SCHEMA = "azelficoast.sql-policy-action-statistics-corpus"
FIXTURE_SCHEMA_VERSION = 1
RESULT_SCHEMA = "azelficoast.sql-policy-parameter-sweep"
RESULT_SCHEMA_VERSION = 1

DEFAULT_PLAN = Path(__file__).with_name("plans") / "risk_adjusted_policy_sweep.json"


class PolicySweepError(ValueError):
    """Raised when a SQL policy sweep is incomplete or ambiguous."""


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PolicySweepError(f"{path}: expected a JSON object")
    return value


def _numeric(value: object, *, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicySweepError(f"{path}: expected an integer or real")
    number = float(value)
    if not math.isfinite(number):
        raise PolicySweepError(f"{path}: value must be finite")
    return value


def _load_policy_source(resource: str) -> str:
    if (
        not resource
        or "/" in resource
        or "\\" in resource
        or not resource.endswith(".sql")
    ):
        raise PolicySweepError("policy_resource must name one packaged SQL file")
    try:
        return (
            files("azelficoast.queries")
            .joinpath(resource)
            .read_text(encoding="utf-8")
            .strip()
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise PolicySweepError(f"unknown policy resource {resource!r}") from exc


def _parameter_grid(
    raw_grid: object,
    *,
    policy_sql: str,
) -> tuple[tuple[dict[str, int | float], PreparedSQLQuery], ...]:
    if not isinstance(raw_grid, Mapping) or not raw_grid:
        raise PolicySweepError("parameter_grid must be a non-empty object")

    if not all(isinstance(name, str) and name for name in raw_grid):
        raise PolicySweepError("parameter_grid names must be non-empty strings")
    names = tuple(sorted(str(name) for name in raw_grid))

    axes: list[tuple[int | float, ...]] = []
    for name in names:
        raw_values = raw_grid[name]
        if not isinstance(raw_values, list) or not raw_values:
            raise PolicySweepError(f"parameter_grid.{name} must be a non-empty list")
        axes.append(
            tuple(
                _numeric(value, path=f"parameter_grid.{name}[{index}]")
                for index, value in enumerate(raw_values)
            )
        )

    points: list[tuple[dict[str, int | float], PreparedSQLQuery]] = []
    seen_bindings: set[str] = set()
    semantic_identity: str | None = None
    for values in itertools.product(*axes):
        parameters = dict(zip(names, values, strict=True))
        prepared = prepare_policy_query(policy_sql, parameters=parameters)
        if prepared.semantic_identity is None or prepared.bound_semantic_identity is None:
            raise PolicySweepError("policy grid point lacks semantic identity")
        if semantic_identity is None:
            semantic_identity = prepared.semantic_identity
        elif prepared.semantic_identity != semantic_identity:
            raise PolicySweepError("parameter grid changed policy source semantics")
        if prepared.parameter_bindings_identity in seen_bindings:
            raise PolicySweepError("parameter grid contains duplicate typed bindings")
        seen_bindings.add(prepared.parameter_bindings_identity)
        points.append((parameters, prepared))

    return tuple(points)


def _validate_fixture_rows(
    raw_fixture: object,
    *,
    fixture_index: int,
) -> tuple[str, tuple[dict[str, object], ...], str]:
    if not isinstance(raw_fixture, Mapping):
        raise PolicySweepError(f"fixtures[{fixture_index}] must be an object")
    fixture_id = raw_fixture.get("fixture_id")
    rows = raw_fixture.get("action_statistics")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise PolicySweepError(f"fixtures[{fixture_index}] lacks fixture_id")
    if not isinstance(rows, list) or not rows:
        raise PolicySweepError(f"{fixture_id}: action_statistics must be non-empty")

    checked: list[dict[str, object]] = []
    seen_actions: set[str] = set()
    for row_index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise PolicySweepError(
                f"{fixture_id}: action_statistics[{row_index}] must be an object"
            )
        action_id = raw_row.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise PolicySweepError(
                f"{fixture_id}: action_statistics[{row_index}] lacks action_id"
            )
        if action_id in seen_actions:
            raise PolicySweepError(f"{fixture_id}: duplicate action {action_id!r}")
        seen_actions.add(action_id)

        row = {
            "action_id": action_id,
            "posterior_mass": _numeric(
                raw_row.get("posterior_mass"),
                path=f"{fixture_id}.{action_id}.posterior_mass",
            ),
            "expected_value": _numeric(
                raw_row.get("expected_value"),
                path=f"{fixture_id}.{action_id}.expected_value",
            ),
            "worst_value": _numeric(
                raw_row.get("worst_value"),
                path=f"{fixture_id}.{action_id}.worst_value",
            ),
            "best_value": _numeric(
                raw_row.get("best_value"),
                path=f"{fixture_id}.{action_id}.best_value",
            ),
        }
        checked.append(row)

    checked_rows = tuple(sorted(checked, key=lambda row: str(row["action_id"])))
    fixture_digest = stable_digest(
        {
            "fixture_id": fixture_id,
            "action_statistics": list(checked_rows),
        }
    )
    return fixture_id, checked_rows, fixture_digest


def _execute_policy(
    prepared: PreparedSQLQuery,
    *,
    parameters: Mapping[str, int | float],
    rows: Sequence[Mapping[str, object]],
    require_all_actions: bool,
) -> tuple[dict[str, object], ...]:
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute(
            """
            CREATE TABLE action_statistics (
                action_id TEXT PRIMARY KEY,
                posterior_mass REAL NOT NULL,
                expected_value REAL NOT NULL,
                worst_value REAL NOT NULL,
                best_value REAL NOT NULL
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO action_statistics (
                action_id,
                posterior_mass,
                expected_value,
                worst_value,
                best_value
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    row["action_id"],
                    row["posterior_mass"],
                    row["expected_value"],
                    row["worst_value"],
                    row["best_value"],
                )
                for row in rows
            ],
        )
        result_rows = connection.execute(
            prepared.execution_sql,
            dict(parameters),
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise PolicySweepError(f"policy execution failed: {exc}") from exc
    finally:
        connection.close()

    if not result_rows:
        raise PolicySweepError("policy returned no actions")

    expected_actions = {str(row["action_id"]) for row in rows}
    seen_actions: set[str] = set()
    checked: list[dict[str, object]] = []
    previous: tuple[float, str] | None = None
    for index, raw_row in enumerate(result_rows):
        if len(raw_row) != 2:
            raise PolicySweepError("policy result row must contain action_id and score")
        action_id, raw_score = raw_row
        if not isinstance(action_id, str) or action_id not in expected_actions:
            raise PolicySweepError(f"policy returned unknown action {action_id!r}")
        if action_id in seen_actions:
            raise PolicySweepError(f"policy returned duplicate action {action_id!r}")
        seen_actions.add(action_id)
        score = _numeric(raw_score, path=f"policy_result[{index}].score")
        numeric_score = float(score)
        if previous is not None:
            previous_score, previous_action = previous
            if numeric_score > previous_score or (
                numeric_score == previous_score and action_id < previous_action
            ):
                raise PolicySweepError("policy result violates deterministic ranking")
        previous = (numeric_score, action_id)
        checked.append({"action_id": action_id, "score": score})

    if require_all_actions and seen_actions != expected_actions:
        missing = sorted(expected_actions - seen_actions)
        raise PolicySweepError(f"policy omitted required actions: {missing!r}")

    return tuple(checked)


def run_policy_sweep(
    *,
    plan: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        raise PolicySweepError("unexpected policy sweep plan schema")
    if (
        fixtures.get("schema") != FIXTURE_SCHEMA
        or fixtures.get("schema_version") != FIXTURE_SCHEMA_VERSION
    ):
        raise PolicySweepError("unexpected policy sweep fixture schema")

    policy_resource = plan.get("policy_resource")
    if not isinstance(policy_resource, str):
        raise PolicySweepError("plan lacks policy_resource")
    policy_sql = _load_policy_source(policy_resource)
    points = _parameter_grid(plan.get("parameter_grid"), policy_sql=policy_sql)
    require_all_actions = plan.get("require_all_actions")
    if not isinstance(require_all_actions, bool):
        raise PolicySweepError("require_all_actions must be boolean")

    raw_fixtures = fixtures.get("fixtures")
    if not isinstance(raw_fixtures, list) or not raw_fixtures:
        raise PolicySweepError("fixture corpus must contain fixtures")

    checked_fixtures = tuple(
        _validate_fixture_rows(raw_fixture, fixture_index=index)
        for index, raw_fixture in enumerate(raw_fixtures)
    )
    fixture_ids = [fixture_id for fixture_id, _rows, _digest in checked_fixtures]
    if len(set(fixture_ids)) != len(fixture_ids):
        raise PolicySweepError("fixture IDs must be unique")

    semantic_identities = {
        prepared.semantic_identity for _parameters, prepared in points
    }
    if len(semantic_identities) != 1 or None in semantic_identities:
        raise PolicySweepError("grid must preserve one policy semantic identity")
    policy_semantic_identity = next(iter(semantic_identities))

    results: list[dict[str, object]] = []
    for grid_index, (parameters, prepared) in enumerate(points):
        for fixture_id, rows, fixture_digest in checked_fixtures:
            ranked = _execute_policy(
                prepared,
                parameters=parameters,
                rows=rows,
                require_all_actions=require_all_actions,
            )
            results.append(
                {
                    "grid_index": grid_index,
                    "fixture_id": fixture_id,
                    "fixture_digest": fixture_digest,
                    "policy_semantic_identity": prepared.semantic_identity,
                    "parameter_bindings_identity": (
                        prepared.parameter_bindings_identity
                    ),
                    "bound_semantic_identity": prepared.bound_semantic_identity,
                    "parameters": {
                        binding.name: {
                            "type": binding.value_type,
                            "canonical_value": binding.canonical_value,
                        }
                        for binding in prepared.parameter_bindings
                    },
                    "chosen_action": ranked[0]["action_id"],
                    "chosen_score": ranked[0]["score"],
                    "ranking": list(ranked),
                    "action_count": len(ranked),
                }
            )

    expected_keys = {
        (grid_index, fixture_id)
        for grid_index in range(len(points))
        for fixture_id in fixture_ids
    }
    observed_keys = {
        (int(row["grid_index"]), str(row["fixture_id"])) for row in results
    }
    if observed_keys != expected_keys:
        raise PolicySweepError("policy sweep matrix is incomplete")

    summaries: list[dict[str, object]] = []
    for fixture_id in fixture_ids:
        rows = [row for row in results if row["fixture_id"] == fixture_id]
        chosen_actions = [str(row["chosen_action"]) for row in rows]
        summaries.append(
            {
                "fixture_id": fixture_id,
                "grid_point_count": len(rows),
                "chosen_actions": chosen_actions,
                "unique_chosen_actions": sorted(set(chosen_actions)),
                "policy_changes_across_grid": len(set(chosen_actions)) > 1,
            }
        )

    point_records = [
        {
            "grid_index": index,
            "parameter_bindings": [
                binding.as_record() for binding in prepared.parameter_bindings
            ],
            "parameter_bindings_identity": prepared.parameter_bindings_identity,
            "bound_semantic_identity": prepared.bound_semantic_identity,
        }
        for index, (_parameters, prepared) in enumerate(points)
    ]

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "plan_identity": stable_digest(plan),
        "fixture_corpus_identity": stable_digest(fixtures),
        "policy": {
            "resource": policy_resource,
            "semantic_identity": policy_semantic_identity,
            "sql_sha256": points[0][1].sql_sha256,
            "parameter_names": list(points[0][1].parameter_names),
        },
        "grid": {
            "parameter_names": list(points[0][1].parameter_names),
            "point_count": len(points),
            "points": point_records,
        },
        "fixtures": {
            "fixture_count": len(checked_fixtures),
            "identities": [
                {"fixture_id": fixture_id, "fixture_digest": fixture_digest}
                for fixture_id, _rows, fixture_digest in checked_fixtures
            ],
        },
        "matched_evidence": {
            "matrix_complete": observed_keys == expected_keys,
            "same_policy_semantics_across_grid": True,
            "same_fixture_identity_across_grid": True,
            "result_count": len(results),
            "expected_result_count": len(expected_keys),
        },
        "fixture_summaries": summaries,
        "results": results,
        "passed": True,
    }


def run_plan(path: Path) -> dict[str, Any]:
    plan = _load_object(path)
    fixture_resource = plan.get("fixture_resource")
    if not isinstance(fixture_resource, str) or not fixture_resource:
        raise PolicySweepError("plan lacks fixture_resource")
    fixture_path = (path.parent / fixture_resource).resolve()
    fixtures = _load_object(fixture_path)
    return run_policy_sweep(plan=plan, fixtures=fixtures)


def run_default_plan() -> dict[str, Any]:
    return run_plan(DEFAULT_PLAN)
