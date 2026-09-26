from __future__ import annotations

import pytest

from azelficoast.core.planning import DEFAULT_DECISION_PLAN
from azelficoast.core.sql import (
    DEFAULT_DECISION_SQL,
    SQLDecisionQueryError,
    explain_decision_query,
    prepare_decision_query,
)


def test_default_decision_sql_is_admitted_by_real_sqlite_parser() -> None:
    prepared = prepare_decision_query(DEFAULT_DECISION_SQL)

    assert prepared.sql == DEFAULT_DECISION_SQL
    assert prepared.relations == (
        "evaluations",
        "hidden_worlds",
        "legal_actions",
        "transitions",
    )
    assert prepared.functions == ("sum",)
    assert prepared.logical == DEFAULT_DECISION_PLAN
    assert prepared.sqlite_version
    assert any(
        "SCAN" in step or "SEARCH" in step
        for step in prepared.sqlite_query_plan
    )


def test_explain_binds_exact_sql_and_planner_environment() -> None:
    prepared = prepare_decision_query(DEFAULT_DECISION_SQL)
    repeated = prepare_decision_query(DEFAULT_DECISION_SQL)
    explanation = explain_decision_query(prepared)

    assert prepared.sql_sha256 == repeated.sql_sha256
    assert explanation["schema"] == "azelficoast.core.sql-query-explain"
    assert explanation["schema_version"] == 1
    assert explanation["sql_sha256"] == prepared.sql_sha256
    assert explanation["relations"] == list(prepared.relations)
    assert explanation["functions"] == ["sum"]
    assert explanation["logical_operators"] == [
        operator.value for operator in DEFAULT_DECISION_PLAN.operators
    ]
    assert explanation["sqlite"]["version"] == prepared.sqlite_version
    assert explanation["sqlite"]["query_plan"] == list(prepared.sqlite_query_plan)


def test_sql_dsl_fails_closed_on_mutation() -> None:
    with pytest.raises(SQLDecisionQueryError, match="not authorized"):
        prepare_decision_query("DELETE FROM hidden_worlds")


def test_sql_dsl_fails_closed_on_unapproved_function() -> None:
    hostile = DEFAULT_DECISION_SQL.replace(
        "SUM(weight * value)",
        "random()",
    )

    with pytest.raises(SQLDecisionQueryError, match="not authorized"):
        prepare_decision_query(hostile)


def test_sql_dsl_requires_all_authoritative_relations() -> None:
    incomplete = """
    SELECT
        world_id AS action_id,
        SUM(weight) AS expected_value
    FROM hidden_worlds
    GROUP BY world_id
    """

    with pytest.raises(
        SQLDecisionQueryError,
        match="evaluations, legal_actions, transitions",
    ):
        prepare_decision_query(incomplete)


def test_sql_dsl_requires_decision_result_shape() -> None:
    wrong_shape = DEFAULT_DECISION_SQL.replace(
        "AS expected_value",
        "AS score",
    ).replace(
        "ORDER BY expected_value DESC",
        "ORDER BY score DESC",
    )

    with pytest.raises(
        SQLDecisionQueryError,
        match="exactly action_id, expected_value",
    ):
        prepare_decision_query(wrong_shape)


def test_sql_dsl_rejects_multiple_statements() -> None:
    with pytest.raises(SQLDecisionQueryError):
        prepare_decision_query(DEFAULT_DECISION_SQL + "; SELECT 1")
