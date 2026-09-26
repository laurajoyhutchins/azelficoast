from __future__ import annotations

import pytest

from azelficoast.core.planning import DEFAULT_DECISION_PLAN
from azelficoast.core.sql import (
    DECISION_QUERY_SEMANTIC_ID,
    DECISION_SQL_SURFACE_ID,
    DEFAULT_DECISION_SQL,
    SQLDecisionQueryError,
    describe_decision_sql_surface,
    explain_decision_query,
    prepare_decision_query,
)


def test_default_decision_sql_is_admitted_by_real_sqlite_parser() -> None:
    prepared = prepare_decision_query(DEFAULT_DECISION_SQL)

    assert prepared.sql == DEFAULT_DECISION_SQL
    assert prepared.relations == (
        "action_value_terms",
        "active_worlds",
        "evaluations",
        "hidden_worlds",
        "legal_actions",
        "transitions",
    )
    assert prepared.functions == ("sum",)
    assert prepared.semantic_identity == DECISION_QUERY_SEMANTIC_ID
    assert prepared.equivalence_rule == "writer-view"
    assert prepared.equivalence_scope == "reviewed-relational"
    assert prepared.sql_surface_identity == DECISION_SQL_SURFACE_ID
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
    assert explanation["schema_version"] == 4
    assert explanation["sql_sha256"] == prepared.sql_sha256
    assert explanation["relations"] == list(prepared.relations)
    assert explanation["functions"] == ["sum"]
    assert explanation["semantic_identity"] == DECISION_QUERY_SEMANTIC_ID
    assert explanation["equivalence_rule"] == "writer-view"
    assert explanation["equivalence_scope"] == "reviewed-relational"
    assert explanation["sql_surface_identity"] == DECISION_SQL_SURFACE_ID
    assert explanation["logical_operators"] == [
        operator.value for operator in DEFAULT_DECISION_PLAN.operators
    ]
    assert explanation["sqlite"]["version"] == prepared.sqlite_version
    assert explanation["sqlite"]["query_plan"] == list(prepared.sqlite_query_plan)
    assert explanation["sqlite"]["program_sha256"] == prepared.sqlite_program_sha256


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



INLINE_REORDERED_DECISION_SQL = """
SELECT
    t.action_id AS action_id,
    SUM(w.weight * e.value) AS expected_value
FROM evaluations AS e
JOIN transitions AS t ON e.successor_id = t.successor_id
JOIN legal_actions AS a ON a.action_id = t.action_id
JOIN hidden_worlds AS w ON t.world_id = w.world_id
WHERE w.weight > 0 AND w.active = 1
GROUP BY t.action_id
ORDER BY expected_value DESC, action_id ASC
""".strip()


def test_relational_rewrites_share_one_decision_semantic_identity() -> None:
    canonical = prepare_decision_query(DEFAULT_DECISION_SQL)
    rewritten = prepare_decision_query(INLINE_REORDERED_DECISION_SQL)

    assert rewritten.sql_sha256 != canonical.sql_sha256
    assert rewritten.semantic_identity == canonical.semantic_identity
    assert rewritten.semantic_identity == DECISION_QUERY_SEMANTIC_ID
    assert rewritten.logical == canonical.logical == DEFAULT_DECISION_PLAN
    assert rewritten.equivalence_rule == (
        "cte-inlining+inner-join-commutativity+predicate-commutativity"
    )
    assert rewritten.equivalence_scope == "reviewed-relational"


def test_relational_equivalence_ignores_incidental_case_and_whitespace() -> None:
    rewritten = INLINE_REORDERED_DECISION_SQL.lower().replace(
        "select\n",
        "SELECT     \n",
        1,
    ) + ";"

    prepared = prepare_decision_query(rewritten)

    assert prepared.semantic_identity == DECISION_QUERY_SEMANTIC_ID
    assert prepared.logical == DEFAULT_DECISION_PLAN


def test_admitted_sql_without_reviewed_equivalence_gets_no_logical_authority() -> None:
    changed = INLINE_REORDERED_DECISION_SQL.replace(
        "w.weight > 0",
        "w.weight >= 0",
    )

    prepared = prepare_decision_query(changed)
    explanation = explain_decision_query(prepared)

    assert prepared.semantic_identity is None
    assert prepared.equivalence_rule is None
    assert prepared.equivalence_scope is None
    assert prepared.logical is None
    assert explanation["semantic_identity"] is None
    assert explanation["equivalence_rule"] is None
    assert explanation["equivalence_scope"] is None
    assert explanation["logical_operators"] == []



def test_writer_surface_exposes_small_stable_relations() -> None:
    surface = describe_decision_sql_surface()

    assert surface["surface_identity"] == DECISION_SQL_SURFACE_ID
    assert surface["relations"] == {
        "active_worlds": {"columns": ["world_id", "weight"]},
        "action_value_terms": {
            "columns": ["action_id", "weight", "value"]
        },
    }
    assert surface["required_result_columns"] == [
        "action_id",
        "expected_value",
    ]
    assert surface["canonical_query"] == DEFAULT_DECISION_SQL


def test_default_sql_is_writer_facing_instead_of_physical_join_ceremony() -> None:
    assert "FROM action_value_terms" in DEFAULT_DECISION_SQL
    assert "hidden_worlds" not in DEFAULT_DECISION_SQL
    assert "transitions" not in DEFAULT_DECISION_SQL
    assert "JOIN" not in DEFAULT_DECISION_SQL



def test_writer_comments_do_not_change_decision_semantics() -> None:
    commented = DEFAULT_DECISION_SQL.replace(
        "SELECT",
        "-- rank legal actions by expected value\nSELECT",
        1,
    ).replace(
        "FROM action_value_terms",
        "FROM /* admitted decision terms */ action_value_terms",
    )

    canonical = prepare_decision_query(DEFAULT_DECISION_SQL)
    prepared = prepare_decision_query(commented)

    assert prepared.sql_sha256 != canonical.sql_sha256
    assert prepared.semantic_identity == DECISION_QUERY_SEMANTIC_ID
    assert prepared.equivalence_rule == "writer-view"



PROGRAM_EQUIVALENT_DECISION_SQL = """
SELECT
    terms.action_id AS action_id,
    SUM((terms.weight * terms.value)) AS expected_value
FROM action_value_terms AS terms
GROUP BY terms.action_id
ORDER BY 2 DESC, 1 ASC
""".strip()


def test_sqlite_program_equivalence_accepts_natural_writer_variants() -> None:
    canonical = prepare_decision_query(DEFAULT_DECISION_SQL)
    variant = prepare_decision_query(PROGRAM_EQUIVALENT_DECISION_SQL)

    assert variant.sql_sha256 != canonical.sql_sha256
    assert variant.semantic_identity == DECISION_QUERY_SEMANTIC_ID
    assert variant.logical == DEFAULT_DECISION_PLAN
    assert variant.equivalence_rule == "sqlite-program-equivalence"
    assert variant.equivalence_scope == "sqlite-version-bound"
    assert variant.sqlite_program_sha256 == canonical.sqlite_program_sha256


def test_sqlite_program_equivalence_does_not_accept_changed_execution() -> None:
    changed = DEFAULT_DECISION_SQL.replace(
        "SUM(weight * value)",
        "SUM(weight * value) + 0",
    )

    prepared = prepare_decision_query(changed)

    assert prepared.semantic_identity is None
    assert prepared.logical is None
    assert prepared.equivalence_rule is None
    assert prepared.equivalence_scope is None
