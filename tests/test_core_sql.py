from __future__ import annotations

import hashlib
from importlib.resources import files

import pytest

from azelficoast.core.planning import DEFAULT_DECISION_PLAN
from azelficoast.core.sql import (
    ACTION_SUMMARY_QUERY,
    ACTION_SUMMARY_SEMANTIC_ID,
    ACTION_SUMMARY_SQL,
    DECISION_EXPECTED_VALUE_QUERY,
    DECISION_MAXIMIN_QUERY,
    DECISION_QUERY_SEMANTIC_ID,
    DECISION_SQL_SURFACE_ID,
    DEFAULT_DECISION_SQL,
    MAXIMIN_SEMANTIC_ID,
    MAXIMIN_SQL,
    SQLQueryError,
    describe_decision_sql_surface,
    explain_decision_query,
    explain_sql_query,
    prepare_action_summary_query,
    prepare_decision_query,
    prepare_maximin_query,
    prepare_sql_query,
)


def test_default_decision_sql_is_admitted_by_real_sqlite_parser() -> None:
    prepared = prepare_decision_query(DEFAULT_DECISION_SQL)

    assert prepared.query_class == DECISION_EXPECTED_VALUE_QUERY
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
    assert explanation["schema_version"] == 5
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
    with pytest.raises(SQLQueryError, match="not authorized"):
        prepare_decision_query("DELETE FROM hidden_worlds")


def test_sql_dsl_fails_closed_on_unapproved_function() -> None:
    hostile = DEFAULT_DECISION_SQL.replace(
        "SUM(weight * value)",
        "random()",
    )

    with pytest.raises(SQLQueryError, match="not authorized"):
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
        SQLQueryError,
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
        SQLQueryError,
        match="exactly action_id, expected_value",
    ):
        prepare_decision_query(wrong_shape)


def test_sql_dsl_rejects_multiple_statements() -> None:
    with pytest.raises(SQLQueryError):
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



def test_writer_surface_exposes_named_query_classes() -> None:
    surface = describe_decision_sql_surface()

    assert surface["surface_identity"] == DECISION_SQL_SURFACE_ID
    assert surface["relations"]["active_worlds"] == {
        "columns": ["world_id", "weight"]
    }
    assert surface["relations"]["action_value_terms"] == {
        "columns": ["action_id", "weight", "value"]
    }
    assert surface["relations"]["action_statistics"] == {
        "columns": [
            "action_id",
            "posterior_mass",
            "expected_value",
            "worst_value",
            "best_value",
        ]
    }
    assert surface["query_classes"][DECISION_EXPECTED_VALUE_QUERY][
        "semantic_identity"
    ] == DECISION_QUERY_SEMANTIC_ID
    assert surface["query_classes"][DECISION_EXPECTED_VALUE_QUERY][
        "packed_jax_lowering"
    ] is True
    assert surface["query_classes"][DECISION_MAXIMIN_QUERY][
        "semantic_identity"
    ] == MAXIMIN_SEMANTIC_ID
    assert surface["query_classes"][DECISION_MAXIMIN_QUERY][
        "packed_jax_lowering"
    ] is False
    assert surface["query_classes"][ACTION_SUMMARY_QUERY][
        "semantic_identity"
    ] == ACTION_SUMMARY_SEMANTIC_ID


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



def test_sql_query_classes_are_first_class_packaged_sources() -> None:
    query_source = (
        files("azelficoast.queries")
        .joinpath("decision.sql")
        .read_text(encoding="utf-8")
        .strip()
    )
    maximin_source = (
        files("azelficoast.queries")
        .joinpath("maximin.sql")
        .read_text(encoding="utf-8")
        .strip()
    )
    summary_source = (
        files("azelficoast.queries")
        .joinpath("action_summary.sql")
        .read_text(encoding="utf-8")
        .strip()
    )
    schema_source = (
        files("azelficoast.queries")
        .joinpath("decision_schema.sql")
        .read_text(encoding="utf-8")
        .strip()
    )
    surface = describe_decision_sql_surface()

    assert query_source == DEFAULT_DECISION_SQL
    assert maximin_source == MAXIMIN_SQL
    assert summary_source == ACTION_SUMMARY_SQL
    assert "CREATE VIEW action_statistics" in schema_source
    assert surface["schema_source"] == (
        "azelficoast.queries/decision_schema.sql"
    )
    assert surface["schema_source_sha256"] == (
        "sha256:" + hashlib.sha256(schema_source.encode("utf-8")).hexdigest()
    )
    assert surface["query_classes"][DECISION_EXPECTED_VALUE_QUERY]["source"] == (
        "azelficoast.queries/decision.sql"
    )
    assert surface["query_classes"][DECISION_MAXIMIN_QUERY]["source"] == (
        "azelficoast.queries/maximin.sql"
    )
    assert surface["query_classes"][ACTION_SUMMARY_QUERY]["source"] == (
        "azelficoast.queries/action_summary.sql"
    )
    assert surface["schema_version"] == 3


def test_action_summary_is_a_distinct_admitted_semantic_query_class() -> None:
    prepared = prepare_action_summary_query()
    explanation = explain_sql_query(prepared)

    assert prepared.query_class == ACTION_SUMMARY_QUERY
    assert prepared.semantic_identity == ACTION_SUMMARY_SEMANTIC_ID
    assert prepared.equivalence_rule == "canonical-source"
    assert prepared.equivalence_scope == "reviewed-source"
    assert prepared.logical is None
    assert prepared.functions == ("max", "min", "sum")
    assert explanation["query_class"] == ACTION_SUMMARY_QUERY
    assert explanation["logical_operators"] == []


def test_maximin_is_distinct_policy_semantics_without_jax_authority() -> None:
    prepared = prepare_maximin_query()
    explanation = explain_sql_query(prepared)

    assert prepared.query_class == DECISION_MAXIMIN_QUERY
    assert prepared.semantic_identity == MAXIMIN_SEMANTIC_ID
    assert prepared.semantic_identity != DECISION_QUERY_SEMANTIC_ID
    assert prepared.equivalence_rule == "canonical-source"
    assert prepared.equivalence_scope == "reviewed-source"
    assert prepared.logical is None
    assert explanation["query_class"] == DECISION_MAXIMIN_QUERY


def test_named_query_contract_rejects_wrong_result_shape() -> None:
    with pytest.raises(SQLQueryError, match="action_id, score"):
        prepare_sql_query(
            ACTION_SUMMARY_SQL,
            query_class=DECISION_MAXIMIN_QUERY,
        )


def test_unknown_query_class_fails_closed() -> None:
    with pytest.raises(SQLQueryError, match="unknown SQL query class"):
        prepare_sql_query(
            DEFAULT_DECISION_SQL,
            query_class="decision.unknown",
        )

