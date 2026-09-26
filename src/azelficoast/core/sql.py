"""Read-only SQL front end for finite partial-information decision queries.

SQLite supplies the parser and query planner. Azelficoast supplies the authority:
SQL may describe relational composition over already-admitted relations, but it does not
implement mechanics, belief semantics, or evaluator semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import sqlite3
from typing import Any

from azelficoast.core.planning import DEFAULT_DECISION_PLAN, LogicalPlan

SQL_EXPLAIN_SCHEMA = "azelficoast.core.sql-query-explain"
SQL_EXPLAIN_SCHEMA_VERSION = 1

DEFAULT_DECISION_SQL = """
WITH active_worlds AS (
    SELECT world_id, weight
    FROM hidden_worlds
    WHERE active = 1 AND weight > 0
),
weighted_successors AS (
    SELECT
        t.action_id,
        w.weight,
        e.value
    FROM active_worlds AS w
    JOIN transitions AS t
      ON t.world_id = w.world_id
    JOIN legal_actions AS a
      ON a.action_id = t.action_id
    JOIN evaluations AS e
      ON e.successor_id = t.successor_id
)
SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM weighted_successors
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC
""".strip()

_SCHEMA = """
CREATE TABLE hidden_worlds (
    world_id INTEGER PRIMARY KEY,
    weight INTEGER NOT NULL,
    active INTEGER NOT NULL
);
CREATE TABLE legal_actions (
    action_id INTEGER PRIMARY KEY
);
CREATE TABLE transitions (
    world_id INTEGER NOT NULL,
    action_id INTEGER NOT NULL,
    successor_id INTEGER NOT NULL
);
CREATE TABLE evaluations (
    successor_id INTEGER PRIMARY KEY,
    value REAL NOT NULL
);
"""

_REQUIRED_RELATIONS = frozenset(
    {"hidden_worlds", "legal_actions", "transitions", "evaluations"}
)
_ALLOWED_FUNCTIONS = frozenset({"sum"})
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
)


class SQLDecisionQueryError(ValueError):
    """Raised when SQL cannot be admitted as a read-only decision query."""


@dataclass(frozen=True)
class PreparedDecisionQuery:
    sql: str
    sql_sha256: str
    relations: tuple[str, ...]
    functions: tuple[str, ...]
    sqlite_version: str
    sqlite_query_plan: tuple[str, ...]
    logical: LogicalPlan = DEFAULT_DECISION_PLAN


def _sql_sha256(sql: str) -> str:
    return "sha256:" + hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _prepare_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(_SCHEMA)
    return connection


def prepare_decision_query(sql: str) -> PreparedDecisionQuery:
    """Parse and admit one read-only SQL decision query.

    SQLite is deliberately used as the SQL parser rather than maintaining an
    Azelficoast-specific SQL grammar. The query runs only against an empty structural
    schema during admission, so mechanics and evaluation code cannot execute here.
    """

    canonical = sql.strip()
    if not canonical:
        raise SQLDecisionQueryError("decision SQL must be non-empty")

    relations: set[str] = set()
    functions: set[str] = set()

    def authorize(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database: str | None,
        trigger: str | None,
    ) -> int:
        del database, trigger
        if action not in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            if arg1 is not None:
                relations.add(arg1)
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            name = (arg2 or arg1 or "").lower()
            if name not in _ALLOWED_FUNCTIONS:
                return sqlite3.SQLITE_DENY
            functions.add(name)
        return sqlite3.SQLITE_OK

    connection = _prepare_connection()
    try:
        connection.set_authorizer(authorize)
        try:
            query_plan_rows = connection.execute(
                "EXPLAIN QUERY PLAN " + canonical
            ).fetchall()
            cursor = connection.execute(canonical)
        except sqlite3.DatabaseError as exc:
            raise SQLDecisionQueryError(str(exc)) from exc
        finally:
            connection.set_authorizer(None)

        columns = tuple(
            description[0] for description in (cursor.description or ())
        )
        if columns != ("action_id", "expected_value"):
            raise SQLDecisionQueryError(
                "decision SQL must return exactly action_id, expected_value"
            )

        missing = _REQUIRED_RELATIONS.difference(relations)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise SQLDecisionQueryError(
                f"decision SQL must read required relations: {missing_text}"
            )

        extra = relations.difference(_REQUIRED_RELATIONS)
        if extra:
            extra_text = ", ".join(sorted(extra))
            raise SQLDecisionQueryError(
                f"decision SQL read unsupported relations: {extra_text}"
            )

        return PreparedDecisionQuery(
            sql=canonical,
            sql_sha256=_sql_sha256(canonical),
            relations=tuple(sorted(relations)),
            functions=tuple(sorted(functions)),
            sqlite_version=sqlite3.sqlite_version,
            sqlite_query_plan=tuple(str(row[3]) for row in query_plan_rows),
        )
    finally:
        connection.close()


def explain_decision_query(query: PreparedDecisionQuery) -> dict[str, Any]:
    """Return admission evidence plus SQLite's environment-bound query outline."""

    return {
        "schema": SQL_EXPLAIN_SCHEMA,
        "schema_version": SQL_EXPLAIN_SCHEMA_VERSION,
        "sql_sha256": query.sql_sha256,
        "relations": list(query.relations),
        "functions": list(query.functions),
        "logical_operators": [operator.value for operator in query.logical.operators],
        "sqlite": {
            "version": query.sqlite_version,
            "query_plan": list(query.sqlite_query_plan),
        },
    }
