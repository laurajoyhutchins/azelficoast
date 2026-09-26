"""Read-only SQL front end for finite partial-information decision queries.

SQLite supplies parsing and admission. Azelficoast owns semantic authority. A small,
reviewed relational rewrite system recognizes SQL forms that are equivalent to the
canonical decision query without pretending that arbitrary admitted SQL has known battle
semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from importlib.resources import files
import itertools
import json
import sqlite3
from typing import Any

from azelficoast.core.planning import DEFAULT_DECISION_PLAN, LogicalPlan

SQL_EXPLAIN_SCHEMA = "azelficoast.core.sql-query-explain"
SQL_EXPLAIN_SCHEMA_VERSION = 6
DECISION_QUERY_SEMANTIC_SCHEMA = "azelficoast.core.decision-query-semantics"
DECISION_QUERY_SEMANTIC_VERSION = 1
DECISION_SQL_SURFACE_SCHEMA = "azelficoast.core.decision-sql-surface"
DECISION_SQL_SURFACE_VERSION = 4

DECISION_EXPECTED_VALUE_QUERY = "decision.expected_value"
DECISION_POLICY_QUERY = "decision.policy"
ACTION_SUMMARY_QUERY = "analysis.action_summary"

_SQL_RESOURCE_PACKAGE = "azelficoast.queries"
_DECISION_SQL_RESOURCE = "decision.sql"
_MAXIMIN_SQL_RESOURCE = "maximin.sql"
_RISK_ADJUSTED_SQL_RESOURCE = "risk_adjusted.sql"
_ACTION_SUMMARY_SQL_RESOURCE = "action_summary.sql"
_DECISION_SCHEMA_RESOURCE = "decision_schema.sql"


def _read_sql_resource(name: str) -> str:
    return (
        files(_SQL_RESOURCE_PACKAGE)
        .joinpath(name)
        .read_text(encoding="utf-8")
        .strip()
    )


DEFAULT_DECISION_SQL = _read_sql_resource(_DECISION_SQL_RESOURCE)
MAXIMIN_SQL = _read_sql_resource(_MAXIMIN_SQL_RESOURCE)
RISK_ADJUSTED_SQL = _read_sql_resource(_RISK_ADJUSTED_SQL_RESOURCE)
ACTION_SUMMARY_SQL = _read_sql_resource(_ACTION_SUMMARY_SQL_RESOURCE)
_SCHEMA = _read_sql_resource(_DECISION_SCHEMA_RESOURCE)
_SCHEMA_SOURCE_SHA256 = "sha256:" + hashlib.sha256(
    _SCHEMA.encode("utf-8")
).hexdigest()

_REQUIRED_RELATIONS = frozenset(
    {"hidden_worlds", "legal_actions", "transitions", "evaluations"}
)
_WRITER_RELATIONS = (
    ("active_worlds", ("world_id", "weight")),
    ("action_value_terms", ("action_id", "weight", "value")),
    (
        "action_statistics",
        (
            "action_id",
            "posterior_mass",
            "expected_value",
            "worst_value",
            "best_value",
        ),
    ),
)
_ALLOWED_RELATIONS = _REQUIRED_RELATIONS | frozenset(
    name for name, _ in _WRITER_RELATIONS
)
_ALLOWED_FUNCTIONS = frozenset({"sum", "min", "max"})
_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
    }
)

_DECISION_RELATIONAL_SEMANTICS = {
    "schema": DECISION_QUERY_SEMANTIC_SCHEMA,
    "schema_version": DECISION_QUERY_SEMANTIC_VERSION,
    "relations": [
        "hidden_worlds",
        "transitions",
        "legal_actions",
        "evaluations",
    ],
    "filter": [
        ["hidden_worlds.active", "=", 1],
        ["hidden_worlds.weight", ">", 0],
    ],
    "joins": [
        ["hidden_worlds.world_id", "=", "transitions.world_id"],
        ["transitions.action_id", "=", "legal_actions.action_id"],
        ["transitions.successor_id", "=", "evaluations.successor_id"],
    ],
    "projection": ["transitions.action_id"],
    "aggregate": [
        "sum",
        ["hidden_worlds.weight", "*", "evaluations.value"],
        "expected_value",
    ],
    "group_by": ["transitions.action_id"],
    "order_by": [
        ["expected_value", "desc"],
        ["transitions.action_id", "asc"],
    ],
}
DECISION_QUERY_SEMANTIC_ID = "sha256:" + hashlib.sha256(
    json.dumps(
        _DECISION_RELATIONAL_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()

_ACTION_SUMMARY_SEMANTICS = {
    "schema": DECISION_QUERY_SEMANTIC_SCHEMA,
    "schema_version": DECISION_QUERY_SEMANTIC_VERSION,
    "query_class": ACTION_SUMMARY_QUERY,
    "source_relation": "action_statistics",
    "projection": [
        "action_id",
        "posterior_mass",
        "expected_value",
        "worst_value",
        "best_value",
    ],
    "order_by": [["action_id", "asc"]],
}
ACTION_SUMMARY_SEMANTIC_ID = "sha256:" + hashlib.sha256(
    json.dumps(
        _ACTION_SUMMARY_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()

@dataclass(frozen=True)
class SQLQueryClass:
    name: str
    canonical_sql: str | None
    source_resource: str | None
    result_columns: tuple[str, ...]
    semantic_identity: str | None
    logical: LogicalPlan | None
    semantic_identity_mode: str


_QUERY_CLASSES = {
    DECISION_EXPECTED_VALUE_QUERY: SQLQueryClass(
        name=DECISION_EXPECTED_VALUE_QUERY,
        canonical_sql=DEFAULT_DECISION_SQL,
        source_resource=_DECISION_SQL_RESOURCE,
        result_columns=("action_id", "expected_value"),
        semantic_identity=DECISION_QUERY_SEMANTIC_ID,
        logical=DEFAULT_DECISION_PLAN,
        semantic_identity_mode="fixed",
    ),
    DECISION_POLICY_QUERY: SQLQueryClass(
        name=DECISION_POLICY_QUERY,
        canonical_sql=None,
        source_resource=None,
        result_columns=("action_id", "score"),
        semantic_identity=None,
        logical=None,
        semantic_identity_mode="source-derived",
    ),
    ACTION_SUMMARY_QUERY: SQLQueryClass(
        name=ACTION_SUMMARY_QUERY,
        canonical_sql=ACTION_SUMMARY_SQL,
        source_resource=_ACTION_SUMMARY_SQL_RESOURCE,
        result_columns=(
            "action_id",
            "posterior_mass",
            "expected_value",
            "worst_value",
            "best_value",
        ),
        semantic_identity=ACTION_SUMMARY_SEMANTIC_ID,
        logical=None,
        semantic_identity_mode="fixed",
    ),
}


_DECISION_SQL_SURFACE = {
    "schema": DECISION_SQL_SURFACE_SCHEMA,
    "schema_version": DECISION_SQL_SURFACE_VERSION,
    "relations": {
        name: {"columns": list(columns)}
        for name, columns in _WRITER_RELATIONS
    },
    "schema_source_sha256": _SCHEMA_SOURCE_SHA256,
    "schema_source": f"{_SQL_RESOURCE_PACKAGE}/{_DECISION_SCHEMA_RESOURCE}",
    "policy_examples": {
        "maximin": f"{_SQL_RESOURCE_PACKAGE}/{_MAXIMIN_SQL_RESOURCE}",
        "risk_adjusted": f"{_SQL_RESOURCE_PACKAGE}/{_RISK_ADJUSTED_SQL_RESOURCE}",
    },
    "query_classes": {
        name: {
            "source": (
                f"{_SQL_RESOURCE_PACKAGE}/{query.source_resource}"
                if query.source_resource is not None
                else None
            ),
            "result_columns": list(query.result_columns),
            "semantic_identity": query.semantic_identity,
            "semantic_identity_mode": query.semantic_identity_mode,
            "ranking": (
                [["score", "desc"], ["action_id", "asc"]]
                if name == DECISION_POLICY_QUERY
                else None
            ),
            "packed_jax_lowering": (
                name == DECISION_EXPECTED_VALUE_QUERY
            ),
        }
        for name, query in _QUERY_CLASSES.items()
    },
}
DECISION_SQL_SURFACE_ID = "sha256:" + hashlib.sha256(
    json.dumps(
        _DECISION_SQL_SURFACE,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


def describe_decision_sql_surface() -> dict[str, Any]:
    """Return the stable writer-facing relations and named SQL query classes."""

    return {
        "schema": DECISION_SQL_SURFACE_SCHEMA,
        "schema_version": DECISION_SQL_SURFACE_VERSION,
        "surface_identity": DECISION_SQL_SURFACE_ID,
        "relations": {
            name: {"columns": list(columns)}
            for name, columns in _WRITER_RELATIONS
        },
        "schema_source_sha256": _SCHEMA_SOURCE_SHA256,
        "schema_source": f"{_SQL_RESOURCE_PACKAGE}/{_DECISION_SCHEMA_RESOURCE}",
        "query_classes": {
            name: {
                "source": (
                    f"{_SQL_RESOURCE_PACKAGE}/{query.source_resource}"
                    if query.source_resource is not None
                    else None
                ),
                "result_columns": list(query.result_columns),
                "semantic_identity": query.semantic_identity,
                "semantic_identity_mode": query.semantic_identity_mode,
                "ranking": (
                    [["score", "desc"], ["action_id", "asc"]]
                    if name == DECISION_POLICY_QUERY
                    else None
                ),
                "packed_jax_lowering": (
                    name == DECISION_EXPECTED_VALUE_QUERY
                ),
            }
            for name, query in _QUERY_CLASSES.items()
        },
        "policy_examples": {
            "maximin": f"{_SQL_RESOURCE_PACKAGE}/{_MAXIMIN_SQL_RESOURCE}",
            "risk_adjusted": (
                f"{_SQL_RESOURCE_PACKAGE}/{_RISK_ADJUSTED_SQL_RESOURCE}"
            ),
        },
    }


class SQLQueryError(ValueError):
    """Raised when SQL cannot be admitted as a read-only decision query."""


@dataclass(frozen=True)
class PreparedSQLQuery:
    query_class: str
    sql: str
    sql_sha256: str
    execution_sql: str
    execution_sql_sha256: str
    writer_relations: tuple[str, ...]
    relations: tuple[str, ...]
    functions: tuple[str, ...]
    sqlite_version: str
    sqlite_query_plan: tuple[str, ...]
    sqlite_program_sha256: str
    sql_surface_identity: str
    semantic_identity: str | None
    equivalence_rule: str | None
    equivalence_scope: str | None
    logical: LogicalPlan | None


def _sql_sha256(sql: str) -> str:
    return "sha256:" + hashlib.sha256(sql.encode("utf-8")).hexdigest()


def _prepare_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(_SCHEMA)
    return connection


def _sqlite_program_sha256_from_rows(rows: list[tuple[Any, ...]]) -> str:
    """Hash SQLite's executable program without unstable human comments."""

    material = [list(row[:-1]) for row in rows]
    payload = json.dumps(
        material,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sqlite_program_sha256(connection: sqlite3.Connection, sql: str) -> str:
    rows = connection.execute("EXPLAIN " + sql).fetchall()
    return _sqlite_program_sha256_from_rows(rows)


@lru_cache(maxsize=None)
def _canonical_sqlite_program_sha256(query_class: str) -> str:
    contract = _QUERY_CLASSES[query_class]
    if contract.canonical_sql is None:
        raise SQLQueryError(
            f"{query_class} has source-derived semantics, not canonical bytecode"
        )
    connection = _prepare_connection()
    try:
        return _sqlite_program_sha256(connection, contract.canonical_sql)
    finally:
        connection.close()


def _normalize_reviewed_sql_source(sql: str) -> str:
    """Normalize incidental presentation outside quoted SQL tokens.

    SQLite remains the parser. This helper only removes writer-level differences already
    known to be semantically inert for a parsed query: case, whitespace, a trailing
    semicolon, and SQL comments. Quoted material is preserved byte-for-byte.
    """

    source = sql.strip()
    if source.endswith(";"):
        source = source[:-1].rstrip()

    output: list[str] = []
    pending_space = False
    quote: str | None = None
    index = 0
    while index < len(source):
        character = source[index]
        if quote is not None:
            output.append(character)
            if character == quote:
                if index + 1 < len(source) and source[index + 1] == quote:
                    output.append(source[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue

        if source.startswith("--", index):
            pending_space = True
            index += 2
            while index < len(source) and source[index] not in "\r\n":
                index += 1
            continue

        if source.startswith("/*", index):
            pending_space = True
            end = source.find("*/", index + 2)
            index = len(source) if end < 0 else end + 2
            continue

        if character in {"'", '"', "`"}:
            if pending_space and output and output[-1] != " ":
                output.append(" ")
            pending_space = False
            quote = character
            output.append(character)
            index += 1
            continue

        if character.isspace():
            pending_space = True
            index += 1
            continue

        if pending_space and output and output[-1] != " ":
            output.append(" ")
        pending_space = False
        output.append(character.lower())
        index += 1

    return "".join(output).strip()


def _policy_execution_sql(sql: str) -> str:
    source = sql.strip()
    if source.endswith(";"):
        source = source[:-1].rstrip()
    return f"""WITH __azelficoast_policy AS (
{source}
)
SELECT
    action_id,
    score
FROM __azelficoast_policy
ORDER BY score DESC, action_id ASC"""


def policy_semantic_identity(sql: str) -> str:
    """Identify exact policy semantics without requiring Python registration."""

    material = {
        "schema": DECISION_QUERY_SEMANTIC_SCHEMA,
        "schema_version": DECISION_QUERY_SEMANTIC_VERSION,
        "query_class": DECISION_POLICY_QUERY,
        "sql_surface_identity": DECISION_SQL_SURFACE_ID,
        "writer_sql": _normalize_reviewed_sql_source(sql),
        "result_columns": ["action_id", "score"],
        "ranking": [["score", "desc"], ["action_id", "asc"]],
    }
    payload = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


_RELATION_SOURCE = {
    "w": "hidden_worlds AS w",
    "t": "transitions AS t",
    "a": "legal_actions AS a",
    "e": "evaluations AS e",
}
_JOIN_EDGE = {
    frozenset(("w", "t")): "t.world_id = w.world_id",
    frozenset(("t", "a")): "a.action_id = t.action_id",
    frozenset(("t", "e")): "e.successor_id = t.successor_id",
}


def _connected_join_orders() -> tuple[tuple[str, ...], ...]:
    orders: list[tuple[str, ...]] = []
    for candidate in itertools.permutations(("w", "t", "a", "e")):
        admitted = {candidate[0]}
        valid = True
        for relation in candidate[1:]:
            if not any(
                frozenset((relation, prior)) in _JOIN_EDGE
                for prior in admitted
            ):
                valid = False
                break
            admitted.add(relation)
        if valid:
            orders.append(candidate)
    return tuple(orders)


def _join_sql(
    order: tuple[str, ...],
    *,
    world_source: str,
) -> str:
    sources = dict(_RELATION_SOURCE)
    sources["w"] = world_source
    clauses = [f"FROM {sources[order[0]]}"]
    admitted = {order[0]}
    for relation in order[1:]:
        edge = next(
            _JOIN_EDGE[frozenset((relation, prior))]
            for prior in admitted
            if frozenset((relation, prior)) in _JOIN_EDGE
        )
        clauses.append(f"JOIN {sources[relation]} ON {edge}")
        admitted.add(relation)
    return "\n".join(clauses)


def _inline_decision_sql(
    order: tuple[str, ...],
    predicates: tuple[str, str],
) -> str:
    return f"""
SELECT
    t.action_id AS action_id,
    SUM(w.weight * e.value) AS expected_value
{_join_sql(order, world_source="hidden_worlds AS w")}
WHERE {predicates[0]} AND {predicates[1]}
GROUP BY t.action_id
ORDER BY expected_value DESC, action_id ASC
""".strip()


def _cte_decision_sql(
    order: tuple[str, ...],
    predicates: tuple[str, str],
) -> str:
    return f"""
WITH active_worlds AS (
    SELECT world_id, weight
    FROM hidden_worlds
    WHERE {predicates[0]} AND {predicates[1]}
),
weighted_successors AS (
    SELECT
        t.action_id,
        w.weight,
        e.value
    {_join_sql(order, world_source="active_worlds AS w")}
)
SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM weighted_successors
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC
""".strip()


@lru_cache(maxsize=1)
def _reviewed_equivalence_index() -> dict[str, str]:
    """Generate the finite SQL equivalence class admitted for compiled execution.

    The rules are algebraic and deliberately small:

    * inner-join commutativity/associativity over the fixed join graph;
    * conjunction commutativity for the two active-support predicates;
    * inlining/elimination of the two non-recursive projection CTEs.

    Each generated query is still parsed and authorized by SQLite before this identity
    can be used.
    """

    predicates = (
        "w.active = 1",
        "w.weight > 0",
    )
    cte_predicates = (
        "active = 1",
        "weight > 0",
    )
    index: dict[str, str] = {
        _normalize_reviewed_sql_source(DEFAULT_DECISION_SQL): "writer-view"
    }
    for order in _connected_join_orders():
        for predicate_order in (predicates, tuple(reversed(predicates))):
            source = _inline_decision_sql(order, predicate_order)
            index.setdefault(
                _normalize_reviewed_sql_source(source),
                "cte-inlining+inner-join-commutativity+predicate-commutativity",
            )
        for predicate_order in (cte_predicates, tuple(reversed(cte_predicates))):
            source = _cte_decision_sql(order, predicate_order)
            index.setdefault(
                _normalize_reviewed_sql_source(source),
                "inner-join-commutativity+predicate-commutativity",
            )
    return index


def _reviewed_sql_equivalence(
    sql: str,
    *,
    query_class: str,
    sqlite_program_sha256: str,
) -> tuple[str, str, str] | None:
    contract = _QUERY_CLASSES[query_class]
    normalized = _normalize_reviewed_sql_source(sql)

    if query_class == DECISION_POLICY_QUERY:
        return (
            policy_semantic_identity(sql),
            "policy-source",
            "source-bound",
        )

    if query_class == DECISION_EXPECTED_VALUE_QUERY:
        rule = _reviewed_equivalence_index().get(normalized)
        if rule is not None:
            assert contract.semantic_identity is not None
            return contract.semantic_identity, rule, "reviewed-relational"
    else:
        assert contract.canonical_sql is not None
        if normalized == _normalize_reviewed_sql_source(contract.canonical_sql):
            assert contract.semantic_identity is not None
            return (
                contract.semantic_identity,
                "canonical-source",
                "reviewed-source",
            )

    if sqlite_program_sha256 == _canonical_sqlite_program_sha256(query_class):
        assert contract.semantic_identity is not None
        return (
            contract.semantic_identity,
            "sqlite-program-equivalence",
            "sqlite-version-bound",
        )
    return None


def prepare_sql_query(
    sql: str,
    *,
    query_class: str,
) -> PreparedSQLQuery:
    """Parse, authorize, and recognize one named writer SQL query class."""

    try:
        contract = _QUERY_CLASSES[query_class]
    except KeyError as exc:
        raise SQLQueryError(f"unknown SQL query class: {query_class}") from exc

    canonical = sql.strip()
    if not canonical:
        raise SQLQueryError("SQL query must be non-empty")

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

    execution_sql = (
        _policy_execution_sql(canonical)
        if query_class == DECISION_POLICY_QUERY
        else canonical
    )

    connection = _prepare_connection()
    try:
        connection.set_authorizer(authorize)
        try:
            writer_cursor = connection.execute(canonical)
            writer_columns = tuple(
                description[0]
                for description in (writer_cursor.description or ())
            )
            if writer_columns != contract.result_columns:
                expected = ", ".join(contract.result_columns)
                raise SQLQueryError(
                    f"{query_class} must return exactly {expected}"
                )

            query_plan_rows = connection.execute(
                "EXPLAIN QUERY PLAN " + execution_sql
            ).fetchall()
            program_rows = connection.execute("EXPLAIN " + execution_sql).fetchall()
            connection.execute(execution_sql)
        except SQLQueryError:
            raise
        except sqlite3.DatabaseError as exc:
            raise SQLQueryError(str(exc)) from exc
        finally:
            connection.set_authorizer(None)

        missing = _REQUIRED_RELATIONS.difference(relations)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise SQLQueryError(
                f"SQL query must read required relations: {missing_text}"
            )

        writer_relation_names = frozenset(name for name, _ in _WRITER_RELATIONS)
        writer_relations = relations.intersection(writer_relation_names)
        if (
            query_class == DECISION_POLICY_QUERY
            and "action_statistics" not in writer_relations
        ):
            raise SQLQueryError(
                "decision.policy must read action_statistics directly"
            )

        extra = relations.difference(_ALLOWED_RELATIONS)
        if extra:
            extra_text = ", ".join(sorted(extra))
            raise SQLQueryError(
                f"SQL query read unsupported relations: {extra_text}"
            )

        sqlite_program_sha256 = _sqlite_program_sha256_from_rows(program_rows)
        equivalent = _reviewed_sql_equivalence(
            canonical,
            query_class=query_class,
            sqlite_program_sha256=sqlite_program_sha256,
        )
        semantic_identity = equivalent[0] if equivalent is not None else None
        equivalence_rule = equivalent[1] if equivalent is not None else None
        equivalence_scope = equivalent[2] if equivalent is not None else None
        logical = contract.logical if equivalent is not None else None

        return PreparedSQLQuery(
            query_class=query_class,
            sql=canonical,
            sql_sha256=_sql_sha256(canonical),
            execution_sql=execution_sql,
            execution_sql_sha256=_sql_sha256(execution_sql),
            writer_relations=tuple(sorted(writer_relations)),
            relations=tuple(sorted(relations)),
            functions=tuple(sorted(functions)),
            sqlite_version=sqlite3.sqlite_version,
            sqlite_query_plan=tuple(str(row[3]) for row in query_plan_rows),
            sqlite_program_sha256=sqlite_program_sha256,
            sql_surface_identity=DECISION_SQL_SURFACE_ID,
            semantic_identity=semantic_identity,
            equivalence_rule=equivalence_rule,
            equivalence_scope=equivalence_scope,
            logical=logical,
        )
    finally:
        connection.close()


def prepare_decision_query(sql: str = DEFAULT_DECISION_SQL) -> PreparedSQLQuery:
    return prepare_sql_query(
        sql,
        query_class=DECISION_EXPECTED_VALUE_QUERY,
    )


def prepare_policy_query(sql: str) -> PreparedSQLQuery:
    return prepare_sql_query(
        sql,
        query_class=DECISION_POLICY_QUERY,
    )


def prepare_action_summary_query(
    sql: str = ACTION_SUMMARY_SQL,
) -> PreparedSQLQuery:
    return prepare_sql_query(
        sql,
        query_class=ACTION_SUMMARY_QUERY,
    )


def explain_sql_query(query: PreparedSQLQuery) -> dict[str, Any]:
    """Return admission, semantic-recognition, and SQLite planner evidence."""

    return {
        "schema": SQL_EXPLAIN_SCHEMA,
        "schema_version": SQL_EXPLAIN_SCHEMA_VERSION,
        "query_class": query.query_class,
        "sql_sha256": query.sql_sha256,
        "execution_sql_sha256": query.execution_sql_sha256,
        "writer_relations": list(query.writer_relations),
        "relations": list(query.relations),
        "functions": list(query.functions),
        "sql_surface_identity": query.sql_surface_identity,
        "semantic_identity": query.semantic_identity,
        "equivalence_rule": query.equivalence_rule,
        "equivalence_scope": query.equivalence_scope,
        "logical_operators": (
            [operator.value for operator in query.logical.operators]
            if query.logical is not None
            else []
        ),
        "sqlite": {
            "version": query.sqlite_version,
            "query_plan": list(query.sqlite_query_plan),
            "program_sha256": query.sqlite_program_sha256,
        },
    }


def explain_decision_query(query: PreparedSQLQuery) -> dict[str, Any]:
    if query.query_class != DECISION_EXPECTED_VALUE_QUERY:
        raise SQLQueryError("expected decision.expected_value query")
    return explain_sql_query(query)

