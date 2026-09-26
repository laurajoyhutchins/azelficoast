"""SQLite reference implementation of information-set posterior transport.

The compiled search topology remains authoritative for which chance edges belong to one
information set. This module takes that already-authorized incidence relation and
expresses the numeric posterior transport itself as ordinary SQL:

    normalize worlds
    -> join worlds to chance edges
    -> aggregate mass by leaf/world
    -> aggregate mass by leaf
    -> condition world mass inside each leaf

It is a verification/reference backend, not mechanics authority and not a replacement
for the packed JAX hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.resources import files
import json
import math
import sqlite3
from typing import Any, Sequence

from azelficoast.core.planning import LogicalOperator, LogicalPlan

SQL_TRANSPORT_SCHEMA = "azelficoast.core.sql-information-set-transport"
SQL_TRANSPORT_SCHEMA_VERSION = 1
SQL_TRANSPORT_SEMANTIC_SCHEMA = "azelficoast.core.sql-transport-semantics"
SQL_TRANSPORT_SEMANTIC_VERSION = 1

_SQL_RESOURCE_PACKAGE = "azelficoast.queries"
_TRANSPORT_SQL_RESOURCE = "information_set_transport.sql"
_TRANSPORT_SCHEMA_RESOURCE = "information_set_transport_schema.sql"


def _read_sql_resource(name: str) -> str:
    return (
        files(_SQL_RESOURCE_PACKAGE)
        .joinpath(name)
        .read_text(encoding="utf-8")
        .strip()
    )


INFORMATION_SET_TRANSPORT_SQL = _read_sql_resource(_TRANSPORT_SQL_RESOURCE)
_INFORMATION_SET_TRANSPORT_SCHEMA = _read_sql_resource(_TRANSPORT_SCHEMA_RESOURCE)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


INFORMATION_SET_TRANSPORT_SQL_SHA256 = _sha256_text(INFORMATION_SET_TRANSPORT_SQL)
INFORMATION_SET_TRANSPORT_SCHEMA_SHA256 = _sha256_text(
    _INFORMATION_SET_TRANSPORT_SCHEMA
)

INFORMATION_SET_TRANSPORT_PLAN = LogicalPlan(
    operators=(
        LogicalOperator.SCAN,
        LogicalOperator.FILTER,
        LogicalOperator.PROJECT,
        LogicalOperator.UPDATE_BELIEF,
        LogicalOperator.AGGREGATE,
    )
)

_INFORMATION_SET_TRANSPORT_SEMANTICS = {
    "schema": SQL_TRANSPORT_SEMANTIC_SCHEMA,
    "schema_version": SQL_TRANSPORT_SEMANTIC_VERSION,
    "authority_boundary": "authorized leaf/world/chance incidence supplied by caller",
    "input_relations": {
        "worlds": ["world_index", "weight"],
        "leaves": ["leaf_index"],
        "edges": ["leaf_index", "world_index", "chance"],
    },
    "operations": [
        "normalize-world-weights",
        "join-worlds-to-authorized-edges",
        "multiply-world-weight-by-chance",
        "aggregate-leaf-world-mass",
        "aggregate-leaf-mass",
        "normalize-world-mass-within-leaf",
    ],
    "output": [
        "leaf_index",
        "world_index",
        "leaf_world_mass",
        "leaf_mass",
        "conditional_weight",
        "normalized_world_weight",
    ],
}
INFORMATION_SET_TRANSPORT_SEMANTIC_ID = "sha256:" + hashlib.sha256(
    json.dumps(
        _INFORMATION_SET_TRANSPORT_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
).hexdigest()


class SQLTransportError(ValueError):
    """Raised when relational posterior transport cannot be validated."""


@dataclass(frozen=True, slots=True)
class SQLTransportedMass:
    """Dense relational result corresponding to the compiled JAX transport."""

    normalized_world_weights: tuple[float, ...]
    leaf_mass: tuple[float, ...]
    leaf_world_mass: tuple[tuple[float, ...], ...]
    leaf_world_weights: tuple[tuple[float, ...], ...]
    sqlite_version: str
    sqlite_query_plan: tuple[str, ...]
    sqlite_program_sha256: str

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": SQL_TRANSPORT_SCHEMA,
            "schema_version": SQL_TRANSPORT_SCHEMA_VERSION,
            "semantic_identity": INFORMATION_SET_TRANSPORT_SEMANTIC_ID,
            "sql_sha256": INFORMATION_SET_TRANSPORT_SQL_SHA256,
            "schema_sha256": INFORMATION_SET_TRANSPORT_SCHEMA_SHA256,
            "logical_operators": [
                operator.value for operator in INFORMATION_SET_TRANSPORT_PLAN.operators
            ],
            "normalized_world_weights": list(self.normalized_world_weights),
            "leaf_mass": list(self.leaf_mass),
            "leaf_world_mass": [list(row) for row in self.leaf_world_mass],
            "leaf_world_weights": [list(row) for row in self.leaf_world_weights],
            "sqlite": {
                "version": self.sqlite_version,
                "query_plan": list(self.sqlite_query_plan),
                "program_sha256": self.sqlite_program_sha256,
            },
            "authority": {
                "topology": "caller-supplied-authorized-incidence",
                "mechanics": "outside-sql",
                "numeric_transport": "sqlite-reference",
            },
        }


def describe_information_set_transport() -> dict[str, Any]:
    """Describe the packaged relational transport contract without executing it."""

    return {
        "schema": SQL_TRANSPORT_SEMANTIC_SCHEMA,
        "schema_version": SQL_TRANSPORT_SEMANTIC_VERSION,
        "semantic_identity": INFORMATION_SET_TRANSPORT_SEMANTIC_ID,
        "query_source": f"{_SQL_RESOURCE_PACKAGE}/{_TRANSPORT_SQL_RESOURCE}",
        "query_sha256": INFORMATION_SET_TRANSPORT_SQL_SHA256,
        "schema_source": f"{_SQL_RESOURCE_PACKAGE}/{_TRANSPORT_SCHEMA_RESOURCE}",
        "schema_sha256": INFORMATION_SET_TRANSPORT_SCHEMA_SHA256,
        "logical_operators": [
            operator.value for operator in INFORMATION_SET_TRANSPORT_PLAN.operators
        ],
        "relations": _INFORMATION_SET_TRANSPORT_SEMANTICS["input_relations"],
        "operations": list(_INFORMATION_SET_TRANSPORT_SEMANTICS["operations"]),
        "authority_boundary": _INFORMATION_SET_TRANSPORT_SEMANTICS[
            "authority_boundary"
        ],
    }


def _sqlite_program_sha256(rows: Sequence[Sequence[Any]]) -> str:
    material = [list(row[:-1]) for row in rows]
    payload = json.dumps(
        material,
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _validate_transport_inputs(
    *,
    world_weights: Sequence[float],
    leaf_count: int,
    edge_world_index: Sequence[int],
    edge_leaf_index: Sequence[int],
    edge_chance: Sequence[float],
) -> None:
    if isinstance(leaf_count, bool) or not isinstance(leaf_count, int) or leaf_count <= 0:
        raise SQLTransportError("leaf_count must be a positive integer")
    if not world_weights:
        raise SQLTransportError("transport requires at least one world")
    for weight in world_weights:
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise SQLTransportError("world weights must be positive and finite")

    edge_count = len(edge_world_index)
    if edge_count <= 0:
        raise SQLTransportError("transport requires at least one chance edge")
    if len(edge_leaf_index) != edge_count or len(edge_chance) != edge_count:
        raise SQLTransportError("chance-edge arrays must have equal length")

    world_count = len(world_weights)
    observed_leaves: set[int] = set()
    for world_index, leaf_index, chance in zip(
        edge_world_index,
        edge_leaf_index,
        edge_chance,
        strict=True,
    ):
        if (
            isinstance(world_index, bool)
            or not isinstance(world_index, int)
            or not 0 <= world_index < world_count
        ):
            raise SQLTransportError("chance edge references an unknown world")
        if (
            isinstance(leaf_index, bool)
            or not isinstance(leaf_index, int)
            or not 0 <= leaf_index < leaf_count
        ):
            raise SQLTransportError("chance edge references an unknown leaf")
        if (
            isinstance(chance, bool)
            or not isinstance(chance, (int, float))
            or not math.isfinite(float(chance))
            or float(chance) <= 0.0
        ):
            raise SQLTransportError("chance probabilities must be positive and finite")
        observed_leaves.add(leaf_index)

    if observed_leaves != set(range(leaf_count)):
        raise SQLTransportError("every leaf must receive positive chance mass")


def transport_information_set_mass_sql(
    *,
    world_weights: Sequence[float],
    leaf_count: int,
    edge_world_index: Sequence[int],
    edge_leaf_index: Sequence[int],
    edge_chance: Sequence[float],
) -> SQLTransportedMass:
    """Execute posterior mass transport through the packaged relational query."""

    _validate_transport_inputs(
        world_weights=world_weights,
        leaf_count=leaf_count,
        edge_world_index=edge_world_index,
        edge_leaf_index=edge_leaf_index,
        edge_chance=edge_chance,
    )

    world_count = len(world_weights)
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(_INFORMATION_SET_TRANSPORT_SCHEMA)
        connection.executemany(
            "INSERT INTO worlds(world_index, weight) VALUES (?, ?)",
            [
                (index, float(weight))
                for index, weight in enumerate(world_weights)
            ],
        )
        connection.executemany(
            "INSERT INTO leaves(leaf_index) VALUES (?)",
            [(index,) for index in range(leaf_count)],
        )
        connection.executemany(
            "INSERT INTO edges(leaf_index, world_index, chance) VALUES (?, ?, ?)",
            [
                (leaf_index, world_index, float(chance))
                for world_index, leaf_index, chance in zip(
                    edge_world_index,
                    edge_leaf_index,
                    edge_chance,
                    strict=True,
                )
            ],
        )

        query_plan_rows = connection.execute(
            "EXPLAIN QUERY PLAN " + INFORMATION_SET_TRANSPORT_SQL
        ).fetchall()
        program_rows = connection.execute(
            "EXPLAIN " + INFORMATION_SET_TRANSPORT_SQL
        ).fetchall()
        rows = connection.execute(INFORMATION_SET_TRANSPORT_SQL).fetchall()
    except sqlite3.DatabaseError as exc:
        raise SQLTransportError(str(exc)) from exc
    finally:
        connection.close()

    expected_rows = leaf_count * world_count
    if len(rows) != expected_rows:
        raise SQLTransportError(
            "relational transport did not materialize the full leaf/world matrix"
        )

    normalized: list[float | None] = [None] * world_count
    leaf_mass: list[float | None] = [None] * leaf_count
    leaf_world_mass = [
        [0.0 for _ in range(world_count)]
        for _ in range(leaf_count)
    ]
    leaf_world_weights = [
        [0.0 for _ in range(world_count)]
        for _ in range(leaf_count)
    ]

    for raw_row in rows:
        (
            raw_leaf_index,
            raw_world_index,
            raw_world_mass,
            raw_leaf_mass,
            raw_conditional,
            raw_normalized,
        ) = raw_row
        leaf_index = int(raw_leaf_index)
        world_index = int(raw_world_index)
        world_mass = float(raw_world_mass)
        mass = float(raw_leaf_mass)
        conditional = float(raw_conditional)
        normalized_weight = float(raw_normalized)

        for value in (world_mass, mass, conditional, normalized_weight):
            if not math.isfinite(value) or value < 0.0:
                raise SQLTransportError("relational transport returned invalid mass")

        prior = normalized[world_index]
        if prior is not None and abs(prior - normalized_weight) > 1e-12:
            raise SQLTransportError(
                "normalized world weight changed across relational leaves"
            )
        normalized[world_index] = normalized_weight

        prior_mass = leaf_mass[leaf_index]
        if prior_mass is not None and abs(prior_mass - mass) > 1e-12:
            raise SQLTransportError("leaf mass changed across relational rows")
        leaf_mass[leaf_index] = mass
        leaf_world_mass[leaf_index][world_index] = world_mass
        leaf_world_weights[leaf_index][world_index] = conditional

    if any(value is None for value in normalized) or any(
        value is None for value in leaf_mass
    ):
        raise SQLTransportError("relational transport returned incomplete cardinality")

    normalized_values = tuple(float(value) for value in normalized if value is not None)
    leaf_mass_values = tuple(float(value) for value in leaf_mass if value is not None)
    if abs(math.fsum(normalized_values) - 1.0) > 1e-10:
        raise SQLTransportError("normalized world weights do not sum to one")

    for index in range(leaf_count):
        if abs(
            math.fsum(leaf_world_mass[index]) - leaf_mass_values[index]
        ) > 1e-10:
            raise SQLTransportError("leaf/world mass does not sum to leaf mass")
        if abs(math.fsum(leaf_world_weights[index]) - 1.0) > 1e-10:
            raise SQLTransportError("conditional leaf posterior does not sum to one")

    return SQLTransportedMass(
        normalized_world_weights=normalized_values,
        leaf_mass=leaf_mass_values,
        leaf_world_mass=tuple(tuple(row) for row in leaf_world_mass),
        leaf_world_weights=tuple(tuple(row) for row in leaf_world_weights),
        sqlite_version=sqlite3.sqlite_version,
        sqlite_query_plan=tuple(str(row[3]) for row in query_plan_rows),
        sqlite_program_sha256=_sqlite_program_sha256(program_rows),
    )
