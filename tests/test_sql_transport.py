from __future__ import annotations

import hashlib
from importlib.resources import files
import json
import sqlite3

import pytest

from azelficoast.core.sql_transport import (
    INFORMATION_SET_TRANSPORT_SEMANTIC_ID,
    SQLTransportError,
    _transport_information_set_mass_rows_sql,
    describe_information_set_transport,
)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_sql_transport_normalizes_and_conditions_mass_relationally() -> None:
    result = _transport_information_set_mass_rows_sql(
        world_weights=(2.0, 3.0),
        leaf_count=2,
        edge_world_index=(0, 1, 1),
        edge_leaf_index=(0, 0, 1),
        edge_chance=(1.0, 0.5, 0.5),
    )

    assert result.normalized_world_weights == pytest.approx((0.4, 0.6))
    assert result.leaf_mass == pytest.approx((0.7, 0.3))
    assert result.leaf_world_mass[0] == pytest.approx((0.4, 0.3))
    assert result.leaf_world_mass[1] == pytest.approx((0.0, 0.3))
    assert result.leaf_world_weights[0] == pytest.approx((4.0 / 7.0, 3.0 / 7.0))
    assert result.leaf_world_weights[1] == pytest.approx((0.0, 1.0))
    assert result.sqlite_query_plan
    assert result.sqlite_program_sha256.startswith("sha256:")


def test_sql_transport_aggregates_duplicate_edge_mass_before_conditioning() -> None:
    result = _transport_information_set_mass_rows_sql(
        world_weights=(1.0, 1.0),
        leaf_count=2,
        edge_world_index=(0, 0, 1, 1),
        edge_leaf_index=(0, 0, 0, 1),
        edge_chance=(0.25, 0.75, 0.5, 0.5),
    )

    assert result.leaf_world_mass[0] == pytest.approx((0.5, 0.25))
    assert result.leaf_mass[0] == pytest.approx(0.75)
    assert result.leaf_world_weights[0] == pytest.approx((2.0 / 3.0, 1.0 / 3.0))


def test_sql_transport_description_binds_exact_packaged_sources() -> None:
    description = describe_information_set_transport()
    query_source = (
        files("azelficoast.queries")
        .joinpath("information_set_transport.sql")
        .read_text(encoding="utf-8")
        .strip()
    )
    schema_source = (
        files("azelficoast.queries")
        .joinpath("information_set_transport_schema.sql")
        .read_text(encoding="utf-8")
        .strip()
    )

    assert description["semantic_identity"] == INFORMATION_SET_TRANSPORT_SEMANTIC_ID
    assert description["query_source"] == (
        "azelficoast.queries/information_set_transport.sql"
    )
    assert description["schema_source"] == (
        "azelficoast.queries/information_set_transport_schema.sql"
    )
    assert description["query_sha256"] == _sha256_text(query_source)
    assert description["schema_sha256"] == _sha256_text(schema_source)
    assert description["logical_operators"] == [
        "scan",
        "project",
        "update_belief",
        "aggregate",
    ]
    assert description["authority_boundary"] == (
        "content-addressed compiled search topology supplied by caller"
    )

    semantic_material = {
        "schema": description["schema"],
        "schema_version": description["schema_version"],
        "authority_boundary": description["authority_boundary"],
        "query_sha256": description["query_sha256"],
        "schema_sha256": description["schema_sha256"],
        "input_relations": description["relations"],
        "operations": description["operations"],
        "output": description["output"],
    }
    expected_identity = "sha256:" + hashlib.sha256(
        json.dumps(
            semantic_material,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    assert description["semantic_identity"] == expected_identity


def test_transport_schema_rejects_unknown_incidence() -> None:
    schema_source = (
        files("azelficoast.queries")
        .joinpath("information_set_transport_schema.sql")
        .read_text(encoding="utf-8")
    )
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(schema_source)
        connection.execute("INSERT INTO worlds(world_index, weight) VALUES (0, 1.0)")
        connection.execute("INSERT INTO leaves(leaf_index) VALUES (0)")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO edges(leaf_index, world_index, chance) VALUES (1, 0, 1.0)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO edges(leaf_index, world_index, chance) VALUES (0, 1, 1.0)"
            )
    finally:
        connection.close()


def test_sql_transport_fails_closed_on_invalid_incidence() -> None:
    with pytest.raises(SQLTransportError, match="unknown leaf"):
        _transport_information_set_mass_rows_sql(
            world_weights=(1.0,),
            leaf_count=1,
            edge_world_index=(0,),
            edge_leaf_index=(1,),
            edge_chance=(1.0,),
        )

    with pytest.raises(SQLTransportError, match="unknown world"):
        _transport_information_set_mass_rows_sql(
            world_weights=(1.0,),
            leaf_count=1,
            edge_world_index=(1,),
            edge_leaf_index=(0,),
            edge_chance=(1.0,),
        )

    with pytest.raises(SQLTransportError, match="equal length"):
        _transport_information_set_mass_rows_sql(
            world_weights=(1.0,),
            leaf_count=1,
            edge_world_index=(0,),
            edge_leaf_index=(0, 0),
            edge_chance=(1.0,),
        )

    with pytest.raises(SQLTransportError, match="positive and finite"):
        _transport_information_set_mass_rows_sql(
            world_weights=(1.0,),
            leaf_count=1,
            edge_world_index=(0,),
            edge_leaf_index=(0,),
            edge_chance=(0.0,),
        )

    with pytest.raises(SQLTransportError, match="every leaf"):
        _transport_information_set_mass_rows_sql(
            world_weights=(1.0,),
            leaf_count=2,
            edge_world_index=(0,),
            edge_leaf_index=(0,),
            edge_chance=(1.0,),
        )
