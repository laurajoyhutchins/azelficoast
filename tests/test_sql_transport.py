from __future__ import annotations

import pytest

from azelficoast.core.sql_transport import (
    INFORMATION_SET_TRANSPORT_SEMANTIC_ID,
    SQLTransportError,
    describe_information_set_transport,
    transport_information_set_mass_sql,
)


def test_sql_transport_normalizes_and_conditions_mass_relationally() -> None:
    result = transport_information_set_mass_sql(
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
    result = transport_information_set_mass_sql(
        world_weights=(1.0, 1.0),
        leaf_count=2,
        edge_world_index=(0, 0, 1, 1),
        edge_leaf_index=(0, 0, 0, 1),
        edge_chance=(0.25, 0.75, 0.5, 0.5),
    )

    assert result.leaf_world_mass[0] == pytest.approx((0.5, 0.25))
    assert result.leaf_mass[0] == pytest.approx(0.75)
    assert result.leaf_world_weights[0] == pytest.approx((2.0 / 3.0, 1.0 / 3.0))


def test_sql_transport_description_binds_packaged_relational_semantics() -> None:
    description = describe_information_set_transport()

    assert description["semantic_identity"] == INFORMATION_SET_TRANSPORT_SEMANTIC_ID
    assert description["query_source"] == (
        "azelficoast.queries/information_set_transport.sql"
    )
    assert description["schema_source"] == (
        "azelficoast.queries/information_set_transport_schema.sql"
    )
    assert description["logical_operators"] == [
        "scan",
        "project",
        "update_belief",
        "aggregate",
    ]
    assert description["authority_boundary"] == (
        "authorized leaf/world/chance incidence supplied by caller"
    )


def test_sql_transport_fails_closed_on_invalid_incidence() -> None:
    with pytest.raises(SQLTransportError, match="unknown leaf"):
        transport_information_set_mass_sql(
            world_weights=(1.0,),
            leaf_count=1,
            edge_world_index=(0,),
            edge_leaf_index=(1,),
            edge_chance=(1.0,),
        )

    with pytest.raises(SQLTransportError, match="every leaf"):
        transport_information_set_mass_sql(
            world_weights=(1.0,),
            leaf_count=2,
            edge_world_index=(0,),
            edge_leaf_index=(0,),
            edge_chance=(1.0,),
        )
