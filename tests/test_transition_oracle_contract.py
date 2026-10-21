"""Fail-closed structural tests for the shared transition-oracle boundary."""

from __future__ import annotations

import pytest

from azelficoast.core.transition import validate_transition_oracle


class OracleContractError(ValueError):
    """Caller-owned validation exception."""


def _valid_oracle() -> dict[str, object]:
    return {
        "worlds": [{"world_id": "w1", "hidden": {"item": "x"}, "weight": 1.0}],
        "legal_actions": ["move"],
        "transitions": [
            {
                "world_id": "w1",
                "action": "move",
                "outcomes": [{"probability": 1.0}],
            }
        ],
    }


@pytest.mark.parametrize(
    ("section", "invalid", "message"),
    [
        ("worlds", [None], "every world must be an object"),
        ("worlds", [[("world_id", "w1")]], "every world must be an object"),
        (
            "worlds",
            [{"world_id": None, "hidden": {}, "weight": 1.0}],
            "every world must contain a non-empty world_id",
        ),
        (
            "worlds",
            [{"world_id": 7, "hidden": {}, "weight": 1.0}],
            "every world must contain a non-empty world_id",
        ),
        ("legal_actions", [None], "root actions must be non-empty strings"),
        ("legal_actions", [7], "root actions must be non-empty strings"),
        (
            "transitions",
            [{"world_id": 7, "action": "move", "outcomes": [{"probability": 1.0}]}],
            "transition world_id must be a non-empty string",
        ),
        (
            "transitions",
            [{"world_id": "w1", "action": 7, "outcomes": [{"probability": 1.0}]}],
            "transition action must be a non-empty string",
        ),
    ],
)
def test_malformed_oracle_is_rejected_by_caller_error(
    section: str, invalid: object, message: str
) -> None:
    oracle = _valid_oracle()
    oracle[section] = invalid

    with pytest.raises(OracleContractError, match=message):
        validate_transition_oracle(
            oracle,
            error_type=OracleContractError,
            expected_schema=None,
            expected_schema_version=None,
        )


def test_valid_generic_oracle_keeps_exact_ids() -> None:
    worlds, actions, transitions, candidates = validate_transition_oracle(
        _valid_oracle(),
        error_type=OracleContractError,
        expected_schema=None,
        expected_schema_version=None,
    )

    assert worlds[0]["world_id"] == "w1"
    assert actions == ["move"]
    assert list(transitions) == [("w1", "move")]
    assert candidates == ["item"]
