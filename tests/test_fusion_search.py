from __future__ import annotations

import json

import pytest

from azelficoast.corpus import DecisionFixture
from azelficoast.fusion_search import FusionSearchError, freeze_selection


def _fixture(fixture_id: str) -> DecisionFixture:
    return DecisionFixture(
        fixture_id=fixture_id,
        state={
            "turn": 8,
            "player": "Azelficoast",
            "opponent": "Rival",
            "active": {
                "species": "Tinkaton",
                "tera_type": "Steel",
                "current_hp": 100,
            },
            "opponent_active": {
                "species": "Zapdos-Galar",
                "level": 77,
                "item": None,
            },
            "team": {"p1: Tinkaton": {"species": "Tinkaton"}},
            "legal_actions": [
                "/choose move protect",
                "/choose move gigatonhammer",
                "/choose switch Lapras",
            ],
        },
        protocol_prefix=(
            (
                ("", "player", "p1", "Azelficoast"),
                ("", "player", "p2", "Rival"),
                ("", "move", "p2a: Zapdos-Galar", "U-turn"),
            ),
        ),
        control_decisions=(),
    )


def _candidate(fixture_id: str) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "active_species": "Tinkaton",
        "opponent_species": "Zapdos-Galar",
        "locked_move": "U-turn",
        "revealed_moves": ["uturn"],
        "item_counts": {"Choice Band": 50, "Choice Scarf": 50},
        "item_weights": {"Choice Band": 0.5, "Choice Scarf": 0.5},
        "persistent_protect_actions": [{"action": "/choose move protect"}],
        "persistent_switches": [],
    }


def _world(
    *,
    item: str,
    count: int,
    own_speed: int,
    opponent_speed: int,
    damage_min: int,
    damage_max: int,
    ko_rolls: int,
) -> dict[str, object]:
    return {
        "item": item,
        "count": count,
        "incoming": {
            "own_speed": own_speed,
            "opponent_speed": opponent_speed,
            "damage_min": damage_min,
            "damage_max": damage_max,
            "ko_rolls": ko_rolls,
        },
    }


def _case(
    fixture_id: str,
    *,
    band: tuple[int, int, int, int],
    scarf: tuple[int, int, int, int],
) -> dict[str, object]:
    # Tuple is opponent_speed, damage_min, damage_max, ko_rolls.
    return {
        "fixture_id": fixture_id,
        "by_item": {
            "Choice Band": {
                "worlds": [
                    _world(
                        item="Choice Band",
                        count=50,
                        own_speed=200,
                        opponent_speed=band[0],
                        damage_min=band[1],
                        damage_max=band[2],
                        ko_rolls=band[3],
                    )
                ]
            },
            "Choice Scarf": {
                "worlds": [
                    _world(
                        item="Choice Scarf",
                        count=50,
                        own_speed=200,
                        opponent_speed=scarf[0],
                        damage_min=scarf[1],
                        damage_max=scarf[2],
                        ko_rolls=scarf[3],
                    )
                ]
            },
        },
    }


def _plan(top_k: int = 2) -> dict[str, object]:
    return {
        "schema": "azelficoast.large-margin-strategy-fusion-plan",
        "schema_version": 1,
        "source_artifact": {"artifact_id": 1},
        "showdown_commit": "pinned",
        "discovery": {
            "generator_rounds": 2048,
            "persistent_only": True,
            "top_k": top_k,
            "rank_order": ["mechanics only"],
        },
        "exact_treatment": {
            "root_chance_samples": 16,
            "continuation_chance_samples": 16,
            "chance_seed_family": 0,
            "large_margin_threshold": 0.02,
        },
    }


def test_freeze_selection_ranks_mechanics_before_policy_results(tmp_path) -> None:
    rich = "rich"
    poor = "poor"
    candidates = {
        "schema": "azelficoast.natural-fusion-candidates",
        "schema_version": 1,
        "persistent_only": True,
        "candidates": [_candidate(poor), _candidate(rich)],
    }
    mechanics = {
        "schema": "azelficoast.public-belief-speed-fork-mechanics",
        "schema_version": 1,
        "showdown_commit": "pinned",
        "cases": [
            _case(
                poor,
                band=(180, 40, 40, 0),
                scarf=(180, 42, 42, 0),
            ),
            _case(
                rich,
                band=(180, 60, 70, 16),
                scarf=(260, 20, 30, 0),
            ),
        ],
    }

    result = freeze_selection(
        plan=_plan(),
        candidates_document=candidates,
        mechanics_document=mechanics,
        fixtures=[_fixture(poor), _fixture(rich)],
        output_dir=tmp_path,
    )

    assert result["ranking_uses_policy_result"] is False
    assert [row["fixture_id"] for row in result["selected"]] == [rich, poor]

    [rich_row, poor_row] = result["selected"]
    assert rich_row["signals"]["separation_channel_count"] == 3
    assert poor_row["signals"]["separation_channel_count"] == 1

    source = json.loads((tmp_path / rich_row["filename"]).read_text())
    assert source["selection"]["ranking_uses_policy_result"] is False
    assert source["selection"]["selection_rank"] == 1
    assert source["plausible_items"] == ["Choice Band", "Choice Scarf"]


def test_freeze_selection_rejects_item_support_drift(tmp_path) -> None:
    fixture_id = "drift"
    candidate = _candidate(fixture_id)
    candidate["item_weights"] = {"Choice Band": 0.5, "Choice Specs": 0.5}

    with pytest.raises(FusionSearchError, match="item supports disagree"):
        freeze_selection(
            plan=_plan(top_k=1),
            candidates_document={
                "schema": "azelficoast.natural-fusion-candidates",
                "persistent_only": True,
                "candidates": [candidate],
            },
            mechanics_document={
                "schema": "azelficoast.public-belief-speed-fork-mechanics",
                "showdown_commit": "pinned",
                "cases": [
                    _case(
                        fixture_id,
                        band=(180, 60, 70, 16),
                        scarf=(260, 20, 30, 0),
                    )
                ],
            },
            fixtures=[_fixture(fixture_id)],
            output_dir=tmp_path,
        )
