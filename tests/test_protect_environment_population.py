from __future__ import annotations

import json

from azelficoast.protect_environment_population import _classify_rows


def _row(
    fixture_id: str,
    filename: str,
    *,
    legal_actions: int,
    protect: bool,
) -> dict[str, object]:
    return {
        "fixture_id": fixture_id,
        "battle_tag": f"battle-{fixture_id}",
        "filename": filename,
        "predictors": {
            "legal_action_count": legal_actions,
            "persistent_protect_present": protect,
        },
    }


def _source(path, *, weather=None, fields=None) -> None:
    path.write_text(
        json.dumps(
            {
                "fixture": {
                    "state": {
                        "weather": dict(weather or {}),
                        "fields": dict(fields or {}),
                    }
                }
            }
        ),
        encoding="utf-8",
    )


def test_environment_selection_enriches_protect_plus_high_branching(tmp_path) -> None:
    rows = []
    specs = [
        ("a", 12, True, {"RAINDANCE": 1}, {}),
        ("b", 11, True, {"SUNNYDAY": 1}, {}),
        ("c", 10, True, {}, {"GRASSY_TERRAIN": 1}),
        ("d", 4, False, {"SANDSTORM": 1}, {}),
        ("e", 3, False, {}, {"ELECTRIC_TERRAIN": 1}),
        ("f", 2, False, {}, {}),
    ]
    for fixture_id, actions, protect, weather, fields in specs:
        filename = f"{fixture_id}.json"
        _source(tmp_path / filename, weather=weather, fields=fields)
        rows.append(
            _row(
                fixture_id,
                filename,
                legal_actions=actions,
                protect=protect,
            )
        )

    selected, thresholds = _classify_rows(
        rows,
        all_sources=tmp_path,
        cap=6,
        minimum_weather_states=1,
        minimum_terrain_states=1,
    )

    assert thresholds["environment_legal_action_q75"] >= 10
    assert any(
        row["depth_stratum"] == "predictor-enriched"
        and row["environment"]["environment_present"]
        and row["predictors"]["persistent_protect_present"] is True
        for row in selected
    )
    assert any(
        row["depth_stratum"] == "representative-lower"
        and row["environment"]["environment_present"]
        and row["predictors"]["persistent_protect_present"] is False
        for row in selected
    )

    assert any(row["environment"]["weather_present"] for row in selected)
    assert any(row["environment"]["terrain_present"] for row in selected)


def test_environment_selection_fails_without_new_frontier(tmp_path) -> None:
    filename = "plain.json"
    _source(tmp_path / filename)
    rows = [_row("plain", filename, legal_actions=8, protect=True)]

    try:
        _classify_rows(rows, all_sources=tmp_path, cap=1)
    except ValueError as error:
        assert "no weather/terrain states" in str(error)
    else:
        raise AssertionError("plain-only cohort must fail closed")
