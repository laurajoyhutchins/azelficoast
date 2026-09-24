from __future__ import annotations

from azelficoast.corpus import DecisionFixture
from azelficoast.natural_disagreements import mine_candidates


def _fixture() -> DecisionFixture:
    state = {
        "turn": 2,
        "player": "azelficoast",
        "opponent": "opponent",
        "active": {
            "species": "Jirachi",
            "level": 80,
            "status": None,
            "item": "Leftovers",
            "ability": "Serene Grace",
            "stats": {"spe": 206},
            "boosts": {
                "atk": 0,
                "def": 0,
                "spa": 0,
                "spd": 0,
                "spe": 0,
                "accuracy": 0,
                "evasion": 0,
            },
        },
        "opponent_active": {
            "species": "Gardevoir",
            "level": 83,
            "status": None,
            "item": None,
            "ability": None,
            "boosts": {
                "atk": 0,
                "def": 0,
                "spa": 0,
                "spd": 0,
                "spe": 0,
                "accuracy": 0,
                "evasion": 0,
            },
        },
        "weather": {},
        "fields": {},
        "side_conditions": {},
        "opponent_side_conditions": {},
        "legal_actions": [
            "/choose move ironhead",
            "/choose move protect",
        ],
    }
    prefix = (
        (
            ("", "player", "p1", "azelficoast"),
            ("", "player", "p2", "opponent"),
            ("", "switch", "p1a: Jirachi", "Jirachi, L80", "200/200"),
            ("", "switch", "p2a: Gardevoir", "Gardevoir, L83", "200/200"),
            ("", "turn", "1"),
            ("", "move", "p2a: Gardevoir", "Moonblast", "p1a: Jirachi"),
            ("", "-damage", "p1a: Jirachi", "120/200"),
            ("", "turn", "2"),
        ),
    )
    return DecisionFixture(
        fixture_id="fixture",
        state=state,
        protocol_prefix=prefix,
        control_decisions=({"chosen_action": "/choose move protect"},),
    )


def _sample(*, extra_item: bool = False) -> dict[str, object]:
    item_counts = {
        "Choice Scarf": 40,
        "Choice Specs": 60,
    }
    if extra_item:
        item_counts["Leftovers"] = 10
    matched = sum(item_counts.values())
    return {
        "schema": "azelficoast.showdown-world-sample",
        "schema_version": 1,
        "species": "gardevoir",
        "observed_moves": ["moonblast"],
        "showdown_commit": "showdown-head",
        "seed_family": "[i,i,i,i]",
        "generator_context": {
            "format": "gen9randombattle",
            "teamDetails": {},
            "isLead": True,
            "isDoubles": False,
        },
        "rounds": matched,
        "matched": matched,
        "item_counts": item_counts,
        "variants": [
            {
                "ability": "Synchronize",
                "item": item,
                "level": 83,
                "moves": ["moonblast", "psychic", "shadowball", "trick"],
                "role": "Fast Attacker",
                "teraType": "Psychic",
                "count": count,
                "weight": count / matched,
            }
            for item, count in item_counts.items()
        ],
    }


def test_legacy_unknown_item_sentinel_is_treated_as_hidden(monkeypatch) -> None:
    fixture = _fixture()
    state = dict(fixture.state)
    opponent = dict(state["opponent_active"])
    opponent["item"] = "unknown_item"
    state["opponent_active"] = opponent
    legacy = DecisionFixture(
        fixture_id="legacy-sentinel",
        state=state,
        protocol_prefix=fixture.protocol_prefix,
        control_decisions=fixture.control_decisions,
    )
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [legacy],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 1


def test_mines_real_shape_choice_item_speed_fork(monkeypatch) -> None:
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [_fixture()],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["active_speed"] == 206
    assert candidate["opponent_base_speed"] == 180
    assert candidate["opponent_scarf_speed"] == 270
    assert candidate["locked_move"] == "Moonblast"
    assert candidate["item_weights"] == {
        "Choice Scarf": 0.4,
        "Choice Specs": 0.6,
    }


def test_rejects_support_with_unresolved_non_choice_item(monkeypatch) -> None:
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(extra_item=True),
    )

    result = mine_candidates(
        [_fixture()],
        showdown_root="/tmp/showdown",
        rounds=110,
    )

    assert result["candidate_count"] == 0
    assert result["skipped"]["not-choice-pair-after-public-evidence"] == 1


def test_absent_life_orb_recoil_removes_life_orb_before_pair_check(monkeypatch) -> None:
    sample = _sample()
    sample["item_counts"] = {
        "Choice Scarf": 40,
        "Choice Specs": 60,
        "Life Orb": 100,
    }
    sample["matched"] = 200
    sample["rounds"] = 200
    sample["variants"].append(
        {
            "ability": "Synchronize",
            "item": "Life Orb",
            "level": 83,
            "moves": ["moonblast", "psychic", "shadowball", "trick"],
            "role": "Fast Attacker",
            "teraType": "Psychic",
            "count": 100,
            "weight": 0.5,
        }
    )
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: sample,
    )

    result = mine_candidates(
        [_fixture()],
        showdown_root="/tmp/showdown",
        rounds=200,
    )

    assert result["candidate_count"] == 1
    assert set(result["candidates"][0]["item_counts"]) == {
        "Choice Scarf",
        "Choice Specs",
    }


def test_public_speed_boosts_are_modeled_instead_of_rejected(monkeypatch) -> None:
    fixture = _fixture()
    state = dict(fixture.state)
    active = dict(state["active"])
    active["boosts"] = dict(active["boosts"])
    active["boosts"]["spe"] = 1
    state["active"] = active
    boosted = DecisionFixture(
        fixture_id="boosted",
        state=state,
        protocol_prefix=fixture.protocol_prefix,
        control_decisions=fixture.control_decisions,
    )
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [boosted],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    # 206 * 1.5 is faster than both Gardevoir worlds, so the fork disappears.
    assert result["candidate_count"] == 0
    assert result["skipped"]["no-speed-order-fork"] == 1
