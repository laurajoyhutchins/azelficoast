from __future__ import annotations

from azelficoast.corpus import DecisionFixture
from azelficoast.natural_disagreements import UnsupportedWorldSample, mine_candidates


def _fixture() -> DecisionFixture:
    state = {
        "turn": 2,
        "player": "azelficoast",
        "opponent": "opponent",
        "active": {
            "species": "Jirachi",
            "level": 80,
            "transformed": False,
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
            "species": "gardevoir",
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
    assert result["persistent_candidate_count"] == 1
    assert candidate["persistent_protect_actions"] == [
        {
            "action": "/choose move protect",
            "kind": "protect",
            "observation": "blocked-no-item-reveal",
        }
    ]


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

    # 206 * 1.5 is faster than both Gardevoir worlds, so the speed fork disappears.
    # Protect still preserves the hidden item information set.
    assert result["candidate_count"] == 1
    assert result["persistent_candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["current_speed_fork"] is False
    assert candidate["persistent_protect_actions"][0]["action"] == "/choose move protect"




def test_unsupported_generator_world_is_counted_not_fatal(monkeypatch) -> None:
    def unsupported(**_kwargs):
        raise UnsupportedWorldSample("transformed Ditto moves are not generator moves")

    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        unsupported,
    )

    result = mine_candidates(
        [_fixture()],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 0
    assert result["skipped"]["world-sample-unsupported"] == 1


def test_transformed_active_state_is_excluded_until_copied_stats_are_modeled(
    monkeypatch,
) -> None:
    fixture = _fixture()
    state = dict(fixture.state)
    active = dict(state["active"])
    active["transformed"] = True
    state["active"] = active
    transformed = DecisionFixture(
        fixture_id="transformed",
        state=state,
        protocol_prefix=fixture.protocol_prefix,
        control_decisions=fixture.control_decisions,
    )
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [transformed],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 0
    assert result["skipped"]["active-transformed"] == 1



def _persistent_fixture(*, immune: bool = True) -> DecisionFixture:
    fixture = _fixture()
    state = dict(fixture.state)

    active = dict(state["active"])
    active["stats"] = {"spe": 300}
    state["active"] = active

    switch_species = "Umbreon" if immune else "Vaporeon"
    switch_types = ["DARK"] if immune else ["WATER"]
    state["available_switches"] = [switch_species.lower()]
    state["legal_actions"] = [
        "/choose move ironhead",
        f"/choose switch {switch_species}",
    ]
    state["team"] = {
        f"p1: {switch_species}": {
            "species": switch_species.lower(),
            "level": 80,
            "transformed": False,
            "fainted": False,
            "status": None,
            "item": "leftovers",
            "ability": "synchronize" if immune else "waterabsorb",
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
            "types": switch_types,
        }
    }

    prefix = list(list(message) for message in fixture.protocol_prefix[0])
    for message in prefix:
        if len(message) >= 4 and message[1] == "move":
            message[3] = "Psychic"
    return DecisionFixture(
        fixture_id="persistent" if immune else "nonimmune",
        state=state,
        protocol_prefix=(tuple(tuple(field for field in message) for message in prefix),),
        control_decisions=fixture.control_decisions,
    )


def test_mines_immunity_switch_that_preserves_hidden_item_worlds(monkeypatch) -> None:
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [_persistent_fixture()],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 1
    assert result["persistent_candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["current_speed_fork"] is False
    assert candidate["persistent_switches"] == [
        {
            "action": "/choose switch Umbreon",
            "species": "umbreon",
            "speed": 206,
            "observation": "immune:psychic",
        }
    ]


def test_nonimmune_switch_does_not_create_persistent_information_set(monkeypatch) -> None:
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [_persistent_fixture(immune=False)],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 0
    assert result["persistent_candidate_count"] == 0
    assert result["skipped"]["no-speed-or-persistent-information-fork"] == 1



def test_protect_preserves_worlds_even_without_current_speed_fork(monkeypatch) -> None:
    fixture = _fixture()
    state = dict(fixture.state)
    active = dict(state["active"])
    active["stats"] = {"spe": 300}
    state["active"] = active
    no_speed_fork = DecisionFixture(
        fixture_id="protect-persistent",
        state=state,
        protocol_prefix=fixture.protocol_prefix,
        control_decisions=fixture.control_decisions,
    )
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [no_speed_fork],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 1
    assert result["persistent_candidate_count"] == 1
    candidate = result["candidates"][0]
    assert candidate["current_speed_fork"] is False
    assert candidate["persistent_protect_actions"][0]["action"] == "/choose move protect"



def test_persistent_only_skips_nonpersistent_state_before_sampling(monkeypatch) -> None:
    fixture = _persistent_fixture(immune=False)

    def should_not_sample(**_kwargs):
        raise AssertionError("generator sampling should not run")

    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        should_not_sample,
    )

    result = mine_candidates(
        [fixture],
        showdown_root="/tmp/showdown",
        rounds=100,
        persistent_only=True,
    )

    assert result["persistent_only"] is True
    assert result["candidate_count"] == 0
    assert result["sampled_world_queries"] == 0
    assert result["skipped"]["no-persistent-information-branch"] == 1



def test_unseen_fist_contact_move_is_not_protect_persistence(monkeypatch) -> None:
    fixture = _fixture()
    state = dict(fixture.state)
    opponent = dict(state["opponent_active"])
    opponent.update(
        {
            "species": "urshifu",
            "level": 74,
            "ability": "unseenfist",
        }
    )
    state["opponent_active"] = opponent

    prefix = [list(message) for message in fixture.protocol_prefix[0]]
    for message in prefix:
        if len(message) >= 4 and message[1] == "switch" and message[2].startswith("p2"):
            message[2] = "p2a: Urshifu"
            message[3] = "Urshifu, L74"
        if len(message) >= 4 and message[1] == "move" and message[2].startswith("p2"):
            message[2] = "p2a: Urshifu"
            message[3] = "Close Combat"
    unseen_fist = DecisionFixture(
        fixture_id="unseen-fist",
        state=state,
        protocol_prefix=(tuple(tuple(field for field in message) for message in prefix),),
        control_decisions=fixture.control_decisions,
    )

    def should_not_sample(**_kwargs):
        raise AssertionError("Unseen Fist Protect bypass should fail before sampling")

    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        should_not_sample,
    )

    result = mine_candidates(
        [unseen_fist],
        showdown_root="/tmp/showdown",
        rounds=100,
        persistent_only=True,
    )

    assert result["candidate_count"] == 0
    assert result["sampled_world_queries"] == 0
    assert result["skipped"]["no-persistent-information-branch"] == 1



def test_prior_equal_priority_move_order_eliminates_scarf_world(monkeypatch) -> None:
    fixture = _fixture()
    prefix = [
        [
            ("", "player", "p1", "azelficoast"),
            ("", "player", "p2", "opponent"),
            ("", "switch", "p1a: Jirachi", "Jirachi, L80", "200/200"),
            ("", "switch", "p2a: Gardevoir", "Gardevoir, L83", "200/200"),
            ("", "turn", "1"),
            ("", "move", "p1a: Jirachi", "Iron Head", "p2a: Gardevoir"),
            ("", "-damage", "p2a: Gardevoir", "150/200"),
            ("", "move", "p2a: Gardevoir", "Moonblast", "p1a: Jirachi"),
            ("", "-damage", "p1a: Jirachi", "120/200"),
            ("", "turn", "2"),
        ]
    ]
    ordered = DecisionFixture(
        fixture_id="prior-order",
        state=fixture.state,
        protocol_prefix=tuple(
            tuple(tuple(field for field in message) for message in batch)
            for batch in prefix
        ),
        control_decisions=fixture.control_decisions,
    )
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [ordered],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    # Jirachi's exact 206 Speed lies between base Gardevoir (180) and
    # Scarf Gardevoir (270). Seeing Jirachi move first on the prior turn
    # therefore makes the Scarf world inconsistent with public evidence.
    assert result["candidate_count"] == 0
    assert (
        result["skipped"]["choice-world-eliminated-by-prior-speed-order"]
        == 1
    )


def test_priority_mismatch_does_not_overinterpret_move_order(monkeypatch) -> None:
    fixture = _fixture()
    prefix = [
        [
            ("", "player", "p1", "azelficoast"),
            ("", "player", "p2", "opponent"),
            ("", "switch", "p1a: Jirachi", "Jirachi, L80", "200/200"),
            ("", "switch", "p2a: Gardevoir", "Gardevoir, L83", "200/200"),
            ("", "turn", "1"),
            ("", "move", "p1a: Jirachi", "Quick Attack", "p2a: Gardevoir"),
            ("", "-damage", "p2a: Gardevoir", "190/200"),
            ("", "move", "p2a: Gardevoir", "Moonblast", "p1a: Jirachi"),
            ("", "-damage", "p1a: Jirachi", "120/200"),
            ("", "turn", "2"),
        ]
    ]
    priority = DecisionFixture(
        fixture_id="priority-order",
        state=fixture.state,
        protocol_prefix=tuple(
            tuple(tuple(field for field in message) for message in batch)
            for batch in prefix
        ),
        control_decisions=fixture.control_decisions,
    )
    monkeypatch.setattr(
        "azelficoast.natural_disagreements._sample_worlds",
        lambda **_kwargs: _sample(),
    )

    result = mine_candidates(
        [priority],
        showdown_root="/tmp/showdown",
        rounds=100,
    )

    assert result["candidate_count"] == 1
    assert result["candidates"][0]["prior_speed_order_evidence"] is None
