from __future__ import annotations

from dataclasses import replace

import pytest

from azelficoast.replay_worlds import (
    DamageObservation,
    ReplayError,
    ReplayEvidence,
    build_replay_belief,
    collect_observations,
    damage_rolls_for_item,
    extract_damage_observations,
    infer_item_posterior,
    parse_protocol,
    replay_json_url,
)


def _turn_19_log() -> str:
    return """|turn|18
|switch|p2a: Infernape|Infernape, L82, F|259/259
|switch|p1a: Kingambit|Kingambit, L74, F|239/270
|turn|19
|-terastallize|p1a: Kingambit|Flying
|move|p2a: Infernape|Close Combat|p1a: Kingambit
|-resisted|p1a: Kingambit
|-damage|p1a: Kingambit|185/270
|-unboost|p2a: Infernape|def|1
|-unboost|p2a: Infernape|spd|1
"""


def _sample() -> dict[str, object]:
    return {
        "schema": "azelficoast.showdown-world-sample",
        "schema_version": 1,
        "species": "infernape",
        "observed_moves": ["closecombat"],
        "seed_family": "[i,i,i,i]",
        "rounds": 8,
        "matched": 8,
        "item_counts": {
            "Choice Band": 2,
            "Choice Scarf": 1,
            "Life Orb": 5,
        },
    }


def _damage() -> DamageObservation:
    [observation] = extract_damage_observations(parse_protocol(_turn_19_log()))
    return observation


def test_replay_json_url_strips_viewpoint_query() -> None:
    assert replay_json_url(
        "https://replay.pokemonshowdown.com/gen9randombattle-1-secret?p2"
    ) == "https://replay.pokemonshowdown.com/gen9randombattle-1-secret.json"


def test_protocol_observations_follow_active_species() -> None:
    log = """|turn|1
|switch|p1a: Ape|Infernape, L82|100/100
|switch|p2a: Gambit|Kingambit, L74|100/100
|move|p1a: Ape|Close Combat|p2a: Gambit
|-damage|p2a: Gambit|79/100
|turn|2
|move|p1a: Ape|Flare Blitz|p2a: Gambit
|-enditem|p1a: Ape|Choice Scarf
"""
    observations = collect_observations(parse_protocol(log))

    infernape = observations[("p1", "Infernape")]
    assert infernape.level == 82
    assert infernape.moves == [(1, "Close Combat"), (2, "Flare Blitz")]
    assert infernape.item_events == [(2, "-enditem", "Choice Scarf")]

    kingambit = observations[("p2", "Kingambit")]
    assert kingambit.hp_events == [(1, "79/100")]


def test_extracts_exact_public_damage_and_tera_state() -> None:
    observation = _damage()

    assert observation.turn == 19
    assert observation.attacker_species == "Infernape"
    assert observation.attacker_level == 82
    assert observation.move == "Close Combat"
    assert observation.target_species == "Kingambit"
    assert observation.target_level == 74
    assert observation.target_tera_type == "Flying"
    assert observation.before_hp == 239
    assert observation.after_hp == 185
    assert observation.max_hp == 270
    assert observation.damage == 54


def test_item_damage_ranges_separate_the_replay_worlds() -> None:
    observation = _damage()

    assert (
        min(damage_rolls_for_item(observation, item="Choice Scarf")),
        max(damage_rolls_for_item(observation, item="Choice Scarf")),
    ) == (51, 61)
    assert (
        min(damage_rolls_for_item(observation, item="Choice Band")),
        max(damage_rolls_for_item(observation, item="Choice Band")),
    ) == (77, 91)
    assert (
        min(damage_rolls_for_item(observation, item="Life Orb")),
        max(damage_rolls_for_item(observation, item="Life Orb")),
    ) == (66, 79)


def test_observed_54_damage_collapses_sampled_item_belief_to_scarf() -> None:
    posterior = infer_item_posterior(_sample(), _damage())

    assert posterior["prior"] == {
        "Choice Band": 0.25,
        "Choice Scarf": 0.125,
        "Life Orb": 0.625,
    }
    assert posterior["posterior"] == {"Choice Scarf": 1.0}
    assert posterior["compatible"]["Choice Scarf"]["matching_rolls"] == 3


def test_alternative_damage_falsifier_selects_band_instead() -> None:
    observation = replace(_damage(), after_hp=159, damage=80)

    posterior = infer_item_posterior(_sample(), observation)

    assert posterior["posterior"] == {"Choice Band": 1.0}


def test_unknown_sampled_item_fails_closed() -> None:
    sample = _sample()
    sample["item_counts"] = {
        "Choice Scarf": 1,
        "Mystery Orb": 1,
    }

    with pytest.raises(ReplayError, match="no damage model"):
        infer_item_posterior(sample, _damage())


def test_same_public_evidence_produces_same_belief_digest() -> None:
    replay = ReplayEvidence(
        replay_id="gen9randombattle-test",
        format="[Gen 9] Random Battle",
        log=_turn_19_log(),
    )
    events = parse_protocol(replay.log)

    first = build_replay_belief(replay, events, _sample())
    second = build_replay_belief(replay, events, _sample())

    assert first == second
    assert first["posterior"] == {"Choice Scarf": 1.0}
    assert len(first["belief_sha256"]) == 64
