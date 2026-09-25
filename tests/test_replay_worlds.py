from __future__ import annotations

from dataclasses import replace

import pytest

from azelficoast.research.verification.replay_worlds import (
    SHOWDOWN_COMMIT,
    DamageObservation,
    ReplayError,
    ReplayEvidence,
    build_replay_belief,
    condition_generator_prior_on_public_history,
    damage_rolls_for_item,
    extract_damage_observations,
    infer_item_posterior,
    load_world_sample,
    parse_protocol,
    replay_json_url,
)


def _turn_19_log(*, life_orb_recoil: bool = False) -> str:
    recoil = (
        "|-damage|p2a: Infernape|233/259|[from] item: Life Orb\n"
        if life_orb_recoil
        else ""
    )
    return f"""|turn|18
|switch|p2a: Infernape|Infernape, L82, F|259/259
|switch|p1a: Thundurus|Thundurus, L80, M|67/258
|move|p2a: Infernape|Close Combat|p1a: Thundurus
|-resisted|p1a: Thundurus
|-damage|p1a: Thundurus|0 fnt
|-unboost|p2a: Infernape|def|1
|-unboost|p2a: Infernape|spd|1
{recoil}|faint|p1a: Thundurus
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
        "showdown_commit": SHOWDOWN_COMMIT,
        "seed_family": "[i,i,i,i]",
        "generator_context": {
            "format": "gen9randombattle",
            "teamDetails": {},
            "isLead": False,
            "isDoubles": False,
        },
        "rounds": 8,
        "matched": 8,
        "item_counts": {
            "Choice Band": 2,
            "Choice Scarf": 1,
            "Life Orb": 5,
        },
    }


def _damage() -> DamageObservation:
    observations = extract_damage_observations(parse_protocol(_turn_19_log()))
    [observation] = [
        item
        for item in observations
        if item.turn == 19 and item.target_species == "Kingambit"
    ]
    return observation


def test_replay_json_url_strips_viewpoint_query() -> None:
    assert replay_json_url(
        "https://replay.pokemonshowdown.com/gen9randombattle-1-secret?p2"
    ) == "https://replay.pokemonshowdown.com/gen9randombattle-1-secret.json"


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


def test_complete_turn_without_life_orb_recoil_removes_life_orb() -> None:
    conditioned = condition_generator_prior_on_public_history(
        _sample(),
        parse_protocol(_turn_19_log()),
    )

    assert conditioned["item_counts"] == {
        "Choice Band": 2,
        "Choice Scarf": 1,
    }
    assert conditioned["updates"] == [
        {
            "turn": 18,
            "kind": "life-orb-recoil",
            "observed": False,
            "authority": "complete-public-turn",
            "remaining_items": ["Choice Band", "Choice Scarf"],
        }
    ]


def test_observed_life_orb_recoil_collapses_history_prior_to_life_orb() -> None:
    conditioned = condition_generator_prior_on_public_history(
        _sample(),
        parse_protocol(_turn_19_log(life_orb_recoil=True)),
    )

    assert conditioned["item_counts"] == {"Life Orb": 5}
    assert conditioned["updates"][0]["observed"] is True


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


def test_world_sample_must_be_bound_to_historical_showdown_revision(tmp_path) -> None:
    sample = _sample()
    sample["showdown_commit"] = "wrong"
    path = tmp_path / "sample.json"
    import json

    path.write_text(json.dumps(sample), encoding="utf-8")

    with pytest.raises(ReplayError, match="not bound"):
        load_world_sample(path)


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
    assert first["generator_prior"] == {
        "Choice Band": 0.25,
        "Choice Scarf": 0.125,
        "Life Orb": 0.625,
    }
    assert first["prior"] == {
        "Choice Band": pytest.approx(2 / 3),
        "Choice Scarf": pytest.approx(1 / 3),
    }
    assert first["posterior"] == {"Choice Scarf": 1.0}
    assert len(first["belief_sha256"]) == 64
