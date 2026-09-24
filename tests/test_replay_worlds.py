from __future__ import annotations

from azelficoast.replay_worlds import (
    collect_observations,
    parse_protocol,
    replay_json_url,
)


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
