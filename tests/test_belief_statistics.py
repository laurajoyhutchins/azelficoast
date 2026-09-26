from __future__ import annotations

import copy

from azelficoast.belief.statistics import posterior_correlation_profile


def _member(
    species: str,
    *,
    item: str,
    ability: str,
    role: str,
    moves: list[str],
    was_lead: bool,
) -> dict[str, object]:
    return {
        "species": species,
        "level": 80,
        "gender": "N",
        "ability": ability,
        "item": item,
        "moves": moves,
        "tera_type": "Water",
        "role": role,
        "nature": "Timid",
        "evs": {},
        "ivs": {},
        "was_lead": was_lead,
    }


def _posterior(*, correlated: bool) -> dict[str, object]:
    if correlated:
        hidden_rows = [
            (
                _member(
                    "Common",
                    item="Leftovers",
                    ability="Pressure",
                    role="Wall",
                    moves=["Protect", "Recover"],
                    was_lead=True,
                ),
                _member(
                    "Alpha",
                    item="Choice Scarf",
                    ability="Swift Swim",
                    role="Fast",
                    moves=["Hydro Pump", "Ice Beam"],
                    was_lead=False,
                ),
            ),
            (
                _member(
                    "Common",
                    item="Leftovers",
                    ability="Pressure",
                    role="Wall",
                    moves=["Protect", "Recover"],
                    was_lead=True,
                ),
                _member(
                    "Beta",
                    item="Choice Specs",
                    ability="Levitate",
                    role="Breaker",
                    moves=["Shadow Ball", "Thunderbolt"],
                    was_lead=False,
                ),
            ),
        ]
    else:
        hidden_rows = [
            (
                _member(
                    "Common",
                    item="Leftovers",
                    ability="Pressure",
                    role="Wall",
                    moves=["Protect", "Recover"],
                    was_lead=True,
                ),
                _member(
                    "Alpha",
                    item="Choice Scarf",
                    ability="Swift Swim",
                    role="Fast",
                    moves=["Hydro Pump", "Ice Beam"],
                    was_lead=False,
                ),
            ),
            (
                _member(
                    "Common",
                    item="Leftovers",
                    ability="Pressure",
                    role="Wall",
                    moves=["Protect", "Recover"],
                    was_lead=True,
                ),
                _member(
                    "Alpha",
                    item="Choice Specs",
                    ability="Levitate",
                    role="Breaker",
                    moves=["Shadow Ball", "Thunderbolt"],
                    was_lead=False,
                ),
            ),
            (
                _member(
                    "Common",
                    item="Leftovers",
                    ability="Pressure",
                    role="Wall",
                    moves=["Protect", "Recover"],
                    was_lead=True,
                ),
                _member(
                    "Beta",
                    item="Choice Scarf",
                    ability="Swift Swim",
                    role="Fast",
                    moves=["Hydro Pump", "Ice Beam"],
                    was_lead=False,
                ),
            ),
            (
                _member(
                    "Common",
                    item="Leftovers",
                    ability="Pressure",
                    role="Wall",
                    moves=["Protect", "Recover"],
                    was_lead=True,
                ),
                _member(
                    "Beta",
                    item="Choice Specs",
                    ability="Levitate",
                    role="Breaker",
                    moves=["Shadow Ball", "Thunderbolt"],
                    was_lead=False,
                ),
            ),
        ]

    weight = 1.0 / len(hidden_rows)
    return {
        "schema": "azelficoast.joint-random-battle-posterior",
        "schema_version": 1,
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "support_status": "sufficient",
        "construction": {
            "kind": "full-team-generator-rejection-particles",
            "posterior_treatment": "generator_faithful_joint_empirical",
            "preserves_joint_team_set_correlations": True,
        },
        "worlds": [
            {
                "world_id": f"w{index}",
                "weight": weight,
                "hidden": {"team": [copy.deepcopy(row[0]), copy.deepcopy(row[1])]},
            }
            for index, row in enumerate(hidden_rows)
        ],
        "public_evidence": {
            "opponent_team_size": 2,
            "revealed": [{"species": "Common", "was_lead": True}],
        },
    }


def test_posterior_correlation_profile_separates_joint_regimes() -> None:
    correlated = posterior_correlation_profile(_posterior(correlated=True))
    independent = posterior_correlation_profile(_posterior(correlated=False))

    assert correlated.signature != independent.signature
    assert correlated.world_count == 2
    assert independent.world_count == 4
    assert correlated.team_size == independent.team_size == 2

    correlated_species_item = correlated.dependencies[0]
    independent_species_item = independent.dependencies[0]
    assert (
        correlated_species_item.total_variation_from_independence
        > independent_species_item.total_variation_from_independence
    )


def test_posterior_correlation_signature_ignores_world_order() -> None:
    document = _posterior(correlated=False)
    reversed_document = copy.deepcopy(document)
    reversed_document["worlds"] = list(reversed(reversed_document["worlds"]))

    assert posterior_correlation_profile(document).signature == (
        posterior_correlation_profile(reversed_document).signature
    )
