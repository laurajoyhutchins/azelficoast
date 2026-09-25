from __future__ import annotations

import copy

import pytest

from azelficoast.research.verification.showdown_transition_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    ShowdownTransitionCorpusError,
    analyze_document,
)


def _document() -> dict[str, object]:
    fixtures: list[dict[str, object]] = []
    seeds = ([1, 2, 3, 4], [5, 6, 7, 8])
    for scenario in ("protect", "damage"):
        for seed_index, seed in enumerate(seeds):
            for item in ("Choice Scarf", "Choice Specs"):
                for bench_signature, bench_item in enumerate(("Leftovers", "Lum Berry")):
                    if scenario == "protect":
                        after_hp = 400
                        transition_log = (
                            "|move|p1a: Lapras|Protect|p1a: Lapras",
                            "|move|p2a: Mew|Aura Sphere|p1a: Lapras",
                            "|-activate|p1a: Lapras|move: Protect",
                        )
                    else:
                        after_hp = 310 if item == "Choice Scarf" else 265
                        transition_log = (
                            "|move|p2a: Mew|Aura Sphere|p1a: Lapras",
                            f"|-damage|p1a: Lapras|{after_hp}/400",
                        )
                    fixtures.append(
                        {
                            "scenario": scenario,
                            "seed_index": seed_index,
                            "seed": list(seed),
                            "opponent_item": item,
                            "opponent_move": "Aura Sphere",
                            "own_species": "Lapras",
                            "opponent_species": "Mew",
                            "bench_signature": bench_signature,
                            "bench_item": bench_item,
                            "before_hp": 400,
                            "after_hp": after_hp,
                            "hp_delta": after_hp - 400,
                            "transition_log": list(transition_log),
                        }
                    )
    return {
        "schema": "azelficoast.showdown-transition-fixtures",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "seed_count": 2,
        "item_count": 2,
        "bench_variant_count": 2,
        "fixture_count": len(fixtures),
        "fixtures": fixtures,
    }


def test_real_fixture_analyzer_collapses_only_irrelevant_dimensions() -> None:
    result = analyze_document(_document())

    assert result["passed"] is True
    assert result["fixture_count"] == 16
    assert result["protect"]["dependency_classes"] == 2
    assert result["protect"]["reduction_factor"] == 4.0
    assert result["damage"]["dependency_classes"] == 4
    assert result["damage"]["reduction_factor"] == 2.0
    assert result["missing_item_negative_control_detected"] is True


def test_real_fixture_analyzer_rejects_bench_sensitive_outcome() -> None:
    document = _document()
    mutated = copy.deepcopy(document)
    fixtures = mutated["fixtures"]
    assert isinstance(fixtures, list)
    victim = next(
        fixture
        for fixture in fixtures
        if fixture["scenario"] == "damage"
        and fixture["seed_index"] == 0
        and fixture["opponent_item"] == "Choice Scarf"
        and fixture["bench_signature"] == 1
    )
    victim["after_hp"] = 309
    victim["hp_delta"] = -91
    victim["transition_log"] = [
        "|move|p2a: Mew|Aura Sphere|p1a: Lapras",
        "|-damage|p1a: Lapras|309/400",
    ]

    with pytest.raises(ValueError, match="Showdown outcomes"):
        analyze_document(mutated)


def test_real_fixture_analyzer_rejects_wrong_showdown_revision() -> None:
    document = _document()
    document["showdown_commit"] = "wrong"

    with pytest.raises(
        ShowdownTransitionCorpusError,
        match="does not match the pinned oracle",
    ):
        analyze_document(document)


def test_real_fixture_analyzer_rejects_duplicate_matrix_cell() -> None:
    document = _document()
    fixtures = document["fixtures"]
    assert isinstance(fixtures, list)
    damage = [
        fixture
        for fixture in fixtures
        if fixture["scenario"] == "damage"
    ]
    damage[-1] = copy.deepcopy(damage[0])
    document["fixtures"] = [
        fixture
        for fixture in fixtures
        if fixture["scenario"] == "protect"
    ] + damage

    with pytest.raises(
        ShowdownTransitionCorpusError,
        match="duplicate fixture matrix cell",
    ):
        analyze_document(document)
