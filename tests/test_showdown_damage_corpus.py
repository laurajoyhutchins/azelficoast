from __future__ import annotations

import copy

import pytest

from azelficoast.research.verification.showdown_damage_corpus import (
    PINNED_SHOWDOWN_COMMIT,
    ShowdownDamageCorpusError,
    analyze_document,
)


def _fixture(
    *,
    scenario: str,
    item: str = "",
    tera: str | None = None,
    type_mod: int = 1,
    burned: bool = False,
    attacker_level: int = 100,
    defender_level: int = 100,
    defender_stat_modifier: int = 4096,
) -> dict[str, object]:
    # Values here are generated from the local scalar kernel solely to test corpus
    # structure. Hosted evidence is generated independently by Showdown.
    from azelficoast.research.mechanics.gen9_damage import DamageContext, damage

    context = DamageContext(
        attacker_level=attacker_level,
        defender_level=defender_level,
        base_power=120 if burned else 80,
        category="Physical" if burned else "Special",
        move_id="closecombat" if burned else "aurasphere",
        move_type="Fighting",
        attacker_types=("Fighting", "Steel"),
        tera_type=tera,
        attacker_base_stat=110 if burned else 115,
        attacker_iv=31,
        attacker_ev=252,
        attacker_nature_percent=110,
        defender_base_stat=80 if burned else 95,
        defender_iv=31,
        defender_ev=252,
        defender_nature_percent=110,
        attacker_item=item,
        type_mod=type_mod,
        burned=burned,
        defender_stat_modifier=defender_stat_modifier,
    )
    raw = {
        "attacker_level": context.attacker_level,
        "defender_level": context.defender_level,
        "base_power": context.base_power,
        "category": context.category,
        "move_id": context.move_id,
        "move_type": context.move_type,
        "attacker_types": list(context.attacker_types),
        "tera_type": context.tera_type,
        "attacker_base_stat": context.attacker_base_stat,
        "attacker_iv": context.attacker_iv,
        "attacker_ev": context.attacker_ev,
        "attacker_nature_percent": context.attacker_nature_percent,
        "defender_base_stat": context.defender_base_stat,
        "defender_iv": context.defender_iv,
        "defender_ev": context.defender_ev,
        "defender_nature_percent": context.defender_nature_percent,
        "attacker_item": context.attacker_item,
        "type_mod": context.type_mod,
        "burned": context.burned,
        "defender_stat_modifier": context.defender_stat_modifier,
    }
    return {
        "scenario": scenario,
        "context": raw,
        "rolls": [
            {"roll": roll, "damage": damage(context, roll)}
            for roll in range(16)
        ],
    }


def _document() -> dict[str, object]:
    fixtures = [
        _fixture(scenario="type", type_mod=1),
        _fixture(scenario="item", item="Choice Specs", type_mod=0),
        _fixture(scenario="tera", tera="Fighting", type_mod=0),
        _fixture(scenario="burn", burned=True, type_mod=0),
        _fixture(
            scenario="unequal-levels",
            type_mod=0,
            attacker_level=78,
            defender_level=88,
        ),
        _fixture(
            scenario="defender-modifier",
            type_mod=0,
            defender_stat_modifier=3072,
        ),
    ]
    return {
        "schema": "azelficoast.showdown-gen9-damage-fixtures",
        "schema_version": 1,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "scenario_count": len(fixtures),
        "fixtures": fixtures,
    }


def test_damage_corpus_requires_exact_roll_agreement_and_negative_controls() -> None:
    result = analyze_document(_document())
    assert result["passed"] is True
    assert result["roll_case_count"] == 96
    assert result["exact_case_count"] == 96
    assert result["type_negative_control_detected"] is True
    assert result["item_negative_control_detected"] is True
    assert result["tera_negative_control_detected"] is True
    assert result["burn_negative_control_detected"] is True
    assert result["unequal_level_negative_control_detected"] is True
    assert result["defender_stat_modifier_negative_control_detected"] is True


def test_damage_corpus_rejects_one_wrong_damage_roll() -> None:
    document = copy.deepcopy(_document())
    fixtures = document["fixtures"]
    assert isinstance(fixtures, list)
    rolls = fixtures[0]["rolls"]
    assert isinstance(rolls, list)
    rolls[7]["damage"] = int(rolls[7]["damage"]) + 1

    with pytest.raises(ShowdownDamageCorpusError, match="disagrees with Showdown"):
        analyze_document(document)


def test_damage_corpus_rejects_wrong_showdown_revision() -> None:
    document = _document()
    document["showdown_commit"] = "wrong"
    with pytest.raises(ShowdownDamageCorpusError, match="does not match"):
        analyze_document(document)
