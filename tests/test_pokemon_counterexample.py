from __future__ import annotations

from fractions import Fraction

from azelficoast.pokemon_counterexample import (
    FURRET_SCARF_PAYOFF,
    FURRET_SPECS_PAYOFF,
    GARDEVOIR_SCARF_SPEED,
    GARDEVOIR_SPEED,
    IRON_HEAD,
    JIRACHI_CURRENT_HP,
    JIRACHI_SPEED,
    KNOCK_OFF_TO_GARDEVOIR,
    MOONBLAST_TO_JIRACHI,
    WORLD_SCARF,
    WORLD_SPECS,
    _fixture,
    run_pokemon_counterexample,
    validate_mechanics,
)
from azelficoast.search.imperfect_information import DeterminizationPolicy, PublicBeliefPolicy


def test_randbats_speed_fork_is_real() -> None:
    assert GARDEVOIR_SPEED == 180
    assert JIRACHI_SPEED == 206
    assert GARDEVOIR_SCARF_SPEED == 270
    assert GARDEVOIR_SPEED < JIRACHI_SPEED < GARDEVOIR_SCARF_SPEED


def test_damage_thresholds_force_opposite_iron_head_outcomes() -> None:
    assert (min(IRON_HEAD), max(IRON_HEAD)) == (185, 218)
    assert (min(MOONBLAST_TO_JIRACHI), max(MOONBLAST_TO_JIRACHI)) == (53, 63)
    assert JIRACHI_CURRENT_HP == 29

    # Specs Gardevoir is slower, so Jirachi's Iron Head removes the 83-HP threat.
    assert min(IRON_HEAD) >= 83
    # Scarf Gardevoir is faster, so even the minimum Moonblast removes Jirachi first.
    assert min(MOONBLAST_TO_JIRACHI) >= JIRACHI_CURRENT_HP


def test_frisk_switch_line_is_roll_enumerated_not_hand_assigned() -> None:
    assert (min(KNOCK_OFF_TO_GARDEVOIR), max(KNOCK_OFF_TO_GARDEVOIR)) == (82, 97)
    assert FURRET_SCARF_PAYOFF == Fraction(897, 2048)
    assert FURRET_SPECS_PAYOFF == Fraction(-1, 31)
    assert float((FURRET_SCARF_PAYOFF + FURRET_SPECS_PAYOFF) / 2) > 0


def test_hidden_item_treatment_exposes_strategy_fusion() -> None:
    fixture = _fixture()
    determinization = DeterminizationPolicy().solve(fixture)
    belief = PublicBeliefPolicy().solve(fixture)

    assert determinization.action == "/choose move protect"
    assert determinization.action_values["/choose move protect"] == 0.5

    assert belief.action == "/choose switch Furret"
    assert belief.action_values["/choose move protect"] == 0
    assert belief.action_values["/choose switch Furret"] > 0


def test_known_item_controls_make_both_solvers_agree() -> None:
    for world in (WORLD_SCARF, WORLD_SPECS):
        fixture = _fixture(item_known=world)
        determinization = DeterminizationPolicy().solve(fixture)
        belief = PublicBeliefPolicy().solve(fixture)

        assert determinization.action == belief.action
        assert determinization.action_values == belief.action_values


def test_mechanical_falsifiers_and_experiment_pass() -> None:
    assert validate_mechanics()["falsifiers_passed"] is True

    result = run_pokemon_counterexample()
    assert result["passed"] is True
    assert result["treatment"]["passed"] is True
    assert all(control["passed"] for control in result["known_item_controls"].values())
