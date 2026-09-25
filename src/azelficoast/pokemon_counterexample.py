"""Pokémon-shaped strategy-fusion counterexample grounded in Gen 9 randbats."""

from __future__ import annotations

import json
import math
from fractions import Fraction
from itertools import product
from typing import Any

from azelficoast.corpus import DecisionFixture
from azelficoast.search.imperfect_information import (
    GAME_KEY,
    DeterminizationPolicy,
    PublicBeliefPolicy,
)

# Current Pokémon Showdown Gen 9 randbats facts used by this experiment.
# Source: data/random-battles/gen9/sets.json and teams.ts.
LEVEL_JIRACHI = 80
LEVEL_GARDEVOIR = 83
LEVEL_FURRET = 94
EV = 85
IV = 31

BASE = {
    "jirachi": {"hp": 100, "atk": 100, "spd": 100, "spe": 100},
    "gardevoir": {"hp": 68, "def": 65, "spa": 125, "spe": 80},
    "furret": {"hp": 85, "atk": 76, "spd": 55, "spe": 90},
}

GARDEVOIR_CURRENT_HP = 83
JIRACHI_CURRENT_HP = 29

WORLD_SCARF = "choice-scarf"
WORLD_SPECS = "choice-specs"


def neutral_stat(base: int, level: int) -> int:
    return math.floor((2 * base + IV + EV // 4) * level / 100) + 5


def hp_stat(base: int, level: int) -> int:
    return math.floor((2 * base + IV + EV // 4 + 100) * level / 100) + 10


def damage_rolls(
    *,
    level: int,
    power: int,
    attack: int,
    defense: int,
    modifiers: tuple[Fraction, ...],
) -> tuple[int, ...]:
    """Exact 85..100 random rolls for the experiment's ordinary damage cases."""
    base_damage = (
        math.floor(
            math.floor((math.floor(2 * level / 5) + 2) * power * attack / defense)
            / 50
        )
        + 2
    )
    rolls = []
    for random_percent in range(85, 101):
        damage = base_damage
        for modifier in modifiers:
            damage = math.floor(damage * modifier)
        damage = math.floor(damage * random_percent / 100)
        rolls.append(damage)
    return tuple(rolls)


JIRACHI_SPEED = neutral_stat(BASE["jirachi"]["spe"], LEVEL_JIRACHI)
GARDEVOIR_SPEED = neutral_stat(BASE["gardevoir"]["spe"], LEVEL_GARDEVOIR)
GARDEVOIR_SCARF_SPEED = math.floor(GARDEVOIR_SPEED * 1.5)
FURRET_SPEED = neutral_stat(BASE["furret"]["spe"], LEVEL_FURRET)

JIRACHI_HP = hp_stat(BASE["jirachi"]["hp"], LEVEL_JIRACHI)
GARDEVOIR_HP = hp_stat(BASE["gardevoir"]["hp"], LEVEL_GARDEVOIR)
FURRET_HP = hp_stat(BASE["furret"]["hp"], LEVEL_FURRET)

IRON_HEAD = damage_rolls(
    level=LEVEL_JIRACHI,
    power=80,
    attack=neutral_stat(BASE["jirachi"]["atk"], LEVEL_JIRACHI),
    defense=neutral_stat(BASE["gardevoir"]["def"], LEVEL_GARDEVOIR),
    modifiers=(Fraction(3, 2), Fraction(2, 1)),
)

MOONBLAST_TO_JIRACHI = damage_rolls(
    level=LEVEL_GARDEVOIR,
    power=95,
    attack=neutral_stat(BASE["gardevoir"]["spa"], LEVEL_GARDEVOIR),
    defense=neutral_stat(BASE["jirachi"]["spd"], LEVEL_JIRACHI),
    modifiers=(Fraction(3, 2), Fraction(1, 2)),
)

MOONBLAST_TO_FURRET = damage_rolls(
    level=LEVEL_GARDEVOIR,
    power=95,
    attack=neutral_stat(BASE["gardevoir"]["spa"], LEVEL_GARDEVOIR),
    defense=neutral_stat(BASE["furret"]["spd"], LEVEL_FURRET),
    modifiers=(Fraction(3, 2),),
)

SPECS_MOONBLAST_TO_FURRET = damage_rolls(
    level=LEVEL_GARDEVOIR,
    power=95,
    attack=neutral_stat(BASE["gardevoir"]["spa"], LEVEL_GARDEVOIR),
    defense=neutral_stat(BASE["furret"]["spd"], LEVEL_FURRET),
    modifiers=(Fraction(3, 2), Fraction(3, 2)),
)

KNOCK_OFF_TO_GARDEVOIR = damage_rolls(
    level=LEVEL_FURRET,
    power=65,
    attack=neutral_stat(BASE["furret"]["atk"], LEVEL_FURRET),
    defense=neutral_stat(BASE["gardevoir"]["def"], LEVEL_GARDEVOIR),
    # Knock Off's 1.5x held-item boost. Dark is net neutral into Psychic/Fairy.
    modifiers=(Fraction(3, 2),),
)


def furret_switch_line_payoff(world: str) -> Fraction:
    """Expected two-turn material result after Frisk reveals the Choice item.

    Payoff is +1 if Gardevoir is removed, -1 if Furret is removed first,
    and 0 if neither happens by the bounded horizon.
    """
    total = 0
    outcomes = 0

    if world == WORLD_SCARF:
        # Turn 1: switch Furret into locked Moonblast and Frisk the Scarf.
        # Turn 2: Scarf Gardevoir attacks first, then Furret uses Knock Off if alive.
        for first_hit, second_hit, knock_off in product(
            MOONBLAST_TO_FURRET,
            MOONBLAST_TO_FURRET,
            KNOCK_OFF_TO_GARDEVOIR,
        ):
            outcomes += 1
            remaining = FURRET_HP - first_hit
            if second_hit >= remaining:
                total -= 1
            elif knock_off >= GARDEVOIR_CURRENT_HP:
                total += 1
    elif world == WORLD_SPECS:
        # Turn 1: Furret survives one Specs-boosted Moonblast and Frisk reveals Specs.
        # Turn 2: Furret is faster, uses Knock Off, and removes Specs before the reply.
        for first_hit, knock_off in product(
            SPECS_MOONBLAST_TO_FURRET,
            KNOCK_OFF_TO_GARDEVOIR,
        ):
            remaining = FURRET_HP - first_hit
            if knock_off >= GARDEVOIR_CURRENT_HP:
                outcomes += 1
                total += 1
                continue
            for reply in MOONBLAST_TO_FURRET:
                outcomes += 1
                if reply >= remaining:
                    total -= 1
    else:
        raise ValueError(f"unknown world: {world}")

    return Fraction(total, outcomes)


FURRET_SCARF_PAYOFF = furret_switch_line_payoff(WORLD_SCARF)
FURRET_SPECS_PAYOFF = furret_switch_line_payoff(WORLD_SPECS)


def _fixture(*, item_known: str | None = None) -> DecisionFixture:
    if item_known not in (None, WORLD_SCARF, WORLD_SPECS):
        raise ValueError(f"invalid known item: {item_known}")

    worlds = (
        [{"name": item_known, "weight": 1.0}]
        if item_known
        else [
            {"name": WORLD_SCARF, "weight": 0.5},
            {"name": WORLD_SPECS, "weight": 0.5},
        ]
    )

    iron_head_payoffs = {
        WORLD_SCARF: -1.0,
        WORLD_SPECS: 1.0,
    }
    switch_payoffs = {
        WORLD_SCARF: float(FURRET_SCARF_PAYOFF),
        WORLD_SPECS: float(FURRET_SPECS_PAYOFF),
    }

    if item_known:
        iron_head_payoffs = {item_known: iron_head_payoffs[item_known]}
        switch_payoffs = {item_known: switch_payoffs[item_known]}

    observations = {
        world["name"]: (
            f"moonblast;item={world['name']}"
            if item_known
            else "moonblast;item-unrevealed"
        )
        for world in worlds
    }

    actions = {
        "/choose move ironhead": {
            "terminal_payoffs": iron_head_payoffs,
        },
        "/choose move protect": {
            "observations": observations,
            "continuations": {
                "iron-head": iron_head_payoffs,
                "switch-furret": {
                    world["name"]: 0.0 for world in worlds
                },
            },
        },
        "/choose switch Furret": {
            "terminal_payoffs": switch_payoffs,
        },
    }

    state: dict[str, Any] = {
        "legal_actions": sorted(actions),
        "pokemon_counterexample": {
            "active": {
                "species": "Jirachi",
                "level": LEVEL_JIRACHI,
                "current_hp": JIRACHI_CURRENT_HP,
                "speed": JIRACHI_SPEED,
                "moves": ["Iron Head", "Protect"],
            },
            "opponent": {
                "species": "Gardevoir",
                "level": LEVEL_GARDEVOIR,
                "current_hp": GARDEVOIR_CURRENT_HP,
                "base_speed": GARDEVOIR_SPEED,
                "scarf_speed": GARDEVOIR_SCARF_SPEED,
                "revealed_move": "Moonblast",
                "hidden_item_worlds": [world["name"] for world in worlds],
            },
            "frisk_switch": {
                "species": "Furret",
                "level": LEVEL_FURRET,
                "speed": FURRET_SPEED,
                "ability": "Frisk",
                "follow_up": "Knock Off",
            },
        },
        GAME_KEY: {
            "worlds": worlds,
            "actions": actions,
        },
    }
    return DecisionFixture(
        fixture_id=f"pokemon-counterexample-{item_known or 'hidden'}",
        state=state,
        protocol_prefix=(),
        control_decisions=(),
    )


def validate_mechanics() -> dict[str, Any]:
    """Mechanical falsifiers for the bounded tactical position."""
    return {
        "jirachi_speed": JIRACHI_SPEED,
        "gardevoir_speed": GARDEVOIR_SPEED,
        "gardevoir_scarf_speed": GARDEVOIR_SCARF_SPEED,
        "furret_speed": FURRET_SPEED,
        "iron_head_rolls": [min(IRON_HEAD), max(IRON_HEAD)],
        "moonblast_to_jirachi_rolls": [
            min(MOONBLAST_TO_JIRACHI),
            max(MOONBLAST_TO_JIRACHI),
        ],
        "knock_off_to_gardevoir_rolls": [
            min(KNOCK_OFF_TO_GARDEVOIR),
            max(KNOCK_OFF_TO_GARDEVOIR),
        ],
        "furret_scarf_payoff": float(FURRET_SCARF_PAYOFF),
        "furret_specs_payoff": float(FURRET_SPECS_PAYOFF),
        "falsifiers_passed": (
            GARDEVOIR_SPEED < JIRACHI_SPEED < GARDEVOIR_SCARF_SPEED
            and min(IRON_HEAD) >= GARDEVOIR_CURRENT_HP
            and min(MOONBLAST_TO_JIRACHI) >= JIRACHI_CURRENT_HP
            and min(SPECS_MOONBLAST_TO_FURRET) < FURRET_HP
        ),
    }


def run_pokemon_counterexample() -> dict[str, Any]:
    determinization = DeterminizationPolicy()
    public_belief = PublicBeliefPolicy()

    hidden = _fixture()
    hidden_det = determinization.solve(hidden)
    hidden_belief = public_belief.solve(hidden)

    known_controls = {}
    controls_passed = True
    for world in (WORLD_SCARF, WORLD_SPECS):
        known = _fixture(item_known=world)
        det = determinization.solve(known)
        belief = public_belief.solve(known)
        same = (
            det.action == belief.action
            and det.action_values == belief.action_values
        )
        controls_passed = controls_passed and same
        known_controls[world] = {
            "determinization": {
                "action": det.action,
                "value": det.value,
                "action_values": dict(det.action_values),
            },
            "public_belief": {
                "action": belief.action,
                "value": belief.value,
                "action_values": dict(belief.action_values),
            },
            "passed": same,
        }

    treatment_passed = (
        hidden_det.action == "/choose move protect"
        and hidden_belief.action == "/choose switch Furret"
        and hidden_det.action_values["/choose move protect"] == 0.5
        and hidden_belief.action_values["/choose move protect"] == 0.0
        and hidden_belief.action_values["/choose switch Furret"] > 0.0
    )

    mechanics = validate_mechanics()
    return {
        "schema": "azelficoast.pokemon-counterexample",
        "schema_version": 1,
        "hypothesis": (
            "perfect-information determinization overvalues Protect because it "
            "conditions the next move on an unrevealed Choice item"
        ),
        "source_facts": {
            "format": "gen9randombattle",
            "jirachi_moves": ["Iron Head", "Protect"],
            "furret_ability": "Frisk",
            "gardevoir_item_worlds": [WORLD_SCARF, WORLD_SPECS],
            "gardevoir_prior": {WORLD_SCARF: 0.5, WORLD_SPECS: 0.5},
        },
        "mechanics": mechanics,
        "treatment": {
            "determinization": {
                "action": hidden_det.action,
                "value": hidden_det.value,
                "action_values": dict(hidden_det.action_values),
                "continuations": dict(hidden_det.continuation_by_action),
            },
            "public_belief": {
                "action": hidden_belief.action,
                "value": hidden_belief.value,
                "action_values": dict(hidden_belief.action_values),
                "continuations": dict(hidden_belief.continuation_by_action),
            },
            "passed": treatment_passed,
        },
        "known_item_controls": known_controls,
        "passed": mechanics["falsifiers_passed"] and treatment_passed and controls_passed,
    }


def main() -> int:
    result = run_pokemon_counterexample()
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
