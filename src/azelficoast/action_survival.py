"""Analyze whether hidden worlds change survival of one public continuation."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.gen9_damage import (
    MOD_ONE,
    DamageContext,
    damage,
)

PINNED_SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc"


class ActionSurvivalError(ValueError):
    """Raised when a frozen survival witness is malformed."""


def _damage_context(
    continuation: Mapping[str, Any],
    item: str,
) -> DamageContext:
    attacker = continuation["attacker"]
    defender = continuation["defender"]
    move = continuation["move"]
    if not all(isinstance(value, Mapping) for value in (attacker, defender, move)):
        raise ActionSurvivalError("continuation lacks attacker, defender, or move")

    return DamageContext(
        attacker_level=int(attacker["level"]),
        defender_level=int(defender["level"]),
        base_power=int(move["base_power"]),
        category=str(move["category"]),  # type: ignore[arg-type]
        move_id=str(move["id"]),
        move_type=str(move["type"]),
        attacker_types=tuple(str(value) for value in attacker["types"]),
        tera_type=None,
        attacker_base_stat=int(attacker["base_spa"]),
        attacker_iv=int(attacker["iv"]),
        attacker_ev=int(attacker["ev"]),
        attacker_nature_percent=int(attacker["nature_percent"]),
        defender_base_stat=int(defender["base_spd"]),
        defender_iv=int(defender["iv"]),
        defender_ev=int(defender["ev"]),
        defender_nature_percent=int(defender["nature_percent"]),
        attacker_item=item,
        type_mod=int(move["type_mod"]),
        burned=False,
        defender_stat_modifier=int(continuation["defender_stat_modifier"]),
    )


def _range(context: DamageContext) -> tuple[int, int]:
    values = tuple(damage(context, roll) for roll in range(16))
    return min(values), max(values)


def analyze_document(document: Mapping[str, Any]) -> dict[str, object]:
    if document.get("schema") != "azelficoast.protect-action-survival":
        raise ActionSurvivalError("unexpected experiment schema")
    if document.get("schema_version") != 1:
        raise ActionSurvivalError("unexpected experiment schema version")

    source = document.get("source")
    information_set = document.get("information_set")
    continuation = document.get("continuation")
    if not all(
        isinstance(value, Mapping)
        for value in (source, information_set, continuation)
    ):
        raise ActionSurvivalError("experiment document is incomplete")
    if source.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        raise ActionSurvivalError("witness does not match pinned Showdown revision")

    item_counts = information_set.get("hidden_item_counts")
    if not isinstance(item_counts, Mapping):
        raise ActionSurvivalError("hidden item counts are missing")
    if int(item_counts.get("Choice Scarf", 0)) <= 0:
        raise ActionSurvivalError("Choice Scarf world has no support")
    if int(item_counts.get("Choice Specs", 0)) <= 0:
        raise ActionSurvivalError("Choice Specs world has no support")

    defender = continuation.get("defender")
    if not isinstance(defender, Mapping):
        raise ActionSurvivalError("defender is missing")
    full_hp = int(defender["max_hp"])
    post_spikes_hp = int(defender["hp_after_known_spikes"])
    if not 0 < post_spikes_hp <= full_hp:
        raise ActionSurvivalError("invalid defender HP bounds")

    scarf = _damage_context(continuation, "Choice Scarf")
    specs = _damage_context(continuation, "Choice Specs")
    scarf_min, scarf_max = _range(scarf)
    specs_min, specs_max = _range(specs)

    no_ruin_scarf_min, no_ruin_scarf_max = _range(
        replace(scarf, defender_stat_modifier=MOD_ONE)
    )
    no_ruin_specs_min, no_ruin_specs_max = _range(
        replace(specs, defender_stat_modifier=MOD_ONE)
    )

    full_hp_split = scarf_max < full_hp <= specs_min
    observed_hp_split = scarf_max < post_spikes_hp <= specs_min
    no_ruin_full_hp_split = (
        no_ruin_scarf_max < full_hp <= no_ruin_specs_min
    )

    passed = (
        full_hp_split
        and observed_hp_split
        and not no_ruin_full_hp_split
    )
    return {
        "schema": "azelficoast.action-survival-analysis",
        "schema_version": 1,
        "fixture_id": source["fixture_id"],
        "showdown_commit": source["showdown_commit"],
        "preserving_action": information_set["preserving_action"],
        "conditioned_observation": information_set["conditioned_observation"],
        "continuation_action": continuation["action"],
        "hidden_item_counts": {
            "Choice Scarf": int(item_counts["Choice Scarf"]),
            "Choice Specs": int(item_counts["Choice Specs"]),
        },
        "defender_full_hp": full_hp,
        "defender_hp_after_known_spikes": post_spikes_hp,
        "damage": {
            "Choice Scarf": {"min": scarf_min, "max": scarf_max},
            "Choice Specs": {"min": specs_min, "max": specs_max},
        },
        "without_defender_stat_modifier": {
            "Choice Scarf": {
                "min": no_ruin_scarf_min,
                "max": no_ruin_scarf_max,
            },
            "Choice Specs": {
                "min": no_ruin_specs_min,
                "max": no_ruin_specs_max,
            },
        },
        "full_hp_survival_split": full_hp_split,
        "observed_hp_survival_split": observed_hp_split,
        "negative_control_collapses_full_hp_split": not no_ruin_full_hp_split,
        "passed": passed,
        "non_claim": (
            "This proves a natural hidden-world survival/reachability divergence "
            "for one fixed opponent continuation. It does not prove that the "
            "optimal public-belief action differs from determinization."
        ),
    }


def analyze_file(path: str | Path) -> dict[str, object]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ActionSurvivalError("experiment document must be an object")
    return analyze_document(document)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_file(args.experiment)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
