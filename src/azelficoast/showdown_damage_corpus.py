"""Differential analysis of the Gen 9 damage kernel against pinned Showdown."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.gen9_damage import DamageContext, damage

PINNED_SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc"


class ShowdownDamageCorpusError(ValueError):
    """Raised when Showdown damage evidence is malformed or incomplete."""


def _context(raw: Mapping[str, Any]) -> DamageContext:
    attacker_types = raw.get("attacker_types")
    if not isinstance(attacker_types, Sequence) or isinstance(attacker_types, (str, bytes)):
        raise ShowdownDamageCorpusError("attacker_types must be a sequence")

    tera = raw.get("tera_type")
    if tera is not None and not isinstance(tera, str):
        raise ShowdownDamageCorpusError("tera_type must be a string or null")

    category = str(raw["category"])
    if category not in ("Physical", "Special"):
        raise ShowdownDamageCorpusError(f"unsupported category {category!r}")

    return DamageContext(
        attacker_level=int(raw["attacker_level"]),
        defender_level=int(raw["defender_level"]),
        base_power=int(raw["base_power"]),
        category=category,  # type: ignore[arg-type]
        move_id=str(raw["move_id"]),
        move_type=str(raw["move_type"]),
        attacker_types=tuple(str(value) for value in attacker_types),
        tera_type=tera,
        attacker_base_stat=int(raw["attacker_base_stat"]),
        attacker_iv=int(raw["attacker_iv"]),
        attacker_ev=int(raw["attacker_ev"]),
        attacker_nature_percent=int(raw["attacker_nature_percent"]),
        defender_base_stat=int(raw["defender_base_stat"]),
        defender_iv=int(raw["defender_iv"]),
        defender_ev=int(raw["defender_ev"]),
        defender_nature_percent=int(raw["defender_nature_percent"]),
        attacker_item=str(raw.get("attacker_item") or ""),
        type_mod=int(raw["type_mod"]),
        burned=bool(raw.get("burned", False)),
    )


def contexts_from_document(document: Mapping[str, Any]) -> tuple[DamageContext, ...]:
    fixtures = document.get("fixtures")
    if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
        raise ShowdownDamageCorpusError("fixture document lacks fixtures")
    return tuple(
        _context(fixture["context"])
        for fixture in fixtures
        if isinstance(fixture, Mapping) and isinstance(fixture.get("context"), Mapping)
    )


def _rolls(fixture: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = fixture.get("rolls")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ShowdownDamageCorpusError("fixture rolls must be a sequence")
    values = tuple(value for value in raw if isinstance(value, Mapping))
    if len(values) != 16:
        raise ShowdownDamageCorpusError("each fixture must contain all 16 damage rolls")
    observed = {int(value["roll"]) for value in values}
    if observed != set(range(16)):
        raise ShowdownDamageCorpusError("damage-roll matrix must contain rolls 0 through 15")
    return values


def _mismatch_count(context: DamageContext, rolls: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        damage(context, int(entry["roll"])) != int(entry["damage"])
        for entry in rolls
    )


def analyze_document(
    document: Mapping[str, Any],
    *,
    expected_showdown_commit: str = PINNED_SHOWDOWN_COMMIT,
) -> dict[str, object]:
    if document.get("schema") != "azelficoast.showdown-gen9-damage-fixtures":
        raise ShowdownDamageCorpusError("unexpected fixture schema")
    if document.get("schema_version") != 1:
        raise ShowdownDamageCorpusError("unexpected fixture schema version")
    if document.get("showdown_commit") != expected_showdown_commit:
        raise ShowdownDamageCorpusError("fixture Showdown revision does not match the pinned oracle")

    fixtures = document.get("fixtures")
    if not isinstance(fixtures, Sequence) or isinstance(fixtures, (str, bytes)):
        raise ShowdownDamageCorpusError("fixture document lacks fixtures")
    if len(fixtures) != int(document.get("scenario_count") or 0):
        raise ShowdownDamageCorpusError("scenario_count does not match fixture count")

    scenario_ids: set[str] = set()
    exact_cases = 0
    total_cases = 0
    type_negative_detected = False
    item_negative_detected = False
    tera_negative_detected = False
    burn_negative_detected = False

    for fixture in fixtures:
        if not isinstance(fixture, Mapping):
            raise ShowdownDamageCorpusError("fixture must be an object")
        scenario = str(fixture.get("scenario") or "")
        if not scenario or scenario in scenario_ids:
            raise ShowdownDamageCorpusError("scenario IDs must be non-empty and unique")
        scenario_ids.add(scenario)

        context_raw = fixture.get("context")
        if not isinstance(context_raw, Mapping):
            raise ShowdownDamageCorpusError("fixture lacks damage context")
        context = _context(context_raw)
        rolls = _rolls(fixture)

        mismatches = _mismatch_count(context, rolls)
        total_cases += len(rolls)
        exact_cases += len(rolls) - mismatches
        if mismatches:
            raise ShowdownDamageCorpusError(
                f"{scenario} disagrees with Showdown on {mismatches} of 16 damage rolls"
            )

        if context.type_mod:
            type_negative_detected |= _mismatch_count(
                replace(context, type_mod=0),
                rolls,
            ) > 0
        if context.attacker_item:
            item_negative_detected |= _mismatch_count(context.without_item(), rolls) > 0
        if context.tera_type:
            tera_negative_detected |= _mismatch_count(
                replace(context, tera_type=None),
                rolls,
            ) > 0
        if context.burned:
            burn_negative_detected |= _mismatch_count(
                replace(context, burned=False),
                rolls,
            ) > 0

    passed = (
        exact_cases == total_cases
        and total_cases > 0
        and type_negative_detected
        and item_negative_detected
        and tera_negative_detected
        and burn_negative_detected
    )
    return {
        "schema": "azelficoast.showdown-gen9-damage-analysis",
        "schema_version": 1,
        "showdown_commit": expected_showdown_commit,
        "scenario_count": len(fixtures),
        "roll_case_count": total_cases,
        "exact_case_count": exact_cases,
        "type_negative_control_detected": type_negative_detected,
        "item_negative_control_detected": item_negative_detected,
        "tera_negative_control_detected": tera_negative_detected,
        "burn_negative_control_detected": burn_negative_detected,
        "passed": passed,
    }


def analyze_file(
    path: str | Path,
    *,
    expected_showdown_commit: str = PINNED_SHOWDOWN_COMMIT,
) -> dict[str, object]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ShowdownDamageCorpusError("fixture document must be an object")
    return analyze_document(document, expected_showdown_commit=expected_showdown_commit)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--expected-showdown-commit", default=PINNED_SHOWDOWN_COMMIT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_file(
        args.fixtures,
        expected_showdown_commit=args.expected_showdown_commit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
