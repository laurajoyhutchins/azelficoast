"""Validate simulator dependency classes against Pokémon Showdown-produced transitions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.simulator_ir import (
    ITEM_CHOICE_SCARF,
    ITEM_CHOICE_SPECS,
    MOVE_MOONBLAST,
    PROTECT_BLOCK,
    SPECIAL_DAMAGE,
    DependencyViolation,
    EffectSpec,
    StateField,
    World,
    field_mask,
)

PINNED_SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc"

ITEM_CODES = {
    "Choice Scarf": ITEM_CHOICE_SCARF,
    "Choice Specs": ITEM_CHOICE_SPECS,
}
MOVE_CODES = {
    "Aura Sphere": MOVE_MOONBLAST,
}


class ShowdownTransitionCorpusError(ValueError):
    """Raised when hosted Showdown transition evidence is incomplete or inconsistent."""


@dataclass(frozen=True)
class OracleOutcome:
    after_hp: int
    hp_delta: int
    transition_log: tuple[str, ...]


@dataclass(frozen=True)
class OraclePartition:
    fixture_count: int
    class_count: int

    @property
    def reduction_factor(self) -> float:
        if not self.class_count:
            return 1.0
        return self.fixture_count / self.class_count


def _seed_key(fixture: Mapping[str, Any]) -> tuple[int, ...]:
    raw = fixture.get("seed")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ShowdownTransitionCorpusError("fixture seed must be a sequence")
    seed = tuple(int(value) for value in raw)
    if len(seed) != 4:
        raise ShowdownTransitionCorpusError("fixture seed must contain four integers")
    return seed


def _world(fixture: Mapping[str, Any]) -> World:
    item = str(fixture.get("opponent_item") or "")
    if item not in ITEM_CODES:
        raise ShowdownTransitionCorpusError(f"unsupported fixture item {item!r}")
    move = str(fixture.get("opponent_move") or "")
    if move not in MOVE_CODES:
        raise ShowdownTransitionCorpusError(f"unsupported fixture move {move!r}")

    return World.from_values(
        {
            StateField.OWN_HP: int(fixture["before_hp"]),
            StateField.OPPONENT_ITEM: ITEM_CODES[item],
            StateField.OPPONENT_MOVE: MOVE_CODES[move],
            StateField.BENCH_SIGNATURE: int(fixture["bench_signature"]),
        }
    )


def _outcome(fixture: Mapping[str, Any]) -> OracleOutcome:
    raw_log = fixture.get("transition_log")
    if not isinstance(raw_log, Sequence) or isinstance(raw_log, (str, bytes)):
        raise ShowdownTransitionCorpusError("fixture transition_log must be a sequence")
    return OracleOutcome(
        after_hp=int(fixture["after_hp"]),
        hp_delta=int(fixture["hp_delta"]),
        transition_log=tuple(str(line) for line in raw_log),
    )


def _partition(
    spec: EffectSpec,
    fixtures: Sequence[Mapping[str, Any]],
) -> OraclePartition:
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], set[OracleOutcome]] = {}
    for fixture in fixtures:
        world = _world(fixture)
        key = (world.project(spec.reads), _seed_key(fixture))
        groups.setdefault(key, set()).add(_outcome(fixture))

    for key, outcomes in groups.items():
        if len(outcomes) != 1:
            raise DependencyViolation(
                f"{spec.name} produced {len(outcomes)} Showdown outcomes for one "
                f"declared dependency class {key!r}"
            )

    return OraclePartition(fixture_count=len(fixtures), class_count=len(groups))


def _fixtures_for(
    document: Mapping[str, Any],
    scenario: str,
) -> tuple[Mapping[str, Any], ...]:
    raw = document.get("fixtures")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ShowdownTransitionCorpusError("fixture document lacks fixtures")
    fixtures = tuple(
        fixture
        for fixture in raw
        if isinstance(fixture, Mapping) and fixture.get("scenario") == scenario
    )
    if not fixtures:
        raise ShowdownTransitionCorpusError(f"no fixtures for scenario {scenario!r}")
    return fixtures


def analyze_document(
    document: Mapping[str, Any],
    *,
    expected_showdown_commit: str = PINNED_SHOWDOWN_COMMIT,
) -> dict[str, object]:
    if document.get("schema") != "azelficoast.showdown-transition-fixtures":
        raise ShowdownTransitionCorpusError("unexpected fixture schema")
    if document.get("schema_version") != 1:
        raise ShowdownTransitionCorpusError("unexpected fixture schema version")
    if document.get("showdown_commit") != expected_showdown_commit:
        raise ShowdownTransitionCorpusError(
            "fixture Showdown revision does not match the pinned oracle"
        )

    protect = _fixtures_for(document, "protect")
    damage = _fixtures_for(document, "damage")
    seed_count = int(document.get("seed_count") or 0)
    item_count = int(document.get("item_count") or 0)
    bench_count = int(document.get("bench_variant_count") or 0)
    expected_per_scenario = seed_count * item_count * bench_count
    if expected_per_scenario <= 0:
        raise ShowdownTransitionCorpusError("invalid fixture dimensions")
    if len(protect) != expected_per_scenario or len(damage) != expected_per_scenario:
        raise ShowdownTransitionCorpusError("fixture matrix is incomplete")

    protect_partition = _partition(PROTECT_BLOCK, protect)
    damage_partition = _partition(SPECIAL_DAMAGE, damage)

    missing_item = EffectSpec(
        name="showdown-damage-without-item",
        op=SPECIAL_DAMAGE.op,
        reads=field_mask(StateField.OWN_HP),
        writes=SPECIAL_DAMAGE.writes,
        random=SPECIAL_DAMAGE.random,
    )
    negative_control_detected = False
    try:
        _partition(missing_item, damage)
    except DependencyViolation:
        negative_control_detected = True

    expected_protect_classes = seed_count
    expected_damage_classes = seed_count * item_count
    passed = (
        protect_partition.class_count == expected_protect_classes
        and damage_partition.class_count == expected_damage_classes
        and negative_control_detected
    )

    return {
        "schema": "azelficoast.showdown-transition-dependency-analysis",
        "schema_version": 1,
        "showdown_commit": expected_showdown_commit,
        "fixture_count": len(protect) + len(damage),
        "protect": {
            "fixture_count": protect_partition.fixture_count,
            "dependency_classes": protect_partition.class_count,
            "expected_dependency_classes": expected_protect_classes,
            "reduction_factor": protect_partition.reduction_factor,
        },
        "damage": {
            "fixture_count": damage_partition.fixture_count,
            "dependency_classes": damage_partition.class_count,
            "expected_dependency_classes": expected_damage_classes,
            "reduction_factor": damage_partition.reduction_factor,
        },
        "missing_item_negative_control_detected": negative_control_detected,
        "passed": passed,
    }


def analyze_file(
    path: str | Path,
    *,
    expected_showdown_commit: str = PINNED_SHOWDOWN_COMMIT,
) -> dict[str, object]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ShowdownTransitionCorpusError("fixture document must be an object")
    return analyze_document(
        document,
        expected_showdown_commit=expected_showdown_commit,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixtures", type=Path)
    parser.add_argument(
        "--expected-showdown-commit",
        default=PINNED_SHOWDOWN_COMMIT,
    )
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
