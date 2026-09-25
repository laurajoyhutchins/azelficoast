"""Outcome-blind ranking and freezing for large-margin strategy-fusion search."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.corpus import DecisionFixture, load_corpus
from azelficoast.live_belief import build_probe_source

PLAN_SCHEMA = "azelficoast.large-margin-strategy-fusion-plan"
PLAN_SCHEMA_VERSION = 1
SELECTION_SCHEMA = "azelficoast.large-margin-strategy-fusion-selection"
SELECTION_SCHEMA_VERSION = 1


class FusionSearchError(ValueError):
    """Raised when discovery evidence cannot support a frozen treatment."""


@dataclass(frozen=True)
class CandidateSignals:
    separation_channel_count: int
    ko_roll_probability_gap_micros: int
    incoming_damage_fraction_gap_micros: int
    minimum_item_mass_micros: int
    persistent_branch_count: int
    legal_action_count: int

    @property
    def rank_key(self) -> tuple[int, ...]:
        return (
            self.separation_channel_count,
            self.ko_roll_probability_gap_micros,
            self.incoming_damage_fraction_gap_micros,
            self.minimum_item_mass_micros,
            self.persistent_branch_count,
            self.legal_action_count,
        )


def _load_object(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise FusionSearchError(f"{path}: expected a JSON object")
    return document


def _plan(path: str | Path) -> dict[str, Any]:
    document = _load_object(path)
    if (
        document.get("schema") != PLAN_SCHEMA
        or document.get("schema_version") != PLAN_SCHEMA_VERSION
    ):
        raise FusionSearchError("unexpected large-margin treatment plan schema")
    discovery = document.get("discovery")
    exact = document.get("exact_treatment")
    source = document.get("source_artifact")
    if not all(isinstance(value, Mapping) for value in (discovery, exact, source)):
        raise FusionSearchError("treatment plan is incomplete")
    return document


def _weighted_mean(
    worlds: Sequence[Mapping[str, Any]],
    value,
) -> float:
    weighted = 0.0
    total = 0
    for world in worlds:
        count = int(world.get("count", 0))
        if count <= 0:
            continue
        weighted += count * float(value(world))
        total += count
    if total <= 0:
        raise FusionSearchError("mechanics item support has no positive sample mass")
    return weighted / total


def _incoming_item_metrics(item_case: Mapping[str, Any], active_hp: float) -> dict[str, float]:
    raw_worlds = item_case.get("worlds")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise FusionSearchError("mechanics item case has no worlds")

    worlds = [world for world in raw_worlds if isinstance(world, Mapping)]
    if len(worlds) != len(raw_worlds):
        raise FusionSearchError("mechanics item world must be an object")

    def damage_fraction(world: Mapping[str, Any]) -> float:
        incoming = world.get("incoming")
        if not isinstance(incoming, Mapping):
            raise FusionSearchError("mechanics world lacks incoming evidence")
        if incoming.get("non_damage") is True:
            return 0.0
        low = incoming.get("damage_min")
        high = incoming.get("damage_max")
        if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
            raise FusionSearchError("mechanics world lacks incoming damage bounds")
        return ((float(low) + float(high)) / 2.0) / active_hp

    def ko_probability(world: Mapping[str, Any]) -> float:
        incoming = world.get("incoming")
        if not isinstance(incoming, Mapping):
            raise FusionSearchError("mechanics world lacks incoming evidence")
        if incoming.get("non_damage") is True:
            return 0.0
        return float(incoming.get("ko_rolls", 0)) / 16.0

    def opponent_first(world: Mapping[str, Any]) -> float:
        incoming = world.get("incoming")
        if not isinstance(incoming, Mapping):
            raise FusionSearchError("mechanics world lacks incoming evidence")
        own_speed = incoming.get("own_speed")
        opponent_speed = incoming.get("opponent_speed")
        if not isinstance(own_speed, (int, float)) or not isinstance(
            opponent_speed, (int, float)
        ):
            raise FusionSearchError("mechanics world lacks speed evidence")
        return float(opponent_speed > own_speed)

    return {
        "incoming_damage_fraction": _weighted_mean(worlds, damage_fraction),
        "incoming_ko_roll_probability": _weighted_mean(worlds, ko_probability),
        "opponent_acts_first_probability": _weighted_mean(worlds, opponent_first),
    }


def _candidate_signals(
    candidate: Mapping[str, Any],
    mechanics: Mapping[str, Any],
    fixture: DecisionFixture,
) -> tuple[CandidateSignals, dict[str, Any]]:
    item_weights = candidate.get("item_weights")
    if not isinstance(item_weights, Mapping) or len(item_weights) != 2:
        raise FusionSearchError("candidate must have exactly two hidden item weights")
    masses = [float(value) for value in item_weights.values()]
    if any(value <= 0 for value in masses):
        raise FusionSearchError("candidate item masses must be positive")

    by_item = mechanics.get("by_item")
    if not isinstance(by_item, Mapping):
        raise FusionSearchError("mechanics case lacks item partition")
    if set(by_item) != set(item_weights):
        raise FusionSearchError("candidate and mechanics item supports disagree")

    active = fixture.state.get("active")
    if not isinstance(active, Mapping):
        raise FusionSearchError("fixture lacks active state")
    active_hp = active.get("current_hp")
    if not isinstance(active_hp, (int, float)) or active_hp <= 0:
        raise FusionSearchError("fixture active HP must be positive")

    item_metrics = {
        str(item): _incoming_item_metrics(case, float(active_hp))
        for item, case in by_item.items()
        if isinstance(case, Mapping)
    }
    if len(item_metrics) != len(by_item):
        raise FusionSearchError("mechanics item case must be an object")

    damage_values = [
        metrics["incoming_damage_fraction"] for metrics in item_metrics.values()
    ]
    ko_values = [
        metrics["incoming_ko_roll_probability"] for metrics in item_metrics.values()
    ]
    order_values = [
        metrics["opponent_acts_first_probability"] for metrics in item_metrics.values()
    ]
    damage_gap = max(damage_values) - min(damage_values)
    ko_gap = max(ko_values) - min(ko_values)
    order_flip = min(order_values) < 0.5 < max(order_values)

    channels = int(order_flip) + int(ko_gap > 1e-12) + int(damage_gap > 1e-12)
    persistent = candidate.get("persistent_protect_actions", [])
    switches = candidate.get("persistent_switches", [])
    if not isinstance(persistent, list) or not isinstance(switches, list):
        raise FusionSearchError("candidate persistent branches are malformed")

    signals = CandidateSignals(
        separation_channel_count=channels,
        ko_roll_probability_gap_micros=round(ko_gap * 1_000_000),
        incoming_damage_fraction_gap_micros=round(damage_gap * 1_000_000),
        minimum_item_mass_micros=round(min(masses) * 1_000_000),
        persistent_branch_count=len(persistent) + len(switches),
        legal_action_count=len(fixture.legal_actions),
    )
    diagnostics = {
        "hidden_item_metrics": item_metrics,
        "relative_move_order_changes": order_flip,
        "incoming_ko_roll_probability_gap": ko_gap,
        "incoming_damage_fraction_gap": damage_gap,
        "minimum_hidden_item_mass": min(masses),
        "persistent_protect_action_count": len(persistent),
        "persistent_switch_count": len(switches),
    }
    return signals, diagnostics


def freeze_selection(
    *,
    plan: Mapping[str, Any],
    candidates_document: Mapping[str, Any],
    mechanics_document: Mapping[str, Any],
    fixtures: Sequence[DecisionFixture],
    output_dir: str | Path,
) -> dict[str, Any]:
    discovery = plan["discovery"]
    if candidates_document.get("schema") != "azelficoast.natural-fusion-candidates":
        raise FusionSearchError("unexpected candidate discovery schema")
    if mechanics_document.get("schema") != "azelficoast.public-belief-speed-fork-mechanics":
        raise FusionSearchError("unexpected mechanics screening schema")
    if candidates_document.get("persistent_only") is not True:
        raise FusionSearchError("candidate discovery must be persistent-only")
    if candidates_document.get("sample_rounds") not in (None, discovery["generator_rounds"]):
        raise FusionSearchError("candidate discovery generator resolution changed")
    if mechanics_document.get("showdown_commit") != plan["showdown_commit"]:
        raise FusionSearchError("mechanics screen used another Showdown revision")

    candidates = candidates_document.get("candidates")
    cases = mechanics_document.get("cases")
    if not isinstance(candidates, list) or not isinstance(cases, list):
        raise FusionSearchError("discovery evidence lacks candidates or mechanics cases")

    fixture_by_id = {fixture.fixture_id: fixture for fixture in fixtures}
    case_by_id = {
        str(case["fixture_id"]): case
        for case in cases
        if isinstance(case, Mapping) and isinstance(case.get("fixture_id"), str)
    }

    ranked: list[dict[str, Any]] = []
    ineligible_reason_counts: dict[str, int] = {}

    def ineligible(reason: str) -> None:
        ineligible_reason_counts[reason] = ineligible_reason_counts.get(reason, 0) + 1

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise FusionSearchError("candidate must be an object")
        fixture_id = candidate.get("fixture_id")
        if not isinstance(fixture_id, str):
            raise FusionSearchError("candidate lacks fixture ID")
        fixture = fixture_by_id.get(fixture_id)
        mechanics = case_by_id.get(fixture_id)
        if fixture is None or mechanics is None:
            raise FusionSearchError(f"missing evidence for candidate {fixture_id}")

        active = fixture.state.get("active")
        active_hp = active.get("current_hp") if isinstance(active, Mapping) else None
        if not isinstance(active_hp, (int, float)) or active_hp <= 0:
            ineligible("active-hp-nonpositive")
            continue

        source, admission = build_probe_source(fixture)
        if source is None:
            ineligible(f"live-admission:{admission}")
            continue
        if admission != "admitted":
            raise FusionSearchError("live admission returned source without admitted status")

        candidate_items = set(str(item) for item in candidate["item_weights"])
        source_items = set(str(item) for item in source.get("plausible_items", []))
        if source_items != candidate_items:
            raise FusionSearchError(
                f"{fixture_id}: live source item support differs from discovery"
            )

        signals, diagnostics = _candidate_signals(candidate, mechanics, fixture)
        ranked.append(
            {
                "fixture_id": fixture_id,
                "rank_key": list(signals.rank_key),
                "signals": {
                    "separation_channel_count": signals.separation_channel_count,
                    "ko_roll_probability_gap_micros": (
                        signals.ko_roll_probability_gap_micros
                    ),
                    "incoming_damage_fraction_gap_micros": (
                        signals.incoming_damage_fraction_gap_micros
                    ),
                    "minimum_item_mass_micros": signals.minimum_item_mass_micros,
                    "persistent_branch_count": signals.persistent_branch_count,
                    "legal_action_count": signals.legal_action_count,
                },
                "diagnostics": diagnostics,
                "candidate": dict(candidate),
                "source": source,
            }
        )

    ranked.sort(
        key=lambda row: (
            tuple(-int(value) for value in row["rank_key"]),
            row["fixture_id"],
        )
    )

    top_k = int(discovery["top_k"])
    selected = ranked[:top_k]
    if len(selected) != top_k:
        raise FusionSearchError(
            f"preregistered treatment requires {top_k} eligible candidates, "
            f"found {len(selected)}"
        )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_artifact = dict(plan["source_artifact"])
    selected_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        source = dict(row["source"])
        source["observed_opponent_moves"] = list(row["candidate"]["revealed_moves"])
        source["source_artifact"] = {
            **source_artifact,
            "selection_rank": rank,
            "selection_fixture_id": row["fixture_id"],
        }
        source["selection"] = {
            "schema": SELECTION_SCHEMA,
            "schema_version": SELECTION_SCHEMA_VERSION,
            "selection_rank": rank,
            "ranking_uses_policy_result": False,
            "rank_key": row["rank_key"],
            "signals": row["signals"],
            "diagnostics": row["diagnostics"],
        }
        filename = f"{rank:02d}-{row['fixture_id']}.json"
        (destination / filename).write_text(
            json.dumps(source, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_rows.append(
            {
                "selection_rank": rank,
                "fixture_id": row["fixture_id"],
                "filename": filename,
                "rank_key": row["rank_key"],
                "signals": row["signals"],
                "diagnostics": row["diagnostics"],
                "active_species": row["candidate"].get("active_species"),
                "opponent_species": row["candidate"].get("opponent_species"),
                "locked_move": row["candidate"].get("locked_move"),
                "item_weights": row["candidate"].get("item_weights"),
                "persistent_protect_actions": row["candidate"].get(
                    "persistent_protect_actions"
                ),
                "persistent_switches": row["candidate"].get("persistent_switches"),
            }
        )

    return {
        "schema": SELECTION_SCHEMA,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "source_artifact": source_artifact,
        "showdown_commit": plan["showdown_commit"],
        "ranking_uses_policy_result": False,
        "eligible_candidate_count": len(ranked),
        "ineligible_reason_counts": dict(sorted(ineligible_reason_counts.items())),
        "selected_count": len(selected_rows),
        "top_k": top_k,
        "rank_order": list(discovery["rank_order"]),
        "selected": selected_rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("mechanics", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _plan(args.plan)
    candidates = _load_object(args.candidates)
    mechanics = _load_object(args.mechanics)
    fixtures = load_corpus(args.corpus)
    result = freeze_selection(
        plan=plan,
        candidates_document=candidates,
        mechanics_document=mechanics,
        fixtures=fixtures,
        output_dir=args.output_dir,
    )
    args.manifest.write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
