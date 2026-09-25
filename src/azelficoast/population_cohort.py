"""Frozen cohort selection and preregistered admission checks."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.corpus import DecisionFixture
from azelficoast.fusion_search import FusionSearchError, _candidate_signals
from azelficoast.live_belief import build_probe_source

COHORT_SCHEMA = "azelficoast.natural-population-strategy-fusion-cohort"
COHORT_SCHEMA_VERSION = 1
SELECTION_SALT = "population-v1:"


class PopulationStudyError(ValueError):
    """Raised when population evidence violates the preregistered contract."""


def _battle_tag(fixture: DecisionFixture) -> str:
    tags = {
        str(control.get("battle_tag"))
        for control in fixture.control_decisions
        if isinstance(control, Mapping) and control.get("battle_tag")
    }
    if len(tags) == 1:
        return next(iter(tags))
    if not tags:
        return f"fixture:{fixture.fixture_id}"
    raise PopulationStudyError(f"{fixture.fixture_id}: controls span multiple battle tags")


def _to_id(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _exact_opponent_bench_status(fixture: DecisionFixture) -> str:
    """Mirror the exact Showdown adapter's public bench reconstruction gate."""

    own_name = _to_id(fixture.state.get("player"))
    opponent_name = _to_id(fixture.state.get("opponent"))
    own_side: str | None = None
    opponent_side: str | None = None

    for batch in fixture.protocol_prefix:
        for message in batch:
            if (
                len(message) >= 4
                and message[0] == ""
                and message[1] == "player"
                and message[2] in {"p1", "p2"}
            ):
                name = _to_id(message[3])
                if name == own_name:
                    own_side = message[2]
                if name == opponent_name:
                    opponent_side = message[2]

    if own_side is None and opponent_side is not None:
        own_side = "p2" if opponent_side == "p1" else "p1"
    if opponent_side is None and own_side is not None:
        opponent_side = "p2" if own_side == "p1" else "p1"
    if opponent_side is None:
        return "opponent-side-unresolved"

    active: str | None = None
    seen: list[str] = []
    fainted: set[str] = set()
    team_size: int | None = None

    for batch in fixture.protocol_prefix:
        for message in batch:
            if len(message) < 2 or message[0] != "":
                continue
            if len(message) >= 4 and message[1] == "teamsize" and message[2] == opponent_side:
                try:
                    team_size = int(message[3])
                except ValueError:
                    pass

            actor = str(message[2]) if len(message) >= 3 else ""
            if (
                message[1] in {"switch", "drag"}
                and actor.startswith(opponent_side)
                and len(message) >= 4
            ):
                active = str(message[3]).split(",", 1)[0]
                if active and not any(_to_id(species) == _to_id(active) for species in seen):
                    seen.append(active)
            if message[1] == "faint" and actor.startswith(opponent_side) and active:
                fainted.add(_to_id(active))

    opponent_active = fixture.state.get("opponent_active")
    if not isinstance(opponent_active, Mapping):
        return "opponent-active-missing"
    current = _to_id(opponent_active.get("species"))
    if any(_to_id(species) != current and _to_id(species) not in fainted for species in seen):
        return "known-surviving-bench"

    opponent_team = fixture.state.get("opponent_team")
    public_team = list(opponent_team.values()) if isinstance(opponent_team, Mapping) else []
    known_species = {
        _to_id(view.get("species"))
        for view in public_team
        if isinstance(view, Mapping) and _to_id(view.get("species"))
    }
    if (
        team_size is not None
        and len(known_species) >= team_size
        and all(
            _to_id(view.get("species")) == current or bool(view.get("fainted"))
            for view in public_team
            if isinstance(view, Mapping)
        )
    ):
        return "public-bench-exhausted"

    return "opponent-bench-unresolved"


def _hp_fraction(view: Mapping[str, Any]) -> float | None:
    current = view.get("current_hp")
    maximum = view.get("max_hp")
    if not isinstance(current, (int, float)) or not isinstance(maximum, (int, float)):
        return None
    if maximum <= 0:
        return None
    return float(current) / float(maximum)


def _opponent_public_hp_fraction(view: Mapping[str, Any]) -> float | None:
    fraction = view.get("hp_fraction")
    if isinstance(fraction, (int, float)):
        return float(fraction)
    current = view.get("current_hp")
    maximum = view.get("max_hp")
    if isinstance(current, (int, float)) and isinstance(maximum, (int, float)) and maximum > 0:
        return float(current) / float(maximum)
    if isinstance(current, (int, float)) and 0 <= float(current) <= 100:
        return float(current) / 100.0
    return None


def _entropy(weights: Mapping[str, Any]) -> float:
    values = [float(value) for value in weights.values()]
    if not values or any(value <= 0 for value in values):
        raise PopulationStudyError("hidden-item weights must be positive")
    total = sum(values)
    if total <= 0:
        raise PopulationStudyError("hidden-item weights have no mass")
    return -sum((value / total) * math.log2(value / total) for value in values)


def _selection_key(fixture_id: str) -> str:
    return hashlib.sha256((SELECTION_SALT + fixture_id).encode("utf-8")).hexdigest()


def _predictors(
    fixture: DecisionFixture,
    candidate: Mapping[str, Any],
    signals: Any,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    active = fixture.state.get("active")
    opponent = fixture.state.get("opponent_active")
    if not isinstance(active, Mapping) or not isinstance(opponent, Mapping):
        raise PopulationStudyError("admitted fixture lacks active state")
    item_weights = candidate.get("item_weights")
    if not isinstance(item_weights, Mapping):
        raise PopulationStudyError("candidate lacks hidden-item weights")
    protects = candidate.get("persistent_protect_actions")
    switches = candidate.get("persistent_switches")
    if not isinstance(protects, list) or not isinstance(switches, list):
        raise PopulationStudyError("candidate persistent branches are malformed")

    return {
        "turn": int(fixture.state.get("turn", 0)),
        "own_active_hp_fraction": _hp_fraction(active),
        "opponent_public_hp_fraction": _opponent_public_hp_fraction(opponent),
        "hidden_item_entropy_bits": _entropy(item_weights),
        "relative_move_order_changes": bool(diagnostics["relative_move_order_changes"]),
        "incoming_ko_roll_probability_gap": float(diagnostics["incoming_ko_roll_probability_gap"]),
        "incoming_damage_fraction_gap": float(diagnostics["incoming_damage_fraction_gap"]),
        "minimum_hidden_item_mass": float(diagnostics["minimum_hidden_item_mass"]),
        "persistent_protect_present": bool(protects),
        "persistent_switch_present": bool(switches),
        "persistent_branch_count": int(signals.persistent_branch_count),
        "legal_action_count": int(signals.legal_action_count),
    }


def freeze_population(
    *,
    plan: Mapping[str, Any],
    candidates_document: Mapping[str, Any],
    mechanics_document: Mapping[str, Any],
    fixtures: Sequence[DecisionFixture],
    output_dir: str | Path,
) -> dict[str, Any]:
    admission = plan["admissibility"]
    if candidates_document.get("schema") != "azelficoast.natural-fusion-candidates":
        raise PopulationStudyError("unexpected candidate discovery schema")
    if candidates_document.get("persistent_only") is not True:
        raise PopulationStudyError("candidate discovery must be persistent-only")
    if mechanics_document.get("schema") != "azelficoast.public-belief-speed-fork-mechanics":
        raise PopulationStudyError("unexpected mechanics-screen schema")
    if mechanics_document.get("showdown_commit") != plan["showdown_commit"]:
        raise PopulationStudyError("mechanics screen used another Showdown revision")
    if int(mechanics_document.get("rounds", 0)) != int(admission["mechanics_screen_rounds"]):
        raise PopulationStudyError("mechanics-screen sampling resolution changed")

    source_count = int(plan["source_artifact"]["decision_state_count"])
    if len(fixtures) != source_count:
        raise PopulationStudyError(
            f"source corpus has {len(fixtures)} fixtures; preregistered {source_count}"
        )

    candidates = candidates_document.get("candidates")
    excluded = candidates_document.get("excluded_fixtures")
    cases = mechanics_document.get("cases")
    if (
        not isinstance(candidates, list)
        or not isinstance(excluded, list)
        or not isinstance(cases, list)
    ):
        raise PopulationStudyError(
            "discovery evidence lacks candidates, exclusions, or mechanics cases"
        )
    if len(candidates) + len(excluded) != source_count:
        raise PopulationStudyError(
            "candidate/exclusion ledger does not cover the source population"
        )
    candidate_ids = {str(row.get("fixture_id")) for row in candidates if isinstance(row, Mapping)}
    excluded_ids = {str(row.get("fixture_id")) for row in excluded if isinstance(row, Mapping)}
    if (
        len(candidate_ids) != len(candidates)
        or len(excluded_ids) != len(excluded)
        or candidate_ids & excluded_ids
        or candidate_ids | excluded_ids != {fixture.fixture_id for fixture in fixtures}
    ):
        raise PopulationStudyError(
            "candidate/exclusion ledger is incomplete, duplicated, or overlapping"
        )

    fixture_by_id = {fixture.fixture_id: fixture for fixture in fixtures}
    case_by_id = {
        str(case["fixture_id"]): case
        for case in cases
        if isinstance(case, Mapping) and isinstance(case.get("fixture_id"), str)
    }
    eligible: list[dict[str, Any]] = []
    ineligible_reason_counts: dict[str, int] = {}
    seen_candidates: set[str] = set()

    def exclude(reason: str) -> None:
        ineligible_reason_counts[reason] = ineligible_reason_counts.get(reason, 0) + 1

    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise PopulationStudyError("candidate must be an object")
        fixture_id = candidate.get("fixture_id")
        if not isinstance(fixture_id, str):
            raise PopulationStudyError("candidate lacks fixture ID")
        if fixture_id in seen_candidates:
            raise PopulationStudyError(f"duplicate candidate {fixture_id}")
        seen_candidates.add(fixture_id)
        if int(candidate.get("sample_rounds", 0)) != int(admission["generator_rounds"]):
            raise PopulationStudyError(f"{fixture_id}: generator sampling resolution changed")

        fixture = fixture_by_id.get(fixture_id)
        mechanics = case_by_id.get(fixture_id)
        if fixture is None:
            raise PopulationStudyError(f"candidate {fixture_id} missing from source corpus")
        if mechanics is None:
            raise PopulationStudyError(f"candidate {fixture_id} lacks mechanics screen")

        active = fixture.state.get("active")
        opponent_active = fixture.state.get("opponent_active")
        active_hp = active.get("current_hp") if isinstance(active, Mapping) else None
        if not isinstance(active_hp, (int, float)) or active_hp <= 0:
            exclude("active-hp-nonpositive")
            continue
        if (
            isinstance(opponent_active, Mapping)
            and isinstance(opponent_active.get("tera_type"), str)
            and opponent_active.get("tera_type")
        ):
            exclude("opponent-terastallized")
            continue

        source, status = build_probe_source(fixture)
        if source is None:
            exclude(f"live-admission:{status}")
            continue
        if status != "admitted":
            raise PopulationStudyError("live admission returned source without admitted status")

        bench_status = _exact_opponent_bench_status(fixture)
        if bench_status not in {
            "known-surviving-bench",
            "public-bench-exhausted",
        }:
            exclude(f"exact-reconstruction:{bench_status}")
            continue

        weights = candidate.get("item_weights")
        if not isinstance(weights, Mapping):
            raise PopulationStudyError(f"{fixture_id}: candidate item weights missing")
        if set(map(str, weights)) != set(map(str, source.get("plausible_items", []))):
            exclude("item-support-mismatch")
            continue

        try:
            signals, diagnostics = _candidate_signals(candidate, mechanics, fixture)
        except FusionSearchError as error:
            raise PopulationStudyError(
                f"{fixture_id}: invalid mechanics evidence: {error}"
            ) from error

        eligible.append(
            {
                "fixture_id": fixture_id,
                "battle_tag": _battle_tag(fixture),
                "selection_key": _selection_key(fixture_id),
                "predictors": _predictors(
                    fixture,
                    candidate,
                    signals,
                    diagnostics,
                ),
                "candidate": dict(candidate),
                "source": source,
            }
        )

    eligible.sort(key=lambda row: (row["selection_key"], row["fixture_id"]))
    cap = int(admission["max_exact_states"])
    if cap < 1:
        raise PopulationStudyError("max_exact_states must be positive")
    selected = eligible[:cap]

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_artifact = dict(plan["source_artifact"])
    selected_rows: list[dict[str, Any]] = []

    for index, row in enumerate(selected, start=1):
        source = dict(row["source"])
        candidate = row["candidate"]
        source["observed_opponent_moves"] = list(candidate["revealed_moves"])
        source["opponent_is_lead"] = bool(candidate["is_lead"])
        source["expected_item_counts"] = {
            str(item): int(count) for item, count in candidate["item_counts"].items()
        }
        source["expected_generator_rounds"] = int(candidate["sample_rounds"])
        source["source_artifact"] = {
            **source_artifact,
            "population_index": index,
            "selection_fixture_id": row["fixture_id"],
        }
        source["population_selection"] = {
            "schema": COHORT_SCHEMA,
            "schema_version": COHORT_SCHEMA_VERSION,
            "frozen_before_policy_values": True,
            "selection_uses_policy_result": False,
            "population_index": index,
            "battle_tag": row["battle_tag"],
            "selection_key": row["selection_key"],
            "predictors": row["predictors"],
        }
        filename = f"{index:03d}-{row['fixture_id']}.json"
        (destination / filename).write_text(
            json.dumps(source, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        selected_rows.append(
            {
                "population_index": index,
                "fixture_id": row["fixture_id"],
                "battle_tag": row["battle_tag"],
                "selection_key": row["selection_key"],
                "filename": filename,
                "predictors": row["predictors"],
            }
        )

    outside = len(excluded)

    return {
        "schema": COHORT_SCHEMA,
        "schema_version": COHORT_SCHEMA_VERSION,
        "source_artifact": source_artifact,
        "showdown_commit": plan["showdown_commit"],
        "frozen_before_policy_values": True,
        "selection_uses_policy_result": False,
        "source_decision_state_count": source_count,
        "bounded_candidate_count": len(candidates),
        "source_outside_bounded_model_count": outside,
        "source_exclusion_reason_counts": dict(
            sorted(
                {
                    reason: sum(
                        1
                        for row in excluded
                        if isinstance(row, Mapping) and row.get("reason") == reason
                    )
                    for reason in {
                        str(row.get("reason")) for row in excluded if isinstance(row, Mapping)
                    }
                }.items()
            )
        ),
        "source_exclusions": [dict(row) for row in excluded],
        "eligible_count": len(eligible),
        "ineligible_reason_counts": dict(sorted(ineligible_reason_counts.items())),
        "max_exact_states": cap,
        "overflow_sampling_applied": len(eligible) > cap,
        "overflow_selection": admission["overflow_selection"],
        "selected_count": len(selected_rows),
        "eligible_fixture_ids": [row["fixture_id"] for row in eligible],
        "selected": selected_rows,
    }
