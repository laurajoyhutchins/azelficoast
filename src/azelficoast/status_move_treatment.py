"""Freeze one natural status-move decision from a hosted public trace."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.corpus import DecisionFixture
from azelficoast.live_belief import opponent_move_from_protocol

SPEC_SCHEMA = "azelficoast.natural-status-move-treatment"
SPEC_SCHEMA_VERSION = 1
SOURCE_SCHEMA = "azelficoast.real-belief-source-fixture"
SOURCE_SCHEMA_VERSION = 1


class StatusMoveTreatmentError(ValueError):
    """Raised when the frozen source does not match preregistered evidence."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _to_id(value: Any) -> str:
    return "".join(
        character for character in str(value or "").lower() if character.isalnum()
    )


def _fixture_id(
    state: Mapping[str, Any],
    protocol_prefix: Sequence[Sequence[Sequence[str]]],
) -> str:
    material = {"state": state, "protocol_prefix": protocol_prefix}
    return hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()


def _load_spec(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        document.get("schema") != SPEC_SCHEMA
        or document.get("schema_version") != SPEC_SCHEMA_VERSION
    ):
        raise StatusMoveTreatmentError("unexpected treatment schema")
    if not isinstance(document.get("selection"), Mapping):
        raise StatusMoveTreatmentError("treatment lacks selection")
    if not isinstance(document.get("source_artifact"), Mapping):
        raise StatusMoveTreatmentError("treatment lacks source artifact")
    return document


def freeze_source(
    treatment: Mapping[str, Any],
    trace_path: str | Path,
) -> dict[str, Any]:
    selection = treatment["selection"]
    run_id = str(selection["trace_run_id"])
    battle_tag = str(selection["battle_tag"])
    target_event = int(selection["event_index"])
    target_decision = int(selection["decision_index"])

    protocol_prefix: list[list[list[str]]] = []
    decision: dict[str, Any] | None = None

    with Path(trace_path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise StatusMoveTreatmentError(
                    f"{trace_path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if record.get("run_id") != run_id:
                continue

            event_index = record.get("event_index")
            if not isinstance(event_index, int):
                continue

            if (
                event_index < target_event
                and record.get("kind") == "protocol"
                and record.get("room") == battle_tag
            ):
                messages = record.get("messages")
                if not isinstance(messages, list):
                    raise StatusMoveTreatmentError("protocol record lacks messages")
                batch: list[list[str]] = []
                for message in messages:
                    if not isinstance(message, list) or not all(
                        isinstance(field, str) for field in message
                    ):
                        raise StatusMoveTreatmentError(
                            "protocol message is not an array of strings"
                        )
                    batch.append(list(message))
                if batch:
                    protocol_prefix.append(batch)
                continue

            if event_index == target_event and record.get("kind") == "decision":
                if decision is not None:
                    raise StatusMoveTreatmentError("duplicate selected decision event")
                decision = record

    if decision is None:
        raise StatusMoveTreatmentError("selected decision event was not found")
    if decision.get("battle_tag") != battle_tag:
        raise StatusMoveTreatmentError("selected decision belongs to another battle")
    if decision.get("decision_index") != target_decision:
        raise StatusMoveTreatmentError("selected decision index changed")

    raw_state = decision.get("state")
    chosen_action = decision.get("chosen_action")
    if not isinstance(raw_state, Mapping) or not isinstance(chosen_action, str):
        raise StatusMoveTreatmentError("selected decision is incomplete")

    state = copy.deepcopy(dict(raw_state))
    state.pop("battle_tag", None)

    if int(state.get("turn", -1)) != int(selection["turn"]):
        raise StatusMoveTreatmentError("selected turn changed")

    active = state.get("active")
    opponent = state.get("opponent_active")
    if not isinstance(active, Mapping) or not isinstance(opponent, Mapping):
        raise StatusMoveTreatmentError("selected decision lacks active state")
    if _to_id(active.get("species")) != _to_id(selection["own_species"]):
        raise StatusMoveTreatmentError("selected own species changed")
    if _to_id(opponent.get("species")) != _to_id(selection["opponent_species"]):
        raise StatusMoveTreatmentError("selected opponent species changed")

    legal_actions = state.get("legal_actions")
    if (
        not isinstance(legal_actions, list)
        or not all(isinstance(action, str) for action in legal_actions)
        or chosen_action not in legal_actions
    ):
        raise StatusMoveTreatmentError("selected decision lacks valid legal actions")

    fixture_id = _fixture_id(state, protocol_prefix)
    if fixture_id != selection["expected_fixture_id"]:
        raise StatusMoveTreatmentError(
            "frozen fixture identity changed: "
            f"expected {selection['expected_fixture_id']}, got {fixture_id}"
        )

    fixture = DecisionFixture(
        fixture_id=fixture_id,
        state=state,
        protocol_prefix=tuple(
            tuple(tuple(field for field in message) for message in batch)
            for batch in protocol_prefix
        ),
        control_decisions=(
            {
                "run_id": run_id,
                "decision_index": target_decision,
                "chosen_action": chosen_action,
            },
        ),
    )

    observed_response = opponent_move_from_protocol(fixture)
    expected_response = str(selection["opponent_response_move"])
    if _to_id(observed_response) != _to_id(expected_response):
        raise StatusMoveTreatmentError(
            "current opponent response evidence changed: "
            f"expected {expected_response!r}, got {observed_response!r}"
        )

    plausible_items = selection.get("plausible_items")
    if (
        not isinstance(plausible_items, list)
        or not plausible_items
        or not all(isinstance(item, str) and item for item in plausible_items)
    ):
        raise StatusMoveTreatmentError("selection lacks plausible item support")

    return {
        "schema": SOURCE_SCHEMA,
        "schema_version": SOURCE_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "showdown_commit": treatment["showdown_commit"],
        "fixture": fixture.as_record(),
        "plausible_items": plausible_items,
        "observed_opponent_moves": [str(selection["observed_opponent_move"])],
        "opponent_response_move": expected_response,
        "own_active_tera_type": str(selection["own_active_tera_type"]),
        "opponent_bench_species": str(selection["opponent_bench_species"]),
        "source_artifact": {
            **dict(treatment["source_artifact"]),
            "trace_run_id": run_id,
            "battle_tag": battle_tag,
            "event_index": target_event,
            "decision_index": target_decision,
        },
        "source_projection": (
            "natural status-move promotion treatment with generator-conditioned "
            "singleton item support"
        ),
        "hypothesis": treatment.get("hypothesis"),
        "acceptance": treatment.get("acceptance"),
        "non_claims": treatment.get("non_claims", []),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("treatment", type=Path)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    treatment = _load_spec(args.treatment)
    source = freeze_source(treatment, args.trace)
    args.output.write_text(
        json.dumps(source, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "fixture_id": source["fixture_id"],
                "legal_action_count": len(source["fixture"]["state"]["legal_actions"]),
                "opponent_response_move": source["opponent_response_move"],
                "plausible_items": source["plausible_items"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
