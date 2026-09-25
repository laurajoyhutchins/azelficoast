"""Build leakage-safe policy/value training records from real decision traces."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.corpus import DecisionFixture, build_fixtures
from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION

TRAINING_SCHEMA = "azelficoast.training-record"
TRAINING_SCHEMA_VERSION = 1
SEARCH_ANNOTATION_SCHEMA = "azelficoast.public-belief-search-target"
SEARCH_ANNOTATION_SCHEMA_VERSION = 1
DEFAULT_SPLIT_SEED = "azelficoast.training-records"


class TrainingRecordError(ValueError):
    """Raised when training evidence is incomplete or internally inconsistent."""


class TrainingRecordConflictError(TrainingRecordError):
    """Raised when an immutable training output path contains different evidence."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = _canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_trace_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise TrainingRecordError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise TrainingRecordError(
                        f"{path}:{line_number}: trace record must be an object"
                    )
                if record.get("schema") != TRACE_SCHEMA:
                    raise TrainingRecordError(
                        f"{path}:{line_number}: unexpected trace schema "
                        f"{record.get('schema')!r}"
                    )
                if record.get("schema_version") != TRACE_SCHEMA_VERSION:
                    raise TrainingRecordError(
                        f"{path}:{line_number}: unsupported trace schema version "
                        f"{record.get('schema_version')!r}"
                    )
                if not isinstance(record.get("run_id"), str):
                    raise TrainingRecordError(f"{path}:{line_number}: missing run_id")
                if not isinstance(record.get("event_index"), int):
                    raise TrainingRecordError(
                        f"{path}:{line_number}: missing event_index"
                    )
                records.append(record)
    return records


def _terminal_outcomes(
    trace_paths: Sequence[str | Path],
) -> dict[tuple[str, str], dict[str, Any]]:
    terminals: dict[tuple[str, str], dict[str, Any]] = {}
    for record in _load_trace_records(trace_paths):
        if record.get("kind") != "terminal":
            continue
        battle_tag = record.get("battle_tag")
        if not isinstance(battle_tag, str):
            raise TrainingRecordError("terminal record lacks battle_tag")
        key = (record["run_id"], battle_tag)
        if key in terminals:
            raise TrainingRecordError(
                f"battle {key!r} contains more than one terminal record"
            )

        won = record.get("won") is True
        lost = record.get("lost") is True
        tied = record.get("tied") is True
        if sum((won, lost, tied)) != 1:
            raise TrainingRecordError(
                f"battle {key!r} terminal outcome is not exactly one of win/loss/tie"
            )
        outcome = 1.0 if won else -1.0 if lost else 0.0
        terminals[key] = {
            "outcome": outcome,
            "event_index": record["event_index"],
        }
    return terminals


def _load_json_documents(path: Path) -> list[Mapping[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return []
    try:
        document = json.loads(text)
    except json.JSONDecodeError:
        documents: list[Mapping[str, Any]] = []
        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingRecordError(
                    f"{path}:{line_number}: invalid search annotation JSON: {error.msg}"
                ) from error
            if not isinstance(row, Mapping):
                raise TrainingRecordError(
                    f"{path}:{line_number}: search annotation must be an object"
                )
            documents.append(row)
        return documents

    if isinstance(document, Mapping):
        return [document]
    if isinstance(document, list) and all(isinstance(row, Mapping) for row in document):
        return list(document)
    raise TrainingRecordError(
        f"{path}: search annotation must be an object, a list of objects, or JSONL"
    )


def _numeric_root_values(raw: Any) -> dict[str, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or not raw:
        raise TrainingRecordError("search root_values must be a non-empty object")
    values: dict[str, float] = {}
    for action, value in raw.items():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise TrainingRecordError("search root value must be a finite number")
        values[str(action)] = float(value)
    return values


def _annotation_target(document: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    public = document.get("public_belief")
    if isinstance(public, Mapping):
        fixture_id = document.get("source_fixture_id")
        payload = public
        kind = "public-belief-analysis"
    elif (
        document.get("schema") == SEARCH_ANNOTATION_SCHEMA
        and document.get("schema_version") == SEARCH_ANNOTATION_SCHEMA_VERSION
    ):
        fixture_id = document.get("fixture_id")
        payload = document
        kind = "public-belief-search-annotation"
    else:
        raise TrainingRecordError(
            "search annotation must be a real-belief analysis or "
            f"{SEARCH_ANNOTATION_SCHEMA} v{SEARCH_ANNOTATION_SCHEMA_VERSION}"
        )

    if not isinstance(fixture_id, str) or not fixture_id:
        raise TrainingRecordError("search annotation lacks fixture identity")
    if document.get("experiment_valid") is False:
        raise TrainingRecordError(
            f"search annotation for fixture {fixture_id} is explicitly invalid"
        )

    selected = payload.get("chosen_action", payload.get("selected_action"))
    if not isinstance(selected, str) or not selected:
        raise TrainingRecordError(
            f"search annotation for fixture {fixture_id} lacks chosen_action"
        )
    root_values = _numeric_root_values(
        payload.get("root_values", payload.get("action_values"))
    )
    value = payload.get("value")
    if value is None and root_values is not None:
        value = root_values.get(selected)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise TrainingRecordError(
            f"search annotation for fixture {fixture_id} lacks a finite value"
        )

    source: dict[str, Any] = {
        "kind": kind,
        "schema": document.get("schema"),
        "schema_version": document.get("schema_version"),
    }
    for key in (
        "showdown_commit",
        "continuation_decision_horizons",
        "search_depth",
        "search_budget",
    ):
        if key in document:
            source[key] = document[key]

    return fixture_id, {
        "selected_action": selected,
        "root_values": root_values,
        "value": float(value),
        "source": source,
    }


def _load_search_annotations(
    paths: Sequence[str | Path],
) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        for document in _load_json_documents(Path(raw_path)):
            fixture_id, target = _annotation_target(document)
            existing = targets.get(fixture_id)
            if existing is not None and _canonical_json(existing) != _canonical_json(target):
                raise TrainingRecordError(
                    f"conflicting search annotations for fixture {fixture_id}"
                )
            targets[fixture_id] = target
    return targets


def _live_search_target(control: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = control.get("decision_metadata")
    if not isinstance(metadata, Mapping):
        return None
    belief = metadata.get("belief")
    if not isinstance(belief, Mapping):
        return None
    if belief.get("status") != "selected" or belief.get("reason") != "bounded-public-belief":
        return None
    selected = belief.get("action")
    diagnostics = belief.get("diagnostics")
    if not isinstance(selected, str) or not isinstance(diagnostics, Mapping):
        return None

    value = diagnostics.get("public_belief_value")
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    root_values = _numeric_root_values(
        diagnostics.get("public_belief_root_values")
    )

    source: dict[str, Any] = {
        "kind": "live-public-belief-search",
        "showdown_commit": diagnostics.get("showdown_commit"),
    }
    if diagnostics.get("public_belief_search_horizons") is not None:
        source["continuation_decision_horizons"] = diagnostics[
            "public_belief_search_horizons"
        ]
    return {
        "selected_action": selected,
        "root_values": root_values,
        "value": float(value),
        "source": source,
    }


def _validate_search_target(
    fixture: DecisionFixture,
    target: Mapping[str, Any],
) -> None:
    legal_actions = tuple(fixture.legal_actions)
    selected = target.get("selected_action")
    if selected not in legal_actions:
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: search selected nonlegal action {selected!r}"
        )

    root_values = target.get("root_values")
    if root_values is None:
        return
    if not isinstance(root_values, Mapping):
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: malformed search root values"
        )
    if set(root_values) != set(legal_actions):
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: searched actions do not match legal actions"
        )
    selected_value = float(root_values[selected])
    if not math.isclose(
        selected_value,
        float(target["value"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: chosen search value disagrees with root values"
        )


def _battle_id(run_id: str, battle_tag: str) -> str:
    return _sha256({"run_id": run_id, "battle_tag": battle_tag})


def _split_for_battle(
    battle_id: str,
    *,
    split_seed: str,
    train_fraction: float,
    validation_fraction: float,
) -> str:
    if not 0.0 < train_fraction <= 1.0:
        raise TrainingRecordError("train_fraction must be within (0, 1]")
    if not 0.0 <= validation_fraction <= 1.0:
        raise TrainingRecordError("validation_fraction must be within [0, 1]")
    if train_fraction + validation_fraction > 1.0:
        raise TrainingRecordError(
            "train_fraction + validation_fraction must not exceed 1"
        )
    digest = hashlib.sha256(
        f"{split_seed}\0{battle_id}".encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "test"


def build_training_records(
    trace_paths: Sequence[str | Path],
    *,
    search_annotation_paths: Sequence[str | Path] = (),
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Produce one target-bearing record per searched real decision state.

    Search annotations, when supplied, override the live bounded-search target for
    the same content-addressed fixture. This lets deeper or otherwise stronger
    public-belief search relabel existing real battle states without changing the
    battle/outcome evidence.
    """

    fixtures = build_fixtures(trace_paths)
    terminals = _terminal_outcomes(trace_paths)
    annotations = _load_search_annotations(search_annotation_paths)

    records: list[dict[str, Any]] = []
    source_decisions = 0
    skipped_without_search = 0

    for fixture in fixtures:
        for control in fixture.control_decisions:
            source_decisions += 1
            run_id = control.get("run_id")
            battle_tag = control.get("battle_tag")
            decision_index = control.get("decision_index")
            if (
                not isinstance(run_id, str)
                or not isinstance(battle_tag, str)
                or not isinstance(decision_index, int)
            ):
                raise TrainingRecordError(
                    f"fixture {fixture.fixture_id}: malformed decision provenance"
                )
            terminal = terminals.get((run_id, battle_tag))
            if terminal is None:
                raise TrainingRecordError(
                    f"battle {(run_id, battle_tag)!r} has a decision but no terminal outcome"
                )

            target = annotations.get(fixture.fixture_id)
            if target is None:
                target = _live_search_target(control)
            if target is None:
                skipped_without_search += 1
                continue
            _validate_search_target(fixture, target)

            battle_id = _battle_id(run_id, battle_tag)
            split = _split_for_battle(
                battle_id,
                split_seed=split_seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
            selected_action = str(target["selected_action"])
            policy_distribution = {
                action: 1.0 if action == selected_action else 0.0
                for action in fixture.legal_actions
            }
            behavior_action = control.get("chosen_action")
            record_material = {
                "battle_id": battle_id,
                "fixture_id": fixture.fixture_id,
                "decision_index": decision_index,
            }
            records.append(
                {
                    "schema": TRAINING_SCHEMA,
                    "schema_version": TRAINING_SCHEMA_VERSION,
                    "record_id": _sha256(record_material),
                    "battle_id": battle_id,
                    "decision_index": decision_index,
                    "split": split,
                    "input": {
                        "fixture_id": fixture.fixture_id,
                        "public_state": fixture.state,
                        "protocol_prefix": [
                            [list(message) for message in batch]
                            for batch in fixture.protocol_prefix
                        ],
                    },
                    "targets": {
                        "value": {
                            "eventual_battle_outcome": terminal["outcome"],
                            "public_belief_search_return": target["value"],
                        },
                        "policy": {
                            "selected_action": selected_action,
                            "action_probabilities": policy_distribution,
                            "searched_action_values": target["root_values"],
                        },
                    },
                    "provenance": {
                        "run_id": run_id,
                        "battle_tag": battle_tag,
                        "terminal_event_index": terminal["event_index"],
                        "behavior_action": behavior_action,
                        "behavior_matches_policy_target": (
                            behavior_action == selected_action
                        ),
                        "search": target["source"],
                    },
                }
            )

    records.sort(
        key=lambda row: (
            row["battle_id"],
            row["decision_index"],
            row["record_id"],
        )
    )
    if not records:
        raise TrainingRecordError(
            "no decisions had public-belief search targets; provide stronger "
            "search annotations or record exact public-belief decisions"
        )

    split_records = Counter(row["split"] for row in records)
    battle_split: dict[str, str] = {}
    for row in records:
        previous = battle_split.setdefault(row["battle_id"], row["split"])
        if previous != row["split"]:
            raise AssertionError("one battle was assigned to multiple dataset splits")
    split_battles = Counter(battle_split.values())

    summary = {
        "schema": "azelficoast.training-record-build",
        "schema_version": 1,
        "source_decision_count": source_decisions,
        "record_count": len(records),
        "skipped_without_search_target": skipped_without_search,
        "battle_count": len(battle_split),
        "split_record_counts": dict(sorted(split_records.items())),
        "split_battle_counts": dict(sorted(split_battles.items())),
        "search_target_sources": dict(
            sorted(Counter(row["provenance"]["search"]["kind"] for row in records).items())
        ),
    }
    return records, summary


def write_training_records(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> str:
    """Write deterministic JSONL and refuse a different immutable overwrite."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        _canonical_json(record) + "\n"
        for record in records
    )
    if destination.exists():
        existing = destination.read_text(encoding="utf-8")
        if existing != payload:
            raise TrainingRecordConflictError(
                f"{destination} already contains different training evidence"
            )
    else:
        destination.write_text(payload, encoding="utf-8")
    return _sha256(payload)


def build_training_dataset(
    trace_paths: Sequence[str | Path],
    output: str | Path,
    *,
    search_annotation_paths: Sequence[str | Path] = (),
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    records, summary = build_training_records(
        trace_paths,
        search_annotation_paths=search_annotation_paths,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    digest = write_training_records(records, output)
    return {
        **summary,
        "output": str(output),
        "sha256": digest,
    }
