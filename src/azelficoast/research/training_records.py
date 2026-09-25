"""Build policy/value training records from settled matched-search evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.corpus import DecisionFixture, build_fixtures
from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION
from azelficoast.research.matched_comparison import (
    PACKET_SCHEMA,
    PACKET_SCHEMA_VERSION,
    MatchedComparisonError,
    _sha256 as matched_digest,
    settle_packet,
)

TRAINING_SCHEMA = "azelficoast.training-record"
TRAINING_SCHEMA_VERSION = 2
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


def _load_documents(paths: Sequence[str | Path], *, kind: str) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, Mapping):
            documents.append(dict(parsed))
            continue
        if isinstance(parsed, list) and all(isinstance(row, Mapping) for row in parsed):
            documents.extend(dict(row) for row in parsed)
            continue

        for line_number, raw_line in enumerate(text.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise TrainingRecordError(
                    f"{path}:{line_number}: invalid {kind} JSON: {error.msg}"
                ) from error
            if not isinstance(row, Mapping):
                raise TrainingRecordError(
                    f"{path}:{line_number}: {kind} must be an object"
                )
            documents.append(dict(row))
    return documents


def _load_trace_records(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    records = _load_documents(paths, kind="trace record")
    for record in records:
        if record.get("schema") != TRACE_SCHEMA:
            raise TrainingRecordError(
                f"unexpected trace schema {record.get('schema')!r}"
            )
        if record.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise TrainingRecordError(
                f"unsupported trace schema version {record.get('schema_version')!r}"
            )
        if not isinstance(record.get("run_id"), str):
            raise TrainingRecordError("trace record is missing run_id")
        if not isinstance(record.get("event_index"), int):
            raise TrainingRecordError("trace record is missing event_index")
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
        terminals[key] = {
            "outcome": 1.0 if won else -1.0 if lost else 0.0,
            "event_index": record["event_index"],
        }
    return terminals


def _load_posteriors(
    paths: Sequence[str | Path],
) -> dict[str, dict[str, Any]]:
    posteriors: dict[str, dict[str, Any]] = {}
    for posterior in _load_documents(paths, kind="posterior"):
        if posterior.get("conditioned_on_public_history") is not True:
            raise TrainingRecordError(
                "training posterior must be conditioned on public history"
            )
        if posterior.get("realized_hidden_state_revealed") is not False:
            raise TrainingRecordError(
                "training posterior may not reveal the realized hidden state"
            )
        worlds = posterior.get("worlds")
        if not isinstance(worlds, list) or not worlds:
            raise TrainingRecordError("training posterior has no hidden-world support")
        digest = matched_digest(posterior)
        existing = posteriors.get(digest)
        if existing is not None and _canonical_json(existing) != _canonical_json(posterior):
            raise TrainingRecordError(f"posterior digest collision for {digest}")
        posteriors[digest] = posterior
    return posteriors


def _load_settled_targets(
    packet_paths: Sequence[str | Path],
    receipt_paths: Sequence[str | Path],
    posterior_paths: Sequence[str | Path],
) -> dict[tuple[str, str], dict[str, Any]]:
    packets = _load_documents(packet_paths, kind="matched-search packet")
    receipts = _load_documents(receipt_paths, kind="matched-search receipt")
    posteriors = _load_posteriors(posterior_paths)
    if not packets:
        raise TrainingRecordError("at least one matched-search packet is required")
    if not receipts:
        raise TrainingRecordError("matched-search receipts are required")
    if not posteriors:
        raise TrainingRecordError("matched-search posterior artifacts are required")

    packet_digests = {
        str(packet.get("packet_digest"))
        for packet in packets
        if isinstance(packet.get("packet_digest"), str)
        and packet.get("packet_digest")
    }
    if len(packet_digests) != len(packets):
        raise TrainingRecordError(
            "matched-search packets must have unique packet_digest values"
        )

    receipts_by_packet: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for receipt in receipts:
        packet_digest = receipt.get("packet_digest")
        if not isinstance(packet_digest, str) or not packet_digest:
            raise TrainingRecordError("search receipt lacks packet_digest")
        if packet_digest not in packet_digests:
            raise TrainingRecordError(
                f"search receipt references packet not supplied to training: "
                f"{packet_digest}"
            )
        receipts_by_packet[packet_digest].append(receipt)

    targets: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for packet in packets:
        if (
            packet.get("schema") != PACKET_SCHEMA
            or packet.get("schema_version") != PACKET_SCHEMA_VERSION
        ):
            raise TrainingRecordError("unexpected matched-search packet schema")
        packet_digest = packet.get("packet_digest")
        if not isinstance(packet_digest, str) or not packet_digest:
            raise TrainingRecordError("matched-search packet lacks packet_digest")
        packet_receipts = receipts_by_packet.pop(packet_digest, [])
        try:
            settled = settle_packet(packet=packet, receipts=packet_receipts)
        except MatchedComparisonError as error:
            raise TrainingRecordError(
                f"cannot settle search packet {packet_digest}: {error}"
            ) from error

        posterior_digest = packet.get("posterior_digest")
        posterior = posteriors.get(str(posterior_digest))
        if posterior is None:
            raise TrainingRecordError(
                f"packet {packet_digest} has no posterior matching "
                f"{posterior_digest!r}"
            )
        if posterior.get("treatment") != packet.get("posterior_treatment"):
            raise TrainingRecordError(
                f"packet {packet_digest}: posterior treatment drifted"
            )

        fixture_id = packet.get("fixture_id")
        battle_tag = packet.get("battle_tag")
        run_id = packet.get("run_id")
        if not isinstance(fixture_id, str) or not isinstance(battle_tag, str):
            raise TrainingRecordError("settled search target lacks fixture/battle identity")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise TrainingRecordError("settled search target has malformed run identity")
        key = (fixture_id, battle_tag, run_id if isinstance(run_id, str) else None)
        if key in targets:
            raise TrainingRecordError(
                f"multiple settled search targets for fixture/battle {key!r}; "
                "select one authoritative teacher search"
            )

        root_values = settled.get("root_values", {}).get("information_set")
        chosen_action = settled.get("chosen_actions", {}).get("information_set")
        if not isinstance(root_values, Mapping) or not root_values:
            raise TrainingRecordError(
                f"packet {packet_digest}: settlement lacks information-set root values"
            )
        numeric_values: dict[str, float] = {}
        for action, value in root_values.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise TrainingRecordError(
                    f"packet {packet_digest}: non-finite information-set root value"
                )
            numeric_values[str(action)] = float(value)
        if chosen_action not in numeric_values:
            raise TrainingRecordError(
                f"packet {packet_digest}: settlement chose a non-evaluated action"
            )

        targets[key] = {
            "packet": packet,
            "settled": settled,
            "posterior": posterior,
            "selected_action": str(chosen_action),
            "root_values": numeric_values,
            "value": numeric_values[str(chosen_action)],
        }

    return targets


def _validate_target_for_fixture(
    fixture: DecisionFixture,
    run_id: str,
    battle_tag: str,
    target: Mapping[str, Any],
) -> None:
    packet = target["packet"]
    legal_actions = list(fixture.legal_actions)
    if packet.get("fixture_id") != fixture.fixture_id:
        raise TrainingRecordError("search packet fixture identity drifted")
    if packet.get("battle_tag") != battle_tag:
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: search packet belongs to another battle"
        )
    packet_run_id = packet.get("run_id")
    if packet_run_id is not None and packet_run_id != run_id:
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: search packet belongs to another run"
        )
    if packet.get("legal_actions") != legal_actions:
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: search/legal actions drifted"
        )
    state_digest = matched_digest(
        {
            "fixture_id": fixture.fixture_id,
            "battle_tag": battle_tag,
            "public_state": fixture.state,
            "legal_actions": legal_actions,
        }
    )
    if packet.get("state_digest") != state_digest:
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: search packet used another public state"
        )
    if set(target["root_values"]) != set(legal_actions):
        raise TrainingRecordError(
            f"fixture {fixture.fixture_id}: searched actions do not match legal actions"
        )


def _battle_id(run_id: str, battle_tag: str) -> str:
    return _sha256({"run_id": run_id, "battle_tag": battle_tag})


def _split_for_group(
    split_group_id: str,
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
        f"{split_seed}\0{split_group_id}".encode("utf-8")
    ).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < train_fraction:
        return "train"
    if unit < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _assign_leakage_safe_splits(
    rows: list[dict[str, Any]],
    *,
    split_seed: str,
    train_fraction: float,
    validation_fraction: float,
) -> None:
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    battles_by_fixture: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        battle_id = row["_battle_id"]
        find(battle_id)
        battles_by_fixture[row["_fixture_id"]].append(battle_id)

    for battle_ids in battles_by_fixture.values():
        first = battle_ids[0]
        for battle_id in battle_ids[1:]:
            union(first, battle_id)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for battle_id in sorted(parent):
        members_by_root[find(battle_id)].append(battle_id)

    group_by_battle: dict[str, tuple[str, str]] = {}
    for members in members_by_root.values():
        group_id = _sha256({"battle_ids": sorted(members)})
        split = _split_for_group(
            group_id,
            split_seed=split_seed,
            train_fraction=train_fraction,
            validation_fraction=validation_fraction,
        )
        for battle_id in members:
            group_by_battle[battle_id] = (group_id, split)

    for row in rows:
        group_id, split = group_by_battle[row.pop("_battle_id")]
        row.pop("_fixture_id")
        row["split_group_id"] = group_id
        row["split"] = split


def build_training_records(
    trace_paths: Sequence[str | Path],
    *,
    search_packet_paths: Sequence[str | Path],
    search_receipt_paths: Sequence[str | Path],
    posterior_paths: Sequence[str | Path],
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join real decisions to independently settled information-set search targets."""

    fixtures = build_fixtures(trace_paths)
    terminals = _terminal_outcomes(trace_paths)
    targets = _load_settled_targets(
        search_packet_paths,
        search_receipt_paths,
        posterior_paths,
    )

    rows: list[dict[str, Any]] = []
    source_decisions = 0
    skipped_without_search = 0
    used_targets: set[tuple[str, str, str | None]] = set()

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

            target_key = (fixture.fixture_id, battle_tag, run_id)
            target = targets.get(target_key)
            if target is None:
                legacy_key = (fixture.fixture_id, battle_tag, None)
                target = targets.get(legacy_key)
                if target is None:
                    skipped_without_search += 1
                    continue
                target_key = legacy_key
            _validate_target_for_fixture(fixture, run_id, battle_tag, target)
            used_targets.add(target_key)

            selected_action = target["selected_action"]
            policy_distribution = {
                action: 1.0 if action == selected_action else 0.0
                for action in fixture.legal_actions
            }
            battle_id = _battle_id(run_id, battle_tag)
            packet = target["packet"]
            settled = target["settled"]
            posterior = target["posterior"]
            record_material = {
                "battle_id": battle_id,
                "fixture_id": fixture.fixture_id,
                "decision_index": decision_index,
                "packet_digest": packet["packet_digest"],
            }
            behavior_action = control.get("chosen_action")
            rows.append(
                {
                    "schema": TRAINING_SCHEMA,
                    "schema_version": TRAINING_SCHEMA_VERSION,
                    "record_id": _sha256(record_material),
                    "battle_id": battle_id,
                    "decision_index": decision_index,
                    "_battle_id": battle_id,
                    "_fixture_id": fixture.fixture_id,
                    "input": {
                        "fixture_id": fixture.fixture_id,
                        "public_state": fixture.state,
                        "protocol_prefix": [
                            [list(message) for message in batch]
                            for batch in fixture.protocol_prefix
                        ],
                        "posterior": posterior,
                        "posterior_digest": packet["posterior_digest"],
                        "posterior_treatment": packet["posterior_treatment"],
                        "legal_actions": list(fixture.legal_actions),
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
                        "behavior_matches_policy_target": behavior_action == selected_action,
                        "search": {
                            "kind": "settled-matched-information-set-search",
                            "packet_digest": packet["packet_digest"],
                            "input_digest": packet["input_digest"],
                            "state_digest": packet["state_digest"],
                            "showdown_commit": packet["showdown_commit"],
                            "depth": packet["depth"],
                            "opponent_model": packet["opponent_model"],
                            "compute_budget": packet["compute_budget"],
                            "compute_consumed": settled["compute_consumed"][
                                "information_set"
                            ],
                            "evaluator": settled["evaluator"],
                            "evaluator_digest": settled["evaluator_digest"],
                            "evaluator_checkpoint_digest": settled[
                                "evaluator_checkpoint_digest"
                            ],
                            "evaluator_calls": settled["evaluator_calls"][
                                "information_set"
                            ],
                        },
                    },
                }
            )

    unused = set(targets) - used_targets
    if unused:
        raise TrainingRecordError(
            "settled search targets do not correspond to supplied real decisions: "
            f"{sorted(unused)[:3]!r}"
        )
    if not rows:
        raise TrainingRecordError(
            "no real decisions had settled matched information-set search targets"
        )

    _assign_leakage_safe_splits(
        rows,
        split_seed=split_seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    rows.sort(
        key=lambda row: (
            row["battle_id"],
            row["decision_index"],
            row["record_id"],
        )
    )

    split_records = Counter(row["split"] for row in rows)
    battle_split: dict[str, str] = {}
    fixture_split: dict[str, str] = {}
    split_groups: set[str] = set()
    for row in rows:
        split_groups.add(row["split_group_id"])
        battle_previous = battle_split.setdefault(row["battle_id"], row["split"])
        if battle_previous != row["split"]:
            raise AssertionError("one battle was assigned to multiple dataset splits")
        fixture_id = row["input"]["fixture_id"]
        fixture_previous = fixture_split.setdefault(fixture_id, row["split"])
        if fixture_previous != row["split"]:
            raise AssertionError("one fixture was assigned to multiple dataset splits")

    summary = {
        "schema": "azelficoast.training-record-build",
        "schema_version": 2,
        "source_decision_count": source_decisions,
        "record_count": len(rows),
        "skipped_without_search_target": skipped_without_search,
        "battle_count": len(battle_split),
        "fixture_count": len(fixture_split),
        "split_group_count": len(split_groups),
        "split_record_counts": dict(sorted(split_records.items())),
        "split_battle_counts": dict(
            sorted(Counter(
                next(
                    row["split"]
                    for row in rows
                    if row["battle_id"] == battle_id
                )
                for battle_id in battle_split
            ).items())
        ),
        "search_target_source": "settled-matched-information-set-search",
    }
    return rows, summary


def write_training_records(
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
) -> str:
    """Write deterministic JSONL and refuse a different immutable overwrite."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(_canonical_json(record) + "\n" for record in records)
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
    search_packet_paths: Sequence[str | Path],
    search_receipt_paths: Sequence[str | Path],
    posterior_paths: Sequence[str | Path],
    split_seed: str = DEFAULT_SPLIT_SEED,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> dict[str, Any]:
    records, summary = build_training_records(
        trace_paths,
        search_packet_paths=search_packet_paths,
        search_receipt_paths=search_receipt_paths,
        posterior_paths=posterior_paths,
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
