"""Build and evaluate immutable decision fixtures from battle traces."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

from azelficoast.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION

CORPUS_SCHEMA = "azelficoast.decision-corpus"
CORPUS_SCHEMA_VERSION = 1


class CorpusError(ValueError):
    """Raised when trace or corpus evidence violates the expected contract."""


class CorpusConflictError(CorpusError):
    """Raised when an immutable corpus path already contains different evidence."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionFixture:
    """One immutable information state against which decision algorithms can run."""

    fixture_id: str
    state: Mapping[str, Any]
    protocol_prefix: tuple[tuple[tuple[str, ...], ...], ...]
    control_decisions: tuple[Mapping[str, Any], ...]

    @property
    def legal_actions(self) -> tuple[str, ...]:
        actions = self.state.get("legal_actions", ())
        return tuple(str(action) for action in actions)

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": CORPUS_SCHEMA,
            "schema_version": CORPUS_SCHEMA_VERSION,
            "kind": "fixture",
            "fixture_id": self.fixture_id,
            "state": self.state,
            "protocol_prefix": [
                [list(message) for message in batch] for batch in self.protocol_prefix
            ],
            "control_decisions": list(self.control_decisions),
        }


class FixturePolicy(Protocol):
    """A deterministic decision algorithm over a frozen information state."""

    name: str

    def choose(self, fixture: DecisionFixture) -> str:
        """Return exactly one action from fixture.legal_actions."""


class FirstLegalPolicy:
    """Minimal deterministic control that selects the first legal action."""

    name = "first-legal"

    def choose(self, fixture: DecisionFixture) -> str:
        if not fixture.legal_actions:
            raise CorpusError(f"fixture {fixture.fixture_id} has no legal actions")
        return fixture.legal_actions[0]


class RecordedModePolicy:
    """Replay the most frequent recorded control action for a fixture."""

    name = "recorded-mode"

    def choose(self, fixture: DecisionFixture) -> str:
        actions = [
            str(control["chosen_action"])
            for control in fixture.control_decisions
            if "chosen_action" in control
        ]
        if not actions:
            raise CorpusError(f"fixture {fixture.fixture_id} has no control decisions")
        counts = Counter(actions)
        return min(
            (action for action, count in counts.items() if count == max(counts.values())),
            default="",
        )


BUILTIN_POLICIES: dict[str, FixturePolicy] = {
    policy.name: policy for policy in (FirstLegalPolicy(), RecordedModePolicy())
}


def _load_trace_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CorpusError(
                        f"{path}:{line_number}: invalid JSON: {error.msg}"
                    ) from error
                if record.get("schema") != TRACE_SCHEMA:
                    raise CorpusError(
                        f"{path}:{line_number}: unexpected trace schema "
                        f"{record.get('schema')!r}"
                    )
                if record.get("schema_version") != TRACE_SCHEMA_VERSION:
                    raise CorpusError(
                        f"{path}:{line_number}: unsupported trace schema version "
                        f"{record.get('schema_version')!r}"
                    )
                if not isinstance(record.get("run_id"), str):
                    raise CorpusError(f"{path}:{line_number}: missing run_id")
                if not isinstance(record.get("event_index"), int):
                    raise CorpusError(f"{path}:{line_number}: missing event_index")
                record["_source_path"] = str(path)
                record["_source_line"] = line_number
                records.append(record)
    return records


def _fixture_material(
    state: Mapping[str, Any],
    protocol_prefix: Sequence[Sequence[Sequence[str]]],
) -> dict[str, Any]:
    return {
        "state": state,
        "protocol_prefix": protocol_prefix,
    }


def _fixture_id(
    state: Mapping[str, Any],
    protocol_prefix: Sequence[Sequence[Sequence[str]]],
) -> str:
    return _sha256_text(_canonical_json(_fixture_material(state, protocol_prefix)))


def build_fixtures(trace_paths: Sequence[str | Path]) -> list[DecisionFixture]:
    """Extract deterministic decision fixtures from one or more trace JSONL files."""
    paths = [Path(path) for path in trace_paths]
    if not paths:
        raise CorpusError("at least one trace path is required")

    records = _load_trace_records(paths)
    by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_run[record["run_id"]].append(record)

    fixtures: dict[str, dict[str, Any]] = {}

    for run_id in sorted(by_run):
        run_records = sorted(
            by_run[run_id],
            key=lambda record: (
                record["event_index"],
                record["_source_path"],
                record["_source_line"],
            ),
        )
        seen_event_indexes: set[int] = set()
        history: dict[str, list[list[list[str]]]] = defaultdict(list)

        for record in run_records:
            event_index = record["event_index"]
            if event_index in seen_event_indexes:
                raise CorpusError(
                    f"run {run_id!r} contains duplicate event_index {event_index}"
                )
            seen_event_indexes.add(event_index)

            kind = record.get("kind")
            if kind == "protocol":
                room = record.get("room")
                messages = record.get("messages")
                if not isinstance(room, str) or not isinstance(messages, list):
                    raise CorpusError(
                        f"run {run_id!r} event {event_index}: malformed protocol record"
                    )
                history[room].append(messages)
                continue

            if kind != "decision":
                continue

            battle_tag = record.get("battle_tag")
            state = record.get("state")
            chosen_action = record.get("chosen_action")
            decision_index = record.get("decision_index")
            if not isinstance(battle_tag, str) or not isinstance(state, dict):
                raise CorpusError(
                    f"run {run_id!r} event {event_index}: malformed decision record"
                )
            if not isinstance(chosen_action, str) or not isinstance(decision_index, int):
                raise CorpusError(
                    f"run {run_id!r} event {event_index}: incomplete decision evidence"
                )

            legal_actions = state.get("legal_actions")
            if not isinstance(legal_actions, list) or not all(
                isinstance(action, str) for action in legal_actions
            ):
                raise CorpusError(
                    f"run {run_id!r} event {event_index}: state lacks legal_actions"
                )
            if chosen_action not in legal_actions:
                raise CorpusError(
                    f"run {run_id!r} event {event_index}: chosen action is not legal"
                )

            protocol_prefix = history.get(battle_tag, [])
            fixture_id = _fixture_id(state, protocol_prefix)
            control = {
                "run_id": run_id,
                "battle_tag": battle_tag,
                "decision_index": decision_index,
                "event_index": event_index,
                "chosen_action": chosen_action,
            }

            if fixture_id not in fixtures:
                fixtures[fixture_id] = {
                    "state": state,
                    "protocol_prefix": protocol_prefix,
                    "controls": [],
                }
            fixtures[fixture_id]["controls"].append(control)

    built: list[DecisionFixture] = []
    for fixture_id in sorted(fixtures):
        raw = fixtures[fixture_id]
        controls = tuple(
            sorted(
                raw["controls"],
                key=lambda control: (
                    control["run_id"],
                    control["event_index"],
                    control["chosen_action"],
                ),
            )
        )
        protocol_prefix = tuple(
            tuple(tuple(str(field) for field in message) for message in batch)
            for batch in raw["protocol_prefix"]
        )
        built.append(
            DecisionFixture(
                fixture_id=fixture_id,
                state=raw["state"],
                protocol_prefix=protocol_prefix,
                control_decisions=controls,
            )
        )
    return built


def _corpus_bytes(fixtures: Sequence[DecisionFixture]) -> bytes:
    fixture_lines = [
        _canonical_json(fixture.as_record()) + "\n"
        for fixture in sorted(fixtures, key=lambda item: item.fixture_id)
    ]
    fixture_blob = "".join(fixture_lines)
    manifest = {
        "schema": CORPUS_SCHEMA,
        "schema_version": CORPUS_SCHEMA_VERSION,
        "kind": "manifest",
        "fixture_count": len(fixture_lines),
        "fixtures_sha256": _sha256_text(fixture_blob),
    }
    return (_canonical_json(manifest) + "\n" + fixture_blob).encode("utf-8")


def write_corpus(fixtures: Sequence[DecisionFixture], output: str | Path) -> str:
    """Write deterministic corpus bytes and refuse to mutate different evidence."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _corpus_bytes(fixtures)
    digest = hashlib.sha256(content).hexdigest()

    if path.exists():
        existing = path.read_bytes()
        if existing != content:
            raise CorpusConflictError(
                f"{path} already exists with different corpus evidence"
            )
        return digest

    path.write_bytes(content)
    return digest


def build_corpus(trace_paths: Sequence[str | Path], output: str | Path) -> dict[str, Any]:
    fixtures = build_fixtures(trace_paths)
    digest = write_corpus(fixtures, output)
    return {
        "schema": CORPUS_SCHEMA,
        "schema_version": CORPUS_SCHEMA_VERSION,
        "fixture_count": len(fixtures),
        "corpus_sha256": digest,
        "output": str(output),
    }


def load_corpus(path: str | Path) -> list[DecisionFixture]:
    corpus_path = Path(path)
    lines = [line for line in corpus_path.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise CorpusError(f"{corpus_path} is empty")

    manifest = json.loads(lines[0])
    if manifest.get("schema") != CORPUS_SCHEMA or manifest.get("kind") != "manifest":
        raise CorpusError(f"{corpus_path}: invalid corpus manifest")
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise CorpusError(f"{corpus_path}: unsupported corpus schema version")

    fixture_blob = "".join(line + "\n" for line in lines[1:])
    if manifest.get("fixtures_sha256") != _sha256_text(fixture_blob):
        raise CorpusError(f"{corpus_path}: fixture digest mismatch")
    if manifest.get("fixture_count") != len(lines) - 1:
        raise CorpusError(f"{corpus_path}: fixture count mismatch")

    fixtures: list[DecisionFixture] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines[1:], start=2):
        record = json.loads(line)
        if (
            record.get("schema") != CORPUS_SCHEMA
            or record.get("schema_version") != CORPUS_SCHEMA_VERSION
            or record.get("kind") != "fixture"
        ):
            raise CorpusError(f"{corpus_path}:{line_number}: invalid fixture record")
        fixture_id = record.get("fixture_id")
        state = record.get("state")
        prefix = record.get("protocol_prefix")
        controls = record.get("control_decisions")
        if (
            not isinstance(fixture_id, str)
            or not isinstance(state, dict)
            or not isinstance(prefix, list)
            or not isinstance(controls, list)
        ):
            raise CorpusError(f"{corpus_path}:{line_number}: malformed fixture")

        expected_id = _fixture_id(state, prefix)
        if fixture_id != expected_id:
            raise CorpusError(f"{corpus_path}:{line_number}: fixture identity mismatch")
        if fixture_id in seen_ids:
            raise CorpusError(f"{corpus_path}:{line_number}: duplicate fixture_id")
        seen_ids.add(fixture_id)

        fixtures.append(
            DecisionFixture(
                fixture_id=fixture_id,
                state=state,
                protocol_prefix=tuple(
                    tuple(tuple(str(field) for field in message) for message in batch)
                    for batch in prefix
                ),
                control_decisions=tuple(controls),
            )
        )
    return fixtures


def evaluate_fixtures(
    fixtures: Iterable[DecisionFixture],
    policy: FixturePolicy,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one policy against frozen fixtures without mutating the corpus."""
    rows: list[dict[str, Any]] = []
    comparisons = 0
    matches = 0
    illegal = 0

    for fixture in fixtures:
        action = policy.choose(fixture)
        legal = action in fixture.legal_actions
        if not legal:
            illegal += 1

        controls = [
            str(control["chosen_action"])
            for control in fixture.control_decisions
            if "chosen_action" in control
        ]
        fixture_matches = sum(control == action for control in controls)
        comparisons += len(controls)
        matches += fixture_matches
        rows.append(
            {
                "schema": "azelficoast.decision-evaluation",
                "schema_version": 1,
                "fixture_id": fixture.fixture_id,
                "policy": policy.name,
                "chosen_action": action,
                "legal": legal,
                "control_actions": controls,
                "control_matches": fixture_matches,
                "control_comparisons": len(controls),
            }
        )

    summary = {
        "policy": policy.name,
        "fixture_count": len(rows),
        "illegal_action_count": illegal,
        "control_comparisons": comparisons,
        "control_matches": matches,
        "control_action_agreement": (matches / comparisons) if comparisons else None,
    }
    return summary, rows


def evaluate_corpus(
    corpus_path: str | Path,
    policy_name: str,
    output: str | Path | None = None,
) -> dict[str, Any]:
    if policy_name not in BUILTIN_POLICIES:
        raise CorpusError(
            f"unknown policy {policy_name!r}; choose from {', '.join(sorted(BUILTIN_POLICIES))}"
        )
    fixtures = load_corpus(corpus_path)
    summary, rows = evaluate_fixtures(fixtures, BUILTIN_POLICIES[policy_name])

    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = "".join(_canonical_json(row) + "\n" for row in rows)
        output_path.write_text(content, encoding="utf-8")
        summary["output"] = str(output_path)
        summary["evaluation_sha256"] = _sha256_text(content)

    return summary
