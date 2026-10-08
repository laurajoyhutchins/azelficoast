"""Read-only operational telemetry for a long-running source acquisition."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class TraceProgress:
    """Tail complete trace records without reading actions or terminal outcomes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.pending = b""
        self.decisions = 0
        self.terminals: set[str] = set()
        self.reasons: Counter[str] = Counter()
        self.malformed = 0
        self.last_activity: float | None = None
        self.last_poll = 0.0
        self.previous_decisions = 0
        self.previous_battles = 0

    def poll(self, elapsed_seconds: float) -> dict[str, Any]:
        if self.path.exists():
            with self.path.open("rb") as handle:
                if self.path.stat().st_size < self.offset:
                    raise ValueError("source trace was truncated")
                handle.seek(self.offset)
                added = handle.read()
                self.offset += len(added)
            lines = (self.pending + added).split(b"\n")
            self.pending = lines.pop()
            for line in lines:
                if not line:
                    continue
                self.last_activity = elapsed_seconds
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    self.malformed += 1
                    continue
                if not isinstance(record, dict):
                    self.malformed += 1
                    continue
                if record.get("kind") == "decision":
                    self.decisions += 1
                    metadata = record.get("decision_metadata")
                    belief = metadata.get("belief") if isinstance(metadata, dict) else None
                    reason = belief.get("reason") if isinstance(belief, dict) else None
                    if isinstance(reason, str):
                        self.reasons[reason] += 1
                elif record.get("kind") == "terminal":
                    tag = record.get("battle_tag")
                    if isinstance(tag, str):
                        self.terminals.add(tag)
        interval = elapsed_seconds - self.last_poll
        battles = len(self.terminals)
        result = {
            "schema": "azelficoast.source-generation-progress/v1",
            "elapsed_seconds": elapsed_seconds,
            "decisions": self.decisions,
            "finished_battles": battles,
            "trace_bytes": self.offset,
            "seconds_since_trace_activity": (
                None if self.last_activity is None else elapsed_seconds - self.last_activity
            ),
            "decisions_per_second": (
                (self.decisions - self.previous_decisions) / interval if interval > 0 else 0.0
            ),
            "battles_per_second": (
                (battles - self.previous_battles) / interval if interval > 0 else 0.0
            ),
            "policy_reasons": dict(self.reasons),
            "malformed_records": self.malformed,
        }
        self.last_poll = elapsed_seconds
        self.previous_decisions = self.decisions
        self.previous_battles = battles
        return result


@contextmanager
def observe_generation(
    trace: Path, output: Path, *, interval_seconds: float = 60.0
) -> Iterator[None]:
    """Observe on a separate thread, including a final scan after source failure."""
    progress = TraceProgress(trace)
    stopped = threading.Event()
    started = time.monotonic()

    def publish() -> None:
        record = progress.poll(time.monotonic() - started)
        line = json.dumps(record, sort_keys=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)

    def observe() -> None:
        try:
            publish()
            while not stopped.wait(interval_seconds):
                publish()
            publish()
        except Exception as error:
            # Operational telemetry cannot change the acquisition's decisions.
            print(f"source progress observer failed: {error}", file=sys.stderr, flush=True)

    worker = threading.Thread(target=observe, daemon=True, name="source-progress")
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=5.0)
        if worker.is_alive():
            print("source progress final snapshot is incomplete", file=sys.stderr, flush=True)
