"""Persistent line-protocol client for live Pokémon Showdown probes.

The worker keeps Node and the pinned Showdown module graph warm across decisions. A
posterior request also leaves its reconstructed JavaScript session resident long enough
for an uncertain learned decision to compile the matching TransitionProgram without
reconstructing the posterior or reloading Showdown.
"""

from __future__ import annotations

import hashlib
import json
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import Any, Mapping, TextIO


class ShowdownProbeRuntimeError(RuntimeError):
    """Raised when the persistent Showdown worker violates its runtime contract."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def probe_session_key(source: Mapping[str, Any]) -> str:
    """Return the content-addressed identity used to resume one posterior context."""

    return hashlib.sha256(_canonical_json(source).encode("utf-8")).hexdigest()


class PersistentShowdownProbe:
    """One lazily started Node worker shared by successive live decisions."""

    def __init__(self, showdown_root: str | Path) -> None:
        self.showdown_root = Path(showdown_root)
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=32)
        self._lock = threading.Lock()
        self._request_id = 0

    def _worker_script(self) -> Path:
        return Path(__file__).resolve().parents[3] / "showdown" / "runtime" / "probe_real_belief_worker.cjs"

    @staticmethod
    def _drain_stdout(stream: TextIO, responses: queue.Queue[str | None]) -> None:
        try:
            for line in stream:
                responses.put(line)
        finally:
            responses.put(None)

    @staticmethod
    def _drain_stderr(stream: TextIO, lines: deque[str]) -> None:
        for line in stream:
            lines.append(line.rstrip())

    def _start_unlocked(self) -> subprocess.Popen[str]:
        process = self._process
        if process is not None and process.poll() is None:
            return process

        self._responses = queue.Queue()
        self._stderr.clear()
        process = subprocess.Popen(
            [
                "node",
                str(self._worker_script()),
                str(self.showdown_root),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise ShowdownProbeRuntimeError("Showdown probe worker pipes are unavailable")

        self._process = process
        threading.Thread(
            target=self._drain_stdout,
            args=(process.stdout, self._responses),
            daemon=True,
            name="azelficoast-showdown-probe-stdout",
        ).start()
        threading.Thread(
            target=self._drain_stderr,
            args=(process.stderr, self._stderr),
            daemon=True,
            name="azelficoast-showdown-probe-stderr",
        ).start()
        return process

    def _stop_unlocked(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)

    def close(self) -> None:
        """Stop the worker and discard any resumable posterior sessions."""

        with self._lock:
            self._stop_unlocked()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _request(
        self,
        *,
        op: str,
        session: str,
        timeout_seconds: float,
        source: Mapping[str, Any] | None = None,
        cache_mode: str | None = None,
    ) -> Mapping[str, Any]:
        if timeout_seconds <= 0:
            raise ValueError("probe timeout must be positive")

        with self._lock:
            process = self._start_unlocked()
            assert process.stdin is not None
            request_id = self._request_id
            self._request_id += 1
            payload: dict[str, Any] = {
                "id": request_id,
                "op": op,
                "session": session,
            }
            if source is not None:
                payload["source"] = dict(source)
            if cache_mode is not None:
                payload["cache_mode"] = cache_mode

            try:
                process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
                process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                detail = "\n".join(self._stderr)
                self._stop_unlocked()
                raise ShowdownProbeRuntimeError(
                    f"Showdown probe worker write failed: {detail or error}"
                ) from error

            try:
                line = self._responses.get(timeout=timeout_seconds)
            except queue.Empty as error:
                self._stop_unlocked()
                raise subprocess.TimeoutExpired(
                    cmd=["node", str(self._worker_script())],
                    timeout=timeout_seconds,
                ) from error

            if line is None:
                detail = "\n".join(self._stderr)
                returncode = process.poll()
                self._stop_unlocked()
                raise ShowdownProbeRuntimeError(
                    "Showdown probe worker exited unexpectedly"
                    + (f" with status {returncode}" if returncode is not None else "")
                    + (f": {detail}" if detail else "")
                )

            try:
                response = json.loads(line)
            except json.JSONDecodeError as error:
                self._stop_unlocked()
                raise ShowdownProbeRuntimeError(
                    "Showdown probe worker returned invalid JSON"
                ) from error

            if not isinstance(response, Mapping) or response.get("id") != request_id:
                self._stop_unlocked()
                raise ShowdownProbeRuntimeError(
                    "Showdown probe worker response identity mismatch"
                )
            if response.get("ok") is not True:
                raise ShowdownProbeRuntimeError(
                    str(response.get("error") or "Showdown probe worker request failed")
                )
            return response

    def posterior(
        self,
        source: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        """Reconstruct a posterior and retain its exact JS context for optional search."""

        session = probe_session_key(source)
        response = self._request(
            op="posterior",
            session=session,
            source=source,
            timeout_seconds=timeout_seconds,
        )
        document = response.get("document")
        if not isinstance(document, Mapping):
            raise ShowdownProbeRuntimeError("posterior worker response lacks a document")
        return document

    def transition_program(
        self,
        source: Mapping[str, Any],
        *,
        timeout_seconds: float,
        cache_mode: str = "projection",
    ) -> Mapping[str, Any]:
        """Compile the TransitionProgram from the retained posterior session."""

        session = probe_session_key(source)
        response = self._request(
            op="transition_program",
            session=session,
            timeout_seconds=timeout_seconds,
            cache_mode=cache_mode,
        )
        document = response.get("document")
        if not isinstance(document, Mapping):
            raise ShowdownProbeRuntimeError(
                "transition-program worker response lacks a document"
            )
        return document

    def release(
        self,
        source: Mapping[str, Any],
        *,
        timeout_seconds: float,
    ) -> None:
        """Release a high-confidence posterior context that will not be searched."""

        session = probe_session_key(source)
        self._request(
            op="release",
            session=session,
            timeout_seconds=timeout_seconds,
        )
