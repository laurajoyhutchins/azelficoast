"""Run one durable hosted training iteration.

The workflow is infrastructure. Training policy lives here so it is versioned, testable,
and shared by manual and scheduled runs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.belief.evaluator import BeliefEvaluatorRuntime
from azelficoast.live.harness import main as harness_main

TRAINING_STATE_SCHEMA = "azelficoast.hosted-training-state"
TRAINING_STATE_SCHEMA_VERSION = 1
TRAINING_RUN_SCHEMA = "azelficoast.hosted-training-run"
TRAINING_RUN_SCHEMA_VERSION = 1

BOOTSTRAP_PAGE_BATTLES = 24
BOOTSTRAP_MAX_PAGES = 4
BOOTSTRAP_MIN_ADMITTED_BATTLES = 12
PUBLIC_REPLAY_MIN_RATING = 1500

AUTO_GENERATIONS = 1
AUTO_BATTLES_PER_GENERATION = 24
AUTO_CONCURRENCY = 4
AUTO_PUBLIC_REPLAYS_PER_GENERATION = 4
AUTO_PROMOTION_BATTLES = 32
AUTO_MAX_TEACHER_FIXTURES = 64


class HostedTrainingError(RuntimeError):
    """Raised when one hosted training iteration cannot establish trustworthy state."""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _invoke(argv: Sequence[str]) -> dict[str, Any]:
    """Invoke the normal CLI surface and recover its final JSON summary."""

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        return_code = harness_main(tuple(argv))
    if return_code != 0:
        raise HostedTrainingError(f"azelficoast command returned {return_code}")

    documents: list[dict[str, Any]] = []
    for line in output.getvalue().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            document = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict):
            documents.append(document)
    if not documents:
        raise HostedTrainingError("azelficoast command emitted no JSON summary")
    return documents[-1]


def _current_identity(promotion: Path) -> dict[str, Any]:
    runtime = BeliefEvaluatorRuntime.from_checkpoint(promotion)
    return dict(runtime.identity)


def _bootstrap(
    *,
    showdown_root: Path,
    state_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    traces: list[str] = []
    batches: list[dict[str, Any]] = []
    before: int | None = None
    admitted_total = 0

    for page in range(BOOTSTRAP_MAX_PAGES):
        public_root = work_root / "public-replays" / f"{page + 1:02d}"
        argv = [
            "--showdown-root",
            str(showdown_root),
            "corpus",
            "import-public",
            "--output-root",
            str(public_root),
            "--max-battles",
            str(BOOTSTRAP_PAGE_BATTLES),
            "--min-rating",
            str(PUBLIC_REPLAY_MIN_RATING),
        ]
        if before is not None:
            argv.extend(("--before", str(before)))
        batch = _invoke(argv)
        batches.append(batch)

        admitted = int(batch.get("admitted_count", 0))
        decisions = int(batch.get("decision_count", 0))
        trace = batch.get("trace")
        if admitted > 0 and decisions > 0 and isinstance(trace, str) and trace:
            traces.append(trace)
            admitted_total += admitted
        if admitted_total >= BOOTSTRAP_MIN_ADMITTED_BATTLES:
            break

        next_before = batch.get("next_before")
        if not isinstance(next_before, int) or isinstance(next_before, bool):
            break
        if before is not None and next_before >= before:
            break
        before = next_before

    if admitted_total < BOOTSTRAP_MIN_ADMITTED_BATTLES or not traces:
        return {
            "status": "not-ready",
            "reason": "insufficient-revision-matched-public-replays",
            "required_admitted_battles": BOOTSTRAP_MIN_ADMITTED_BATTLES,
            "admitted_battles": admitted_total,
            "batches": batches,
        }

    evaluator_root = state_root / "evaluators"
    result = _invoke(
        [
            "--showdown-root",
            str(showdown_root),
            "training",
            "bootstrap-public",
            *traces,
            "--output",
            str(work_root / "public-pretraining.jsonl"),
            "--models-dir",
            str(evaluator_root / "candidates"),
            "--receipts-dir",
            str(evaluator_root / "receipts"),
            "--promotion",
            str(evaluator_root / "current.json"),
        ]
    )
    return {
        "status": "promoted" if result.get("promoted") is True else "rejected",
        "admitted_battles": admitted_total,
        "batches": batches,
        "training": result,
    }


def _automatic_generation(
    *,
    showdown_root: Path,
    state_root: Path,
    work_root: Path,
) -> dict[str, Any]:
    evaluator_root = state_root / "evaluators"
    promotion = evaluator_root / "current.json"
    return _invoke(
        [
            "--showdown-root",
            str(showdown_root),
            "training",
            "auto",
            "--incumbent",
            str(promotion),
            "--workspace",
            str(work_root / "self-improvement"),
            "--models-dir",
            str(evaluator_root / "candidates"),
            "--receipts-dir",
            str(evaluator_root / "receipts"),
            "--promotion",
            str(promotion),
            "--generations",
            str(AUTO_GENERATIONS),
            "--battles-per-generation",
            str(AUTO_BATTLES_PER_GENERATION),
            "--concurrency",
            str(AUTO_CONCURRENCY),
            "--public-replays-per-generation",
            str(AUTO_PUBLIC_REPLAYS_PER_GENERATION),
            "--public-replay-min-rating",
            str(PUBLIC_REPLAY_MIN_RATING),
            "--promotion-battles",
            str(AUTO_PROMOTION_BATTLES),
            "--max-teacher-fixtures",
            str(AUTO_MAX_TEACHER_FIXTURES),
        ]
    )


def run_training_iteration(
    *,
    showdown_root: str | Path,
    state_root: str | Path,
    work_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    """Cold-start if needed, then execute one bounded self-improvement generation."""

    showdown = Path(showdown_root)
    state = Path(state_root)
    work = Path(work_root)
    promotion = state / "evaluators" / "current.json"
    state.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    bootstrap: dict[str, Any] | None = None
    if not promotion.is_file():
        bootstrap = _bootstrap(
            showdown_root=showdown,
            state_root=state,
            work_root=work,
        )

    automatic: dict[str, Any] | None = None
    if promotion.is_file():
        automatic = _automatic_generation(
            showdown_root=showdown,
            state_root=state,
            work_root=work,
        )

    identity = _current_identity(promotion) if promotion.is_file() else None
    status = (
        "trained"
        if automatic is not None
        else "bootstrap-promoted"
        if identity is not None
        else "bootstrap-not-ready"
    )
    policy = {
        "bootstrap_page_battles": BOOTSTRAP_PAGE_BATTLES,
        "bootstrap_max_pages": BOOTSTRAP_MAX_PAGES,
        "bootstrap_min_admitted_battles": BOOTSTRAP_MIN_ADMITTED_BATTLES,
        "public_replay_min_rating": PUBLIC_REPLAY_MIN_RATING,
        "auto_generations": AUTO_GENERATIONS,
        "auto_battles_per_generation": AUTO_BATTLES_PER_GENERATION,
        "auto_concurrency": AUTO_CONCURRENCY,
        "auto_public_replays_per_generation": AUTO_PUBLIC_REPLAYS_PER_GENERATION,
        "auto_promotion_battles": AUTO_PROMOTION_BATTLES,
        "auto_max_teacher_fixtures": AUTO_MAX_TEACHER_FIXTURES,
    }
    summary = {
        "schema": TRAINING_RUN_SCHEMA,
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "status": status,
        "policy": policy,
        "bootstrap": bootstrap,
        "automatic": automatic,
        "current_evaluator": identity,
    }
    _write_json(work / "run-summary.json", summary)

    state_projection = {
        "schema": TRAINING_STATE_SCHEMA,
        "schema_version": TRAINING_STATE_SCHEMA_VERSION,
        "last_run_id": run_id,
        "last_run_status": status,
        "current_evaluator": identity,
        "policy": policy,
    }
    _write_json(state / "state.json", state_projection)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Azelficoast hosted training iteration.")
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary = run_training_iteration(
        showdown_root=args.showdown_root,
        state_root=args.state_root,
        work_root=args.work_root,
        run_id=args.run_id,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
