"""Battle-evidence production for evaluator promotion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from azelficoast.belief.battle_promotion import (
    BattlePromotionPolicy,
    CANDIDATE_PRIMARY_MODE,
    INCUMBENT_PRIMARY_MODE,
    settle_battle_panel,
)
from azelficoast.belief.evaluator import BeliefEvaluatorRuntime
from azelficoast.live.battle_runtime import BATTLE_FORMAT, prepare_output_paths, run_local
from azelficoast.live.player import AzelficoastPlayer


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def checkpoint_digest(path: Path) -> str:
    runtime = BeliefEvaluatorRuntime.from_checkpoint(path)
    return str(runtime.identity["checkpoint_digest"])


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


async def run_promotion_panel(
    *,
    root: Path,
    candidate_checkpoint: Path,
    incumbent_checkpoint: Path,
    showdown_root: Path,
    belief_timeout: float,
    search_policy_margin: float,
    concurrency: int,
    policy: BattlePromotionPolicy,
) -> dict[str, Any]:
    """Run and settle a side-balanced candidate-vs-incumbent battle panel."""

    candidate_digest = checkpoint_digest(candidate_checkpoint)
    incumbent_digest = checkpoint_digest(incumbent_checkpoint)
    results = root / "results.jsonl"
    decisions = root / "decisions.jsonl"
    replays = root / "replays"
    evidence_path = root / "evidence.json"
    prepare_output_paths(results, decisions, replays)

    common_metadata = {
        "candidate_checkpoint_digest": candidate_digest,
        "incumbent_checkpoint_digest": incumbent_digest,
    }
    half = policy.expected_battles // 2

    incumbent_opponent = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
        evaluator_checkpoint=incumbent_checkpoint,
        search_policy_margin=search_policy_margin,
    )
    await run_local(
        half,
        concurrency,
        results,
        decisions,
        replays,
        showdown_root=showdown_root,
        belief_timeout=belief_timeout,
        evaluator_checkpoint=candidate_checkpoint,
        search_policy_margin=search_policy_margin,
        opponent=incumbent_opponent,
        trace_source={"kind": "promotion-candidate-primary"},
        mode=CANDIDATE_PRIMARY_MODE,
        print_summary=False,
        result_metadata=common_metadata,
    )

    candidate_opponent = AzelficoastPlayer(
        battle_format=BATTLE_FORMAT,
        max_concurrent_battles=concurrency,
        showdown_root=showdown_root,
        belief_timeout_seconds=belief_timeout,
        evaluator_checkpoint=candidate_checkpoint,
        search_policy_margin=search_policy_margin,
    )
    await run_local(
        half,
        concurrency,
        results,
        decisions,
        replays,
        showdown_root=showdown_root,
        belief_timeout=belief_timeout,
        evaluator_checkpoint=incumbent_checkpoint,
        search_policy_margin=search_policy_margin,
        opponent=candidate_opponent,
        trace_source={"kind": "promotion-incumbent-primary"},
        mode=INCUMBENT_PRIMARY_MODE,
        print_summary=False,
        result_metadata=common_metadata,
    )

    evidence = settle_battle_panel(
        _read_jsonl_objects(results),
        candidate_checkpoint_digest=candidate_digest,
        incumbent_checkpoint_digest=incumbent_digest,
        policy=policy,
    )
    evidence = {
        **evidence,
        "results": str(results),
        "results_digest": file_sha256(results),
        "decision_trace": str(decisions),
        "decision_trace_digest": file_sha256(decisions),
    }
    evidence_path.write_text(
        json.dumps(evidence, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return evidence


