"""Structural mining over real public-belief transition oracles.

Candidate ranking deliberately ignores policy values and policy disagreement. It uses
only the shape of hidden-state dependence, dependency collapse, and public observation
branching. The policy comparison is inspected only after candidates have been ranked.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.research.verification.real_belief_trace import analyze_oracle


class BeliefMiningError(ValueError):
    """Raised when a mining corpus is empty or malformed."""


@dataclass(frozen=True)
class StructuralSignals:
    hidden_sensitive_actions: int
    collapsing_actions: int
    observation_branching_actions: int
    total_observation_classes: int
    max_observation_classes: int
    world_count: int
    legal_action_count: int

    @property
    def rank_key(self) -> tuple[int, ...]:
        """Higher is structurally more informative, without policy-result leakage."""
        return (
            self.hidden_sensitive_actions,
            self.collapsing_actions,
            self.observation_branching_actions,
            self.total_observation_classes,
            self.max_observation_classes,
            self.world_count,
            self.legal_action_count,
        )


def structural_signals(trace: Mapping[str, Any]) -> StructuralSignals:
    actions = trace.get("actions")
    if not isinstance(actions, list) or not actions:
        raise BeliefMiningError("trace must contain action reports")

    world_count = int(trace.get("world_count", 0))
    legal_action_count = int(trace.get("legal_action_count", 0))
    if world_count < 1 or legal_action_count < 1:
        raise BeliefMiningError("trace must contain positive world and legal-action counts")

    hidden_sensitive_actions = 0
    collapsing_actions = 0
    observation_branching_actions = 0
    total_observation_classes = 0
    max_observation_classes = 0

    for action in actions:
        if not isinstance(action, Mapping):
            raise BeliefMiningError("action report must be an object")
        signature = action.get("dependency_signature")
        if not isinstance(signature, Mapping):
            raise BeliefMiningError("action report is missing dependency signature")

        classes_out = int(signature.get("classes_out", 0))
        observable_classes_out = int(signature.get("observable_classes_out", 0))
        if not 1 <= classes_out <= world_count:
            raise BeliefMiningError("dependency class count is outside belief support")
        if observable_classes_out < 1:
            raise BeliefMiningError("observable class count must be positive")

        if classes_out > 1:
            hidden_sensitive_actions += 1
        if classes_out < world_count:
            collapsing_actions += 1
        if observable_classes_out > 1:
            observation_branching_actions += 1
        total_observation_classes += observable_classes_out
        max_observation_classes = max(max_observation_classes, observable_classes_out)

    return StructuralSignals(
        hidden_sensitive_actions=hidden_sensitive_actions,
        collapsing_actions=collapsing_actions,
        observation_branching_actions=observation_branching_actions,
        total_observation_classes=total_observation_classes,
        max_observation_classes=max_observation_classes,
        world_count=world_count,
        legal_action_count=legal_action_count,
    )


def _candidate(trace: Mapping[str, Any], source: str) -> dict[str, Any]:
    signals = structural_signals(trace)
    return {
        "source": source,
        "source_fixture_id": trace.get("source_fixture_id"),
        "showdown_commit": trace.get("showdown_commit"),
        "signals": {
            "hidden_sensitive_actions": signals.hidden_sensitive_actions,
            "collapsing_actions": signals.collapsing_actions,
            "observation_branching_actions": signals.observation_branching_actions,
            "total_observation_classes": signals.total_observation_classes,
            "max_observation_classes": signals.max_observation_classes,
            "world_count": signals.world_count,
            "legal_action_count": signals.legal_action_count,
        },
        "structural_rank_key": list(signals.rank_key),
        "strategy_fusion_observation_count": int(
            trace.get("strategy_fusion_observation_count", 0)
        ),
        "max_world_aware_choices_per_observation": int(
            trace.get("max_world_aware_choices_per_observation", 0)
        ),
        "policy_disagreement": bool(trace.get("policy_disagreement")),
        "hypothesis_supported": bool(trace.get("hypothesis_supported")),
        "determinization_action": trace.get("determinization", {}).get("chosen_action"),
        "public_belief_action": trace.get("public_belief", {}).get("chosen_action"),
    }


def mine_oracles(documents: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, Any]:
    if not documents:
        raise BeliefMiningError("at least one oracle is required")

    candidates = [
        _candidate(analyze_oracle(document), source)
        for source, document in documents
    ]
    candidates.sort(
        key=lambda row: (
            tuple(-int(value) for value in row["structural_rank_key"]),
            str(row.get("source_fixture_id") or ""),
            row["source"],
        )
    )

    first_strategy_fusion_candidate = next(
        (
            candidate
            for candidate in candidates
            if int(candidate["strategy_fusion_observation_count"]) > 0
        ),
        None,
    )
    first_disagreement = next(
        (candidate for candidate in candidates if candidate["policy_disagreement"]),
        None,
    )
    return {
        "schema": "azelficoast.real-belief-trace-mining",
        "schema_version": 1,
        "ranking_uses_policy_result": False,
        "evaluated_count": len(candidates),
        "strategy_fusion_candidate_count": sum(
            int(candidate["strategy_fusion_observation_count"]) > 0
            for candidate in candidates
        ),
        "disagreement_count": sum(
            bool(candidate["policy_disagreement"]) for candidate in candidates
        ),
        "ranked_candidates": candidates,
        "first_strategy_fusion_candidate": first_strategy_fusion_candidate,
        "first_disagreement": first_disagreement,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracles", nargs="+", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    documents: list[tuple[str, Mapping[str, Any]]] = []
    for path in args.oracles:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise BeliefMiningError(f"{path}: oracle must be an object")
        documents.append((str(path), raw))
    print(json.dumps(mine_oracles(documents), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
