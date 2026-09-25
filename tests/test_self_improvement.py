from __future__ import annotations

import copy
import json

import pytest

import azelficoast.belief.self_improvement as self_improvement
from azelficoast.belief.evaluator import BeliefEvaluatorSpec, BeliefPrediction
from azelficoast.belief.self_improvement import (
    TeacherArtifacts,
    _challenger_consensus,
    _challenger_teacher_required,
    _mine_informative_fixtures,
    generate_teacher_evidence,
)
from azelficoast.live.corpus import DecisionFixture
from azelficoast.live.instrumentation import TRACE_SCHEMA, TRACE_SCHEMA_VERSION
from azelficoast.core.whole_turn_program import compile_whole_turn_programs


class _FakeEvaluator:
    spec = BeliefEvaluatorSpec(
        public_width=8,
        world_width=8,
        action_width=8,
        hidden_width=8,
        world_hidden_width=8,
    )
    identity = {
        "schema": "azelficoast.belief-policy-value-evaluator",
        "schema_version": 1,
        "checkpoint_digest": "sha256:" + "a" * 64,
        "observability": "public_belief_only",
        "architecture": "weighted_deep_sets_policy_value",
        "spec": spec.as_dict(),
    }

    def predict(self, inputs) -> BeliefPrediction:
        probability = 1.0 / len(inputs.legal_actions)
        probabilities = tuple(probability for _ in inputs.legal_actions)
        return BeliefPrediction(
            value=1.0 - max(inputs.world_weights),
            legal_actions=inputs.legal_actions,
            probabilities=probabilities,
            selected_action=min(inputs.legal_actions),
            policy_margin=0.0,
            policy_entropy_bits=0.0,
        )


class _FakeTeacherSource:
    showdown_commit = "pinned"

    def artifacts(self, fixture):
        worlds = [
            {
                "world_id": "scarf",
                "weight": 0.5,
                "hidden": {"opponent.active.item": "Scarf"},
            },
            {
                "world_id": "specs",
                "weight": 0.5,
                "hidden": {"opponent.active.item": "Specs"},
            },
        ]
        transitions = []
        for world in worlds:
            item = world["hidden"]["opponent.active.item"]
            common_successor = {
                "turn": 9,
                "request_state": "move",
                "p1": [{"species": "rotom", "hp": 61, "maxhp": 100}],
                "p2_active": {"species": "garchomp", "hp": "<unchanged>"},
            }
            transitions.extend(
                [
                    {
                        "world_id": world["world_id"],
                        "action": "wait",
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"kind": "same"},
                                "successor": common_successor,
                                "hidden_reads": [],
                                "continuations": {
                                    "fast": 4000.0 if item == "Specs" else -4000.0,
                                    "safe": 1000.0,
                                },
                            }
                        ],
                    },
                    {
                        "world_id": world["world_id"],
                        "action": "reveal",
                        "outcomes": [
                            {
                                "probability": 1.0,
                                "observation": {"kind": item},
                                "successor": {
                                    **common_successor,
                                    "revealed_item": item,
                                },
                                "hidden_reads": ["opponent.active.item"],
                                "continuations": {
                                    "fast": 3000.0 if item == "Specs" else -3000.0,
                                    "safe": 0.0,
                                },
                            }
                        ],
                    },
                ]
            )
        oracle = {
            "schema": "azelficoast.core.transition-oracle",
            "schema_version": 1,
            "source_fixture_id": fixture.fixture_id,
            "showdown_commit": self.showdown_commit,
            "worlds": copy.deepcopy(worlds),
            "legal_actions": list(fixture.legal_actions),
            "dependency_candidates": ["opponent.active.item"],
            "declared_reads": {
                "wait": [],
                "reveal": ["opponent.active.item"],
            },
            "transitions": transitions,
        }
        posterior = {
            "treatment": "generator_faithful",
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "worlds": copy.deepcopy(worlds),
        }
        return TeacherArtifacts(
            posterior=posterior,
            transition_program=compile_whole_turn_programs(oracle),
        )


def _record(event: int, kind: str, **extra: object) -> dict[str, object]:
    return {
        "schema": TRACE_SCHEMA,
        "schema_version": TRACE_SCHEMA_VERSION,
        "run_id": "run",
        "event_index": event,
        "observed_at": "2026-09-25T00:00:00+00:00",
        "kind": kind,
        **extra,
    }


def _trace(path) -> None:
    rows = [
        _record(
            0,
            "decision",
            battle_tag="battle",
            decision_index=0,
            state={
                "battle_tag": "battle",
                "turn": 8,
                "legal_actions": ["wait", "reveal"],
            },
            chosen_action="wait",
        ),
        _record(
            1,
            "terminal",
            battle_tag="battle",
            won=True,
            lost=False,
            tied=False,
            final_state={},
        ),
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mark_trace_hard(path) -> None:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    decision = next(row for row in rows if row.get("kind") == "decision")
    decision["decision_metadata"] = {
        "belief": {
            "status": "search",
            "reason": "learned-policy-uncertain",
            "diagnostics": {
                "learned_prediction": {
                    "policy_margin": 0.0,
                    "policy_entropy_bits": 1.0,
                }
            },
        }
    }
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_challenger_teacher_runs_only_for_hard_public_evidence() -> None:
    assert _challenger_teacher_required(
        {"uncertainty": 0.8, "search_count": 0, "fallback_count": 0},
        uncertainty_threshold=0.75,
    )
    assert _challenger_teacher_required(
        {"uncertainty": 0.1, "search_count": 1, "fallback_count": 0},
        uncertainty_threshold=0.75,
    )
    assert not _challenger_teacher_required(
        {"uncertainty": 0.2, "search_count": 0, "fallback_count": 0},
        uncertainty_threshold=0.75,
    )


def test_challenger_consensus_rejects_prior_sensitive_teacher_action() -> None:
    stable = _challenger_consensus(
        "attack",
        {"flattened": "attack", "sharpened": "attack"},
    )
    unstable = _challenger_consensus(
        "attack",
        {"flattened": "switch", "sharpened": "attack"},
    )

    assert stable["passed"] is True
    assert unstable["passed"] is False
    assert unstable["actions"]["flattened"] == "switch"

def test_teacher_evidence_is_reproducible_and_settled(tmp_path) -> None:
    trace = tmp_path / "trace.jsonl"
    _trace(trace)
    evaluator = _FakeEvaluator()
    source = _FakeTeacherSource()

    first = generate_teacher_evidence(
        [trace],
        evaluator=evaluator,
        source=source,
        output_root=tmp_path / "teachers",
        compute_budget=4,
    )
    second = generate_teacher_evidence(
        [trace],
        evaluator=evaluator,
        source=source,
        output_root=tmp_path / "teachers",
        compute_budget=4,
    )

    assert first.manifest_digest == second.manifest_digest
    assert first.admitted_decision_count == 1
    assert first.excluded_decision_count == 0
    assert len(first.packet_paths) == 1
    assert len(first.receipt_paths) == 2
    assert len(first.posterior_paths) == 1

    det = json.loads(first.receipt_paths[0].read_text(encoding="utf-8"))
    info = json.loads(first.receipt_paths[1].read_text(encoding="utf-8"))
    assert "resource_accounting" not in det
    assert "resource_accounting" not in info
    assert det["packet_digest"] == info["packet_digest"]
    assert det["evaluator_checkpoint_digest"] == _FakeEvaluator.identity["checkpoint_digest"]
    packet = json.loads(first.packet_paths[0].read_text(encoding="utf-8"))
    assert packet["run_id"] == "run"

    settled = json.loads(
        (first.packet_paths[0].parent / "settled.json").read_text(encoding="utf-8")
    )
    assert settled["matched_authorized_compute"] is True
    assert settled["matched_evaluator_checkpoint"] is True
    assert settled["matched_transition_program"] is True


def test_hard_teacher_state_requires_prior_stress_consensus(tmp_path) -> None:
    trace = tmp_path / "hard.jsonl"
    _trace(trace)
    _mark_trace_hard(trace)

    evidence = generate_teacher_evidence(
        [trace],
        evaluator=_FakeEvaluator(),
        source=_FakeTeacherSource(),
        output_root=tmp_path / "teachers",
        compute_budget=4,
    )

    assert evidence.admitted_decision_count == 1
    assert evidence.excluded_decision_count == 0
    assert len(evidence.packet_paths) == 1
    assert len(evidence.receipt_paths) == 2

    manifest = json.loads(evidence.manifest_path.read_text(encoding="utf-8"))
    challenger = manifest["admitted"][0]["challenger"]
    assert challenger["required"] is True
    assert challenger["passed"] is True
    assert len(challenger["runs"]) == 2
    assert {row["treatment"] for row in challenger["runs"]} == {
        "flattened",
        "sharpened",
    }
    assert manifest["challenger_policy"]["failure_mode"] == "exclude-training-target"


def test_prior_sensitive_challenger_withholds_teacher_label(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "hard.jsonl"
    _trace(trace)
    _mark_trace_hard(trace)
    original_execute = self_improvement.execute_method

    def disagreeing_execute_method(**kwargs):
        result = dict(original_execute(**kwargs))
        packet = kwargs["packet"]
        if (
            packet.get("posterior_treatment") == "flattened"
            and kwargs.get("method") == "information_set"
        ):
            legal_actions = list(packet["legal_actions"])
            alternate = next(
                action
                for action in legal_actions
                if action != result["chosen_action"]
            )
            result["chosen_action"] = alternate
            result["root_values"] = {
                action: (1.0 if action == alternate else 0.0)
                for action in legal_actions
            }
        return result

    monkeypatch.setattr(
        self_improvement,
        "execute_method",
        disagreeing_execute_method,
    )
    evidence = generate_teacher_evidence(
        [trace],
        evaluator=_FakeEvaluator(),
        source=_FakeTeacherSource(),
        output_root=tmp_path / "teachers",
        compute_budget=4,
    )

    assert evidence.admitted_decision_count == 0
    assert evidence.excluded_decision_count == 1
    assert evidence.packet_paths == ()
    assert evidence.receipt_paths == ()

    manifest = json.loads(evidence.manifest_path.read_text(encoding="utf-8"))
    assert manifest["excluded"][0]["reason"] == "posterior-sensitive-teacher-target"
    assert manifest["excluded"][0]["challenger"]["passed"] is False


def test_duplicate_trace_content_does_not_double_weight_teacher_data(tmp_path) -> None:
    first_trace = tmp_path / "one.jsonl"
    second_trace = tmp_path / "two.jsonl"
    _trace(first_trace)
    second_trace.write_bytes(first_trace.read_bytes())

    evidence = generate_teacher_evidence(
        [first_trace, second_trace],
        evaluator=_FakeEvaluator(),
        source=_FakeTeacherSource(),
        output_root=tmp_path / "teachers",
        compute_budget=4,
    )

    assert len(evidence.trace_paths) == 1
    assert evidence.admitted_decision_count == 1


def _mining_fixture(
    fixture_id: str,
    battle_tag: str,
    *,
    status: str,
    margin: float,
) -> DecisionFixture:
    return DecisionFixture(
        fixture_id=fixture_id,
        state={"turn": 8, "legal_actions": ["move a", "move b"]},
        protocol_prefix=(),
        control_decisions=(
            {
                "run_id": f"run-{battle_tag}",
                "battle_tag": battle_tag,
                "event_index": 1,
                "decision_metadata": {
                    "belief": {
                        "status": status,
                        "reason": (
                            "learned-policy-uncertain"
                            if status == "search"
                            else "learned-public-belief"
                        ),
                        "diagnostics": {
                            "learned_prediction": {
                                "policy_margin": margin,
                                "policy_entropy_bits": 0.75,
                            }
                        },
                    }
                },
            },
        ),
    )


def test_state_mining_spends_budget_across_battles_before_refilling() -> None:
    fixtures = [
        _mining_fixture("a-hard", "battle-a", status="search", margin=0.0),
        _mining_fixture("a-second", "battle-a", status="search", margin=0.01),
        _mining_fixture("b-state", "battle-b", status="selected", margin=0.9),
    ]

    selected, manifest = _mine_informative_fixtures(fixtures, max_fixtures=3)

    assert [fixture.fixture_id for fixture in selected] == ["a-hard", "b-state"]
    assert manifest["candidate_fixture_count"] == 3
    assert manifest["candidate_decision_count"] == 3
    assert manifest["selected_fixture_count"] == 2
    assert manifest["selected_decision_count"] == 2
    assert manifest["one_decision_per_battle"] is True
    assert manifest["kind"] == "public-evidence-debt-curriculum"



def test_unbounded_state_mining_preserves_existing_cycle_population() -> None:
    fixtures = [
        _mining_fixture("a-one", "battle-a", status="search", margin=0.0),
        _mining_fixture("a-two", "battle-a", status="search", margin=0.1),
        _mining_fixture("b-one", "battle-b", status="selected", margin=0.9),
    ]

    selected, manifest = _mine_informative_fixtures(fixtures, max_fixtures=None)

    assert selected == fixtures
    assert manifest["kind"] == "all-public-decisions"
    assert manifest["selected_fixture_count"] == 3
    assert manifest["selected_decision_count"] == 3
    assert manifest["one_decision_per_battle"] is False
