from __future__ import annotations

import hashlib
import json

import pytest

from azelficoast.belief.evaluator import (
    BeliefEvaluatorSpec,
    BeliefPrediction,
    init_params,
    load_checkpoint,
    write_checkpoint,
)
from azelficoast.belief.improvement import (
    AdmissionPolicy,
    EvaluationMetrics,
    ImprovementError,
    _promote,
    decide_admission,
    evaluate_hostile_invariants,
    improve_checkpoint,
    load_training_dataset,
    promote_deferred_candidate,
)
from azelficoast.belief.battle_promotion import (
    BattlePromotionPolicy,
    CANDIDATE_PRIMARY_MODE,
    INCUMBENT_PRIMARY_MODE,
    settle_battle_panel,
)
from azelficoast.research.training_records import TRAINING_SCHEMA, TRAINING_SCHEMA_VERSION


def _metrics(*, total: float, value: float, policy: float) -> EvaluationMetrics:
    return EvaluationMetrics(
        count=4,
        total_loss=total,
        value_mse=value,
        policy_cross_entropy=policy,
        policy_accuracy=0.5,
    )


def test_admission_requires_total_improvement_without_component_regression() -> None:
    incumbent = _metrics(total=1.0, value=0.4, policy=0.6)
    admitted = decide_admission(
        incumbent,
        _metrics(total=0.8, value=0.3, policy=0.5),
    )
    value_regression = decide_admission(
        incumbent,
        _metrics(total=0.9, value=0.5, policy=0.4),
    )

    assert admitted["admitted"] is True
    assert value_regression["admitted"] is False
    assert "validation_value_mse_regression" in value_regression["failed_checks"]


def _record(record_id: str, split: str) -> dict[str, object]:
    return {
        "schema": TRAINING_SCHEMA,
        "schema_version": TRAINING_SCHEMA_VERSION,
        "record_id": record_id,
        "battle_id": f"{split}-battle",
        "decision_index": 0,
        "split_group_id": f"{split}-group",
        "split": split,
        "input": {
            "fixture_id": f"{split}-fixture",
            "public_state": {"turn": 4, "active_hp_fraction": 0.7},
            "posterior": {
                "conditioned_on_public_history": True,
                "realized_hidden_state_revealed": False,
                "worlds": [
                    {"world_id": "a", "weight": 0.5, "hidden": {"item": "leftovers"}},
                    {"world_id": "b", "weight": 0.5, "hidden": {"item": "choicescarf"}},
                ],
            },
            "legal_actions": ["attack", "switch"],
        },
        "targets": {
            "value": {
                "eventual_battle_outcome": 1.0,
                "public_belief_search_return": 0.8,
            },
            "policy": {
                "selected_action": "attack",
                "action_probabilities": {"attack": 1.0, "switch": 0.0},
                "searched_action_values": {"attack": 0.8, "switch": 0.1},
            },
        },
        "provenance": {},
    }


def _records() -> list[dict[str, object]]:
    return [_record("train-record", "train"), _record("validation-record", "validation"), _record("test-record", "test")]


def _write(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _spec() -> BeliefEvaluatorSpec:
    return BeliefEvaluatorSpec(
        public_width=8,
        world_width=8,
        action_width=8,
        hidden_width=8,
        world_hidden_width=8,
    )


def test_dataset_digest_is_invariant_to_record_order(tmp_path) -> None:
    rows = _records()
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write(first, rows)
    _write(second, list(reversed(rows)))

    a = load_training_dataset(first, spec=_spec())
    b = load_training_dataset(second, spec=_spec())

    assert a.digest == b.digest
    assert a.record_counts == {"train": 1, "validation": 1, "test": 1}


def test_dataset_rejects_split_group_leakage(tmp_path) -> None:
    rows = _records()
    rows[1]["split_group_id"] = rows[0]["split_group_id"]
    path = tmp_path / "training.jsonl"
    _write(path, rows)

    with pytest.raises(ImprovementError, match="crosses dataset splits"):
        load_training_dataset(path, spec=_spec())


def test_rejected_candidate_does_not_replace_promotion_pointer(tmp_path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("numpy")
    incumbent = tmp_path / "incumbent"
    write_checkpoint(incumbent, init_params(_spec(), seed=5), _spec())
    dataset = tmp_path / "training.jsonl"
    _write(dataset, _records())
    promotion = tmp_path / "current.json"
    promotion.write_text('{"sentinel":true}\n', encoding="utf-8")

    result = improve_checkpoint(
        dataset,
        incumbent_checkpoint=incumbent,
        models_dir=tmp_path / "candidates",
        receipts_dir=tmp_path / "receipts",
        promotion_file=promotion,
        admission_policy=AdmissionPolicy(min_validation_total_improvement=1e9),
    )

    assert result["admitted"] is False
    assert promotion.read_text(encoding="utf-8") == '{"sentinel":true}\n'


def test_admitted_candidate_pointer_is_loadable_and_digest_bound(tmp_path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("numpy")
    incumbent = tmp_path / "incumbent"
    write_checkpoint(incumbent, init_params(_spec(), seed=7), _spec())
    dataset = tmp_path / "training.jsonl"
    _write(dataset, _records())
    promotion = tmp_path / "current.json"

    result = improve_checkpoint(
        dataset,
        incumbent_checkpoint=incumbent,
        models_dir=tmp_path / "candidates",
        receipts_dir=tmp_path / "receipts",
        promotion_file=promotion,
        epochs=4,
        learning_rate=1e-2,
        admission_policy=AdmissionPolicy(
            min_validation_total_improvement=0.0,
            max_validation_value_mse_regression=10.0,
            max_validation_policy_cross_entropy_regression=10.0,
        ),
    )

    assert result["admitted"] is True
    _, _, manifest = load_checkpoint(promotion)
    assert manifest["evaluator"]["checkpoint_digest"] == result["candidate_checkpoint_digest"]



def test_promotion_is_compare_and_swap_fenced(tmp_path) -> None:
    promotion = tmp_path / "current.json"
    promotion.write_text(
        json.dumps(
            {
                "schema": "azelficoast.evaluator-promotion",
                "schema_version": 1,
                "checkpoint": "other",
                "checkpoint_digest": "sha256:" + "b" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    before = promotion.read_text(encoding="utf-8")

    with pytest.raises(ImprovementError, match="authority changed"):
        _promote(
            promotion,
            candidate_path=tmp_path / "candidate",
            candidate_digest="sha256:" + "c" * 64,
            incumbent_digest="sha256:" + "a" * 64,
            dataset_digest="sha256:" + "d" * 64,
            receipt_digest="sha256:" + "e" * 64,
        )

    assert promotion.read_text(encoding="utf-8") == before


def test_hostile_gate_rejects_action_order_sensitive_candidate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = tmp_path / "training.jsonl"
    _write(dataset_path, _records())
    dataset = load_training_dataset(dataset_path, spec=_spec())

    def order_sensitive_predict(params, inputs):
        del params
        probabilities = tuple(
            1.0 if index == 0 else 0.0
            for index, _ in enumerate(inputs.legal_actions)
        )
        return BeliefPrediction(
            value=0.0,
            legal_actions=inputs.legal_actions,
            probabilities=probabilities,
            selected_action=inputs.legal_actions[0],
            policy_margin=1.0,
            policy_entropy_bits=0.0,
        )

    monkeypatch.setattr(
        "azelficoast.belief.improvement.predict",
        order_sensitive_predict,
    )
    result = evaluate_hostile_invariants({}, dataset.examples("validation"))

    assert result["passed"] is False
    assert result["failure_count"] >= 1
    assert {
        failure["mutation"] for failure in result["failures"]
    } >= {"reverse-legal-action-order"}


def test_deferred_candidate_cannot_mutate_authority_before_battle_gate(tmp_path) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("numpy")
    incumbent = tmp_path / "incumbent"
    write_checkpoint(incumbent, init_params(_spec(), seed=11), _spec())
    dataset = tmp_path / "training.jsonl"
    _write(dataset, _records())
    promotion = tmp_path / "current.json"

    improvement = improve_checkpoint(
        dataset,
        incumbent_checkpoint=incumbent,
        models_dir=tmp_path / "candidates",
        receipts_dir=tmp_path / "receipts",
        promotion_file=promotion,
        epochs=4,
        learning_rate=1e-2,
        admission_policy=AdmissionPolicy(
            min_validation_total_improvement=0.0,
            max_validation_value_mse_regression=10.0,
            max_validation_policy_cross_entropy_regression=10.0,
        ),
        promote=False,
    )

    assert improvement["admitted"] is True
    assert improvement["promotion_deferred"] is True
    assert not promotion.exists()

    raw_results = tmp_path / "promotion-results.jsonl"
    raw_rows = [
        {
            "battle_tag": f"battle-{index}",
            "mode": (
                CANDIDATE_PRIMARY_MODE
                if index < 2
                else INCUMBENT_PRIMARY_MODE
            ),
            "won": index < 2,
            "lost": index >= 2,
            "candidate_checkpoint_digest": improvement["candidate_checkpoint_digest"],
            "incumbent_checkpoint_digest": improvement["incumbent_checkpoint_digest"],
        }
        for index in range(4)
    ]
    _write(raw_results, raw_rows)
    battle_evidence = settle_battle_panel(
        raw_rows,
        candidate_checkpoint_digest=improvement["candidate_checkpoint_digest"],
        incumbent_checkpoint_digest=improvement["incumbent_checkpoint_digest"],
        policy=BattlePromotionPolicy(
            expected_battles=4,
            max_superiority_p_value=0.5,
        ),
    )
    battle_evidence = {
        **battle_evidence,
        "results": str(raw_results),
        "results_digest": hashlib.sha256(raw_results.read_bytes()).hexdigest(),
    }
    settled = promote_deferred_candidate(
        improvement,
        battle_evidence=battle_evidence,
        promotion_file=promotion,
        receipts_dir=tmp_path / "receipts",
    )

    assert settled["admitted"] is True
    _, _, manifest = load_checkpoint(promotion)
    assert (
        manifest["evaluator"]["checkpoint_digest"]
        == improvement["candidate_checkpoint_digest"]
    )


def test_deferred_promotion_rejects_mismatched_battle_identity(tmp_path) -> None:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    improvement = {
        "admitted": True,
        "promotion_deferred": True,
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_digest": "sha256:candidate",
        "incumbent_checkpoint_digest": "sha256:incumbent",
        "dataset_digest": "sha256:dataset",
        "receipt_digest": "sha256:improvement",
    }
    raw_results = tmp_path / "mismatched-results.jsonl"
    raw_rows = [
        {
            "battle_tag": f"battle-{index}",
            "mode": (
                CANDIDATE_PRIMARY_MODE
                if index < 2
                else INCUMBENT_PRIMARY_MODE
            ),
            "won": index < 2,
            "lost": index >= 2,
            "candidate_checkpoint_digest": "sha256:other",
            "incumbent_checkpoint_digest": "sha256:incumbent",
        }
        for index in range(4)
    ]
    _write(raw_results, raw_rows)
    battle_evidence = settle_battle_panel(
        raw_rows,
        candidate_checkpoint_digest="sha256:other",
        incumbent_checkpoint_digest="sha256:incumbent",
        policy=BattlePromotionPolicy(
            expected_battles=4,
            max_superiority_p_value=0.5,
        ),
    )
    battle_evidence = {
        **battle_evidence,
        "results": str(raw_results),
        "results_digest": hashlib.sha256(raw_results.read_bytes()).hexdigest(),
    }

    with pytest.raises(ImprovementError, match="candidate checkpoint does not match"):
        promote_deferred_candidate(
            improvement,
            battle_evidence=battle_evidence,
            promotion_file=tmp_path / "current.json",
            receipts_dir=tmp_path / "receipts",
        )
