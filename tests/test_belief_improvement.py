from __future__ import annotations

import json

import pytest

from azelficoast.belief.evaluator import (
    BeliefEvaluatorSpec,
    init_params,
    load_checkpoint,
    write_checkpoint,
)
from azelficoast.belief.improvement import (
    AdmissionPolicy,
    EvaluationMetrics,
    ImprovementError,
    decide_admission,
    improve_checkpoint,
    load_training_dataset,
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
