"""Evidence-gated candidate training and promotion for the learned evaluator.

Training may propose a checkpoint. Deterministic software decides whether that checkpoint
becomes authoritative. Validation participates in admission; the held-out test split is
reported only.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.belief.evaluator import (
    PROMOTION_SCHEMA,
    PROMOTION_SCHEMA_VERSION,
    BeliefEvaluatorError,
    BeliefEvaluatorInput,
    BeliefEvaluatorSpec,
    build_evaluator_input,
    checkpoint_digest,
    load_checkpoint,
    predict,
    write_checkpoint,
)
from azelficoast.belief.training import TrainingExample, train_examples
from azelficoast.research.training_records import TRAINING_SCHEMA, TRAINING_SCHEMA_VERSION

IMPROVEMENT_RECEIPT_SCHEMA = "azelficoast.evaluator-improvement-receipt"
IMPROVEMENT_RECEIPT_SCHEMA_VERSION = 2
VALUE_TARGET_SOURCES = ("public_belief_search_return", "eventual_battle_outcome")


class ImprovementError(ValueError):
    """Raised when a model-update transaction cannot be trusted."""


@dataclass(frozen=True)
class EvaluationMetrics:
    count: int
    total_loss: float
    value_mse: float
    policy_cross_entropy: float
    policy_accuracy: float

    def as_record(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_loss": self.total_loss,
            "value_mse": self.value_mse,
            "policy_cross_entropy": self.policy_cross_entropy,
            "policy_accuracy": self.policy_accuracy,
        }


@dataclass(frozen=True)
class AdmissionPolicy:
    min_validation_total_improvement: float = 1e-6
    max_validation_value_mse_regression: float = 0.0
    max_validation_policy_cross_entropy_regression: float = 0.0

    def __post_init__(self) -> None:
        for name, value in self.as_record().items():
            if not math.isfinite(value) or value < 0.0:
                raise ImprovementError(f"{name} must be finite and non-negative")

    def as_record(self) -> dict[str, float]:
        return {
            "min_validation_total_improvement": self.min_validation_total_improvement,
            "max_validation_value_mse_regression": self.max_validation_value_mse_regression,
            "max_validation_policy_cross_entropy_regression": (
                self.max_validation_policy_cross_entropy_regression
            ),
        }


@dataclass(frozen=True)
class FrozenTrainingDataset:
    digest: str
    examples_by_split: Mapping[str, tuple[TrainingExample, ...]]
    record_counts: Mapping[str, int]
    split_group_counts: Mapping[str, int]

    def examples(self, split: str) -> tuple[TrainingExample, ...]:
        return self.examples_by_split.get(split, ())


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, str) else _canonical(value)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as error:
        raise ImprovementError(f"cannot read training dataset: {error}") from error
    if not text.strip():
        raise ImprovementError("training dataset is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        return [dict(parsed)]
    if isinstance(parsed, list):
        if not all(isinstance(row, Mapping) for row in parsed):
            raise ImprovementError("training dataset array must contain objects")
        return [dict(row) for row in parsed]

    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ImprovementError(f"{source}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(row, Mapping):
            raise ImprovementError(f"{source}:{line_number}: training record must be an object")
        result.append(dict(row))
    if not result:
        raise ImprovementError("training dataset is empty")
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ImprovementError(f"{field} must be an object")
    return value


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ImprovementError(f"{field} must be a non-empty string")
    return value


def _example(
    record: Mapping[str, Any],
    *,
    spec: BeliefEvaluatorSpec,
    value_target_source: str,
) -> TrainingExample:
    inputs = _mapping(record.get("input"), "input")
    public_state = _mapping(inputs.get("public_state"), "input.public_state")
    posterior = _mapping(inputs.get("posterior"), "input.posterior")
    raw_actions = inputs.get("legal_actions")
    if not isinstance(raw_actions, list) or not raw_actions or not all(
        isinstance(action, str) and action for action in raw_actions
    ):
        raise ImprovementError("input.legal_actions must be a non-empty list of strings")
    actions = tuple(raw_actions)
    if len(set(actions)) != len(actions):
        raise ImprovementError("input.legal_actions must be unique")

    targets = _mapping(record.get("targets"), "targets")
    value_targets = _mapping(targets.get("value"), "targets.value")
    raw_value = value_targets.get(value_target_source)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise ImprovementError(f"targets.value.{value_target_source} must be numeric")
    value_target = float(raw_value)
    if not math.isfinite(value_target) or not -1.0 <= value_target <= 1.0:
        raise ImprovementError(f"targets.value.{value_target_source} must be within [-1, 1]")

    policy = _mapping(targets.get("policy"), "targets.policy")
    probabilities = _mapping(
        policy.get("action_probabilities"),
        "targets.policy.action_probabilities",
    )
    if set(probabilities) != set(actions):
        raise ImprovementError("policy target actions must exactly match legal actions")
    policy_target: list[float] = []
    for action in actions:
        raw = probabilities[action]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ImprovementError(f"policy target for {action!r} must be numeric")
        probability = float(raw)
        if not math.isfinite(probability) or probability < 0.0:
            raise ImprovementError(f"policy target for {action!r} must be non-negative")
        policy_target.append(probability)

    try:
        return TrainingExample(
            inputs=build_evaluator_input(
                public_state=public_state,
                posterior=posterior,
                legal_actions=actions,
                spec=spec,
            ),
            value_target=value_target,
            policy_target=tuple(policy_target),
        )
    except BeliefEvaluatorError as error:
        raise ImprovementError(str(error)) from error


def load_training_dataset(
    path: str | Path,
    *,
    spec: BeliefEvaluatorSpec,
    value_target_source: str = "public_belief_search_return",
) -> FrozenTrainingDataset:
    """Validate leakage boundaries and canonicalize record order before training."""
    if value_target_source not in VALUE_TARGET_SOURCES:
        raise ImprovementError("unsupported value target source")
    records = _records(path)
    seen_ids: set[str] = set()
    identity_split: dict[tuple[str, str], str] = {}
    groups = {split: set() for split in ("train", "validation", "test")}

    for record in records:
        if (
            record.get("schema") != TRAINING_SCHEMA
            or record.get("schema_version") != TRAINING_SCHEMA_VERSION
        ):
            raise ImprovementError("unexpected training record schema")
        record_id = _text(record.get("record_id"), "record_id")
        if record_id in seen_ids:
            raise ImprovementError(f"duplicate training record_id {record_id!r}")
        seen_ids.add(record_id)
        split = _text(record.get("split"), "split")
        if split not in groups:
            raise ImprovementError(f"unexpected dataset split {split!r}")
        split_group = _text(record.get("split_group_id"), "split_group_id")
        battle = _text(record.get("battle_id"), "battle_id")
        fixture = _text(_mapping(record.get("input"), "input").get("fixture_id"), "input.fixture_id")
        for kind, identity in (("split_group", split_group), ("battle", battle), ("fixture", fixture)):
            key = (kind, identity)
            previous = identity_split.setdefault(key, split)
            if previous != split:
                raise ImprovementError(
                    f"{kind} {identity!r} crosses dataset splits: {previous!r} vs {split!r}"
                )
        groups[split].add(split_group)

    for split, split_groups in groups.items():
        if not split_groups:
            raise ImprovementError(f"self-improvement requires a non-empty {split!r} split")

    records.sort(key=lambda row: str(row["record_id"]))
    examples = {split: [] for split in groups}
    for record in records:
        examples[str(record["split"])].append(
            _example(record, spec=spec, value_target_source=value_target_source)
        )

    return FrozenTrainingDataset(
        digest=_sha256(
            {
                "schema": TRAINING_SCHEMA,
                "schema_version": TRAINING_SCHEMA_VERSION,
                "records": records,
            }
        ),
        examples_by_split={split: tuple(rows) for split, rows in examples.items()},
        record_counts={split: len(rows) for split, rows in examples.items()},
        split_group_counts={split: len(rows) for split, rows in groups.items()},
    )


def evaluate_examples(
    params: Mapping[str, Any],
    examples: Sequence[TrainingExample],
    *,
    policy_weight: float = 1.0,
) -> EvaluationMetrics:
    if not examples:
        raise ImprovementError("evaluation requires at least one example")
    if not math.isfinite(policy_weight) or policy_weight < 0.0:
        raise ImprovementError("policy_weight must be finite and non-negative")

    value_sum = 0.0
    policy_sum = 0.0
    correct = 0
    for example in examples:
        prediction = predict(params, example.inputs)
        value_sum += (prediction.value - example.value_target) ** 2
        mass = sum(example.policy_target)
        target = tuple(value / mass for value in example.policy_target)
        policy_sum += -sum(
            wanted * math.log(max(actual, 1e-12))
            for wanted, actual in zip(target, prediction.probabilities, strict=True)
            if wanted > 0.0
        )
        maximum = max(target)
        best = {
            action
            for action, wanted in zip(example.inputs.legal_actions, target, strict=True)
            if abs(wanted - maximum) <= 1e-15
        }
        correct += int(prediction.selected_action in best)

    count = len(examples)
    value_mse = value_sum / count
    policy_cross_entropy = policy_sum / count
    return EvaluationMetrics(
        count=count,
        total_loss=value_mse + policy_weight * policy_cross_entropy,
        value_mse=value_mse,
        policy_cross_entropy=policy_cross_entropy,
        policy_accuracy=correct / count,
    )


def evaluate_hostile_invariants(
    params: Mapping[str, Any],
    examples: Sequence[TrainingExample],
) -> dict[str, Any]:
    """Falsify candidate dependence on incidental support/action ordering."""

    failures: list[dict[str, Any]] = []
    checked = 0

    def signature(inputs: BeliefEvaluatorInput) -> tuple[float, dict[str, float], str]:
        prediction = predict(params, inputs)
        return (
            prediction.value,
            dict(zip(prediction.legal_actions, prediction.probabilities, strict=True)),
            prediction.selected_action,
        )

    def equivalent(
        expected: tuple[float, dict[str, float], str],
        actual: tuple[float, dict[str, float], str],
    ) -> bool:
        if expected[2] != actual[2] or set(expected[1]) != set(actual[1]):
            return False
        if not math.isclose(expected[0], actual[0], rel_tol=1e-6, abs_tol=1e-6):
            return False
        return all(
            math.isclose(expected[1][action], actual[1][action], rel_tol=1e-6, abs_tol=1e-6)
            for action in expected[1]
        )

    for index, example in enumerate(examples):
        base = example.inputs
        expected = signature(base)

        world_permuted = BeliefEvaluatorInput(
            public_features=base.public_features,
            world_features=tuple(reversed(base.world_features)),
            world_weights=tuple(reversed(base.world_weights)),
            action_features=base.action_features,
            legal_actions=base.legal_actions,
        )
        checked += 1
        if not equivalent(expected, signature(world_permuted)):
            failures.append({"example_index": index, "mutation": "reverse-hidden-world-order"})

        action_permuted = BeliefEvaluatorInput(
            public_features=base.public_features,
            world_features=base.world_features,
            world_weights=base.world_weights,
            action_features=tuple(reversed(base.action_features)),
            legal_actions=tuple(reversed(base.legal_actions)),
        )
        checked += 1
        if not equivalent(expected, signature(action_permuted)):
            failures.append({"example_index": index, "mutation": "reverse-legal-action-order"})

    return {
        "passed": not failures,
        "checked_mutations": checked,
        "failure_count": len(failures),
        "failures": failures,
    }


def decide_admission(
    incumbent: EvaluationMetrics,
    candidate: EvaluationMetrics,
    *,
    policy: AdmissionPolicy = AdmissionPolicy(),
) -> dict[str, Any]:
    """Decide promotion from validation metrics only."""
    if incumbent.count != candidate.count or incumbent.count <= 0:
        raise ImprovementError("incumbent and candidate must evaluate the same records")
    measurements = {
        "validation_total_improvement": incumbent.total_loss - candidate.total_loss,
        "validation_value_mse_regression": candidate.value_mse - incumbent.value_mse,
        "validation_policy_cross_entropy_regression": (
            candidate.policy_cross_entropy - incumbent.policy_cross_entropy
        ),
    }
    checks = {
        "validation_total_improvement": (
            measurements["validation_total_improvement"]
            >= policy.min_validation_total_improvement
        ),
        "validation_value_mse_regression": (
            measurements["validation_value_mse_regression"]
            <= policy.max_validation_value_mse_regression
        ),
        "validation_policy_cross_entropy_regression": (
            measurements["validation_policy_cross_entropy_regression"]
            <= policy.max_validation_policy_cross_entropy_regression
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "admitted": not failed,
        "policy": policy.as_record(),
        "measurements": measurements,
        "checks": checks,
        "failed_checks": failed,
    }


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = _canonical(value) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise ImprovementError(f"{path} already contains different immutable evidence")
        return
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _candidate_checkpoint(
    models_dir: Path,
    *,
    params: Mapping[str, Any],
    spec: BeliefEvaluatorSpec,
    metadata: Mapping[str, Any],
) -> tuple[Path, str]:
    digest = checkpoint_digest(params, spec)
    destination = models_dir / digest.removeprefix("sha256:")
    if destination.exists():
        _, existing_spec, manifest = load_checkpoint(destination)
        evaluator = _mapping(manifest.get("evaluator"), "candidate.evaluator")
        if existing_spec != spec or evaluator.get("checkpoint_digest") != digest:
            raise ImprovementError("existing candidate checkpoint conflicts with candidate digest")
        return destination, digest
    write_checkpoint(destination, params, spec, metadata=metadata)
    return destination, digest


def _promote(
    path: Path,
    *,
    candidate_path: Path,
    candidate_digest: str,
    incumbent_digest: str,
    dataset_digest: str,
    receipt_digest: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ImprovementError(f"cannot read current promotion pointer: {error}") from error
        if not isinstance(current, Mapping):
            raise ImprovementError("current promotion pointer must be an object")
        current_digest = current.get("checkpoint_digest")
        if current_digest != incumbent_digest:
            raise ImprovementError(
                "promotion authority changed since candidate evaluation: "
                f"expected {incumbent_digest}, found {current_digest!r}"
            )
    pointer = {
        "schema": PROMOTION_SCHEMA,
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "checkpoint": os.path.relpath(candidate_path, path.parent),
        "checkpoint_digest": candidate_digest,
        "previous_checkpoint_digest": incumbent_digest,
        "dataset_digest": dataset_digest,
        "improvement_receipt_digest": receipt_digest,
    }
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_canonical(pointer) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def improve_checkpoint(
    dataset_path: str | Path,
    *,
    incumbent_checkpoint: str | Path,
    models_dir: str | Path,
    receipts_dir: str | Path,
    promotion_file: str | Path,
    expected_incumbent_digest: str | None = None,
    epochs: int = 1,
    learning_rate: float = 3e-4,
    policy_weight: float = 1.0,
    value_target_source: str = "public_belief_search_return",
    admission_policy: AdmissionPolicy = AdmissionPolicy(),
) -> dict[str, Any]:
    """Train one candidate and atomically promote it only after admission."""
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ImprovementError("epochs must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ImprovementError("learning_rate must be positive and finite")

    try:
        incumbent_params, spec, incumbent_manifest = load_checkpoint(incumbent_checkpoint)
    except BeliefEvaluatorError as error:
        raise ImprovementError(str(error)) from error
    incumbent_digest = _text(
        _mapping(incumbent_manifest.get("evaluator"), "incumbent.evaluator").get(
            "checkpoint_digest"
        ),
        "incumbent.evaluator.checkpoint_digest",
    )
    if (
        expected_incumbent_digest is not None
        and incumbent_digest != expected_incumbent_digest
    ):
        raise ImprovementError(
            "incumbent checkpoint changed before candidate training: "
            f"expected {expected_incumbent_digest}, found {incumbent_digest}"
        )
    dataset = load_training_dataset(
        dataset_path,
        spec=spec,
        value_target_source=value_target_source,
    )
    candidate_params, losses = train_examples(
        incumbent_params,
        dataset.examples("train"),
        epochs=epochs,
        learning_rate=learning_rate,
        policy_weight=policy_weight,
    )
    training = {
        "epochs": epochs,
        "learning_rate": learning_rate,
        "policy_weight": policy_weight,
        "value_target_source": value_target_source,
        "training_record_count": len(dataset.examples("train")),
        "optimizer": "deterministic_adam",
    }
    candidate_path, candidate_digest = _candidate_checkpoint(
        Path(models_dir),
        params=candidate_params,
        spec=spec,
        metadata={
            "kind": "self-improvement-candidate",
            "dataset_digest": dataset.digest,
            "incumbent_checkpoint_digest": incumbent_digest,
            "training": training,
        },
    )

    incumbent_validation = evaluate_examples(
        incumbent_params, dataset.examples("validation"), policy_weight=policy_weight
    )
    candidate_validation = evaluate_examples(
        candidate_params, dataset.examples("validation"), policy_weight=policy_weight
    )
    incumbent_test = evaluate_examples(
        incumbent_params, dataset.examples("test"), policy_weight=policy_weight
    )
    candidate_test = evaluate_examples(
        candidate_params, dataset.examples("test"), policy_weight=policy_weight
    )
    admission = decide_admission(
        incumbent_validation,
        candidate_validation,
        policy=admission_policy,
    )
    hostile = evaluate_hostile_invariants(
        candidate_params,
        dataset.examples("validation"),
    )
    admission = {
        **admission,
        "admitted": bool(admission["admitted"] and hostile["passed"]),
        "checks": {
            **dict(admission["checks"]),
            "hostile_representation_invariance": bool(hostile["passed"]),
        },
        "failed_checks": [
            *list(admission["failed_checks"]),
            *([] if hostile["passed"] else ["hostile_representation_invariance"]),
        ],
    }

    receipt_material = {
        "schema": IMPROVEMENT_RECEIPT_SCHEMA,
        "schema_version": IMPROVEMENT_RECEIPT_SCHEMA_VERSION,
        "dataset": {
            "digest": dataset.digest,
            "record_counts": dict(dataset.record_counts),
            "split_group_counts": dict(dataset.split_group_counts),
        },
        "incumbent_checkpoint_digest": incumbent_digest,
        "candidate_checkpoint_digest": candidate_digest,
        "training": {
            **training,
            "step_count": len(losses),
            "first_observed_loss": losses[0],
            "last_observed_loss": losses[-1],
        },
        "evaluation": {
            "validation": {
                "incumbent": incumbent_validation.as_record(),
                "candidate": candidate_validation.as_record(),
                "used_for_admission": True,
            },
            "test": {
                "incumbent": incumbent_test.as_record(),
                "candidate": candidate_test.as_record(),
                "used_for_admission": False,
            },
            "hostile": {
                **hostile,
                "used_for_admission": True,
                "target_labels_consulted": False,
            },
        },
        "admission": admission,
    }
    receipt_digest = _sha256(receipt_material)
    receipt = {**receipt_material, "receipt_digest": receipt_digest}
    receipt_path = Path(receipts_dir) / f"{receipt_digest.removeprefix('sha256:')}.json"
    _write_immutable_json(receipt_path, receipt)

    promotion_path: Path | None = None
    if admission["admitted"]:
        promotion_path = Path(promotion_file)
        _promote(
            promotion_path,
            candidate_path=candidate_path,
            candidate_digest=candidate_digest,
            incumbent_digest=incumbent_digest,
            dataset_digest=dataset.digest,
            receipt_digest=receipt_digest,
        )

    return {
        "admitted": bool(admission["admitted"]),
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_digest": candidate_digest,
        "incumbent_checkpoint_digest": incumbent_digest,
        "dataset_digest": dataset.digest,
        "receipt": str(receipt_path),
        "receipt_digest": receipt_digest,
        "promotion_file": str(promotion_path) if promotion_path is not None else None,
        "failed_checks": list(admission["failed_checks"]),
        "validation": {
            "incumbent": incumbent_validation.as_record(),
            "candidate": candidate_validation.as_record(),
        },
        "test": {
            "incumbent": incumbent_test.as_record(),
            "candidate": candidate_test.as_record(),
        },
        "hostile": hostile,
    }
