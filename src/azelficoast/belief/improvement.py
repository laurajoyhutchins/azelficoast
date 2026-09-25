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
from azelficoast.belief.battle_promotion import (
    BATTLE_PROMOTION_SCHEMA,
    BATTLE_PROMOTION_SCHEMA_VERSION,
    BattlePromotionError,
    verify_battle_panel_evidence,
)
from azelficoast.belief.training import TrainingExample, train_examples
from azelficoast.research.training_records import TRAINING_SCHEMA, TRAINING_SCHEMA_VERSION

IMPROVEMENT_RECEIPT_SCHEMA = "azelficoast.evaluator-improvement-receipt"
IMPROVEMENT_RECEIPT_SCHEMA_VERSION = 4
POSTERIOR_STRESS_TREATMENTS = ("flattened", "sharpened")
PROMOTION_SETTLEMENT_SCHEMA = "azelficoast.evaluator-promotion-settlement"
PROMOTION_SETTLEMENT_SCHEMA_VERSION = 1
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
    max_validation_posterior_stress_regression: float = 0.0

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
            "max_validation_posterior_stress_regression": (
                self.max_validation_posterior_stress_regression
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


def _posterior_stress_input(
    inputs: BeliefEvaluatorInput,
    *,
    treatment: str,
) -> BeliefEvaluatorInput:
    """Reweight one validated belief input without changing its support."""

    if treatment not in POSTERIOR_STRESS_TREATMENTS:
        raise ImprovementError(f"unsupported posterior stress treatment {treatment!r}")
    if not inputs.world_weights:
        raise ImprovementError("posterior stress requires hidden-world support")

    if treatment == "flattened":
        raw = tuple(1.0 for _ in inputs.world_weights)
    else:
        raw = tuple(weight * weight for weight in inputs.world_weights)
    mass = sum(raw)
    if not math.isfinite(mass) or mass <= 0.0:
        raise ImprovementError("posterior stress produced invalid probability mass")
    weights = tuple(value / mass for value in raw)
    return BeliefEvaluatorInput(
        public_features=inputs.public_features,
        world_features=inputs.world_features,
        world_weights=weights,
        action_features=inputs.action_features,
        legal_actions=inputs.legal_actions,
    )


def evaluate_posterior_stress(
    params: Mapping[str, Any],
    examples: Sequence[TrainingExample],
    *,
    policy_weight: float = 1.0,
) -> dict[str, Any]:
    """Measure held-out evaluator sensitivity to plausible prior reweighting.

    Teacher targets are intentionally held fixed. This is a robustness/sensitivity
    check, not an alternate-posterior oracle-label experiment.
    """

    results: dict[str, dict[str, float | int]] = {}
    for treatment in POSTERIOR_STRESS_TREATMENTS:
        stressed = tuple(
            TrainingExample(
                inputs=_posterior_stress_input(example.inputs, treatment=treatment),
                value_target=example.value_target,
                policy_target=example.policy_target,
            )
            for example in examples
        )
        results[treatment] = evaluate_examples(
            params,
            stressed,
            policy_weight=policy_weight,
        ).as_record()
    worst_treatment = max(
        sorted(results),
        key=lambda name: float(results[name]["total_loss"]),
    )
    return {
        "treatments": results,
        "worst_treatment": worst_treatment,
        "worst_total_loss": float(results[worst_treatment]["total_loss"]),
        "target_semantics": "nominal-settled-search-target-held-fixed",
    }


def evaluate_posterior_robustness(
    incumbent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    policy: AdmissionPolicy,
) -> dict[str, Any]:
    """Require the candidate not to become more fragile than the incumbent."""

    incumbent_treatments = _mapping(incumbent.get("treatments"), "incumbent.treatments")
    candidate_treatments = _mapping(candidate.get("treatments"), "candidate.treatments")
    regressions: dict[str, float] = {}
    for treatment in POSTERIOR_STRESS_TREATMENTS:
        incumbent_metrics = _mapping(
            incumbent_treatments.get(treatment),
            f"incumbent.treatments.{treatment}",
        )
        candidate_metrics = _mapping(
            candidate_treatments.get(treatment),
            f"candidate.treatments.{treatment}",
        )
        regressions[treatment] = float(candidate_metrics["total_loss"]) - float(
            incumbent_metrics["total_loss"]
        )
    worst_treatment = max(sorted(regressions), key=regressions.__getitem__)
    worst_regression = regressions[worst_treatment]
    passed = (
        worst_regression <= policy.max_validation_posterior_stress_regression
    )
    return {
        "passed": passed,
        "policy": {
            "max_validation_posterior_stress_regression": (
                policy.max_validation_posterior_stress_regression
            )
        },
        "regressions": regressions,
        "worst_treatment": worst_treatment,
        "worst_regression": worst_regression,
        "target_semantics": "nominal-settled-search-target-held-fixed",
    }


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



def promote_deferred_candidate(
    improvement: Mapping[str, Any],
    *,
    battle_evidence: Mapping[str, Any],
    promotion_file: str | Path,
    receipts_dir: str | Path,
) -> dict[str, Any]:
    """Promote a candidate only after independent battle evidence also admits it."""

    if improvement.get("admitted") is not True:
        raise ImprovementError("cannot promote a candidate rejected by model admission")
    if improvement.get("promotion_deferred") is not True:
        raise ImprovementError("candidate was not produced by a deferred promotion transaction")
    if (
        battle_evidence.get("schema") != BATTLE_PROMOTION_SCHEMA
        or battle_evidence.get("schema_version") != BATTLE_PROMOTION_SCHEMA_VERSION
    ):
        raise ImprovementError("unexpected battle promotion evidence schema")
    try:
        verified_battle_evidence = verify_battle_panel_evidence(battle_evidence)
    except BattlePromotionError as error:
        raise ImprovementError(str(error)) from error
    if verified_battle_evidence.get("admitted") is not True:
        raise ImprovementError("battle promotion evidence did not admit the candidate")

    candidate_digest = _text(
        improvement.get("candidate_checkpoint_digest"),
        "candidate_checkpoint_digest",
    )
    incumbent_digest = _text(
        improvement.get("incumbent_checkpoint_digest"),
        "incumbent_checkpoint_digest",
    )
    dataset_digest = _text(improvement.get("dataset_digest"), "dataset_digest")
    improvement_receipt_digest = _text(
        improvement.get("receipt_digest"),
        "improvement.receipt_digest",
    )
    if verified_battle_evidence.get("candidate_checkpoint_digest") != candidate_digest:
        raise ImprovementError("battle evidence candidate checkpoint does not match")
    if verified_battle_evidence.get("incumbent_checkpoint_digest") != incumbent_digest:
        raise ImprovementError("battle evidence incumbent checkpoint does not match")

    candidate_path = Path(
        _text(improvement.get("candidate_checkpoint"), "candidate_checkpoint")
    )
    if not candidate_path.is_dir():
        raise ImprovementError("candidate checkpoint no longer exists")

    material = {
        "schema": PROMOTION_SETTLEMENT_SCHEMA,
        "schema_version": PROMOTION_SETTLEMENT_SCHEMA_VERSION,
        "candidate_checkpoint_digest": candidate_digest,
        "incumbent_checkpoint_digest": incumbent_digest,
        "dataset_digest": dataset_digest,
        "improvement_receipt_digest": improvement_receipt_digest,
        "battle_evidence": dict(verified_battle_evidence),
        "checks": {
            "model_admission": True,
            "battle_strength": True,
        },
        "admitted": True,
    }
    receipt_digest = _sha256(material)
    receipt = {**material, "receipt_digest": receipt_digest}
    receipt_path = Path(receipts_dir) / f"{receipt_digest.removeprefix('sha256:')}.json"
    _write_immutable_json(receipt_path, receipt)

    promotion_path = Path(promotion_file)
    _promote(
        promotion_path,
        candidate_path=candidate_path,
        candidate_digest=candidate_digest,
        incumbent_digest=incumbent_digest,
        dataset_digest=dataset_digest,
        receipt_digest=receipt_digest,
    )
    return {
        "admitted": True,
        "promotion_file": str(promotion_path),
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_digest": candidate_digest,
        "incumbent_checkpoint_digest": incumbent_digest,
        "receipt": str(receipt_path),
        "receipt_digest": receipt_digest,
        "battle_evidence": dict(verified_battle_evidence),
    }

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
    promote: bool = True,
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
    incumbent_posterior_stress = evaluate_posterior_stress(
        incumbent_params,
        dataset.examples("validation"),
        policy_weight=policy_weight,
    )
    candidate_posterior_stress = evaluate_posterior_stress(
        candidate_params,
        dataset.examples("validation"),
        policy_weight=policy_weight,
    )
    posterior_robustness = evaluate_posterior_robustness(
        incumbent_posterior_stress,
        candidate_posterior_stress,
        policy=admission_policy,
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
        "admitted": bool(
            admission["admitted"]
            and hostile["passed"]
            and posterior_robustness["passed"]
        ),
        "checks": {
            **dict(admission["checks"]),
            "hostile_representation_invariance": bool(hostile["passed"]),
            "posterior_weight_robustness": bool(posterior_robustness["passed"]),
        },
        "failed_checks": [
            *list(admission["failed_checks"]),
            *([] if hostile["passed"] else ["hostile_representation_invariance"]),
            *(
                []
                if posterior_robustness["passed"]
                else ["posterior_weight_robustness"]
            ),
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
            "posterior_stress": {
                "incumbent": incumbent_posterior_stress,
                "candidate": candidate_posterior_stress,
                "comparison": posterior_robustness,
                "used_for_admission": True,
                "alternate_teacher_targets_recomputed": False,
            },
        },
        "admission": admission,
        "promotion_mode": "immediate" if promote else "deferred",
    }
    receipt_digest = _sha256(receipt_material)
    receipt = {**receipt_material, "receipt_digest": receipt_digest}
    receipt_path = Path(receipts_dir) / f"{receipt_digest.removeprefix('sha256:')}.json"
    _write_immutable_json(receipt_path, receipt)

    promotion_path: Path | None = None
    if admission["admitted"] and promote:
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
        "promotion_deferred": bool(admission["admitted"] and not promote),
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
        "posterior_stress": {
            "incumbent": incumbent_posterior_stress,
            "candidate": candidate_posterior_stress,
            "comparison": posterior_robustness,
        },
    }
