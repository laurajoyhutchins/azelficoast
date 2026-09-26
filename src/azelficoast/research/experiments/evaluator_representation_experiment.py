"""Matched experiment for hashed versus Showdown-packed hidden-world representations.

The experiment freezes records, targets, split labels, optimizer settings, public-state
encoding, and action encoding. Only the hidden-world representation/model encoder differs.
It is intentionally research-only: neither treatment is promoted by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.belief.evaluator import (
    BeliefEvaluatorSpec,
    build_evaluator_input,
    forward,
    init_params,
    loss,
)
from azelficoast.belief.packed_evaluator import (
    PackedBeliefEvaluatorInput,
    PackedBeliefEvaluatorSpec,
    build_packed_evaluator_input,
    forward_packed,
    init_packed_params,
    packed_loss,
)
from azelficoast.belief.showdown_packing import ShowdownVocabulary
from azelficoast.belief.training import TrainingExample, train_examples
from azelficoast.research.training_records import (
    TRAINING_SCHEMA,
    TRAINING_SCHEMA_VERSION,
)

EXPERIMENT_SCHEMA = "azelficoast.evaluator-representation-comparison"
EXPERIMENT_SCHEMA_VERSION = 1


class RepresentationExperimentError(ValueError):
    """Raised when a matched representation comparison cannot be constructed."""


@dataclass(frozen=True)
class TreatmentMetrics:
    count: int
    total_loss: float
    value_mse: float
    policy_cross_entropy: float
    policy_accuracy: float
    examples_per_second: float

    def as_record(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "total_loss": self.total_loss,
            "value_mse": self.value_mse,
            "policy_cross_entropy": self.policy_cross_entropy,
            "policy_accuracy": self.policy_accuracy,
            "examples_per_second": self.examples_per_second,
        }


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RepresentationExperimentError(f"{field} must be an object")
    return value


def _records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise RepresentationExperimentError("training dataset is empty")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, Mapping):
        rows = [dict(parsed)]
    elif isinstance(parsed, list):
        if not all(isinstance(row, Mapping) for row in parsed):
            raise RepresentationExperimentError("dataset array contains non-objects")
        rows = [dict(row) for row in parsed]
    else:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RepresentationExperimentError(
                    f"{source}:{line_number}: invalid JSON: {error.msg}"
                ) from error
            if not isinstance(row, Mapping):
                raise RepresentationExperimentError(
                    f"{source}:{line_number}: training record must be an object"
                )
            rows.append(dict(row))
    if not rows:
        raise RepresentationExperimentError("training dataset contains no records")
    return rows


def _target(
    record: Mapping[str, Any],
    *,
    value_target_source: str,
) -> tuple[tuple[str, ...], float, tuple[float, ...]]:
    if (
        record.get("schema") != TRAINING_SCHEMA
        or record.get("schema_version") != TRAINING_SCHEMA_VERSION
    ):
        raise RepresentationExperimentError("unexpected training record schema")
    inputs = _mapping(record.get("input"), "input")
    raw_actions = inputs.get("legal_actions")
    if not isinstance(raw_actions, list) or not raw_actions or not all(
        isinstance(action, str) and action for action in raw_actions
    ):
        raise RepresentationExperimentError(
            "input.legal_actions must be a non-empty string list"
        )
    actions = tuple(raw_actions)
    if len(set(actions)) != len(actions):
        raise RepresentationExperimentError("input.legal_actions must be unique")

    targets = _mapping(record.get("targets"), "targets")
    values = _mapping(targets.get("value"), "targets.value")
    raw_value = values.get(value_target_source)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise RepresentationExperimentError(
            f"targets.value.{value_target_source} must be numeric"
        )
    value = float(raw_value)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise RepresentationExperimentError("value target must be within [-1, 1]")

    policy = _mapping(targets.get("policy"), "targets.policy")
    probabilities = _mapping(
        policy.get("action_probabilities"),
        "targets.policy.action_probabilities",
    )
    if set(probabilities) != set(actions):
        raise RepresentationExperimentError(
            "policy target actions must exactly match legal actions"
        )
    policy_target = tuple(float(probabilities[action]) for action in actions)
    if any(not math.isfinite(value) or value < 0 for value in policy_target):
        raise RepresentationExperimentError(
            "policy target probabilities must be finite and non-negative"
        )
    if sum(policy_target) <= 0:
        raise RepresentationExperimentError("policy target has no positive mass")
    return actions, value, policy_target


def _hashed_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    spec: BeliefEvaluatorSpec,
    value_target_source: str,
) -> tuple[TrainingExample, ...]:
    examples: list[TrainingExample] = []
    for record in records:
        inputs = _mapping(record.get("input"), "input")
        actions, value, policy = _target(
            record,
            value_target_source=value_target_source,
        )
        examples.append(
            TrainingExample(
                inputs=build_evaluator_input(
                    public_state=_mapping(
                        inputs.get("public_state"),
                        "input.public_state",
                    ),
                    posterior=_mapping(inputs.get("posterior"), "input.posterior"),
                    legal_actions=actions,
                    spec=spec,
                ),
                value_target=value,
                policy_target=policy,
            )
        )
    return tuple(examples)


def _packed_examples(
    records: Sequence[Mapping[str, Any]],
    *,
    vocabulary: ShowdownVocabulary,
    spec: PackedBeliefEvaluatorSpec,
    value_target_source: str,
) -> tuple[TrainingExample, ...]:
    examples: list[TrainingExample] = []
    for record in records:
        inputs = _mapping(record.get("input"), "input")
        actions, value, policy = _target(
            record,
            value_target_source=value_target_source,
        )
        examples.append(
            TrainingExample(
                inputs=build_packed_evaluator_input(
                    public_state=_mapping(
                        inputs.get("public_state"),
                        "input.public_state",
                    ),
                    posterior=_mapping(inputs.get("posterior"), "input.posterior"),
                    legal_actions=actions,
                    vocabulary=vocabulary,
                    spec=spec,
                ),
                value_target=value,
                policy_target=policy,
            )
        )
    return tuple(examples)


def _parameter_count(params: Mapping[str, Any]) -> int:
    import numpy as np

    return sum(int(np.asarray(value).size) for value in params.values())


def _hashed_dense_bytes(examples: Sequence[TrainingExample]) -> int:
    total = 0
    for example in examples:
        inputs = example.inputs
        total += 4 * len(inputs.public_features)
        total += 4 * sum(len(row) for row in inputs.world_features)
        total += 4 * len(inputs.world_weights)
        total += 4 * sum(len(row) for row in inputs.action_features)
    return total


def _packed_dense_bytes(examples: Sequence[TrainingExample]) -> int:
    import numpy as np

    total = 0
    for example in examples:
        inputs = example.inputs
        if not isinstance(inputs, PackedBeliefEvaluatorInput):
            raise RepresentationExperimentError("packed example has wrong input type")
        arrays = (
            np.asarray(inputs.public_features, dtype=np.float32),
            np.asarray(inputs.world_weights, dtype=np.float32),
            np.asarray(inputs.species_num, dtype=np.int32),
            np.asarray(inputs.species_forme, dtype=np.int16),
            np.asarray(inputs.level, dtype=np.int16),
            np.asarray(inputs.ability_num, dtype=np.int16),
            np.asarray(inputs.item_num, dtype=np.int16),
            np.asarray(inputs.move_num, dtype=np.int16),
            np.asarray(inputs.move_mask, dtype=np.bool_),
            np.asarray(inputs.tera_type, dtype=np.int8),
            np.asarray(inputs.nature, dtype=np.int8),
            np.asarray(inputs.role, dtype=np.int16),
            np.asarray(inputs.gender, dtype=np.int8),
            np.asarray(inputs.evs, dtype=np.int16),
            np.asarray(inputs.ivs, dtype=np.int8),
            np.asarray(inputs.was_lead, dtype=np.bool_),
            np.asarray(inputs.action_features, dtype=np.float32),
        )
        total += sum(int(array.nbytes) for array in arrays)
    return total


def _metrics(
    params: Mapping[str, Any],
    examples: Sequence[TrainingExample],
    *,
    packed: bool,
) -> TreatmentMetrics:
    import numpy as np

    if not examples:
        raise RepresentationExperimentError("evaluation split is empty")

    start = time.perf_counter_ns()
    total_loss = 0.0
    value_error = 0.0
    policy_error = 0.0
    correct = 0
    for example in examples:
        if packed:
            value, logits = forward_packed(params, example.inputs)
        else:
            value, logits = forward(params, example.inputs)
        value_float = float(value)
        raw_logits = np.asarray(logits, dtype=np.float64)
        shifted = raw_logits - float(np.max(raw_logits))
        probabilities = np.exp(shifted)
        probabilities /= float(np.sum(probabilities))
        target = np.asarray(example.policy_target, dtype=np.float64)
        target /= float(np.sum(target))
        mse = (value_float - example.value_target) ** 2
        cross_entropy = -float(
            np.sum(target * np.log(np.maximum(probabilities, 1e-12)))
        )
        total_loss += mse + cross_entropy
        value_error += mse
        policy_error += cross_entropy
        if int(np.argmax(probabilities)) == int(np.argmax(target)):
            correct += 1
    elapsed = max((time.perf_counter_ns() - start) / 1_000_000_000.0, 1e-12)
    count = len(examples)
    return TreatmentMetrics(
        count=count,
        total_loss=total_loss / count,
        value_mse=value_error / count,
        policy_cross_entropy=policy_error / count,
        policy_accuracy=correct / count,
        examples_per_second=count / elapsed,
    )


def run_representation_comparison(
    records: Sequence[Mapping[str, Any]],
    *,
    vocabulary: ShowdownVocabulary,
    value_target_source: str = "public_belief_search_return",
    epochs: int = 1,
    learning_rate: float = 3e-4,
    policy_weight: float = 1.0,
    seed: int = 0,
    hashed_spec: BeliefEvaluatorSpec = BeliefEvaluatorSpec(),
    packed_spec: PackedBeliefEvaluatorSpec | None = None,
) -> dict[str, Any]:
    """Train and evaluate both representations on exactly the same frozen records."""

    if not records:
        raise RepresentationExperimentError("comparison requires records")
    packed_spec = packed_spec or PackedBeliefEvaluatorSpec.from_vocabulary(
        vocabulary,
        public_width=hashed_spec.public_width,
        action_width=hashed_spec.action_width,
        world_hidden_width=hashed_spec.world_hidden_width,
        hidden_width=hashed_spec.hidden_width,
    )
    if packed_spec.public_width != hashed_spec.public_width:
        raise RepresentationExperimentError("public feature widths must match")
    if packed_spec.action_width != hashed_spec.action_width:
        raise RepresentationExperimentError("action feature widths must match")

    by_split: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        split = record.get("split")
        if split not in {"train", "validation", "test"}:
            raise RepresentationExperimentError(
                "every comparison record needs train/validation/test split"
            )
        by_split.setdefault(str(split), []).append(record)
    train_records = by_split.get("train", [])
    validation_records = by_split.get("validation", [])
    if not train_records or not validation_records:
        raise RepresentationExperimentError(
            "comparison requires non-empty train and validation splits"
        )

    encoding_started = time.perf_counter_ns()
    hashed_train = _hashed_examples(
        train_records,
        spec=hashed_spec,
        value_target_source=value_target_source,
    )
    hashed_validation = _hashed_examples(
        validation_records,
        spec=hashed_spec,
        value_target_source=value_target_source,
    )
    hashed_encoding_seconds = (
        time.perf_counter_ns() - encoding_started
    ) / 1_000_000_000.0

    encoding_started = time.perf_counter_ns()
    packed_train = _packed_examples(
        train_records,
        vocabulary=vocabulary,
        spec=packed_spec,
        value_target_source=value_target_source,
    )
    packed_validation = _packed_examples(
        validation_records,
        vocabulary=vocabulary,
        spec=packed_spec,
        value_target_source=value_target_source,
    )
    packed_encoding_seconds = (
        time.perf_counter_ns() - encoding_started
    ) / 1_000_000_000.0

    hashed_params = init_params(hashed_spec, seed=seed)
    packed_params = init_packed_params(packed_spec, seed=seed)

    hashed_before = _metrics(hashed_params, hashed_validation, packed=False)
    packed_before = _metrics(packed_params, packed_validation, packed=True)

    started = time.perf_counter_ns()
    hashed_trained, hashed_losses = train_examples(
        hashed_params,
        hashed_train,
        epochs=epochs,
        learning_rate=learning_rate,
        policy_weight=policy_weight,
        loss_function=loss,
    )
    hashed_training_seconds = (
        time.perf_counter_ns() - started
    ) / 1_000_000_000.0

    started = time.perf_counter_ns()
    packed_trained, packed_losses = train_examples(
        packed_params,
        packed_train,
        epochs=epochs,
        learning_rate=learning_rate,
        policy_weight=policy_weight,
        loss_function=packed_loss,
    )
    packed_training_seconds = (
        time.perf_counter_ns() - started
    ) / 1_000_000_000.0

    hashed_after = _metrics(hashed_trained, hashed_validation, packed=False)
    packed_after = _metrics(packed_trained, packed_validation, packed=True)

    return {
        "schema": EXPERIMENT_SCHEMA,
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "matched": {
            "record_count": len(records),
            "train_count": len(train_records),
            "validation_count": len(validation_records),
            "test_count": len(by_split.get("test", [])),
            "value_target_source": value_target_source,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "policy_weight": policy_weight,
            "seed": seed,
            "public_width": hashed_spec.public_width,
            "action_width": hashed_spec.action_width,
            "vocabulary_sha256": vocabulary.vocabulary_sha256,
        },
        "hashed_control": {
            "architecture": "hashed-hidden-world-deep-sets",
            "parameter_count": _parameter_count(hashed_params),
            "encoding_seconds": hashed_encoding_seconds,
            "dense_input_bytes": _hashed_dense_bytes(
                (*hashed_train, *hashed_validation)
            ),
            "training_seconds": hashed_training_seconds,
            "training_examples_per_second": (
                len(hashed_train) * epochs / max(hashed_training_seconds, 1e-12)
            ),
            "initial_validation": hashed_before.as_record(),
            "trained_validation": hashed_after.as_record(),
            "first_training_loss": float(hashed_losses[0]),
            "last_training_loss": float(hashed_losses[-1]),
        },
        "showdown_packed_treatment": {
            "architecture": "showdown-native-categorical-joint-team-deep-sets",
            "parameter_count": _parameter_count(packed_params),
            "encoding_seconds": packed_encoding_seconds,
            "dense_input_bytes": _packed_dense_bytes(
                (*packed_train, *packed_validation)
            ),
            "training_seconds": packed_training_seconds,
            "training_examples_per_second": (
                len(packed_train) * epochs / max(packed_training_seconds, 1e-12)
            ),
            "initial_validation": packed_before.as_record(),
            "trained_validation": packed_after.as_record(),
            "first_training_loss": float(packed_losses[0]),
            "last_training_loss": float(packed_losses[-1]),
        },
        "claim": (
            "This compares representation treatments on identical frozen records and "
            "targets. It does not promote either evaluator or claim battle strength."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("vocabulary", type=Path)
    parser.add_argument(
        "--value-target-source",
        default="public_belief_search_return",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    vocabulary_document = json.loads(args.vocabulary.read_text(encoding="utf-8"))
    if not isinstance(vocabulary_document, Mapping):
        raise RepresentationExperimentError("vocabulary must be an object")
    result = run_representation_comparison(
        _records(args.dataset),
        vocabulary=ShowdownVocabulary.from_record(vocabulary_document),
        value_target_source=args.value_target_source,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        policy_weight=args.policy_weight,
        seed=args.seed,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
