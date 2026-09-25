"""Shared learned value/policy evaluator over public belief states.

The evaluator consumes only public state, a posterior over plausible hidden worlds,
and legal actions. It never consumes the realized hidden state. Hidden-world support is
pooled permutation-invariantly so determinization and information-set search can share
one learned evaluator without giving either method a private-information side channel.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.core.evaluation import EvaluationLeaf
from azelficoast.research.contracts import (
    BeliefInput,
    PublicDecisionInput,
    PublicSuccessorState,
    ResearchContractError,
)

EVALUATOR_SCHEMA = "azelficoast.belief-policy-value-evaluator"
EVALUATOR_SCHEMA_VERSION = 1
INPUT_SCHEMA = "azelficoast.public-belief-evaluator-input"
INPUT_SCHEMA_VERSION = 1


class BeliefEvaluatorError(ValueError):
    """Raised when evaluator input or checkpoint metadata violates the contract."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _bucket(token: str, width: int) -> tuple[int, float]:
    if width <= 0:
        raise BeliefEvaluatorError("feature width must be positive")
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % width
    sign = 1.0 if digest[8] & 1 else -1.0
    return index, sign


def _flatten(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, Mapping):
        rows: list[tuple[str, Any]] = []
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.extend(_flatten(value[key], child))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for index, child_value in enumerate(value):
            child = f"{prefix}[{index}]"
            rows.extend(_flatten(child_value, child))
        return rows
    return [(prefix, value)]


def hashed_features(value: Any, *, width: int) -> tuple[float, ...]:
    """Map arbitrary structured public data to a stable fixed-width numeric vector."""
    output = [0.0] * width
    for path, scalar in _flatten(value):
        if scalar is None:
            index, sign = _bucket(f"{path}=<none>", width)
            output[index] += sign
            continue
        if isinstance(scalar, bool):
            index, sign = _bucket(f"{path}=bool", width)
            output[index] += sign * (1.0 if scalar else -1.0)
            continue
        if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
            numeric = float(scalar)
            if not math.isfinite(numeric):
                raise BeliefEvaluatorError(f"non-finite numeric feature at {path}")
            index, sign = _bucket(f"{path}=number", width)
            output[index] += sign * math.copysign(math.log1p(abs(numeric)), numeric)
            continue
        index, sign = _bucket(f"{path}={scalar}", width)
        output[index] += sign
    return tuple(output)


@dataclass(frozen=True)
class BeliefEvaluatorSpec:
    public_width: int = 256
    world_width: int = 192
    action_width: int = 96
    hidden_width: int = 256
    world_hidden_width: int = 192

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BeliefEvaluatorError(f"{name} must be a positive integer")

    def as_dict(self) -> dict[str, int]:
        return {
            "public_width": self.public_width,
            "world_width": self.world_width,
            "action_width": self.action_width,
            "hidden_width": self.hidden_width,
            "world_hidden_width": self.world_hidden_width,
        }


@dataclass(frozen=True)
class BeliefEvaluatorInput:
    public_features: tuple[float, ...]
    world_features: tuple[tuple[float, ...], ...]
    world_weights: tuple[float, ...]
    action_features: tuple[tuple[float, ...], ...]
    legal_actions: tuple[str, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": INPUT_SCHEMA,
            "schema_version": INPUT_SCHEMA_VERSION,
            "public_features": list(self.public_features),
            "world_features": [list(row) for row in self.world_features],
            "world_weights": list(self.world_weights),
            "action_features": [list(row) for row in self.action_features],
            "legal_actions": list(self.legal_actions),
        }


def build_evaluator_input(
    *,
    public_state: Mapping[str, Any],
    posterior: Mapping[str, Any],
    legal_actions: Sequence[str],
    spec: BeliefEvaluatorSpec = BeliefEvaluatorSpec(),
) -> BeliefEvaluatorInput:
    """Build one evaluator input while enforcing the no-realized-state boundary."""
    if posterior.get("conditioned_on_public_history") is not True:
        raise BeliefEvaluatorError("posterior must be conditioned on public history")
    if posterior.get("realized_hidden_state_revealed") is not False:
        raise BeliefEvaluatorError("evaluator may not consume a revealed hidden state")
    try:
        public = PublicSuccessorState.from_record(public_state)
        belief = BeliefInput.from_record(posterior)
        return build_evaluator_input_for_contract(
            public_state=public,
            belief=belief,
            legal_actions=legal_actions,
            spec=spec,
        )
    except ResearchContractError as error:
        raise BeliefEvaluatorError(str(error)) from error


_MECHANICS_ONLY_HIDDEN_FIELDS = frozenset({"opponent.bench"})


def _evaluator_world_record(world: Any) -> dict[str, Any]:
    """Project mechanics-private state out of the learned evaluator surface."""

    record = world.features.to_record()
    hidden = record.get("hidden")
    if isinstance(hidden, Mapping):
        record = {
            **record,
            "hidden": {
                str(key): value
                for key, value in hidden.items()
                if str(key) not in _MECHANICS_ONLY_HIDDEN_FIELDS
            },
        }
    return record


def build_evaluator_input_for_contract(
    *,
    public_state: PublicDecisionInput | PublicSuccessorState,
    belief: BeliefInput,
    legal_actions: Sequence[str],
    spec: BeliefEvaluatorSpec = BeliefEvaluatorSpec(),
) -> BeliefEvaluatorInput:
    """Build model features from validated public and posterior contracts only."""
    actions = tuple(legal_actions)
    if not actions or any(not action for action in actions):
        raise BeliefEvaluatorError("legal actions must be non-empty strings")
    if len(set(actions)) != len(actions):
        raise BeliefEvaluatorError("legal actions must be unique")

    state_record = public_state.public_state.to_record()
    return BeliefEvaluatorInput(
        public_features=hashed_features(state_record, width=spec.public_width),
        world_features=tuple(
            hashed_features(_evaluator_world_record(world), width=spec.world_width)
            for world in belief.model_worlds
        ),
        world_weights=tuple(world.weight for world in belief.model_worlds),
        action_features=tuple(
            hashed_features({"action": action}, width=spec.action_width)
            for action in actions
        ),
        legal_actions=actions,
    )


def checkpoint_digest(params: Mapping[str, Any], spec: BeliefEvaluatorSpec) -> str:
    """Content-address a parameter tree independent of container serialization."""
    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError("numpy is required to digest evaluator parameters") from error

    chunks = [_canonical(spec.as_dict()).encode("utf-8")]
    for name in sorted(params):
        value = np.asarray(params[name])
        chunks.extend(
            (
                name.encode("utf-8"),
                str(value.dtype).encode("ascii"),
                _canonical(list(value.shape)).encode("ascii"),
                value.tobytes(order="C"),
            )
        )
    return _sha256_bytes(b"\0".join(chunks))


def evaluator_identity(
    *,
    checkpoint_digest_value: str,
    spec: BeliefEvaluatorSpec,
) -> dict[str, Any]:
    if not (
        checkpoint_digest_value.startswith("sha256:")
        and len(checkpoint_digest_value) == 71
        and all(ch in "0123456789abcdef" for ch in checkpoint_digest_value[7:])
    ):
        raise BeliefEvaluatorError("checkpoint digest must be sha256:<64 lowercase hex>")
    return {
        "schema": EVALUATOR_SCHEMA,
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "checkpoint_digest": checkpoint_digest_value,
        "observability": "public_belief_only",
        "architecture": "weighted_deep_sets_policy_value",
        "spec": spec.as_dict(),
    }


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as error:
        raise BeliefEvaluatorError(
            "JAX is required for learned evaluator initialization and inference; "
            "install the simulator extra"
        ) from error
    return jax, jnp


def _glorot(key: Any, fan_in: int, fan_out: int) -> Any:
    jax, jnp = _require_jax()
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(
        key,
        (fan_in, fan_out),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )


def init_params(spec: BeliefEvaluatorSpec, *, seed: int = 0) -> dict[str, Any]:
    """Initialize a dependency-light JAX network without Flax or Optax."""
    jax, jnp = _require_jax()
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 10))

    def dense(name: str, fan_in: int, fan_out: int, params: dict[str, Any]) -> None:
        params[f"{name}.weight"] = _glorot(next(keys), fan_in, fan_out)
        params[f"{name}.bias"] = jnp.zeros((fan_out,), dtype=jnp.float32)

    params: dict[str, Any] = {}
    dense("public", spec.public_width, spec.hidden_width, params)
    dense("world", spec.world_width, spec.world_hidden_width, params)
    dense("world_post", spec.world_hidden_width * 2 + 2, spec.hidden_width, params)
    dense("trunk", spec.hidden_width * 2, spec.hidden_width, params)
    dense("action", spec.action_width, spec.hidden_width, params)
    dense("policy_context", spec.hidden_width, spec.hidden_width, params)
    dense("value_hidden", spec.hidden_width, spec.hidden_width, params)
    dense("value", spec.hidden_width, 1, params)
    return params


def _dense(jnp: Any, x: Any, params: Mapping[str, Any], name: str) -> Any:
    return x @ params[f"{name}.weight"] + params[f"{name}.bias"]


def _belief_trunk(
    jnp: Any,
    params: Mapping[str, Any],
    public: Any,
    worlds: Any,
    weights: Any,
) -> Any:
    """Pool one public belief without exposing transport or realized-world identity."""

    public_hidden = jnp.tanh(_dense(jnp, public, params, "public"))
    world_hidden = jnp.tanh(_dense(jnp, worlds, params, "world"))
    mean = jnp.sum(world_hidden * weights[:, None], axis=0)
    centered = world_hidden - mean[None, :]
    variance = jnp.sum(centered * centered * weights[:, None], axis=0)
    entropy = -jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-12)))
    effective_support = jnp.exp(entropy)
    belief_summary = jnp.concatenate(
        (mean, variance, jnp.stack((entropy, effective_support)).astype(jnp.float32))
    )
    belief_hidden = jnp.tanh(_dense(jnp, belief_summary, params, "world_post"))
    return jnp.tanh(
        _dense(
            jnp,
            jnp.concatenate((public_hidden, belief_hidden)),
            params,
            "trunk",
        )
    )


def _value_from_trunk(jnp: Any, params: Mapping[str, Any], trunk: Any) -> Any:
    value_hidden = jnp.tanh(_dense(jnp, trunk, params, "value_hidden"))
    return jnp.tanh(_dense(jnp, value_hidden, params, "value"))[0]


def forward(
    params: Mapping[str, Any],
    inputs: BeliefEvaluatorInput,
) -> tuple[Any, Any]:
    """Return value and policy logits for one public-belief state."""
    _, jnp = _require_jax()
    public = jnp.asarray(inputs.public_features, dtype=jnp.float32)
    worlds = jnp.asarray(inputs.world_features, dtype=jnp.float32)
    weights = jnp.asarray(inputs.world_weights, dtype=jnp.float32)
    actions = jnp.asarray(inputs.action_features, dtype=jnp.float32)

    trunk = _belief_trunk(jnp, params, public, worlds, weights)
    action_hidden = jnp.tanh(_dense(jnp, actions, params, "action"))
    policy_context = jnp.tanh(_dense(jnp, trunk, params, "policy_context"))
    logits = action_hidden @ policy_context / math.sqrt(float(policy_context.shape[-1]))
    return _value_from_trunk(jnp, params, trunk), logits


_BATCH_VALUE_FUNCTION: Any | None = None


def _batched_value_function() -> Any:
    """Return one cached JIT for shape-bucketed successor-belief evaluation."""

    global _BATCH_VALUE_FUNCTION
    if _BATCH_VALUE_FUNCTION is None:
        jax, jnp = _require_jax()

        def evaluate(
            params: Mapping[str, Any],
            public: Any,
            worlds: Any,
            weights: Any,
        ) -> Any:
            def one(public_row: Any, world_rows: Any, weight_row: Any) -> Any:
                trunk = _belief_trunk(
                    jnp,
                    params,
                    public_row,
                    world_rows,
                    weight_row,
                )
                return _value_from_trunk(jnp, params, trunk)

            return jax.vmap(one)(public, worlds, weights)

        _BATCH_VALUE_FUNCTION = jax.jit(evaluate)
    return _BATCH_VALUE_FUNCTION


def _shape_bucket(size: int) -> int:
    if size <= 0:
        raise BeliefEvaluatorError("batched evaluator dimensions must be positive")
    return 1 << (size - 1).bit_length()


def predict_values(
    params: Mapping[str, Any],
    inputs: Sequence[BeliefEvaluatorInput],
) -> tuple[float, ...]:
    """Evaluate a whole search frontier in one shape-bucketed JAX dispatch."""

    if not inputs:
        return ()
    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError("numpy is required for batched evaluator inference") from error

    public_width = len(inputs[0].public_features)
    if not inputs[0].world_features:
        raise BeliefEvaluatorError("batched evaluator input has no posterior worlds")
    world_width = len(inputs[0].world_features[0])
    batch_size = _shape_bucket(len(inputs))
    world_count = _shape_bucket(max(len(row.world_features) for row in inputs))

    public = np.zeros((batch_size, public_width), dtype=np.float32)
    worlds = np.zeros((batch_size, world_count, world_width), dtype=np.float32)
    weights = np.zeros((batch_size, world_count), dtype=np.float32)

    for index, row in enumerate(inputs):
        if len(row.public_features) != public_width:
            raise BeliefEvaluatorError("batched evaluator public feature widths differ")
        if not row.world_features or len(row.world_features) != len(row.world_weights):
            raise BeliefEvaluatorError("batched evaluator posterior shape is invalid")
        if any(len(world) != world_width for world in row.world_features):
            raise BeliefEvaluatorError("batched evaluator world feature widths differ")
        count = len(row.world_features)
        public[index] = np.asarray(row.public_features, dtype=np.float32)
        worlds[index, :count] = np.asarray(row.world_features, dtype=np.float32)
        weights[index, :count] = np.asarray(row.world_weights, dtype=np.float32)

    raw = _batched_value_function()(params, public, worlds, weights)
    values = np.asarray(raw, dtype=np.float64)[: len(inputs)]
    if values.ndim != 1 or values.shape[0] != len(inputs):
        raise BeliefEvaluatorError("batched value head returned the wrong shape")
    if not np.all(np.isfinite(values)):
        raise BeliefEvaluatorError("batched value head returned non-finite values")
    return tuple(float(value) for value in values)


def loss(
    params: Mapping[str, Any],
    inputs: BeliefEvaluatorInput,
    *,
    value_target: float,
    policy_target: Sequence[float],
    policy_weight: float = 1.0,
) -> Any:
    """Joint value MSE plus policy cross-entropy objective."""
    _, jnp = _require_jax()
    if len(policy_target) != len(inputs.legal_actions):
        raise BeliefEvaluatorError("policy target must cover every legal action")
    target = jnp.asarray(policy_target, dtype=jnp.float32)
    target = target / jnp.maximum(jnp.sum(target), 1e-12)
    value, logits = forward(params, inputs)
    maximum = jnp.max(logits)
    log_probs = logits - maximum - jnp.log(jnp.sum(jnp.exp(logits - maximum)))
    value_error = (value - jnp.asarray(value_target, dtype=jnp.float32)) ** 2
    policy_error = -jnp.sum(target * log_probs)
    return value_error + float(policy_weight) * policy_error


CHECKPOINT_SCHEMA = "azelficoast.belief-policy-value-checkpoint"
CHECKPOINT_SCHEMA_VERSION = 1

PROMOTION_SCHEMA = "azelficoast.evaluator-promotion"
PROMOTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BeliefPrediction:
    """One learned evaluation with diagnostics suitable for search routing."""

    value: float
    legal_actions: tuple[str, ...]
    probabilities: tuple[float, ...]
    selected_action: str
    policy_margin: float
    policy_entropy_bits: float

    def as_record(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "legal_actions": list(self.legal_actions),
            "probabilities": list(self.probabilities),
            "selected_action": self.selected_action,
            "policy_margin": self.policy_margin,
            "policy_entropy_bits": self.policy_entropy_bits,
        }


def predict(
    params: Mapping[str, Any],
    inputs: BeliefEvaluatorInput,
) -> BeliefPrediction:
    """Evaluate one belief state and expose calibrated-routing diagnostics."""
    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError("numpy is required for evaluator inference") from error

    raw_value, raw_logits = forward(params, inputs)
    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.ndim != 1 or logits.shape[0] != len(inputs.legal_actions):
        raise BeliefEvaluatorError("policy head returned the wrong action shape")
    if not np.all(np.isfinite(logits)):
        raise BeliefEvaluatorError("policy head returned non-finite logits")

    shifted = logits - float(np.max(logits))
    probabilities_array = np.exp(shifted)
    probabilities_array /= float(np.sum(probabilities_array))
    probabilities = tuple(float(value) for value in probabilities_array)

    maximum = max(probabilities)
    selected_action = min(
        action
        for action, probability in zip(inputs.legal_actions, probabilities, strict=True)
        if abs(probability - maximum) <= 1e-15
    )
    ordered = sorted(probabilities, reverse=True)
    margin = 1.0 if len(ordered) == 1 else ordered[0] - ordered[1]
    entropy = -sum(
        probability * math.log2(probability)
        for probability in probabilities
        if probability > 0.0
    )
    value = float(raw_value)
    if not math.isfinite(value):
        raise BeliefEvaluatorError("value head returned a non-finite value")

    return BeliefPrediction(
        value=value,
        legal_actions=inputs.legal_actions,
        probabilities=probabilities,
        selected_action=selected_action,
        policy_margin=float(margin),
        policy_entropy_bits=float(entropy),
    )


def write_checkpoint(
    directory: str | Path,
    params: Mapping[str, Any],
    spec: BeliefEvaluatorSpec,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a content-addressed checkpoint and return its verified manifest."""
    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError("numpy is required to write evaluator checkpoints") from error

    destination = Path(directory)
    manifest_path = destination / "manifest.json"
    params_path = destination / "params.npz"
    if not overwrite and (manifest_path.exists() or params_path.exists()):
        raise BeliefEvaluatorError(
            f"checkpoint destination already contains evaluator files: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)

    ordered_names = sorted(params)
    arrays: dict[str, Any] = {}
    parameter_map: dict[str, str] = {}
    for index, name in enumerate(ordered_names):
        key = f"p{index:04d}"
        value = np.asarray(params[name])
        if value.dtype.kind not in {"f", "i", "u", "b"}:
            raise BeliefEvaluatorError(f"unsupported parameter dtype for {name}: {value.dtype}")
        arrays[key] = value
        parameter_map[name] = key

    digest = checkpoint_digest(params, spec)
    identity = evaluator_identity(checkpoint_digest_value=digest, spec=spec)
    manifest = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "evaluator": identity,
        "parameter_map": parameter_map,
        "metadata": dict(metadata or {}),
    }
    np.savez_compressed(params_path, **arrays)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_checkpoint(
    directory: str | Path,
) -> tuple[dict[str, Any], BeliefEvaluatorSpec, dict[str, Any]]:
    """Load a checkpoint only after recomputing and matching its content digest."""
    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError("numpy is required to load evaluator checkpoints") from error

    source = Path(directory)
    expected_promoted_digest: str | None = None
    if source.is_file():
        pointer_path = source
        try:
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise BeliefEvaluatorError(f"cannot read checkpoint promotion pointer: {error}") from error
        if not isinstance(pointer, Mapping):
            raise BeliefEvaluatorError("checkpoint promotion pointer must be an object")
        if (
            pointer.get("schema") != PROMOTION_SCHEMA
            or pointer.get("schema_version") != PROMOTION_SCHEMA_VERSION
        ):
            raise BeliefEvaluatorError("unexpected checkpoint promotion schema")
        checkpoint_reference = pointer.get("checkpoint")
        expected_promoted_digest = pointer.get("checkpoint_digest")
        if not isinstance(checkpoint_reference, str) or not checkpoint_reference:
            raise BeliefEvaluatorError("checkpoint promotion pointer is missing checkpoint")
        if not (
            isinstance(expected_promoted_digest, str)
            and expected_promoted_digest.startswith("sha256:")
            and len(expected_promoted_digest) == 71
        ):
            raise BeliefEvaluatorError("checkpoint promotion pointer has an invalid digest")
        source = Path(checkpoint_reference)
        if not source.is_absolute():
            source = pointer_path.parent / source

    try:
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BeliefEvaluatorError(f"cannot read checkpoint manifest: {error}") from error
    if not isinstance(manifest, Mapping):
        raise BeliefEvaluatorError("checkpoint manifest must be an object")
    if (
        manifest.get("schema") != CHECKPOINT_SCHEMA
        or manifest.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise BeliefEvaluatorError("unexpected checkpoint schema")

    evaluator = manifest.get("evaluator")
    parameter_map = manifest.get("parameter_map")
    if not isinstance(evaluator, Mapping) or not isinstance(parameter_map, Mapping):
        raise BeliefEvaluatorError("checkpoint manifest is incomplete")
    raw_spec = evaluator.get("spec")
    if not isinstance(raw_spec, Mapping):
        raise BeliefEvaluatorError("checkpoint evaluator spec is missing")
    try:
        spec = BeliefEvaluatorSpec(**{str(key): int(value) for key, value in raw_spec.items()})
    except (TypeError, ValueError) as error:
        raise BeliefEvaluatorError(f"invalid checkpoint evaluator spec: {error}") from error

    try:
        archive = np.load(source / "params.npz", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise BeliefEvaluatorError(f"cannot read checkpoint parameters: {error}") from error
    try:
        params: dict[str, Any] = {}
        expected_keys = {str(value) for value in parameter_map.values()}
        if set(archive.files) != expected_keys:
            raise BeliefEvaluatorError("checkpoint parameter archive differs from manifest")
        for name, key in parameter_map.items():
            if not isinstance(name, str) or not isinstance(key, str):
                raise BeliefEvaluatorError("checkpoint parameter map must contain strings")
            params[name] = np.asarray(archive[key])
    finally:
        archive.close()

    actual = checkpoint_digest(params, spec)
    if evaluator.get("checkpoint_digest") != actual:
        raise BeliefEvaluatorError("checkpoint content digest does not match manifest")
    if expected_promoted_digest is not None and actual != expected_promoted_digest:
        raise BeliefEvaluatorError(
            "promoted checkpoint content digest does not match promotion pointer"
        )
    expected_identity = evaluator_identity(checkpoint_digest_value=actual, spec=spec)
    if dict(evaluator) != expected_identity:
        raise BeliefEvaluatorError("checkpoint evaluator identity is inconsistent")
    return params, spec, dict(manifest)


class BeliefEvaluatorRuntime:
    """Loaded learned evaluator with immutable, content-addressed identity."""

    def __init__(
        self,
        params: Mapping[str, Any],
        spec: BeliefEvaluatorSpec,
        identity: Mapping[str, Any],
    ) -> None:
        actual = checkpoint_digest(params, spec)
        expected = evaluator_identity(checkpoint_digest_value=actual, spec=spec)
        if dict(identity) != expected:
            raise BeliefEvaluatorError("runtime evaluator identity does not match parameters")
        self.params = dict(params)
        self.spec = spec
        self.identity = expected

    @classmethod
    def from_checkpoint(cls, directory: str | Path) -> "BeliefEvaluatorRuntime":
        params, spec, manifest = load_checkpoint(directory)
        evaluator = manifest["evaluator"]
        assert isinstance(evaluator, Mapping)
        return cls(params, spec, evaluator)

    def predict(self, inputs: BeliefEvaluatorInput) -> BeliefPrediction:
        return predict(self.params, inputs)

    def predict_values(
        self,
        inputs: Sequence[BeliefEvaluatorInput],
    ) -> tuple[float, ...]:
        return predict_values(self.params, inputs)


class BeliefSearchValueAdapter:
    """Expose a learned belief evaluator through the domain-neutral search contract."""

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    def values(self, leaves: Sequence[EvaluationLeaf]) -> tuple[float, ...]:
        """Evaluate an already-constructed search frontier in one backend dispatch."""

        inputs: list[BeliefEvaluatorInput] = []
        for leaf in leaves:
            if not isinstance(leaf.public_state, Mapping):
                raise BeliefEvaluatorError("search leaf public state must be a mapping")
            if not isinstance(leaf.posterior, Sequence):
                raise BeliefEvaluatorError("search leaf posterior must be a sequence")
            posterior = {
                "conditioned_on_public_history": True,
                "realized_hidden_state_revealed": False,
                "worlds": [dict(world) for world in leaf.posterior],
            }
            inputs.append(
                build_evaluator_input(
                    public_state=leaf.public_state,
                    posterior=posterior,
                    legal_actions=leaf.legal_actions,
                    spec=self.evaluator.spec,
                )
            )
        predict_many = getattr(self.evaluator, "predict_values", None)
        if callable(predict_many):
            return tuple(float(value) for value in predict_many(tuple(inputs)))
        return tuple(float(self.evaluator.predict(row).value) for row in inputs)

    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        posterior = {
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "worlds": [dict(world) for world in posterior_worlds],
        }
        try:
            inputs = build_evaluator_input(
                public_state=public_state,
                posterior=posterior,
                legal_actions=legal_actions,
                spec=self.evaluator.spec,
            )
            prediction = self.evaluator.predict(inputs)
        except Exception as error:
            raise BeliefEvaluatorError(
                f"learned evaluator failed at successor leaf: {error}"
            ) from error

        value = float(prediction.value)
        if not math.isfinite(value):
            raise BeliefEvaluatorError(
                "learned evaluator returned a non-finite successor value"
            )
        return value
