"""JAX evaluator over Showdown-native packed joint posterior data.

This is a matched alternative to the hashed hidden-world representation in
:mod:`azelficoast.belief.evaluator`. Public-state and action encoding remain hashed so
experiments can isolate the effect of replacing hidden-world feature hashing with the
revision-bound categorical coordinates produced by `showdown_packing`.

The packed representation is not semantic authority. The builder first validates the
joint posterior and its exact Showdown-bound vocabulary, then lowers it into arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from azelficoast.belief.evaluator import (
    BeliefEvaluatorError,
    BeliefPrediction,
    _require_jax,
    _shape_bucket,
    hashed_features,
)
from azelficoast.belief.showdown_packing import (
    PackedJointPosterior,
    ShowdownPackingError,
    ShowdownVocabulary,
    pack_joint_posterior,
)
from azelficoast.research.contracts import PublicSuccessorState, ResearchContractError

PACKED_EVALUATOR_SCHEMA = "azelficoast.showdown-packed-belief-evaluator"
PACKED_EVALUATOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PackedBeliefEvaluatorSpec:
    """Shape and embedding contract for one Showdown-bound evaluator treatment."""

    vocabulary_sha256: str
    species_vocab_size: int
    forme_vocab_size: int
    move_vocab_size: int
    item_vocab_size: int
    ability_vocab_size: int
    type_vocab_size: int
    nature_vocab_size: int
    role_vocab_size: int
    public_width: int = 256
    action_width: int = 96
    embedding_width: int = 24
    member_hidden_width: int = 128
    world_hidden_width: int = 192
    hidden_width: int = 256

    def __post_init__(self) -> None:
        if not (
            self.vocabulary_sha256.startswith("sha256:")
            and len(self.vocabulary_sha256) == 71
        ):
            raise BeliefEvaluatorError("packed evaluator requires a vocabulary digest")
        for name, value in self.as_dict().items():
            if name == "vocabulary_sha256":
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise BeliefEvaluatorError(f"{name} must be a positive integer")

    @classmethod
    def from_vocabulary(
        cls,
        vocabulary: ShowdownVocabulary,
        **overrides: int,
    ) -> "PackedBeliefEvaluatorSpec":
        """Derive safe embedding table sizes from one exact vocabulary."""

        def maximum(values: Sequence[int]) -> int:
            if not values:
                raise BeliefEvaluatorError("packed evaluator vocabulary table is empty")
            return max(values) + 1

        species_values = [pair[0] for pair in vocabulary.species.values()]
        forme_values = [pair[1] for pair in vocabulary.species.values()]
        base: dict[str, Any] = {
            "vocabulary_sha256": vocabulary.vocabulary_sha256,
            "species_vocab_size": maximum(species_values),
            "forme_vocab_size": maximum(forme_values),
            "move_vocab_size": maximum(list(vocabulary.moves.values())),
            "item_vocab_size": maximum(list(vocabulary.items.values())),
            "ability_vocab_size": maximum(list(vocabulary.abilities.values())),
            "type_vocab_size": maximum(list(vocabulary.types.values())),
            "nature_vocab_size": maximum(list(vocabulary.natures.values())),
            "role_vocab_size": maximum(list(vocabulary.roles.values())),
        }
        base.update(overrides)
        return cls(**base)

    def as_dict(self) -> dict[str, str | int]:
        return {
            "vocabulary_sha256": self.vocabulary_sha256,
            "species_vocab_size": self.species_vocab_size,
            "forme_vocab_size": self.forme_vocab_size,
            "move_vocab_size": self.move_vocab_size,
            "item_vocab_size": self.item_vocab_size,
            "ability_vocab_size": self.ability_vocab_size,
            "type_vocab_size": self.type_vocab_size,
            "nature_vocab_size": self.nature_vocab_size,
            "role_vocab_size": self.role_vocab_size,
            "public_width": self.public_width,
            "action_width": self.action_width,
            "embedding_width": self.embedding_width,
            "member_hidden_width": self.member_hidden_width,
            "world_hidden_width": self.world_hidden_width,
            "hidden_width": self.hidden_width,
        }


@dataclass(frozen=True)
class PackedBeliefEvaluatorInput:
    """One model input with structured hidden worlds and hashed public/action context."""

    public_features: tuple[float, ...]
    world_weights: tuple[float, ...]
    species_num: tuple[tuple[int, ...], ...]
    species_forme: tuple[tuple[int, ...], ...]
    level: tuple[tuple[int, ...], ...]
    ability_num: tuple[tuple[int, ...], ...]
    item_num: tuple[tuple[int, ...], ...]
    move_num: tuple[tuple[tuple[int, ...], ...], ...]
    move_mask: tuple[tuple[tuple[bool, ...], ...], ...]
    tera_type: tuple[tuple[int, ...], ...]
    nature: tuple[tuple[int, ...], ...]
    role: tuple[tuple[int, ...], ...]
    gender: tuple[tuple[int, ...], ...]
    evs: tuple[tuple[tuple[int, ...], ...], ...]
    ivs: tuple[tuple[tuple[int, ...], ...], ...]
    was_lead: tuple[tuple[bool, ...], ...]
    action_features: tuple[tuple[float, ...], ...]
    legal_actions: tuple[str, ...]
    source_digest: str
    vocabulary_sha256: str

    @property
    def world_count(self) -> int:
        return len(self.world_weights)

    @property
    def team_size(self) -> int:
        return len(self.species_num[0]) if self.species_num else 0


def _check_ids(
    packed: PackedJointPosterior,
    spec: PackedBeliefEvaluatorSpec,
) -> None:
    checks: tuple[tuple[str, Sequence[Any], int], ...] = (
        ("species", packed.species_num, spec.species_vocab_size),
        ("forme", packed.species_forme, spec.forme_vocab_size),
        ("ability", packed.ability_num, spec.ability_vocab_size),
        ("item", packed.item_num, spec.item_vocab_size),
        ("tera type", packed.tera_type, spec.type_vocab_size),
        ("nature", packed.nature, spec.nature_vocab_size),
        ("role", packed.role, spec.role_vocab_size),
    )
    for label, rows, size in checks:
        for row in rows:
            for raw in row:
                value = int(raw)
                if not 0 <= value < size:
                    raise BeliefEvaluatorError(
                        f"packed {label} id {value} exceeds evaluator table size {size}"
                    )
    for world in packed.move_num:
        for member in world:
            for raw in member:
                value = int(raw)
                if not 0 <= value < spec.move_vocab_size:
                    raise BeliefEvaluatorError(
                        f"packed move id {value} exceeds evaluator table size "
                        f"{spec.move_vocab_size}"
                    )


def build_packed_evaluator_input(
    *,
    public_state: Mapping[str, Any],
    posterior: Mapping[str, Any],
    legal_actions: Sequence[str],
    vocabulary: ShowdownVocabulary,
    spec: PackedBeliefEvaluatorSpec,
) -> PackedBeliefEvaluatorInput:
    """Build one matched evaluator input from validated joint posterior evidence."""

    if spec.vocabulary_sha256 != vocabulary.vocabulary_sha256:
        raise BeliefEvaluatorError("packed evaluator spec and vocabulary differ")
    actions = tuple(legal_actions)
    if not actions or any(not action for action in actions):
        raise BeliefEvaluatorError("legal actions must be non-empty strings")
    if len(set(actions)) != len(actions):
        raise BeliefEvaluatorError("legal actions must be unique")

    try:
        public = PublicSuccessorState.from_record(public_state)
        packed = pack_joint_posterior(posterior, vocabulary)
    except (ResearchContractError, ShowdownPackingError) as error:
        raise BeliefEvaluatorError(str(error)) from error

    _check_ids(packed, spec)
    if packed.vocabulary_sha256 != spec.vocabulary_sha256:
        raise BeliefEvaluatorError("packed posterior vocabulary identity drifted")

    return PackedBeliefEvaluatorInput(
        public_features=hashed_features(
            public.public_state.to_record(),
            width=spec.public_width,
        ),
        world_weights=packed.weights,
        species_num=packed.species_num,
        species_forme=packed.species_forme,
        level=packed.level,
        ability_num=packed.ability_num,
        item_num=packed.item_num,
        move_num=packed.move_num,
        move_mask=packed.move_mask,
        tera_type=packed.tera_type,
        nature=packed.nature,
        role=packed.role,
        gender=packed.gender,
        evs=packed.evs,
        ivs=packed.ivs,
        was_lead=packed.was_lead,
        action_features=tuple(
            hashed_features({"action": action}, width=spec.action_width)
            for action in actions
        ),
        legal_actions=actions,
        source_digest=packed.source_digest,
        vocabulary_sha256=packed.vocabulary_sha256,
    )


def _glorot(key: Any, fan_in: int, fan_out: int) -> Any:
    jax, jnp = _require_jax()
    limit = math.sqrt(6.0 / float(fan_in + fan_out))
    return jax.random.uniform(
        key,
        (fan_in, fan_out),
        minval=-limit,
        maxval=limit,
        dtype=jnp.float32,
    )


def _embedding(key: Any, size: int, width: int) -> Any:
    jax, jnp = _require_jax()
    values = jax.random.normal(key, (size, width), dtype=jnp.float32) * 0.02
    return values.at[0].set(jnp.zeros((width,), dtype=jnp.float32))


def init_packed_params(
    spec: PackedBeliefEvaluatorSpec,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Initialize the packed treatment without Flax/Optax dependencies."""

    jax, jnp = _require_jax()
    keys = iter(jax.random.split(jax.random.PRNGKey(seed), 24))
    params: dict[str, Any] = {}

    for name, size in (
        ("species", spec.species_vocab_size),
        ("forme", spec.forme_vocab_size),
        ("move", spec.move_vocab_size),
        ("item", spec.item_vocab_size),
        ("ability", spec.ability_vocab_size),
        ("type", spec.type_vocab_size),
        ("nature", spec.nature_vocab_size),
        ("role", spec.role_vocab_size),
        ("gender", 4),
    ):
        params[f"{name}.embedding"] = _embedding(
            next(keys),
            size,
            spec.embedding_width,
        )

    def dense(name: str, fan_in: int, fan_out: int) -> None:
        params[f"{name}.weight"] = _glorot(next(keys), fan_in, fan_out)
        params[f"{name}.bias"] = jnp.zeros((fan_out,), dtype=jnp.float32)

    member_input = spec.embedding_width * 9 + 14
    dense("member", member_input, spec.member_hidden_width)
    dense("world", spec.member_hidden_width * 2, spec.world_hidden_width)
    dense(
        "belief",
        spec.world_hidden_width * 2 + 2,
        spec.hidden_width,
    )
    dense("public", spec.public_width, spec.hidden_width)
    dense("trunk", spec.hidden_width * 2, spec.hidden_width)
    dense("action", spec.action_width, spec.hidden_width)
    dense("policy_context", spec.hidden_width, spec.hidden_width)
    dense("value_hidden", spec.hidden_width, spec.hidden_width)
    dense("value", spec.hidden_width, 1)
    return params


def _dense(jnp: Any, x: Any, params: Mapping[str, Any], name: str) -> Any:
    return x @ params[f"{name}.weight"] + params[f"{name}.bias"]


def _masked_team_moments(
    jnp: Any,
    hidden: Any,
    mask: Any,
) -> tuple[Any, Any]:
    numeric = mask.astype(jnp.float32)
    count = jnp.maximum(jnp.sum(numeric, axis=-1, keepdims=True), 1.0)
    mean = jnp.sum(hidden * numeric[..., None], axis=-2) / count
    centered = hidden - mean[..., None, :]
    variance = (
        jnp.sum(centered * centered * numeric[..., None], axis=-2) / count
    )
    return mean, variance


def _packed_belief_trunk(
    jnp: Any,
    params: Mapping[str, Any],
    public: Any,
    weights: Any,
    species_num: Any,
    species_forme: Any,
    level: Any,
    ability_num: Any,
    item_num: Any,
    move_num: Any,
    move_mask: Any,
    tera_type: Any,
    nature: Any,
    role: Any,
    gender: Any,
    evs: Any,
    ivs: Any,
    was_lead: Any,
    team_mask: Any,
) -> Any:
    """Encode one complete joint posterior without hashing hidden categories."""

    species_emb = params["species.embedding"][species_num]
    forme_emb = params["forme.embedding"][species_forme]
    ability_emb = params["ability.embedding"][ability_num]
    item_emb = params["item.embedding"][item_num]
    tera_emb = params["type.embedding"][tera_type]
    nature_emb = params["nature.embedding"][nature]
    role_emb = params["role.embedding"][role]
    gender_emb = params["gender.embedding"][gender]

    raw_move_emb = params["move.embedding"][move_num]
    numeric_move_mask = move_mask.astype(jnp.float32)
    move_count = jnp.maximum(
        jnp.sum(numeric_move_mask, axis=-1, keepdims=True),
        1.0,
    )
    move_emb = (
        jnp.sum(raw_move_emb * numeric_move_mask[..., None], axis=-2)
        / move_count
    )

    scalars = jnp.concatenate(
        (
            level[..., None].astype(jnp.float32) / 100.0,
            evs.astype(jnp.float32) / 252.0,
            ivs.astype(jnp.float32) / 31.0,
            was_lead[..., None].astype(jnp.float32),
        ),
        axis=-1,
    )
    member_input = jnp.concatenate(
        (
            species_emb,
            forme_emb,
            ability_emb,
            item_emb,
            move_emb,
            tera_emb,
            nature_emb,
            role_emb,
            gender_emb,
            scalars,
        ),
        axis=-1,
    )
    member_hidden = jnp.tanh(_dense(jnp, member_input, params, "member"))
    member_mean, member_variance = _masked_team_moments(
        jnp,
        member_hidden,
        team_mask,
    )
    world_hidden = jnp.tanh(
        _dense(
            jnp,
            jnp.concatenate((member_mean, member_variance), axis=-1),
            params,
            "world",
        )
    )

    posterior_mean = jnp.sum(world_hidden * weights[:, None], axis=0)
    centered = world_hidden - posterior_mean[None, :]
    posterior_variance = jnp.sum(
        centered * centered * weights[:, None],
        axis=0,
    )
    entropy = -jnp.sum(weights * jnp.log(jnp.maximum(weights, 1e-12)))
    effective_support = jnp.exp(entropy)
    belief_hidden = jnp.tanh(
        _dense(
            jnp,
            jnp.concatenate(
                (
                    posterior_mean,
                    posterior_variance,
                    jnp.stack((entropy, effective_support)).astype(jnp.float32),
                )
            ),
            params,
            "belief",
        )
    )

    public_hidden = jnp.tanh(_dense(jnp, public, params, "public"))
    return jnp.tanh(
        _dense(
            jnp,
            jnp.concatenate((public_hidden, belief_hidden)),
            params,
            "trunk",
        )
    )


def _value_from_trunk(jnp: Any, params: Mapping[str, Any], trunk: Any) -> Any:
    hidden = jnp.tanh(_dense(jnp, trunk, params, "value_hidden"))
    return jnp.tanh(_dense(jnp, hidden, params, "value"))[0]


def forward_packed(
    params: Mapping[str, Any],
    inputs: PackedBeliefEvaluatorInput,
) -> tuple[Any, Any]:
    """Return value and policy logits for one packed public-belief state."""

    _, jnp = _require_jax()
    trunk = _packed_belief_trunk(
        jnp,
        params,
        jnp.asarray(inputs.public_features, dtype=jnp.float32),
        jnp.asarray(inputs.world_weights, dtype=jnp.float32),
        jnp.asarray(inputs.species_num, dtype=jnp.int32),
        jnp.asarray(inputs.species_forme, dtype=jnp.int32),
        jnp.asarray(inputs.level, dtype=jnp.int32),
        jnp.asarray(inputs.ability_num, dtype=jnp.int32),
        jnp.asarray(inputs.item_num, dtype=jnp.int32),
        jnp.asarray(inputs.move_num, dtype=jnp.int32),
        jnp.asarray(inputs.move_mask, dtype=jnp.bool_),
        jnp.asarray(inputs.tera_type, dtype=jnp.int32),
        jnp.asarray(inputs.nature, dtype=jnp.int32),
        jnp.asarray(inputs.role, dtype=jnp.int32),
        jnp.asarray(inputs.gender, dtype=jnp.int32),
        jnp.asarray(inputs.evs, dtype=jnp.int32),
        jnp.asarray(inputs.ivs, dtype=jnp.int32),
        jnp.asarray(inputs.was_lead, dtype=jnp.bool_),
        jnp.ones(
            (inputs.world_count, inputs.team_size),
            dtype=jnp.bool_,
        ),
    )
    actions = jnp.asarray(inputs.action_features, dtype=jnp.float32)
    action_hidden = jnp.tanh(_dense(jnp, actions, params, "action"))
    policy_context = jnp.tanh(_dense(jnp, trunk, params, "policy_context"))
    logits = action_hidden @ policy_context / math.sqrt(
        float(policy_context.shape[-1])
    )
    return _value_from_trunk(jnp, params, trunk), logits


def packed_loss(
    params: Mapping[str, Any],
    inputs: PackedBeliefEvaluatorInput,
    *,
    value_target: float,
    policy_target: Sequence[float],
    policy_weight: float = 1.0,
) -> Any:
    """Use the same value-MSE + policy-cross-entropy objective as the control."""

    _, jnp = _require_jax()
    if len(policy_target) != len(inputs.legal_actions):
        raise BeliefEvaluatorError("policy target must cover every legal action")
    target = jnp.asarray(policy_target, dtype=jnp.float32)
    target = target / jnp.maximum(jnp.sum(target), 1e-12)
    value, logits = forward_packed(params, inputs)
    maximum = jnp.max(logits)
    log_probs = logits - maximum - jnp.log(
        jnp.sum(jnp.exp(logits - maximum))
    )
    value_error = (
        value - jnp.asarray(value_target, dtype=jnp.float32)
    ) ** 2
    policy_error = -jnp.sum(target * log_probs)
    return value_error + float(policy_weight) * policy_error


def predict_packed(
    params: Mapping[str, Any],
    inputs: PackedBeliefEvaluatorInput,
) -> BeliefPrediction:
    """Expose the same prediction diagnostics as the hashed control evaluator."""

    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError(
            "NumPy is required for packed evaluator inference"
        ) from error

    raw_value, raw_logits = forward_packed(params, inputs)
    logits = np.asarray(raw_logits, dtype=np.float64)
    if logits.ndim != 1 or logits.shape[0] != len(inputs.legal_actions):
        raise BeliefEvaluatorError("packed policy head returned the wrong action shape")
    if not np.all(np.isfinite(logits)):
        raise BeliefEvaluatorError("packed policy head returned non-finite logits")

    shifted = logits - float(np.max(logits))
    probabilities_array = np.exp(shifted)
    probabilities_array /= float(np.sum(probabilities_array))
    probabilities = tuple(float(value) for value in probabilities_array)
    maximum = max(probabilities)
    selected_action = min(
        action
        for action, probability in zip(
            inputs.legal_actions,
            probabilities,
            strict=True,
        )
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
        raise BeliefEvaluatorError("packed value head returned a non-finite value")
    return BeliefPrediction(
        value=value,
        legal_actions=inputs.legal_actions,
        probabilities=probabilities,
        selected_action=selected_action,
        policy_margin=float(margin),
        policy_entropy_bits=float(entropy),
    )


_PACKED_BATCH_VALUE_FUNCTION: Any | None = None


def _batched_value_function() -> Any:
    global _PACKED_BATCH_VALUE_FUNCTION
    if _PACKED_BATCH_VALUE_FUNCTION is None:
        jax, jnp = _require_jax()

        def evaluate(
            params: Mapping[str, Any],
            public: Any,
            weights: Any,
            species_num: Any,
            species_forme: Any,
            level: Any,
            ability_num: Any,
            item_num: Any,
            move_num: Any,
            move_mask: Any,
            tera_type: Any,
            nature: Any,
            role: Any,
            gender: Any,
            evs: Any,
            ivs: Any,
            was_lead: Any,
            team_mask: Any,
        ) -> Any:
            def one(
                public_row: Any,
                weight_row: Any,
                species_row: Any,
                forme_row: Any,
                level_row: Any,
                ability_row: Any,
                item_row: Any,
                move_row: Any,
                move_mask_row: Any,
                tera_row: Any,
                nature_row: Any,
                role_row: Any,
                gender_row: Any,
                evs_row: Any,
                ivs_row: Any,
                lead_row: Any,
                team_mask_row: Any,
            ) -> Any:
                trunk = _packed_belief_trunk(
                    jnp,
                    params,
                    public_row,
                    weight_row,
                    species_row,
                    forme_row,
                    level_row,
                    ability_row,
                    item_row,
                    move_row,
                    move_mask_row,
                    tera_row,
                    nature_row,
                    role_row,
                    gender_row,
                    evs_row,
                    ivs_row,
                    lead_row,
                    team_mask_row,
                )
                return _value_from_trunk(jnp, params, trunk)

            return jax.vmap(one)(
                public,
                weights,
                species_num,
                species_forme,
                level,
                ability_num,
                item_num,
                move_num,
                move_mask,
                tera_type,
                nature,
                role,
                gender,
                evs,
                ivs,
                was_lead,
                team_mask,
            )

        _PACKED_BATCH_VALUE_FUNCTION = jax.jit(evaluate)
    return _PACKED_BATCH_VALUE_FUNCTION


def predict_packed_values(
    params: Mapping[str, Any],
    inputs: Sequence[PackedBeliefEvaluatorInput],
) -> tuple[float, ...]:
    """Evaluate a shape-bucketed packed frontier through one JAX dispatch."""

    if not inputs:
        return ()
    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError(
            "NumPy is required for packed batched inference"
        ) from error

    public_width = len(inputs[0].public_features)
    batch_size = _shape_bucket(len(inputs))
    world_count = _shape_bucket(max(row.world_count for row in inputs))
    team_size = _shape_bucket(max(row.team_size for row in inputs))

    public = np.zeros((batch_size, public_width), dtype=np.float32)
    weights = np.zeros((batch_size, world_count), dtype=np.float32)
    categorical = {
        "species_num": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "species_forme": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "level": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "ability_num": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "item_num": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "tera_type": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "nature": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "role": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
        "gender": np.zeros((batch_size, world_count, team_size), dtype=np.int32),
    }
    move_num = np.zeros(
        (batch_size, world_count, team_size, 4),
        dtype=np.int32,
    )
    move_mask = np.zeros(
        (batch_size, world_count, team_size, 4),
        dtype=np.bool_,
    )
    evs = np.zeros(
        (batch_size, world_count, team_size, 6),
        dtype=np.int32,
    )
    ivs = np.zeros(
        (batch_size, world_count, team_size, 6),
        dtype=np.int32,
    )
    was_lead = np.zeros(
        (batch_size, world_count, team_size),
        dtype=np.bool_,
    )
    team_mask = np.zeros(
        (batch_size, world_count, team_size),
        dtype=np.bool_,
    )

    for index, row in enumerate(inputs):
        if len(row.public_features) != public_width:
            raise BeliefEvaluatorError(
                "packed batched evaluator public feature widths differ"
            )
        wc = row.world_count
        tc = row.team_size
        public[index] = np.asarray(row.public_features, dtype=np.float32)
        weights[index, :wc] = np.asarray(row.world_weights, dtype=np.float32)
        for name in categorical:
            categorical[name][index, :wc, :tc] = np.asarray(
                getattr(row, name),
                dtype=np.int32,
            )
        move_num[index, :wc, :tc] = np.asarray(row.move_num, dtype=np.int32)
        move_mask[index, :wc, :tc] = np.asarray(row.move_mask, dtype=np.bool_)
        evs[index, :wc, :tc] = np.asarray(row.evs, dtype=np.int32)
        ivs[index, :wc, :tc] = np.asarray(row.ivs, dtype=np.int32)
        was_lead[index, :wc, :tc] = np.asarray(row.was_lead, dtype=np.bool_)
        team_mask[index, :wc, :tc] = True

    raw = _batched_value_function()(
        params,
        public,
        weights,
        categorical["species_num"],
        categorical["species_forme"],
        categorical["level"],
        categorical["ability_num"],
        categorical["item_num"],
        move_num,
        move_mask,
        categorical["tera_type"],
        categorical["nature"],
        categorical["role"],
        categorical["gender"],
        evs,
        ivs,
        was_lead,
        team_mask,
    )
    values = np.asarray(raw, dtype=np.float64)[: len(inputs)]
    if values.ndim != 1 or values.shape[0] != len(inputs):
        raise BeliefEvaluatorError("packed batched value head returned the wrong shape")
    if not np.all(np.isfinite(values)):
        raise BeliefEvaluatorError("packed batched value head returned non-finite values")
    return tuple(float(value) for value in values)



_SHARED_WORLD_FRONTIER_VALUE_FUNCTION: Any | None = None


def _shared_world_frontier_value_function() -> Any:
    """JIT one frontier whose leaves reweight the same packed hidden worlds."""

    global _SHARED_WORLD_FRONTIER_VALUE_FUNCTION
    if _SHARED_WORLD_FRONTIER_VALUE_FUNCTION is None:
        jax, jnp = _require_jax()

        def evaluate(
            params: Mapping[str, Any],
            public: Any,
            weights: Any,
            species_num: Any,
            species_forme: Any,
            level: Any,
            ability_num: Any,
            item_num: Any,
            move_num: Any,
            move_mask: Any,
            tera_type: Any,
            nature: Any,
            role: Any,
            gender: Any,
            evs: Any,
            ivs: Any,
            was_lead: Any,
            team_mask: Any,
        ) -> Any:
            def one(public_row: Any, weight_row: Any) -> Any:
                trunk = _packed_belief_trunk(
                    jnp,
                    params,
                    public_row,
                    weight_row,
                    species_num,
                    species_forme,
                    level,
                    ability_num,
                    item_num,
                    move_num,
                    move_mask,
                    tera_type,
                    nature,
                    role,
                    gender,
                    evs,
                    ivs,
                    was_lead,
                    team_mask,
                )
                return _value_from_trunk(jnp, params, trunk)

            return jax.vmap(one)(public, weights)

        _SHARED_WORLD_FRONTIER_VALUE_FUNCTION = jax.jit(evaluate)
    return _SHARED_WORLD_FRONTIER_VALUE_FUNCTION


def predict_packed_shared_world_values(
    params: Mapping[str, Any],
    packed: PackedJointPosterior,
    *,
    public_features: Sequence[Sequence[float]],
    leaf_world_weights: Any,
    expected_vocabulary_sha256: str | None = None,
) -> tuple[float, ...]:
    """Evaluate many successor beliefs without duplicating hidden-world tensors.

    All leaves share one immutable packed world tensor. Only public-state features and
    posterior weights vary across the frontier. This is the numerical shape produced by
    compiled information-set transport.
    """

    try:
        import numpy as np
    except ImportError as error:
        raise BeliefEvaluatorError(
            "NumPy is required for shared-world packed frontier inference"
        ) from error

    if (
        expected_vocabulary_sha256 is not None
        and packed.vocabulary_sha256 != expected_vocabulary_sha256
    ):
        raise BeliefEvaluatorError(
            "packed frontier and evaluator vocabulary identities differ"
        )

    public = np.asarray(public_features, dtype=np.float32)
    weights = np.asarray(leaf_world_weights, dtype=np.float32)
    if public.ndim != 2 or public.shape[0] == 0:
        raise BeliefEvaluatorError(
            "packed frontier public features must be a non-empty matrix"
        )
    if weights.ndim != 2 or weights.shape != (public.shape[0], packed.world_count):
        raise BeliefEvaluatorError(
            "packed frontier posterior-weight matrix has the wrong shape"
        )
    if not np.all(np.isfinite(public)):
        raise BeliefEvaluatorError("packed frontier public features must be finite")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise BeliefEvaluatorError(
            "packed frontier posterior weights must be finite and non-negative"
        )
    row_mass = weights.sum(axis=1, dtype=np.float64)
    if not np.allclose(row_mass, np.ones(len(row_mass)), atol=1e-6, rtol=0):
        raise BeliefEvaluatorError(
            "packed frontier posterior weights must normalize per leaf"
        )

    arrays = packed.as_numpy()
    team_mask = np.ones(
        (packed.world_count, packed.team_size),
        dtype=np.bool_,
    )
    raw = _shared_world_frontier_value_function()(
        params,
        public,
        weights,
        arrays["species_num"].astype(np.int32, copy=False),
        arrays["species_forme"].astype(np.int32, copy=False),
        arrays["level"].astype(np.int32, copy=False),
        arrays["ability_num"].astype(np.int32, copy=False),
        arrays["item_num"].astype(np.int32, copy=False),
        arrays["move_num"].astype(np.int32, copy=False),
        arrays["move_mask"].astype(np.bool_, copy=False),
        arrays["tera_type"].astype(np.int32, copy=False),
        arrays["nature"].astype(np.int32, copy=False),
        arrays["role"].astype(np.int32, copy=False),
        arrays["gender"].astype(np.int32, copy=False),
        arrays["evs"].astype(np.int32, copy=False),
        arrays["ivs"].astype(np.int32, copy=False),
        arrays["was_lead"].astype(np.bool_, copy=False),
        team_mask,
    )
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != (public.shape[0],) or not np.all(np.isfinite(values)):
        raise BeliefEvaluatorError(
            "shared-world packed frontier evaluator returned invalid values"
        )
    return tuple(float(value) for value in values)
