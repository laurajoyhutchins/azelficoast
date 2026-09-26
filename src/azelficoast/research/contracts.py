"""Immutable contracts separating public evidence from private belief support.

JSON is accepted only at artifact edges. The values in this module are validated,
canonical, and free of mutable dictionaries after construction.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from azelficoast.core.program import PROGRAM_SET_SCHEMA, PROGRAM_SET_SCHEMA_VERSION
from azelficoast.core.transition import (
    ORACLE_SCHEMA as ORACLE_SCHEMA,
    ORACLE_SCHEMA_VERSION as ORACLE_SCHEMA_VERSION,
)
COMPUTE_RECEIPT_SCHEMA = "azelficoast.matched-search-receipt"
COMPUTE_RECEIPT_SCHEMA_VERSION = 4
COMPUTE_BUDGET_UNIT_DEFINITION = (
    "one transition_evaluation per verified whole-turn execution class consumed"
)
EVALUATOR_CALL_UNIT_DEFINITION = (
    "one learned value prediction per successor public information set"
)


class ResearchContractError(ValueError):
    """Raised when data crosses a research boundary without a valid contract."""


def _json_value(value: object, *, path: str = "$") -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResearchContractError(f"{path}: JSON numbers must be finite")
        return value
    if isinstance(value, FrozenJSONObject):
        return value.to_record()
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ResearchContractError(f"{path}: JSON object keys must be strings")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item, path=f"{path}[{index}]") for index, item in enumerate(value)]
    raise ResearchContractError(f"{path}: value is not JSON-compatible")


def canonical_json(value: object) -> str:
    """Serialize JSON data with the same canonical rules used for evidence hashes."""
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def stable_digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _semantic_mass(value: float) -> float:
    """Drop binary normalization noise from identity, retaining full model precision."""
    return float(format(value, ".15g"))


def _belief_semantic_digest(
    treatment: str,
    worlds: Sequence[SemanticWorld],
) -> str:
    return stable_digest(
        {
            "treatment": treatment,
            "support": [
                {
                    "semantic_identity": world.semantic_identity,
                    "weight": _semantic_mass(world.weight),
                }
                for world in worlds
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenJSONObject:
    """A JSON object retained canonically so nested mutable values cannot leak in."""

    canonical: str

    def __post_init__(self) -> None:
        try:
            parsed = json.loads(self.canonical)
        except json.JSONDecodeError as error:
            raise ResearchContractError("frozen JSON object is malformed") from error
        if not isinstance(parsed, dict) or canonical_json(parsed) != self.canonical:
            raise ResearchContractError("frozen JSON object must be a canonical object")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> FrozenJSONObject:
        encoded = canonical_json(value)
        parsed = json.loads(encoded)
        if not isinstance(parsed, dict):
            raise ResearchContractError("expected a JSON object")
        return cls(encoded)

    def to_record(self) -> dict[str, object]:
        value = json.loads(self.canonical)
        if not isinstance(value, dict):
            raise ResearchContractError("frozen JSON object has invalid canonical data")
        return value


@dataclass(frozen=True, slots=True)
class MechanicsIdentity:
    """Pinned Showdown revision and admitted transition-program semantics."""

    revision: str
    semantics_schema: str = PROGRAM_SET_SCHEMA
    semantics_schema_version: int = PROGRAM_SET_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.revision:
            raise ResearchContractError("mechanics revision must be non-empty")
        if self.semantics_schema != PROGRAM_SET_SCHEMA:
            raise ResearchContractError("unsupported mechanics semantics schema")
        if self.semantics_schema_version != PROGRAM_SET_SCHEMA_VERSION:
            raise ResearchContractError("unsupported mechanics semantics version")

    @classmethod
    def from_showdown_commit(cls, revision: str) -> MechanicsIdentity:
        return cls(revision=revision)

    def to_record(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "semantics_schema": self.semantics_schema,
            "semantics_schema_version": self.semantics_schema_version,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MechanicsIdentity:
        revision = record.get("revision")
        schema = record.get("semantics_schema")
        version = record.get("semantics_schema_version")
        if not isinstance(revision, str):
            raise ResearchContractError("mechanics identity lacks a revision")
        if not isinstance(schema, str):
            raise ResearchContractError("mechanics identity lacks a semantics schema")
        if not isinstance(version, int) or isinstance(version, bool):
            raise ResearchContractError("mechanics identity lacks a semantics version")
        return cls(revision, schema, version)

    @property
    def identity_digest(self) -> str:
        return stable_digest(self.to_record())


_FORBIDDEN_PUBLIC_FIELDS = {
    "counterfactualhiddenstate",
    "futureinformation",
    "futureobservation",
    "futurestate",
    "hiddenworld",
    "opponentprivate",
    "postdecisionstate",
    "privatedata",
    "privatefields",
    "realizedhiddenstate",
    "realizedhiddenworld",
    "transportid",
    "worldid",
}


def _reject_realized_world_payload(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ResearchContractError(f"{path} contains a non-string field name")
            normalized = "".join(character.lower() for character in key if character.isalnum())
            if normalized in {"realizedhiddenstaterevealed", "realizedhiddenstateused"}:
                if child is False:
                    continue
                raise ResearchContractError(
                    f"{path} may not reveal the realized hidden state or carry realized hidden-world information; forbidden private or future field {key!r}"
                )
            marker = ("realized" in normalized or "sampled" in normalized or "actual" in normalized) and (
                "hidden" in normalized or "world" in normalized
            )
            if marker:
                raise ResearchContractError(
                    f"{path} may not reveal the realized hidden state or carry realized hidden-world information; forbidden private or future field {key!r}"
                )
            _reject_realized_world_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_realized_world_payload(child, path=f"{path}[{index}]")


def _public_payload(value: object, *, path: str = "public decision input") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ResearchContractError(f"{path} contains a non-string field name")
            normalized = "".join(character.lower() for character in key if character.isalnum())
            if (
                normalized in _FORBIDDEN_PUBLIC_FIELDS
                or normalized.startswith("future")
                or "private" in normalized
                or "afterdecision" in normalized
                or "transportid" in normalized
                or normalized.endswith("worldid")
            ):
                raise ResearchContractError(
                    f"{path} contains forbidden private or future field {key!r}"
                )
            _public_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _public_payload(child, path=f"{path}[{index}]")


def _reject_future_payload(value: object, *, path: str) -> None:
    """Reject time-travel fields even inside otherwise valid hidden-world records."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ResearchContractError(f"{path} contains a non-string field name")
            normalized = "".join(character.lower() for character in key if character.isalnum())
            if (
                normalized.startswith("future")
                or "afterdecision" in normalized
                or "postdecision" in normalized
            ):
                raise ResearchContractError(f"{path} contains future-only field {key!r}")
            _reject_future_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_future_payload(child, path=f"{path}[{index}]")


def _reject_transport_payload(
    value: object,
    *,
    path: str,
    root_world_record: bool = False,
) -> None:
    """Keep opaque world identifiers at the root adapter field only."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ResearchContractError(f"{path} contains a non-string field name")
            normalized = "".join(character.lower() for character in key if character.isalnum())
            is_transport_field = "transportid" in normalized or normalized.endswith("worldid")
            if is_transport_field and not (
                root_world_record and key in {"world_id", "id", "transport_id"}
            ):
                raise ResearchContractError(
                    f"{path} contains transport-only identifier field {key!r}"
                )
            _reject_transport_payload(child, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _reject_transport_payload(child, path=f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class PublicDecisionInput:
    """Only public state, public history identity, legal actions, and mechanics."""

    fixture_id: str
    battle_tag: str
    public_state: FrozenJSONObject
    legal_actions: tuple[str, ...]
    public_history_identity: str
    mechanics_identity: MechanicsIdentity

    def __post_init__(self) -> None:
        if not self.fixture_id or not self.battle_tag or not self.public_history_identity:
            raise ResearchContractError("public decision identity fields must be non-empty")
        if not isinstance(self.public_state, FrozenJSONObject):
            raise ResearchContractError("public decision state must be frozen")
        _public_payload(self.public_state.to_record())
        if (
            not isinstance(self.legal_actions, tuple)
            or not self.legal_actions
            or not all(isinstance(action, str) and action for action in self.legal_actions)
            or len(set(self.legal_actions)) != len(self.legal_actions)
        ):
            raise ResearchContractError("public decision actions must be non-empty and unique")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        mechanics_identity: MechanicsIdentity,
    ) -> PublicDecisionInput:
        fixture_id = record.get("fixture_id")
        battle_tag = record.get("battle_tag")
        public_state = record.get("public_state")
        raw_actions = record.get("legal_actions")
        if not isinstance(fixture_id, str) or not fixture_id:
            raise ResearchContractError("public decision input lacks fixture identity")
        if not isinstance(battle_tag, str) or not battle_tag:
            raise ResearchContractError("public decision input lacks battle identity")
        if not isinstance(public_state, Mapping):
            raise ResearchContractError("public decision input requires public state")
        _reject_realized_world_payload(record, path="public decision input")
        _public_payload(record)
        if (
            not isinstance(raw_actions, list)
            or not raw_actions
            or not all(isinstance(action, str) and action for action in raw_actions)
        ):
            raise ResearchContractError(
                "public decision input requires non-empty legal public actions"
            )
        if len(set(raw_actions)) != len(raw_actions):
            raise ResearchContractError("public decision actions must be unique")

        frozen_state = FrozenJSONObject.from_mapping(public_state)
        state_identity = stable_digest(
            {
                "fixture_id": fixture_id,
                "battle_tag": battle_tag,
                "public_state": frozen_state.to_record(),
                "legal_actions": raw_actions,
            }
        )
        raw_history_identity = record.get("public_history_identity")
        if raw_history_identity is not None and (
            not isinstance(raw_history_identity, str) or not raw_history_identity
        ):
            raise ResearchContractError("public history identity must be a non-empty string")
        return cls(
            fixture_id=fixture_id,
            battle_tag=battle_tag,
            public_state=frozen_state,
            legal_actions=tuple(raw_actions),
            public_history_identity=(
                raw_history_identity if isinstance(raw_history_identity, str) else state_identity
            ),
            mechanics_identity=mechanics_identity,
        )

    @property
    def state_digest(self) -> str:
        return stable_digest(
            {
                "fixture_id": self.fixture_id,
                "battle_tag": self.battle_tag,
                "public_state": self.public_state.to_record(),
                "legal_actions": list(self.legal_actions),
            }
        )


@dataclass(frozen=True, slots=True)
class PublicSuccessorState:
    """A public state after a mechanics transition and a public observation."""

    public_state: FrozenJSONObject

    def __post_init__(self) -> None:
        if not isinstance(self.public_state, FrozenJSONObject):
            raise ResearchContractError("public successor state must be frozen")
        _public_payload(self.public_state.to_record(), path="public successor state")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PublicSuccessorState:
        _reject_realized_world_payload(record, path="public successor state")
        _public_payload(record, path="public successor state")
        return cls(public_state=FrozenJSONObject.from_mapping(record))


@dataclass(frozen=True, slots=True)
class SemanticWorld:
    """One semantic support atom; opaque transport identifiers are not represented."""

    semantic_identity: str
    features: FrozenJSONObject
    hidden_state: FrozenJSONObject | None
    weight: float

    def __post_init__(self) -> None:
        if not self.semantic_identity:
            raise ResearchContractError("semantic world identity must be non-empty")
        if not isinstance(self.features, FrozenJSONObject):
            raise ResearchContractError("semantic world features must be frozen")
        if self.hidden_state is not None and not isinstance(
            self.hidden_state, FrozenJSONObject
        ):
            raise ResearchContractError("semantic hidden state must be frozen")
        if (
            isinstance(self.weight, bool)
            or not isinstance(self.weight, (int, float))
            or not math.isfinite(float(self.weight))
            or self.weight <= 0
        ):
            raise ResearchContractError("semantic world weight must be positive and finite")
        if stable_digest(self.features.to_record()) != self.semantic_identity:
            raise ResearchContractError("semantic world identity does not match its features")

    def to_record(self) -> dict[str, object]:
        return {**self.features.to_record(), "weight": self.weight}


@dataclass(frozen=True, slots=True)
class BeliefTransportIndex:
    """Private adapter-side association from opaque artifact IDs to semantic worlds."""

    bindings: tuple[tuple[str, str], ...]
    weights: tuple[tuple[str, float], ...] = ()

    def semantic_identity_for(self, transport_id: str) -> str | None:
        for identifier, semantic_identity in self.bindings:
            if identifier == transport_id:
                return semantic_identity
        return None

    def transport_ids_for(self, semantic_identity: str) -> tuple[str, ...]:
        return tuple(
            identifier for identifier, identity in self.bindings if identity == semantic_identity
        )

    def weight_for(self, transport_id: str) -> float | None:
        return next(
            (weight for identifier, weight in self.weights if identifier == transport_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class BeliefInput:
    """A normalized posterior over canonical semantic worlds, without transport IDs."""

    treatment: str
    model_worlds: tuple[SemanticWorld, ...]
    semantic_digest: str

    def __post_init__(self) -> None:
        if not self.treatment:
            raise ResearchContractError("belief treatment identity must be non-empty")
        if not isinstance(self.model_worlds, tuple) or not self.model_worlds:
            raise ResearchContractError("belief must contain semantic support")
        identities = [world.semantic_identity for world in self.model_worlds]
        if len(set(identities)) != len(identities):
            raise ResearchContractError("belief semantic support must be coalesced")
        worlds = tuple(sorted(self.model_worlds, key=lambda world: world.features.canonical))
        if worlds != self.model_worlds:
            object.__setattr__(self, "model_worlds", worlds)
        total_weight = math.fsum(world.weight for world in worlds)
        if not math.isfinite(total_weight) or abs(total_weight - 1.0) > 1e-12:
            raise ResearchContractError("belief weights must be normalized")
        expected_digest = _belief_semantic_digest(self.treatment, worlds)
        if self.semantic_digest != expected_digest:
            raise ResearchContractError("belief semantic digest does not match its support")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BeliefInput:
        return parse_belief_artifact(record)[0]

    def to_evaluator_record(self) -> dict[str, object]:
        return {
            "treatment": self.treatment,
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "worlds": [world.to_record() for world in self.model_worlds],
        }

    def reweighted(self, mass_by_identity: Mapping[str, float]) -> BeliefInput:
        """Return a normalized conditional posterior over a semantic support subset."""
        known = {world.semantic_identity: world for world in self.model_worlds}
        if not mass_by_identity or set(mass_by_identity) - set(known):
            raise ResearchContractError("conditional belief references unsupported semantics")
        masses: dict[str, float] = {}
        for identity, mass in mass_by_identity.items():
            try:
                number = float(mass)
            except (OverflowError, TypeError, ValueError):
                number = math.nan
            if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(number) or number <= 0:
                raise ResearchContractError("conditional belief masses must be positive and finite")
            masses[identity] = number
        scale = max(masses.values())
        scaled_masses = {identity: mass / scale for identity, mass in masses.items()}
        total = math.fsum(scaled_masses.values())
        if not math.isfinite(total) or total <= 0:
            raise ResearchContractError("conditional belief has no finite positive mass")
        worlds = tuple(
            SemanticWorld(
                semantic_identity=identity,
                features=known[identity].features,
                hidden_state=known[identity].hidden_state,
                weight=scaled_masses[identity] / total,
            )
            for identity, mass in sorted(
                scaled_masses.items(), key=lambda row: known[row[0]].features.canonical
            )
        )
        digest = _belief_semantic_digest(self.treatment, worlds)
        return BeliefInput(
            treatment=self.treatment,
            model_worlds=worlds,
            semantic_digest=digest,
        )


def parse_belief_artifact(
    record: Mapping[str, object],
) -> tuple[BeliefInput, BeliefTransportIndex]:
    _reject_realized_world_payload(record, path="belief")
    _public_payload(
        {key: value for key, value in record.items() if key != "worlds"},
        path="belief metadata",
    )
    _reject_future_payload(record, path="belief")
    if record.get("conditioned_on_public_history") is not True:
        raise ResearchContractError("belief must be conditioned on public history")
    if record.get("realized_hidden_state_revealed") is not False:
        raise ResearchContractError("belief may not reveal the realized hidden state")
    treatment = record.get("treatment", "unspecified")
    if not isinstance(treatment, str) or not treatment:
        raise ResearchContractError("belief treatment identity must be non-empty")
    raw_worlds = record.get("worlds")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise ResearchContractError("belief must contain hidden-world support")

    weights: list[float] = []
    feature_by_canonical: dict[str, FrozenJSONObject] = {}
    mass_by_canonical: dict[str, list[float]] = {}
    hidden_by_canonical: dict[str, FrozenJSONObject | None] = {}
    bindings: list[tuple[str, str]] = []
    transport_weights: list[tuple[str, float]] = []
    seen_transport_ids: set[str] = set()

    for index, raw_world in enumerate(raw_worlds):
        if not isinstance(raw_world, Mapping):
            raise ResearchContractError("belief world must be an object")
        _reject_future_payload(raw_world, path=f"belief world {index}")
        _reject_transport_payload(
            raw_world,
            path=f"belief world {index}",
            root_world_record=True,
        )
        weight = raw_world.get("weight")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ResearchContractError("belief world weight must be positive and finite")
        try:
            numeric_weight = float(weight)
        except (OverflowError, TypeError, ValueError):
            numeric_weight = math.nan
        if (
            not math.isfinite(numeric_weight)
            or numeric_weight <= 0
        ):
            raise ResearchContractError("belief weights must be positive and finite")
        world_id = raw_world.get("world_id", raw_world.get("id"))
        if world_id is not None:
            if not isinstance(world_id, str) or not world_id:
                raise ResearchContractError("opaque belief transport ID must be non-empty")
            if world_id in seen_transport_ids:
                raise ResearchContractError(f"duplicate belief transport ID {world_id!r}")
            seen_transport_ids.add(world_id)

        raw_hidden = raw_world.get("hidden")
        if raw_hidden is not None and not isinstance(raw_hidden, Mapping):
            raise ResearchContractError("belief hidden semantic state must be an object")
        features = {
            str(key): value
            for key, value in raw_world.items()
            if key
            not in {
                "weight",
                "world_id",
                "id",
                "transport_id",
                "provenance",
                "sample_count",
            }
        }
        frozen_features = FrozenJSONObject.from_mapping(features)
        canonical = frozen_features.canonical
        feature_by_canonical[canonical] = frozen_features
        mass_by_canonical.setdefault(canonical, []).append(numeric_weight)
        weights.append(numeric_weight)
        if isinstance(raw_hidden, Mapping):
            frozen_hidden: FrozenJSONObject | None = FrozenJSONObject.from_mapping(raw_hidden)
        else:
            frozen_hidden = None
        previous_hidden = hidden_by_canonical.get(canonical)
        if previous_hidden is not None and frozen_hidden != previous_hidden:
            raise ResearchContractError("equivalent belief worlds disagree on hidden state")
        hidden_by_canonical[canonical] = frozen_hidden

        if isinstance(world_id, str):
            bindings.append((world_id, stable_digest(frozen_features.to_record())))
            transport_weights.append((world_id, numeric_weight))

    scale = max(weights)
    scaled_weights = [weight / scale for weight in weights]
    total_weight = math.fsum(sorted(scaled_weights))
    if not math.isfinite(total_weight) or total_weight <= 0:
        raise ResearchContractError("belief has no finite positive probability mass")

    grouped: list[tuple[str, SemanticWorld]] = []
    for canonical in sorted(feature_by_canonical):
        frozen_features = feature_by_canonical[canonical]
        semantic_identity = stable_digest(frozen_features.to_record())
        mass = math.fsum(
            sorted(weight / scale for weight in mass_by_canonical[canonical])
        ) / total_weight
        grouped.append(
            (
                semantic_identity,
                SemanticWorld(
                    semantic_identity=semantic_identity,
                    features=frozen_features,
                    hidden_state=hidden_by_canonical[canonical],
                    weight=mass,
                ),
            )
        )
    normalized_total = math.fsum(world.weight for _, world in grouped)
    semantic_worlds = tuple(
        SemanticWorld(
            semantic_identity=world.semantic_identity,
            features=world.features,
            hidden_state=world.hidden_state,
            weight=world.weight / normalized_total,
        )
        for _, world in grouped
    )
    belief = BeliefInput(
        treatment=treatment,
        model_worlds=semantic_worlds,
        semantic_digest=_belief_semantic_digest(treatment, semantic_worlds),
    )
    return belief, BeliefTransportIndex(
        bindings=tuple(sorted(bindings)),
        weights=tuple(
            sorted(
                (identifier, (weight / scale) / total_weight)
                for identifier, weight in transport_weights
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class ComputeBudget:
    unit: str
    authorized: int

    def __post_init__(self) -> None:
        if not self.unit:
            raise ResearchContractError("compute budget unit must be non-empty")
        if (
            not isinstance(self.authorized, int)
            or isinstance(self.authorized, bool)
            or self.authorized <= 0
        ):
            raise ResearchContractError("authorized compute budget must be positive")

    def to_record(self) -> dict[str, object]:
        return {"unit": self.unit, "authorized": self.authorized}

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ComputeBudget:
        unit = record.get("unit")
        authorized = record.get("authorized")
        if not isinstance(unit, str):
            raise ResearchContractError("compute budget lacks its unit")
        if not isinstance(authorized, int) or isinstance(authorized, bool):
            raise ResearchContractError("compute budget lacks an integer limit")
        return cls(unit, authorized)


@dataclass(frozen=True, slots=True)
class MatchedExperimentSpec:
    """The immutable inputs that must be shared by both treatment arms."""

    fixture_id: str
    battle_tag: str
    public_history_identity: str
    public_state: FrozenJSONObject
    public_state_digest: str
    legal_actions: tuple[str, ...]
    posterior_semantic_digest: str
    mechanics_identity: MechanicsIdentity
    evaluator_identity_digest: str
    chance_treatment: str
    compute_budget: ComputeBudget
    depth: int
    opponent_model: str

    def __post_init__(self) -> None:
        for name, value in (
            ("fixture identity", self.fixture_id),
            ("battle identity", self.battle_tag),
            ("public history identity", self.public_history_identity),
            ("public state digest", self.public_state_digest),
            ("posterior identity", self.posterior_semantic_digest),
            ("evaluator identity", self.evaluator_identity_digest),
            ("chance treatment", self.chance_treatment),
            ("opponent model", self.opponent_model),
        ):
            if not value:
                raise ResearchContractError(f"matched specification lacks {name}")
        if not self.legal_actions or len(set(self.legal_actions)) != len(self.legal_actions):
            raise ResearchContractError(
                "matched specification actions must be non-empty and unique"
            )
        if not isinstance(self.public_state, FrozenJSONObject):
            raise ResearchContractError("matched specification public state must be frozen")
        _public_payload(self.public_state.to_record(), path="matched specification public state")
        expected_public_digest = stable_digest(
            {
                "fixture_id": self.fixture_id,
                "battle_tag": self.battle_tag,
                "public_state": self.public_state.to_record(),
                "legal_actions": list(self.legal_actions),
            }
        )
        if self.public_state_digest != expected_public_digest:
            raise ResearchContractError("matched specification public state digest is invalid")
        if self.depth <= 0:
            raise ResearchContractError("matched specification depth must be positive")

    @classmethod
    def build(
        cls,
        *,
        public: PublicDecisionInput,
        belief: BeliefInput,
        evaluator_identity_digest: str,
        chance_treatment: str,
        compute_budget: ComputeBudget,
        depth: int,
        opponent_model: str,
    ) -> MatchedExperimentSpec:
        return cls(
            fixture_id=public.fixture_id,
            battle_tag=public.battle_tag,
            public_history_identity=public.public_history_identity,
            public_state=public.public_state,
            public_state_digest=public.state_digest,
            legal_actions=public.legal_actions,
            posterior_semantic_digest=belief.semantic_digest,
            mechanics_identity=public.mechanics_identity,
            evaluator_identity_digest=evaluator_identity_digest,
            chance_treatment=chance_treatment,
            compute_budget=compute_budget,
            depth=depth,
            opponent_model=opponent_model,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "fixture_id": self.fixture_id,
            "battle_tag": self.battle_tag,
            "public_history_identity": self.public_history_identity,
            "public_state": self.public_state.to_record(),
            "public_state_digest": self.public_state_digest,
            "legal_actions": list(self.legal_actions),
            "posterior_semantic_digest": self.posterior_semantic_digest,
            "mechanics_identity": self.mechanics_identity.to_record(),
            "mechanics_identity_digest": self.mechanics_identity.identity_digest,
            "evaluator_identity_digest": self.evaluator_identity_digest,
            "chance_treatment": self.chance_treatment,
            "compute_budget": self.compute_budget.to_record(),
            "depth": self.depth,
            "opponent_model": self.opponent_model,
        }

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MatchedExperimentSpec:
        fields = (
            "fixture_id",
            "battle_tag",
            "public_history_identity",
            "public_state_digest",
            "posterior_semantic_digest",
            "evaluator_identity_digest",
            "chance_treatment",
            "opponent_model",
        )
        values: dict[str, str] = {}
        for field_name in fields:
            value = record.get(field_name)
            if not isinstance(value, str):
                raise ResearchContractError(f"matched specification lacks {field_name}")
            values[field_name] = value
        raw_actions = record.get("legal_actions")
        if not isinstance(raw_actions, list) or not all(
            isinstance(action, str) and action for action in raw_actions
        ):
            raise ResearchContractError("matched specification actions are malformed")
        mechanics = record.get("mechanics_identity")
        public_state = record.get("public_state")
        budget = record.get("compute_budget")
        depth = record.get("depth")
        if not isinstance(mechanics, Mapping):
            raise ResearchContractError("matched specification lacks mechanics identity")
        if not isinstance(public_state, Mapping):
            raise ResearchContractError("matched specification lacks public state")
        if not isinstance(budget, Mapping):
            raise ResearchContractError("matched specification lacks compute budget")
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise ResearchContractError("matched specification depth is malformed")
        parsed_mechanics = MechanicsIdentity.from_record(mechanics)
        raw_mechanics_digest = record.get("mechanics_identity_digest")
        if raw_mechanics_digest != parsed_mechanics.identity_digest:
            raise ResearchContractError("matched specification mechanics digest is invalid")
        return cls(
            fixture_id=values["fixture_id"],
            battle_tag=values["battle_tag"],
            public_history_identity=values["public_history_identity"],
            public_state=FrozenJSONObject.from_mapping(public_state),
            public_state_digest=values["public_state_digest"],
            legal_actions=tuple(raw_actions),
            posterior_semantic_digest=values["posterior_semantic_digest"],
            mechanics_identity=parsed_mechanics,
            evaluator_identity_digest=values["evaluator_identity_digest"],
            chance_treatment=values["chance_treatment"],
            compute_budget=ComputeBudget.from_record(budget),
            depth=depth,
            opponent_model=values["opponent_model"],
        )

    @property
    def digest(self) -> str:
        return stable_digest(self.to_record())

    def require_compatible(self, other: MatchedExperimentSpec) -> None:
        comparisons: Sequence[tuple[str, object, object]] = (
            (
                "source state",
                (self.fixture_id, self.battle_tag),
                (other.fixture_id, other.battle_tag),
            ),
            ("public history", self.public_history_identity, other.public_history_identity),
            ("public state", self.public_state, other.public_state),
            ("public state", self.public_state_digest, other.public_state_digest),
            ("legal action set", set(self.legal_actions), set(other.legal_actions)),
            (
                "posterior semantics",
                self.posterior_semantic_digest,
                other.posterior_semantic_digest,
            ),
            ("mechanics revision", self.mechanics_identity, other.mechanics_identity),
            ("evaluator identity", self.evaluator_identity_digest, other.evaluator_identity_digest),
            ("chance treatment", self.chance_treatment, other.chance_treatment),
            ("compute budget", self.compute_budget, other.compute_budget),
            ("depth", self.depth, other.depth),
            ("opponent model", self.opponent_model, other.opponent_model),
        )
        for label, left, right in comparisons:
            if left != right:
                raise ResearchContractError(f"matched specification differs in {label}")


@dataclass(frozen=True, slots=True)
class ResourceAccounting:
    """Auditable class-budget, evaluator-call, and wall-clock measurements."""

    verified_execution_classes_consumed: int
    evaluator_calls: int
    evaluator_batches: int
    executor_preparation_wall_ms: float
    search_wall_ms: float
    executor_wall_ms: float
    transition_program_generation_included: bool
    transition_program_verification_included: bool
    posterior_construction_included: bool
    scope_note: str
    frontier_builds: int = 1
    frontier_memo_hit: bool = False
    frontier_group_identity: str = ""

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ResourceAccounting:
        count_fields = ("verified_execution_classes_consumed", "evaluator_calls")
        counts: dict[str, int] = {}
        for name in count_fields:
            value = record.get(name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ResearchContractError(f"resource accounting lacks valid {name}")
            counts[name] = value
        evaluator_batches = record.get("evaluator_batches", counts["evaluator_calls"])
        if (
            not isinstance(evaluator_batches, int)
            or isinstance(evaluator_batches, bool)
            or evaluator_batches < 0
            or evaluator_batches > counts["evaluator_calls"]
        ):
            raise ResearchContractError(
                "resource accounting lacks valid evaluator_batches"
            )
        timings: dict[str, float] = {}
        for name in (
            "executor_preparation_wall_ms",
            "search_wall_ms",
            "executor_wall_ms",
        ):
            value = record.get(name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ResearchContractError(f"resource accounting lacks valid {name}")
            timings[name] = float(value)
        flags: dict[str, bool] = {}
        for name in (
            "transition_program_generation_included",
            "transition_program_verification_included",
            "posterior_construction_included",
        ):
            value = record.get(name)
            if not isinstance(value, bool):
                raise ResearchContractError(f"resource accounting lacks boolean {name}")
            flags[name] = value
        scope_note = record.get("scope_note")
        if not isinstance(scope_note, str) or not scope_note:
            raise ResearchContractError("resource accounting lacks its scope note")
        frontier_builds = record.get("frontier_builds", 1)
        if (
            not isinstance(frontier_builds, int)
            or isinstance(frontier_builds, bool)
            or frontier_builds < 0
        ):
            raise ResearchContractError(
                "resource accounting lacks valid frontier_builds"
            )
        frontier_memo_hit = record.get("frontier_memo_hit", False)
        if not isinstance(frontier_memo_hit, bool):
            raise ResearchContractError(
                "resource accounting lacks boolean frontier_memo_hit"
            )
        frontier_group_identity = record.get("frontier_group_identity", "")
        if not isinstance(frontier_group_identity, str):
            raise ResearchContractError(
                "resource accounting lacks valid frontier_group_identity"
            )
        return cls(
            verified_execution_classes_consumed=counts[
                "verified_execution_classes_consumed"
            ],
            evaluator_calls=counts["evaluator_calls"],
            evaluator_batches=evaluator_batches,
            executor_preparation_wall_ms=timings["executor_preparation_wall_ms"],
            search_wall_ms=timings["search_wall_ms"],
            executor_wall_ms=timings["executor_wall_ms"],
            transition_program_generation_included=flags[
                "transition_program_generation_included"
            ],
            transition_program_verification_included=flags[
                "transition_program_verification_included"
            ],
            posterior_construction_included=flags["posterior_construction_included"],
            scope_note=scope_note,
            frontier_builds=frontier_builds,
            frontier_memo_hit=frontier_memo_hit,
            frontier_group_identity=frontier_group_identity,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "verified_execution_classes_consumed": self.verified_execution_classes_consumed,
            "evaluator_calls": self.evaluator_calls,
            "evaluator_batches": self.evaluator_batches,
            "executor_preparation_wall_ms": self.executor_preparation_wall_ms,
            "search_wall_ms": self.search_wall_ms,
            "executor_wall_ms": self.executor_wall_ms,
            "transition_program_generation_included": self.transition_program_generation_included,
            "transition_program_verification_included": self.transition_program_verification_included,
            "posterior_construction_included": self.posterior_construction_included,
            "scope_note": self.scope_note,
            "frontier_builds": self.frontier_builds,
            "frontier_memo_hit": self.frontier_memo_hit,
            "frontier_group_identity": self.frontier_group_identity,
        }


@dataclass(frozen=True, slots=True)
class ComputeReceipt:
    """Measured outcome from one treatment arm under a frozen matched spec."""

    method: str
    packet_digest: str
    matched_spec_digest: str
    input_digest: str
    posterior_digest: str
    posterior_semantic_digest: str
    evaluator_digest: str
    evaluator_checkpoint_digest: str
    mechanics_identity_digest: str
    mechanics_evidence_digest: str
    showdown_commit: str
    chance_treatment: str
    transition_oracle_digest: str
    transition_program_digest: str
    transition_artifact_digest: str
    transition_program_source: str
    compute_budget: ComputeBudget
    budget_unit_definition: str
    evaluator_call_unit_definition: str
    consumed: int
    evaluator_calls: int
    evaluator_batches: int
    chosen_action: str
    root_values: tuple[tuple[str, float], ...]
    resource_accounting: ResourceAccounting | None

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ComputeReceipt:
        if (
            record.get("schema") != COMPUTE_RECEIPT_SCHEMA
            or record.get("schema_version") != COMPUTE_RECEIPT_SCHEMA_VERSION
        ):
            raise ResearchContractError("unexpected matched-search receipt schema")
        string_fields = (
            "method",
            "packet_digest",
            "matched_spec_digest",
            "input_digest",
            "posterior_digest",
            "posterior_semantic_digest",
            "evaluator_digest",
            "evaluator_checkpoint_digest",
            "mechanics_identity_digest",
            "mechanics_evidence_digest",
            "showdown_commit",
            "chance_treatment",
            "transition_oracle_digest",
            "transition_program_digest",
            "transition_artifact_digest",
            "transition_program_source",
            "budget_unit_definition",
            "evaluator_call_unit_definition",
            "chosen_action",
        )
        values: dict[str, str] = {}
        for field_name in string_fields:
            value = record.get(field_name)
            if not isinstance(value, str) or not value:
                raise ResearchContractError(f"receipt lacks {field_name}")
            values[field_name] = value
        if values["transition_program_source"] not in {
            "verified-transition-program",
            "compiled-legacy-oracle",
        }:
            raise ResearchContractError("receipt declares an unsupported mechanics source")
        if values["transition_oracle_digest"] != values["transition_artifact_digest"]:
            raise ResearchContractError("receipt mechanics artifact digest is inconsistent")
        if values["budget_unit_definition"] != COMPUTE_BUDGET_UNIT_DEFINITION:
            raise ResearchContractError("receipt uses an unsupported compute-budget unit definition")
        if values["evaluator_call_unit_definition"] != EVALUATOR_CALL_UNIT_DEFINITION:
            raise ResearchContractError("receipt uses an unsupported evaluator-call unit definition")
        raw_budget = record.get("compute_budget")
        if not isinstance(raw_budget, Mapping):
            raise ResearchContractError("receipt lacks a compute budget")
        consumed = record.get("consumed")
        evaluator_calls = record.get("evaluator_calls")
        evaluator_batches = record.get("evaluator_batches", evaluator_calls)
        if not isinstance(consumed, int) or isinstance(consumed, bool) or consumed < 0:
            raise ResearchContractError("receipt consumed must be a non-negative integer")
        if (
            not isinstance(evaluator_calls, int)
            or isinstance(evaluator_calls, bool)
            or evaluator_calls < 0
        ):
            raise ResearchContractError(
                "receipt evaluator call count must be a non-negative integer"
            )
        if (
            not isinstance(evaluator_batches, int)
            or isinstance(evaluator_batches, bool)
            or evaluator_batches < 0
            or evaluator_batches > evaluator_calls
        ):
            raise ResearchContractError(
                "receipt evaluator batch count must be between zero and evaluator calls"
            )
        raw_values = record.get("root_values")
        if not isinstance(raw_values, Mapping) or not raw_values:
            raise ResearchContractError("receipt root values must be a non-empty object")
        root_values: list[tuple[str, float]] = []
        for action, value in raw_values.items():
            if not isinstance(action, str) or not action:
                raise ResearchContractError("receipt root action must be a non-empty string")
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ResearchContractError("receipt root values must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ResearchContractError("receipt root values must be finite")
            root_values.append((action, number))
        raw_resources = record.get("resource_accounting")
        if raw_resources is not None and not isinstance(raw_resources, Mapping):
            raise ResearchContractError("receipt resource_accounting must be an object")
        resources = (
            ResourceAccounting.from_record(raw_resources)
            if isinstance(raw_resources, Mapping)
            else None
        )
        if resources is not None and (
            resources.verified_execution_classes_consumed != consumed
            or resources.evaluator_calls != evaluator_calls
            or resources.evaluator_batches != evaluator_batches
        ):
            raise ResearchContractError("receipt resource accounting disagrees with measured counts")
        return cls(
            method=values["method"],
            packet_digest=values["packet_digest"],
            matched_spec_digest=values["matched_spec_digest"],
            input_digest=values["input_digest"],
            posterior_digest=values["posterior_digest"],
            posterior_semantic_digest=values["posterior_semantic_digest"],
            evaluator_digest=values["evaluator_digest"],
            evaluator_checkpoint_digest=values["evaluator_checkpoint_digest"],
            mechanics_identity_digest=values["mechanics_identity_digest"],
            mechanics_evidence_digest=values["mechanics_evidence_digest"],
            showdown_commit=values["showdown_commit"],
            chance_treatment=values["chance_treatment"],
            transition_oracle_digest=values["transition_oracle_digest"],
            transition_program_digest=values["transition_program_digest"],
            transition_artifact_digest=values["transition_artifact_digest"],
            transition_program_source=values["transition_program_source"],
            compute_budget=ComputeBudget.from_record(raw_budget),
            budget_unit_definition=values["budget_unit_definition"],
            evaluator_call_unit_definition=values["evaluator_call_unit_definition"],
            consumed=consumed,
            evaluator_calls=evaluator_calls,
            evaluator_batches=evaluator_batches,
            chosen_action=values["chosen_action"],
            root_values=tuple(sorted(root_values)),
            resource_accounting=resources,
        )
