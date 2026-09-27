"""Matched-experiment specifications, budgets, accounting, and receipts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from azelficoast.research.decision_contracts import (
    BeliefInput,
    FrozenJSONObject,
    MechanicsIdentity,
    PublicDecisionInput,
    ResearchContractError,
    _public_payload,
    stable_digest,
)

COMPUTE_RECEIPT_SCHEMA = "azelficoast.matched-search-receipt"
COMPUTE_RECEIPT_SCHEMA_VERSION = 4
COMPUTE_BUDGET_UNIT_DEFINITION = (
    "one transition_evaluation per verified whole-turn execution class consumed"
)
EVALUATOR_CALL_UNIT_DEFINITION = (
    "one learned value prediction per successor public information set"
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
            "verified-material-oracle",
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
