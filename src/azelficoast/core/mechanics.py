"""Validated mechanics boundary for whole-turn transition programs.

Serialized transition artifacts are admitted here once. Search receives this
immutable executor instead of interpreting an oracle or transition-program mapping.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping, Protocol

from azelficoast.research.contracts import (
    BeliefInput,
    BeliefTransportIndex,
    FrozenJSONObject,
    MechanicsIdentity,
    ResearchContractError,
    stable_digest,
)
from azelficoast.transition_oracle import (
    ORACLE_SCHEMA,
    ORACLE_SCHEMA_VERSION,
    canonical_json,
    sha256_json,
)
from azelficoast.whole_turn_program import (
    PROGRAM_SET_SCHEMA,
    PROGRAM_SET_SCHEMA_VERSION,
    WholeTurnProgramError,
    compile_whole_turn_programs,
    program_for_action,
)


class MechanicsContractError(ValueError):
    """Raised when exact admitted mechanics evidence is absent or inconsistent."""


@dataclass(frozen=True, slots=True)
class MechanicsExecutionRequest:
    mechanics_identity: MechanicsIdentity
    action: str


@dataclass(frozen=True, slots=True)
class MechanicsExecutionResult:
    action: str
    program: FrozenJSONObject
    dependency_evidence_digest: str


class MechanicsExecutor(Protocol):
    """The only mechanics interface used by transition-program search."""

    @property
    def identity(self) -> MechanicsIdentity:
        """Pinned mechanics revision and admitted semantics schema."""

    @property
    def legal_actions(self) -> tuple[str, ...]:
        """Frozen public root action set."""

    @property
    def semantic_evidence_digest(self) -> str:
        """Identity for the dependency/refinement evidence consumed by search."""

    @property
    def transition_program_digest(self) -> str:
        """Content identity of the admitted transition program set."""

    @property
    def world_ids(self) -> tuple[str, ...]:
        """Opaque adapter IDs used only to join the frozen program artifact."""

    @property
    def program_set(self) -> FrozenJSONObject:
        """Canonical verified artifact retained behind the mechanics interface."""

    def execute(self, request: MechanicsExecutionRequest) -> MechanicsExecutionResult:
        """Return the exact admitted whole-turn program for a legal root action."""


def _artifact_digest(value: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
            "utf-8"
        )
    ).hexdigest()


def _reject_incomplete_oracle(artifact: Mapping[str, object]) -> None:
    candidates = artifact.get("dependency_candidates")
    if not isinstance(candidates, list) or not all(
        isinstance(field, str) and field for field in candidates
    ):
        raise MechanicsContractError(
            "transition oracle lacks a complete dependency_candidates list"
        )
    declared = artifact.get("declared_reads")
    actions = artifact.get("legal_actions")
    transitions = artifact.get("transitions")
    if not isinstance(declared, Mapping) or not isinstance(actions, list):
        raise MechanicsContractError("transition oracle lacks declared dependency reads")
    for action in actions:
        fields = declared.get(action)
        if not isinstance(fields, list) or not all(isinstance(field, str) for field in fields):
            raise MechanicsContractError(f"{action}: declared dependency reads are incomplete")
    if not isinstance(transitions, list):
        raise MechanicsContractError("transition oracle lacks its transition matrix")
    for row in transitions:
        if not isinstance(row, Mapping):
            raise MechanicsContractError("transition oracle contains a malformed transition")
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, list):
            raise MechanicsContractError("transition oracle transition has no outcomes")
        for outcome in outcomes:
            if not isinstance(outcome, Mapping) or not isinstance(
                outcome.get("hidden_reads"), list
            ):
                raise MechanicsContractError("transition outcome lacks hidden-read evidence")
            if any(not isinstance(field, str) for field in outcome["hidden_reads"]):
                raise MechanicsContractError("transition outcome has malformed hidden-read evidence")
            if "continuation_transitions" in outcome:
                raise MechanicsContractError(
                    "depth-1 receipt cannot consume deeper continuation evidence"
                )


def _validate_program_shape(
    program_set: Mapping[str, object],
    *,
    identity: MechanicsIdentity,
    fixture_id: str,
    belief: BeliefInput,
    transport_index: BeliefTransportIndex,
    expected_actions: tuple[str, ...],
    expected_world_ids: set[str],
) -> None:
    raw_actions = program_set.get("legal_actions")
    raw_world_ids = program_set.get("world_ids")
    candidates = program_set.get("dependency_candidates")
    programs = program_set.get("programs")
    if raw_actions != list(expected_actions):
        raise MechanicsContractError("transition-program legal actions differ from matched input")
    if (
        not isinstance(raw_world_ids, list)
        or not all(isinstance(world_id, str) and world_id for world_id in raw_world_ids)
        or len(set(raw_world_ids)) != len(raw_world_ids)
        or set(raw_world_ids) != expected_world_ids
    ):
        raise MechanicsContractError("posterior and transition program have different support")
    if not isinstance(candidates, list) or not all(
        isinstance(field, str) and field for field in candidates
    ):
        raise MechanicsContractError("transition program lacks dependency evidence")
    semantic_worlds = {world.semantic_identity: world for world in belief.model_worlds}
    semantic_by_transport = dict(transport_index.bindings)
    hidden_by_semantic: dict[str, Mapping[str, object]] = {}
    for world in belief.model_worlds:
        if world.hidden_state is None:
            raise MechanicsContractError(
                "posterior must retain correlated hidden state for mechanics execution"
            )
        hidden_by_semantic[world.semantic_identity] = world.hidden_state.to_record()
    observed_hidden_fields = {
        field for hidden in hidden_by_semantic.values() for field in hidden
    }
    if set(candidates) - observed_hidden_fields:
        raise MechanicsContractError("transition program names unknown dependency candidates")
    if not isinstance(programs, list) or len(programs) != len(expected_actions):
        raise MechanicsContractError("transition program lacks one program per legal action")

    seen_actions: set[str] = set()
    for action in expected_actions:
        try:
            program = program_for_action(program_set, action)
        except WholeTurnProgramError as error:
            raise MechanicsContractError(str(error)) from error
        if action in seen_actions:
            raise MechanicsContractError("transition program repeats a root action")
        seen_actions.add(action)
        dependency_fields = program.get("dependency_fields")
        partition_method = program.get("partition_method")
        effect_signature = program.get("effect_signature")
        partition_key_hash = program.get("partition_key_hash")
        classes = program.get("classes")
        if (
            not isinstance(dependency_fields, list)
            or not all(isinstance(field, str) for field in dependency_fields)
            or set(dependency_fields) - set(candidates)
        ):
            raise MechanicsContractError(f"{action}: dependency fields are incomplete")
        if not isinstance(partition_method, str) or not partition_method:
            raise MechanicsContractError(f"{action}: refinement method is missing")
        if partition_method not in {
            "finite-support-minimal-semantics",
            "dynamic-read-refinement",
        }:
            raise MechanicsContractError(f"{action}: unsupported refinement method")
        if not isinstance(effect_signature, str) or not effect_signature:
            raise MechanicsContractError(f"{action}: mechanics effect signature is missing")
        if not isinstance(partition_key_hash, str) or not partition_key_hash:
            raise MechanicsContractError(f"{action}: dependency partition identity is missing")
        if not isinstance(classes, list) or not classes:
            raise MechanicsContractError(f"{action}: transition program has no classes")

        covered: set[str] = set()
        signature_rows: list[dict[str, object]] = []
        for row in classes:
            if not isinstance(row, Mapping):
                raise MechanicsContractError(f"{action}: execution class is malformed")
            members = row.get("member_world_ids")
            reads = row.get("read_fields")
            projection_key = row.get("projection_key")
            class_id = row.get("class_id")
            outcomes = row.get("outcomes")
            representative = row.get("representative_world_id")
            semantic_hash = row.get("semantic_hash")
            if (
                not isinstance(members, list)
                or not members
                or not all(isinstance(world_id, str) for world_id in members)
                or not isinstance(reads, list)
                or not all(isinstance(field, str) for field in reads)
                or not isinstance(projection_key, list)
                or not isinstance(class_id, str)
                or not class_id
                or not isinstance(outcomes, list)
                or not outcomes
                or not isinstance(representative, str)
                or representative not in members
                or not isinstance(semantic_hash, str)
                or not semantic_hash
                or len(set(members)) != len(members)
            ):
                raise MechanicsContractError(f"{action}: incomplete class refinement evidence")
            if set(reads) - set(dependency_fields) or set(reads) - set(candidates):
                raise MechanicsContractError(f"{action}: class reads unknown dependency fields")
            expected_projection = [
                canonical_json(hidden_by_semantic[
                    semantic_by_transport[str(representative)]
                ].get(field))
                for field in reads
            ]
            if projection_key != expected_projection:
                raise MechanicsContractError(f"{action}: representative dependency projection is invalid")
            for member_id in members:
                semantic_identity = semantic_by_transport.get(member_id)
                if semantic_identity not in semantic_worlds:
                    raise MechanicsContractError(f"{action}: class member has unknown semantics")
                member_hidden = hidden_by_semantic[semantic_identity]
                member_projection = [canonical_json(member_hidden.get(field)) for field in reads]
                if member_projection != projection_key:
                    raise MechanicsContractError(
                        f"{action}: class members disagree on their dependency projection"
                    )
            expected_class_id = "transition-class-" + sha256_json(
                {
                    "action": action,
                    "read_fields": reads,
                    "key": projection_key,
                    "semantic_hash": semantic_hash,
                }
            )[:24]
            if class_id != expected_class_id:
                raise MechanicsContractError(f"{action}: execution class identity is invalid")
            if covered.intersection(members):
                raise MechanicsContractError(f"{action}: execution classes overlap")
            if set(members) - expected_world_ids:
                raise MechanicsContractError(f"{action}: class references an unknown world")
            chance_total = 0.0
            for outcome in outcomes:
                if not isinstance(outcome, Mapping):
                    raise MechanicsContractError(f"{action}: transition outcome is malformed")
                probability = outcome.get("probability")
                successor = outcome.get("successor")
                legal = outcome.get("legal_actions")
                if (
                    not isinstance(probability, (int, float))
                    or isinstance(probability, bool)
                    or not math.isfinite(float(probability))
                    or float(probability) <= 0
                    or not isinstance(successor, Mapping)
                    or not isinstance(legal, list)
                    or not legal
                    or not all(isinstance(choice, str) and choice for choice in legal)
                ):
                    raise MechanicsContractError(f"{action}: transition outcome is incomplete")
                chance_total += float(probability)
            if abs(chance_total - 1.0) > 1e-9:
                raise MechanicsContractError(f"{action}: transition chance mass is not normalized")
            if semantic_hash != sha256_json(outcomes):
                raise MechanicsContractError(f"{action}: class transition signature is invalid")
            signature_rows.append(
                {
                    "class_id": class_id,
                    "members": members,
                    "semantic_hash": semantic_hash,
                }
            )
            covered.update(members)
        if covered != expected_world_ids:
            raise MechanicsContractError(f"{action}: execution classes do not cover support")
        if program.get("classes_out") != len(classes):
            raise MechanicsContractError(f"{action}: class count differs from its evidence")
        if program.get("worlds_in") != len(expected_world_ids):
            raise MechanicsContractError(f"{action}: support count differs from its evidence")
        expected_partition_hash = sha256_json(
            {
                "action": action,
                "partition_method": partition_method,
                "fields": dependency_fields,
                "classes": signature_rows,
            }
        )
        if partition_key_hash != expected_partition_hash:
            raise MechanicsContractError(f"{action}: dependency partition signature is invalid")
        expected_effect_signature = "sha256:" + sha256_json(
            {
                "showdown_commit": identity.revision,
                "source_fixture_id": fixture_id,
                "action": action,
                "dependency_fields": dependency_fields,
                "partition_key_hash": partition_key_hash,
            }
        )
        if effect_signature != expected_effect_signature:
            raise MechanicsContractError(f"{action}: mechanics effect signature is invalid")


@dataclass(frozen=True, slots=True)
class VerifiedTransitionProgramSet:
    """Immutable admitted mechanics artifact bound to public actions and belief support."""

    identity: MechanicsIdentity
    fixture_id: str
    legal_actions: tuple[str, ...]
    world_ids: tuple[str, ...]
    program_set: FrozenJSONObject
    source: str
    transition_artifact_digest: str
    transition_program_digest: str
    semantic_evidence_digest: str

    @classmethod
    def from_artifact(
        cls,
        *,
        artifact: Mapping[str, object],
        identity: MechanicsIdentity,
        fixture_id: str,
        legal_actions: tuple[str, ...],
        belief: BeliefInput,
        transport_index: BeliefTransportIndex,
    ) -> VerifiedTransitionProgramSet:
        if not fixture_id or not legal_actions:
            raise MechanicsContractError("mechanics input lacks source or legal actions")
        if len(set(legal_actions)) != len(legal_actions):
            raise MechanicsContractError("mechanics input legal actions are not unique")
        if artifact.get("showdown_commit") != identity.revision:
            raise MechanicsContractError("mechanics artifact used another Showdown revision")
        if artifact.get("source_fixture_id") != fixture_id:
            raise MechanicsContractError("mechanics artifact belongs to another fixture")

        schema = artifact.get("schema")
        version = artifact.get("schema_version")
        if schema == PROGRAM_SET_SCHEMA and version == PROGRAM_SET_SCHEMA_VERSION:
            program_set = dict(artifact)
            source = "verified-transition-program"
        elif schema == ORACLE_SCHEMA and version == ORACLE_SCHEMA_VERSION:
            _reject_incomplete_oracle(artifact)
            try:
                program_set = compile_whole_turn_programs(artifact)
            except (WholeTurnProgramError, ResearchContractError) as error:
                raise MechanicsContractError(str(error)) from error
            source = "compiled-legacy-oracle"
        else:
            raise MechanicsContractError("unsupported mechanics artifact schema")

        if program_set.get("schema") != PROGRAM_SET_SCHEMA:
            raise MechanicsContractError("mechanics adapter produced an unsupported program schema")
        if program_set.get("schema_version") != PROGRAM_SET_SCHEMA_VERSION:
            raise MechanicsContractError("mechanics adapter produced an unsupported program version")
        if program_set.get("showdown_commit") != identity.revision:
            raise MechanicsContractError("transition program used another Showdown revision")
        if program_set.get("source_fixture_id") != fixture_id:
            raise MechanicsContractError("transition program belongs to another fixture")

        transport_ids = {identifier for identifier, _identity in transport_index.bindings}
        if not transport_ids:
            raise MechanicsContractError("posterior lacks opaque IDs required by mechanics evidence")
        semantic_ids = {world.semantic_identity for world in belief.model_worlds}
        if any(identity not in semantic_ids for _identifier, identity in transport_index.bindings):
            raise MechanicsContractError("posterior transport index references unknown semantics")
        transport_weight_by_id = dict(transport_index.weights)
        if set(transport_weight_by_id) != transport_ids:
            raise MechanicsContractError("posterior transport weights are incomplete")
        if any(
            not math.isfinite(weight) or weight <= 0
            for weight in transport_weight_by_id.values()
        ) or abs(math.fsum(transport_weight_by_id.values()) - 1.0) > 1e-12:
            raise MechanicsContractError("posterior transport weights are not normalized")
        transport_semantics = dict(transport_index.bindings)
        for world in belief.model_worlds:
            projected_mass = math.fsum(
                weight
                for identifier, weight in transport_weight_by_id.items()
                if transport_semantics[identifier] == world.semantic_identity
            )
            if abs(projected_mass - world.weight) > 1e-12:
                raise MechanicsContractError("posterior transport mass differs from semantics")
        _validate_program_shape(
            program_set,
            identity=identity,
            fixture_id=fixture_id,
            belief=belief,
            transport_index=transport_index,
            expected_actions=legal_actions,
            expected_world_ids=transport_ids,
        )
        try:
            frozen_program = FrozenJSONObject.from_mapping(program_set)
        except ResearchContractError as error:
            raise MechanicsContractError(str(error)) from error

        program_digest = sha256_json(program_set)
        admitted_programs = program_set.get("programs")
        if not isinstance(admitted_programs, list):
            raise MechanicsContractError("transition program has no admitted programs")
        evidence = stable_digest(
            {
                "program_digest": program_digest,
                "dependency_candidates": program_set["dependency_candidates"],
                "programs": [
                    {
                        "action": program.get("action"),
                        "dependency_fields": program.get("dependency_fields"),
                        "partition_method": program.get("partition_method"),
                        "partition_key_hash": program.get("partition_key_hash"),
                        "classes": [
                            {
                                "read_fields": row.get("read_fields"),
                                "projection_key": row.get("projection_key"),
                                "member_world_ids": row.get("member_world_ids"),
                                "semantic_hash": row.get("semantic_hash"),
                            }
                            for row in program["classes"]
                            if isinstance(row, Mapping)
                        ],
                    }
                    for program in admitted_programs
                    if isinstance(program, Mapping)
                ],
            }
        )
        return cls(
            identity=identity,
            fixture_id=fixture_id,
            legal_actions=legal_actions,
            world_ids=tuple(sorted(transport_ids)),
            program_set=frozen_program,
            source=source,
            transition_artifact_digest=_artifact_digest(artifact),
            transition_program_digest=program_digest,
            semantic_evidence_digest=evidence,
        )

    def execute(self, request: MechanicsExecutionRequest) -> MechanicsExecutionResult:
        if request.mechanics_identity != self.identity:
            raise MechanicsContractError("mechanics request used another revision")
        if request.action not in self.legal_actions:
            raise MechanicsContractError("mechanics request used a non-legal public action")
        record = self.program_set.to_record()
        try:
            raw_program = program_for_action(record, request.action)
        except WholeTurnProgramError as error:
            raise MechanicsContractError(str(error)) from error
        try:
            frozen_program = FrozenJSONObject.from_mapping(raw_program)
        except ResearchContractError as error:
            raise MechanicsContractError(str(error)) from error
        evidence_digest = stable_digest(
            {
                "action": request.action,
                "dependency_fields": raw_program.get("dependency_fields"),
                "partition_method": raw_program.get("partition_method"),
                "partition_key_hash": raw_program.get("partition_key_hash"),
                "classes": raw_program.get("classes"),
            }
        )
        return MechanicsExecutionResult(
            action=request.action,
            program=frozen_program,
            dependency_evidence_digest=evidence_digest,
        )
