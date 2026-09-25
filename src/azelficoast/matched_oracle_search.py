"""Budgeted matched search over validated whole-turn mechanics programs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from azelficoast.belief_evaluator import (
    BeliefEvaluatorInput,
    BeliefEvaluatorRuntime,
    BeliefEvaluatorSpec,
    BeliefPrediction,
)
from azelficoast.matched_comparison import (
    METHODS,
    MatchedComparisonError,
    PACKET_SCHEMA,
    PACKET_SCHEMA_VERSION,
    RECEIPT_SCHEMA,
    RECEIPT_SCHEMA_VERSION,
    _validate_packet,
    _sha256,
)
from azelficoast.mechanics_contracts import (
    MechanicsContractError,
    MechanicsExecutionRequest,
    VerifiedTransitionProgramSet,
)
from azelficoast.research_contracts import (
    BeliefInput,
    BeliefTransportIndex,
    COMPUTE_BUDGET_UNIT_DEFINITION,
    EVALUATOR_CALL_UNIT_DEFINITION,
    MatchedExperimentSpec,
    PublicDecisionInput,
    ResearchContractError,
    parse_belief_artifact,
)
from azelficoast.transition_program_search import (
    TransitionProgramSearchError,
    search_transition_program,
)
from azelficoast.whole_turn_program import PROGRAM_SET_SCHEMA


class MatchedSearchExecutionError(ValueError):
    """Raised when a frozen matched-search work item cannot execute faithfully."""


class Evaluator(Protocol):
    spec: BeliefEvaluatorSpec

    @property
    def identity(self) -> Mapping[str, Any]:
        """Stable evaluator identity including its checkpoint digest."""

    def predict(self, inputs: BeliefEvaluatorInput) -> BeliefPrediction:
        """Evaluate one validated public-belief input."""


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedSearchExecutionError(f"{path}: expected a JSON object")
    return value


def _validated_evaluator(packet: Mapping[str, Any], evaluator: Evaluator) -> str:
    expected = packet.get("evaluator")
    identity = getattr(evaluator, "identity", None)
    if not isinstance(expected, Mapping) or not isinstance(identity, Mapping):
        raise MatchedSearchExecutionError(
            "matched execution requires a frozen evaluator identity"
        )
    if dict(identity) != dict(expected):
        raise MatchedSearchExecutionError(
            "loaded evaluator identity differs from the frozen packet"
        )
    checkpoint_digest = identity.get("checkpoint_digest")
    if not isinstance(checkpoint_digest, str) or not checkpoint_digest:
        raise MatchedSearchExecutionError("frozen evaluator identity lacks checkpoint digest")
    if getattr(evaluator, "spec", None) is None or not callable(
        getattr(evaluator, "predict", None)
    ):
        raise MatchedSearchExecutionError(
            "loaded evaluator does not expose the frozen inference contract"
        )
    return checkpoint_digest


def _validated_inputs(
    *,
    packet: Mapping[str, Any],
    posterior: Mapping[str, Any],
    transition_artifact: Mapping[str, Any],
    method: str,
) -> tuple[BeliefInput, BeliefTransportIndex, MatchedExperimentSpec, VerifiedTransitionProgramSet]:
    if (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("schema_version") != PACKET_SCHEMA_VERSION
    ):
        raise MatchedSearchExecutionError("unexpected matched-search packet schema")
    if method not in METHODS:
        raise MatchedSearchExecutionError(f"unknown search method {method!r}")
    if int(packet.get("depth", 0)) != 1:
        raise MatchedSearchExecutionError(
            "TransitionProgram receipt executor currently supports depth 1 only"
        )
    if packet.get("posterior_digest") != _sha256(posterior):
        raise MatchedSearchExecutionError("posterior artifact does not match frozen packet")

    raw_spec = packet.get("matched_spec")
    if not isinstance(raw_spec, Mapping):
        raise MatchedSearchExecutionError("matched packet lacks its frozen specification")
    try:
        spec = MatchedExperimentSpec.from_record(raw_spec)
        belief, transport_index = parse_belief_artifact(posterior)
    except ResearchContractError as error:
        raise MatchedSearchExecutionError(str(error)) from error
    if packet.get("matched_spec_digest") != spec.digest:
        raise MatchedSearchExecutionError("matched packet specification digest is invalid")
    if packet.get("posterior_treatment") != belief.treatment:
        raise MatchedSearchExecutionError("posterior treatment drifted after freezing")
    if (
        packet.get("posterior_semantic_digest") != belief.semantic_digest
        or spec.posterior_semantic_digest != belief.semantic_digest
    ):
        raise MatchedSearchExecutionError("posterior semantics differ from frozen packet")

    raw_actions = packet.get("legal_actions")
    if (
        not isinstance(raw_actions, list)
        or not raw_actions
        or not all(isinstance(action, str) and action for action in raw_actions)
        or len(set(raw_actions)) != len(raw_actions)
    ):
        raise MatchedSearchExecutionError("frozen packet legal actions are malformed")
    if tuple(raw_actions) != spec.legal_actions:
        raise MatchedSearchExecutionError("matched specification action set drifted")

    try:
        public = PublicDecisionInput(
            fixture_id=spec.fixture_id,
            battle_tag=spec.battle_tag,
            public_state=spec.public_state,
            legal_actions=spec.legal_actions,
            public_history_identity=spec.public_history_identity,
            mechanics_identity=spec.mechanics_identity,
        )
        if public.state_digest != spec.public_state_digest:
            raise ResearchContractError("matched specification public state digest is invalid")
        mechanics = VerifiedTransitionProgramSet.from_artifact(
            artifact=transition_artifact,
            identity=spec.mechanics_identity,
            fixture_id=spec.fixture_id,
            legal_actions=spec.legal_actions,
            belief=belief,
            transport_index=transport_index,
        )
    except (MechanicsContractError, ResearchContractError) as error:
        raise MatchedSearchExecutionError(str(error)) from error
    return belief, transport_index, spec, mechanics


def _required_transition_evaluations(
    mechanics: VerifiedTransitionProgramSet,
) -> int:
    total = 0
    for action in mechanics.legal_actions:
        try:
            execution = mechanics.execute(
                MechanicsExecutionRequest(mechanics_identity=mechanics.identity, action=action)
            )
        except MechanicsContractError as error:
            raise MatchedSearchExecutionError(str(error)) from error
        classes = execution.program.to_record().get("classes")
        if not isinstance(classes, list) or not classes:
            raise MatchedSearchExecutionError(
                f"{action}: transition program has no execution classes"
            )
        total += len(classes)
    return total


def execute_method(
    *,
    packet: Mapping[str, Any],
    posterior: Mapping[str, Any],
    oracle: Mapping[str, Any] | None = None,
    transition_program: Mapping[str, Any] | None = None,
    method: str,
    evaluator: Evaluator,
) -> dict[str, Any]:
    """Execute one matched method through the mechanics and evaluator contracts."""
    if (oracle is None) == (transition_program is None):
        raise MatchedSearchExecutionError(
            "provide exactly one of transition_program or legacy oracle"
        )
    artifact = transition_program if transition_program is not None else oracle
    assert artifact is not None

    executor_started_ns = time.perf_counter_ns()
    try:
        _validate_packet(packet)
    except MatchedComparisonError as error:
        raise MatchedSearchExecutionError(str(error)) from error
    checkpoint_digest = _validated_evaluator(packet, evaluator)
    belief, transport_index, spec, mechanics = _validated_inputs(
        packet=packet,
        posterior=posterior,
        transition_artifact=artifact,
        method=method,
    )
    if packet.get("mechanics_identity_digest") != spec.mechanics_identity.identity_digest:
        raise MatchedSearchExecutionError("matched mechanics identity digest is invalid")
    if packet.get("evaluator_digest") != spec.evaluator_identity_digest:
        raise MatchedSearchExecutionError("matched evaluator identity differs from specification")
    prepared_ns = time.perf_counter_ns()
    budget = packet.get("compute_budget")
    if not isinstance(budget, Mapping) or budget != spec.compute_budget.to_record():
        raise MatchedSearchExecutionError("matched specification compute budget drifted")
    if budget.get("unit") != "transition_evaluations":
        raise MatchedSearchExecutionError("receipt executor requires transition_evaluations budget")

    required = _required_transition_evaluations(mechanics)
    authorized = spec.compute_budget.authorized
    if required > authorized:
        raise MatchedSearchExecutionError(
            f"{method}: requires {required} transitions but budget authorizes {authorized}"
        )

    search_started_ns = time.perf_counter_ns()
    try:
        search = search_transition_program(
            mechanics=mechanics,
            belief=belief,
            transport_index=transport_index,
            method=method,
            evaluator=evaluator,
        )
    except TransitionProgramSearchError as error:
        raise MatchedSearchExecutionError(str(error)) from error
    search_finished_ns = time.perf_counter_ns()
    if int(search["transition_evaluations"]) != required:
        raise MatchedSearchExecutionError(
            "TransitionProgram search consumed an unexpected execution-class count"
        )

    artifact_digest = mechanics.transition_artifact_digest
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "method": method,
        "input_digest": packet["input_digest"],
        "evaluator_digest": packet["evaluator_digest"],
        "evaluator_checkpoint_digest": checkpoint_digest,
        "evaluator_calls": int(search["evaluator_calls"]),
        "evaluator_call_unit_definition": EVALUATOR_CALL_UNIT_DEFINITION,
        "packet_digest": packet["packet_digest"],
        "matched_spec_digest": spec.digest,
        "posterior_digest": packet["posterior_digest"],
        "posterior_semantic_digest": belief.semantic_digest,
        "mechanics_identity_digest": spec.mechanics_identity.identity_digest,
        "mechanics_evidence_digest": str(search["mechanics_evidence_digest"]),
        "showdown_commit": spec.mechanics_identity.revision,
        "chance_treatment": spec.chance_treatment,
        "transition_oracle_digest": artifact_digest,
        "transition_program_digest": str(search["transition_program_digest"]),
        "transition_artifact_digest": artifact_digest,
        "transition_program_source": mechanics.source,
        "compute_budget": spec.compute_budget.to_record(),
        "consumed": required,
        "budget_unit_definition": COMPUTE_BUDGET_UNIT_DEFINITION,
        "chosen_action": search["chosen_action"],
        "root_values": dict(search["root_values"]),
        "resource_accounting": {
            "verified_execution_classes_consumed": required,
            "evaluator_calls": int(search["evaluator_calls"]),
            "executor_preparation_wall_ms": (prepared_ns - executor_started_ns) / 1_000_000.0,
            "search_wall_ms": (search_finished_ns - search_started_ns) / 1_000_000.0,
            "executor_wall_ms": (search_finished_ns - executor_started_ns) / 1_000_000.0,
            "transition_program_generation_included": (
                mechanics.source == "compiled-legacy-oracle"
            ),
            "transition_program_verification_included": False,
            "posterior_construction_included": False,
            "scope_note": (
                "Wall-clock fields cover this receipt executor only. Posterior construction "
                "and externally supplied TransitionProgram generation/verification must be "
                "reported by their producing stages rather than hidden inside the class budget."
            ),
        },
    }
    return receipt


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("posterior", type=Path)
    parser.add_argument("transitions", type=Path)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--evaluator-checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    evaluator = BeliefEvaluatorRuntime.from_checkpoint(args.evaluator_checkpoint)
    transition_artifact = _load_object(args.transitions)
    kwargs: dict[str, object] = {
        "packet": _load_object(args.packet),
        "posterior": _load_object(args.posterior),
        "method": args.method,
        "evaluator": evaluator,
    }
    if transition_artifact.get("schema") == PROGRAM_SET_SCHEMA:
        kwargs["transition_program"] = transition_artifact
    else:
        kwargs["oracle"] = transition_artifact
    result = execute_method(**kwargs)  # type: ignore[arg-type]
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
