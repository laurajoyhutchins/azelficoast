"""Budgeted matched search over verified whole-turn TransitionPrograms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.belief_evaluator import BeliefEvaluatorRuntime
from azelficoast.matched_comparison import (
    METHODS,
    PACKET_SCHEMA,
    PACKET_SCHEMA_VERSION,
    RECEIPT_SCHEMA,
    RECEIPT_SCHEMA_VERSION,
    _sha256,
)
from azelficoast.transition_oracle import ORACLE_SCHEMA, ORACLE_SCHEMA_VERSION
from azelficoast.transition_program_search import (
    TransitionProgramSearchError,
    search_transition_program,
)
from azelficoast.whole_turn_program import (
    PROGRAM_SET_SCHEMA,
    PROGRAM_SET_SCHEMA_VERSION,
    compile_whole_turn_programs,
    program_for_action,
)

class MatchedSearchExecutionError(ValueError):
    """Raised when a frozen matched-search work item cannot execute faithfully."""


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedSearchExecutionError(f"{path}: expected a JSON object")
    return value


def _artifact_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _validated_evaluator(
    packet: Mapping[str, Any],
    evaluator: Any,
) -> str:
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
        raise MatchedSearchExecutionError(
            "frozen evaluator identity lacks checkpoint digest"
        )
    if getattr(evaluator, "spec", None) is None or not callable(
        getattr(evaluator, "predict", None)
    ):
        raise MatchedSearchExecutionError(
            "loaded evaluator does not expose the frozen inference contract"
        )
    return checkpoint_digest


def _validated_program(
    *,
    packet: Mapping[str, Any],
    posterior: Mapping[str, Any],
    transition_artifact: Mapping[str, Any],
    method: str,
) -> tuple[dict[str, Any], str]:
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
    if posterior.get("treatment") != packet.get("posterior_treatment"):
        raise MatchedSearchExecutionError("posterior treatment drifted after freezing")

    schema = transition_artifact.get("schema")
    schema_version = transition_artifact.get("schema_version")
    if schema == PROGRAM_SET_SCHEMA and schema_version == PROGRAM_SET_SCHEMA_VERSION:
        program_set = dict(transition_artifact)
        source = "verified-transition-program"
    elif schema == ORACLE_SCHEMA and schema_version == ORACLE_SCHEMA_VERSION:
        raw_transitions = transition_artifact.get("transitions")
        if not isinstance(raw_transitions, list):
            raise MatchedSearchExecutionError(
                "legacy transition oracle has no transition matrix"
            )
        for transition in raw_transitions:
            if not isinstance(transition, Mapping):
                raise MatchedSearchExecutionError("legacy transition must be an object")
            outcomes = transition.get("outcomes")
            if not isinstance(outcomes, list):
                raise MatchedSearchExecutionError(
                    "legacy transition has no chance outcomes"
                )
            if any(
                isinstance(outcome, Mapping)
                and "continuation_transitions" in outcome
                for outcome in outcomes
            ):
                raise MatchedSearchExecutionError(
                    "depth-1 receipt cannot consume deeper continuation evidence"
                )
        program_set = compile_whole_turn_programs(transition_artifact)
        source = "compiled-legacy-oracle"
    else:
        raise MatchedSearchExecutionError(
            "expected a whole-turn transition program or compatible frozen oracle"
        )

    if program_set.get("source_fixture_id") != packet.get("fixture_id"):
        raise MatchedSearchExecutionError(
            "transition program belongs to another fixture"
        )
    if program_set.get("showdown_commit") != packet.get("showdown_commit"):
        raise MatchedSearchExecutionError(
            "transition program used another Showdown revision"
        )
    legal_actions = packet.get("legal_actions")
    if (
        not isinstance(legal_actions, list)
        or program_set.get("legal_actions") != legal_actions
    ):
        raise MatchedSearchExecutionError(
            "transition-program legal actions differ from frozen packet"
        )

    posterior_worlds = posterior.get("worlds")
    program_world_ids = program_set.get("world_ids")
    if not isinstance(posterior_worlds, list) or not posterior_worlds:
        raise MatchedSearchExecutionError("posterior has no hidden-world support")
    if not isinstance(program_world_ids, list) or not program_world_ids:
        raise MatchedSearchExecutionError(
            "transition program has no hidden-world support"
        )
    posterior_ids = [
        str(world.get("world_id"))
        for world in posterior_worlds
        if isinstance(world, Mapping)
    ]
    if len(posterior_ids) != len(posterior_worlds) or set(posterior_ids) != set(
        map(str, program_world_ids)
    ):
        raise MatchedSearchExecutionError(
            "posterior and transition program have different hidden-world support"
        )

    return program_set, source


def _required_transition_evaluations(
    program_set: Mapping[str, Any],
    legal_actions: Sequence[str],
) -> int:
    total = 0
    for action in legal_actions:
        try:
            program = program_for_action(program_set, action)
        except Exception as error:
            raise MatchedSearchExecutionError(str(error)) from error
        classes = program.get("classes")
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
    evaluator: Any,
) -> dict[str, Any]:
    """Execute one matched method using a TransitionProgram successor surface."""

    if (oracle is None) == (transition_program is None):
        raise MatchedSearchExecutionError(
            "provide exactly one of transition_program or legacy oracle"
        )
    artifact = transition_program if transition_program is not None else oracle
    assert artifact is not None

    checkpoint_digest = _validated_evaluator(packet, evaluator)
    program_set, source = _validated_program(
        packet=packet,
        posterior=posterior,
        transition_artifact=artifact,
        method=method,
    )
    budget = packet.get("compute_budget")
    if not isinstance(budget, Mapping) or budget.get("unit") != "transition_evaluations":
        raise MatchedSearchExecutionError(
            "receipt executor requires transition_evaluations budget"
        )

    legal_actions = list(packet["legal_actions"])
    required = _required_transition_evaluations(program_set, legal_actions)
    authorized = int(budget.get("authorized", 0))
    if required > authorized:
        raise MatchedSearchExecutionError(
            f"{method}: requires {required} transitions but budget authorizes {authorized}"
        )

    try:
        search = search_transition_program(
            program_set=program_set,
            posterior=posterior,
            method=method,
            evaluator=evaluator,
        )
    except TransitionProgramSearchError as error:
        raise MatchedSearchExecutionError(str(error)) from error
    if int(search["transition_evaluations"]) != required:
        raise MatchedSearchExecutionError(
            "TransitionProgram search consumed an unexpected execution-class count"
        )

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "method": method,
        "input_digest": packet["input_digest"],
        "evaluator_digest": packet["evaluator_digest"],
        "evaluator_checkpoint_digest": checkpoint_digest,
        "evaluator_calls": int(search["evaluator_calls"]),
        "evaluator_call_unit_definition": (
            "one learned value prediction per successor public information set"
        ),
        "packet_digest": packet["packet_digest"],
        "posterior_digest": packet["posterior_digest"],
        "transition_program_digest": str(search["transition_program_digest"]),
        "transition_program_source": source,
        "transition_artifact_digest": _artifact_digest(artifact),
        "showdown_commit": packet["showdown_commit"],
        "compute_budget": dict(budget),
        "consumed": required,
        "budget_unit_definition": (
            "one transition_evaluation per verified whole-turn execution class consumed"
        ),
        "chosen_action": search["chosen_action"],
        "root_values": dict(search["root_values"]),
    }
    if source == "compiled-legacy-oracle":
        receipt["transition_oracle_digest"] = _artifact_digest(artifact)
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
    kwargs = {
        "packet": _load_object(args.packet),
        "posterior": _load_object(args.posterior),
        "method": args.method,
        "evaluator": evaluator,
    }
    if transition_artifact.get("schema") == PROGRAM_SET_SCHEMA:
        kwargs["transition_program"] = transition_artifact
    else:
        kwargs["oracle"] = transition_artifact
    result = execute_method(**kwargs)
    _write_json(args.output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
