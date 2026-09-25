"""Executable contract for matched determinization/information-set comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.belief.validity import (
    PosteriorValidityError,
    posterior_diagnostics,
)
from azelficoast.research.contracts import (
    BeliefInput,
    COMPUTE_BUDGET_UNIT_DEFINITION,
    COMPUTE_RECEIPT_SCHEMA,
    COMPUTE_RECEIPT_SCHEMA_VERSION,
    EVALUATOR_CALL_UNIT_DEFINITION,
    ComputeBudget,
    ComputeReceipt,
    MatchedExperimentSpec,
    MechanicsIdentity,
    PublicDecisionInput,
    ResearchContractError,
)

PLAN_SCHEMA = "azelficoast.matched-search-comparison-plan"
PLAN_SCHEMA_VERSION = 2
PACKET_SCHEMA = "azelficoast.matched-search-comparison-packet"
PACKET_SCHEMA_VERSION = 3
RESULT_SCHEMA = "azelficoast.matched-search-comparison-result"
RESULT_SCHEMA_VERSION = 4
RECEIPT_SCHEMA = COMPUTE_RECEIPT_SCHEMA
RECEIPT_SCHEMA_VERSION = COMPUTE_RECEIPT_SCHEMA_VERSION
EVALUATOR_SCHEMA = "azelficoast.belief-policy-value-evaluator"
EVALUATOR_SCHEMA_VERSION = 1

METHODS = ("determinization", "information_set")
POSTERIOR_TREATMENTS = (
    "oracle",
    "generator_faithful",
    "practical",
    "flattened",
    "sharpened",
    "widened_support",
)
OPPONENT_MODELS = ("fixed_observed_response", "two_sided_information_sets")
BUDGET_UNITS = ("transition_evaluations",)


class MatchedComparisonError(ValueError):
    """Raised when an experiment attempts an unmatched comparison."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedComparisonError(f"{path}: expected a JSON object")
    return value


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    if plan.get("schema") != PLAN_SCHEMA or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise MatchedComparisonError("unexpected matched-comparison plan schema")

    treatments = plan.get("posterior_treatments")
    if not isinstance(treatments, list) or not treatments:
        raise MatchedComparisonError("plan must preregister posterior treatments")
    normalized_treatments = [str(value) for value in treatments]
    if len(set(normalized_treatments)) != len(normalized_treatments):
        raise MatchedComparisonError("posterior treatments must be unique")
    unknown_treatments = set(normalized_treatments) - set(POSTERIOR_TREATMENTS)
    if unknown_treatments:
        raise MatchedComparisonError(
            f"unknown posterior treatments: {sorted(unknown_treatments)!r}"
        )

    budget = plan.get("compute_budget")
    if not isinstance(budget, Mapping):
        raise MatchedComparisonError("plan must declare a compute budget")
    unit = budget.get("unit")
    limit = budget.get("per_method_limit")
    if unit not in BUDGET_UNITS:
        raise MatchedComparisonError(f"compute budget unit must be one of {BUDGET_UNITS!r}")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise MatchedComparisonError("per-method compute limit must be a positive integer")

    opponent_model = plan.get("opponent_model")
    if opponent_model not in OPPONENT_MODELS:
        raise MatchedComparisonError(f"opponent model must be one of {OPPONENT_MODELS!r}")

    depths = plan.get("depths")
    if not isinstance(depths, list) or not depths:
        raise MatchedComparisonError("plan must preregister at least one search depth")
    normalized_depths = [int(depth) for depth in depths]
    if any(depth <= 0 for depth in normalized_depths):
        raise MatchedComparisonError("search depths must be positive")
    if sorted(set(normalized_depths)) != normalized_depths:
        raise MatchedComparisonError("search depths must be unique and ascending")

    predictors = plan.get("confirmatory_predictors")
    if not isinstance(predictors, list) or not 2 <= len(predictors) <= 3:
        raise MatchedComparisonError("plan must freeze two or three confirmatory predictors")
    if len(set(map(str, predictors))) != len(predictors):
        raise MatchedComparisonError("confirmatory predictors must be unique")

    if plan.get("cluster_unit") != "battle_tag":
        raise MatchedComparisonError("population inference must cluster by battle_tag")

    inference = plan.get("inference")
    if not isinstance(inference, Mapping):
        raise MatchedComparisonError("plan must declare population inference")
    replicates = inference.get("bootstrap_replicates")
    seed = inference.get("bootstrap_seed")
    if not isinstance(replicates, int) or isinstance(replicates, bool) or replicates < 1:
        raise MatchedComparisonError("bootstrap replicates must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise MatchedComparisonError("bootstrap seed must be an integer")

    evaluator = plan.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise MatchedComparisonError("plan must pin one learned evaluator")
    if (
        evaluator.get("schema") != EVALUATOR_SCHEMA
        or evaluator.get("schema_version") != EVALUATOR_SCHEMA_VERSION
    ):
        raise MatchedComparisonError("unexpected evaluator schema")
    checkpoint_digest = evaluator.get("checkpoint_digest")
    if not (
        isinstance(checkpoint_digest, str)
        and checkpoint_digest.startswith("sha256:")
        and len(checkpoint_digest) == 71
        and all(character in "0123456789abcdef" for character in checkpoint_digest[7:])
    ):
        raise MatchedComparisonError(
            "evaluator checkpoint_digest must be sha256:<64 lowercase hex>"
        )
    if evaluator.get("observability") != "public_belief_only":
        raise MatchedComparisonError("matched evaluator must be blind to the realized hidden state")
    if not isinstance(evaluator.get("architecture"), str) or not evaluator["architecture"]:
        raise MatchedComparisonError("evaluator must identify its architecture")
    if not isinstance(evaluator.get("spec"), Mapping):
        raise MatchedComparisonError("evaluator must pin its input/model spec")

    showdown_commit = plan.get("showdown_commit")
    if not isinstance(showdown_commit, str) or not showdown_commit:
        raise MatchedComparisonError("plan must pin a Showdown commit")

    return dict(plan)


def freeze_packet(
    *,
    plan: Mapping[str, Any],
    state: Mapping[str, Any],
    posterior: Mapping[str, Any],
    posterior_treatment: str,
    depth: int,
) -> dict[str, Any]:
    checked_plan = validate_plan(plan)
    if posterior_treatment not in checked_plan["posterior_treatments"]:
        raise MatchedComparisonError(
            f"posterior treatment {posterior_treatment!r} was not preregistered"
        )
    if int(depth) not in checked_plan["depths"]:
        raise MatchedComparisonError(f"depth {depth} was not preregistered")

    mechanics_identity = MechanicsIdentity.from_showdown_commit(
        str(checked_plan["showdown_commit"])
    )
    try:
        public = PublicDecisionInput.from_record(
            state,
            mechanics_identity=mechanics_identity,
        )
        belief = BeliefInput.from_record(posterior)
    except ResearchContractError as error:
        raise MatchedComparisonError(str(error)) from error

    fixture_id = public.fixture_id
    battle_tag = public.battle_tag
    public_state = public.public_state.to_record()
    legal_actions = list(public.legal_actions)
    if belief.treatment != posterior_treatment:
        raise MatchedComparisonError("posterior artifact treatment does not match packet")

    predictors = state.get("predictors")
    if not isinstance(predictors, Mapping):
        raise MatchedComparisonError("state must contain preregistered predictors")
    frozen_predictors: dict[str, Any] = {}
    for predictor in checked_plan["confirmatory_predictors"]:
        if predictor not in predictors:
            raise MatchedComparisonError(f"state lacks confirmatory predictor {predictor!r}")
        value = predictors[predictor]
        if not isinstance(value, (bool, int, float)):
            raise MatchedComparisonError(
                f"confirmatory predictor {predictor!r} must be numeric or boolean"
            )
        frozen_predictors[str(predictor)] = value

    if posterior.get("treatment") != posterior_treatment:
        raise MatchedComparisonError("posterior artifact treatment does not match packet")
    if posterior.get("conditioned_on_public_history") is not True:
        raise MatchedComparisonError("posterior must be conditioned on public history")
    if posterior.get("realized_hidden_state_revealed") is not False:
        raise MatchedComparisonError(
            "posterior may not reveal the realized hidden state"
        )
    worlds = posterior.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise MatchedComparisonError("posterior must contain hidden-world support")
    try:
        posterior_support = posterior_diagnostics(posterior)
    except PosteriorValidityError as error:
        raise MatchedComparisonError(f"invalid posterior support: {error}") from error
    state_digest = _sha256(
        {
            "fixture_id": fixture_id,
            "battle_tag": battle_tag,
            "public_state": public_state,
            "legal_actions": legal_actions,
        }
    )
    posterior_digest = _sha256(posterior)
    evaluator = dict(checked_plan["evaluator"])
    evaluator_digest = _sha256(evaluator)
    input_digest = _sha256(
        {
            "state_digest": state_digest,
            "posterior_digest": posterior_digest,
            "evaluator_digest": evaluator_digest,
            "showdown_commit": checked_plan["showdown_commit"],
            "depth": int(depth),
        }
    )
    budget = {
        "unit": checked_plan["compute_budget"]["unit"],
        "authorized": int(checked_plan["compute_budget"]["per_method_limit"]),
    }
    try:
        typed_budget = ComputeBudget.from_record(budget)
    except ResearchContractError as error:
        raise MatchedComparisonError(str(error)) from error
    chance_treatment = checked_plan.get(
        "chance_treatment",
        "shared_frozen_transition_oracle",
    )
    if not isinstance(chance_treatment, str) or not chance_treatment:
        raise MatchedComparisonError("chance treatment must be a non-empty string")
    matched_spec = MatchedExperimentSpec.build(
        public=public,
        belief=belief,
        evaluator_identity_digest=evaluator_digest,
        chance_treatment=chance_treatment,
        compute_budget=typed_budget,
        depth=int(depth),
        opponent_model=str(checked_plan["opponent_model"]),
    )

    packet = {
        "schema": PACKET_SCHEMA,
        "schema_version": PACKET_SCHEMA_VERSION,
        "fixture_id": fixture_id,
        "battle_tag": battle_tag,
        "posterior_treatment": posterior_treatment,
        "opponent_model": checked_plan["opponent_model"],
        "showdown_commit": checked_plan["showdown_commit"],
        "depth": int(depth),
        "state_digest": state_digest,
        "posterior_digest": posterior_digest,
        "posterior_support": posterior_support,
        "posterior_semantic_digest": belief.semantic_digest,
        "mechanics_identity_digest": mechanics_identity.identity_digest,
        "chance_treatment": chance_treatment,
        "matched_spec": matched_spec.to_record(),
        "matched_spec_digest": matched_spec.digest,
        "evaluator": evaluator,
        "evaluator_digest": evaluator_digest,
        "input_digest": input_digest,
        "legal_actions": list(legal_actions),
        "predictors": frozen_predictors,
        "compute_budget": budget,
        "work": [
            {
                "method": method,
                "input_digest": input_digest,
                "evaluator_digest": evaluator_digest,
                "compute_budget": dict(budget),
            }
            for method in METHODS
        ],
    }
    packet["packet_digest"] = _sha256(packet)
    return packet


def _validate_packet(packet: Mapping[str, Any]) -> None:
    if (
        packet.get("schema") != PACKET_SCHEMA
        or packet.get("schema_version") != PACKET_SCHEMA_VERSION
    ):
        raise MatchedComparisonError("unexpected comparison packet schema")
    packet_digest = packet.get("packet_digest")
    unsigned_packet = dict(packet)
    unsigned_packet.pop("packet_digest", None)
    if not isinstance(packet_digest, str) or _sha256(unsigned_packet) != packet_digest:
        raise MatchedComparisonError("comparison packet digest is invalid")
    work = packet.get("work")
    if not isinstance(work, list) or len(work) != len(METHODS):
        raise MatchedComparisonError("comparison packet must contain both methods")
    methods = [row.get("method") for row in work if isinstance(row, Mapping)]
    if tuple(methods) != METHODS:
        raise MatchedComparisonError("comparison methods are missing or reordered")
    expected_digest = packet.get("input_digest")
    expected_budget = packet.get("compute_budget")
    evaluator = packet.get("evaluator")
    expected_evaluator_digest = packet.get("evaluator_digest")
    if not isinstance(evaluator, Mapping):
        raise MatchedComparisonError("comparison packet lacks evaluator identity")
    if _sha256(evaluator) != expected_evaluator_digest:
        raise MatchedComparisonError("comparison packet evaluator digest is invalid")
    for row in work:
        if not isinstance(row, Mapping):
            raise MatchedComparisonError("comparison work item must be an object")
        if row.get("input_digest") != expected_digest:
            raise MatchedComparisonError("comparison methods do not share one input")
        if row.get("evaluator_digest") != expected_evaluator_digest:
            raise MatchedComparisonError("comparison methods do not share one evaluator")
        if row.get("compute_budget") != expected_budget:
            raise MatchedComparisonError("comparison methods do not share one budget")

    raw_spec = packet.get("matched_spec")
    if not isinstance(raw_spec, Mapping):
        raise MatchedComparisonError("comparison packet lacks its matched experiment spec")
    try:
        spec = MatchedExperimentSpec.from_record(raw_spec)
    except ResearchContractError as error:
        raise MatchedComparisonError(str(error)) from error
    if packet.get("matched_spec_digest") != spec.digest:
        raise MatchedComparisonError("comparison packet matched spec digest is invalid")
    if not spec.public_state_digest.startswith("sha256:") or packet.get(
        "state_digest"
    ) != spec.public_state_digest.removeprefix("sha256:"):
        raise MatchedComparisonError("matched spec public state differs from its packet")
    raw_actions = packet.get("legal_actions")
    if not isinstance(raw_actions, list) or set(raw_actions) != set(spec.legal_actions):
        raise MatchedComparisonError("matched spec action set differs from its packet")
    if packet.get("fixture_id") != spec.fixture_id or packet.get("battle_tag") != spec.battle_tag:
        raise MatchedComparisonError("matched spec source identity differs from its packet")
    if packet.get("posterior_semantic_digest") != spec.posterior_semantic_digest:
        raise MatchedComparisonError("matched spec posterior differs from its packet")
    if packet.get("mechanics_identity_digest") != spec.mechanics_identity.identity_digest:
        raise MatchedComparisonError("matched spec mechanics identity differs from its packet")
    if packet.get("showdown_commit") != spec.mechanics_identity.revision:
        raise MatchedComparisonError("matched spec mechanics revision differs from its packet")
    if packet.get("evaluator_digest") != spec.evaluator_identity_digest:
        raise MatchedComparisonError("matched spec evaluator differs from its packet")
    if packet.get("compute_budget") != spec.compute_budget.to_record():
        raise MatchedComparisonError("matched spec compute budget differs from its packet")
    if packet.get("chance_treatment") != spec.chance_treatment:
        raise MatchedComparisonError("matched spec chance treatment differs from its packet")
    if packet.get("depth") != spec.depth or packet.get("opponent_model") != spec.opponent_model:
        raise MatchedComparisonError("matched spec search treatment differs from its packet")


def settle_packet(
    *,
    packet: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _validate_packet(packet)
    raw_spec = packet.get("matched_spec")
    assert isinstance(raw_spec, Mapping)
    spec = MatchedExperimentSpec.from_record(raw_spec)
    if len(receipts) != len(METHODS):
        raise MatchedComparisonError("exactly two method receipts are required")
    parsed_receipts: list[ComputeReceipt] = []
    for raw_receipt in receipts:
        try:
            parsed_receipts.append(ComputeReceipt.from_record(raw_receipt))
        except ResearchContractError as error:
            raise MatchedComparisonError(str(error)) from error
    by_method = {receipt.method: receipt for receipt in parsed_receipts}
    if set(by_method) != set(METHODS) or len(by_method) != len(METHODS):
        raise MatchedComparisonError("receipts must contain each comparison method once")

    legal_actions = list(packet["legal_actions"])
    root_values: dict[str, dict[str, float]] = {}
    consumed: dict[str, int] = {}
    evaluator_calls: dict[str, int] = {}
    resource_accounting: dict[str, dict[str, Any]] = {}
    transition_program_digests: dict[str, str] = {}
    chosen: dict[str, str] = {}
    oracle_digest: str | None = None
    mechanics_evidence_digest: str | None = None
    transition_artifact_digest: str | None = None
    transition_program_source: str | None = None

    for method in METHODS:
        receipt = by_method[method]
        if receipt.packet_digest != packet["packet_digest"]:
            raise MatchedComparisonError(f"{method}: receipt belongs to another frozen packet")
        if receipt.matched_spec_digest != packet["matched_spec_digest"]:
            raise MatchedComparisonError(f"{method}: receipt used another matched spec")
        if receipt.input_digest != packet["input_digest"]:
            raise MatchedComparisonError(f"{method}: receipt used another frozen input")
        if receipt.posterior_digest != packet["posterior_digest"]:
            raise MatchedComparisonError(f"{method}: receipt used another posterior artifact")
        if receipt.posterior_semantic_digest != spec.posterior_semantic_digest:
            raise MatchedComparisonError(f"{method}: receipt used another posterior")
        if receipt.evaluator_digest != packet["evaluator_digest"]:
            raise MatchedComparisonError(f"{method}: receipt used another evaluator")
        checkpoint_digest = packet["evaluator"].get("checkpoint_digest")
        if receipt.evaluator_checkpoint_digest != checkpoint_digest:
            raise MatchedComparisonError(f"{method}: receipt used another evaluator checkpoint")
        if receipt.mechanics_identity_digest != spec.mechanics_identity.identity_digest:
            raise MatchedComparisonError(f"{method}: receipt used another mechanics identity")
        if receipt.showdown_commit != spec.mechanics_identity.revision:
            raise MatchedComparisonError(f"{method}: receipt used another mechanics revision")
        if receipt.chance_treatment != spec.chance_treatment:
            raise MatchedComparisonError(f"{method}: receipt used another chance treatment")
        if receipt.compute_budget != spec.compute_budget:
            raise MatchedComparisonError(f"{method}: receipt used another authorized budget")
        transition_program_digests[method] = receipt.transition_program_digest
        if receipt.consumed > spec.compute_budget.authorized:
            raise MatchedComparisonError(f"{method}: exceeded the authorized budget")
        consumed[method] = receipt.consumed
        evaluator_calls[method] = receipt.evaluator_calls
        if oracle_digest is None:
            oracle_digest = receipt.transition_oracle_digest
        elif receipt.transition_oracle_digest != oracle_digest:
            raise MatchedComparisonError(
                "matched methods report different frozen mechanics evidence"
            )
        if mechanics_evidence_digest is None:
            mechanics_evidence_digest = receipt.mechanics_evidence_digest
        elif receipt.mechanics_evidence_digest != mechanics_evidence_digest:
            raise MatchedComparisonError(
                "matched methods report different dependency/refinement evidence"
            )
        if transition_artifact_digest is None:
            transition_artifact_digest = receipt.transition_artifact_digest
        elif receipt.transition_artifact_digest != transition_artifact_digest:
            raise MatchedComparisonError("matched methods consumed different mechanics artifacts")
        if transition_program_source is None:
            transition_program_source = receipt.transition_program_source
        elif receipt.transition_program_source != transition_program_source:
            raise MatchedComparisonError("matched methods used different program sources")
        if receipt.resource_accounting is not None:
            resource_accounting[method] = receipt.resource_accounting.to_record()

        normalized = dict(receipt.root_values)
        if set(normalized) != set(legal_actions):
            raise MatchedComparisonError(
                f"{method}: root values do not cover the frozen legal actions"
            )
        root_values[method] = normalized
        best_value = max(normalized.values())
        expected_action = min(action for action, value in normalized.items() if value == best_value)
        action = receipt.chosen_action
        if action != expected_action:
            raise MatchedComparisonError(
                f"{method}: chosen action is inconsistent with its root values"
            )
        chosen[method] = str(action)

    if len(set(transition_program_digests.values())) != 1:
        raise MatchedComparisonError(
            "comparison methods consumed different transition programs"
        )
    if len(set(consumed.values())) != 1:
        raise MatchedComparisonError("matched methods report different counted compute consumption")

    det_values = root_values["determinization"]
    info_values = root_values["information_set"]
    gaps = {action: det_values[action] - info_values[action] for action in legal_actions}
    max_gap = max(gaps.values())
    max_gap_action = min(action for action, gap in gaps.items() if gap == max_gap)
    info_best = max(info_values.values())
    regret = max(0.0, info_best - info_values[chosen["determinization"]])

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "packet_digest": packet["packet_digest"],
        "input_digest": packet["input_digest"],
        "state_digest": packet["state_digest"],
        "posterior_digest": packet["posterior_digest"],
        "posterior_support": dict(packet["posterior_support"]),
        "posterior_semantic_digest": packet["posterior_semantic_digest"],
        "matched_spec_digest": packet["matched_spec_digest"],
        "mechanics_identity_digest": packet["mechanics_identity_digest"],
        "transition_oracle_digest": oracle_digest,
        "mechanics_evidence_digest": mechanics_evidence_digest,
        "transition_artifact_digest": transition_artifact_digest,
        "transition_program_source": transition_program_source,
        "chance_treatment": packet["chance_treatment"],
        "fixture_id": packet["fixture_id"],
        "battle_tag": packet["battle_tag"],
        "legal_actions": list(packet["legal_actions"]),
        "posterior_treatment": packet["posterior_treatment"],
        "showdown_commit": packet["showdown_commit"],
        "opponent_model": packet["opponent_model"],
        "depth": packet["depth"],
        "matched_input": True,
        "matched_authorized_compute": True,
        "matched_evaluator": True,
        "matched_evaluator_checkpoint": True,
        "matched_transition_program": True,
        "transition_program_digest": transition_program_digests["determinization"],
        "evaluator": dict(packet["evaluator"]),
        "evaluator_digest": packet["evaluator_digest"],
        "evaluator_checkpoint_digest": packet["evaluator"]["checkpoint_digest"],
        "budget_unit_definition": COMPUTE_BUDGET_UNIT_DEFINITION,
        "evaluator_call_unit_definition": EVALUATOR_CALL_UNIT_DEFINITION,
        "evaluator_calls": evaluator_calls,
        "compute_budget": dict(packet["compute_budget"]),
        "compute_consumed": consumed,
        "resource_accounting": resource_accounting,
        "predictors": dict(packet["predictors"]),
        "two_player_information_sets_preserved": (
            packet["opponent_model"] == "two_sided_information_sets"
        ),
        "max_determinization_value_optimism": max_gap,
        "max_optimism_action": max_gap_action,
        "information_set_regret_of_determinization_action": regret,
        "policy_disagreement": (chosen["determinization"] != chosen["information_set"]),
        "chosen_actions": chosen,
        "root_values": root_values,
    }


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze")
    freeze.add_argument("plan", type=Path)
    freeze.add_argument("state", type=Path)
    freeze.add_argument("posterior", type=Path)
    freeze.add_argument("--posterior-treatment", required=True, choices=POSTERIOR_TREATMENTS)
    freeze.add_argument("--depth", required=True, type=int)
    freeze.add_argument("--output", required=True, type=Path)

    settle = commands.add_parser("settle")
    settle.add_argument("packet", type=Path)
    settle.add_argument("receipts", nargs=2, type=Path)
    settle.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        packet = freeze_packet(
            plan=_load_object(args.plan),
            state=_load_object(args.state),
            posterior=_load_object(args.posterior),
            posterior_treatment=args.posterior_treatment,
            depth=args.depth,
        )
        _write_json(args.output, packet)
        print(json.dumps(packet, sort_keys=True))
        return 0
    if args.command == "settle":
        result = settle_packet(
            packet=_load_object(args.packet),
            receipts=[_load_object(path) for path in args.receipts],
        )
        _write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        return 0
    raise AssertionError(f"unexpected command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
