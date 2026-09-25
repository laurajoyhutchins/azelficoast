from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest

from azelficoast.core.decision_relevance import (
    DecisionRelevanceError,
    decision_relevance_quotient,
)
from azelficoast.core.program import program_for_action
from azelficoast.core.transition import ORACLE_SCHEMA, canonical_json, sha256_json
from azelficoast.core.whole_turn_program import (
    WholeTurnProgramError,
    compile_whole_turn_programs,
    execute_whole_turn_program,
    verify_whole_turn_program_set,
)
from hostile.fixtures import (
    PROBABILITY_CONSERVATION_ABS_TOLERANCE,
    hostile_case,
)
from hostile.transforms import permute_oracle_support


PINNED_SHOWDOWN_COMMIT = "a5df8274e85b0889bf2a9b3422a08b39732374fc"


def _small_oracle() -> dict[str, Any]:
    worlds = [
        {
            "world_id": f"{semantic}-{replica}",
            "weight": 0.25,
            "hidden": {"relevant": semantic, "unread_noise": replica},
        }
        for semantic in ("same", "different")
        for replica in (1, 2)
    ]
    transitions = []
    for world in worlds:
        value = world["hidden"]["relevant"]
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "hold",
                "outcomes": [
                    {
                        "probability": 0.25,
                        "observation": {"kind": "public", "coin": "heads"},
                        "successor": {"state": value, "noise": "ignored"},
                        "continuations": {"future": 1.0},
                        "transition_reads": ["relevant"],
                    },
                    {
                        "probability": 0.75,
                        "observation": {"kind": "public", "coin": "tails"},
                        "successor": {"state": value, "noise": "ignored"},
                        "continuations": {"future": 1.0},
                        "transition_reads": ["relevant"],
                    },
                ],
            }
        )
    return {
        "schema": ORACLE_SCHEMA,
        "schema_version": 1,
        "source_fixture_id": "hostile-transition-program",
        "showdown_commit": "synthetic-pinned-test",
        "worlds": worlds,
        "legal_actions": ["hold"],
        "dependency_candidates": ["relevant", "unread_noise"],
        "transitions": transitions,
    }


def _multi_action_oracle() -> dict[str, Any]:
    oracle = _small_oracle()
    oracle["legal_actions"] = ["hold", "switch"]
    oracle["transitions"] = [
        {
            **copy.deepcopy(row),
            "action": action,
        }
        for row in oracle["transitions"]
        for action in oracle["legal_actions"]
    ]
    return oracle


def _immediate_surface(transition: dict[str, Any]) -> list[dict[str, Any]]:
    surface = [
        {
            "probability": float(outcome["probability"]),
            "observation": copy.deepcopy(outcome.get("observation")),
            "successor": copy.deepcopy(outcome.get("successor")),
        }
        for outcome in transition["outcomes"]
    ]
    return sorted(surface, key=canonical_json)


def _showdown_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    oracle_path = os.environ.get("AZELFICOAST_HOSTILE_ORACLE_PATH")
    lazy_path = os.environ.get("AZELFICOAST_HOSTILE_LAZY_PROGRAM_PATH")
    if not oracle_path or not lazy_path:
        pytest.skip("pinned exact-head workflow supplies direct and lazy oracle paths")
    oracle = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    lazy = json.loads(Path(lazy_path).read_text(encoding="utf-8"))
    if not isinstance(lazy, dict):
        pytest.fail("pinned lazy transition program set must be an object")
    assert oracle.get("showdown_commit") == PINNED_SHOWDOWN_COMMIT
    return oracle, lazy


@hostile_case(
    mutation="clear the recorded runtime reads for a field that separates direct transition semantics",
    expected="dynamic-read compilation rejects the stale trace instead of merging the distinct worlds",
    threat="a read trace from a different context is accepted as timeless proof of semantic equivalence",
    layer="whole-turn dependency evidence validation",
)
def test_stale_dynamic_read_trace_cannot_merge_distinct_worlds() -> None:
    oracle = _small_oracle()
    stale = copy.deepcopy(oracle)
    for transition in stale["transitions"]:
        for outcome in transition["outcomes"]:
            outcome["transition_reads"] = []

    with pytest.raises(WholeTurnProgramError, match="reads are incomplete"):
        compile_whole_turn_programs(stale, partition_strategy="dynamic_reads")


@hostile_case(
    mutation="swap a class representative and reweight posterior atoms without changing their semantic support",
    expected="the independent verifier accepts the alternate equivalent representative and reweighted outcome mass remains one",
    threat="compiled transitions depend on an arbitrary representative or refinement loses posterior mass",
    layer="whole-turn class execution and posterior reweighting",
)
def test_representative_swap_and_posterior_reweight_preserve_distribution() -> None:
    oracle = _small_oracle()
    program_set = compile_whole_turn_programs(oracle, partition_strategy="dynamic_reads")
    program = program_for_action(program_set, "hold")
    class_row = next(row for row in program["classes"] if len(row["member_world_ids"]) > 1)
    swapped = copy.deepcopy(program_set)
    swapped_program = program_for_action(swapped, "hold")
    swapped_class = next(
        row for row in swapped_program["classes"] if row["class_id"] == class_row["class_id"]
    )
    current = swapped_class["representative_world_id"]
    alternate = next(
        world_id for world_id in swapped_class["member_world_ids"] if world_id != current
    )
    swapped_class["representative_world_id"] = alternate
    certificate = verify_whole_turn_program_set(swapped, oracle)
    assert certificate["verified_class_count"] == 2

    posterior = {world["world_id"]: world["weight"] for world in oracle["worlds"]}
    execution = execute_whole_turn_program(program_set, action="hold", posterior=posterior)
    assert math.fsum(edge["mass"] for edge in execution["edges"]) == pytest.approx(
        1.0,
        abs=PROBABILITY_CONSERVATION_ABS_TOLERANCE,
        rel=0,
    )


@hostile_case(
    mutation="add a valid hidden-field read to dependency evidence for every direct transition",
    expected="refinement may split equivalent classes, never merges distinct outcomes, and preserves total posterior mass",
    threat="new dependency evidence is used to collapse worlds despite their independently verified transition differences",
    layer="dependency-based class refinement",
)
def test_stronger_read_evidence_only_refines_classes_and_preserves_mass() -> None:
    oracle = _small_oracle()
    baseline = compile_whole_turn_programs(oracle, partition_strategy="dynamic_reads")
    refined_oracle = copy.deepcopy(oracle)
    for transition in refined_oracle["transitions"]:
        for outcome in transition["outcomes"]:
            outcome["transition_reads"] = ["relevant", "unread_noise"]
    refined = compile_whole_turn_programs(
        refined_oracle,
        partition_strategy="dynamic_reads",
    )

    baseline_program = program_for_action(baseline, "hold")
    refined_program = program_for_action(refined, "hold")
    assert baseline_program["classes_out"] == 2
    assert refined_program["classes_out"] == 4
    assert all(len(row["member_world_ids"]) == 1 for row in refined_program["classes"])

    posterior = {world["world_id"]: world["weight"] for world in oracle["worlds"]}
    baseline_result = execute_whole_turn_program(
        baseline,
        action="hold",
        posterior=posterior,
    )
    refined_result = execute_whole_turn_program(
        refined,
        action="hold",
        posterior=posterior,
    )

    def semantic_mass(result):
        totals: dict[str, list[float]] = {}
        for edge in result["edges"]:
            key = canonical_json((edge["observation"], edge["successor"]))
            totals.setdefault(key, []).append(edge["mass"])
        return {key: math.fsum(values) for key, values in totals.items()}

    assert semantic_mass(refined_result) == semantic_mass(baseline_result)
    assert math.fsum(edge["mass"] for edge in refined_result["edges"]) == pytest.approx(
        1.0,
        abs=PROBABILITY_CONSERVATION_ABS_TOLERANCE,
        rel=0,
    )


@hostile_case(
    mutation="reverse or fixed-shuffle world support and its direct transition rows",
    expected="the compiled whole-turn program and each weighted action distribution are exactly unchanged",
    threat="support iteration order changes transition classes, representative selection, or posterior mass",
    layer="whole-turn transition compilation",
)
@pytest.mark.parametrize("order", [(3, 2, 1, 0), (2, 0, 3, 1)])
def test_transition_program_is_invariant_to_support_order(order) -> None:
    oracle = _small_oracle()
    baseline_program = compile_whole_turn_programs(oracle)
    reordered_oracle = permute_oracle_support(oracle, order)
    reordered_program = compile_whole_turn_programs(reordered_oracle)
    assert reordered_program == baseline_program

    posterior = {world["world_id"]: world["weight"] for world in oracle["worlds"]}
    for action in oracle["legal_actions"]:
        baseline = execute_whole_turn_program(baseline_program, action=action, posterior=posterior)
        reordered = execute_whole_turn_program(
            reordered_program, action=action, posterior=posterior
        )
        assert reordered == baseline


@hostile_case(
    mutation="reverse legal-action order and reorder direct transition records",
    expected="each action retains its own program semantics regardless of input position",
    threat="values or transition programs are associated with the wrong action by position",
    layer="whole-turn transition compilation",
)
def test_transition_program_preserves_action_associations_under_action_order_change() -> None:
    oracle = _multi_action_oracle()
    baseline = compile_whole_turn_programs(oracle)
    reordered = copy.deepcopy(oracle)
    reordered["legal_actions"].reverse()
    reordered["transitions"].reverse()
    actual = compile_whole_turn_programs(reordered)
    for action in oracle["legal_actions"]:
        assert program_for_action(actual, action) == program_for_action(baseline, action)


@hostile_case(
    mutation="inject NaN or infinity in hidden state, immediate observations, successors, or continuation utilities",
    expected="all oracle consumers reject the artifact explicitly before hashing or inference",
    threat="non-finite semantic evidence can be certified, pooled, or published as a plausible result",
    layer="transition-oracle structural and utility validation",
)
@pytest.mark.parametrize(
    "path,value",
    [
        (("worlds", 0, "hidden", "poison"), math.nan),
        (("transitions", 0, "outcomes", 0, "observation", "poison"), math.inf),
        (("transitions", 0, "outcomes", 0, "successor", "poison"), -math.inf),
        (("transitions", 0, "outcomes", 0, "continuations", "next"), math.inf),
    ],
)
def test_nonfinite_nested_oracle_evidence_fails_closed(path, value) -> None:
    oracle = _small_oracle()
    cursor = oracle
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value

    with pytest.raises(WholeTurnProgramError, match="finite|JSON"):
        compile_whole_turn_programs(oracle)
    with pytest.raises(DecisionRelevanceError, match="finite|JSON"):
        decision_relevance_quotient(oracle)


@hostile_case(
    mutation="place NaN or infinity in terminal and continuation utilities used by the public-belief trace analyzer",
    expected="decision analysis and quotient construction reject the utility before value aggregation",
    threat="a non-finite utility can poison a published expected value or fail later with an unrelated error",
    layer="continuation utility validation",
)
@pytest.mark.parametrize("utility", [math.nan, math.inf, -math.inf])
def test_nonfinite_continuation_utility_fails_closed(utility: float) -> None:
    oracle = _small_oracle()
    oracle["transitions"][0]["outcomes"][0]["continuations"]["next"] = utility

    with pytest.raises(DecisionRelevanceError, match="finite"):
        decision_relevance_quotient(oracle)


@hostile_case(
    mutation="inject NaN, infinity, zero, or negative probability into one direct transition outcome",
    expected="both structural oracle validation and bounded decision quotient reject malformed probability mass explicitly",
    threat="non-finite or invalid chance mass is silently accepted as a plausible mechanics result",
    layer="transition-oracle validation",
)
@pytest.mark.parametrize("probability", [math.nan, math.inf, 0.0, -0.25])
def test_transition_probability_validation_fails_closed(probability: float) -> None:
    oracle = _small_oracle()
    oracle["transitions"][0]["outcomes"][0]["probability"] = probability

    with pytest.raises(WholeTurnProgramError, match="probabilit|finite|JSON"):
        compile_whole_turn_programs(oracle)
    with pytest.raises(DecisionRelevanceError, match="probabilit|finite|JSON"):
        decision_relevance_quotient(oracle)


@hostile_case(
    mutation="replace one hidden-world prior with NaN, infinity, zero, or negative mass",
    expected="transition-oracle consumers reject the posterior instead of publishing a zero or non-finite class weight",
    threat="malformed support mass is ignored by transition compression or decision-relevance analysis",
    layer="shared transition-oracle support validation",
)
@pytest.mark.parametrize("weight", [math.nan, math.inf, 0.0, -0.25])
def test_transition_world_weight_validation_fails_closed(weight: float) -> None:
    oracle = _small_oracle()
    oracle["worlds"][0]["weight"] = weight

    with pytest.raises(WholeTurnProgramError, match="weight|finite|JSON"):
        compile_whole_turn_programs(oracle)
    with pytest.raises(DecisionRelevanceError, match="weight|finite|JSON"):
        decision_relevance_quotient(oracle)


@hostile_case(
    mutation="compare every lazily proposed class with the independent exhaustive pinned-Showdown world/action oracle",
    expected="each class member has exactly the representative's full chance distribution, public observation, and successor state",
    threat="dependency-based compression publishes a transition that only matched one representative by coincidence",
    layer="pinned Showdown transition certification",
    tier="showdown",
)
def test_pinned_lazy_program_matches_exhaustive_direct_oracle() -> None:
    oracle, lazy = _showdown_artifacts()
    mechanics = oracle.get("mechanics")
    producer = lazy.get("producer")
    assert isinstance(mechanics, dict)
    assert isinstance(producer, dict)
    assert producer.get("root_chance_samples") == mechanics.get("root_chance_samples")
    assert sorted(lazy.get("legal_actions", [])) == sorted(oracle.get("legal_actions", []))
    verification = verify_whole_turn_program_set(lazy, oracle)
    assert verification["showdown_commit"] == PINNED_SHOWDOWN_COMMIT
    assert verification["verified_class_count"] >= 1
    assert (
        verification["representative_world_executions"]
        <= verification["exhaustive_world_action_product"]
    )

    for transition in oracle["transitions"]:
        assert math.fsum(
            float(outcome["probability"]) for outcome in transition["outcomes"]
        ) == pytest.approx(
            1.0,
            abs=PROBABILITY_CONSERVATION_ABS_TOLERANCE,
            rel=0,
        )

    posterior = {world["world_id"]: float(world["weight"]) for world in oracle["worlds"]}
    for action in oracle["legal_actions"]:
        result = execute_whole_turn_program(lazy, action=action, posterior=posterior)
        assert math.fsum(edge["mass"] for edge in result["edges"]) == pytest.approx(
            1.0,
            abs=PROBABILITY_CONSERVATION_ABS_TOLERANCE,
            rel=0,
        )


@hostile_case(
    mutation="change one hidden field between directly executed worlds while preserving all other hidden fields",
    expected="unread fields have identical complete immediate outcomes, and every outcome-changing field appears in runtime-read evidence",
    threat="lazy compression drops a field that changes a pinned transition or preserves an unnecessary hidden distinction as if it were evidence",
    layer="pinned Showdown dependency evidence and direct oracle",
    tier="showdown",
)
def test_pinned_direct_oracle_separates_read_and_unread_fields() -> None:
    oracle, lazy = _showdown_artifacts()
    witnessed_fields = set()
    witnessed_unread_fields = set()
    witnessed_relevant = False
    worlds = oracle["worlds"]
    transitions = {
        (transition["world_id"], transition["action"]): transition
        for transition in oracle["transitions"]
    }

    for left_index, left in enumerate(worlds):
        for right in worlds[left_index + 1 :]:
            left_hidden = left["hidden"]
            right_hidden = right["hidden"]
            if set(left_hidden) != set(right_hidden):
                continue
            differences = [
                field for field in left_hidden if left_hidden[field] != right_hidden[field]
            ]
            if len(differences) != 1:
                continue
            field = differences[0]
            witnessed_fields.add(field)
            for action in oracle["legal_actions"]:
                left_transition = transitions[(left["world_id"], action)]
                right_transition = transitions[(right["world_id"], action)]
                left_surface = _immediate_surface(left_transition)
                right_surface = _immediate_surface(right_transition)
                left_reads = {
                    value
                    for outcome in left_transition["outcomes"]
                    for value in outcome.get("transition_reads", [])
                }
                right_reads = {
                    value
                    for outcome in right_transition["outcomes"]
                    for value in outcome.get("transition_reads", [])
                }
                changed = left_surface != right_surface
                if changed:
                    assert field in left_reads | right_reads, (
                        field,
                        left["world_id"],
                        right["world_id"],
                        action,
                    )
                    witnessed_relevant = True
                elif field not in left_reads | right_reads:
                    witnessed_unread_fields.add(field)

    assert witnessed_fields, "fixture has no one-field hidden-world pair"
    assert witnessed_unread_fields, "fixture has no directly verified unread-field pair"
    assert witnessed_relevant, "fixture has no directly verified read-field separation"


@hostile_case(
    mutation="remove a transition read field from an older trace while direct outcomes still distinguish hidden worlds",
    expected="the pinned oracle detects stale dependency evidence and rejects dynamic-read compression",
    threat="a stale runtime trace is reused as a timeless proof after the semantic context changes",
    layer="pinned Showdown read-trace validation",
    tier="showdown",
)
def test_pinned_stale_read_trace_cannot_certify_a_distinct_transition() -> None:
    oracle, _ = _showdown_artifacts()
    transitions = oracle["transitions"]
    actions = oracle["legal_actions"]
    selected_action = None
    for action in actions:
        surfaces = {
            canonical_json(
                _immediate_surface(
                    next(
                        transition
                        for transition in transitions
                        if transition["world_id"] == world["world_id"]
                        and transition["action"] == action
                    )
                )
            )
            for world in oracle["worlds"]
        }
        if len(surfaces) > 1:
            selected_action = action
            break
    assert selected_action is not None, "pinned fixture has no direct transition separation"

    stale = copy.deepcopy(oracle)
    for transition in stale["transitions"]:
        if transition["action"] != selected_action:
            continue
        for outcome in transition["outcomes"]:
            outcome["transition_reads"] = []

    with pytest.raises(WholeTurnProgramError, match="reads are incomplete"):
        compile_whole_turn_programs(stale, partition_strategy="dynamic_reads")


@hostile_case(
    mutation="swap each available pinned class representative and compare semantic program identity after a JSON round trip",
    expected="equivalent members remain interchangeable, while serialized and parsed artifacts retain the same semantic hash",
    threat="representative selection or incidental JSON formatting changes a certified mechanics result",
    layer="pinned transition artifact identity",
    tier="showdown",
)
def test_pinned_representative_swap_and_artifact_round_trip() -> None:
    oracle, lazy = _showdown_artifacts()
    swapped = copy.deepcopy(lazy)
    swapped_count = 0
    for program in swapped["programs"]:
        for row in program["classes"]:
            members = row["member_world_ids"]
            if len(members) < 2:
                continue
            current = row["representative_world_id"]
            row["representative_world_id"] = next(member for member in members if member != current)
            swapped_count += 1
    assert swapped_count > 0, "pinned support does not exercise a mergeable class"
    verify_whole_turn_program_set(swapped, oracle)

    round_tripped = json.loads(json.dumps(lazy, indent=5, ensure_ascii=False))
    assert sha256_json(round_tripped) == sha256_json(lazy)
    assert verify_whole_turn_program_set(round_tripped, oracle)["program_digest"] == sha256_json(
        lazy
    )
