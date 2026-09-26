"""Candidate evidence for compiled partial-information search topology.

The experiment is confirmatory about semantics and descriptive about speed. It compares
the existing Python frontier search with the research-only compiled JAX path on the same
synthetic finite-support game, then measures repeated transport through one already
compiled topology. Host timing is reported but does not decide correctness.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Sequence

from azelficoast.core.compiled_search import (
    CardinalityEnvelope,
    compile_search_topology,
    materialize_compiled_frontier,
    reduce_compiled_root_values,
    search_transition_program_adaptive,
    transport_posterior_mass,
)
from azelficoast.core.search import search_transition_program

WORLD_COUNT = 512
ACTION_COUNT = 4
CLASSES_PER_ACTION = 64
OBSERVATIONS_PER_ACTION = 8
TRANSPORT_REPEATS = 20


class SyntheticEvaluator:
    """Deterministic public-belief value with no private realized-world access."""

    def values(self, leaves: Sequence[Any]) -> tuple[float, ...]:
        values: list[float] = []
        for leaf in leaves:
            expected_hidden = math.fsum(
                float(world["weight"]) * float(world["hidden"]["score"])
                for world in leaf.posterior
            )
            values.append(float(leaf.public_state["bias"]) + 0.25 * expected_hidden)
        return tuple(values)


def _problem() -> tuple[dict[str, Any], dict[str, Any]]:
    world_ids = [f"w{index:04d}" for index in range(WORLD_COUNT)]
    raw_weights = [1 + ((index * 17) % 23) for index in range(WORLD_COUNT)]
    total = float(sum(raw_weights))
    posterior = {
        "worlds": [
            {
                "world_id": world_id,
                "weight": raw_weights[index] / total,
                "hidden": {
                    "score": (((index * 37) % 101) - 50) / 50.0,
                },
            }
            for index, world_id in enumerate(world_ids)
        ]
    }

    programs = []
    for action_index in range(ACTION_COUNT):
        classes = []
        for class_index in range(CLASSES_PER_ACTION):
            members = [
                world_id
                for index, world_id in enumerate(world_ids)
                if index % CLASSES_PER_ACTION == class_index
            ]
            observation_group = class_index % OBSERVATIONS_PER_ACTION
            classes.append(
                {
                    "member_world_ids": members,
                    "outcomes": [
                        {
                            "probability": 0.35,
                            "observation": {
                                "group": observation_group,
                                "branch": 0,
                            },
                            "successor": {
                                "turn": 2,
                                "action_index": action_index,
                                "group": observation_group,
                                "branch": 0,
                                "bias": (
                                    action_index * 0.03
                                    + observation_group * 0.002
                                    - 0.01
                                ),
                            },
                            "legal_actions": ["continue:a", "continue:b"],
                        },
                        {
                            "probability": 0.65,
                            "observation": {
                                "group": observation_group,
                                "branch": 1,
                            },
                            "successor": {
                                "turn": 2,
                                "action_index": action_index,
                                "group": observation_group,
                                "branch": 1,
                                "bias": (
                                    action_index * 0.03
                                    + observation_group * 0.002
                                    + 0.01
                                ),
                            },
                            "legal_actions": ["continue:a"],
                        },
                    ],
                }
            )
        programs.append(
            {
                "action": f"action:{action_index}",
                "classes_out": len(classes),
                "classes": classes,
            }
        )

    return (
        {
            "schema": "azelficoast.synthetic-transition-program-set",
            "schema_version": 1,
            "world_ids": world_ids,
            "legal_actions": [f"action:{index}" for index in range(ACTION_COUNT)],
            "programs": programs,
        },
        posterior,
    )


def _milliseconds(start_ns: int, end_ns: int) -> float:
    return (end_ns - start_ns) / 1_000_000.0


def main() -> int:
    program, posterior = _problem()
    evaluator = SyntheticEvaluator()

    start = time.perf_counter_ns()
    reference = search_transition_program(
        program_set=program,
        posterior=posterior,
        method="information_set",
        evaluator=evaluator,
        expected_program_schema="azelficoast.synthetic-transition-program-set",
        expected_program_schema_version=1,
    )
    reference_end = time.perf_counter_ns()

    compile_start = time.perf_counter_ns()
    topology = compile_search_topology(
        program_set=program,
        posterior=posterior,
        method="information_set",
        expected_program_schema="azelficoast.synthetic-transition-program-set",
        expected_program_schema_version=1,
    )
    compile_end = time.perf_counter_ns()

    first_start = time.perf_counter_ns()
    compiled = search_transition_program_adaptive(
        program_set=program,
        posterior=posterior,
        method="information_set",
        evaluator=evaluator,
        envelope=CardinalityEnvelope(
            max_world_count=WORLD_COUNT,
            max_class_count=ACTION_COUNT * CLASSES_PER_ACTION,
            max_chance_edge_count=ACTION_COUNT * WORLD_COUNT * 2,
            max_leaf_count=ACTION_COUNT * OBSERVATIONS_PER_ACTION * 2,
            max_dense_leaf_world_cells=(
                ACTION_COUNT * OBSERVATIONS_PER_ACTION * 2 * WORLD_COUNT
            ),
        ),
        expected_program_schema="azelficoast.synthetic-transition-program-set",
        expected_program_schema_version=1,
    )
    first_end = time.perf_counter_ns()

    # Warm the topology-specific transport/reduction kernels once, then time repeated
    # posterior routing without recompiling semantic topology.
    frontier, transported = materialize_compiled_frontier(topology, posterior)
    leaf_values = evaluator.values(frontier.leaves)
    reduce_compiled_root_values(topology, transported, leaf_values)

    transport_start = time.perf_counter_ns()
    last_transport = None
    for _ in range(TRANSPORT_REPEATS):
        last_transport = transport_posterior_mass(topology, posterior)
    transport_end = time.perf_counter_ns()
    assert last_transport is not None

    reduction_start = time.perf_counter_ns()
    last_roots = None
    for _ in range(TRANSPORT_REPEATS):
        last_roots = reduce_compiled_root_values(
            topology,
            last_transport,
            leaf_values,
        )
    reduction_end = time.perf_counter_ns()
    assert last_roots is not None

    differences = {
        action: abs(
            float(reference["root_values"][action])
            - float(compiled["root_values"][action])
        )
        for action in reference["root_values"]
    }
    maximum_difference = max(differences.values())
    action_mass = math.fsum(float(value) for value in last_transport.leaf_mass)
    expected_action_mass = float(ACTION_COUNT)

    transport_seconds = max(
        (transport_end - transport_start) / 1_000_000_000.0,
        1e-12,
    )
    reduction_seconds = max(
        (reduction_end - reduction_start) / 1_000_000_000.0,
        1e-12,
    )
    warm_reduction_difference = max(
        abs(float(last_roots[action]) - float(compiled["root_values"][action]))
        for action in last_roots
    )
    passed = (
        reference["chosen_action"] == compiled["chosen_action"]
        and maximum_difference <= 1e-5
        and abs(action_mass - expected_action_mass) <= 1e-5
        and warm_reduction_difference <= 1e-5
        and compiled["physical_search_path"] == "compiled-jax"
        and compiled["cardinality_plan"]["replanned"] is False
    )

    result = {
        "schema": "azelficoast.compiled-search-topology-experiment",
        "schema_version": 1,
        "passed": passed,
        "problem": {
            "world_count": WORLD_COUNT,
            "action_count": ACTION_COUNT,
            "classes_per_action": CLASSES_PER_ACTION,
            "observations_per_action": OBSERVATIONS_PER_ACTION,
            "chance_outcomes_per_class": 2,
        },
        "compiled_shape": {
            "class_count": topology.class_count,
            "edge_count": topology.edge_count,
            "leaf_count": topology.leaf_count,
            "observation_count": len(topology.observation_keys),
            "successor_state_count": len(topology.successor_states),
            "successor_action_count": len(topology.successor_action_vocabulary),
        },
        "cardinality_plan": compiled["cardinality_plan"],
        "correctness": {
            "reference_action": reference["chosen_action"],
            "compiled_action": compiled["chosen_action"],
            "maximum_root_value_abs_difference": maximum_difference,
            "root_value_differences": differences,
            "transported_action_mass": action_mass,
            "expected_action_mass": expected_action_mass,
            "program_digest_matches": (
                reference["transition_program_digest"]
                == compiled["transition_program_digest"]
            ),
            "transition_evaluations_match": (
                reference["transition_evaluations"]
                == compiled["transition_evaluations"]
            ),
            "warm_reduction_max_abs_difference": warm_reduction_difference,
        },
        "timing": {
            "reference_search_ms": _milliseconds(start, reference_end),
            "topology_compile_ms": _milliseconds(compile_start, compile_end),
            "first_compiled_search_ms": _milliseconds(first_start, first_end),
            "warm_transport_repeats": TRANSPORT_REPEATS,
            "warm_transport_total_ms": _milliseconds(
                transport_start,
                transport_end,
            ),
            "warm_transport_mean_ms": _milliseconds(
                transport_start,
                transport_end,
            )
            / TRANSPORT_REPEATS,
            "warm_chance_edges_per_second": (
                topology.edge_count * TRANSPORT_REPEATS / transport_seconds
            ),
            "warm_reduction_total_ms": _milliseconds(
                reduction_start,
                reduction_end,
            ),
            "warm_reduction_mean_ms": _milliseconds(
                reduction_start,
                reduction_end,
            )
            / TRANSPORT_REPEATS,
            "warm_leaf_contributions_per_second": (
                topology.leaf_count * TRANSPORT_REPEATS / reduction_seconds
            ),
        },
        "claim": (
            "Compiled JAX mass transport and root reduction reproduce the existing "
            "Python information-set search on this frozen synthetic finite-support "
            "problem. Timing is descriptive and host-specific."
        ),
        "non_claim": (
            "This does not establish live latency, battle strength, unseen-program "
            "generality, or permission to replace the live search path."
        ),
    }
    print(json.dumps(result, sort_keys=True))
    return 0 if passed else 1



if __name__ == "__main__":
    raise SystemExit(main())
