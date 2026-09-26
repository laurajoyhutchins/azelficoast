from __future__ import annotations

import copy
from pathlib import Path

import pytest

from azelficoast.research.verification.real_belief_trace import BeliefTraceError, analyze_oracle


def _oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": "scarf-a",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Scarf", "noise": 1},
        },
        {
            "world_id": "scarf-b",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Scarf", "noise": 2},
        },
        {
            "world_id": "specs-a",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Specs", "noise": 1},
        },
        {
            "world_id": "specs-b",
            "weight": 0.25,
            "hidden": {"opponent.active.item": "Specs", "noise": 2},
        },
    ]
    transitions = []
    for world in worlds:
        item = world["hidden"]["opponent.active.item"]
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "wait",
                "outcomes": [
                    {
                        "probability": 0.5,
                        "observation": {"kind": "same", "roll": "low"},
                        "successor": {"hp": 10},
                        "continuations": {
                            "fast": 4 if item == "Specs" else -4,
                            "safe": 1,
                        },
                    },
                    {
                        "probability": 0.5,
                        "observation": {"kind": "same", "roll": "high"},
                        "successor": {"hp": 9},
                        "continuations": {
                            "fast": 4 if item == "Specs" else -4,
                            "safe": 1,
                        },
                    },
                ],
            }
        )
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "reveal",
                "outcomes": [
                    {
                        "probability": 1.0,
                        "observation": {"kind": item},
                        "successor": {"hp": 9},
                        "continuations": {
                            "fast": 3 if item == "Specs" else -3,
                            "safe": 0,
                        },
                    }
                ],
            }
        )
    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "real",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["wait", "reveal"],
        "dependency_candidates": ["opponent.active.item"],
        # wait's immediate transition is item-independent even though its future
        # continuation value is not; reveal's observation depends on the item.
        "declared_reads": {"wait": [], "reveal": ["opponent.active.item"]},
        "transitions": transitions,
    }


def test_information_set_policy_cannot_branch_on_hidden_world_or_future_chance() -> None:
    result = analyze_oracle(_oracle())

    assert result["determinization"]["chosen_action"] == "wait"
    assert result["public_belief"]["chosen_action"] == "reveal"
    assert result["policy_disagreement"] is True
    assert result["hypothesis_supported"] is True
    assert result["experiment_valid"] is True
    assert result["strategy_fusion_observation_count"] > 0

    wait = next(row for row in result["actions"] if row["action"] == "wait")
    assert wait["dependency_signature"]["empirically_required_reads"] == []
    assert wait["dependency_signature"]["classes_out"] == 1
    assert wait["dependency_signature"]["observable_classes_out"] == 2

    # Each chance observation still retains both hidden item worlds. The policy
    # may react to the observed roll, but not to which hidden item generated it.
    assert all(
        {member["world_id"].split("-")[0] for member in belief["members"]}
        == {"scarf", "specs"}
        for belief in wait["successor_beliefs"]
    )


def test_missing_declared_dependency_fails_closed() -> None:
    document = _oracle()
    declared = document["declared_reads"]
    assert isinstance(declared, dict)
    declared["reveal"] = []

    with pytest.raises(BeliefTraceError, match="missing from declaration"):
        analyze_oracle(document)


def test_unread_hidden_field_perturbation_preserves_transition_refinement() -> None:
    original = _oracle()
    perturbed = copy.deepcopy(original)
    worlds = perturbed["worlds"]
    assert isinstance(worlds, list)
    for index, world in enumerate(worlds):
        hidden = world["hidden"]
        hidden["noise"] = 100 + index

    baseline = analyze_oracle(original)
    changed = analyze_oracle(perturbed)
    assert changed["determinization"] == baseline["determinization"]
    assert changed["public_belief"] == baseline["public_belief"]
    baseline_signatures = {
        row["action"]: row["dependency_signature"] for row in baseline["actions"]
    }
    changed_signatures = {
        row["action"]: row["dependency_signature"] for row in changed["actions"]
    }
    assert changed_signatures == baseline_signatures


def test_incomplete_transition_matrix_fails_closed() -> None:
    document = copy.deepcopy(_oracle())
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    transitions.pop()

    with pytest.raises(BeliefTraceError, match="omitted root transitions"):
        analyze_oracle(document)

def test_determinization_cannot_condition_on_unobserved_chance() -> None:
    document = {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "chance-alias",
        "showdown_commit": "pinned",
        "worlds": [
            {
                "world_id": "only-world",
                "weight": 1.0,
                "hidden": {},
            }
        ],
        "legal_actions": ["hold"],
        "dependency_candidates": [],
        "declared_reads": {"hold": []},
        "transitions": [
            {
                "world_id": "only-world",
                "action": "hold",
                "outcomes": [
                    {
                        "probability": 0.5,
                        "observation": {"same": True},
                        "successor": {"public": "same"},
                        "continuations": {"a": 4, "b": 0},
                    },
                    {
                        "probability": 0.5,
                        "observation": {"same": True},
                        "successor": {"public": "same"},
                        "continuations": {"a": 0, "b": 4},
                    },
                ],
            }
        ],
    }

    result = analyze_oracle(document)

    assert result["schema_version"] == 3
    assert result["determinization"]["value"] == 2
    assert result["public_belief"]["value"] == 2
    assert result["policy_disagreement"] is False
    assert result["strategy_fusion_observation_count"] == 0

    [action] = result["actions"]
    assert action["determinization_continuations"] == [
        {
            "world_id": "only-world",
            "outcome_indices": [0, 1],
            "observation_hash": action["determinization_continuations"][0][
                "observation_hash"
            ],
            "choice": "a",
            "value": 2.0,
        }
    ]
    [public] = action["public_belief_continuations"]
    assert public["world_aware_choices"] == ["a"]
    assert public["strategy_fusion_possible"] is False



def _add_hidden_read_witnesses(
    document: dict[str, object],
    *reads: str,
) -> None:
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        outcomes = transition["outcomes"]
        assert isinstance(outcomes, list)
        for outcome in outcomes:
            assert isinstance(outcome, dict)
            outcome["hidden_reads"] = list(reads)


def test_unread_factored_hidden_field_stays_out_of_cartesian_worlds() -> None:
    document = _oracle()
    _add_hidden_read_witnesses(document)
    document["factored_hidden"] = {
        "opponent.bench.species": {
            "distribution": [
                {"value": "alpha", "weight": 0.6},
                {"value": "beta", "weight": 0.4},
            ],
            "unread_actions": ["wait", "reveal"],
            "evidence": {
                "kind": "bounded-read-audit",
                "audited_support_count": 2,
            },
        }
    }

    result = analyze_oracle(document)

    assert result["world_count"] == 4
    assert result["materialized_world_count"] == 4
    assert result["latent_world_count"] == 8
    assert result["factoring_ratio"] == 2
    assert result["factored_hidden"]["opponent.bench.species"]["support_count"] == 2
    assert all(
        row["dependency_signature"]["marginalized_hidden_factors"]
        == ["opponent.bench.species"]
        for row in result["actions"]
    )


def test_factored_hidden_field_fails_closed_when_any_action_reads_it() -> None:
    document = _oracle()
    _add_hidden_read_witnesses(document)
    document["factored_hidden"] = {
        "opponent.bench.species": {
            "distribution": [
                {"value": "alpha", "weight": 0.6},
                {"value": "beta", "weight": 0.4},
            ],
            "unread_actions": ["wait"],
            "evidence": {"kind": "bounded-read-audit"},
        }
    }

    with pytest.raises(BeliefTraceError, match="materialize that factor"):
        analyze_oracle(document)


def test_factored_hidden_distribution_must_be_normalized() -> None:
    document = _oracle()
    _add_hidden_read_witnesses(document)
    document["factored_hidden"] = {
        "opponent.bench.species": {
            "distribution": [
                {"value": "alpha", "weight": 0.7},
                {"value": "beta", "weight": 0.4},
            ],
            "unread_actions": ["wait", "reveal"],
        }
    }

    with pytest.raises(BeliefTraceError, match="weights sum to"):
        analyze_oracle(document)


def test_factored_hidden_field_requires_per_outcome_read_witnesses() -> None:
    document = _oracle()
    document["factored_hidden"] = {
        "opponent.bench.species": {
            "distribution": [
                {"value": "alpha", "weight": 0.6},
                {"value": "beta", "weight": 0.4},
            ],
            "unread_actions": ["wait", "reveal"],
        }
    }

    with pytest.raises(BeliefTraceError, match="hidden_reads witness"):
        analyze_oracle(document)


def test_factored_hidden_field_rejects_contradictory_read_witness() -> None:
    document = _oracle()
    _add_hidden_read_witnesses(document)
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    first = transitions[0]
    assert isinstance(first, dict)
    outcomes = first["outcomes"]
    assert isinstance(outcomes, list)
    first_outcome = outcomes[0]
    assert isinstance(first_outcome, dict)
    first_outcome["hidden_reads"] = ["opponent.bench.species"]

    document["factored_hidden"] = {
        "opponent.bench.species": {
            "distribution": [
                {"value": "alpha", "weight": 0.6},
                {"value": "beta", "weight": 0.4},
            ],
            "unread_actions": ["wait", "reveal"],
        }
    }

    with pytest.raises(BeliefTraceError, match="was read"):
        analyze_oracle(document)


def _deep_oracle() -> dict[str, object]:
    worlds = [
        {
            "world_id": "red",
            "weight": 0.5,
            "hidden": {"opponent.active.item": "Red"},
        },
        {
            "world_id": "blue",
            "weight": 0.5,
            "hidden": {"opponent.active.item": "Blue"},
        },
    ]
    transitions: list[dict[str, object]] = []
    for world in worlds:
        world_id = str(world["world_id"])
        for action in ("safe", "trap"):
            if action == "safe":
                leaf = {"hold": 2.0}
            elif world_id == "red":
                leaf = {"red": 3.0, "blue": 0.0}
            else:
                leaf = {"red": 0.0, "blue": 3.0}
            transitions.append(
                {
                    "world_id": world_id,
                    "action": action,
                    "outcomes": [
                        {
                            "probability": 1.0,
                            "observation": {"root": "same"},
                            "successor": {"public": "same"},
                            "continuation_transitions": {
                                "continue": [
                                    {
                                        "probability": 1.0,
                                        "observation": {"second": "same"},
                                        "continuations": leaf,
                                    }
                                ]
                            },
                        }
                    ],
                }
            )
    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": "deep",
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["safe", "trap"],
        "dependency_candidates": ["opponent.active.item"],
        "declared_reads": {"safe": [], "trap": []},
        "transitions": transitions,
    }


def test_second_public_horizon_cannot_branch_on_hidden_world() -> None:
    result = analyze_oracle(_deep_oracle())

    assert result["continuation_decision_horizons"] == 2
    assert result["determinization"]["root_values"] == {
        "safe": 2.0,
        "trap": 3.0,
    }
    assert result["public_belief"]["root_values"] == {
        "safe": 2.0,
        "trap": 1.5,
    }
    assert result["determinization"]["chosen_action"] == "trap"
    assert result["public_belief"]["chosen_action"] == "safe"
    assert result["policy_disagreement"] is True


def test_deeper_information_set_rejects_mixed_horizon_evidence() -> None:
    document = _deep_oracle()
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    trap = next(
        transition
        for transition in transitions
        if transition["world_id"] == "blue" and transition["action"] == "trap"
    )
    outcomes = trap["outcomes"]
    assert isinstance(outcomes, list)
    outcome = outcomes[0]
    assert isinstance(outcome, dict)
    outcome.pop("continuation_transitions")
    outcome["continuations"] = {"continue": 1.0}

    with pytest.raises(BeliefTraceError, match="mixes shallow and deeper"):
        analyze_oracle(document)


def test_deeper_information_set_is_bounded_to_one_extra_horizon() -> None:
    document = _deep_oracle()
    transitions = document["transitions"]
    assert isinstance(transitions, list)
    first = transitions[0]
    outcomes = first["outcomes"]
    assert isinstance(outcomes, list)
    outcome = outcomes[0]
    assert isinstance(outcome, dict)
    branches = outcome["continuation_transitions"]
    assert isinstance(branches, dict)
    nested = branches["continue"]
    assert isinstance(nested, list)
    leaf = nested[0]
    assert isinstance(leaf, dict)
    leaf["continuation_transitions"] = {
        "too-deep": [
            {
                "probability": 1.0,
                "observation": {"third": "same"},
                "continuations": {"finish": 0.0},
            }
        ]
    }
    leaf.pop("continuations")

    with pytest.raises(BeliefTraceError, match="exceeds the supported"):
        analyze_oracle(document)


def test_showdown_probe_preserves_semantic_support_before_execution_projection() -> None:
    showdown_runtime = Path(__file__).resolve().parents[1] / "showdown" / "runtime"
    source = (showdown_runtime / "probe_real_belief_trace.cjs").read_text(encoding="utf-8")
    generator_source = (
        scripts / "real_belief_probe" / "generator_population.cjs"
    ).read_text(encoding="utf-8")
    opponent_source = (
        scripts / "real_belief_probe" / "opponent_policy.cjs"
    ).read_text(encoding="utf-8")
    compiler_source = (
        scripts / "real_belief_probe" / "transition_program_compiler.cjs"
    ).read_text(encoding="utf-8")

    assert '"--historical-showdown-commit"' in source
    assert 'historicalShowdownCommit && !posteriorOnly' in source
    assert 'historicalShowdownCommit || PINNED_SHOWDOWN_COMMIT' in source
    assert '"--generator-cache-dir"' in source
    assert "generatorCacheDir && !posteriorOnly" in source

    cache_block = generator_source.split(
        "function generatorPopulationMaterial(species)", 1
    )[1].split("function generatorVariants()", 1)[0]
    assert "showdown_commit: actualCommit" in cache_block
    assert 'format: "gen9randombattle"' in cache_block
    assert "species," in cache_block
    assert "opponent_is_lead: source.opponent_is_lead === true" in cache_block
    assert "generator_rounds: GENERATOR_ROUNDS" in cache_block
    assert "document.variants_sha256 !== sha256(document.variants)" in cache_block
    assert "sampled !== GENERATOR_ROUNDS" in cache_block

    conditioning_block = generator_source.split("function generatorVariants()", 1)[1].split(
        "function mechanicsProjectionVariantCount", 1
    )[0]
    assert "const population = generatorPopulation(species, requested)" in conditioning_block
    assert "const observed = new Set(observedOpponentMoves())" in conditioning_block
    assert "publicAbility && toID(set.ability) !== publicAbility" in conditioning_block
    assert "plausibleItemIds && !plausibleItemIds.has(toID(set.item))" in conditioning_block
    assert "[...observed].every(move => moves.includes(move))" in conditioning_block
    assert "function mechanicsProjectionVariantCount(variants)" in generator_source

    assert '"opponent.active.item": entry.set.item' in source
    assert '"opponent.active.ability": entry.set.ability' in source
    assert '"opponent.active.evs": entry.set.evs' in source
    assert '"opponent.active.ivs": entry.set.ivs' in source
    assert '"opponent.active.exact_hp": exactHp' in source

    hp_block = source.split("function exactMaxHpForVariant(variant)", 1)[1].split(
        "function applyFixtureState", 1
    )[0]
    assert "species.baseStats.hp" in hp_block
    assert "variant.level" in hp_block
    assert "variant.ivs && variant.ivs.hp" in hp_block
    assert "variant.evs && variant.evs.hp" in hp_block
    assert "species.maxHP" in hp_block
    assert "Math.floor(ev / 4)" in hp_block
    assert "buildBattle(" not in hp_block

    world_block = source.split("const worldById = new Map();", 1)[1].split(
        "const worlds = [...worldById.values()];", 1
    )[0]
    assert '"opponent.active.moves": entry.set.moves' in world_block
    assert '"opponent.active.tera_type": entry.set.teraType' in world_block
    assert "for (const entry of variants)" in world_block
    assert "mechanicsProjection" not in world_block

    dependency_block = source.split("const DEPENDENCY_CANDIDATES = [", 1)[1].split(
        "];", 1
    )[0]
    assert '"opponent.active.moves"' in dependency_block
    assert '"opponent.active.tera_type"' in dependency_block

    assert "mechanics_projection_variant_count" in source
    assert (
        "execution optimization only; semantic posterior support retains every "
        "generator variant"
    ) in source
    assert "function opponentActionDistribution(" in opponent_source
    assert "const currentSpecies = toID(" in source
    assert "moveSpecies === currentSpecies" in source
    assert '"simple-heuristics"' in opponent_source
    assert '"dirty-tricks"' in opponent_source
    assert '"max-damage"' in opponent_source
    assert "function moveDamageHeuristic(" in opponent_source
    assert "function simpleHeuristicsDistribution(" in opponent_source
    assert "function legalOpponentSwitches(" in opponent_source
    assert "function voluntarySwitchDistribution(" in opponent_source
    assert "function dirtyTricksDistribution(" in opponent_source
    assert "function equalStrategyMixture(" in opponent_source
    assert '"uniform-forced-switch"' in opponent_source
    assert 'hiddenReads.add("opponent.active.moves")' in opponent_source
    assert 'hiddenReads.add("opponent.active.evs")' in opponent_source
    assert 'hiddenReads.add("opponent.active.ivs")' in opponent_source
    assert 'hiddenReads.add("opponent.active.exact_hp")' in opponent_source
    assert 'hiddenReads.add("opponent.active.tera_type")' in opponent_source
    assert 'row.choice + " terastallize"' in opponent_source

    assert "showdown_turn_executions: showdownTurnExecutions" in compiler_source
    assert '"projection-first"' in compiler_source
    assert 'cacheMode === "projection-first"' in compiler_source
    assert '["projection", "exact"]' in compiler_source
    assert "for (const cacheKind of cacheProbeOrder)" in compiler_source
    assert "exactExecutionCacheMisses++" in compiler_source
    assert "projectedExecutionCacheMisses++" in compiler_source
    assert "route_execution_count: uniqueExecutions" in compiler_source
    assert "route_total_ms:" in compiler_source
    assert "uniqueExecutions * ROOT_CHANCE_SAMPLES" not in compiler_source
    assert "opponent_policy: OPPONENT_POLICY" in compiler_source

    posterior_block = source.split("if (posteriorOnly)", 1)[1].split(
        "createTransitionProgramCompiler", 1
    )[0]
    assert "generator_cache" not in posterior_block.lower()
    assert "marginalized_hidden" not in source

    # The entrypoint now composes authorities rather than re-implementing them.
    assert "createGeneratorPopulationSource" in source
    assert "createOpponentPolicyEngine" in source
    assert "createTransitionProgramCompiler" in source
    assert "function generatorPopulationMaterial" not in source
    assert "function moveDamageHeuristic(" not in source
    assert "function compileLazyWholeTurnPrograms(" not in source
