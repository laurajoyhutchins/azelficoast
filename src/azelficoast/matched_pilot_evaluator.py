"""Deterministically build the frozen learned evaluator used by the matched pilot."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.belief_evaluator import (
    BeliefEvaluatorSpec,
    build_evaluator_input,
    init_params,
    write_checkpoint,
)
from azelficoast.belief_training import TrainingExample, train_examples

SOURCE_SCHEMA = "azelficoast.frozen-evaluator-source-state"
SOURCE_SCHEMA_VERSION = 1
SEARCH_UTILITY_MIN = -1.0
SEARCH_UTILITY_MAX = 6.0
SEED = 1729
EPOCHS = 64
LEARNING_RATE = 3e-4
POLICY_WEIGHT = 1.0
SPEC = BeliefEvaluatorSpec(
    public_width=128,
    world_width=128,
    action_width=64,
    hidden_width=96,
    world_hidden_width=96,
)


class MatchedPilotEvaluatorError(ValueError):
    """Raised when frozen training evidence cannot produce the pilot checkpoint."""


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MatchedPilotEvaluatorError(f"{path}: expected a JSON object")
    return value


def expand_posterior(source: Mapping[str, Any]) -> dict[str, Any]:
    components = source.get("posterior_components")
    if not isinstance(components, list) or not components:
        raise MatchedPilotEvaluatorError("source lacks frozen posterior components")

    worlds: list[dict[str, Any]] = []
    for component in components:
        if not isinstance(component, Mapping):
            raise MatchedPilotEvaluatorError("posterior component must be an object")
        hidden = component.get("hidden")
        provenance = component.get("provenance")
        exact_hp_values = component.get("exact_hp_values")
        weight = component.get("weight")
        if not isinstance(hidden, Mapping) or not isinstance(provenance, Mapping):
            raise MatchedPilotEvaluatorError(
                "posterior component lacks hidden state or provenance"
            )
        if not isinstance(exact_hp_values, list) or not exact_hp_values:
            raise MatchedPilotEvaluatorError("posterior component lacks exact HP support")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise MatchedPilotEvaluatorError(
                "posterior component weight must be positive and finite"
            )

        for exact_hp in exact_hp_values:
            if not isinstance(exact_hp, int) or isinstance(exact_hp, bool):
                raise MatchedPilotEvaluatorError("exact HP support must be integral")
            world_hidden = dict(hidden)
            world_hidden["opponent.active.exact_hp"] = exact_hp
            worlds.append(
                {
                    "weight": float(weight),
                    "hidden": world_hidden,
                    "provenance": dict(provenance),
                }
            )

    return {
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": worlds,
    }


def training_targets(
    source: Mapping[str, Any],
) -> tuple[str, float, tuple[float, ...], dict[str, float]]:
    public_state = source.get("public_state")
    target = source.get("search_target")
    if not isinstance(public_state, Mapping) or not isinstance(target, Mapping):
        raise MatchedPilotEvaluatorError("source lacks public state or search target")
    legal_actions = public_state.get("legal_actions")
    if (
        not isinstance(legal_actions, list)
        or not legal_actions
        or not all(isinstance(action, str) and action for action in legal_actions)
    ):
        raise MatchedPilotEvaluatorError("source lacks frozen legal actions")

    selected = target.get("selected_action")
    if not isinstance(selected, str) or selected not in legal_actions:
        raise MatchedPilotEvaluatorError("search target selected a nonlegal action")
    search_return = target.get("value")
    outcome = source.get("eventual_battle_outcome")
    if (
        not isinstance(search_return, (int, float))
        or isinstance(search_return, bool)
        or not math.isfinite(float(search_return))
    ):
        raise MatchedPilotEvaluatorError("search return must be finite")
    if (
        not isinstance(outcome, (int, float))
        or isinstance(outcome, bool)
        or not math.isfinite(float(outcome))
        or not -1.0 <= float(outcome) <= 1.0
    ):
        raise MatchedPilotEvaluatorError("battle outcome must be within [-1, 1]")

    normalized_search = max(
        -1.0,
        min(
            1.0,
            (
                2.0 * float(search_return)
                - (SEARCH_UTILITY_MAX + SEARCH_UTILITY_MIN)
            )
            / (SEARCH_UTILITY_MAX - SEARCH_UTILITY_MIN),
        ),
    )
    value_target = 0.5 * float(outcome) + 0.5 * normalized_search
    policy_target = tuple(
        1.0 if action == selected else 0.0 for action in legal_actions
    )
    return (
        selected,
        value_target,
        policy_target,
        {
            "search_return": float(search_return),
            "normalized_search_return": normalized_search,
            "eventual_battle_outcome": float(outcome),
            "combined_value_target": value_target,
        },
    )


def build_checkpoint(
    source: Mapping[str, Any],
    output: str | Path,
) -> dict[str, Any]:
    if (
        source.get("schema") != SOURCE_SCHEMA
        or source.get("schema_version") != SOURCE_SCHEMA_VERSION
    ):
        raise MatchedPilotEvaluatorError("unexpected frozen evaluator source schema")
    if source.get("showdown_commit") != (
        "a5df8274e85b0889bf2a9b3422a08b39732374fc"
    ):
        raise MatchedPilotEvaluatorError("frozen evaluator Showdown revision drifted")

    public_state = source.get("public_state")
    if not isinstance(public_state, Mapping):
        raise MatchedPilotEvaluatorError("source lacks public state")
    legal_actions = public_state.get("legal_actions")
    assert isinstance(legal_actions, list)

    posterior = expand_posterior(source)
    selected, value_target, policy_target, target_metadata = training_targets(source)
    inputs = build_evaluator_input(
        public_state=public_state,
        posterior=posterior,
        legal_actions=legal_actions,
        spec=SPEC,
    )
    params = init_params(SPEC, seed=SEED)
    trained, losses = train_examples(
        params,
        [
            TrainingExample(
                inputs=inputs,
                value_target=value_target,
                policy_target=policy_target,
            )
        ],
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        policy_weight=POLICY_WEIGHT,
    )

    provenance = source.get("provenance")
    if not isinstance(provenance, Mapping):
        raise MatchedPilotEvaluatorError("source lacks provenance")
    metadata = {
        "purpose": "matched-population pipeline checkpoint; not a model-strength claim",
        "training_example_count": 1,
        "training_battle_tag": source.get("battle_tag"),
        "training_fixture_id": source.get("fixture_id"),
        "terminal_event_index": source.get("terminal_event_index"),
        "search_artifact_id": provenance.get("search_artifact_id"),
        "source_artifact_id": provenance.get("source_artifact_id"),
        "source_artifact_digest": provenance.get("source_artifact_digest"),
        "policy_target": "exact_public_belief_search_action",
        "selected_action": selected,
        "value_target": (
            "0.5*eventual_battle_outcome + "
            "0.5*normalized_exact_public_belief_search_return"
        ),
        "search_utility_bounds": [SEARCH_UTILITY_MIN, SEARCH_UTILITY_MAX],
        **target_metadata,
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "policy_weight": POLICY_WEIGHT,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    destination = Path(output)
    manifest = write_checkpoint(
        destination,
        trained,
        SPEC,
        metadata=metadata,
    )
    summary = {
        "schema": "azelficoast.matched-pilot-evaluator-checkpoint-build",
        "schema_version": 1,
        "evaluator": manifest["evaluator"],
        "training_battle_tag": source.get("battle_tag"),
        "training_fixture_id": source.get("fixture_id"),
        "posterior_world_count": len(posterior["worlds"]),
        **target_metadata,
        "selected_action": selected,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
    }
    (destination / "build-summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    summary = build_checkpoint(_load_object(args.source), args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
