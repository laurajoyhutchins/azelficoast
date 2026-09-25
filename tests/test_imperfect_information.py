from __future__ import annotations

import pytest

from azelficoast.corpus import DecisionFixture, load_policy
from azelficoast.search.imperfect_information import (
    GAME_KEY,
    DeterminizationPolicy,
    ImperfectInformationError,
    PublicBeliefPolicy,
    run_reference_experiment,
)


def _fixture(*, reveal: bool) -> DecisionFixture:
    actions = {
        "/choose move safe": {
            "terminal_payoffs": {
                "red": 0.25,
                "blue": 0.25,
            }
        },
        "/choose move guess": {
            "observations": {
                "red": "red" if reveal else "unrevealed",
                "blue": "blue" if reveal else "unrevealed",
            },
            "continuations": {
                "guess-red": {"red": 1.0, "blue": -1.0},
                "guess-blue": {"red": -1.0, "blue": 1.0},
            },
        },
        "/choose move scout": {
            "observations": {"red": "red", "blue": "blue"},
            "continuations": {
                "guess-red": {"red": 0.8, "blue": -1.2},
                "guess-blue": {"red": -1.2, "blue": 0.8},
            },
        },
    }
    return DecisionFixture(
        fixture_id="test",
        state={
            "legal_actions": sorted(actions),
            GAME_KEY: {
                "worlds": [
                    {"name": "red", "weight": 0.5},
                    {"name": "blue", "weight": 0.5},
                ],
                "actions": actions,
            },
        },
        protocol_prefix=(),
        control_decisions=(),
    )


def test_determinization_exhibits_strategy_fusion_on_hidden_worlds() -> None:
    solution = DeterminizationPolicy().solve(_fixture(reveal=False))

    assert solution.action == "/choose move guess"
    assert solution.action_values["/choose move guess"] == 1
    assert solution.continuation_by_action["/choose move guess"] == {
        "red": "guess-red",
        "blue": "guess-blue",
    }


def test_public_belief_enforces_one_continuation_per_information_set() -> None:
    solution = PublicBeliefPolicy().solve(_fixture(reveal=False))

    assert solution.action == "/choose move scout"
    assert solution.action_values["/choose move guess"] == 0
    assert solution.action_values["/choose move scout"] == pytest.approx(0.8)
    assert solution.continuation_by_action["/choose move guess"] == {
        "unrevealed": "guess-blue"
    }


def test_negative_control_agrees_when_hidden_world_is_revealed() -> None:
    fixture = _fixture(reveal=True)
    determinization = DeterminizationPolicy().solve(fixture)
    public_belief = PublicBeliefPolicy().solve(fixture)

    assert determinization.action == "/choose move guess"
    assert public_belief.action == "/choose move guess"
    assert determinization.action_values == public_belief.action_values


def test_policies_load_through_corpus_plugin_boundary() -> None:
    determinization = load_policy(
        "azelficoast.search.imperfect_information:determinization"
    )
    public_belief = load_policy(
        "azelficoast.search.imperfect_information:public_belief"
    )

    fixture = _fixture(reveal=False)
    assert determinization.choose(fixture) == "/choose move guess"
    assert public_belief.choose(fixture) == "/choose move scout"


def test_reference_experiment_passes_treatment_and_negative_control() -> None:
    result = run_reference_experiment()

    assert result["passed"] is True
    assert result["treatment"]["passed"] is True
    assert result["negative_control"]["passed"] is True


def test_game_rejects_actions_not_legal_in_fixture() -> None:
    fixture = _fixture(reveal=False)
    bad_state = dict(fixture.state)
    bad_state["legal_actions"] = ["/choose move safe"]

    with pytest.raises(ImperfectInformationError, match="not legal"):
        DeterminizationPolicy().choose(
            DecisionFixture(
                fixture_id="bad",
                state=bad_state,
                protocol_prefix=(),
                control_decisions=(),
            )
        )
