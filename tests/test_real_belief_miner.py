from __future__ import annotations

import copy

from azelficoast.research.verification.real_belief_miner import mine_oracles


def _oracle(
    *,
    fixture_id: str,
    hidden_sensitive: bool,
    disagreement: bool,
) -> dict[str, object]:
    worlds = [
        {
            "world_id": "left",
            "weight": 0.5,
            "hidden": {"opponent.active.item": "left"},
        },
        {
            "world_id": "right",
            "weight": 0.5,
            "hidden": {"opponent.active.item": "right"},
        },
    ]

    transitions = []
    for world in worlds:
        item = world["hidden"]["opponent.active.item"]
        if hidden_sensitive:
            observation = {"item_effect": item}
            successor = {"hp": 1 if item == "left" else 2}
        else:
            observation = {"same": True}
            successor = {"hp": 1}

        if disagreement:
            continuation = {
                "world-aware": 3 if item == "left" else -3,
                "safe": 1,
            }
        else:
            continuation = {
                "world-aware": 3,
                "safe": 1,
            }

        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "hold",
                "outcomes": [
                    {
                        "probability": 1.0,
                        "observation": observation,
                        "successor": successor,
                        "continuations": continuation,
                    }
                ],
            }
        )

        reveal_values = (
            {"world-aware": 1.5, "safe": 1.5}
            if disagreement
            else {"world-aware": 2, "safe": 0}
        )
        transitions.append(
            {
                "world_id": world["world_id"],
                "action": "reveal",
                "outcomes": [
                    {
                        "probability": 1.0,
                        "observation": {"revealed": item},
                        "successor": {"hp": 1},
                        "continuations": reveal_values,
                    }
                ],
            }
        )

    return {
        "schema": "azelficoast.core.transition-oracle",
        "schema_version": 1,
        "source_fixture_id": fixture_id,
        "showdown_commit": "pinned",
        "worlds": worlds,
        "legal_actions": ["hold", "reveal"],
        "dependency_candidates": ["opponent.active.item"],
        "declared_reads": {
            "hold": ["opponent.active.item"] if hidden_sensitive else [],
            "reveal": ["opponent.active.item"],
        },
        "transitions": transitions,
    }


def test_miner_ranks_by_structure_before_inspecting_policy_result() -> None:
    structurally_rich = _oracle(
        fixture_id="rich-no-disagreement",
        hidden_sensitive=True,
        disagreement=False,
    )
    structurally_poor = _oracle(
        fixture_id="poor-disagreement",
        hidden_sensitive=False,
        disagreement=True,
    )

    result = mine_oracles(
        [
            ("poor.json", structurally_poor),
            ("rich.json", structurally_rich),
        ]
    )

    assert result["ranking_uses_policy_result"] is False
    assert result["evaluated_count"] == 2
    assert result["ranked_candidates"][0]["source_fixture_id"] == "rich-no-disagreement"

    # Strategy-fusion and root-policy outcomes are inspected only after structural
    # ordering rather than promoted above a structurally stronger candidate.
    assert result["first_strategy_fusion_candidate"]["source_fixture_id"] == "poor-disagreement"
    assert result["first_disagreement"]["source_fixture_id"] == "poor-disagreement"


def test_miner_reports_valid_negative_corpus_without_manufacturing_disagreement() -> None:
    oracle = _oracle(
        fixture_id="negative",
        hidden_sensitive=True,
        disagreement=False,
    )
    result = mine_oracles([("negative.json", oracle)])

    assert result["evaluated_count"] == 1
    assert result["strategy_fusion_candidate_count"] == 0
    assert result["disagreement_count"] == 0
    assert result["first_strategy_fusion_candidate"] is None
    assert result["first_disagreement"] is None
    assert result["ranked_candidates"][0]["policy_disagreement"] is False


def test_miner_order_is_invariant_to_policy_result_when_structure_is_fixed() -> None:
    first = _oracle(
        fixture_id="a",
        hidden_sensitive=True,
        disagreement=False,
    )
    second = copy.deepcopy(first)
    second["source_fixture_id"] = "b"

    result = mine_oracles([("b.json", second), ("a.json", first)])
    assert [
        row["source_fixture_id"] for row in result["ranked_candidates"]
    ] == ["a", "b"]
