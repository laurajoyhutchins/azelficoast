from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from azelficoast.research.studies.action_survival import ActionSurvivalError, analyze_document


EXPERIMENT = Path("experiments/protect-action-survival.json")


def _document() -> dict[str, object]:
    return json.loads(EXPERIMENT.read_text(encoding="utf-8"))


def test_natural_protect_witness_has_all_roll_survival_split() -> None:
    result = analyze_document(_document())

    assert result["passed"] is True
    assert result["hidden_item_counts"] == {
        "Choice Scarf": 509,
        "Choice Specs": 518,
    }
    assert result["damage"] == {
        "Choice Scarf": {"min": 153, "max": 180},
        "Choice Specs": {"min": 229, "max": 270},
    }
    assert result["defender_full_hp"] == 229
    assert result["defender_hp_after_known_spikes"] == 201
    assert result["full_hp_survival_split"] is True
    assert result["observed_hp_survival_split"] is True


def test_beads_of_ruin_negative_control_collapses_robust_split() -> None:
    result = analyze_document(_document())

    assert result["without_defender_stat_modifier"] == {
        "Choice Scarf": {"min": 115, "max": 136},
        "Choice Specs": {"min": 171, "max": 202},
    }
    assert result["negative_control_collapses_full_hp_split"] is True


def test_witness_is_bound_to_pinned_showdown_revision() -> None:
    document = copy.deepcopy(_document())
    source = document["source"]
    assert isinstance(source, dict)
    source["showdown_commit"] = "stale"

    with pytest.raises(ActionSurvivalError, match="pinned Showdown"):
        analyze_document(document)
