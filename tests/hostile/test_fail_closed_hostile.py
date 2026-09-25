from __future__ import annotations

import pytest

from azelficoast.research.matched_comparison import (
    MatchedComparisonError,
    freeze_packet,
    settle_packet,
)
from hostile.fixtures import hostile_case, matched_plan, matched_posterior, matched_state
from test_matched_comparison import _receipt


def packet(posterior=None):
    return freeze_packet(
        plan=matched_plan(),
        state=matched_state(),
        posterior=posterior or matched_posterior(),
        posterior_treatment="generator_faithful",
        depth=1,
    )


def receipts(value):
    return [
        _receipt(
            value,
            method,
            chosen_action=value["legal_actions"][0],
            root_values={
                action: float(index == 0) for index, action in enumerate(value["legal_actions"])
            },
            consumed=2,
        )
        for method in ("determinization", "information_set")
    ]


@hostile_case(
    mutation="supply negative, zero-total, NaN, infinite, or missing posterior mass",
    expected="packet construction raises an explicit validation error",
    threat="invalid probability mass is silently dropped",
    layer="posterior validation",
)
@pytest.mark.parametrize("attack", ["negative", "zero", "nan", "infinite", "missing"])
def test_malformed_posterior_fails_closed(attack):
    belief = matched_posterior()
    if attack == "negative":
        belief["worlds"][0]["weight"] = -1
    elif attack == "zero":
        for world in belief["worlds"]:
            world["weight"] = 0
    elif attack == "nan":
        belief["worlds"][0]["weight"] = float("nan")
    elif attack == "infinite":
        belief["worlds"][0]["weight"] = float("inf")
    else:
        del belief["worlds"][0]["weight"]
    with pytest.raises(MatchedComparisonError):
        packet(belief)


@hostile_case(
    mutation="inject realized-world data beneath nested auxiliary metadata",
    expected="packet construction rejects the private payload",
    threat="public decisions condition on the realized hidden state",
    layer="public-belief boundary",
)
def test_nested_realized_world_payload_rejected():
    belief = matched_posterior()
    belief["metadata"] = {"sampled_world": {"item": "band"}}
    with pytest.raises(MatchedComparisonError, match="realized hidden-world"):
        packet(belief)


@hostile_case(
    mutation="make one receipt consume one unit above the authorized ceiling",
    expected="settlement rejects before inference",
    threat="one method receives extra compute",
    layer="receipt validation",
)
def test_receipt_budget_plus_one_rejected():
    value = packet()
    pair = receipts(value)
    pair[1]["consumed"] = value["compute_budget"]["authorized"] + 1
    with pytest.raises(MatchedComparisonError, match="exceeded"):
        settle_packet(packet=value, receipts=pair)


@hostile_case(
    mutation="change a receipt's evaluator identity or create inconsistent accounting",
    expected="settlement rejects the forged evidence",
    threat="a plausible result is published from mismatched or internally impossible evidence",
    layer="receipt validation",
)
@pytest.mark.parametrize("attack", ["evaluator", "negative", "missing_digest", "oracle"])
def test_forged_receipts_fail_closed(attack):
    value = packet()
    pair = receipts(value)
    if attack == "evaluator":
        pair[1]["evaluator_digest"] = "f" * 64
    elif attack == "negative":
        pair[1]["consumed"] = -1
    elif attack == "missing_digest":
        del pair[1]["transition_program_digest"]
    else:
        pair[1]["transition_oracle_digest"] = "different-oracle"
    with pytest.raises(MatchedComparisonError):
        settle_packet(packet=value, receipts=pair)


@hostile_case(
    mutation="change packet budget while recomputing only its outer packet digest",
    expected="the packet's frozen spec detects internal drift",
    threat="altered authorization is disguised as the original experiment",
    layer="packet integrity",
)
def test_packet_budget_tampering_rejected():
    value = packet()
    value["compute_budget"]["authorized"] += 1
    from azelficoast.research.matched_comparison import _sha256

    unsigned = dict(value)
    unsigned.pop("packet_digest")
    value["packet_digest"] = _sha256(unsigned)
    with pytest.raises(MatchedComparisonError, match="budget"):
        settle_packet(packet=value, receipts=receipts(packet()))
