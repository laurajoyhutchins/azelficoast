from __future__ import annotations


import pytest

from azelficoast.research.studies.matched_comparison import (
    MatchedComparisonError,
    freeze_packet,
    settle_packet,
)
from azelficoast.research.studies.matched_population import aggregate_population
from hostile.fixtures import hostile_case, matched_plan, matched_posterior, matched_state
from hostile.transforms import (
    perturb_hidden_field,
    rename_world_ids,
    permute_support,
    split_world,
    merge_equivalent_worlds,
)
from test_matched_comparison import _receipt


def packet(*, plan=None, state=None, posterior=None, treatment="generator_faithful", depth=1):
    return freeze_packet(
        plan=plan or matched_plan(),
        state=state or matched_state(),
        posterior=posterior or matched_posterior(),
        posterior_treatment=treatment,
        depth=depth,
    )


def receipts(value):
    actions = value["legal_actions"]
    rows = []
    for method in ("determinization", "information_set"):
        scores = {action: float(index) for index, action in enumerate(actions)}
        rows.append(
            _receipt(
                value,
                method,
                chosen_action=max(scores, key=scores.get),
                root_values=scores,
                consumed=4,
            )
        )
    return rows


@hostile_case(
    mutation="rename, reorder, split, merge, or scale equivalent posterior support",
    expected="semantic input identity and settled scientific metrics remain invariant",
    threat="transport identifiers or support encoding affect a matched result",
    layer="matched input identity",
)
def test_semantically_equivalent_support_has_same_matched_input():
    base = matched_posterior()
    variants = [rename_world_ids(base, ["opaque-900", "opaque-2"]), permute_support(base, (1, 0))]
    split = split_world(base, 0, (0.1, 0.15), identifiers=("copy-a", "copy-b"))
    variants.extend((split, merge_equivalent_worlds(split)))
    base_packet = packet(posterior=base)
    for changed in variants:
        other = packet(posterior=changed)
        assert other["posterior_semantic_digest"] == base_packet["posterior_semantic_digest"]
        assert other["input_digest"] == base_packet["input_digest"]
        # Raw artifact/packet digests intentionally bind serialization evidence.
        assert (
            settle_packet(packet=other, receipts=receipts(other))["input_digest"]
            == base_packet["input_digest"]
        )


@hostile_case(
    mutation="change one semantic hidden field",
    expected="semantic posterior and input identities change",
    threat="a real posterior mismatch is accepted",
    layer="matched packet construction",
)
def test_semantic_posterior_change_changes_identity():
    base = matched_posterior()
    changed = perturb_hidden_field(base, 0, ("hidden", "item"), "specs")
    assert packet(posterior=changed)["input_digest"] != packet(posterior=base)["input_digest"]


@hostile_case(
    mutation="increase one receipt's consumed work by one beyond its authorization",
    expected="settlement rejects the overrun",
    threat="an arm receives unmatched compute",
    layer="receipt settlement",
)
def test_budget_plus_one_fails_closed():
    value = packet()
    pair = receipts(value)
    pair[1]["consumed"] = value["compute_budget"]["authorized"] + 1
    with pytest.raises(MatchedComparisonError, match="exceeded"):
        settle_packet(packet=value, receipts=pair)


@hostile_case(
    mutation="change exactly one frozen public state, posterior, mechanics, evaluator, depth, or budget input",
    expected="the original evidence cannot settle against the changed packet",
    threat="receipts from another experiment are reused",
    layer="matched receipt settlement",
)
@pytest.mark.parametrize(
    "mutation", ["state", "posterior", "revision", "evaluator", "depth", "budget"]
)
def test_one_field_drift_rejects_old_receipts(mutation):
    original = packet()
    plan = matched_plan()
    state = matched_state()
    belief = matched_posterior()
    depth = 1
    if mutation == "state":
        state["public_state"]["turn"] += 1
    elif mutation == "posterior":
        belief["worlds"][0]["weight"] = 0.5
        belief["worlds"][1]["weight"] = 0.5
    elif mutation == "revision":
        plan["showdown_commit"] = "another-pinned-revision"
    elif mutation == "evaluator":
        plan["evaluator"]["checkpoint_digest"] = "sha256:" + "b" * 64
    elif mutation == "depth":
        depth = 2
    elif mutation == "budget":
        plan["compute_budget"]["per_method_limit"] += 1
    altered = packet(plan=plan, state=state, posterior=belief, depth=depth)
    with pytest.raises(MatchedComparisonError):
        settle_packet(packet=original, receipts=receipts(altered))


@hostile_case(
    mutation="permute complete result rows before aggregation",
    expected="fixed-seed population statistics are unchanged",
    threat="input ordering influences inference",
    layer="population aggregation",
)
def test_population_order_invariance():
    plan = matched_plan()
    cohort = {
        "schema": "azelficoast.matched-search-population-cohort",
        "schema_version": 1,
        "selected": [],
    }
    results = []
    # A compact complete treatment × depth matrix.
    for treatment in plan["posterior_treatments"]:
        belief = matched_posterior()
        belief["treatment"] = treatment
        for depth in plan["depths"]:
            value = packet(posterior=belief, treatment=treatment, depth=depth)
            results.append(settle_packet(packet=value, receipts=receipts(value)))
    cohort["selected"] = [{"fixture_id": "hostile-fixture-1", "battle_tag": "hostile-battle-1"}]
    a = aggregate_population(plan=plan, cohort=cohort, results=results)
    b = aggregate_population(plan=plan, cohort=cohort, results=list(reversed(results)))
    assert a == b
