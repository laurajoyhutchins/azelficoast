from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from azelficoast.research.contracts import BeliefInput
from azelficoast.core.transition import sha256_json
from hostile.fixtures import hostile_case, matched_posterior
from hostile.transforms import (
    merge_equivalent_worlds,
    permute_support,
    rename_world_ids,
    split_world,
)


ROOT = Path(__file__).resolve().parents[2]
IDENTITY_FIXTURE = Path(__file__).with_name("semantic_identity_fixture.json")


def _posterior_identity(posterior: dict[str, Any]) -> str:
    return BeliefInput.from_record(posterior).semantic_digest


def _semantic_record(posterior: dict[str, Any]) -> dict[str, Any]:
    return BeliefInput.from_record(posterior).to_evaluator_record()


def _reverse_mapping_order(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _reverse_mapping_order(item) for key, item in reversed(list(value.items()))}
    if isinstance(value, list):
        return [_reverse_mapping_order(item) for item in value]
    return value


@hostile_case(
    mutation="reorder mapping keys, change JSON whitespace, and round-trip one frozen semantic artifact in Python and Node",
    expected="both runtimes produce the frozen SHA-256 identity while a semantic field change produces a different identity",
    threat="cross-language canonicalization drift silently changes transition evidence identity",
    layer="semantic serialization and artifact identity",
)
def test_frozen_cross_language_semantic_identity() -> None:
    # The shared JSON contract currently admits only safe integers in this
    # cross-language fixture; float formatting has a separate contract to define.
    payload = {
        "transition": {"action": "hold", "outcomes": [{"probability": 1, "successor": {"hp": 17}}]},
        "actions": ["a", "b"],
    }
    expected = sha256_json(payload)
    reordered = _reverse_mapping_order(payload)
    assert sha256_json(reordered) == expected
    assert sha256_json(json.loads(json.dumps(payload, indent=8))) == expected
    changed = copy.deepcopy(payload)
    changed["transition"]["outcomes"][0]["successor"]["hp"] -= 1
    assert sha256_json(changed) != expected
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the cross-language identity half")
    result = subprocess.run(
        [node, "scripts/semantic_identity.cjs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        input=json.dumps(payload),
        text=True,
    )
    assert result.stdout.strip() == expected


@hostile_case(
    mutation="encode safe integer values across Python and Node",
    expected="both runtimes produce the same identity for safe integer semantic values",
    threat="cross-language transition identities diverge on ordinary floating-point serialization",
    layer="cross-language semantic serialization",
)
@pytest.mark.parametrize(
    "number",
    [0, 1, -1, 17, 9007199254740991],
)
def test_cross_language_numeric_canonicalization(number: int | float) -> None:
    payload = {"number": number}
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the cross-language identity half")
    result = subprocess.run(
        [node, "scripts/semantic_identity.cjs"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        input=json.dumps(payload),
        text=True,
    )
    assert result.stdout.strip() == sha256_json(payload)


@hostile_case(
    mutation="rename and reorder posterior rows, split and merge equivalent atoms, and reserialize the semantic record",
    expected="semantic posterior digest is exact and stable, while a changed hidden value or posterior mass changes it",
    threat="transport representation is treated as evidence or a real posterior change is erased",
    layer="posterior canonical identity",
)
def test_posterior_identity_round_trip_and_semantic_sensitivity() -> None:
    posterior = matched_posterior()
    split = split_world(posterior, 0, (0.1, 0.15), identifiers=("x", "y"))
    variants = (
        rename_world_ids(posterior, ["id-92841", "id-negative-seven"]),
        permute_support(posterior, (1, 0)),
        split,
        merge_equivalent_worlds(split),
    )
    expected = _posterior_identity(posterior)
    assert _semantic_record(posterior) == _semantic_record(
        json.loads(json.dumps(posterior, ensure_ascii=False, indent=3))
    )
    assert all(_posterior_identity(value) == expected for value in variants)

    changed_semantics = copy.deepcopy(posterior)
    changed_semantics["worlds"][0]["hidden"]["item"] = "specs"
    assert _posterior_identity(changed_semantics) != expected
    changed_mass = copy.deepcopy(posterior)
    changed_mass["worlds"][0]["weight"] += 0.01
    assert _posterior_identity(changed_mass) != expected


@hostile_case(
    mutation="multiply non-binary-friendly posterior masses by a positive scalar",
    expected="normalized posterior identity is unchanged while a resolvable mass change remains visible",
    threat="floating-point normalization noise turns an equivalent posterior into a different experiment",
    layer="posterior canonical identity",
)
def test_posterior_identity_is_stable_under_non_binary_friendly_scaling() -> None:
    posterior = matched_posterior()
    posterior["worlds"] = [
        {"world_id": f"world-{index}", "weight": weight, "hidden": {"kind": str(index)}}
        for index, weight in enumerate((0.1, 0.2, 0.7))
    ]
    scaled = copy.deepcopy(posterior)
    for world in scaled["worlds"]:
        world["weight"] *= 3

    assert _posterior_identity(scaled) == _posterior_identity(posterior)

    changed = copy.deepcopy(posterior)
    changed["worlds"][0]["weight"] += 1e-13
    assert _posterior_identity(changed) != _posterior_identity(posterior)
