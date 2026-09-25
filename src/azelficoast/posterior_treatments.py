"""Posterior treatment construction for matched search experiments."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from azelficoast.real_belief_trace import (
    SCHEMA as ORACLE_SCHEMA,
    SCHEMA_VERSION as ORACLE_SCHEMA_VERSION,
)

POSTERIOR_SCHEMA = "azelficoast.matched-search-posterior"
POSTERIOR_SCHEMA_VERSION = 1


class PosteriorTreatmentError(ValueError):
    """Raised when a requested posterior treatment cannot be justified."""


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PosteriorTreatmentError(f"{path}: expected a JSON object")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _validated_oracle(oracle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if (
        oracle.get("schema") != ORACLE_SCHEMA
        or oracle.get("schema_version") != ORACLE_SCHEMA_VERSION
    ):
        raise PosteriorTreatmentError("unexpected transition-oracle schema")
    worlds = oracle.get("worlds")
    reconstruction = oracle.get("reconstruction")
    if not isinstance(worlds, list) or not worlds:
        raise PosteriorTreatmentError("transition oracle has no hidden-world support")
    if not isinstance(reconstruction, Mapping):
        raise PosteriorTreatmentError("transition oracle lacks reconstruction evidence")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0.0
    for raw in worlds:
        if not isinstance(raw, Mapping):
            raise PosteriorTreatmentError("hidden-world support must contain objects")
        world_id = raw.get("world_id")
        hidden = raw.get("hidden")
        weight = raw.get("weight")
        if not isinstance(world_id, str) or not world_id:
            raise PosteriorTreatmentError("hidden world lacks world_id")
        if world_id in seen:
            raise PosteriorTreatmentError(f"duplicate hidden world {world_id}")
        if not isinstance(hidden, Mapping):
            raise PosteriorTreatmentError(f"{world_id}: hidden state is missing")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
            raise PosteriorTreatmentError(f"{world_id}: hidden-world weight must be positive")
        seen.add(world_id)
        total += float(weight)
        normalized.append(
            {
                "world_id": world_id,
                "weight": float(weight),
                "hidden": copy.deepcopy(dict(hidden)),
                "provenance": copy.deepcopy(dict(raw.get("provenance", {}))),
            }
        )
    if total <= 0:
        raise PosteriorTreatmentError("hidden-world support has no probability mass")
    for world in normalized:
        world["weight"] = float(world["weight"]) / total
    return normalized, reconstruction


def _artifact(
    *,
    oracle: Mapping[str, Any],
    treatment: str,
    worlds: Sequence[Mapping[str, Any]],
    construction: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": POSTERIOR_SCHEMA,
        "schema_version": POSTERIOR_SCHEMA_VERSION,
        "treatment": treatment,
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "source_fixture_id": oracle.get("source_fixture_id"),
        "showdown_commit": oracle.get("showdown_commit"),
        "transition_oracle_digest": _sha256(oracle),
        "construction": dict(construction),
        "worlds": [copy.deepcopy(dict(world)) for world in worlds],
    }


def generator_faithful_posterior(oracle: Mapping[str, Any]) -> dict[str, Any]:
    worlds, reconstruction = _validated_oracle(oracle)
    rounds = reconstruction.get("generator_rounds")
    matches = reconstruction.get("generator_matches")
    if not isinstance(rounds, int) or rounds <= 0:
        raise PosteriorTreatmentError("generator posterior lacks generator-round evidence")
    if not isinstance(matches, int) or matches <= 0:
        raise PosteriorTreatmentError("generator posterior lacks conditioned-match evidence")

    if any(
        not isinstance(world.get("provenance"), Mapping)
        or int(world["provenance"].get("generator_rounds", -1)) != rounds
        for world in worlds
    ):
        raise PosteriorTreatmentError(
            "world provenance is not bound to the reconstruction generator sweep"
        )

    return _artifact(
        oracle=oracle,
        treatment="generator_faithful",
        worlds=worlds,
        construction={
            "kind": "conditioned-generator-frequency",
            "generator_rounds": rounds,
            "generator_matches": matches,
            "preserves_joint_hidden_worlds": True,
            "weight_rule": "normalized transition-oracle generator mass",
        },
    )


def practical_posterior(oracle: Mapping[str, Any]) -> dict[str, Any]:
    worlds, reconstruction = _validated_oracle(oracle)
    rounds = reconstruction.get("generator_rounds")
    if not isinstance(rounds, int) or rounds <= 0:
        raise PosteriorTreatmentError("practical posterior lacks generator evidence")

    item_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_mass: dict[str, float] = defaultdict(float)
    for world in worlds:
        hidden = world["hidden"]
        assert isinstance(hidden, Mapping)
        item = hidden.get("opponent.active.item")
        if not isinstance(item, str) or not item:
            raise PosteriorTreatmentError(
                "practical item-marginal treatment requires hidden item identity"
            )
        item_groups[item].append(world)
        item_mass[item] += float(world["weight"])

    approximated: list[dict[str, Any]] = []
    for item in sorted(item_groups):
        group = item_groups[item]
        per_world = item_mass[item] / len(group)
        for world in group:
            row = copy.deepcopy(world)
            row["weight"] = per_world
            approximated.append(row)

    approximated.sort(key=lambda world: str(world["world_id"]))
    return _artifact(
        oracle=oracle,
        treatment="practical",
        worlds=approximated,
        construction={
            "kind": "item-marginal-uniform-within-item",
            "generator_rounds": rounds,
            "preserves_joint_hidden_world_support": True,
            "preserves_item_marginal_mass": True,
            "discarded_weight_structure": [
                "ability-within-item frequency",
                "EV/IV-within-item frequency",
                "exact-HP-within-item frequency",
            ],
            "purpose": (
                "bounded practical approximation that retains mechanics support "
                "while deliberately discarding generator correlations below item"
            ),
        },
    )


def oracle_posterior(oracle: Mapping[str, Any]) -> dict[str, Any]:
    worlds, reconstruction = _validated_oracle(oracle)
    if reconstruction.get("posterior_authority") != "exact_conditional":
        raise PosteriorTreatmentError(
            "oracle posterior requires exact_conditional reconstruction authority"
        )
    evidence = reconstruction.get("exact_conditional_evidence")
    if not isinstance(evidence, Mapping) or not evidence:
        raise PosteriorTreatmentError(
            "oracle posterior lacks exact conditional evidence"
        )
    return _artifact(
        oracle=oracle,
        treatment="oracle",
        worlds=worlds,
        construction={
            "kind": "exact-conditional",
            "evidence": copy.deepcopy(dict(evidence)),
            "realized_hidden_state_used": False,
        },
    )


def build_posterior(
    oracle: Mapping[str, Any],
    *,
    treatment: str,
) -> dict[str, Any]:
    if treatment == "generator_faithful":
        return generator_faithful_posterior(oracle)
    if treatment == "practical":
        return practical_posterior(oracle)
    if treatment == "oracle":
        return oracle_posterior(oracle)
    raise PosteriorTreatmentError(f"unknown posterior treatment {treatment!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument(
        "--treatment",
        required=True,
        choices=("oracle", "generator_faithful", "practical"),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    result = build_posterior(_load_object(args.oracle), treatment=args.treatment)
    args.output.write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
