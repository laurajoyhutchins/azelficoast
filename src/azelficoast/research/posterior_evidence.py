"""Bind high-resolution conditional belief evidence to one frozen mechanics oracle."""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from azelficoast.research.contracts import stable_digest

REFERENCE_SCHEMA = "azelficoast.live-belief-posterior"
REFERENCE_SCHEMA_VERSION = 1


class PosteriorEvidenceError(ValueError):
    """Raised when conditional posterior evidence cannot be certified."""


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise PosteriorEvidenceError(f"{field} must be a positive integer")
    return value


def _world_index(document: Mapping[str, Any], *, label: str) -> dict[str, Mapping[str, Any]]:
    worlds = document.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise PosteriorEvidenceError(f"{label} has no hidden-world support")
    result: dict[str, Mapping[str, Any]] = {}
    for row in worlds:
        if not isinstance(row, Mapping):
            raise PosteriorEvidenceError(f"{label} contains a malformed hidden world")
        world_id = row.get("world_id")
        hidden = row.get("hidden")
        weight = row.get("weight")
        if not isinstance(world_id, str) or not world_id:
            raise PosteriorEvidenceError(f"{label} hidden world lacks identity")
        if world_id in result:
            raise PosteriorEvidenceError(f"{label} repeats hidden world {world_id}")
        if not isinstance(hidden, Mapping):
            raise PosteriorEvidenceError(f"{label} hidden world lacks semantics")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0
        ):
            raise PosteriorEvidenceError(f"{label} hidden-world weight must be positive")
        result[world_id] = row
    total = math.fsum(float(row["weight"]) for row in result.values())
    if total <= 0:
        raise PosteriorEvidenceError(f"{label} has no probability mass")
    return result


def bind_best_available_conditional(
    *,
    oracle: Mapping[str, Any],
    generator_reference: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Certify a high-resolution finite conditional sweep without calling it exact."""
    if oracle.get("schema") != "azelficoast.core.transition-oracle":
        raise PosteriorEvidenceError("unexpected mechanics oracle schema")
    if (
        generator_reference.get("schema") != REFERENCE_SCHEMA
        or generator_reference.get("schema_version") != REFERENCE_SCHEMA_VERSION
    ):
        raise PosteriorEvidenceError("unexpected generator-reference schema")
    if oracle.get("source_fixture_id") != generator_reference.get("source_fixture_id"):
        raise PosteriorEvidenceError("conditional treatments belong to different fixtures")
    if oracle.get("showdown_commit") != generator_reference.get("showdown_commit"):
        raise PosteriorEvidenceError("conditional treatments use different Showdown revisions")
    if generator_reference.get("conditioned_on_public_history") is not True:
        raise PosteriorEvidenceError("generator reference is not conditioned on public history")
    if generator_reference.get("realized_hidden_state_revealed") is not False:
        raise PosteriorEvidenceError("generator reference reveals the realized hidden state")

    reconstruction = oracle.get("reconstruction")
    reference_reconstruction = generator_reference.get("reconstruction")
    if not isinstance(reconstruction, Mapping) or not isinstance(reference_reconstruction, Mapping):
        raise PosteriorEvidenceError("conditional treatments lack reconstruction evidence")

    if authority.get("kind") != "best_available_conditional":
        raise PosteriorEvidenceError("contract did not authorize best-available conditional evidence")
    if authority.get("realized_hidden_state_used") is not False:
        raise PosteriorEvidenceError("oracle authority may not use realized hidden state")
    oracle_rounds = _positive_int(
        authority.get("oracle_generator_rounds"),
        field="oracle_generator_rounds",
    )
    reference_rounds = _positive_int(
        authority.get("generator_faithful_rounds"),
        field="generator_faithful_rounds",
    )
    if int(reconstruction.get("generator_rounds", 0)) != oracle_rounds:
        raise PosteriorEvidenceError("mechanics oracle used the wrong conditional sweep size")
    if int(reference_reconstruction.get("generator_rounds", 0)) != reference_rounds:
        raise PosteriorEvidenceError("generator-faithful reference used the wrong sweep size")
    if oracle_rounds <= reference_rounds:
        raise PosteriorEvidenceError("oracle treatment must use a denser sweep than generator-faithful")

    expected_schedule = authority.get("generator_seed_schedule")
    if not isinstance(expected_schedule, str) or not expected_schedule:
        raise PosteriorEvidenceError("contract lacks generator seed schedule")
    if reconstruction.get("generator_seed_schedule") != expected_schedule:
        raise PosteriorEvidenceError("mechanics oracle generator schedule drifted")
    if reference_reconstruction.get("generator_seed_schedule") != expected_schedule:
        raise PosteriorEvidenceError("generator reference schedule drifted")
    if reconstruction.get("generator_seed_start") != 0:
        raise PosteriorEvidenceError("mechanics oracle generator sweep must start at zero")
    if reference_reconstruction.get("generator_seed_start") != 0:
        raise PosteriorEvidenceError("generator reference sweep must start at zero")

    oracle_worlds = _world_index(oracle, label="mechanics oracle")
    reference_worlds = _world_index(generator_reference, label="generator reference")
    if set(oracle_worlds) != set(reference_worlds):
        missing = sorted(set(oracle_worlds) ^ set(reference_worlds))
        raise PosteriorEvidenceError(
            f"conditional sweeps disagree on hidden-world support: {missing[:3]!r}"
        )

    weights: list[dict[str, object]] = []
    for world_id in sorted(oracle_worlds):
        oracle_world = oracle_worlds[world_id]
        reference_world = reference_worlds[world_id]
        if stable_digest(oracle_world["hidden"]) != stable_digest(reference_world["hidden"]):
            raise PosteriorEvidenceError(
                f"conditional sweeps disagree on hidden semantics for {world_id}"
            )
        weights.append(
            {
                "world_id": world_id,
                "weight": float(reference_world["weight"]),
            }
        )

    reference_evidence = {
        "generator_rounds": reference_rounds,
        "generator_matches": _positive_int(
            reference_reconstruction.get("generator_matches"),
            field="generator reference matches",
        ),
        "generator_seed_schedule": expected_schedule,
        "generator_seed_start": 0,
        "world_weights": weights,
    }
    oracle_evidence = {
        "authority_kind": "best_available_conditional",
        "exact": False,
        "generator_rounds": oracle_rounds,
        "generator_matches": _positive_int(
            reconstruction.get("generator_matches"),
            field="oracle conditional matches",
        ),
        "generator_seed_schedule": expected_schedule,
        "generator_seed_start": 0,
        "support_count": len(oracle_worlds),
        "conditioned_on_public_history": True,
        "realized_hidden_state_used": False,
        "scope": "finite deterministic generator seed sweep conditioned on public evidence",
        "caveat": "not exhaustive over the full Showdown PRNG state space",
    }
    oracle_evidence["evidence_digest"] = stable_digest(oracle_evidence)
    reference_evidence["evidence_digest"] = stable_digest(reference_evidence)

    result = copy.deepcopy(dict(oracle))
    mutable_reconstruction = result.get("reconstruction")
    assert isinstance(mutable_reconstruction, dict)
    mutable_reconstruction["posterior_authority"] = "best_available_conditional"
    mutable_reconstruction["best_available_conditional_evidence"] = oracle_evidence
    mutable_reconstruction["generator_faithful_reference"] = reference_evidence
    return result


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PosteriorEvidenceError(f"{path}: expected a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("oracle", type=Path)
    parser.add_argument("generator_reference", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    contract = _load_object(args.contract)
    authority = contract.get("oracle_authority")
    if not isinstance(authority, Mapping):
        raise PosteriorEvidenceError("contract lacks oracle authority")
    result = bind_best_available_conditional(
        oracle=_load_object(args.oracle),
        generator_reference=_load_object(args.generator_reference),
        authority=authority,
    )
    args.output.write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "fixture_id": result.get("source_fixture_id"),
        "posterior_authority": result["reconstruction"]["posterior_authority"],
        "evidence_digest": (
            result["reconstruction"]["best_available_conditional_evidence"][
                "evidence_digest"
            ]
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
