"""Validation and projection for joint Gen 9 Random Battle hidden-team posteriors."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

from azelficoast.posterior_validity import posterior_diagnostics

SCHEMA = "azelficoast.joint-random-battle-posterior"
SCHEMA_VERSION = 1


class JointPosteriorError(ValueError):
    """Raised when a joint hidden-team posterior is not safe to consume."""


_REQUIRED_SET_FIELDS = {
    "species",
    "level",
    "gender",
    "ability",
    "item",
    "moves",
    "tera_type",
    "role",
    "nature",
    "evs",
    "ivs",
    "was_lead",
}


def validate_joint_posterior(
    document: Mapping[str, Any],
    *,
    require_sufficient_support: bool = True,
) -> dict[str, Any]:
    """Validate one posterior without factorizing any hidden-team particle."""

    if (
        document.get("schema") != SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise JointPosteriorError("unexpected joint posterior schema")
    if document.get("conditioned_on_public_history") is not True:
        raise JointPosteriorError("posterior is not conditioned on public history")
    if document.get("realized_hidden_state_revealed") is not False:
        raise JointPosteriorError("posterior may not reveal the realized hidden state")
    if require_sufficient_support and document.get("support_status") != "sufficient":
        raise JointPosteriorError("joint posterior has insufficient sampled support")

    construction = document.get("construction")
    if not isinstance(construction, Mapping):
        raise JointPosteriorError("posterior lacks construction evidence")
    if construction.get("preserves_joint_team_set_correlations") is not True:
        raise JointPosteriorError("posterior does not preserve joint team/set correlations")
    if construction.get("kind") not in {
        "full-team-generator-rejection-particles",
        "full-team-conditioned-completion-particles",
    }:
        raise JointPosteriorError("unexpected joint posterior construction")
    treatment = construction.get("posterior_treatment")
    expected_treatment = {
        "full-team-generator-rejection-particles": (
            "generator_faithful_joint_empirical"
        ),
        "full-team-conditioned-completion-particles": (
            "practical_joint_completion"
        ),
    }[str(construction["kind"])]
    if treatment != expected_treatment:
        raise JointPosteriorError("posterior treatment does not match construction")

    worlds = document.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise JointPosteriorError("joint posterior has no hidden-team particles")

    seen_worlds: set[str] = set()
    total = 0.0
    team_size: int | None = None
    normalized_worlds: list[dict[str, Any]] = []
    for raw in worlds:
        if not isinstance(raw, Mapping):
            raise JointPosteriorError("posterior particle must be an object")
        world_id = raw.get("world_id")
        weight = raw.get("weight")
        hidden = raw.get("hidden")
        if not isinstance(world_id, str) or not world_id:
            raise JointPosteriorError("posterior particle lacks world_id")
        if world_id in seen_worlds:
            raise JointPosteriorError(f"duplicate posterior particle {world_id}")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise JointPosteriorError(f"{world_id}: weight must be positive and finite")
        if not isinstance(hidden, Mapping):
            raise JointPosteriorError(f"{world_id}: hidden state is missing")

        team = hidden.get("team")
        if not isinstance(team, list) or not team:
            raise JointPosteriorError(f"{world_id}: hidden team is missing")
        if team_size is None:
            team_size = len(team)
        elif len(team) != team_size:
            raise JointPosteriorError("joint particles disagree on team size")

        species: set[str] = set()
        lead_count = 0
        for raw_set in team:
            if not isinstance(raw_set, Mapping):
                raise JointPosteriorError(f"{world_id}: team member is not an object")
            missing = _REQUIRED_SET_FIELDS - set(raw_set)
            if missing:
                raise JointPosteriorError(
                    f"{world_id}: team member lacks {sorted(missing)!r}"
                )
            species_id = raw_set.get("species")
            if not isinstance(species_id, str) or not species_id:
                raise JointPosteriorError(f"{world_id}: team member lacks species")
            if species_id in species:
                raise JointPosteriorError(
                    f"{world_id}: duplicate species {species_id!r} in one particle"
                )
            species.add(species_id)

            moves = raw_set.get("moves")
            if (
                not isinstance(moves, list)
                or not moves
                or any(not isinstance(move, str) or not move for move in moves)
                or len(set(moves)) != len(moves)
            ):
                raise JointPosteriorError(
                    f"{world_id}/{species_id}: moves must be unique strings"
                )
            if raw_set.get("was_lead") is True:
                lead_count += 1

        if lead_count != 1:
            raise JointPosteriorError(
                f"{world_id}: expected exactly one generated lead, got {lead_count}"
            )

        seen_worlds.add(world_id)
        total += float(weight)
        normalized_worlds.append(copy.deepcopy(dict(raw)))

    if abs(total - 1.0) > 1e-9:
        raise JointPosteriorError(f"posterior weights sum to {total}, not 1")

    public_evidence = document.get("public_evidence")
    if not isinstance(public_evidence, Mapping):
        raise JointPosteriorError("posterior lacks public-evidence certificate")
    revealed = public_evidence.get("revealed")
    if not isinstance(revealed, list) or not revealed:
        raise JointPosteriorError("posterior public evidence lacks revealed opponents")

    public_team_size = public_evidence.get("opponent_team_size")
    if public_team_size is not None:
        if not isinstance(public_team_size, int) or isinstance(public_team_size, bool):
            raise JointPosteriorError("public opponent team size must be an integer")
        if team_size != public_team_size:
            raise JointPosteriorError(
                f"posterior team size {team_size} contradicts public size {public_team_size}"
            )

    for row in revealed:
        if not isinstance(row, Mapping) or not isinstance(row.get("species"), str):
            raise JointPosteriorError("malformed revealed-opponent evidence")
        required_species = str(row["species"])
        for world in normalized_worlds:
            members = {
                str(member["species"]): member
                for member in world["hidden"]["team"]
            }
            candidate = members.get(required_species)
            if candidate is None:
                raise JointPosteriorError(
                    f"{world['world_id']}: missing revealed species {required_species}"
                )
            moves = row.get("moves")
            if isinstance(moves, list) and not set(map(str, moves)).issubset(
                set(map(str, candidate["moves"]))
            ):
                raise JointPosteriorError(
                    f"{world['world_id']}: particle contradicts revealed moves"
                )
            if row.get("was_lead") is True and candidate.get("was_lead") is not True:
                raise JointPosteriorError(
                    f"{world['world_id']}: particle contradicts observed lead"
                )
            for evidence_key, set_key in (
                ("ability", "ability"),
                ("item", "item"),
                ("tera_type", "tera_type"),
                ("level", "level"),
                ("gender", "gender"),
            ):
                observed = row.get(evidence_key)
                if observed is not None and candidate.get(set_key) != observed:
                    raise JointPosteriorError(
                        f"{world['world_id']}: particle contradicts revealed {evidence_key}"
                    )

    return dict(document)


def evaluator_posterior(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return only evaluator-safe posterior fields, preserving atomic team particles."""

    checked = validate_joint_posterior(document)
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "source_fixture_id": checked.get("source_fixture_id"),
        "showdown_commit": checked.get("showdown_commit"),
        "public_evidence_sha256": checked.get("public_evidence_sha256"),
        "posterior_sha256": checked.get("posterior_sha256"),
        "worlds": [
            {
                "world_id": world["world_id"],
                "weight": float(world["weight"]),
                "hidden": copy.deepcopy(world["hidden"]),
            }
            for world in checked["worlds"]
        ],
    }


def posterior_validity_record(document: Mapping[str, Any]) -> dict[str, Any]:
    """Expose support quality as a first-class result for a joint posterior."""

    checked = validate_joint_posterior(document, require_sufficient_support=False)
    diagnostics = posterior_diagnostics(checked)
    construction = checked["construction"]
    return {
        "schema": "azelficoast.posterior-validity",
        "schema_version": 1,
        "source_fixture_id": checked.get("source_fixture_id"),
        "posterior_sha256": checked.get("posterior_sha256"),
        "public_evidence_sha256": checked.get("public_evidence_sha256"),
        "support_status": checked.get("support_status"),
        "posterior_treatment": construction.get("posterior_treatment"),
        "support": diagnostics,
        "sampling": {
            key: construction.get(key)
            for key in (
                "attempted_team_count",
                "accepted_team_count",
                "unique_particle_count",
                "acceptance_rate",
                "effective_sample_size",
                "generation_error_count",
                "proposal_mode",
            )
            if key in construction
        },
        "conditioning_scope": list(construction.get("conditioning_scope", [])),
        "unmodeled_dynamic_evidence": list(
            construction.get("dynamic_battle_evidence_not_yet_likelihood_weighted", [])
        ),
        "proposal_caveats": list(construction.get("proposal_caveats", [])),
        "claim": (
            "This record describes finite sampled support and concentration; it does "
            "not establish that omitted posterior mass is negligible."
        ),
    }
