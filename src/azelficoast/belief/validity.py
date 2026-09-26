"""Offline posterior-validity and robustness diagnostics.

These helpers deliberately keep realized hidden state out of search. Realized-state
information may be supplied only to post-hoc scoring functions after a decision has
been made, so posterior adequacy can be measured without becoming a search side
channel.
"""

from __future__ import annotations

import copy
import json
import math
from typing import Any, Mapping, Sequence, TypedDict


class PosteriorValidityError(ValueError):
    """Raised when posterior-validity evidence is malformed."""


class PosteriorWorld(TypedDict):
    world_id: str
    weight: float
    hidden: Mapping[str, Any]


class PosteriorDiagnostics(TypedDict):
    support_size: int
    entropy_bits: float
    effective_sample_size: float
    minimum_mass: float
    maximum_mass: float


class RealizedSupportScore(TypedDict):
    realized_world_id: str
    realized_state_in_support: bool
    assigned_mass: float
    brier_score: float
    log_loss_if_covered: float | None
    support_size: int


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def normalized_worlds(posterior: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate and normalize one finite-support posterior."""

    raw_worlds = posterior.get("worlds")
    if not isinstance(raw_worlds, list) or not raw_worlds:
        raise PosteriorValidityError("posterior must contain non-empty world support")

    worlds: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0.0
    for raw in raw_worlds:
        if not isinstance(raw, Mapping):
            raise PosteriorValidityError("posterior world must be an object")
        world_id = raw.get("world_id")
        hidden = raw.get("hidden")
        weight = raw.get("weight")
        if not isinstance(world_id, str) or not world_id:
            raise PosteriorValidityError("posterior world lacks world_id")
        if world_id in seen:
            raise PosteriorValidityError(f"duplicate posterior world {world_id}")
        if not isinstance(hidden, Mapping):
            raise PosteriorValidityError(f"{world_id}: hidden state must be an object")
        if (
            not isinstance(weight, (int, float))
            or isinstance(weight, bool)
            or not math.isfinite(float(weight))
            or float(weight) <= 0.0
        ):
            raise PosteriorValidityError(
                f"{world_id}: posterior weight must be positive and finite"
            )
        seen.add(world_id)
        total += float(weight)
        worlds.append(copy.deepcopy(dict(raw)))

    if not math.isfinite(total) or total <= 0.0:
        raise PosteriorValidityError("posterior has no finite positive mass")
    for world in worlds:
        world["weight"] = float(world["weight"]) / total
    return worlds


def posterior_diagnostics(posterior: Mapping[str, Any]) -> PosteriorDiagnostics:
    """Return support concentration diagnostics that should accompany search results."""

    worlds = normalized_worlds(posterior)
    weights = [float(world["weight"]) for world in worlds]
    entropy = -sum(weight * math.log2(weight) for weight in weights)
    ess = 1.0 / sum(weight * weight for weight in weights)
    return {
        "support_size": len(worlds),
        "entropy_bits": entropy,
        "effective_sample_size": ess,
        "minimum_mass": min(weights),
        "maximum_mass": max(weights),
    }


def score_realized_support(
    posterior: Mapping[str, Any],
    *,
    realized_world_id: str,
) -> RealizedSupportScore:
    """Score support/calibration after the hidden world becomes observable.

    This is an offline evaluation primitive. The realized world identifier must never
    be fed back into the decision-time posterior or evaluator.
    """

    if not realized_world_id:
        raise PosteriorValidityError("realized_world_id must be non-empty")
    worlds = normalized_worlds(posterior)
    masses = {str(world["world_id"]): float(world["weight"]) for world in worlds}
    assigned = masses.get(realized_world_id, 0.0)
    brier = 1.0 + sum(probability * probability for probability in masses.values())
    brier -= 2.0 * assigned
    return {
        "realized_world_id": realized_world_id,
        "realized_state_in_support": realized_world_id in masses,
        "assigned_mass": assigned,
        "brier_score": brier,
        "log_loss_if_covered": (-math.log(assigned) if assigned > 0.0 else None),
        "support_size": len(worlds),
    }


def aggregate_realized_support(
    scores: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate post-hoc support coverage without hiding omitted realized worlds."""

    if not scores:
        raise PosteriorValidityError("at least one realized-support score is required")

    normalized: list[RealizedSupportScore] = []
    for raw in scores:
        covered = raw.get("realized_state_in_support")
        mass = raw.get("assigned_mass")
        brier = raw.get("brier_score")
        support_size = raw.get("support_size")
        realized_world_id = raw.get("realized_world_id")
        log_loss = raw.get("log_loss_if_covered")
        if not isinstance(covered, bool):
            raise PosteriorValidityError("support score lacks coverage boolean")
        if not isinstance(realized_world_id, str) or not realized_world_id:
            raise PosteriorValidityError("support score lacks realized_world_id")
        if (
            not isinstance(mass, (int, float))
            or isinstance(mass, bool)
            or not math.isfinite(float(mass))
            or not 0.0 <= float(mass) <= 1.0
        ):
            raise PosteriorValidityError("support score has invalid assigned mass")
        if (
            not isinstance(brier, (int, float))
            or isinstance(brier, bool)
            or not math.isfinite(float(brier))
            or float(brier) < 0.0
        ):
            raise PosteriorValidityError("support score has invalid Brier score")
        if not isinstance(support_size, int) or isinstance(support_size, bool) or support_size < 1:
            raise PosteriorValidityError("support score has invalid support size")
        normalized_log_loss: float | None = None
        if covered:
            if (
                not isinstance(log_loss, (int, float))
                or isinstance(log_loss, bool)
                or not math.isfinite(float(log_loss))
                or float(log_loss) < 0.0
            ):
                raise PosteriorValidityError("covered score must contain finite log loss")
            normalized_log_loss = float(log_loss)
        elif log_loss is not None:
            raise PosteriorValidityError("omitted realized state must not report finite log loss")
        normalized.append(
            {
                "realized_world_id": realized_world_id,
                "realized_state_in_support": covered,
                "assigned_mass": float(mass),
                "brier_score": float(brier),
                "log_loss_if_covered": normalized_log_loss,
                "support_size": support_size,
            }
        )

    covered_rows = [row for row in normalized if row["realized_state_in_support"]]
    covered_log_losses = [
        log_loss
        for row in covered_rows
        if (log_loss := row["log_loss_if_covered"]) is not None
    ]
    if len(covered_log_losses) != len(covered_rows):
        raise PosteriorValidityError("covered support score is missing log loss")
    return {
        "case_count": len(normalized),
        "realized_support_coverage_rate": len(covered_rows) / len(normalized),
        "omitted_realized_state_count": len(normalized) - len(covered_rows),
        "mean_assigned_mass": sum(row["assigned_mass"] for row in normalized) / len(normalized),
        "mean_brier_score": sum(row["brier_score"] for row in normalized) / len(normalized),
        "mean_log_loss_if_covered": (
            sum(covered_log_losses) / len(covered_log_losses)
            if covered_log_losses
            else None
        ),
        "mean_support_size": sum(row["support_size"] for row in normalized) / len(normalized),
    }


def power_reweight_posterior(
    posterior: Mapping[str, Any],
    *,
    exponent: float,
    treatment: str,
) -> dict[str, Any]:
    """Stress prior concentration while preserving exactly the same support.

    exponent < 1 flattens the prior; exponent == 0 is exactly uniform;
    exponent > 1 sharpens it.
    """

    if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
        raise PosteriorValidityError("reweight exponent must be numeric")
    exponent = float(exponent)
    if not math.isfinite(exponent) or exponent < 0.0:
        raise PosteriorValidityError("reweight exponent must be finite and non-negative")
    if not treatment:
        raise PosteriorValidityError("robustness treatment name must be non-empty")

    worlds = normalized_worlds(posterior)
    transformed = [float(world["weight"]) ** exponent for world in worlds]
    total = sum(transformed)
    for world, weight in zip(worlds, transformed, strict=True):
        world["weight"] = weight / total

    result = copy.deepcopy(dict(posterior))
    result["worlds"] = worlds
    result["treatment"] = treatment
    result["robustness_treatment"] = {
        "kind": "power-reweight",
        "name": treatment,
        "exponent": exponent,
        "support_changed": False,
    }
    return result


def widen_posterior_support(
    posterior: Mapping[str, Any],
    *,
    alternative: Mapping[str, Any],
    alternative_mass: float,
    treatment: str = "widened_support",
) -> dict[str, Any]:
    """Mix in an alternative support source for deliberate support sensitivity.

    A matching TransitionProgram must be generated and verified for the resulting union
    before this posterior may be used by search.
    """

    if (
        not isinstance(alternative_mass, (int, float))
        or isinstance(alternative_mass, bool)
        or not math.isfinite(float(alternative_mass))
        or not 0.0 < float(alternative_mass) < 1.0
    ):
        raise PosteriorValidityError("alternative_mass must be finite and within (0, 1)")
    if not treatment:
        raise PosteriorValidityError("robustness treatment name must be non-empty")

    base_worlds = normalized_worlds(posterior)
    alt_worlds = normalized_worlds(alternative)
    combined: dict[str, dict[str, Any]] = {}
    mass: dict[str, float] = {}

    def add(source: Sequence[Mapping[str, Any]], scale: float) -> None:
        for raw in source:
            world_id = str(raw["world_id"])
            existing = combined.get(world_id)
            if existing is not None:
                if _canonical(existing.get("hidden")) != _canonical(raw.get("hidden")):
                    raise PosteriorValidityError(
                        f"{world_id}: alternative support reuses id for another hidden world"
                    )
            else:
                combined[world_id] = copy.deepcopy(dict(raw))
                mass[world_id] = 0.0
            mass[world_id] += scale * float(raw["weight"])

    alt_mass = float(alternative_mass)
    add(base_worlds, 1.0 - alt_mass)
    add(alt_worlds, alt_mass)

    worlds = []
    for world_id in sorted(combined):
        row = combined[world_id]
        row["weight"] = mass[world_id]
        worlds.append(row)

    result = copy.deepcopy(dict(posterior))
    result["worlds"] = worlds
    result["treatment"] = treatment
    result["robustness_treatment"] = {
        "kind": "support-mixture",
        "name": treatment,
        "alternative_mass": alt_mass,
        "support_changed": set(combined) != {str(row["world_id"]) for row in base_worlds},
        "requires_new_verified_transition_program": True,
    }
    return result
