"""Posterior extended statistics for Pokémon hidden-team correlations.

These records summarize the already-authorized joint posterior for planning. They do not
factorize the posterior and do not add belief authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from azelficoast.belief.joint_posterior import validate_joint_posterior
from azelficoast.core.statistics import (
    CategoricalDependency,
    analyze_categorical_dependency,
)

POSTERIOR_CORRELATION_SCHEMA = "azelficoast.posterior-correlation-statistics"
POSTERIOR_CORRELATION_SCHEMA_VERSION = 1

_PAIR_FIELDS = (
    ("species", "item"),
    ("species", "ability"),
    ("species", "role"),
    ("species", "move_set"),
)


@dataclass(frozen=True, slots=True)
class PosteriorCorrelationProfile:
    """Coarse correlation regime plus exact diagnostic dependency metrics."""

    signature: str
    world_count: int
    team_size: int
    posterior_treatment: str
    dependencies: tuple[CategoricalDependency, ...]

    def as_record(self) -> dict[str, Any]:
        return {
            "schema": POSTERIOR_CORRELATION_SCHEMA,
            "schema_version": POSTERIOR_CORRELATION_SCHEMA_VERSION,
            "signature": self.signature,
            "world_count": self.world_count,
            "team_size": self.team_size,
            "posterior_treatment": self.posterior_treatment,
            "dependencies": [
                dependency.as_record() for dependency in self.dependencies
            ],
            "claim": (
                "These are advisory multi-column statistics over the admitted finite "
                "posterior; they do not alter posterior semantics or mechanics authority."
            ),
        }


def _bucket(value: float, *, low: float, high: float) -> str:
    if value < low:
        return "low"
    if value < high:
        return "medium"
    return "high"


def _field_value(member: Mapping[str, Any], field: str) -> str:
    if field == "move_set":
        moves = member.get("moves")
        if not isinstance(moves, list) or not moves:
            raise ValueError("posterior member lacks move set")
        return "|".join(sorted(str(move) for move in moves))
    value = member.get(field)
    if value is None:
        raise ValueError(f"posterior member lacks {field}")
    text = str(value)
    if not text:
        raise ValueError(f"posterior member has empty {field}")
    return text


def posterior_correlation_profile(
    document: Mapping[str, Any],
) -> PosteriorCorrelationProfile:
    """Measure coarse extended statistics without breaking joint particles apart."""

    checked = validate_joint_posterior(document)
    worlds = checked["worlds"]
    team_size = len(worlds[0]["hidden"]["team"])
    dependencies: list[CategoricalDependency] = []
    regime: list[dict[str, str]] = []

    for left_name, right_name in _PAIR_FIELDS:
        samples: list[tuple[str, str, float]] = []
        for world in worlds:
            weight = float(world["weight"])
            for member in world["hidden"]["team"]:
                samples.append(
                    (
                        _field_value(member, left_name),
                        _field_value(member, right_name),
                        weight,
                    )
                )
        dependency = analyze_categorical_dependency(
            samples,
            left_name=left_name,
            right_name=right_name,
        )
        dependencies.append(dependency)
        regime.append(
            {
                "columns": f"{left_name}:{right_name}",
                "dependence": _bucket(
                    dependency.total_variation_from_independence,
                    low=0.05,
                    high=0.20,
                ),
                "forward_functional": _bucket(
                    dependency.left_predicts_right_accuracy,
                    low=0.50,
                    high=0.80,
                ),
            }
        )

    material = {
        "schema": POSTERIOR_CORRELATION_SCHEMA,
        "schema_version": POSTERIOR_CORRELATION_SCHEMA_VERSION,
        "posterior_treatment": checked["construction"]["posterior_treatment"],
        "team_size": team_size,
        "regime": regime,
    }
    encoded = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    signature = "sha256:" + hashlib.sha256(encoded).hexdigest()

    return PosteriorCorrelationProfile(
        signature=signature,
        world_count=len(worlds),
        team_size=team_size,
        posterior_treatment=str(
            checked["construction"]["posterior_treatment"]
        ),
        dependencies=tuple(dependencies),
    )
