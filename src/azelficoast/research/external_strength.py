"""Frozen external playing-strength contract and settlement."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from azelficoast.core.showdown import PINNED_SHOWDOWN_COMMIT

SCHEMA = "azelficoast.external-playing-strength-contract"
SCHEMA_VERSION = 1
RESULT_SCHEMA = "azelficoast.external-playing-strength-result"
RESULT_SCHEMA_VERSION = 1


class ExternalStrengthContractError(ValueError):
    """Raised when external-strength evidence is incomplete or identity-drifted."""


def _full_sha(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExternalStrengthContractError(f"{field} must be a full lowercase Git SHA")
    return value


def _digest(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ExternalStrengthContractError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def load_contract(path: str | Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ExternalStrengthContractError("external-strength contract must be an object")
    return validate_contract(document)


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != SCHEMA or contract.get("schema_version") != SCHEMA_VERSION:
        raise ExternalStrengthContractError("unexpected external-strength contract schema")
    if contract.get("issue") != 198:
        raise ExternalStrengthContractError("external-strength contract must bind issue #198")
    if contract.get("format") != "gen9randombattle":
        raise ExternalStrengthContractError("first external benchmark must remain Gen 9 Random Battle")
    if contract.get("showdown_revision_source") != "showdown/revision.json":
        raise ExternalStrengthContractError("Showdown revision must remain repository-owned")

    planned = contract.get("planned_battles")
    shards = contract.get("shard_count")
    per_shard = contract.get("battles_per_shard")
    per_direction = contract.get("battles_per_direction_per_shard")
    for value, field in (
        (planned, "planned_battles"),
        (shards, "shard_count"),
        (per_shard, "battles_per_shard"),
        (per_direction, "battles_per_direction_per_shard"),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ExternalStrengthContractError(f"{field} must be a positive integer")
    assert isinstance(planned, int)
    assert isinstance(shards, int)
    assert isinstance(per_shard, int)
    assert isinstance(per_direction, int)
    if planned != 1000 or shards != 10 or per_shard != 100:
        raise ExternalStrengthContractError("headline panel must remain 1,000 battles in ten shards")
    if planned != shards * per_shard or per_shard != 2 * per_direction:
        raise ExternalStrengthContractError("battle count and direction balance do not compose")
    if contract.get("challenge_directions") != [
        "foul_play_challenges",
        "azelficoast_challenges",
    ]:
        raise ExternalStrengthContractError("challenge-direction strata drifted")
    if contract.get("no_optional_stopping") is not True:
        raise ExternalStrengthContractError("external benchmark forbids optional stopping")

    minimum_decisive = contract.get("minimum_decisive_fraction")
    if not isinstance(minimum_decisive, (int, float)) or isinstance(minimum_decisive, bool):
        raise ExternalStrengthContractError("minimum_decisive_fraction must be numeric")
    if float(minimum_decisive) != 0.95:
        raise ExternalStrengthContractError("minimum decisive fraction drifted")

    superiority = contract.get("superiority")
    if not isinstance(superiority, Mapping):
        raise ExternalStrengthContractError("superiority test is missing")
    if (
        superiority.get("null_win_probability") != 0.5
        or superiority.get("alternative") != "greater"
        or superiority.get("max_p_value") != 0.01
    ):
        raise ExternalStrengthContractError("superiority test drifted")

    azelficoast = contract.get("azelficoast")
    if not isinstance(azelficoast, Mapping):
        raise ExternalStrengthContractError("Azelficoast identity is missing")
    source = azelficoast.get("evaluator_source")
    if not isinstance(source, Mapping):
        raise ExternalStrengthContractError("evaluator source is missing")
    if source.get("repository") != "laurajoyhutchins/azelficoast":
        raise ExternalStrengthContractError("evaluator must come from Azelficoast authority")
    _full_sha(source.get("commit"), field="training-state commit")
    _digest(source.get("checkpoint_digest"), field="evaluator checkpoint digest")
    if source.get("promotion_path") != "state/evaluators/current.json":
        raise ExternalStrengthContractError("evaluator promotion authority drifted")
    timeout = azelficoast.get("belief_timeout_seconds")
    margin = azelficoast.get("search_policy_margin")
    if not isinstance(timeout, (int, float)) or float(timeout) <= 0:
        raise ExternalStrengthContractError("belief timeout must be positive")
    if not isinstance(margin, (int, float)) or isinstance(margin, bool):
        raise ExternalStrengthContractError("search policy margin must be numeric")

    opponent = contract.get("opponent")
    if not isinstance(opponent, Mapping) or opponent.get("name") != "foul-play":
        raise ExternalStrengthContractError("first serious external opponent must be Foul Play")
    _full_sha(opponent.get("revision"), field="Foul Play revision")
    if opponent.get("execution_boundary") != "external_process":
        raise ExternalStrengthContractError("Foul Play must remain outside Azelficoast")
    for field, expected in (
        ("search_time_ms", 100),
        ("search_parallelism", 1),
        ("search_threads", 1),
    ):
        if opponent.get(field) != expected:
            raise ExternalStrengthContractError(f"Foul Play {field} drifted")

    evidence = contract.get("evidence")
    if not isinstance(evidence, Mapping) or not all(
        evidence.get(field) is True
        for field in (
            "retain_raw_battles",
            "retain_decision_traces",
            "retain_replays",
            "bind_exact_git_head",
            "bind_evaluator_digest",
            "bind_showdown_revision",
            "bind_opponent_revision",
        )
    ):
        raise ExternalStrengthContractError("external evidence binding is incomplete")
    if contract.get("claim_scope") != "playing_strength_only_unmatched_compute":
        raise ExternalStrengthContractError("benchmark must not imply compute efficiency")
    return dict(contract)


def contract_readiness(contract: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_contract(contract)
    source = checked["azelficoast"]["evaluator_source"]
    opponent = checked["opponent"]
    return {
        "schema": "azelficoast.external-playing-strength-contract-check",
        "schema_version": 1,
        "passed": True,
        "planned_battles": checked["planned_battles"],
        "shard_count": checked["shard_count"],
        "battles_per_shard": checked["battles_per_shard"],
        "evaluator_digest": source["checkpoint_digest"],
        "training_state_commit": source["commit"],
        "opponent_revision": opponent["revision"],
        "showdown_revision": PINNED_SHOWDOWN_COMMIT,
        "claim_scope": checked["claim_scope"],
    }


def exact_binomial_superiority_p_value(wins: int, losses: int) -> float:
    decisive = wins + losses
    if decisive <= 0 or wins < 0 or losses < 0:
        raise ExternalStrengthContractError("decisive win/loss counts must be non-negative")
    return math.fsum(
        math.comb(decisive, index) * (0.5 ** decisive)
        for index in range(wins, decisive + 1)
    )


def _canonical_raw_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in sorted(rows, key=lambda row: (int(row["shard"]), str(row["battle_tag"])))
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def settle_external_panel(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    checked = validate_contract(contract)
    planned = int(checked["planned_battles"])
    if len(rows) != planned:
        raise ExternalStrengthContractError(
            f"external panel has {len(rows)} battles; contract requires exactly {planned}"
        )

    source = checked["azelficoast"]["evaluator_source"]
    opponent = checked["opponent"]
    expected_evaluator = str(source["checkpoint_digest"])
    expected_opponent = str(opponent["revision"])
    tags: set[tuple[int, str]] = set()
    git_heads: set[str] = set()
    outcomes: Counter[str] = Counter()
    directions: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    shard_counts: Counter[int] = Counter()
    shard_directions: Counter[tuple[int, str]] = Counter()

    for row in rows:
        tag = row.get("battle_tag")
        if not isinstance(tag, str) or not tag:
            raise ExternalStrengthContractError("battle row lacks battle_tag")
        if row.get("format") != checked["format"]:
            raise ExternalStrengthContractError(f"{tag}: battle format drifted")
        head = _full_sha(row.get("azelficoast_git_sha"), field=f"{tag} Azelficoast head")
        git_heads.add(head)
        if row.get("evaluator_digest") != expected_evaluator:
            raise ExternalStrengthContractError(f"{tag}: evaluator identity drifted")
        if row.get("showdown_revision") != PINNED_SHOWDOWN_COMMIT:
            raise ExternalStrengthContractError(f"{tag}: Showdown revision drifted")
        if row.get("opponent") != "foul-play" or row.get("opponent_revision") != expected_opponent:
            raise ExternalStrengthContractError(f"{tag}: external opponent identity drifted")

        shard = row.get("shard")
        if not isinstance(shard, int) or isinstance(shard, bool) or not 0 <= shard < int(checked["shard_count"]):
            raise ExternalStrengthContractError(f"{tag}: invalid shard")
        identity = (shard, tag)
        if identity in tags:
            raise ExternalStrengthContractError(
                f"duplicate battle identity {tag} in shard {shard}"
            )
        tags.add(identity)
        direction = row.get("direction")
        if direction not in checked["challenge_directions"]:
            raise ExternalStrengthContractError(f"{tag}: invalid challenge direction")
        role = row.get("player_role")
        if role not in {"p1", "p2"}:
            raise ExternalStrengthContractError(f"{tag}: invalid Azelficoast player role")
        outcome = row.get("outcome")
        if outcome not in {"win", "loss", "tie"}:
            raise ExternalStrengthContractError(f"{tag}: invalid outcome")

        shard_counts[shard] += 1
        shard_directions[(shard, str(direction))] += 1
        directions[str(direction)] += 1
        roles[str(role)] += 1
        outcomes[str(outcome)] += 1

    if len(git_heads) != 1:
        raise ExternalStrengthContractError("panel mixes Azelficoast Git heads")
    per_shard = int(checked["battles_per_shard"])
    per_direction = int(checked["battles_per_direction_per_shard"])
    for shard in range(int(checked["shard_count"])):
        if shard_counts[shard] != per_shard:
            raise ExternalStrengthContractError(f"shard {shard} is incomplete")
        for direction in checked["challenge_directions"]:
            if shard_directions[(shard, str(direction))] != per_direction:
                raise ExternalStrengthContractError(
                    f"shard {shard} challenge-direction balance drifted"
                )
    if roles != Counter({"p1": planned // 2, "p2": planned // 2}):
        raise ExternalStrengthContractError(
            f"actual player-role balance drifted: {dict(roles)}"
        )

    wins = outcomes["win"]
    losses = outcomes["loss"]
    ties = outcomes["tie"]
    decisive = wins + losses
    decisive_fraction = decisive / planned
    p_value = exact_binomial_superiority_p_value(wins, losses)
    max_p = float(checked["superiority"]["max_p_value"])
    minimum_decisive = float(checked["minimum_decisive_fraction"])
    superiority_established = (
        decisive_fraction >= minimum_decisive
        and wins > losses
        and p_value <= max_p
    )
    return {
        "schema": RESULT_SCHEMA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "complete_panel": True,
        "superiority_established": superiority_established,
        "claim_scope": checked["claim_scope"],
        "battle_count": planned,
        "azelficoast_git_sha": next(iter(git_heads)),
        "evaluator_digest": expected_evaluator,
        "showdown_revision": PINNED_SHOWDOWN_COMMIT,
        "opponent": "foul-play",
        "opponent_revision": expected_opponent,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "score_rate": (wins + 0.5 * ties) / planned,
        "decisive_fraction": decisive_fraction,
        "one_sided_exact_binomial_p_value": p_value,
        "max_p_value": max_p,
        "directions": dict(sorted(directions.items())),
        "player_roles": dict(sorted(roles.items())),
        "raw_results_digest": _canonical_raw_digest(rows),
    }
