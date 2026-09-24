"""Live adapter from poke-env information states to bounded public-belief search."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from poke_env.data import GenData

from azelficoast.corpus import DecisionFixture
from azelficoast.real_belief_trace import BeliefTraceError, analyze_oracle
from azelficoast.showdown_damage_corpus import PINNED_SHOWDOWN_COMMIT

PROBE_SCHEMA = "azelficoast.real-belief-source-fixture"
PROBE_SCHEMA_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class LiveDecisionResult:
    """One bounded live-policy attempt.

    action is null whenever the bounded public-belief model cannot safely
    handle the current information state. The caller then retains authority to
    use a fallback policy.
    """

    action: str | None
    status: str
    reason: str
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_trace(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "action": self.action,
            "diagnostics": dict(self.diagnostics),
        }


class LiveBeliefPolicyError(ValueError):
    """Raised when the configured live belief engine is internally invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def live_fixture(
    state: Mapping[str, Any],
    protocol_prefix: Sequence[Sequence[Sequence[str]]],
) -> DecisionFixture:
    """Freeze the current live information state using corpus-compatible identity."""

    frozen_state = copy.deepcopy(dict(state))
    frozen_state.pop("battle_tag", None)
    frozen_protocol = tuple(
        tuple(tuple(str(field) for field in message) for message in batch)
        for batch in protocol_prefix
    )
    material = {
        "state": frozen_state,
        "protocol_prefix": [
            [list(message) for message in batch] for batch in frozen_protocol
        ],
    }
    fixture_id = hashlib.sha256(_canonical_json(material).encode("utf-8")).hexdigest()
    return DecisionFixture(
        fixture_id=fixture_id,
        state=frozen_state,
        protocol_prefix=frozen_protocol,
        control_decisions=(),
    )


def _to_id(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _opponent_move_from_protocol(fixture: DecisionFixture) -> str | None:
    """Return the latest publicly observed opponent move for either Showdown side."""

    opponent = _to_id(fixture.state.get("opponent"))
    opponent_side: str | None = None
    for batch in fixture.protocol_prefix:
        for message in batch:
            if (
                len(message) >= 4
                and message[0] == ""
                and message[1] == "player"
                and message[2] in {"p1", "p2"}
                and _to_id(message[3]) == opponent
            ):
                opponent_side = message[2]
    if opponent_side is None:
        return None

    last_move: str | None = None
    for batch in fixture.protocol_prefix:
        for message in batch:
            if (
                len(message) >= 4
                and message[0] == ""
                and message[1] == "move"
                and str(message[2]).startswith(opponent_side)
            ):
                last_move = str(message[3])
    return last_move


def _choice_items_for_move(move_name: str) -> tuple[str, str] | None:
    move = GenData.from_gen(9).moves.get(_to_id(move_name))
    if not isinstance(move, Mapping):
        return None
    category = move.get("category")
    if category == "Physical":
        return ("Choice Band", "Choice Scarf")
    if category == "Special":
        return ("Choice Scarf", "Choice Specs")
    return None


def build_probe_source(fixture: DecisionFixture) -> tuple[dict[str, Any] | None, str]:
    """Admit only the hidden-Choice slice supported by current live evidence."""

    opponent = fixture.state.get("opponent_active")
    active = fixture.state.get("active")
    if not isinstance(opponent, Mapping) or not isinstance(active, Mapping):
        return None, "missing-active-state"
    if opponent.get("item") not in (None, GenData.UNKNOWN_ITEM):
        return None, "opponent-item-known"
    if not isinstance(opponent.get("species"), str) or not isinstance(opponent.get("level"), int):
        return None, "opponent-generator-identity-incomplete"
    if not fixture.legal_actions:
        return None, "no-legal-actions"

    last_move = _opponent_move_from_protocol(fixture)
    if last_move is None:
        return None, "opponent-side-or-last-move-unresolved"
    plausible_items = _choice_items_for_move(last_move)
    if plausible_items is None:
        return None, "last-opponent-move-not-fixed-damage-category"

    tera_type = active.get("tera_type")
    if not isinstance(tera_type, str) or not tera_type:
        return None, "own-active-tera-type-unavailable"

    team = fixture.state.get("team")
    if not isinstance(team, Mapping) or not team:
        return None, "own-team-unavailable"

    source = {
        "schema": PROBE_SCHEMA,
        "schema_version": PROBE_SCHEMA_VERSION,
        "fixture_id": fixture.fixture_id,
        "showdown_commit": PINNED_SHOWDOWN_COMMIT,
        "fixture": fixture.as_record(),
        "plausible_items": list(plausible_items),
        "opponent_response_move": last_move,
        "own_active_tera_type": tera_type,
        "source_projection": "live hidden-Choice bounded public-belief decision",
    }
    return source, "admitted"


def public_belief_result(
    oracle: Mapping[str, Any],
    legal_actions: Sequence[str],
) -> LiveDecisionResult:
    """Analyze one exact mechanics oracle and return its legal public-belief action."""

    try:
        trace = analyze_oracle(oracle)
    except (BeliefTraceError, KeyError, TypeError, ValueError) as error:
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="oracle-analysis-failed",
            diagnostics={"error": str(error)},
        )

    public = trace.get("public_belief")
    action = public.get("chosen_action") if isinstance(public, Mapping) else None
    if not isinstance(action, str) or action not in set(legal_actions):
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="analyzer-returned-nonlegal-action",
            diagnostics={"analyzed_action": action},
        )

    determinization = trace.get("determinization")
    return LiveDecisionResult(
        action=action,
        status="selected",
        reason="bounded-public-belief",
        diagnostics={
            "fixture_id": trace.get("source_fixture_id"),
            "showdown_commit": trace.get("showdown_commit"),
            "world_count": trace.get("world_count"),
            "legal_action_count": trace.get("legal_action_count"),
            "strategy_fusion_observation_count": trace.get(
                "strategy_fusion_observation_count"
            ),
            "policy_disagreement": trace.get("policy_disagreement"),
            "determinization_action": (
                determinization.get("chosen_action")
                if isinstance(determinization, Mapping)
                else None
            ),
            "public_belief_value": public.get("value"),
        },
    )


class PinnedShowdownBeliefPolicy:
    """Run the bounded live public-belief policy through pinned Pokemon Showdown."""

    name = "pinned-showdown-public-belief"

    def __init__(
        self,
        showdown_root: str | Path,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("belief timeout must be positive")
        self.showdown_root = Path(showdown_root)
        self.timeout_seconds = float(timeout_seconds)
        self._configuration_error = self._validate_showdown_root()

    def _validate_showdown_root(self) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.showdown_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=min(self.timeout_seconds, 5.0),
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            return f"cannot-read-showdown-revision: {error}"
        actual = completed.stdout.strip()
        if actual != PINNED_SHOWDOWN_COMMIT:
            return (
                "showdown-revision-mismatch: "
                f"expected {PINNED_SHOWDOWN_COMMIT}, got {actual or '<empty>'}"
            )
        if not (self.showdown_root / "dist" / "sim" / "battle.js").is_file():
            return "showdown-build-missing"
        return None

    @property
    def configured(self) -> bool:
        return self._configuration_error is None

    def _probe(self, source: Mapping[str, Any]) -> Mapping[str, Any]:
        script = Path(__file__).resolve().parents[2] / "scripts" / "probe_real_belief_trace.cjs"
        with tempfile.TemporaryDirectory(prefix="azelficoast-live-belief-") as temp_dir:
            source_path = Path(temp_dir) / "source.json"
            source_path.write_text(
                json.dumps(source, sort_keys=True),
                encoding="utf-8",
            )
            completed = subprocess.run(
                ["node", str(script), str(self.showdown_root), str(source_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        document = json.loads(completed.stdout)
        if not isinstance(document, Mapping):
            raise LiveBeliefPolicyError("probe output is not an object")
        return document

    def choose(self, fixture: DecisionFixture) -> LiveDecisionResult:
        if self._configuration_error is not None:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="belief-engine-unavailable",
                diagnostics={"error": self._configuration_error},
            )

        source, admission = build_probe_source(fixture)
        if source is None:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason=admission,
            )

        try:
            oracle = self._probe(source)
        except subprocess.TimeoutExpired:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="belief-search-timeout",
                diagnostics={"timeout_seconds": self.timeout_seconds},
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            LiveBeliefPolicyError,
        ) as error:
            detail = str(error)
            if isinstance(error, subprocess.CalledProcessError):
                detail = (error.stderr or error.stdout or detail).strip()
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="belief-probe-failed",
                diagnostics={"error": detail[-1000:]},
            )

        if oracle.get("source_fixture_id") != fixture.fixture_id:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="oracle-fixture-mismatch",
                diagnostics={"oracle_fixture_id": oracle.get("source_fixture_id")},
            )
        if oracle.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="oracle-revision-mismatch",
                diagnostics={"showdown_commit": oracle.get("showdown_commit")},
            )

        return public_belief_result(oracle, fixture.legal_actions)
