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

from azelficoast.belief_evaluator import build_evaluator_input
from azelficoast.corpus import DecisionFixture
from azelficoast.decision_relevance import (
    DecisionRelevanceError,
    analyze_quotiented_oracle,
)
from azelficoast.real_belief_trace import BeliefTraceError
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


def opponent_move_from_protocol(fixture: DecisionFixture) -> str | None:
    """Return the latest observed move belonging to the current opponent active.

    A move made before an opponent switch must not become the bounded response
    policy for the newly active Pokémon. Keep move history species-bound and
    return only evidence for the species that is active in the frozen state.
    """

    opponent = _to_id(fixture.state.get("opponent"))
    opponent_active = fixture.state.get("opponent_active")
    if not isinstance(opponent_active, Mapping):
        return None
    current_species = _to_id(opponent_active.get("species"))
    if not current_species:
        return None

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

    active_species: str | None = None
    last_move_by_species: dict[str, str] = {}
    for batch in fixture.protocol_prefix:
        for message in batch:
            if len(message) < 4 or message[0] != "":
                continue
            actor = str(message[2])
            if not actor.startswith(opponent_side):
                continue

            if message[1] in {"switch", "drag", "replace"}:
                active_species = str(message[3]).split(",", 1)[0].strip()
                continue

            if message[1] != "move":
                continue

            observed_species = active_species
            if observed_species is None and ":" in actor:
                observed_species = actor.split(":", 1)[1].strip()
            species_id = _to_id(observed_species)
            if species_id:
                last_move_by_species[species_id] = str(message[3])

    return last_move_by_species.get(current_species)


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


def _own_active_tera_type(fixture: DecisionFixture) -> str | None:
    active = fixture.state.get("active")
    if not isinstance(active, Mapping):
        return None

    current = active.get("tera_type")
    if isinstance(current, str) and current:
        return current

    species = _to_id(active.get("species"))
    if not species:
        return None

    recovered: str | None = None
    for batch in fixture.protocol_prefix:
        for message in batch:
            if len(message) < 3 or message[0] != "" or message[1] != "request":
                continue
            try:
                request = json.loads(message[2])
            except json.JSONDecodeError:
                continue
            if not isinstance(request, Mapping):
                continue

            request_active = request.get("active")
            side = request.get("side")
            if (
                not isinstance(request_active, Sequence)
                or isinstance(request_active, (str, bytes))
                or not request_active
                or not isinstance(request_active[0], Mapping)
                or not isinstance(side, Mapping)
            ):
                continue

            tera = request_active[0].get("canTerastallize")
            if not isinstance(tera, str) or not tera:
                continue

            pokemon = side.get("pokemon")
            if (
                not isinstance(pokemon, Sequence)
                or isinstance(pokemon, (str, bytes))
            ):
                continue
            active_view = next(
                (
                    view
                    for view in pokemon
                    if isinstance(view, Mapping) and view.get("active") is True
                ),
                None,
            )
            if active_view is None:
                continue

            details = active_view.get("details")
            request_species = (
                _to_id(str(details).split(",", 1)[0])
                if isinstance(details, str)
                else ""
            )
            if request_species == species:
                recovered = tera

    return recovered


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

    last_move = opponent_move_from_protocol(fixture)
    if last_move is None:
        return None, "opponent-side-or-last-move-unresolved"
    plausible_items = _choice_items_for_move(last_move)
    if plausible_items is None:
        return None, "last-opponent-move-not-fixed-damage-category"

    tera_type = _own_active_tera_type(fixture)
    if tera_type is None:
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
        trace, certificate = analyze_quotiented_oracle(oracle)
    except (
        BeliefTraceError,
        DecisionRelevanceError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
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
            "source_world_count": certificate.get("worlds_in"),
            "decision_class_count": certificate.get("classes_out"),
            "decision_relevant_hidden_fields": certificate.get("decision_fields"),
            "decision_world_reduction": certificate.get("world_reduction"),
            "decision_reduction_fraction": certificate.get("reduction_fraction"),
            "belief_branching_required": certificate.get(
                "belief_branching_required"
            ),
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


def learned_route_result(
    *,
    fixture: DecisionFixture,
    posterior: Mapping[str, Any],
    evaluator: Any,
    search_gate: Any,
) -> LiveDecisionResult:
    """Decide whether a learned public-belief prediction may bypass exact search."""

    identity = getattr(evaluator, "identity", {})
    gate_record = (
        search_gate.as_record()
        if callable(getattr(search_gate, "as_record", None))
        else {"kind": type(search_gate).__name__}
    )
    common = {
        "evaluator": dict(identity) if isinstance(identity, Mapping) else {},
        "search_gate": gate_record,
    }
    try:
        spec = getattr(evaluator, "spec")
        inputs = build_evaluator_input(
            public_state=fixture.state,
            posterior=posterior,
            legal_actions=fixture.legal_actions,
            spec=spec,
        )
        prediction = evaluator.predict(inputs)
        if prediction.selected_action not in set(fixture.legal_actions):
            raise LiveBeliefPolicyError("learned evaluator returned a nonlegal action")
        should_search = bool(search_gate.should_search(prediction))
    except Exception as error:
        return LiveDecisionResult(
            action=None,
            status="search",
            reason="learned-evaluator-error",
            diagnostics={
                **common,
                "learned_route": "search-after-evaluator-error",
                "learned_evaluator_error": {
                    "type": type(error).__name__,
                    "error": str(error)[-1000:],
                },
            },
        )

    routing = {
        **common,
        "learned_prediction": prediction.as_record(),
    }
    if should_search:
        return LiveDecisionResult(
            action=None,
            status="search",
            reason="learned-policy-uncertain",
            diagnostics={
                **routing,
                "learned_route": "exact-public-belief-search",
            },
        )

    return LiveDecisionResult(
        action=prediction.selected_action,
        status="selected",
        reason="learned-public-belief",
        diagnostics={
            **routing,
            "learned_route": "direct-policy",
        },
    )


def selective_belief_result(
    *,
    fixture: DecisionFixture,
    oracle: Mapping[str, Any],
    evaluator: Any,
    search_gate: Any,
) -> LiveDecisionResult:
    """Evaluate routing against an already-built oracle, primarily for offline evidence."""

    worlds = oracle.get("worlds")
    posterior = {
        "conditioned_on_public_history": True,
        "realized_hidden_state_revealed": False,
        "worlds": copy.deepcopy(worlds) if isinstance(worlds, list) else [],
    }
    route = learned_route_result(
        fixture=fixture,
        posterior=posterior,
        evaluator=evaluator,
        search_gate=search_gate,
    )
    if route.action is not None:
        return route

    exact = public_belief_result(oracle, fixture.legal_actions)
    return LiveDecisionResult(
        action=exact.action,
        status=exact.status,
        reason=exact.reason,
        diagnostics={
            **dict(exact.diagnostics),
            **dict(route.diagnostics),
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
        learned_evaluator: Any | None = None,
        search_gate: Any | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("belief timeout must be positive")
        if (learned_evaluator is None) != (search_gate is None):
            raise ValueError("learned_evaluator and search_gate must be provided together")
        self.showdown_root = Path(showdown_root)
        self.timeout_seconds = float(timeout_seconds)
        self.learned_evaluator = learned_evaluator
        self.search_gate = search_gate
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

    def _probe_document(
        self,
        source: Mapping[str, Any],
        *,
        posterior_only: bool,
    ) -> Mapping[str, Any]:
        script = Path(__file__).resolve().parents[2] / "scripts" / "probe_real_belief_trace.cjs"
        with tempfile.TemporaryDirectory(prefix="azelficoast-live-belief-") as temp_dir:
            source_path = Path(temp_dir) / "source.json"
            source_path.write_text(
                json.dumps(source, sort_keys=True),
                encoding="utf-8",
            )
            command = ["node", str(script), str(self.showdown_root), str(source_path)]
            if posterior_only:
                command.append("--posterior-only")
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        document = json.loads(completed.stdout)
        if not isinstance(document, Mapping):
            raise LiveBeliefPolicyError("probe output is not an object")
        return document

    def _probe(self, source: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._probe_document(source, posterior_only=False)

    def _probe_posterior(self, source: Mapping[str, Any]) -> Mapping[str, Any]:
        document = self._probe_document(source, posterior_only=True)
        if (
            document.get("schema") != "azelficoast.live-belief-posterior"
            or document.get("schema_version") != 1
        ):
            raise LiveBeliefPolicyError("unexpected posterior-only probe schema")
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

        route: LiveDecisionResult | None = None
        if self.learned_evaluator is not None:
            try:
                posterior = self._probe_posterior(source)
                if posterior.get("source_fixture_id") != fixture.fixture_id:
                    raise LiveBeliefPolicyError("posterior fixture identity mismatch")
                if posterior.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
                    raise LiveBeliefPolicyError("posterior Showdown revision mismatch")
                if posterior.get("legal_actions") != list(fixture.legal_actions):
                    raise LiveBeliefPolicyError("posterior legal actions drifted")
                route = learned_route_result(
                    fixture=fixture,
                    posterior=posterior,
                    evaluator=self.learned_evaluator,
                    search_gate=self.search_gate,
                )
                if route.action is not None:
                    return route
            except Exception as error:
                route = LiveDecisionResult(
                    action=None,
                    status="search",
                    reason="learned-posterior-probe-error",
                    diagnostics={
                        "learned_route": "search-after-posterior-probe-error",
                        "learned_posterior_error": {
                            "type": type(error).__name__,
                            "error": str(error)[-1000:],
                        },
                    },
                )

        try:
            oracle = self._probe(source)
        except subprocess.TimeoutExpired:
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="belief-search-timeout",
                diagnostics={
                    "timeout_seconds": self.timeout_seconds,
                    **(dict(route.diagnostics) if route is not None else {}),
                },
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
                diagnostics={
                    "error": detail[-1000:],
                    **(dict(route.diagnostics) if route is not None else {}),
                },
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

        exact = public_belief_result(oracle, fixture.legal_actions)
        if route is None:
            return exact
        return LiveDecisionResult(
            action=exact.action,
            status=exact.status,
            reason=exact.reason,
            diagnostics={
                **dict(exact.diagnostics),
                **dict(route.diagnostics),
            },
        )
