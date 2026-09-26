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

from azelficoast.belief.evaluator import build_evaluator_input
from azelficoast.live.corpus import DecisionFixture
from azelficoast.live.showdown_probe import (
    PersistentShowdownProbe,
    ShowdownProbeRuntimeError,
)
from azelficoast.core.decision_relevance import DecisionRelevanceError
from azelficoast.research.verification.real_belief_trace import BeliefTraceError, analyze_quotiented_oracle
from azelficoast.core.mechanics import (
    MechanicsContractError,
    VerifiedTransitionProgramSet,
)
from azelficoast.research.contracts import (
    MechanicsIdentity,
    ResearchContractError,
    parse_belief_artifact,
)
from azelficoast.research.verification.showdown_damage_corpus import PINNED_SHOWDOWN_COMMIT
from azelficoast.research.typed_search import (
    TransitionProgramSearchError,
    search_transition_program,
)
from azelficoast.core.program import PROGRAM_SET_SCHEMA, PROGRAM_SET_SCHEMA_VERSION

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
    """Build the broadest live reconstruction source justified by public evidence.

    Posterior reconstruction does not require an opponent-response model. Exact search
    uses a bounded mixture of established strategy archetypes over each hidden world's
    legal moves. An observed move contributes one persistence component rather than
    becoming a deterministic prediction of the opponent's next action.
    """

    opponent = fixture.state.get("opponent_active")
    active = fixture.state.get("active")
    if not isinstance(opponent, Mapping) or not isinstance(active, Mapping):
        return None, "missing-active-state"
    if not isinstance(opponent.get("species"), str) or not isinstance(opponent.get("level"), int):
        return None, "opponent-generator-identity-incomplete"
    if not fixture.legal_actions:
        return None, "no-legal-actions"

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
        "own_active_tera_type": tera_type,
        "source_projection": (
            "live generator-faithful active-set posterior with a separately bounded "
            "opponent-response model"
        ),
    }

    known_item = opponent.get("item")
    if known_item not in (None, GenData.UNKNOWN_ITEM):
        source["known_opponent_item"] = str(known_item)

    last_move = opponent_move_from_protocol(fixture)
    strategies: list[dict[str, object]] = [
        {"kind": "simple-heuristics"},
        {"kind": "dirty-tricks"},
        {"kind": "max-damage"},
        {"kind": "uniform-legal-moves"},
    ]
    if last_move is not None:
        source["opponent_response_move"] = last_move
        strategies.append(
            {
                "kind": "repeat-observed-move",
                "move": last_move,
            }
        )
    source["opponent_policy"] = {
        "kind": "strategy-mixture",
        "weighting": "equal-active-strategies",
        "strategies": strategies,
        "voluntary_switches": True,
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
            "public_belief_root_values": (
                dict(public.get("root_values"))
                if isinstance(public.get("root_values"), Mapping)
                else None
            ),
            "public_belief_search_horizons": trace.get(
                "continuation_decision_horizons"
            ),
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


def transition_program_belief_result(
    *,
    fixture: DecisionFixture,
    posterior: Mapping[str, Any],
    transition_program: Mapping[str, Any],
    evaluator: Any,
) -> LiveDecisionResult:
    """Search one verified whole-turn mechanics program under the public belief."""

    if (
        transition_program.get("schema") != PROGRAM_SET_SCHEMA
        or transition_program.get("schema_version") != PROGRAM_SET_SCHEMA_VERSION
    ):
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="transition-program-schema-mismatch",
        )
    if transition_program.get("source_fixture_id") != fixture.fixture_id:
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="transition-program-fixture-mismatch",
            diagnostics={
                "program_fixture_id": transition_program.get("source_fixture_id")
            },
        )
    if transition_program.get("showdown_commit") != PINNED_SHOWDOWN_COMMIT:
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="transition-program-revision-mismatch",
            diagnostics={
                "showdown_commit": transition_program.get("showdown_commit")
            },
        )
    if transition_program.get("legal_actions") != list(fixture.legal_actions):
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="transition-program-legal-actions-drifted",
        )

    try:
        belief, transport_index = parse_belief_artifact(posterior)
        mechanics = VerifiedTransitionProgramSet.from_artifact(
            artifact=transition_program,
            identity=MechanicsIdentity.from_showdown_commit(PINNED_SHOWDOWN_COMMIT),
            fixture_id=fixture.fixture_id,
            legal_actions=tuple(fixture.legal_actions),
            belief=belief,
            transport_index=transport_index,
        )
        search = search_transition_program(
            mechanics=mechanics,
            belief=belief,
            transport_index=transport_index,
            method="information_set",
            evaluator=evaluator,
        )
    except (
        TransitionProgramSearchError,
        MechanicsContractError,
        ResearchContractError,
    ) as error:
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="transition-program-search-failed",
            diagnostics={"error": str(error)[-1000:]},
        )

    action = search.get("chosen_action")
    if not isinstance(action, str) or action not in set(fixture.legal_actions):
        return LiveDecisionResult(
            action=None,
            status="fallback",
            reason="transition-program-returned-nonlegal-action",
            diagnostics={"searched_action": action},
        )

    producer = transition_program.get("producer")
    producer_diagnostics = dict(producer) if isinstance(producer, Mapping) else {}

    return LiveDecisionResult(
        action=action,
        status="selected",
        reason="transition-program-public-belief",
        diagnostics={
            "learned_route": "transition-program-search",
            "fixture_id": fixture.fixture_id,
            "showdown_commit": transition_program.get("showdown_commit"),
            "transition_program_digest": search.get("transition_program_digest"),
            "transition_evaluations": search.get("transition_evaluations"),
            "evaluator_calls": search.get("evaluator_calls"),
            "evaluator_batches": search.get("evaluator_batches"),
            "root_snapshot_builds": producer_diagnostics.get("root_snapshot_builds"),
            "saved_root_snapshot_builds": producer_diagnostics.get(
                "saved_root_snapshot_builds"
            ),
            "transition_execution_cache_hits": producer_diagnostics.get(
                "transition_execution_cache_hits"
            ),
            "exact_transition_execution_cache_hits": producer_diagnostics.get(
                "exact_transition_execution_cache_hits"
            ),
            "public_projection_cache_hits": producer_diagnostics.get(
                "public_projection_cache_hits"
            ),
            "transition_delta_rehydrations": producer_diagnostics.get(
                "transition_delta_rehydrations"
            ),
            "public_successor_delta_schema": producer_diagnostics.get(
                "public_successor_delta_schema"
            ),
            "public_root_dependency_schema": producer_diagnostics.get(
                "public_root_dependency_schema"
            ),
            "public_read_fields": producer_diagnostics.get("public_read_fields"),
            "public_trace_incomplete_executions": producer_diagnostics.get(
                "public_trace_incomplete_executions"
            ),
            "transition_execution_cache_misses": producer_diagnostics.get(
                "transition_execution_cache_misses"
            ),
            "fresh_showdown_turn_executions": producer_diagnostics.get(
                "fresh_showdown_turn_executions"
            ),
            "reused_showdown_turn_executions": producer_diagnostics.get(
                "reused_showdown_turn_executions"
            ),
            "public_belief_root_values": dict(search.get("root_values", {})),
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
        self._probe_runtime = (
            PersistentShowdownProbe(self.showdown_root)
            if self._configuration_error is None and learned_evaluator is not None
            else None
        )

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

    def close(self) -> None:
        """Release the optional persistent Showdown worker."""

        runtime = getattr(self, "_probe_runtime", None)
        if runtime is not None:
            runtime.close()

    def _probe_document(
        self,
        source: Mapping[str, Any],
        *,
        posterior_only: bool = False,
        transition_program_only: bool = False,
    ) -> Mapping[str, Any]:
        script = Path(__file__).resolve().parents[3] / "scripts" / "probe_real_belief_trace.cjs"
        with tempfile.TemporaryDirectory(prefix="azelficoast-live-belief-") as temp_dir:
            source_path = Path(temp_dir) / "source.json"
            source_path.write_text(
                json.dumps(source, sort_keys=True),
                encoding="utf-8",
            )
            command = ["node", str(script), str(self.showdown_root), str(source_path)]
            if posterior_only and transition_program_only:
                raise LiveBeliefPolicyError(
                    "posterior-only and transition-program-only are mutually exclusive"
                )
            if posterior_only:
                command.append("--posterior-only")
            if transition_program_only:
                command.append("--transition-program-only")
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
        return self._probe_document(source)

    def _probe_posterior(self, source: Mapping[str, Any]) -> Mapping[str, Any]:
        runtime = getattr(self, "_probe_runtime", None)
        document = (
            runtime.posterior(source, timeout_seconds=self.timeout_seconds)
            if runtime is not None
            else self._probe_document(source, posterior_only=True)
        )
        if (
            document.get("schema") != "azelficoast.live-belief-posterior"
            or document.get("schema_version") != 1
        ):
            raise LiveBeliefPolicyError("unexpected posterior-only probe schema")
        return document

    def _probe_transition_program(
        self,
        source: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        runtime = getattr(self, "_probe_runtime", None)
        document = (
            runtime.transition_program(source, timeout_seconds=self.timeout_seconds)
            if runtime is not None
            else self._probe_document(source, transition_program_only=True)
        )
        if (
            document.get("schema") != PROGRAM_SET_SCHEMA
            or document.get("schema_version") != PROGRAM_SET_SCHEMA_VERSION
        ):
            raise LiveBeliefPolicyError("unexpected transition-program probe schema")
        return document

    def _release_probe_session(self, source: Mapping[str, Any]) -> None:
        runtime = getattr(self, "_probe_runtime", None)
        if runtime is None:
            return
        try:
            runtime.release(
                source,
                timeout_seconds=min(self.timeout_seconds, 1.0),
            )
        except Exception:
            # This is only memory cleanup for a high-confidence route. The selected
            # action must not depend on whether an optimization session can be released.
            pass

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
        posterior: Mapping[str, Any] | None = None
        program_failure: dict[str, Any] = {}

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
                    self._release_probe_session(source)
                    return route
            except Exception as error:
                if posterior is not None:
                    self._release_probe_session(source)
                posterior = None
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

        opponent_policy = source.get("opponent_policy")
        if not isinstance(opponent_policy, Mapping):
            if posterior is not None:
                self._release_probe_session(source)
            return LiveDecisionResult(
                action=None,
                status="fallback",
                reason="opponent-model-unavailable",
                diagnostics={
                    "posterior_available": posterior is not None,
                    "exact_search_blocker": "no-opponent-policy",
                    **(dict(route.diagnostics) if route is not None else {}),
                },
            )

        if (
            self.learned_evaluator is not None
            and posterior is not None
            and route is not None
            and route.action is None
        ):
            try:
                transition_program = self._probe_transition_program(source)
                searched = transition_program_belief_result(
                    fixture=fixture,
                    posterior=posterior,
                    transition_program=transition_program,
                    evaluator=self.learned_evaluator,
                )
                if searched.action is not None:
                    return LiveDecisionResult(
                        action=searched.action,
                        status=searched.status,
                        reason=searched.reason,
                        diagnostics={
                            **dict(route.diagnostics),
                            **dict(searched.diagnostics),
                        },
                    )
                program_failure = {
                    "transition_program_fallback_reason": searched.reason,
                    **dict(searched.diagnostics),
                }
            except subprocess.TimeoutExpired:
                return LiveDecisionResult(
                    action=None,
                    status="fallback",
                    reason="transition-program-search-timeout",
                    diagnostics={
                        "timeout_seconds": self.timeout_seconds,
                        **dict(route.diagnostics),
                    },
                )
            except (
                OSError,
                subprocess.CalledProcessError,
                json.JSONDecodeError,
                LiveBeliefPolicyError,
                ShowdownProbeRuntimeError,
            ) as error:
                detail = str(error)
                if isinstance(error, subprocess.CalledProcessError):
                    detail = (error.stderr or error.stdout or detail).strip()
                program_failure = {
                    "transition_program_fallback_reason": "program-probe-failed",
                    "transition_program_error": detail[-1000:],
                }

        # Compatibility fallback for unlearned configurations or a failed program
        # path. New learned search does not require this exhaustive matrix.
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
                    **program_failure,
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
                    **program_failure,
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
                **program_failure,
            },
        )
