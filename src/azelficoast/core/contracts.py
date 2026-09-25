"""Small contracts between domain semantics and generic belief/search machinery."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence


class PartialInformationDomain(Protocol):
    """Domain adapter used to expose partial-information decision semantics."""

    def legal_actions(self, public_state: Mapping[str, Any]) -> Sequence[str]:
        """Return actions legal from the supplied public state."""

    def initial_belief(
        self,
        public_state: Mapping[str, Any],
        observation_history: Any,
    ) -> Mapping[str, Any]:
        """Construct finite hidden-world support conditioned on public evidence."""

    def transition(
        self,
        world: Mapping[str, Any],
        public_state: Mapping[str, Any],
        action: str,
    ) -> Mapping[str, Any]:
        """Produce one authoritative transition result for a world/action pair."""

    def public_observation(self, transition_result: Mapping[str, Any]) -> Any:
        """Project a transition result onto information visible to the decision maker."""


class TransitionOracleProducer(Protocol):
    """Producer for an immutable finite-support transition oracle."""

    def build_transition_oracle(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior: Mapping[str, Any],
        legal_actions: Sequence[str],
    ) -> Mapping[str, Any]:
        """Materialize complete world/action transition evidence."""


class BeliefEvaluator(Protocol):
    """Value surface over public state and a correlated hidden-world posterior."""

    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        """Return a finite scalar value without revealing realized hidden state."""
