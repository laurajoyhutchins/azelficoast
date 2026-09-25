"""Azelficoast learned-evaluator adapter for generic partial-information search."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping, Sequence

from azelficoast.belief_evaluator import build_evaluator_input
from azelficoast.core.search import (
    SEARCH_METHODS,
    PartialInformationSearchError,
    search_transition_program as search_partial_information_program,
)
from azelficoast.whole_turn_program import (
    PROGRAM_SET_SCHEMA,
    PROGRAM_SET_SCHEMA_VERSION,
)

SEARCH_SCHEMA = "azelficoast.transition-program-search"
SEARCH_SCHEMA_VERSION = 1
METHODS = SEARCH_METHODS
TransitionProgramSearchError = PartialInformationSearchError


class _LearnedEvaluatorAdapter:
    """Expose Azelficoast's frozen learned evaluator through the generic value contract."""

    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    def value(
        self,
        *,
        public_state: Mapping[str, Any],
        posterior_worlds: Sequence[Mapping[str, Any]],
        legal_actions: Sequence[str],
    ) -> float:
        posterior = {
            "conditioned_on_public_history": True,
            "realized_hidden_state_revealed": False,
            "worlds": [copy.deepcopy(dict(world)) for world in posterior_worlds],
        }
        try:
            inputs = build_evaluator_input(
                public_state=public_state,
                posterior=posterior,
                legal_actions=legal_actions,
                spec=self.evaluator.spec,
            )
            prediction = self.evaluator.predict(inputs)
        except Exception as error:
            raise TransitionProgramSearchError(
                f"learned evaluator failed at successor leaf: {error}"
            ) from error

        value = float(prediction.value)
        if not math.isfinite(value):
            raise TransitionProgramSearchError(
                "learned evaluator returned a non-finite successor value"
            )
        return value


def search_transition_program(
    *,
    program_set: Mapping[str, Any],
    posterior: Mapping[str, Any],
    method: str,
    evaluator: Any,
) -> dict[str, Any]:
    """Evaluate verified Azelficoast mechanics through the generic search kernel."""

    return search_partial_information_program(
        program_set=program_set,
        posterior=posterior,
        method=method,
        evaluator=_LearnedEvaluatorAdapter(evaluator),
        expected_program_schema=PROGRAM_SET_SCHEMA,
        expected_program_schema_version=PROGRAM_SET_SCHEMA_VERSION,
        result_schema=SEARCH_SCHEMA,
        result_schema_version=SEARCH_SCHEMA_VERSION,
    )


__all__ = [
    "METHODS",
    "SEARCH_SCHEMA",
    "SEARCH_SCHEMA_VERSION",
    "TransitionProgramSearchError",
    "search_transition_program",
]
