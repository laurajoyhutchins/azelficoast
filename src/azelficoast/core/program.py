"""Domain-neutral helpers for finite transition-program sets."""

from __future__ import annotations

from typing import Any, Mapping

PROGRAM_SET_SCHEMA = "azelficoast.core.transition-program-set"
PROGRAM_SET_SCHEMA_VERSION = 1
EXECUTION_SCHEMA = "azelficoast.core.weighted-transition-outcomes"
EXECUTION_SCHEMA_VERSION = 1
VERIFICATION_SCHEMA = "azelficoast.core.transition-program-verification"
VERIFICATION_SCHEMA_VERSION = 1


class TransitionProgramError(ValueError):
    """Raised when a finite transition program is structurally invalid."""


def program_for_action(
    program_set: Mapping[str, Any],
    action: str,
    *,
    error_type: type[ValueError] = TransitionProgramError,
) -> Mapping[str, Any]:
    """Return the unique program for one root action."""

    raw_programs = program_set.get("programs")
    if not isinstance(raw_programs, list):
        raise error_type("transition program set has no programs")
    matches = [
        program
        for program in raw_programs
        if isinstance(program, Mapping) and program.get("action") == action
    ]
    if len(matches) != 1:
        raise error_type(f"expected one transition program for action {action!r}")
    return matches[0]
