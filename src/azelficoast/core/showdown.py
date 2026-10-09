"""Repository-owned Pokémon Showdown mechanics revision."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Final


class ShowdownAuthorityError(RuntimeError):
    """Raised when the repository's pinned Showdown authority is malformed."""


SHOWDOWN_REVISION_PATH: Final = (
    Path(__file__).resolve().parents[3] / "showdown" / "revision.json"
)


def _load_pinned_showdown_commit() -> str:
    try:
        document = json.loads(SHOWDOWN_REVISION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ShowdownAuthorityError("cannot read repository Showdown revision authority") from error

    if not isinstance(document, dict):
        raise ShowdownAuthorityError("Showdown revision authority must be a JSON object")
    if document.get("schema") != "azelficoast.showdown-revision":
        raise ShowdownAuthorityError("unexpected Showdown revision schema")
    if document.get("schema_version") != 1:
        raise ShowdownAuthorityError("unsupported Showdown revision schema version")

    commit = document.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ShowdownAuthorityError("Showdown revision must be a 40-hex commit")
    return commit


PINNED_SHOWDOWN_COMMIT: Final = _load_pinned_showdown_commit()
