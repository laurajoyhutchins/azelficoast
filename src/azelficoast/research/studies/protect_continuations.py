"""Mine real Protect branches whose hidden Choice worlds survive the turn."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from azelficoast.live.corpus import DecisionFixture, load_corpus
from azelficoast.research.studies.natural_disagreements import (
    _persistent_protect_actions,
    mine_candidates,
)


def mine_protect_candidates(
    fixtures: Sequence[DecisionFixture],
    *,
    showdown_root: str | Path,
    rounds: int = 2048,
) -> dict[str, Any]:
    """Run hidden-world inference only on fixtures with a Protect-family action."""
    protect_fixtures = [
        fixture for fixture in fixtures if _persistent_protect_actions(fixture)
    ]
    result = mine_candidates(
        protect_fixtures,
        showdown_root=Path(showdown_root),
        rounds=rounds,
        persistent_only=True,
    )

    candidates = [
        candidate
        for candidate in result["candidates"]
        if candidate["persistent_protect_actions"]
    ]
    if len(candidates) != result["candidate_count"]:
        raise ValueError("protect-only treatment emitted a non-Protect candidate")

    return {
        **result,
        "schema": "azelficoast.protect-continuation-candidates",
        "source_fixture_count": len(fixtures),
        "protect_fixture_count": len(protect_fixtures),
        "current_speed_fork_count": sum(
            bool(candidate["current_speed_fork"]) for candidate in candidates
        ),
    }


def mine_protect_corpus(
    corpus_path: str | Path,
    *,
    showdown_root: str | Path,
    rounds: int = 2048,
) -> dict[str, Any]:
    return mine_protect_candidates(
        load_corpus(corpus_path),
        showdown_root=showdown_root,
        rounds=rounds,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--showdown-root", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2048)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("--rounds must be positive")
    result = mine_protect_corpus(
        args.corpus,
        showdown_root=args.showdown_root,
        rounds=args.rounds,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
