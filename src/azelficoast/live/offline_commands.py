"""Offline corpus and coverage commands."""

from __future__ import annotations

import argparse
import json

from azelficoast.belief.coverage import summarize_traces
from azelficoast.live.corpus import build_corpus, evaluate_corpus
from azelficoast.research.public_replays import import_public_replays


def run_corpus(args: argparse.Namespace) -> None:
    if args.corpus_command == "build":
        summary = build_corpus(args.traces, args.output)
    elif args.corpus_command == "import-public":
        if args.showdown_root is None:
            raise ValueError(
                "public replay import requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
            )
        summary = import_public_replays(
            showdown_root=args.showdown_root,
            output_root=args.output_root,
            max_battles=args.max_battles,
            min_rating=args.min_rating,
            before=args.before,
            strict=args.strict,
        )
    elif args.corpus_command == "evaluate":
        summary = evaluate_corpus(args.corpus_path, args.policy, args.output)
    else:
        raise AssertionError(f"unsupported corpus command: {args.corpus_command}")
    print(json.dumps(summary, sort_keys=True))



def run_belief_coverage(args: argparse.Namespace) -> None:
    print(json.dumps(summarize_traces(args.traces), sort_keys=True))


