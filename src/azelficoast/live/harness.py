"""Run Azelficoast command-line workflows."""

from __future__ import annotations

import asyncio
from typing import Sequence

from azelficoast.live.battle_runtime import run_battle_command
from azelficoast.live.cli import build_parser
from azelficoast.live.offline_commands import run_belief_coverage, run_corpus
from azelficoast.live.training_runtime import run_training


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "corpus":
            run_corpus(args)
        elif args.command == "training":
            run_training(args)
        elif args.command == "belief-coverage":
            run_belief_coverage(args)
        else:
            asyncio.run(run_battle_command(args))
    except ValueError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
