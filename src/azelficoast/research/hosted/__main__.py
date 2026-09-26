from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from azelficoast.research.hosted import belief, mechanics, population


Command = Callable[[], None]


COMMANDS: dict[str, Command] = {
    "conditional-team-prior": belief.conditional_team_prior,
    "decision-relevance-quotient": belief.decision_relevance_quotient,
    "factored-hidden-bench-prior": belief.factored_hidden_bench_prior,
    "joint-random-battle-posterior": belief.joint_random_battle_posterior,
    "live-belief-coverage": belief.live_belief_coverage,
    "natural-status-move": belief.natural_status_move,
    "real-belief-decision-trace": belief.real_belief_decision_trace,
    "real-belief-survival-witness": belief.real_belief_survival_witness,
    "status-move-hidden-world-prior": belief.status_move_prior,
    "opponent-team-completion-support": mechanics.opponent_team_completion_support,
    "policy-boundary-refinement": mechanics.policy_boundary_refinement,
    "protect-action-survival": mechanics.protect_action_survival,
    "protect-continuation": mechanics.protect_continuation,
    "protect-speed-fork": mechanics.protect_speed_fork,
    "replay-world": mechanics.replay_world,
    "natural-population-aggregate": population.natural_population_aggregate,
    "natural-depth-restore": population.natural_depth_restore,
    "natural-depth-aggregate": population.natural_depth_aggregate,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute GitHub-hosted research semantics outside workflow YAML."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for command in sorted(COMMANDS):
        commands.add_parser(command)

    matrix = commands.add_parser("matrix")
    matrix.add_argument("suite", choices=("exhausted-bench", "public-belief-exact"))

    exhausted = commands.add_parser("exhausted-bench")
    exhausted.add_argument("name", choices=tuple(belief.EXHAUSTED_FIXTURES))

    public = commands.add_parser("public-belief-exact")
    public.add_argument("name", choices=tuple(belief.PUBLIC_BELIEF_FIXTURES))

    population_shard = commands.add_parser("natural-population-shard")
    population_shard.add_argument("shard", type=int, choices=range(8))

    depth_shard = commands.add_parser("natural-depth-shard")
    depth_shard.add_argument("shard", type=int, choices=range(6))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        cases = (
            belief.EXHAUSTED_FIXTURES
            if args.suite == "exhausted-bench"
            else belief.PUBLIC_BELIEF_FIXTURES
        )
        import json

        print(json.dumps({"include": [{"name": name} for name in cases]}))
    elif args.command == "exhausted-bench":
        belief.exhausted_bench(args.name)
    elif args.command == "public-belief-exact":
        belief.public_belief_exact(args.name)
    elif args.command == "natural-population-shard":
        population.natural_population_shard(args.shard)
    elif args.command == "natural-depth-shard":
        population.natural_depth_shard(args.shard)
    else:
        COMMANDS[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
