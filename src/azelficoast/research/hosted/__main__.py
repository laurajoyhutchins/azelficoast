from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence

from azelficoast.research.hosted import belief, mechanics, population
from azelficoast.research.hosted.contracts import (
    STUDIES_BY_NAME,
    HostedResearchContractError,
    aggregate_matrix,
    run_matrix,
    select_studies,
)


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
        description="Execute repository-owned hosted research contracts."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    for command in sorted(COMMANDS):
        commands.add_parser(command)

    matrix = commands.add_parser("matrix")
    matrix.add_argument(
        "suite",
        choices=("exhausted-bench", "public-belief-exact"),
    )

    exhausted = commands.add_parser("exhausted-bench")
    exhausted.add_argument("name", choices=tuple(belief.EXHAUSTED_FIXTURES))

    public = commands.add_parser("public-belief-exact")
    public.add_argument("name", choices=tuple(belief.PUBLIC_BELIEF_FIXTURES))

    population_shard = commands.add_parser("natural-population-shard")
    population_shard.add_argument("shard", type=int, choices=range(8))

    depth_shard = commands.add_parser("natural-depth-shard")
    depth_shard.add_argument("shard", type=int, choices=range(6))

    plan = commands.add_parser("plan")
    plan.add_argument("--base", default="")
    plan.add_argument("--head", default="")
    plan.add_argument("--requested", default="")

    run_unit = commands.add_parser("run-unit")
    run_unit.add_argument("study", choices=tuple(STUDIES_BY_NAME))
    run_unit.add_argument("unit")

    prepare = commands.add_parser("prepare-aggregate")
    prepare.add_argument("study", choices=tuple(STUDIES_BY_NAME))

    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("study", choices=tuple(STUDIES_BY_NAME))
    return parser


def _run_study_unit(study_name: str, unit: str) -> None:
    study = STUDIES_BY_NAME[study_name]
    contract = study.run
    if unit not in contract.units:
        raise HostedResearchContractError(
            f"{study_name} has no execution unit {unit!r}"
        )

    if contract.kind == "command":
        if unit != "default":
            raise HostedResearchContractError(
                f"{study_name} command contract requires the default unit"
            )
        command = COMMANDS.get(contract.operation)
        if command is None:
            raise HostedResearchContractError(
                f"{study_name} references unknown operation {contract.operation!r}"
            )
        command()
        return
    if contract.kind == "exhausted-bench":
        belief.exhausted_bench(unit)
        return
    if contract.kind == "public-belief-exact":
        belief.public_belief_exact(unit)
        return
    if contract.kind == "natural-population-shard":
        population.natural_population_shard(int(unit))
        return
    if contract.kind == "natural-depth-shard":
        population.natural_depth_shard(int(unit))
        return
    raise HostedResearchContractError(
        f"{study_name} uses unknown execution kind {contract.kind!r}"
    )


def _prepare_aggregate(study_name: str) -> None:
    aggregate = STUDIES_BY_NAME[study_name].aggregate
    if aggregate is None:
        raise HostedResearchContractError(
            f"{study_name} has no aggregate contract"
        )
    if aggregate.prepare is None:
        return
    command = COMMANDS.get(aggregate.prepare)
    if command is None:
        raise HostedResearchContractError(
            f"{study_name} references unknown prepare operation "
            f"{aggregate.prepare!r}"
        )
    command()


def _aggregate(study_name: str) -> None:
    aggregate = STUDIES_BY_NAME[study_name].aggregate
    if aggregate is None:
        raise HostedResearchContractError(
            f"{study_name} has no aggregate contract"
        )
    command = COMMANDS.get(aggregate.operation)
    if command is None:
        raise HostedResearchContractError(
            f"{study_name} references unknown aggregate operation "
            f"{aggregate.operation!r}"
        )
    command()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "matrix":
        cases = (
            belief.EXHAUSTED_FIXTURES
            if args.suite == "exhausted-bench"
            else belief.PUBLIC_BELIEF_FIXTURES
        )
        print(json.dumps({"include": [{"name": name} for name in cases]}))
    elif args.command == "exhausted-bench":
        belief.exhausted_bench(args.name)
    elif args.command == "public-belief-exact":
        belief.public_belief_exact(args.name)
    elif args.command == "natural-population-shard":
        population.natural_population_shard(args.shard)
    elif args.command == "natural-depth-shard":
        population.natural_depth_shard(args.shard)
    elif args.command == "plan":
        selected = select_studies(
            base=args.base,
            head=args.head,
            requested=args.requested,
        )
        print(
            json.dumps(
                {
                    "run": run_matrix(selected),
                    "aggregate": aggregate_matrix(selected),
                },
                separators=(",", ":"),
            )
        )
    elif args.command == "run-unit":
        _run_study_unit(args.study, args.unit)
    elif args.command == "prepare-aggregate":
        _prepare_aggregate(args.study)
    elif args.command == "aggregate":
        _aggregate(args.study)
    else:
        COMMANDS[args.command]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
