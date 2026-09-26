"""Explicit training command dispatch."""

from __future__ import annotations

import argparse
import asyncio
import json

from azelficoast.belief.improvement import AdmissionPolicy, improve_checkpoint
from azelficoast.belief.public_pretraining import run_public_pretraining
from azelficoast.belief.self_improvement import run_self_improvement_cycle
from azelficoast.live.self_improvement_runtime import run_automatic_self_improvement
from azelficoast.research.training_records import build_training_dataset


def _admission_policy(args: argparse.Namespace) -> AdmissionPolicy:
    return AdmissionPolicy(
        min_validation_total_improvement=args.min_validation_improvement,
        max_validation_value_mse_regression=args.max_validation_value_regression,
        max_validation_policy_cross_entropy_regression=(
            args.max_validation_policy_regression
        ),
        max_validation_posterior_stress_regression=(
            args.max_validation_posterior_stress_regression
        ),
    )


def run_training(args: argparse.Namespace) -> None:
    if args.training_command == "build":
        summary = build_training_dataset(
            args.traces,
            args.output,
            search_packet_paths=args.search_packet,
            search_receipt_paths=args.search_receipt,
            posterior_paths=args.posterior,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
    elif args.training_command == "bootstrap-public":
        if args.showdown_root is None:
            raise ValueError(
                "public pretraining requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
            )
        summary = run_public_pretraining(
            args.traces,
            showdown_root=args.showdown_root,
            dataset_path=args.output,
            models_dir=args.models_dir,
            receipts_dir=args.receipts_dir,
            promotion_file=args.promotion,
            posterior_timeout_seconds=args.belief_timeout,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            policy_weight=args.policy_weight,
            admission_policy=_admission_policy(args),
        )
    elif args.training_command == "improve":
        summary = improve_checkpoint(
            args.dataset,
            incumbent_checkpoint=args.incumbent,
            models_dir=args.models_dir,
            receipts_dir=args.receipts_dir,
            promotion_file=args.promotion,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            policy_weight=args.policy_weight,
            value_target_source=args.value_target,
            admission_policy=_admission_policy(args),
        )
    elif args.training_command == "cycle":
        if args.showdown_root is None:
            raise ValueError(
                "training cycle requires --showdown-root or AZELFICOAST_SHOWDOWN_ROOT"
            )
        summary = run_self_improvement_cycle(
            args.traces,
            showdown_root=args.showdown_root,
            incumbent_checkpoint=args.incumbent,
            workspace=args.workspace,
            models_dir=args.models_dir,
            receipts_dir=args.receipts_dir,
            promotion_file=args.promotion,
            teacher_compute_budget=args.teacher_budget,
            max_teacher_fixtures=args.max_teacher_fixtures,
            challenger_uncertainty_threshold=args.teacher_challenger_uncertainty,
            teacher_timeout_seconds=args.belief_timeout,
            split_seed=args.split_seed,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            policy_weight=args.policy_weight,
            value_target_source=args.value_target,
            admission_policy=_admission_policy(args),
        )
    elif args.training_command == "auto":
        summary = asyncio.run(run_automatic_self_improvement(args))
    else:
        raise AssertionError(f"unsupported training command: {args.training_command}")
    print(json.dumps(summary, sort_keys=True))


