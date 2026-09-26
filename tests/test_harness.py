from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from azelficoast.live.battle_runtime import resolve_live_credentials
from azelficoast.live.cli import (
    build_parser,
    nonnegative_int,
    open_unit_float,
    positive_even_int,
    positive_int,
)
from azelficoast.live.self_improvement_runtime import adaptive_public_replay_count




def test_harness_stays_a_dispatch_shell() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "azelficoast"
        / "live"
        / "harness.py"
    ).read_text(encoding="utf-8")

    assert "add_argument(" not in source
    assert "poke_env" not in source
    assert "run_self_improvement_cycle" not in source
    assert "build_training_dataset" not in source



def test_training_runtime_stays_command_dispatch_only() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "azelficoast"
        / "live"
        / "training_runtime.py"
    ).read_text(encoding="utf-8")

    assert "poke_env" not in source
    assert "settle_battle_panel" not in source
    assert "allocate_evidence_budget" not in source
    assert "async def run_automatic_self_improvement" not in source

def test_challenge_parsing() -> None:
    args = build_parser().parse_args(["challenge", "Jaxcalibur", "--battles", "3"])
    assert args.command == "challenge"
    assert args.opponent == "Jaxcalibur"
    assert args.battles == 3


def test_corpus_build_parsing() -> None:
    args = build_parser().parse_args(
        [
            "corpus",
            "build",
            "one.jsonl",
            "two.jsonl",
            "--output",
            "fixtures.jsonl",
        ]
    )
    assert args.command == "corpus"
    assert args.corpus_command == "build"
    assert [path.name for path in args.traces] == ["one.jsonl", "two.jsonl"]
    assert args.output.name == "fixtures.jsonl"


def test_corpus_evaluate_parsing() -> None:
    args = build_parser().parse_args(
        [
            "corpus",
            "evaluate",
            "fixtures.jsonl",
            "--policy",
            "first-legal",
        ]
    )
    assert args.corpus_command == "evaluate"
    assert args.corpus_path.name == "fixtures.jsonl"
    assert args.policy == "first-legal"


def test_positive_int_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_credentials_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOWDOWN_USERNAME", "azelficoast")
    monkeypatch.setenv("SHOWDOWN_PASSWORD", "secret")
    assert resolve_live_credentials(None) == ("azelficoast", "secret")



def test_local_concurrency_parsing() -> None:
    args = build_parser().parse_args(
        ["local", "--battles", "32", "--concurrency", "8"]
    )
    assert args.command == "local"
    assert args.battles == 32
    assert args.concurrency == 8

def test_live_belief_configuration_parsing() -> None:
    args = build_parser().parse_args(
        [
            "--showdown-root",
            "/tmp/pokemon-showdown",
            "--belief-timeout",
            "3.5",
            "local",
        ]
    )
    assert str(args.showdown_root) == "/tmp/pokemon-showdown"
    assert args.belief_timeout == 3.5


def test_belief_coverage_parsing() -> None:
    args = build_parser().parse_args(
        ["belief-coverage", "one.jsonl", "two.jsonl"]
    )
    assert args.command == "belief-coverage"
    assert [str(path) for path in args.traces] == ["one.jsonl", "two.jsonl"]


def test_training_improve_parsing() -> None:
    args = build_parser().parse_args(
        [
            "training",
            "improve",
            "training.jsonl",
            "--incumbent",
            "models/incumbent",
            "--epochs",
            "3",
            "--max-validation-posterior-stress-regression",
            "0.05",
        ]
    )
    assert args.training_command == "improve"
    assert args.dataset.name == "training.jsonl"
    assert str(args.incumbent) == "models/incumbent"
    assert args.epochs == 3
    assert args.max_validation_posterior_stress_regression == 0.05



def test_training_cycle_parsing() -> None:
    args = build_parser().parse_args(
        [
            "--showdown-root",
            "/tmp/pokemon-showdown",
            "training",
            "cycle",
            "decisions.jsonl",
            "--incumbent",
            "artifacts/evaluators/current.json",
            "--teacher-budget",
            "2048",
        ]
    )
    assert args.training_command == "cycle"
    assert [path.name for path in args.traces] == ["decisions.jsonl"]
    assert str(args.showdown_root) == "/tmp/pokemon-showdown"
    assert args.teacher_budget == 2048
    assert args.teacher_challenger_uncertainty == 0.75


def test_training_auto_parsing() -> None:
    args = build_parser().parse_args(
        [
            "--showdown-root",
            "/tmp/pokemon-showdown",
            "training",
            "auto",
            "--incumbent",
            "artifacts/evaluators/current.json",
            "--generations",
            "3",
            "--battles-per-generation",
            "20",
            "--max-teacher-fixtures",
            "48",
        ]
    )
    assert args.training_command == "auto"
    assert args.generations == 3
    assert args.battles_per_generation == 20
    assert args.battle_search_policy_margin == 0.0
    assert args.max_teacher_fixtures == 48
    assert args.teacher_challenger_uncertainty == 0.75


def test_nonnegative_int_allows_zero_and_rejects_negative() -> None:
    assert nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_int("-1")


def test_teacher_challenger_threshold_parsing() -> None:
    args = build_parser().parse_args(
        [
            "training",
            "cycle",
            "decisions.jsonl",
            "--teacher-challenger-uncertainty",
            "0.6",
        ]
    )

    assert args.teacher_challenger_uncertainty == 0.6


def test_training_auto_replenishes_public_curriculum_by_default() -> None:
    args = build_parser().parse_args(
        [
            "--showdown-root",
            "/tmp/pokemon-showdown",
            "training",
            "auto",
        ]
    )

    assert args.public_replays_per_generation == 4
    assert args.public_replay_min_rating == 1500


def test_promotion_battle_controls_are_fail_closed() -> None:
    assert positive_even_int("32") == 32
    with pytest.raises(argparse.ArgumentTypeError):
        positive_even_int("31")
    assert open_unit_float("0.1") == 0.1
    with pytest.raises(argparse.ArgumentTypeError):
        open_unit_float("1")


def test_training_auto_has_battle_strength_gate_by_default() -> None:
    args = build_parser().parse_args(["training", "auto"])

    assert args.promotion_battles == 32
    assert args.promotion_alpha == 0.10


def test_public_replay_acquisition_stays_full_on_cold_start() -> None:
    assert adaptive_public_replay_count(
        4,
        ledger=None,
        generated_source_kinds=["generated-dirty-tricks"],
    ) == 4


def test_public_replay_acquisition_backs_off_when_generated_debt_is_higher() -> None:
    ledger = {
        "claims": [
            {
                "claim_id": "evidence-source:public-showdown-replay",
                "evidence_count": 100,
                "debt": 0.1,
            },
            {
                "claim_id": "evidence-source:generated-dirty-tricks",
                "evidence_count": 10,
                "debt": 0.2,
            },
            {
                "claim_id": "evidence-source-hard:generated-dirty-tricks",
                "evidence_count": 1,
                "debt": 2.0,
            },
        ]
    }

    assert adaptive_public_replay_count(
        4,
        ledger=ledger,
        generated_source_kinds=["generated-dirty-tricks"],
    ) == 1



def test_promotion_evidence_identity_names_are_stable() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "azelficoast"
        / "live"
        / "promotion_runtime.py"
    ).read_text(encoding="utf-8")

    assert '"candidate_checkpoint_digest"' in source
    assert '"incumbent_checkpoint_digest"' in source
    assert "candidate_checkpoint_digest=" in source
    assert "incumbent_checkpoint_digest=" in source
    assert "candidatecheckpoint_digest" not in source
    assert "incumbentcheckpoint_digest" not in source
