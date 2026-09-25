from __future__ import annotations

import argparse

import pytest

from azelficoast.live.harness import _build_parser, _positive_int, _resolve_live_credentials


def test_challenge_parsing() -> None:
    args = _build_parser().parse_args(["challenge", "Jaxcalibur", "--battles", "3"])
    assert args.command == "challenge"
    assert args.opponent == "Jaxcalibur"
    assert args.battles == 3


def test_corpus_build_parsing() -> None:
    args = _build_parser().parse_args(
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
    args = _build_parser().parse_args(
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
        _positive_int("0")


def test_credentials_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOWDOWN_USERNAME", "azelficoast")
    monkeypatch.setenv("SHOWDOWN_PASSWORD", "secret")
    assert _resolve_live_credentials(None) == ("azelficoast", "secret")



def test_local_concurrency_parsing() -> None:
    args = _build_parser().parse_args(
        ["local", "--battles", "32", "--concurrency", "8"]
    )
    assert args.command == "local"
    assert args.battles == 32
    assert args.concurrency == 8

def test_live_belief_configuration_parsing() -> None:
    args = _build_parser().parse_args(
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
    args = _build_parser().parse_args(
        ["belief-coverage", "one.jsonl", "two.jsonl"]
    )
    assert args.command == "belief-coverage"
    assert [str(path) for path in args.traces] == ["one.jsonl", "two.jsonl"]


def test_training_improve_parsing() -> None:
    args = _build_parser().parse_args(
        [
            "training",
            "improve",
            "training.jsonl",
            "--incumbent",
            "models/incumbent",
            "--epochs",
            "3",
        ]
    )
    assert args.training_command == "improve"
    assert args.dataset.name == "training.jsonl"
    assert str(args.incumbent) == "models/incumbent"
    assert args.epochs == 3



def test_training_cycle_parsing() -> None:
    args = _build_parser().parse_args(
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


def test_training_auto_parsing() -> None:
    args = _build_parser().parse_args(
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
