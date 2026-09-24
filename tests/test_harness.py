from __future__ import annotations

import argparse

import pytest

from azelficoast.harness import _build_parser, _positive_int, _resolve_live_credentials


def test_challenge_parsing() -> None:
    args = _build_parser().parse_args(["challenge", "Jaxcalibur", "--battles", "3"])
    assert args.command == "challenge"
    assert args.opponent == "Jaxcalibur"
    assert args.battles == 3


def test_positive_int_rejects_zero() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _positive_int("0")


def test_credentials_come_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHOWDOWN_USERNAME", "azelficoast")
    monkeypatch.setenv("SHOWDOWN_PASSWORD", "secret")
    assert _resolve_live_credentials(None) == ("azelficoast", "secret")
