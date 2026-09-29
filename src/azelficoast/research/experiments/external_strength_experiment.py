"""Cheap exact-head validation for the external playing-strength contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from azelficoast.research.external_strength import contract_readiness, load_contract

ROOT = Path(__file__).resolve().parents[4]
CONTRACT = ROOT / "experiments" / "data" / "external-playing-strength-contract.json"


def run_experiment() -> dict[str, Any]:
    return contract_readiness(load_contract(CONTRACT))
