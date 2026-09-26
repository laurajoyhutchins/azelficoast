"""Cheap candidate evidence for the issue #69 population contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from azelficoast.research.posterior_population_contract import contract_readiness

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = REPOSITORY_ROOT / "experiments" / "posterior-stratified-population-contract.json"


def run_experiment() -> dict[str, Any]:
    document = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("posterior population contract must be a JSON object")
    return contract_readiness(document)
