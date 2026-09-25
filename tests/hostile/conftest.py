from __future__ import annotations

from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "hostile_fast: dependency-light hostile invariant")
    config.addinivalue_line("markers", "hostile_simulator: requires simulator/JAX extra")
    config.addinivalue_line("markers", "hostile_showdown: requires pinned Pokémon Showdown")
    config.addinivalue_line(
        "markers",
        "hostile_contract(mutation, expected, threat, layer): scientific threat guarded by a test",
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Reject hostile tests that do not explain the scientific threat they guard."""
    incomplete: list[str] = []
    for item in items:
        path = Path(str(item.path))
        if "tests/hostile" not in path.as_posix():
            continue
        contracts = list(item.iter_markers("hostile_contract"))
        if len(contracts) != 1:
            incomplete.append(f"{item.nodeid}: expected exactly one hostile_contract marker")
            continue
        required = {"mutation", "expected", "threat", "layer"}
        actual = set(contracts[0].kwargs)
        if actual != required or any(not contracts[0].kwargs[name] for name in required):
            incomplete.append(
                f"{item.nodeid}: hostile_contract requires non-empty {sorted(required)!r}"
            )
    if incomplete:
        raise pytest.UsageError("\n".join(incomplete))
