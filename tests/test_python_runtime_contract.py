from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repository_python_runtime_is_standardized_on_312() -> None:
    assert (ROOT / ".python-version").read_text().strip() == "3.12"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert project["project"]["requires-python"] == ">=3.12"
    assert project["tool"]["ruff"]["target-version"] == "py312"
    assert project["tool"]["mypy"]["python_version"] == "3.12"


def test_shared_python_setup_uses_repository_runtime_authority() -> None:
    setup = (ROOT / ".github/actions/setup-python-environment/action.yml").read_text()
    assert 'python-version-file: ".python-version"' in setup
    assert "inputs.python-version" not in setup


def test_python_workflows_observe_and_use_repository_runtime() -> None:
    raw_python = re.compile(r"^\\s*python3?\\s+-\\s+<<", re.MULTILINE)
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        text = path.read_text()
        if "uses: ./.github/actions/setup-python-environment" not in text:
            continue
        if "paths:" in text:
            assert '- ".python-version"' in text, (
                f"{path.name} must rerun when repository Python changes"
            )
        assert raw_python.search(text) is None, (
            f"{path.name} bypasses the repository-managed Python runtime"
        )
