from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def _event_block(source: str, event: str) -> str:
    marker = f"  {event}:\n"
    start = source.find(marker)
    assert start >= 0, f"missing {event} trigger"
    after = source[start + len(marker) :]
    lines = after.splitlines(keepends=True)
    block: list[str] = []
    for line in lines:
        if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            break
        block.append(line)
    return "".join(block)


def test_pr_ci_cancels_superseded_heads_and_observes_candidate_transition() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    pull_request = _event_block(source, "pull_request")

    assert "types: [opened, synchronize, reopened, ready_for_review]" in pull_request
    assert source.count("\nconcurrency:\n") == 1
    assert "group: ci-${{ github.event.pull_request.number || github.ref }}" in source
    assert "cancel-in-progress: true" in source


def test_expensive_pr_workflows_only_run_for_candidate_heads() -> None:
    checked: list[str] = []

    for path in sorted(WORKFLOWS.glob("*.yml")):
        if path.name == "ci.yml":
            continue

        source = path.read_text(encoding="utf-8")
        if "  pull_request:\n" not in source:
            continue

        pull_request = _event_block(source, "pull_request")
        assert "types: [ready_for_review]" in pull_request, (
            f"{path.name} must not spend research evidence on ordinary PR updates"
        )
        assert "  workflow_dispatch:\n" in source, (
            f"{path.name} must retain an explicit manual evidence path"
        )
        assert "${{ github.event.pull_request.number || github.ref }}" in source, (
            f"{path.name} must key concurrency by PR when available"
        )
        assert "cancel-in-progress: true" in source
        checked.append(path.name)

    assert checked, "expected at least one candidate-only research workflow"


def test_shared_python_environment_owns_locked_dependency_resolution() -> None:
    source = (
        ROOT / ".github" / "actions" / "setup-python-environment" / "action.yml"
    ).read_text(encoding="utf-8")

    assert "uses: actions/setup-python@v7" in source
    assert 'python-version: "3.13"' in source
    assert "python -m pip install uv==0.12.18" in source
    assert "uv sync --locked" in source
    assert "uv sync --locked --extra simulator" in source


def test_uv_managed_workflows_use_shared_python_environment() -> None:
    checked: list[str] = []

    for path in sorted(WORKFLOWS.glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        if "uv run" not in source:
            continue

        assert "uses: ./.github/actions/setup-python-environment" in source, (
            f"{path.name} must use the shared Python environment setup"
        )
        assert "actions/setup-python@v7" not in source
        assert "pip install uv" not in source
        assert "uv sync --locked" not in source

        if path.name != "ci.yml" and "    paths:\n" in source:
            assert '- ".github/actions/setup-python-environment/action.yml"' in source, (
                f"{path.name} must rerun when shared Python setup changes"
            )

        if '- "pyproject.toml"' in source:
            assert '- "uv.lock"' in source, (
                f"{path.name} treats pyproject.toml as dependency-sensitive "
                "and must treat uv.lock the same way"
            )

        checked.append(path.name)

    assert checked, "expected at least one uv-managed workflow"
