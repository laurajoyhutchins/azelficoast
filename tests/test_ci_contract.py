from __future__ import annotations

from pathlib import Path
import re


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


def test_expensive_pr_workflows_are_exact_head_fenced() -> None:
    checked: list[str] = []

    for path in sorted(WORKFLOWS.glob("*.yml")):
        if path.name == "ci.yml":
            continue

        source = path.read_text(encoding="utf-8")
        if "  pull_request:\n" not in source:
            continue

        pull_request = _event_block(source, "pull_request")
        if path.name == "candidate-research.yml":
            assert "types: [opened, synchronize, reopened, ready_for_review]" in pull_request
            assert "github.event.pull_request.draft == false" in source
            assert "github.event.pull_request.head.sha || github.sha" in source
        else:
            assert "types: [ready_for_review]" in pull_request, (
                f"{path.name} must remain candidate-only until migrated"
            )
        assert "  workflow_dispatch:\n" in source, (
            f"{path.name} must retain an explicit manual evidence path"
        )
        assert "${{ github.event.pull_request.number || github.ref }}" in source, (
            f"{path.name} must key concurrency by PR when available"
        )
        assert "cancel-in-progress: true" in source
        checked.append(path.name)

    assert checked, "expected at least one research workflow"


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


def test_repository_evidence_replaces_historical_artifact_runtime_dependencies() -> None:
    manifest = ROOT / "experiments" / "evidence" / "canonical-evidence.json"
    bundle = ROOT / "experiments" / "evidence" / "canonical-evidence.tar.gz"
    action = ROOT / ".github" / "actions" / "setup-research-evidence" / "action.yml"

    assert manifest.is_file()
    assert bundle.is_file()
    assert action.is_file()

    action_source = action.read_text(encoding="utf-8")
    assert "canonical-evidence.json" in action_source
    assert "canonical-evidence.tar.gz" in action_source
    assert "sha256sum" in action_source
    assert "tar -xzf" in action_source

    for path in sorted(WORKFLOWS.glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        assert "actions/artifacts/" not in source, (
            f"{path.name} must not depend on historical Actions artifacts"
        )
        assert "restore-exact-artifact" not in source, (
            f"{path.name} must consume repository evidence instead of artifact restoration"
        )


def test_population_workflow_delegates_frozen_cohort_semantics() -> None:
    source = (WORKFLOWS / "natural-population-strategy-fusion.yml").read_text(
        encoding="utf-8"
    )
    assert "freeze-cohort:" not in source
    assert "setup-research-evidence" in source
    assert "azelficoast.research.hosted natural-population-shard" in source
    assert "/tmp/azelficoast-evidence/population/manifest.json" not in source


def test_active_population_plan_contains_no_superseded_cohort_history() -> None:
    source = (
        ROOT / "experiments" / "natural-population-strategy-fusion-plan.json"
    ).read_text(encoding="utf-8")
    assert '"superseded_cohort"' not in source
    assert '"superseded_cohort_after_forme_bug"' not in source
    assert '"preregistration_amendment' not in source
    assert '"canonical_cohort"' in source


def test_completed_discovery_workflows_are_not_executable_ci() -> None:
    obsolete = {
        "public-belief-feasibility-search.yml",
        "natural-disagreement-experiment.yml",
        "real-belief-candidate-corpus.yml",
        "natural-public-belief-exact.yml",
        "natural-public-belief-robustness.yml",
        "simulator-adaptive-execution-experiment.yml",
        "simulator-attack-transition-experiment.yml",
        "simulator-class-native-belief-experiment.yml",
        "simulator-gen9-damage-experiment.yml",
        "simulator-jax-experiment.yml",
        "simulator-native-damage-experiment.yml",
        "simulator-ordered-attack-experiment.yml",
        "simulator-showdown-experiment.yml",
        "simulator-two-attack-turn-experiment.yml",
        "stateful-protect-turn-experiment.yml",
        "switch-entry-hazard-experiment.yml",
        "switch-intimidate-experiment.yml",
        "voluntary-switch-turn-experiment.yml",
    }
    present = {path.name for path in WORKFLOWS.glob("*.yml")}
    assert obsolete.isdisjoint(present), sorted(obsolete & present)


def test_canonical_evidence_consumers_observe_evidence_changes() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        if "uses: ./.github/actions/setup-research-evidence" not in source:
            continue
        if path.name == "ci.yml":
            continue

        assert '- ".github/actions/setup-research-evidence/action.yml"' in source, (
            f"{path.name} must observe canonical evidence setup changes"
        )
        assert '- "experiments/evidence/canonical-evidence.json"' in source, (
            f"{path.name} must observe canonical evidence manifest changes"
        )
        assert '- "experiments/evidence/canonical-evidence.tar.gz"' in source, (
            f"{path.name} must observe canonical evidence bundle changes"
        )


def test_research_semantics_do_not_live_in_workflow_yaml() -> None:
    forbidden_fragments = (
        "python - <<",
        "node - <<",
        "assert ",
        "--target-particles",
        "--minimum-particles",
        "--max-rounds",
        "--battles ",
        "--rounds ",
        "AZELFICOAST_ROOT_CHANCE_SAMPLES",
        "AZELFICOAST_CONTINUATION_CHANCE_SAMPLES",
        "AZELFICOAST_CHANCE_SEED_FAMILY",
        "AZELFICOAST_CONTINUATION_DECISION_HORIZONS",
    )
    digest = re.compile(r"(?<![0-9a-f])[0-9a-f]{40,64}(?![0-9a-f])")

    for path in sorted(WORKFLOWS.glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, (
                f"{path.name} encodes research semantics via {fragment!r}"
            )
        assert digest.search(source) is None, (
            f"{path.name} must not own fixture, evidence, or revision digests"
        )


def test_showdown_action_reads_authority_from_repository_data() -> None:
    action = (
        ROOT / ".github" / "actions" / "setup-showdown" / "action.yml"
    ).read_text(encoding="utf-8")
    revision = (ROOT / "experiments" / "showdown-revision.txt").read_text(
        encoding="utf-8"
    ).strip()

    assert "experiments/showdown-revision.txt" in action
    assert revision not in action
    assert re.fullmatch(r"[0-9a-f]{40}", revision)


def test_candidate_research_certificate_semantics_live_in_python() -> None:
    source = (WORKFLOWS / "candidate-research.yml").read_text(encoding="utf-8")

    assert "\n  certify:\n" in source
    assert "needs: [plan, exact]" in source
    assert "if: always() && needs.plan.result == 'success'" in source
    assert "src/azelficoast/research/ci.py certify" in source
    assert '"schema": "azelficoast.candidate-research-certificate"' not in source
    assert "python - <<" not in source
    assert "name: candidate-research-certificate" in source


def test_hosted_witness_membership_is_not_encoded_in_yaml() -> None:
    from azelficoast.research.hosted.belief import (
        EXHAUSTED_FIXTURES,
        PUBLIC_BELIEF_FIXTURES,
    )

    suites = {
        "exhausted-bench-real-belief.yml": ("exhausted-bench", EXHAUSTED_FIXTURES),
        "public-belief-exact-corpus.yml": ("public-belief-exact", PUBLIC_BELIEF_FIXTURES),
    }
    for workflow, (suite, cases) in suites.items():
        source = (WORKFLOWS / workflow).read_text(encoding="utf-8")
        assert f"azelficoast.research.hosted matrix {suite}" in source
        for name in cases:
            assert name not in source
