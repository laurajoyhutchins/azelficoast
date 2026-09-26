from __future__ import annotations

from pathlib import Path
import tomllib


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


def test_static_analysis_frontier_is_explicit_and_non_regressing() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    mypy = config["tool"]["mypy"]
    assert mypy["strict"] is True
    assert mypy["follow_imports"] == "silent"
    checked = set(mypy["files"])
    assert {
        "src/azelficoast/core",
        "src/azelficoast/search",
        "src/azelficoast/live",
        "src/azelficoast/belief",
    } <= checked
    assert mypy["exclude"] == [
        "src/azelficoast/core/compiled_search.py",
        "src/azelficoast/belief/compiled_search.py",
        "src/azelficoast/belief/packed_evaluator.py",
        "src/azelficoast/belief/showdown_packing.py",
    ]
    required = {
        "src/azelficoast/research/contracts.py",
        "src/azelficoast/research/matched_comparison.py",
        "src/azelficoast/research/matched_search.py",
        "src/azelficoast/research/typed_search.py",
        "src/azelficoast/research/population_cohort.py",
    }
    assert required <= checked

    ruff_rules = set(config["tool"]["ruff"]["lint"]["select"])
    assert {"E4", "E7", "E9", "F", "B", "RUF012"} <= ruff_rules
    assert config["tool"]["ruff"]["lint"]["flake8-bugbear"]["extend-immutable-calls"] == [
        "azelficoast.belief.battle_promotion.BattlePromotionPolicy",
        "azelficoast.belief.evaluator.BeliefEvaluatorSpec",
        "azelficoast.belief.improvement.AdmissionPolicy",
    ]
    assert config["tool"]["ruff"]["lint"]["per-file-ignores"] == {
        "tests/**/*.py": ["RUF012"]
    }


def test_pr_ci_cancels_superseded_heads_and_observes_candidate_transition() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    pull_request = _event_block(source, "pull_request")

    assert "types: [opened, synchronize, reopened, ready_for_review]" in pull_request
    assert source.count("\nconcurrency:\n") == 1
    assert "group: ci-${{ github.event.pull_request.number || github.ref }}" in source
    assert "cancel-in-progress: true" in source


def test_ci_syntax_checks_showdown_probe_runtime() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    assert "node --check scripts/probe_real_belief_trace.cjs" in source
    assert "node --check scripts/probe_real_belief_worker.cjs" in source
    assert "node --check scripts/transition_successor_delta.cjs" in source


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
    assert 'python-version:' in source
    assert 'default: "3.13"' in source
    assert 'python-version: "${{ inputs.python-version }}"' in source
    assert "python -m pip install uv==0.12.18" in source
    assert "uv sync --locked" in source
    assert "uv sync --locked --extra simulator" in source


def test_base_static_analysis_runs_in_declared_python_version() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    static = source[source.index("  static:\n") : source.index("  test:\n")]
    assert "uses: ./.github/actions/setup-python-environment" in static
    assert 'python-version: "3.11"' in static
    assert "run: uv run mypy" in static


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


def test_training_workflow_observes_replay_bridge_changes() -> None:
    source = (WORKFLOWS / "training.yml").read_text(encoding="utf-8")

    assert '- "scripts/replay_inputlog_to_streams.cjs"' in source


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


def test_population_workflow_starts_from_canonical_frozen_cohort() -> None:
    source = (WORKFLOWS / "natural-population-strategy-fusion.yml").read_text(
        encoding="utf-8"
    )
    assert "freeze-cohort:" not in source
    assert "setup-research-evidence" in source
    assert "/tmp/azelficoast-evidence/population/manifest.json" in source


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


def test_status_move_workflow_uses_only_canonical_source() -> None:
    source = (WORKFLOWS / "natural-status-move-public-belief.yml").read_text(
        encoding="utf-8"
    )
    assert "/tmp/azelficoast-evidence/status-source.json" in source
    assert "/tmp/status-source.json" not in source
    assert "/tmp/status-source-summary.json" not in source


def test_candidate_research_emits_one_exact_head_certificate() -> None:
    source = (WORKFLOWS / "candidate-research.yml").read_text(encoding="utf-8")

    assert "\n  certify:\n" in source
    assert "needs: [plan, exact, accelerator-static]" in source
    assert "if: always() && needs.plan.result == 'success'" in source
    assert 'ACCELERATOR_STATIC_RESULT: ${{ needs.accelerator-static.result }}' in source
    assert "uv run python -m azelficoast.research.ci certify" in source
    assert "python - <<'PY'" not in source
    assert "name: candidate-research-certificate" in source
    assert "Type check accelerator frontier" in source
    assert "src/azelficoast/core/compiled_search.py" in source
    assert "src/azelficoast/belief/showdown_packing.py" in source
    assert "src/azelficoast/belief/packed_evaluator.py" in source
    assert "src/azelficoast/belief/compiled_search.py" in source


def test_showdown_revision_is_declared_once_in_repository_contract() -> None:
    action = (ROOT / ".github" / "actions" / "setup-showdown" / "action.yml").read_text(encoding="utf-8")
    revision = (ROOT / "experiments" / "showdown-revision.txt").read_text(encoding="utf-8").strip()
    runner = (ROOT / "src" / "azelficoast" / "research" / "ci.py").read_text(encoding="utf-8")

    assert len(revision) == 40
    assert "SHOWDOWN_REVISION = revision_path.read_text(encoding=\"utf-8\").strip()" in runner
    assert "default: a5df8274e85b0889bf2a9b3422a08b39732374fc" not in action
    assert "steps.revision.outputs.sha" in action
    assert "revision_path = REPOSITORY_ROOT / \"experiments\" / \"showdown-revision.txt\"" in runner


def test_showdown_consumers_observe_revision_contract_changes() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        if "uses: ./.github/actions/setup-showdown" not in source:
            continue
        assert '- "experiments/showdown-revision.txt"' in source, (
            f"{path.name} must rerun when pinned Showdown revision changes"
        )




def test_oracle_evidence_versions_its_opponent_policy_semantics() -> None:
    contract = json.loads(
        (ROOT / "experiments" / "opponent-policy-semantics.json").read_text(
            encoding="utf-8"
        )
    )
    script = (ROOT / "scripts" / "probe_real_belief_trace.cjs").read_text(
        encoding="utf-8"
    )
    architecture = (ROOT / "docs" / "search-architecture.md").read_text(
        encoding="utf-8"
    )

    assert contract["schema"] == "azelficoast.opponent-policy-semantics"
    assert contract["semantics_version"]
    assert "opponent_policy_semantics_version: OPPONENT_POLICY_SEMANTICS_VERSION" in script
    assert "versioned contract lives in" in architecture
    assert "experiments/opponent-policy-semantics.json" in architecture


def test_oracle_workflows_observe_opponent_semantics_contract_changes() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        source = path.read_text(encoding="utf-8")
        if "scripts/probe_real_belief_trace.cjs" not in source or "    paths:\n" not in source:
            continue
        assert '- "experiments/opponent-policy-semantics.json"' in source, (
            f"{path.name} must rerun when oracle opponent semantics change"
        )
