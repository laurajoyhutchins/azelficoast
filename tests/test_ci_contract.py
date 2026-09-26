from __future__ import annotations

import json
from pathlib import Path
import re
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
        if (
            line.startswith("  ")
            and not line.startswith("    ")
            and line.rstrip().endswith(":")
        ):
            break
        block.append(line)
    return "".join(block)


def test_workflow_surface_is_small_and_authority_specific() -> None:
    assert {path.name for path in WORKFLOWS.glob("*.yml")} == {
        "ci.yml",
        "research.yml",
        "showdown-build-cache.yml",
        "training.yml",
    }


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
    assert {
        "src/azelficoast/research/contracts.py",
        "src/azelficoast/research/matched_comparison.py",
        "src/azelficoast/research/matched_search.py",
        "src/azelficoast/research/typed_search.py",
        "src/azelficoast/research/population_cohort.py",
    } <= checked

    ruff_rules = set(config["tool"]["ruff"]["lint"]["select"])
    assert {"E4", "E7", "E9", "F", "B", "RUF012"} <= ruff_rules


def test_pr_ci_cancels_superseded_heads() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
    pull_request = _event_block(source, "pull_request")

    assert "types: [opened, synchronize, reopened, ready_for_review]" in pull_request
    assert source.count("\nconcurrency:\n") == 1
    assert "group: ci-${{ github.event.pull_request.number || github.ref }}" in source
    assert "cancel-in-progress: true" in source


def test_ci_checks_entire_javascript_script_frontier() -> None:
    source = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")

    assert "npm run lint:scripts" in source
    assert "npm run typecheck:scripts" in source
    assert "find scripts -type f -name '*.cjs' -print0" in source
    assert 'node --check "$script"' in source
    assert "node --check scripts/probe_real_belief_trace.cjs" not in source


def test_research_workflow_is_exact_head_fenced_and_manually_runnable() -> None:
    source = (WORKFLOWS / "research.yml").read_text(encoding="utf-8")
    pull_request = _event_block(source, "pull_request")

    assert "types: [opened, synchronize, reopened, ready_for_review]" in pull_request
    assert "  workflow_dispatch:\n" in source
    assert "github.event.pull_request.draft == false" in source
    assert "github.event.pull_request.head.sha || github.sha" in source
    assert "group: research-${{ github.event.pull_request.number || github.ref }}" in source
    assert "cancel-in-progress: true" in source


def test_hosted_research_preserves_candidate_only_cadence() -> None:
    source = (WORKFLOWS / "research.yml").read_text(encoding="utf-8")
    hosted = source[source.index("  hosted-plan:\n") : source.index("  hosted-run:\n")]

    assert (
        "if: github.event_name == 'workflow_dispatch' "
        "|| github.event.action == 'ready_for_review'"
    ) in hosted
    assert "github.event.pull_request.draft == false" not in hosted


def test_shared_python_environment_owns_locked_dependency_resolution() -> None:
    source = (
        ROOT / ".github" / "actions" / "setup-python-environment" / "action.yml"
    ).read_text(encoding="utf-8")

    assert "uses: actions/setup-python@v7" in source
    assert 'default: "3.13"' in source
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
        assert "uses: ./.github/actions/setup-python-environment" in source
        assert "actions/setup-python@v7" not in source
        assert "pip install uv" not in source
        assert "uv sync --locked" not in source
        checked.append(path.name)

    assert checked == ["ci.yml", "research.yml", "training.yml"]


def test_evidence_setup_runs_after_python_environment() -> None:
    for workflow in ("ci.yml", "research.yml"):
        source = (WORKFLOWS / workflow).read_text(encoding="utf-8")
        if "uses: ./.github/actions/setup-research-evidence" not in source:
            continue
        assert source.index("uses: ./.github/actions/setup-python-environment") < source.index(
            "uses: ./.github/actions/setup-research-evidence"
        )


def test_training_workflow_observes_replay_bridge_changes() -> None:
    source = (WORKFLOWS / "training.yml").read_text(encoding="utf-8")
    assert '- "scripts/replay_inputlog_to_streams.cjs"' in source


def test_repository_evidence_admission_is_owned_by_python() -> None:
    manifest = ROOT / "experiments" / "evidence" / "canonical-evidence.json"
    bundle = ROOT / "experiments" / "evidence" / "canonical-evidence.tar.gz"
    action = ROOT / ".github" / "actions" / "setup-research-evidence" / "action.yml"
    authority = ROOT / "src" / "azelficoast" / "research" / "evidence.py"

    assert manifest.is_file()
    assert bundle.is_file()
    assert authority.is_file()
    source = action.read_text(encoding="utf-8")
    assert "uv run python -m azelficoast.research.evidence unpack" in source
    assert "sha256sum" not in source
    assert "tar -xzf" not in source


def test_active_population_plan_contains_no_superseded_cohort_history() -> None:
    source = (
        ROOT / "experiments" / "natural-population-strategy-fusion-plan.json"
    ).read_text(encoding="utf-8")
    assert '"superseded_cohort"' not in source
    assert '"superseded_cohort_after_forme_bug"' not in source
    assert '"preregistration_amendment' not in source
    assert '"canonical_cohort"' in source


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


def test_research_workflow_calls_only_generic_research_entrypoints() -> None:
    source = (WORKFLOWS / "research.yml").read_text(encoding="utf-8")

    assert "azelficoast.research.ci plan" in source
    assert "azelficoast.research.ci run" in source
    assert "azelficoast.research.ci certify" in source
    assert "azelficoast.research.hosted plan" in source
    assert "azelficoast.research.hosted run-unit" in source
    assert "azelficoast.research.hosted prepare-aggregate" in source
    assert "azelficoast.research.hosted aggregate" in source

    for operation in (
        "natural-depth-shard",
        "natural-population-shard",
        "public-belief-exact",
        "conditional-team-prior",
        "protect-action-survival",
    ):
        assert f"azelficoast.research.hosted {operation}" not in source


def test_hosted_aggregates_are_not_blocked_by_unrelated_matrix_failure() -> None:
    source = (WORKFLOWS / "research.yml").read_text(encoding="utf-8")
    aggregate = source[source.index("  hosted-aggregate:\n") :]

    assert "needs.hosted-run.result != 'cancelled'" in aggregate
    assert "needs.hosted-run.result == 'success'" not in aggregate
    assert "pattern: ${{ matrix.download_pattern }}" in aggregate
    assert "if-no-files-found: ${{ matrix.if_no_files }}" in aggregate


def test_candidate_research_emits_one_exact_head_certificate() -> None:
    source = (WORKFLOWS / "research.yml").read_text(encoding="utf-8")

    assert "\n  candidate-certify:\n" in source
    assert "needs: [candidate-plan, candidate-exact, accelerator-static]" in source
    assert "if: always() && needs.candidate-plan.result == 'success'" in source
    assert (
        'ACCELERATOR_STATIC_RESULT: ${{ needs.accelerator-static.result }}'
        in source
    )
    assert "uv run python -m azelficoast.research.ci certify" in source
    assert "name: candidate-research-certificate" in source
    assert "Type check accelerator frontier" in source


def test_candidate_specs_are_repository_data_not_runner_code() -> None:
    runner = (ROOT / "src" / "azelficoast" / "research" / "ci.py").read_text(
        encoding="utf-8"
    )
    loader = (
        ROOT / "src" / "azelficoast" / "research" / "experiment_contracts.py"
    ).read_text(encoding="utf-8")
    contract = json.loads(
        (ROOT / "experiments" / "candidate-research-contracts.json").read_text(
            encoding="utf-8"
        )
    )

    assert "Check(" not in runner
    assert "paths=(" not in runner
    assert "_spec(" not in loader
    assert "adaptive-execution-experiment.json" not in loader
    assert contract["schema"] == "azelficoast.candidate-research-contracts"
    assert len(contract["experiments"]) == 16
    assert '"experiments/candidate-research-contracts.json"' in loader
    assert '".github/workflows/research.yml"' in loader
    assert "candidate-research.yml" not in loader


def test_showdown_revision_is_declared_once_in_repository_contract() -> None:
    action = (
        ROOT / ".github" / "actions" / "setup-showdown" / "action.yml"
    ).read_text(encoding="utf-8")
    revision = (
        ROOT / "experiments" / "showdown-revision.txt"
    ).read_text(encoding="utf-8").strip()
    runner = (
        ROOT / "src" / "azelficoast" / "research" / "ci.py"
    ).read_text(encoding="utf-8")
    probe = (ROOT / "scripts" / "probe_real_belief_trace.cjs").read_text(
        encoding="utf-8"
    )

    assert re.fullmatch(r"[0-9a-f]{40}", revision)
    assert "steps.revision.outputs.sha" in action
    assert 'REPOSITORY_ROOT / "experiments" / "showdown-revision.txt"' in runner
    assert "../experiments/showdown-revision.txt" in probe
    assert revision not in runner
    assert revision not in probe


def test_showdown_workflow_triggers_observe_revision_contract() -> None:
    research = (WORKFLOWS / "research.yml").read_text(encoding="utf-8")
    training = (WORKFLOWS / "training.yml").read_text(encoding="utf-8")
    cache = (WORKFLOWS / "showdown-build-cache.yml").read_text(encoding="utf-8")

    assert '- "experiments/**"' in research
    assert '- "experiments/showdown-revision.txt"' in training
    assert '- "experiments/showdown-revision.txt"' in cache


def test_oracle_evidence_versions_its_opponent_policy_semantics() -> None:
    contract = json.loads(
        (ROOT / "experiments" / "opponent-policy-semantics.json").read_text(
            encoding="utf-8"
        )
    )
    script = (ROOT / "scripts" / "probe_real_belief_trace.cjs").read_text(
        encoding="utf-8"
    )

    assert contract["schema"] == "azelficoast.opponent-policy-semantics"
    assert contract["semantics_version"]
    assert (
        "opponent_policy_semantics_version: OPPONENT_POLICY_SEMANTICS_VERSION"
        in script
    )
