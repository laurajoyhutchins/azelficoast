from __future__ import annotations

import os
import shutil
from pathlib import Path

from azelficoast.research.hosted.common import (
    EVIDENCE_ROOT,
    SHOWDOWN_ROOT,
    HostedResearchError,
    as_dict,
    load_json,
    node,
    print_json,
    python_module,
    run,
)


DEPTH_SELECTION_RUN_ID = "36140228005"
DEPTH_SELECTION_DIGEST = "6de55a45943002fa81d042f21be70b48305584f9b9d74bb6de0d06695c8ff8e4"


def _population_manifest() -> dict[str, object]:
    manifest = as_dict(
        load_json(EVIDENCE_ROOT / "population" / "manifest.json"),
        label="population manifest",
    )
    assert manifest["source_decision_state_count"] == 6485
    assert manifest["bounded_candidate_count"] == 13
    assert manifest["eligible_count"] == 10
    assert manifest["selected_count"] == 10
    assert manifest["overflow_sampling_applied"] is False
    assert manifest["frozen_before_policy_values"] is True
    assert manifest["selection_uses_policy_result"] is False
    return manifest


def natural_population_shard(shard: int) -> None:
    manifest = _population_manifest()
    selected = manifest["selected"]
    if not isinstance(selected, list):
        raise HostedResearchError("population manifest lacks selected rows")
    result_root = Path("/tmp/results")
    result_root.mkdir(parents=True, exist_ok=True)
    env = {
        "AZELFICOAST_ROOT_CHANCE_SAMPLES": "16",
        "AZELFICOAST_CONTINUATION_CHANCE_SAMPLES": "16",
        "AZELFICOAST_CHANCE_SEED_FAMILY": "0",
    }
    for raw in selected:
        row = as_dict(raw, label="population row")
        index = int(row["population_index"])
        if (index - 1) % 8 != shard:
            continue
        source = EVIDENCE_ROOT / "population" / "selected" / str(row["filename"])
        node(
            "scripts/probe_real_belief_trace.cjs",
            str(SHOWDOWN_ROOT),
            str(source),
            stdout="/tmp/oracle.json",
            env=env,
        )
        trace = result_root / f"trace-{index}.json"
        python_module(
            "azelficoast.research.verification.real_belief_trace",
            "/tmp/oracle.json",
            stdout=trace,
            env=env,
        )
        python_module(
            "azelficoast.research.population",
            "summarize",
            str(source),
            str(trace),
            "--output",
            str(result_root / f"summary-{index}.json"),
            stdout="/tmp/summary-output.json",
            env=env,
        )
        Path("/tmp/oracle.json").unlink(missing_ok=True)


def natural_population_aggregate() -> None:
    manifest = _population_manifest()
    python_module(
        "azelficoast.research.population",
        "aggregate",
        "experiments/natural-population-strategy-fusion-plan.json",
        str(EVIDENCE_ROOT / "population" / "manifest.json"),
        "/tmp/exact",
        "--output",
        "/tmp/natural-population-aggregate.json",
        stdout="/tmp/aggregate-output.json",
    )
    result = as_dict(
        load_json("/tmp/natural-population-aggregate.json"),
        label="population aggregate",
    )
    cohort = as_dict(result["cohort"], label="population cohort")
    outcomes = as_dict(result["outcomes"], label="population outcomes")
    assert cohort["source_decision_state_count"] == manifest["source_decision_state_count"]
    assert int(cohort["selected_count"]) > 0
    print_json(
        {
            "exact_states": cohort["selected_count"],
            "exact_battles": cohort["exact_battle_count"],
            "positive_bias_rate": outcomes["positive_bias_rate"],
            "policy_disagreement_rate": outcomes["policy_disagreement_rate"],
            "mean_bias": outcomes["mean_max_strategy_fusion_value_advantage"],
            "mean_regret": outcomes["mean_determinization_public_regret"],
        }
    )


def _restore_depth_selection() -> Path:
    root = Path("/tmp/selection")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        raise HostedResearchError("GITHUB_REPOSITORY is required")
    run(
        (
            "gh",
            "run",
            "download",
            DEPTH_SELECTION_RUN_ID,
            "--repo",
            repository,
            "--name",
            "natural-depth-regret-selection",
            "--dir",
            str(root),
        )
    )
    return root


def _depth_manifest(root: Path) -> dict[str, object]:
    manifest = as_dict(
        load_json(root / "depth" / "manifest.json"),
        label="depth manifest",
    )
    assert manifest["selection_uses_policy_result"] is False
    assert manifest["source_decision_state_count"] == 25445
    assert manifest["bounded_candidate_count"] == 29
    assert manifest["eligible_count"] == 21
    assert manifest["selected_count"] == 18
    assert manifest["stratum_counts"] == {
        "predictor-enriched": 12,
        "representative-lower": 6,
    }
    selected = manifest["selected"]
    assert isinstance(selected, list)
    assert len({as_dict(row, label="depth row")["battle_tag"] for row in selected}) == 12
    source = as_dict(manifest["source_artifact"], label="depth source artifact")
    assert str(source["workflow_run_id"]) == DEPTH_SELECTION_RUN_ID
    assert source["battle_count"] == 1024
    assert source["decisions_sha256"] == DEPTH_SELECTION_DIGEST
    return manifest


def natural_depth_shard(shard: int) -> None:
    selection = _restore_depth_selection()
    manifest = _depth_manifest(selection)
    selected = manifest["selected"]
    assert isinstance(selected, list)
    result_root = Path("/tmp/results")
    result_root.mkdir(parents=True, exist_ok=True)
    base_env = {
        "AZELFICOAST_ROOT_CHANCE_SAMPLES": "8",
        "AZELFICOAST_CONTINUATION_CHANCE_SAMPLES": "2",
        "AZELFICOAST_CHANCE_SEED_FAMILY": "0",
    }
    for raw in selected:
        row = as_dict(raw, label="depth row")
        index = int(row["depth_index"])
        if (index - 1) % 6 != shard:
            continue
        source = selection / "depth" / "selected" / str(row["filename"])
        shallow_env = {
            **base_env,
            "AZELFICOAST_CONTINUATION_DECISION_HORIZONS": "1",
        }
        deeper_env = {
            **base_env,
            "AZELFICOAST_CONTINUATION_DECISION_HORIZONS": "2",
        }
        node(
            "scripts/probe_real_belief_trace.cjs",
            str(SHOWDOWN_ROOT),
            str(source),
            stdout="/tmp/shallow-oracle.json",
            env=shallow_env,
        )
        python_module(
            "azelficoast.research.verification.real_belief_trace",
            "/tmp/shallow-oracle.json",
            stdout="/tmp/shallow-trace.json",
            env=shallow_env,
        )
        node(
            "scripts/probe_real_belief_trace.cjs",
            str(SHOWDOWN_ROOT),
            str(source),
            stdout="/tmp/deeper-oracle.json",
            env=deeper_env,
        )
        python_module(
            "azelficoast.research.verification.real_belief_trace",
            "/tmp/deeper-oracle.json",
            stdout="/tmp/deeper-trace.json",
            env=deeper_env,
        )
        python_module(
            "azelficoast.research.depth_population",
            "summarize",
            str(source),
            "/tmp/shallow-trace.json",
            "/tmp/deeper-trace.json",
            "--output",
            str(result_root / f"summary-{index}.json"),
            env=base_env,
        )
        for path in (
            "/tmp/shallow-oracle.json",
            "/tmp/shallow-trace.json",
            "/tmp/deeper-oracle.json",
            "/tmp/deeper-trace.json",
        ):
            Path(path).unlink(missing_ok=True)


def natural_depth_restore() -> None:
    selection = _restore_depth_selection()
    _depth_manifest(selection)


def natural_depth_aggregate() -> None:
    selection = Path("/tmp/selection")
    manifest = _depth_manifest(selection)
    python_module(
        "azelficoast.research.depth_population",
        "aggregate",
        "experiments/natural-depth-regret-plan.json",
        str(selection / "depth" / "manifest.json"),
        "/tmp/exact",
        "--output",
        "/tmp/natural-depth-regret-aggregate.json",
        stdout="/tmp/aggregate-output.json",
    )
    result = as_dict(
        load_json("/tmp/natural-depth-regret-aggregate.json"),
        label="depth aggregate",
    )
    cohort = as_dict(result["cohort"], label="depth cohort")
    outcomes = as_dict(result["outcomes"], label="depth outcomes")
    assert cohort["selected_count"] == manifest["selected_count"] == 18
    assert cohort["exact_battle_count"] == 12
    assert cohort["stratum_counts"] == {
        "predictor-enriched": 12,
        "representative-lower": 6,
    }
    print_json(
        {
            "cohort": cohort,
            "deep_value_bias": outcomes["deep_value_bias"],
            "deep_regret": outcomes["deep_regret"],
            "value_bias_change": outcomes["value_bias_change"],
            "regret_change": outcomes["regret_change"],
            "deep_positive_regret_count": outcomes["deep_positive_regret_count"],
            "deep_policy_disagreement_count": outcomes["deep_policy_disagreement_count"],
        }
    )
