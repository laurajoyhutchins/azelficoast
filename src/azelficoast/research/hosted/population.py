from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Mapping

from azelficoast.research.hosted.common import (
    EVIDENCE_ROOT,
    REPOSITORY_ROOT,
    SHOWDOWN_ROOT,
    HostedResearchError,
    as_dict,
    load_json,
    node,
    print_json,
    python_module,
    run,
)


POPULATION_PLAN_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "natural-population-strategy-fusion-plan.json"
)
DEPTH_PLAN_PATH = REPOSITORY_ROOT / "experiments" / "natural-depth-regret-plan.json"


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HostedResearchError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HostedResearchError(f"{label} must be a non-negative integer")
    return value


def _population_plan() -> dict[str, object]:
    plan = as_dict(load_json(POPULATION_PLAN_PATH), label="population plan")
    if plan.get("schema") != "azelficoast.natural-population-strategy-fusion-plan":
        raise HostedResearchError("unexpected natural population plan schema")
    return plan


def _depth_plan() -> dict[str, object]:
    plan = as_dict(load_json(DEPTH_PLAN_PATH), label="depth plan")
    if plan.get("schema") != "azelficoast.natural-depth-regret-plan":
        raise HostedResearchError("unexpected natural depth plan schema")
    return plan


def _population_manifest() -> dict[str, object]:
    plan = _population_plan()
    expected = as_dict(plan["canonical_cohort"], label="canonical population cohort")
    manifest = as_dict(
        load_json(EVIDENCE_ROOT / "population" / "manifest.json"),
        label="population manifest",
    )

    for field in (
        "source_decision_state_count",
        "bounded_candidate_count",
        "eligible_count",
        "selected_count",
        "overflow_sampling_applied",
        "frozen_before_policy_values",
        "selection_uses_policy_result",
    ):
        if manifest.get(field) != expected.get(field):
            raise HostedResearchError(
                f"population manifest {field} does not match its frozen plan"
            )
    return manifest


def _treatment_environment(
    treatment: Mapping[str, object],
    *,
    label: str,
) -> dict[str, str]:
    return {
        "AZELFICOAST_ROOT_CHANCE_SAMPLES": str(
            _positive_int(
                treatment.get("root_chance_samples"),
                label=f"{label}.root_chance_samples",
            )
        ),
        "AZELFICOAST_CONTINUATION_CHANCE_SAMPLES": str(
            _positive_int(
                treatment.get("continuation_chance_samples"),
                label=f"{label}.continuation_chance_samples",
            )
        ),
        "AZELFICOAST_CHANCE_SEED_FAMILY": str(
            _nonnegative_int(
                treatment.get("chance_seed_family"),
                label=f"{label}.chance_seed_family",
            )
        ),
    }


def _validate_shard(shard: int, shard_count: int, *, label: str) -> None:
    if shard_count < 1 or shard < 0 or shard >= shard_count:
        raise HostedResearchError(
            f"{label} shard {shard} is outside 0..{shard_count - 1}"
        )


def natural_population_shard(shard: int, shard_count: int) -> None:
    _validate_shard(shard, shard_count, label="population")
    plan = _population_plan()
    treatment = as_dict(plan["exact_treatment"], label="population treatment")
    manifest = _population_manifest()
    selected = manifest["selected"]
    if not isinstance(selected, list):
        raise HostedResearchError("population manifest lacks selected rows")

    result_root = Path("/tmp/results")
    result_root.mkdir(parents=True, exist_ok=True)
    env = _treatment_environment(treatment, label="population treatment")

    for raw in selected:
        row = as_dict(raw, label="population row")
        index = int(row["population_index"])
        if (index - 1) % shard_count != shard:
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
        str(POPULATION_PLAN_PATH),
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
    if cohort["source_decision_state_count"] != manifest["source_decision_state_count"]:
        raise HostedResearchError("population aggregate changed source cohort")
    if int(cohort["selected_count"]) <= 0:
        raise HostedResearchError("population aggregate contains no exact states")
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


def _frozen_depth_cohort() -> dict[str, object]:
    plan = _depth_plan()
    return as_dict(plan["frozen_cohort"], label="frozen depth cohort")


def _restore_depth_selection() -> Path:
    frozen = _frozen_depth_cohort()
    root = Path("/tmp/selection")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        raise HostedResearchError("GITHUB_REPOSITORY is required")
    run_id = _positive_int(
        frozen.get("workflow_run_id"),
        label="frozen depth workflow_run_id",
    )
    artifact_name = frozen.get("artifact_name")
    if not isinstance(artifact_name, str) or not artifact_name:
        raise HostedResearchError("frozen depth artifact_name must be non-empty")

    run(
        (
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            repository,
            "--name",
            artifact_name,
            "--dir",
            str(root),
        )
    )
    return root


def _depth_manifest(root: Path) -> dict[str, object]:
    plan = _depth_plan()
    frozen = as_dict(plan["frozen_cohort"], label="frozen depth cohort")
    fresh = as_dict(plan["fresh_population"], label="fresh depth population")
    manifest = as_dict(
        load_json(root / "depth" / "manifest.json"),
        label="depth manifest",
    )

    expected_fields = {
        "selection_uses_policy_result": frozen["selection_uses_policy_result"],
        "source_decision_state_count": frozen["source_decision_state_count"],
        "bounded_candidate_count": frozen["bounded_candidate_count"],
        "eligible_count": frozen["eligible_count"],
        "selected_count": frozen["selected_state_count"],
        "stratum_counts": frozen["stratum_counts"],
    }
    for field, expected in expected_fields.items():
        if manifest.get(field) != expected:
            raise HostedResearchError(
                f"depth manifest {field} does not match its frozen plan"
            )

    selected = manifest.get("selected")
    if not isinstance(selected, list):
        raise HostedResearchError("depth manifest lacks selected rows")
    battle_count = len(
        {
            as_dict(row, label="depth row")["battle_tag"]
            for row in selected
        }
    )
    if battle_count != frozen["selected_battle_count"]:
        raise HostedResearchError("depth manifest battle count changed")

    source = as_dict(manifest["source_artifact"], label="depth source artifact")
    if source.get("workflow_run_id") != frozen["workflow_run_id"]:
        raise HostedResearchError("depth source workflow locator changed")
    if source.get("battle_count") != fresh["battle_count"]:
        raise HostedResearchError("depth source battle count changed")
    if source.get("decisions_sha256") != frozen["decisions_sha256"]:
        raise HostedResearchError("depth source digest changed")
    return manifest


def natural_depth_shard(shard: int, shard_count: int) -> None:
    _validate_shard(shard, shard_count, label="depth")
    plan = _depth_plan()
    treatment = as_dict(plan["paired_treatment"], label="depth treatment")
    selection = _restore_depth_selection()
    manifest = _depth_manifest(selection)
    selected = manifest["selected"]
    if not isinstance(selected, list):
        raise HostedResearchError("depth manifest lacks selected rows")

    result_root = Path("/tmp/results")
    result_root.mkdir(parents=True, exist_ok=True)
    base_env = _treatment_environment(treatment, label="depth treatment")
    shallow_horizon = _positive_int(
        treatment.get("shallow_continuation_decision_horizons"),
        label="depth treatment shallow horizon",
    )
    deeper_horizon = _positive_int(
        treatment.get("deeper_continuation_decision_horizons"),
        label="depth treatment deeper horizon",
    )
    if deeper_horizon <= shallow_horizon:
        raise HostedResearchError("deeper treatment must exceed shallow horizon")

    for raw in selected:
        row = as_dict(raw, label="depth row")
        index = int(row["depth_index"])
        if (index - 1) % shard_count != shard:
            continue
        source = selection / "depth" / "selected" / str(row["filename"])
        shallow_env = {
            **base_env,
            "AZELFICOAST_CONTINUATION_DECISION_HORIZONS": str(shallow_horizon),
        }
        deeper_env = {
            **base_env,
            "AZELFICOAST_CONTINUATION_DECISION_HORIZONS": str(deeper_horizon),
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
    frozen = _frozen_depth_cohort()
    selection = Path("/tmp/selection")
    manifest = _depth_manifest(selection)
    python_module(
        "azelficoast.research.depth_population",
        "aggregate",
        str(DEPTH_PLAN_PATH),
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
    if cohort["selected_count"] != manifest["selected_count"]:
        raise HostedResearchError("depth aggregate changed selected count")
    if cohort["selected_count"] != frozen["selected_state_count"]:
        raise HostedResearchError("depth aggregate changed frozen state count")
    if cohort["exact_battle_count"] != frozen["selected_battle_count"]:
        raise HostedResearchError("depth aggregate changed frozen battle count")
    if cohort["stratum_counts"] != frozen["stratum_counts"]:
        raise HostedResearchError("depth aggregate changed frozen strata")

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
