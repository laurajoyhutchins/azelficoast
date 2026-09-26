"""Freeze issue #69's outcome-blind population from an existing source artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from azelficoast.live.corpus import load_corpus
from azelficoast.research.contracts import stable_digest
from azelficoast.research.population_cohort import freeze_population
from azelficoast.research.posterior_population_contract import (
    PosteriorPopulationContractError,
    compile_execution_plan,
    validate_contract,
)


class PosteriorPopulationSelectionError(ValueError):
    """Raised when the frozen issue #69 source cannot produce its preregistered ledger."""


def _load_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PosteriorPopulationSelectionError(f"{path}: expected a JSON object")
    return value


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_binding(
    contract: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    corpus_path: str | Path,
) -> dict[str, Any]:
    checked = validate_contract(contract)
    population = checked["population"]
    assert isinstance(population, Mapping)
    expected = population["source_artifact"]
    assert isinstance(expected, Mapping)

    for field in ("workflow_run_id", "battle_count", "decision_state_count"):
        if source_metadata.get(field) != expected.get(field):
            raise PosteriorPopulationSelectionError(
                f"source metadata {field} differs from frozen contract"
            )
    if source_metadata.get("head_sha") != expected.get("generation_head_sha"):
        raise PosteriorPopulationSelectionError(
            "source generation head differs from frozen contract"
        )
    if source_metadata.get("decisions_sha256") != expected.get("decisions_sha256"):
        raise PosteriorPopulationSelectionError(
            "source decisions digest differs from frozen contract"
        )
    corpus_digest = _sha256_file(corpus_path)
    if corpus_digest != expected.get("corpus_sha256"):
        raise PosteriorPopulationSelectionError(
            "source corpus digest differs from frozen contract"
        )
    return {
        "digest": "sha256:" + corpus_digest,
        "state_count": int(expected["decision_state_count"]),
        "battle_count": int(expected["battle_count"]),
        "frozen_before_treatments": True,
        "workflow_run_id": int(expected["workflow_run_id"]),
        "artifact_id": int(expected["artifact_id"]),
        "artifact_digest": str(expected["artifact_digest"]),
        "generation_head_sha": str(expected["generation_head_sha"]),
        "decisions_sha256": str(expected["decisions_sha256"]),
    }


def freeze_issue_69_population(
    *,
    contract: Mapping[str, Any],
    source_metadata: Mapping[str, Any],
    candidates: Mapping[str, Any],
    mechanics: Mapping[str, Any],
    corpus_path: str | Path,
    selected_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checked = validate_contract(contract)
    population = checked["population"]
    assert isinstance(population, Mapping)
    source_binding = _source_binding(checked, source_metadata, corpus_path)

    plan = {
        "schema": "azelficoast.natural-population-strategy-fusion-plan",
        "schema_version": 1,
        "source_artifact": {
            "workflow_run_id": source_binding["workflow_run_id"],
            "artifact_id": source_binding["artifact_id"],
            "artifact_name": population["source_artifact"]["artifact_name"],
            "artifact_digest": source_binding["artifact_digest"],
            "head_sha": source_binding["generation_head_sha"],
            "battle_count": source_binding["battle_count"],
            "decision_state_count": source_binding["state_count"],
            "decisions_sha256": source_binding["decisions_sha256"],
            "corpus_sha256": source_binding["digest"][7:],
        },
        "showdown_commit": checked_showdown_revision(checked),
        "admissibility": {
            "generator_rounds": int(population["generator_rounds"]),
            "persistent_only": bool(population["persistent_only"]),
            "mechanics_screen_rounds": int(population["mechanics_screen_rounds"]),
            "max_exact_states": int(population["max_selected_states"]),
            "overflow_selection": str(population["overflow_selection"]),
        },
        "exact_treatment": {},
        "inference": dict(checked["inference"]),
    }
    fixtures = load_corpus(corpus_path)
    manifest = freeze_population(
        plan=plan,
        candidates_document=candidates,
        mechanics_document=mechanics,
        fixtures=fixtures,
        output_dir=selected_dir,
    )

    minimum = int(population["minimum_selected_states"])
    if int(manifest["selected_count"]) < minimum:
        raise PosteriorPopulationSelectionError(
            f"outcome-blind admission produced {manifest['selected_count']} states; "
            f"contract requires {minimum}"
        )
    if manifest.get("selection_uses_policy_result") is not False:
        raise PosteriorPopulationSelectionError(
            "population selection used treatment outcomes"
        )
    if manifest.get("frozen_before_policy_values") is not True:
        raise PosteriorPopulationSelectionError(
            "population ledger was not frozen before policy values"
        )

    manifest["issue"] = 69
    manifest["source_corpus_digest"] = source_binding["digest"]
    manifest["frozen_before_treatments"] = True
    manifest["selection_uses_treatment_outcomes"] = False
    manifest["persistent_only"] = bool(population["persistent_only"])
    manifest_digest = stable_digest(manifest)

    authority = checked["oracle_authority"]
    assert isinstance(authority, Mapping)
    bindings = {
        "enabled_treatments": list(checked["completion_posterior_treatments"]),
        "source_corpus": source_binding,
        "inclusion_ledger": {
            "digest": manifest_digest,
            "frozen_before_treatments": True,
            "selection_uses_treatment_outcomes": False,
            "selected": list(manifest["selected"]),
        },
        "evaluator": dict(checked["evaluator"]),
        "compute_budget_per_method": int(checked["compute"]["per_method_limit"]),
        "oracle_authority": {
            "kind": authority["kind"],
            "evidence_digest": stable_digest(authority),
            "realized_hidden_state_used": False,
        },
    }
    execution_plan = compile_execution_plan(checked, bindings)
    execution_plan["population_manifest_digest"] = manifest_digest
    return manifest, execution_plan


def checked_showdown_revision(contract: Mapping[str, Any]) -> str:
    path = Path(__file__).resolve().parents[3] / str(contract["showdown_revision_source"])
    revision = path.read_text(encoding="utf-8").strip()
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise PosteriorPopulationSelectionError("repository Showdown revision is malformed")
    return revision


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(value), sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", type=Path)
    parser.add_argument("source_metadata", type=Path)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("mechanics", type=Path)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--selected-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--execution-plan", required=True, type=Path)
    parser.add_argument("--matched-plan", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        manifest, execution_plan = freeze_issue_69_population(
            contract=_load_object(args.contract),
            source_metadata=_load_object(args.source_metadata),
            candidates=_load_object(args.candidates),
            mechanics=_load_object(args.mechanics),
            corpus_path=args.corpus,
            selected_dir=args.selected_dir,
        )
    except PosteriorPopulationContractError as error:
        raise PosteriorPopulationSelectionError(str(error)) from error

    _write_json(args.manifest, manifest)
    _write_json(args.execution_plan, execution_plan)
    matched_plan = execution_plan.get("matched_plan")
    if not isinstance(matched_plan, Mapping):
        raise PosteriorPopulationSelectionError("execution plan lacks matched plan")
    _write_json(args.matched_plan, matched_plan)
    print(
        json.dumps(
            {
                "selected_count": manifest["selected_count"],
                "eligible_count": manifest["eligible_count"],
                "selected_battle_count": execution_plan["selected_battle_count"],
                "expected_result_count": execution_plan["expected_result_count"],
                "source_corpus_digest": manifest["source_corpus_digest"],
                "population_manifest_digest": execution_plan["population_manifest_digest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
