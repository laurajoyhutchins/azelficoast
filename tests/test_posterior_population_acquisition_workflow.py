from __future__ import annotations

import re
from pathlib import Path


WORKFLOW = Path(__file__).parents[1] / ".github/workflows/research.yml"


def _job(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [a-z0-9-]+:|\Z)", source
    )
    assert match is not None, f"workflow job {name} is missing"
    return match.group(1)


def test_acquisition_dispatch_has_an_explicit_fenced_mode() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")

    assert "posterior_population_acquisition:" in source
    assert "default: false" in source
    assert "posterior-population-acquisition-generate:" in source
    assert "posterior-population-acquisition-validate:" in source
    generate = _job(source, "posterior-population-acquisition-generate")
    assert "artifact_digest: sha256:${{ steps.upload.outputs.artifact-digest }}" in generate
    validate = _job(source, "posterior-population-acquisition-validate")
    assert "jq -r '.digest'" in validate and '"$ARTIFACT_DIGEST"' in validate
    assert validate.index("name: Initialize validation receipt") < validate.index(
        "uses: actions/checkout@v7"
    )
    for job_name in ("candidate-plan", "hosted-plan"):
        assert "inputs.posterior_population_acquisition != true" in _job(source, job_name)
    assert "name: Stage every available source file" in generate
    assert "path: /tmp/posterior-population-source-upload" in generate


def test_acquisition_is_dependency_ordered_and_cannot_start_treatments() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    generate = _job(source, "posterior-population-acquisition-generate")
    validate = _job(source, "posterior-population-acquisition-validate")

    assert "posterior_population_acquisition" in generate
    assert "generate-source" in generate
    assert "./.github/actions/setup-showdown" in generate
    assert "needs: posterior-population-acquisition-generate" in validate
    assert "hosted-run" not in generate + validate
    assert "posterior_population_acquisition" in validate
    assert "validate-source" in validate
    assert "posterior_population_selection" not in validate
    assert "screen_public_belief_speed_forks.cjs" not in validate
    assert "--rounds " not in validate


def test_acquisition_upload_omits_outcome_result_file() -> None:
    source = WORKFLOW.read_text(encoding="utf-8")
    generate = _job(source, "posterior-population-acquisition-generate")
    upload = generate.split("actions/upload-artifact@v4", 1)[1]
    assert "results.jsonl" not in upload
    assert "source-metadata.json" in generate
    assert "corpus.jsonl" in generate
    assert "decisions.jsonl" in generate
