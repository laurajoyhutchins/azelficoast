# Posterior population corpus refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Acquire a fresh, immutable natural decision corpus and produce a durable receipt proving that the unchanged issue #69 population pipeline admits at least 100 distinct states before treatment jobs can start.

**Architecture:** Add an explicitly selected acquisition mode to the already registered `.github/workflows/research.yml`, with separate generation and validation jobs. GitHub only accepts `workflow_dispatch` for a workflow registered on the default branch, so this lets the branch version run before merge. Acquisition mode must skip all existing candidate and hosted study jobs. Generation uses pinned Showdown and existing local battle/corpus machinery to create 3,072 fresh battles, freeze them, record source digests, and upload the corpus. Validation obtains that run's artifact identity and digest, creates a temporary contract overlay bound to the new source, and executes the existing candidate mining, exact fixture extraction, pinned Showdown mechanics screen, and population selector. It uploads all available diagnostics on success and failure. Only after a positive receipt is reviewed will the checked-in contract be rebound to the actual artifact and the existing exact-head hosted study run.

**Tech Stack:** GitHub Actions, pinned Pokémon Showdown setup action, Python CLI and research modules, Node.js mechanics screen, `uv`, `gh` REST API, JSON artifacts.

**Spec:** `docs/superpowers/specs/2026-10-08-posterior-population-corpus-refresh-design.md`

## Global Constraints

- Keep the contract minimum at 100 and maximum at 128. Do not weaken or bypass selector validation.
- Preserve the current exact reconstruction, fixture inclusion, mechanics screen, pinned Showdown revision, candidate mining budget, outcome-blind ordering, and unchanged Python mechanics verifier.
- Generate 3,072 fresh independent battles; do not replay or duplicate the failed 1,024-battle source.
- Do not use treatment values, disagreement, regret, chosen actions, or battle outcomes to select states. Do not start treatment jobs from the acquisition workflow.
- Bind provenance to the exact generation head, source workflow run, artifact ID and digest, battle/state counts, and corpus/decision digests. The temporary contract overlay is validation-only.
- A generation artifact or acquisition receipt is not a positive study result. Keep the merge hold until the existing exact-head hosted study emits a positive receipt with at least 100 distinct admitted states and all current gates pass.

## Review Focus

- Workflow dependency graph ensures validation cannot run before the immutable source artifact is uploaded and cannot launch treatment jobs.
- Failure paths retain the source, exclusion ledger, candidate rejection ledger, mechanics evidence, selector output, and provenance produced up to the failure point.
- Dynamic artifact binding rejects missing, ambiguous, or mismatched artifact/run/head/digest metadata.
- Contract overlay changes only source-artifact binding fields; all admission settings and mechanics authorities remain byte-for-byte equivalent.
- The final committed contract uses actual positive-receipt provenance, never an estimated yield or placeholder ID.

---

### Task 1: Add a testable source-artifact binding and validation receipt helper

**Files:**
- Create: `src/azelficoast/research/posterior_population_acquisition.py`
- Create: `tests/test_posterior_population_acquisition.py`
- Reference: `src/azelficoast/research/posterior_population_selection.py`
- Reference: `src/azelficoast/research/posterior_population_contract.py`

- [x] Add failing unit tests for: binding a source metadata document to a temporary contract overlay; verifying run ID, artifact ID/digest, head SHA, counts, decision digest, and corpus digest; and rejecting missing or conflicting provenance.
- [x] Add a test that compares the original and overlaid contract and proves every field outside `population.source_artifact` is unchanged, including minimum 100, maximum 128, candidate budget, mechanics rounds, and outcome-blind policy.
- [x] Implement the smallest pure helper(s) needed to build the temporary overlay and an acquisition receipt from the existing selector outputs. Reuse contract validation and digest utilities where available; do not change the selector or mechanics admission boundary.
- [x] Add failure-receipt tests confirming a below-minimum result remains negative, records actual selected count and required minimum, and cannot be mistaken for a successful study receipt.
- [x] Run: `uv run pytest tests/test_posterior_population_acquisition.py tests/test_posterior_population_contract.py tests/test_hosted_research_contracts.py` — expect all tests to pass.

### Task 2: Add the 3,072-battle immutable source acquisition workflow

**Files:**
- Update: `.github/workflows/research.yml`
- Update: `src/azelficoast/research/posterior_population_acquisition.py`
- Update: `tests/test_posterior_population_acquisition.py`
- Create: `tests/test_posterior_population_acquisition_workflow.py`

- [x] Use the existing `azelficoast local` and `azelficoast corpus build` commands with pinned local Showdown; record the exact CLI contract in the testable acquisition helper invoked by the workflow.
- [x] Add a Boolean `workflow_dispatch` acquisition input (default false) and fence candidate/hosted-study planning jobs off when it is true. Add a fixed, reviewed generation budget of 3,072.
- [x] Generate fresh independent local Random Battle battles with the existing baseline policy; freeze decisions and corpus before any treatment execution; calculate byte-level corpus and decisions digests and exact counts.
- [x] Write source metadata containing schema, generation run ID/head SHA, pinned Showdown commit, battle count, decision-state count, decisions digest, and corpus digest. Fail closed if counts or digests do not match the frozen files.
- [x] Upload only immutable source decisions, corpus, metadata, and provenance with `if-no-files-found: error`; expose artifact name/run identity for the dependent validation job. Do not upload outcome result files.
- [x] Add workflow tests/assertions that the budget is exactly 3,072, setup uses repository Showdown authority, and acquisition dispatch cannot launch candidate/treatment jobs.
- [x] Run focused workflow-contract tests and YAML/actionlint validation where available — expect valid workflow structure and all assertions to pass.

### Task 3: Validate the new corpus through the unchanged admission pipeline

**Files:**
- Update: `.github/workflows/research.yml`
- Update: `src/azelficoast/research/posterior_population_acquisition.py`
- Update: `src/azelficoast/research/population_cohort.py`
- Update: `src/azelficoast/research/posterior_population_selection.py`
- Test/update: `tests/test_posterior_population_acquisition.py`
- Test/update: `tests/test_population_study.py`
- Reuse: `src/azelficoast/research/posterior_population_selection.py`, current candidate miner, and `showdown/research/belief/screen_public_belief_speed_forks.cjs`

- [x] Add tests for the receipt state machine: validation starts only after source upload; a selector shortfall fails with a durable negative receipt; at least 100 distinct states yields an acquisition-positive receipt but not a study-positive receipt; more than 128 is capped by the unchanged selector.
- [x] Add a dependent validation job that retrieves the exact artifact by run and artifact ID through GitHub's artifact API, verifies its digest and metadata, and rejects ambiguous or mismatched artifacts.
- [x] Build the temporary contract overlay from the checked-in contract and verified source metadata. Assert only source binding changes, then run the existing bounded candidate miner, exact source-fixture extraction, pinned-Showdown mechanics screen, and existing selector in their current configured budgets.
- [x] Preserve the existing Python mechanics verifier and its admission boundary unchanged. Persist per-fixture rejection and cap-overflow ledger rows before the selector's existing minimum check; confirm via diff that no verifier implementation, threshold, or selected-state ordering changed.
- [x] Upload the full source, exclusion and rejection ledgers, mechanics evidence, selected manifest if present, provenance, and success/failure receipt with `if: always()`; ensure a validation failure cannot be converted to success by artifact upload.
- [x] Run: focused Python tests; `uv run python -m azelficoast.research.hosted plan --requested posterior-stratified-population`; and workflow/YAML tests — expect no treatment job in the acquisition graph and all expected validation outputs.

### Task 4: Run acquisition and bind only a positive receipt to the study contract

**Files:**
- Update only after acquisition succeeds: `experiments/data/posterior-stratified-population-contract.json`
- Verify: `.github/workflows/research.yml` and hosted exact-head artifacts

- [ ] Dispatch the registered research workflow's acquisition mode at the implementation head and inspect its source and validation artifacts, including every exclusion/rejection reason and the final receipt.
- [ ] If fewer than 100 distinct states are admitted, retain the negative receipt and merge hold; do not modify the study contract or start treatments.
- [ ] If at least 100 distinct states are admitted, update only the contract's `population.source_artifact` identity/digest fields from the actual generation and acquisition artifacts. Keep minimum 100, maximum 128, all admission settings, treatments, and inference settings unchanged.
- [ ] Run contract and hosted-plan checks, then run the existing hosted study on that exact contract head. Do not accept the acquisition receipt as the study receipt.
- [ ] Verify the final exact-head study receipt reports at least 100 distinct admitted states, matching methods/population, exact source provenance, pinned Showdown, unchanged mechanics admission, and complete durable evidence before lifting the merge hold.

### Task 5: Final verification and review

- [x] Run the unchanged Python mechanics verifier and the relevant posterior-population, source-corpus, and hosted-contract tests.
- [x] Run `git diff --check`; inspect the complete diff for unintended changes to population thresholds, mechanics code, exclusions, treatment settings, or the Python verifier.
- [ ] Record the acquisition run ID, artifact ID/digest, exact selected count, and exact-head hosted study run/receipt in the PR evidence only after they exist.
- [ ] Keep the PR blocked if any exact-head, 100-state, mechanics, provenance, matched-method, or evidence gate fails.
