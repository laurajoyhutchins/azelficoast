# Research Workflow Authority Recut Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the research experiment contract, rather than workflow YAML, the sole authority for scientific parameters and hosted evidence semantics.

**Architecture:** Recut the existing #118 change onto current main, preserve each experiment's current contract and output, and keep workflows limited to triggers, runner topology, shared setup, hosted command invocation, and artifact transport. Validate the authority boundary with repository-wide workflow tests and exact-head GitHub Actions checks.

**Tech Stack:** Python 3.12+, pytest, uv, GitHub Actions YAML, GitHub Contents/Git Data APIs.

**Spec:** `docs/superpowers/specs/2026-09-26-matched-population-experiment-design.md`

## Global Constraints

- Scientific parameters and source identities come from one versioned experiment contract.
- Workflows distribute shards and transport artifacts; they do not duplicate treatment values, corpus digests, inclusion rules, or scientific assertions.
- Preserve existing experiment implementations, plans, acceptance conditions, and evidence outputs.
- The current frozen 18-state cohort remains a pipeline pilot only.
- Preserve fail-closed validation and exact Pokémon Showdown revision authority.
- Keep experimental opponent semantics distinct and content-addressed.
- Do not broaden mechanics as part of this refactor.

## Review Focus

- A workflow can reintroduce scientific semantics through inline Python, Node, CLI flags, environment values, or long digests; pin all forms with `test_research_semantics_do_not_live_in_workflow_yaml`.
- Setup-showdown can drift from the repository's revision authority; pin malformed, absent, and overridden revision handling with `test_showdown_action_reads_authority_from_repository_data` and action validation tests.
- Candidate certification can diverge from the selected exact matrix; pin certificate derivation and failed exact evidence with `test_candidate_research_certificate_semantics_live_in_python` and `test_candidate_certificate_fails_closed_on_missing_exact_evidence`.
- Hosted commands can accept an unregistered witness or fixture; pin membership to the repository-owned registry with `test_hosted_witness_membership_is_not_encoded_in_yaml`.
- Recutting old #118 files can overwrite newer main behavior in overlapping files; preserve main changes and run the full pytest suite plus the exact-head merge gate.

---

### Task 1: Add workflow-authority regression tests

**Files:**
- Modify: `tests/test_ci_contract.py`
- Modify: `tests/test_research_ci.py`
- Modify: `tests/test_replay_worlds.py`

**Interfaces:**
- Consumes: existing `WORKFLOWS`, experiment registry, Showdown revision, and candidate certificate helpers.
- Produces: regression tests that fail on current main because workflow YAML still contains scientific parameters and inline certificate logic.

- [ ] **Step 1: Add the four authority tests from the approved #118 contract**: `test_research_semantics_do_not_live_in_workflow_yaml`, `test_showdown_action_reads_authority_from_repository_data`, `test_candidate_research_certificate_semantics_live_in_python`, and `test_hosted_witness_membership_is_not_encoded_in_yaml`; add fail-closed candidate certificate test if the current helper exposes it.
- [ ] **Step 2: Run the targeted tests on a draft PR exact head** with `uv run pytest tests/test_ci_contract.py tests/test_research_ci.py tests/test_replay_worlds.py -q`. Expected: failures specifically identify inline research semantics, workflow-owned digests/parameters, and workflow-owned Showdown/certificate authority.
- [ ] **Step 3: Commit the failing tests** as `test: enforce hosted research authority boundary`.

### Task 2: Recut hosted contracts onto latest main

**Files:**
- Create: `.github/actions/setup-showdown/action.yml` updates and `experiments/showdown-revision.txt`
- Create: `src/azelficoast/research/hosted/{__init__.py,__main__.py,common.py,belief.py,mechanics.py,population.py}`
- Modify: all 22 workflow files listed in PR #118
- Modify: `src/azelficoast/research/ci.py`
- Modify: `src/azelficoast/research/verification/replay_worlds.py`
- Modify: the three test files in Task 1

**Interfaces:**
- Consumes: failing tests from Task 1 and current-main versions of every overlapping file.
- Produces: `python -m azelficoast.research.hosted <command>` as the workflow boundary; repository-owned revision/evidence/fixture authority; candidate certificate generation in Python.

- [ ] **Step 1: Port the 28 non-overlapping #118 files from its reviewed head and merge the six overlapping files against current main**, preserving current-main changes in `.github/workflows/candidate-research.yml`, `src/azelficoast/research/ci.py`, `src/azelficoast/research/verification/replay_worlds.py`, `tests/test_ci_contract.py`, `tests/test_replay_worlds.py`, and `tests/test_research_ci.py`.
- [ ] **Step 2: Move workflow-embedded selections, parameters, assertions, and digests into hosted Python contracts**; retain only runner topology, setup, hosted invocation, and artifact transport in YAML.
- [ ] **Step 3: Run targeted authority tests** with `uv run pytest tests/test_ci_contract.py tests/test_research_ci.py tests/test_replay_worlds.py -q`. Expected: all pass and each workflow invokes its hosted command.
- [ ] **Step 4: Run the complete pytest suite** with `uv run pytest -q`. Expected: all tests pass; no scientific parameter or source digest remains in workflow YAML.
- [ ] **Step 5: Commit the recut** as `refactor: make hosted research contracts authoritative`.

### Task 3: Verify the workflow-authority recut

**Files:**
- Verify: all changed workflow and hosted contract files.
- Verify: GitHub Actions exact-head run artifacts and statuses.

**Interfaces:**
- Consumes: Task 2 branch commit.
- Produces: exact-head CI evidence proving the hosted refactor preserves existing experiment gates.

- [ ] **Step 1: Run the ordinary pull-request merge gate** on the draft PR and record exact head, test result, and workflow URL.
- [ ] **Step 2: Read every failed job/log; fix only regressions introduced by the recut**, preserving fail-closed behavior.
- [ ] **Step 3: Verify the complete workflow matrix and artifact names remain represented in the repository-owned hosted registry**; require no missing experiment and no duplicate workflow-owned semantics.
- [ ] **Step 4: Mark the PR ready for review only after the cheap gate is green**; do not run candidate-only scientific jobs in this plan.

**Commit order:** Task 1 failing-test commit, Task 2 implementation commit, any test-driven repair commits. Branch remains based on the approved design branch's exact main base.
