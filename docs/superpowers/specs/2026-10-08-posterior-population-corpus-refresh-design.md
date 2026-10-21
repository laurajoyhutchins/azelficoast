# Posterior population corpus refresh

## Purpose

Produce a new, reproducible, outcome-blind natural decision corpus large enough
to admit at least 100 states under the existing issue #69 population contract.
The refresh addresses source scarcity observed in hosted research run
`37729840090`: the frozen source contains 1,024 battles and 25,445 decision
states; bounded mining produced 1,006 candidate world queries, and 51 states
passed live/exact reconstruction and population admission.

## Design

Add an explicitly selected source-acquisition mode to the already registered
`.github/workflows/research.yml`. GitHub only dispatches a `workflow_dispatch`
workflow when that workflow file is present on the default branch, so a new
standalone workflow could not produce a pre-merge receipt from its first branch
revision. The acquisition mode must gate off the existing candidate and hosted
study jobs; its first job generates 3,072 independent local Random Battle
battles using the existing battle policy and pinned Pokémon Showdown revision.
It freezes the decisions and corpus before any treatment runs, computes the
existing source digests, and
uploads the immutable source artifact. A dependent validation job reads the
completed artifact's run ID, artifact ID, and digest from GitHub, then runs
candidate mining at the contract's existing generator-round budget, extracts
the exact source fixtures, runs the existing pinned-Showdown mechanics screen
at its existing round budget, and invokes the existing selector.

The acquisition budget is three times the failed corpus. At the observed
51-per-1,024 yield, that estimates about 153 admissible states. This estimate
sets acquisition size only; it never certifies admission. The validation job
creates a temporary copy of the checked-in study contract with only the new
source identity and digests bound to the newly uploaded artifact. It runs the
unchanged selector and continues only if it admits at least 100 unique states.
The selector retains its current 128-state cap and outcome-blind ordering. The
complete source exclusions, candidate rejection ledger, mechanics evidence,
selection manifest, and provenance are uploaded on success and failure. A
shortfall remains a failed run with a durable negative receipt.

Once an acquisition run admits at least 100 states, update the source-artifact
binding in the posterior population contract with the exact run, artifact,
generation head, and decision/corpus/artifact digests. Then run the existing
hosted study on that exact contract head. Determinization and information-set
treatments use the same frozen population and existing inference settings.

## Authority and invariants

- Keep `minimum_selected_states` at 100 and `max_selected_states` at 128.
- Preserve the current fixture inclusion, exact reconstruction, mechanics,
  pinned Showdown, candidate mining, and outcome-blind selection rules.
- Do not select on treatment values, disagreement, regret, chosen actions, or
  battle outcomes.
- The source acquisition jobs do not run or inspect treatment outcomes.
- Do not retry the frozen old source or duplicate old battles to fill the
  cohort; generate fresh independent battles at the preregistered budget.
- The existing Python mechanics verifier and its admission boundary remain
  unchanged.
- A generated source artifact is not a positive study result. The final
  receipt must be from the exact-head hosted study and document at least 100
  admitted states.

## Failure handling

If generation, corpus freezing, candidate mining, mechanics screening, or
selection fails, upload every artifact produced up to that point, including
the full exclusion and rejection ledgers. The temporary contract overlay is
validation-only; do not commit or publish it as the study contract unless the
receipt confirms at least 100 states. Do not start treatment jobs from the
acquisition workflow. If the new corpus still admits fewer than 100 states,
keep the merge hold and treat the receipt as negative evidence; any further
source acquisition must be preregistered before treatment outcomes are
examined.

## Verification

The acquisition workflow must prove that the frozen corpus and source metadata
match their byte-level digests, that every source fixture is either accounted
for as a candidate or an exclusion, that Showdown and mechanics evidence match
the repository authority, and that selection uses no treatment results. The
hosted-study verifier remains the final gate: exact-head provenance, unchanged
mechanics admission, at least 100 distinct admitted states, matched methods,
and complete durable evidence.

## Scope

This change adds only the missing corpus-acquisition path and its provenance
receipt inside the existing research workflow. It does not change the study's
scientific question, treatments,
predictors, mechanics implementation, inclusion boundary, minimum, maximum,
or merge policy.
