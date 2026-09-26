# Design: Issue #69 matched population experiment at scale

**Status:** proposed for review; no implementation changes are included yet.  
**Base:** `main` at `5fd73fc8b54890b53f8236ad6a0d27414931874d` (2026-09-26).

## Decision

Deliver issue #69 as a staged, preregistered factorial study. The same frozen public decision states will be evaluated under oracle, generator-faithful, and practical posterior treatments; determinization and information-set search; and fixed, preregistered depths. A held-out battle study will follow only after the state-level matrix and evaluator are frozen.

Scientific parameters and source identities will come from one versioned experiment contract. Workflows will dispatch the contract and distribute shards; they will not duplicate treatment values, corpus digests, inclusion rules, or scientific assertions.

## Why this work comes first

The repository already has the core design and most of the per-state/population contracts:

- Issue [#69](https://github.com/laurajoyhutchins/azelficoast/issues/69) specifies the posterior-stratified natural-state experiment and its estimands.
- PRs [#70](https://github.com/laurajoyhutchins/azelficoast/pull/70) and [#73](https://github.com/laurajoyhutchins/azelficoast/pull/73) landed matched packets, posterior boundaries, budget receipts, matrix validation, and battle-clustered aggregation. The current receipt executor still rejects depths other than 1.
- PR [#118](https://github.com/laurajoyhutchins/azelficoast/pull/118) is stale against current main and dirty; recut its intended workflow-authority change rather than merging its old head.
- The latest depth cohort has 18 selected states from 12 battles, after 1,024 source battles yielded 21 eligible states from 25,445 decision states. Its enriched/lower-stratum mix is useful for a pipeline pilot, not a natural-prevalence estimate.
- The exact-shard run [36223699587](https://github.com/laurajoyhutchins/azelficoast/actions/runs/36223699587) failed at `scripts/probe_real_belief_trace.cjs:3199`: `JSON.stringify(..., null, 2)` throws `RangeError: Invalid string length` after constructing the full nested oracle. Other shards were canceled. The Python verifier also reads the entire JSON text before decoding it, so both sides need a bounded-memory design and end-to-end verification.
- Issue [#95](https://github.com/laurajoyhutchins/azelficoast/issues/95) records that no frozen learned evaluator checkpoint is currently published in canonical evidence.

The observed eligibility yield is about 0.0205 selected-eligible states per source battle. If stable, 512 complete-matrix states could require roughly 25,000 source battles. That is a planning projection from an outcome-blind pilot, not a claim that yield is stable or a reason to weaken the eligibility rules.

## Study population and sampling

1. Freeze the source corpus, Showdown revision, and an outcome-blind inclusion/exclusion ledger before policy values or disagreements are computed. The ledger records source decision count, unique battles, each exclusion reason, posterior availability, stratum, inclusion probability, and selected fixture identity.
2. The confirmatory target is **512 states with all required posterior treatments, drawn from at least 256 distinct battles**. Select exactly this cohort from a larger eligible frame using a deterministic seed recorded in the contract. Do not stop early or replace states based on policy values, disagreement, regret, or outcomes.
3. Report the full source denominator, eligibility yield, treatment-availability coverage, selected state count, battle count, and effective design-weighted sample size. If 512 complete-matrix states across 256 battles cannot be obtained under the frozen corpus and compute ceiling, classify the confirmatory run as infeasible; do not relax exactness or support checks.
4. If structural strata are oversampled, record each state's inclusion probability and use design weights for natural-population estimates. Use a battle-clustered, stratum-aware bootstrap that respects the sample design. Unweighted enriched-cohort results may be shown only as explicitly labeled stratum diagnostics.
5. Keep the current 18-state cohort and its artifact digest unchanged as a pipeline/throughput pilot. It does not contribute to confirmatory population estimates.

## Matched factorial treatment

For every selected state, construct a complete matrix across:

- posterior: `oracle`, `generator_faithful`, `practical`;
- search: determinization and information-set;
- depth: one and two public decision horizons (the current registered depth contrast).

The exact oracle posterior must carry reconstruction authority and evidence. If that authority is unavailable, fail closed or label the best available reference honestly; never synthesize oracle mass. Every posterior must exclude the realized hidden state from the solver's information. Within each posterior-depth cell, both search methods receive the same public input, legal actions, posterior identity and mass, mechanics/Showdown semantics, evaluator, opponent model, chance seeds, and authorized compute budget.

Use the existing `transition_evaluations` unit as the primary authorized budget: one charge per verified whole-turn execution class consumed. Extend receipts to account for all expanded transitions at the registered depth. Require the same authorized budget for both search methods in every paired cell, reject overruns and input drift, and record actual charged transitions, simulator work, evaluator calls, and wall-clock time. Results that fail the budget-matching contract are not pooled as matched evidence.

The learned evaluator is frozen before confirmatory execution. Publish the checkpoint or canonical artifact with SHA-256, architecture and inference contract, training data/split manifest, code revision, and training seed/configuration. It may consume public state and posterior information only; it must not receive a realized hidden world. Use the same checkpoint digest in every cell. No synthetic checkpoint may stand in for the learned evaluator.

## Opponent and mechanics identity

For this natural-state study, pin the documented one-sided bounded response model: repeat the opponent's previously observed move when that move is legal; otherwise use a uniform distribution over that hidden world's legal moves. Give this behavior a versioned `opponent_model_id`, canonical semantics record, and digest, and include all three in every packet and result.

Reconcile the probe configuration, search-architecture documentation, contract, and result serializer so the record names the behavior actually executed. Keep any strategy-mixture teacher policy as a different opponent-model identity and do not mix its results into this study. State plainly that the fixed-response experiment does not solve the full two-player imperfect-information game.

Each result also carries the contract digest, source-corpus/cohort digests, Showdown revision, transition-program digest, posterior digest, evaluator digest, opponent-model digest, search method, depth, chance-seed identity, and budget receipt.

## Endpoints and error decomposition

Primary state-level endpoints at each posterior and depth:

1. maximum determinization-minus-information-set value optimism;
2. information-set-evaluated regret of the determinization-chosen root action;
3. common frozen-evaluator decision quality, where its inference contract supports that comparison.

Report design-weighted distributions and battle-clustered intervals. Preserve positive, zero, and negative results. Report depth curves and the preregistered structural predictors from prior frozen evidence; do not add predictors after inspecting this cohort. Policy disagreement remains secondary.

Keep the sources of error separable:

- **Search effect:** paired determinization versus information-set differences within the same posterior, state, depth, evaluator, and budget.
- **Belief effect:** decisions from generator-faithful/practical posteriors evaluated under the frozen oracle reference on shared semantic support, where exact oracle authority exists.
- **Interaction:** the preregistered difference in the search effect between posterior qualities.

Do not report belief effects as oracle-referenced where the reference is unavailable. Report those states and the resulting restricted estimand explicitly.

## Downstream battle outcomes

After the state-level contract, evaluator, and policies are frozen, run a separate held-out, side-balanced battle study. Compare complete policy variants against the same frozen opponent panel using paired battle seeds/lineups where the engine permits, and preserve the battle as the uncertainty cluster. Preregister the opponent panel, policy posterior/search identities, action budget, seed set, primary outcome (win/loss/tie), and sample size before play. Report actual battle outcomes separately from evaluator value and search regret; never use one as a substitute for the other.

The study answers whether decision differences are associated with battle performance under this bounded environment. It does not establish full-game equilibrium quality or generalization outside the frozen formats and opponent panel.

## Implementation slices

1. **Authority and pipeline gate:** recut #118 on current main; make the experiment contract the only source of scientific parameters; move fixed run IDs, digests, treatment settings, cohort checks, and semantic assertions out of workflow YAML. Add a workflow test that rejects duplicated experiment semantics. Repair the producer/verifier serialization path without changing the transition-oracle schema, hidden-world support, probabilities, or validation. Re-run the frozen 18-state pipeline as a throughput check only.
2. **Frozen identities and depth receipts:** reconcile/version opponent semantics; publish and content-address the learned evaluator; extend the matched executor and contracts from depth 1 to both registered horizons. Add tests for matched inputs, complete matrix, same budget, depth accounting, digest drift, oracle-authority failure, and round-trip serialization.
3. **Confirmatory population:** freeze the larger outcome-blind source frame and 512-state/256-battle cohort with inclusion probabilities, execute every factorial cell, and emit the complete weighted aggregate and figure rows.
4. **Outcome study:** preregister and execute the held-out paired battle tournament only after the state-level evidence and policy identities are frozen.

No slice broadens battle mechanics unless measured excluded-state coverage or a specific contract requirement calls for it.

## Alternatives considered

- **Recommended: staged factorial, then outcomes.** It isolates posterior and search effects before expensive full battles, while preserving a clear prerequisite chain.
- **Separate posterior and depth studies.** Simpler to schedule, but loses the posterior-by-search interaction that is the scientific question.
- **Battle tournament first.** More ecological, but conflates posterior quality, search, evaluator calibration, and opponent policy before those identities are controlled.

## Acceptance

The program is ready for confirmatory claims only when:

- the single-source contract and exact-head workflows pass;
- the serializer/verifier round-trip completes the frozen pipeline pilot without losing oracle semantics;
- the frozen evaluator and opponent-model identities are present and identical where required;
- the 512-state/256-battle complete matrix has no missing, duplicate, drifted, or hidden-state-revealing cells;
- each paired search comparison has matching inputs and authorized transition-evaluation budgets;
- the aggregate uses design weights and battle-clustered uncertainty and retains zero/negative effects;
- state-level and held-out battle outcomes are reported separately with complete provenance.

## Review request

Please review the target cohort size, fixed-response opponent semantics, and staged order. Once this design is approved, implementation can proceed slice by slice; any correction should be made in this contract before scientific execution.
