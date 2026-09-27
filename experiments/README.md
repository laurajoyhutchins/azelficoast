# Experiments

This directory is Azelficoast's design record.

The experiments are not a miscellaneous archive of JSON files. Read together, they explain why the bot ended up with its current boundaries: public-information beliefs, information-set-preserving search, pinned Showdown mechanics, verified compilation, matched compute, posterior treatments, and evidence-gated learning.

The short version is:

```text
don't become psychic
        |
        v
reconstruct only what the player can know
        |
        v
prove the mechanics used by search
        |
        v
show where hidden-world search becomes optimistic
        |
        v
measure that effect on natural populations
        |
        v
compress and accelerate without changing semantics
        |
        v
learn only behind the same evidence boundary
```

For the detailed historical notebook, see [`docs/research-notebook.md`](../docs/research-notebook.md). This page is the curated reading order: the experiments as a causal story of the design.

## Directory contract

The root of this directory is intentionally small.

| Path | Role |
| --- | --- |
| [`contracts/`](contracts/) | Executable orchestration authority: what candidate and hosted research must run and settle |
| [`data/`](data/) | Small frozen inputs, selections, treatments, plans, contracts, expected results, and negative controls |
| [`evidence/`](evidence/) | Canonical evidence bundle and digest manifest for larger frozen evidence |
| [`src/azelficoast/research/experiments/`](../src/azelficoast/research/experiments/) | Experiment implementations used by repository-owned contracts |

The distinction matters. Data records scientific state. Contracts decide how repository research is executed. Evidence preserves larger immutable inputs. GitHub Actions transports those authorities but does not redefine them.

## Chapter 1: future-you is not allowed to become psychic

The first question was architectural rather than Pokémon-specific:

> Can a search procedure accidentally use hidden information at future decisions even when the root decision looks uncertainty-aware?

The synthetic imperfect-information experiment and the Pokémon-shaped counterexample established the strategy-fusion failure mode. The follow-up work moved from curated examples to replay-derived beliefs and exact public histories.

Relevant frozen records include:

- [`real-belief-source-fixture.json`](data/real-belief-source-fixture.json) and the named real-belief source fixtures;
- [`real-belief-negative-corpus.json`](data/real-belief-negative-corpus.json), which keeps counterexamples and non-findings visible;
- [`live-belief-coverage-results.json`](data/live-belief-coverage-results.json), which records how much of the live belief surface is actually supported.

**Design consequence:** public state and hidden-world support became separate authorities. Search may condition on a posterior, but it may not smuggle the realized hidden world into future decisions. Unsupported states must fail closed.

## Chapter 2: believable search still needs real mechanics

Once hidden information was handled honestly, the next failure mode was simpler: a correct information-set algorithm is still wrong if its battle transitions are wrong.

The mechanics experiments probe small, falsifiable pieces against pinned Pokémon Showdown behavior:

- [`protect-action-survival.json`](data/protect-action-survival.json);
- [`protect-speed-forks.json`](data/protect-speed-forks.json);
- the natural status-move treatment and result records;
- candidate experiments for damage, ordered attacks, two-action turns, switching, hazards, Intimidate, and stateful Protect.

**Design consequence:** Pokémon Showdown remains the mechanics authority. Native or compiled execution is admitted only on a bounded support that has been checked against that authority. Generated output is not automatically verified output.

## Chapter 3: does strategy fusion matter in real positions?

A synthetic counterexample proves existence. It does not establish practical relevance.

The next experiments therefore mine or select naturally occurring public states and ask whether determinization and public-belief search disagree, whether the disagreement survives exact mechanics, and whether apparently interesting cases collapse under stronger controls.

Read these together:

- [`natural-public-belief-exact-selection.json`](data/natural-public-belief-exact-selection.json) and [`natural-public-belief-exact-results.json`](data/natural-public-belief-exact-results.json);
- [`status-move-prior-selection.json`](data/status-move-prior-selection.json) and [`status-move-prior-results.json`](data/status-move-prior-results.json);
- [`natural-status-move-treatment.json`](data/natural-status-move-treatment.json) and [`natural-status-move-results.json`](data/natural-status-move-results.json);
- [`opponent-policy-semantics.json`](data/opponent-policy-semantics.json), which freezes the opponent-policy semantics used by this line of research.

**Design consequence:** Azelficoast measures value bias, regret, continuation conflicts, and policy disagreement separately. A different root move is interesting evidence, not the only definition of a search failure.

## Chapter 4: stop trusting one posterior or one depth

Once natural examples existed, two confounders became impossible to ignore:

1. a search algorithm can look bad because its posterior is bad;
2. a shallow or tiny sample can exaggerate or hide an effect.

That led to population, depth, and posterior-robustness work:

- [`natural-depth-regret-plan.json`](data/natural-depth-regret-plan.json);
- [`natural-population-strategy-fusion-plan.json`](data/natural-population-strategy-fusion-plan.json);
- [`natural-public-belief-robustness-plan.json`](data/natural-public-belief-robustness-plan.json) and its frozen [result](data/natural-public-belief-robustness-results.json);
- [`posterior-stratified-population-contract.json`](data/posterior-stratified-population-contract.json).

The files named `plan` or `contract` are deliberately not presented as completed findings. They specify the comparisons that must be run under matched populations, depths, compute, and posterior treatments.

**Design consequence:** belief error and search error are separate experimental axes. Matched experiments bind source state, public history, legal actions, mechanics identity, evaluator identity, chance treatment, search depth, and compute accounting before methods are compared.

## Chapter 5: make it fast without changing what it means

Correct search was too expensive to scale by simply repeating full simulator work for every hidden world.

The candidate experiment suite therefore asks a different class of question:

> Which representations and execution strategies can be substituted without changing admitted semantics?

The candidate contract covers progressively more aggressive execution machinery: class-native beliefs, native damage, ordered and multi-action transitions, compiled search, JAX execution, and related equivalence checks.

See [`contracts/candidate.json`](contracts/candidate.json) for the executable set.

**Design consequence:** optimization became an evidence problem. Active-support reduction, execution classes, memoization, compiled data planes, JAX, and SQL-shaped planning are acceptable only when the faster representation preserves the verified decision semantics on the supplied support.

## Chapter 6: learning is downstream of the evidence boundary

A learned evaluator is useful only if it does not become a shortcut around the rules established above.

The policy-boundary experiments and the training/promotion machinery therefore treat learned decisions as candidates, not authority:

- [`policy-boundary-refinement-results.json`](data/policy-boundary-refinement-results.json) records one frozen boundary study;
- candidate and hosted contracts run the research that informs routing and evaluation;
- training and promotion remain separately gated because they can change the live evaluator.

**Design consequence:** search can teach learning, but training cannot grade its own homework. Promotion requires independent evidence, and unsupported states retain a deterministic fallback path.

## How to add the next experiment

A new experiment should extend this story rather than merely add another runnable program.

It should make five things obvious:

1. **Question:** what unsupported claim are we testing?
2. **Authority:** which repository contract owns execution and settlement?
3. **Frozen inputs:** what public state, mechanics revision, treatment, or corpus is held fixed?
4. **Falsifier:** what result would make the claim fail?
5. **Design consequence:** if the evidence survives, what implementation choice becomes justified, constrained, or removable?

Keep negative results. Keep plans distinguishable from results. Keep scientific assertions out of workflow YAML. When a new experiment displaces an old authority, delete the duplicate path instead of preserving two competing truths.
