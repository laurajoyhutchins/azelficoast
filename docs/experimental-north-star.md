# Experimental north star

Azelficoast's main scientific claim should be tested through one progressively stronger comparison:

~~~text
same natural Pokémon decision states
        + same posterior treatment
        + same computational budget
                    |
          +---------+---------+
          |                   |
   determinization    information-set search
          |                   |
          +---------+---------+
                    |
         bias / regret / action quality
                    |
             battle outcome
~~~

The simulator, hidden-world reconstruction, compressed execution, and mechanics work are supporting apparatus. They are not the experimental endpoint.

## Questions

The program separates three questions that otherwise confound one another.

1. **Search:** given the same faithful belief, does determinization systematically overvalue plans by fusing future policies that could not coexist under one information history?
2. **Inference:** how much of any search advantage survives when the hidden-state posterior is approximated?
3. **Game model:** how much changes when the opponent is strategic and also acts from private information?

A positive result on the first question does not imply that the second or third has been solved.

## Posterior treatments

Every comparison should identify which belief treatment generated the hidden-world distribution.

### Oracle posterior

Use the exact or best-available conditional distribution induced by the Random Battle generative process and the complete public history available at the decision point.

"Oracle" does not mean revealing the realized hidden state. A point mass on the actual hidden world would remove the uncertainty under study.

This treatment asks whether information-set-respecting search is valuable when belief error is minimized.

### Generator-faithful posterior

Use finite samples from the pinned generator, conditioned on all available public evidence while preserving dependencies among species, moves, abilities, items, Tera types, teammates, and observed actions.

This treatment asks whether the theoretical correction survives a tractable high-fidelity approximation.

### Approximate practical posterior

Use a bounded inference procedure suitable for live play.

This treatment asks whether posterior error dominates search error in practice.

The main analysis should report both search method and posterior treatment rather than collapsing them into a single "public-belief" number.

## Population experiment

### Source population

Mine naturally occurring Gen 9 Random Battle decision states from instrumented battles. Preserve the immutable source corpus and account for every source state through either inclusion or an explicit exclusion reason.

Freeze admissibility before reading treatment outcomes.

Do not select states because they exhibit policy disagreement, large value bias, or large regret.

### Matched treatments

For every admitted state:

- reconstruct one frozen posterior treatment;
- expose identical legal root actions to both search methods;
- use identical mechanics semantics and Showdown revision;
- match computational budget by an explicit resource contract;
- derive determinization and information-set-search results from the same frozen evidence;
- preserve seeds and all stochastic sampling metadata.

The compute contract should be stated in units that cannot hide asymmetric work. Record at least simulator transitions and wall-clock time; choose one as the preregistered matching authority and report the other diagnostically.

### Primary endpoints

Use continuous endpoints as the primary scientific measurements:

1. maximum determinization-minus-information-set value optimism across root actions;
2. information-set-evaluated regret of the determinization-chosen root action;
3. decision quality under a common information-respecting evaluator when such an evaluator is available.

Root-action disagreement is secondary. It is useful descriptively but should not gate whether a state contains measurable strategy fusion.

### Inference

Cluster resampling by battle rather than treating decisions from the same battle as independent.

Report bootstrap confidence intervals for the population means and rates. Preserve zero-effect and negative results.

Depth should be a controlled treatment, producing depth curves rather than a single hand-selected horizon.

### Structural predictors

Reuse the predictor vocabulary already frozen by the first natural population study. For the next confirmatory population, select only two or three predictors before reading outcomes.

Strong candidates from the existing measurement surface include:

- hidden-item entropy;
- incoming KO-roll probability gap;
- persistent branch count;
- relative move-order changes;
- incoming damage-fraction gap;
- persistent switch or Protect branches.

Selection of the confirmatory subset should be justified from prior frozen evidence, not from the new treatment outcomes.

## Two-player boundary

The current natural-state treatment is deliberately narrower than a full two-player imperfect-information solution.

It enforces that the evaluated player cannot choose future actions using hidden-world identity that is not public at that information set. The current bounded continuation model fixes the observed opponent response. Therefore it does not yet optimize an opponent strategy conditioned on the opponent's private information history.

A later two-sided treatment should make explicit:

- the player's information history;
- the opponent's information history;
- which public observations update both beliefs;
- which private observations are available to only one policy;
- how each policy is constrained to be identical across histories it cannot distinguish.

Only that stronger treatment should support claims about two-player information-set-respecting strategic search.

## Battle outcome

Battle win rate is the final downstream endpoint, not the first one.

First establish whether the matched decision-state comparison produces systematic differences in value bias, regret, or common-evaluator decision quality. Then run paired policy battles under matched resources to determine whether those local corrections accumulate into measurable battle outcomes.

This ordering keeps high-variance battle feedback from obscuring the mechanism being tested.

## Research prioritization rule

A proposed expansion in mechanics, simulator performance, posterior inference, or search depth should answer at least one of these questions:

- Does it admit materially more natural decision states?
- Does it make the posterior treatment more faithful?
- Does it make the compute match fairer or more measurable?
- Does it permit a deeper controlled treatment?
- Does it move the evidence chain from decision quality toward battle outcome?
- Does it close the two-player information-set boundary?

If not, it is probably not on the critical path to the main claim.
