# Research notebook

This document preserves the experiment-by-experiment narrative that originally accumulated in Azelficoast's README.

It is a **historical notebook**, captured from repository state `f9dc0b28da32d2dfd2242e3bfcdf40b8aaeca352`. Statements such as “next rung,” current limitations, and performance measurements describe the project at the point each note was written; they are not the current roadmap or capability contract.

The value of these notes is the progression: each rung states a narrow claim, records the evidence used to test it, names important negative controls, and leaves the next unsupported claim visible. Current public-facing status belongs in the root README, the current search/mechanics boundary is documented in [`search-architecture.md`](search-architecture.md), and frozen machine-readable evidence lives under [`experiments/`](../experiments/).

## Research progression

The notebook follows three intertwined threads:

- **Information-set correctness:** from a synthetic strategy-fusion example to Pokémon-shaped and naturally mined public-belief states.
- **Mechanics fidelity:** from a tiny dependency IR to pinned-Showdown damage, ordering, two-action turns, and additional composed mechanics.
- **Execution compression:** from repeated hidden-world execution to dependency classes, class-native beliefs, adaptive dispatch, and generated native kernels.

---
## Imperfect-information reference experiment

The first search experiment deliberately uses a tiny two-stage hidden-world game rather than full Pokémon mechanics. It exists to test the architectural question cleanly.

```text
hidden world: red or blue

guess
  -> world remains hidden
  -> one future action must serve both worlds

scout
  -> world is revealed
  -> future action may depend on the observation
```

The determinization baseline solves each sampled world as if its hidden state will remain available at future decisions. The public-belief solver instead groups worlds by the public observation available at that future information set and requires one continuation per indistinguishable group.

Run the reference experiment directly:

```bash
uv run python -m azelficoast.imperfect_information
```

The preregistered treatment requires determinization to overvalue the unrevealed `guess` branch while public-belief search chooses the costly but informative `scout` branch. A negative control then reveals the world for free; both solvers must agree there.

Both policies also implement the corpus policy interface:

```bash
uv run azelficoast corpus evaluate artifacts/corpus.jsonl \
  --policy azelficoast.imperfect_information:determinization

uv run azelficoast corpus evaluate artifacts/corpus.jsonl \
  --policy azelficoast.imperfect_information:public_belief
```

This is intentionally not yet a Pokémon-strength search engine. It establishes the strategy-fusion failure and the information-set-preserving correction on the exact interface that later Pokémon world models will use.


## Pokémon-shaped hidden-item counterexample

The abstract red/blue experiment now has a Gen 9 Random Battle analogue grounded in current Pokémon Showdown set-generation data.

The position uses:

- level-80 Jirachi with `Protect` and `Iron Head`;
- level-83 Gardevoir on an all-special Fast Attacker set whose item path can produce either Choice Scarf or Choice Specs;
- level-94 Furret with `Frisk`;
- a revealed `Moonblast` from Gardevoir.

Under Showdown's randbats stat defaults, unboosted Gardevoir is 180 Speed, Jirachi is 206, and Choice Scarf Gardevoir is 270. The experiment pins Gardevoir to 83 HP and Jirachi to 29 HP, so Jirachi's Iron Head removes Gardevoir if Jirachi acts first, while even the minimum Moonblast roll removes Jirachi if Scarf Gardevoir acts first.

`Protect` blocks Moonblast but does not reveal whether the item is Scarf or Specs. Determinization therefore overvalues the protected position by selecting different next moves in the two hidden worlds. The public-belief solver cannot do that. Its information-gathering alternative is the real switch to Frisk Furret.

Furret's bounded two-turn line is evaluated by enumerating the actual 16 damage rolls for Moonblast and Knock Off. No information bonus is inserted by hand.

Run it directly:

```bash
uv run python -m azelficoast.pokemon_counterexample
```

The treatment passes only if determinization chooses `Protect`, public-belief search chooses `Furret`, all mechanical speed/damage falsifiers hold, and both solvers agree again in the known-item controls.

This remains a bounded tactical model, not a claim that the current search machinery is a competitive Pokémon engine. The next rung is to construct these hidden worlds automatically from a real decision trace and the Showdown random-set generator rather than embedding one curated position.


## Replay-derived hidden-world reconstruction

Azelficoast can now reconstruct a hidden-item belief from a real public Gen 9 Random Battle replay without hard-coding the revealed answer.

The hosted experiment uses public replay `gen9randombattle-2405042449` and binds its generator model to Pokémon Showdown commit `6397bfddb3db4e916dd792e03c43355f7366e8ab`, the latest relevant randbats repository revision from July 1, 2025 before that battle.

The pipeline is:

```text
public replay history
        |
        +--> observed Infernape: Close Combat
        |
        v
historical Showdown generator
65,536 deterministic seed samples
        |
        v
empirical generator prior
  Life Orb     63.33%
  Choice Band  24.55%
  Choice Scarf 12.12%
        |
        +--> turn 18:
             damaging Close Combat
             complete public turn
             no Life Orb recoil
        |
        v
public-history prior
  Choice Band  66.94%
  Choice Scarf 33.06%
        |
        +--> turn 19:
             Close Combat
             into Tera-Flying Kingambit
             239/270 -> 185/270
             observed damage = 54
        |
        v
exact damage likelihood
        |
        v
posterior: Choice Scarf = 1.0
```

The seed sweep is a reproducible empirical prior over the chosen deterministic seed family and empty generator-team context, not an exact analytic distribution over every possible Showdown PRNG or team-generation history. Public evidence is applied separately. The absence of mandatory Life Orb recoil after turn 18 removes Life Orb from the sampled support. The observed 54 damage on turn 19 is then compatible with the Choice Scarf damage range (51–61) and incompatible with Choice Band (77–91); the raw Life Orb range is 66–79. The damage calculation mirrors the historical Showdown ordering and fixed-point modifier rounding.

The resulting `azelficoast.replay-belief` record includes the replay ID, historical Showdown revision, revision-bound generator sample and context, raw generator prior, authoritative public-history updates, pre-damage prior, public damage observation, compatible worlds, posterior, and a deterministic SHA-256 belief identity. Unknown sampled item semantics fail closed instead of being treated as neutral damage.

The networked/historical experiment lives in `replay-world-experiment.yml`; ordinary unit CI remains independent of external replay and npm availability.

This validates replay-to-hidden-world reconstruction. It does not yet demonstrate a determinization/public-belief action disagreement from a naturally occurring full Azelficoast decision state, because a public opponent replay does not expose our own hidden bench and complete legal-action state. Azelficoast's instrumented live/local traces do, which is the next search surface.

## Dependency-aware simulator experiment

The first simulator rung is intentionally a tiny reference IR rather than a fast backend. Each
effect declares the state fields it reads, the fields it may write, and its random inputs.
Hidden worlds can then be grouped by the fields that are actually relevant to a transition.

Run the finite treatment:

```bash
uv run python -m azelficoast.simulator_experiment
```

The preregistered slice creates 4,096 hidden worlds. A Protect-like transition must collapse
them to one unique computation because the hidden Choice item and bench signature cannot affect
the blocked result. A Choice-item-sensitive special-damage transition must retain exactly two
classes, one for Scarf and one for Specs, while still discarding the irrelevant bench dimension.
Expanded collapsed execution must exactly equal direct per-world execution. A negative control
deliberately omits the item dependency and must be rejected.

This is a dependency/collapse proof over a synthetic microkernel, not yet a claim of complete
Pokémon Showdown mechanics or accelerator performance. The next rung is to bind the same IR to
real Showdown-derived transition fixtures before introducing JAX/Pallas kernels.

## Showdown-backed simulator dependency experiment

The dependency IR is also checked against transitions produced by a pinned Pokémon Showdown
engine rather than only against the synthetic reference microkernel. Hosted CI creates fixed-seed
Gen 9 battles that vary a hidden Choice Scarf versus Choice Specs world and a genuinely different
benched item.

Two treatments use the same state-dependency declarations:

- a Protect turn must collapse both Choice-item worlds and all bench variants within each fixed
  RNG seed because neither difference can affect the blocked transition;
- an unprotected Aura Sphere turn must preserve the Choice-item distinction while collapsing the
  benched-item dimension.

The fixture matrix is produced directly by Pokémon Showdown at commit
`a5df8274e85b0889bf2a9b3422a08b39732374fc`. The checker compares the resulting HP and
transition protocol slice within each dependency class. A negative control removes the opponent
item from the damage read set and must be rejected.

This validates a narrow state-dependency partition against real Showdown execution. It still does
not claim complete mechanics coverage or accelerator speed. RNG is held by exact Showdown seed
here; the synthetic experiment separately checks explicit RNG dependency declarations.

## JAX simulator lowering

The next simulator rung lowers the validated reference microkernel into JAX without changing its
semantics. JAX is an opt-in dependency so the ordinary battle/evidence harness stays lightweight:

```bash
uv sync --extra simulator
uv run python -m azelficoast.jax_simulator_experiment
```

The hosted experiment checks the JAX batch kernels against the scalar reference, then measures
CPU execution at 2,048, 32,768, and 524,288 logical worlds. It compares direct per-world execution
with already-partitioned dependency classes: one class for the Protect slice and two classes for
Choice-item-sensitive damage. The reduced path carries class weights and verifies the same
aggregate result without expanding every world.

The timing deliberately excludes dependency-partition construction. Reported reduced throughput
therefore means logical worlds represented per second by the class computation, not physical
transition evaluations per second. Hosted GitHub evidence is CPU evidence only. The JAX kernels
still implement the small reference microkernel, not complete Showdown damage or battle mechanics.

## Exact Gen 9 damage kernel

The simulator now has a first real Pokémon mechanic rather than only a synthetic damage
microkernel. A narrow Gen 9 damage implementation reproduces Pokémon Showdown's integer operation
order for ordinary single-target attacks: stat construction, Choice Band/Specs attack
modification, base-damage truncation, the 16 random damage rolls, ordinary/Tera STAB, type
effectiveness, burn, and Life Orb's final modifier.

Hosted evidence generates eleven controlled scenarios directly through
`BattleActions.getDamage` at pinned Showdown commit
`a5df8274e85b0889bf2a9b3422a08b39732374fc`, covering all 16 damage rolls per scenario.
The corpus includes non-STAB, ordinary STAB, new-type Tera STAB, same-base-type Tera STAB,
Choice Specs, Choice Band, Life Orb, super-effective, neutral, heavily resisted, physical, and
burned physical damage. Deliberately removing type, item, Tera, or burn semantics must create a
mismatch.

The same numeric context lowers to JAX and is checked against the scalar implementation. Its
benchmark reports both an already-partitioned reduced path and an end-to-end baseline that
rebuilds dependency classes with `numpy.unique` on every call. The latter intentionally includes
partition construction so an impressive kernel-only speedup cannot hide an expensive grouping
step.

This remains a bounded mechanics slice. Critical hits, weather, spread damage, exceptional
abilities, variable base power, Stellar Tera, multihit sequencing, and other effects remain
outside the claim until separate Showdown-backed evidence covers them.

## Class-native belief state

The simulator no longer needs to treat a belief as a flat particle bag merely to exploit
dependency classes. The class-native experiment represents a finite belief as integer
multiplicities over a compact canonical semantic support. Effect-specific projection maps then
select only the distinctions needed by the current mechanic.

For the Showdown-backed damage corpus, the hosted treatment adds eight nuisance bench variants
and all sixteen damage rolls. That creates 1,536 canonical support classes while representing up
to 524,288 logical worlds. The same belief can be viewed as:

- 1 execution class for Protect;
- 192 classes for the damage kernel;
- 8 classes for a bench-only dependency;
- all 1,536 classes when every modeled distinction matters;
- then 1 class again for Protect.

The belief weights never expand into 524,288 mutable particles during the class-native path.
Exact-damage observations filter canonical weights directly. A raw expansion is retained only as
an oracle and benchmark baseline.

The performance treatment includes the projection cost on every class-native call. It compares
that path both against already-materialized, device-resident direct JAX execution and against
materializing every logical world and rediscovering dependency classes with generic
`numpy.unique`. The direct baseline is intentionally favorable to raw particles: its timed
region excludes materialization and host-to-device transfer, while the class-native timed region
pays projection and transfer each time. A negative control removes the Choice Band/Specs-derived
attack modifier from the
damage signature and must merge mechanically distinct worlds, producing the wrong weighted
damage.

This proves only the bounded finite-class representation used here. It does not establish that
all Pokémon hidden variables factor into a small canonical support, nor that every future
mechanic admits a cheap projection. Those remain empirical questions as mechanics coverage grows.

## Ordered attack transition

The compiled simulator now extends the bounded whole-attack transition through one layer of
turn ordering and one observable secondary effect. The ordered transition resolves:

- move priority before Speed;
- Speed before a final speed-tie outcome;
- the previously validated accuracy/damage/HP/PP/Life Orb attack path;
- a Shadow Ball-shaped 20% Special Defense drop, bounded at -6;
- one packed post-state containing HP, PP, SpD stage, and whether the attacker acted first.

The opponent action in this rung is deliberately a no-op. This isolates ordering semantics from
the much larger problem of executing two mutually interacting attacks.

One dependency signature covers the whole ordered transition. It cryptographically binds the
underlying whole-attack signature and conditionally includes the order-tie outcome and secondary
RNG only where they can change the result. The hosted experiment carries two independent hostile
controls: omitting the tie-order dependency and omitting the secondary dependency must each merge
states that produce different transition results.

Pinned Showdown fixtures cover priority overriding Speed, faster and slower attackers, both final
speed-tie outcomes under Gen 9's dynamic queue re-sorting, and the secondary-effect boundary.
Python, the generated C kernel, and the JAX lowering must all match the full packed Showdown
post-state exactly.

The adaptive dispatcher is calibrated on this entire ordered transition rather than on damage
alone. Its confirmation grid spans the direct/class-native crossover and requires exact aggregate
semantics while measuring both execution paths.

This remains a bounded rung. The opponent move does not yet alter state, and protection,
immunities, multihit sequencing, contact hooks, status secondaries, and mutually interacting
attacks remain outside the claim.

## Two-attack turn

The compiled simulator now executes a bounded singles turn containing two real damaging actions
rather than an attack against a no-op opponent. Priority, Speed, and the final speed-tie outcome
select the first actor. Each queued move consumes PP only if it actually executes.

The treatment uses Moonblast versus Aura Sphere to make causal ordering observable. If Moonblast
acts first and its 30% SpA drop lands, Aura Sphere's later damage is calculated from the reduced
SpA stage. If Aura Sphere has already resolved, the later drop changes only the resulting stage and
cannot retroactively change that damage. If either first action faints the queued second actor, the
second action is cancelled and its PP remains untouched.

One dependency signature covers the entire turn. It binds the previously validated attack and
ordered-attack contracts, then adds the dynamic dependency on the SpA stage at p2 execution. The
hosted treatment carries independent hostile projections that omit the final tie outcome or the
secondary/stage dependency; each must merge states that produce a different aggregate result.

Pinned Showdown evidence covers 1,248 controlled full turns spanning faster and slower actors,
priority, both final tie outcomes, both KO-cancellation directions, Choice Specs versus no item,
damage-roll extremes, and the Moonblast secondary boundary. Python, generated C, and JAX must
match the exact packed post-state, including both HP values, both PP values, p2's SpA stage, and
which queued actions actually executed.

For the class-native treatment, 3,968 canonical states collapse to 128 execution classes. A
separate disjoint confirmation grid calibrates the adaptive direct-versus-class-native dispatcher
on the entire two-action transition, not on an isolated damage primitive.

This remains intentionally bounded. Switching, protection, immunities, redirection, multihit
sequencing, contact hooks, status effects, and residual/end-turn processing remain outside the
claim.

## Adaptive simulator dispatch

The dispatcher models the two execution paths according to the work they actually perform rather
than using one population-size threshold:

```text
direct =
    fixed launch cost
  + logical worlds × per-world work
  + optional logical worlds² × bounded saturation term
  + or optional max(0, worlds - knee) × post-knee saturation work

projected =
    fixed projection/transfer cost
  + optional active canonical classes × projection work
  + active execution classes × compact execution work
```

All coefficients are non-negative, so the model is monotone in every work dimension. The
quadratic direct term is not an algorithmic-complexity claim; it is an optional local
approximation for backend saturation/cache behavior. Profiles refuse to extrapolate beyond the
largest logical population or class counts used during calibration.

Calibration alternates direct and projected measurements to reduce runner drift and weights fits
by observed timing noise. Leave-one-out model selection compares linear, bounded-quadratic, and
hinge-saturation direct work, each with or without an independent canonical-class term.
Selection first maximizes leave-one-out crossover-choice accuracy. If several models tie on the
decision the dispatcher actually makes, the simplest shape wins; latency MAE only breaks ties
between equally simple shapes. This prevents a saturation term from earning complexity merely by
fitting large, decision-irrelevant latency magnitudes far from the crossover.

When calibration contains repeated measurements with the exact same canonical-class and
execution-class counts, and those measurements form one monotone direct-to-projected crossover,
the profile also records the observed crossover bracket. Dispatch uses the bracket midpoint only
for that exact class shape. This prevents far-above-crossover cache saturation from dragging a
local decision boundary away from the directly observed bracket, while multi-dimensional
workloads continue to use the structural cost surface.

Low-world whole-attack calibration may contain fewer logical worlds than the full canonical
support. Those treatments activate a deterministic evenly spaced subset of support classes rather
than truncating a prefix, so the small-population evidence still spans the support geometry.

An uncertainty guard derived from leave-one-out error is reported with each decision, but it is
diagnostic only. Both execution paths are semantically exact, so uncertainty about performance
does not override the selected structural or crossover-fenced path.

The hosted experiment retains the original linear absolute-curve model as a control and evaluates
both on a denser, disjoint crossover-heavy grid. Promotion requires at least 5% lower held-out
projected-minus-direct prediction error, no material increase in dispatch regret, both paths to
remain useful, and exact result agreement throughout.

Calibration is execution-target-specific, not merely backend-specific. Profiles are fenced to a
fingerprint covering the JAX backend, device kind/count, machine architecture, CPU count, and CPU
model where available. JAX CPU, the owned native kernel, different CPU hosts, and future GPU
kernels therefore receive separate profiles rather than sharing coefficients.

## Native damage compiler

The exact numeric damage formula now has one executable definition in `gen9_damage.py`. Ordinary
CPython executes that function directly. Azelficoast's deliberately tiny native compiler extracts
the same function and its three helpers from the Python AST, rejects syntax outside its supported
subset, and emits standalone C99 with 64-bit intermediates and Python-compatible floor division.

```bash
uv sync --extra simulator
uv run python -m azelficoast.native_damage_experiment \
  /tmp/showdown-gen9-damage-fixtures.json
```

Hosted correctness requires the interpreted numeric function, generated native code, JAX lowering,
and pinned Pokémon Showdown corpus to agree exactly. A transitional comparison against POST Python
0.3.0 established that the external compiler was not necessary for this workload: with 524,288
logical worlds represented by 192 execution classes, the owned weighted kernel measured 0.0168 ms
versus 0.0157 ms for POST and 0.136 ms for JAX on the comparison runner. At 524,288 direct
transitions the owned batch path was faster than the POST control in that same treatment.

POST is therefore no longer a project dependency. The compiler remains intentionally narrow rather
than evolving into a general Python implementation. A mechanic that needs new syntax must extend
the supported language explicitly, with rejection tests and Showdown-backed semantic evidence.

## Real public-belief trace mining

The search-level experiment now treats naturally occurring policy agreement as evidence rather
than as a failed test. A pinned real Gliscor versus Urshifu decision reconstructs 33 plausible
hidden worlds, executes all 12 legal root actions in Pokémon Showdown, derives dependency
signatures from the observed transitions, and regroups successor states by the exact public
observation available to the player.

That trace is a durable negative corpus case: both world-aware determinization and public-belief
search choose Earthquake under the bounded continuation model. The result is retained with the
hosted artifact identity instead of changing the utility model to manufacture a disagreement.

`real_belief_miner.py` accepts one or more such mechanics oracles and ranks candidates only by
structural information available before inspecting the policy result:

- how many actions are sensitive to hidden state;
- how many actions collapse the hidden-world support;
- how many actions branch into multiple public observations;
- the size of the resulting observation partition;
- world and legal-action counts.

The rank does not use either policy's value or whether the policies disagree. After structural
ordering, the miner first reports whether any public observation class contains multiple
world-aware best continuations, the direct necessary signal for strategy fusion in the reference
experiment. It then separately reports the first root-level determinization/public-belief
disagreement, if one exists. Neither post-ranking signal can change candidate order. This makes
additional instrumented battle traces a corpus search problem rather than a sequence of
hand-curated tactical examples.

A preregistered three-case natural exact treatment has now found that failure mode in pinned
Showdown mechanics. All three structurally selected hidden-Choice speed-fork cases contained
public successor information sets with competing world-aware continuations. In the third case,
Tinkaton versus Choice Band/Choice Scarf Galarian Zapdos, the original 8×8 chance-sampling
treatment changed the bounded root decision: world-aware determinization chose Protect while
public-belief search chose Gigaton Hammer.

The causal trace is unusually clean. After Protect, all 18 reconstructed worlds produce one
identical public observation and the immediate transition has no empirically required hidden
read. Every Choice Band world prefers Gigaton Hammer at the next decision; every Choice Scarf
world prefers Protect. Determinization can splice those incompatible continuations together,
while public-belief search must choose one continuation for the shared information set.

A preregistered robustness treatment then reran the same frozen decision at 32 root samples × 32
continuation samples under four deterministic seed families. The causal strategy-fusion pattern
replicated in all four families, and determinization overvalued Protect relative to public-belief
search in all four. The specific root-policy flip did not: families 0 and 1 chose Gigaton Hammer
under public-belief search, while families 2 and 3 chose Protect. The durable result is therefore
the information-set continuation conflict and its positive strategy-fusion value bias, not a
stable claim that the corrected root action must be Gigaton Hammer.

This remains bounded evidence rather than a competitive-play claim. The opponent response is
fixed to the observed locked move, the continuation horizon is one further decision, utility is
material-only, and hidden support is an empirical Showdown-generator support rather than a full
analytical posterior.
