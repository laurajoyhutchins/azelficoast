# Azelficoast

Competitive Pokemon battle AI research with a reproducible Gen 9 Random Battle evaluation harness.

The repository deliberately separates battle orchestration from decision machinery. `AzelficoastPlayer` can now route supported live hidden-Choice information states through the same pinned-Showdown public-belief evaluator used by the natural strategy-fusion experiments. States outside that bounded model, probe failures, and search timeouts fail closed to `poke-env`'s simple-heuristics policy rather than inventing unsupported mechanics.

## Install

Requires Python 3.11+ and `uv`.

```bash
uv sync
```

`poke-env` 0.16.1 is pinned because its player/server APIs are part of the experiment boundary.

## Local smoke battle

`poke-env` expects a local Pokemon Showdown server on port 8000 by default. The upstream recommended development setup is:

```bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

Then, from this repository:

```bash
uv run azelficoast local --battles 10
```

This runs Azelficoast's current baseline against `poke-env`'s `RandomPlayer`.

## Challenge Jaxcalibur or another Showdown user

Keep credentials out of command history:

```bash
export SHOWDOWN_USERNAME='your-bot-account'
export SHOWDOWN_PASSWORD='your-password'
uv run azelficoast challenge Jaxcalibur --battles 1
```

The same harness can wait for incoming challenges or enter the ladder:

```bash
uv run azelficoast accept --opponent Jaxcalibur --battles 1
uv run azelficoast ladder --battles 5
```

Official Showdown play uses one concurrent battle.

### Enable bounded live public-belief search

Live belief search needs a local built checkout of the exact Pokemon Showdown revision used by the mechanics evidence:

```bash
git clone https://github.com/smogon/pokemon-showdown.git /tmp/pokemon-showdown
git -C /tmp/pokemon-showdown checkout a5df8274e85b0889bf2a9b3422a08b39732374fc
cd /tmp/pokemon-showdown
npm ci
npm run build

export AZELFICOAST_SHOWDOWN_ROOT=/tmp/pokemon-showdown
uv run azelficoast ladder --battles 5
```

You can also pass `--showdown-root PATH` before the subcommand. `--belief-timeout SECONDS` bounds one exact probe and defaults to 20 seconds. If the checkout is absent, at the wrong revision, unbuilt, the position is outside the admitted hidden-Choice slice, or the probe cannot finish safely, the live player uses the heuristic fallback.

Decision traces record `decision_metadata.selected_policy` plus the belief attempt, including the reason for fallback or the exact public-belief diagnostics used for a selected move.

## Evidence

Every run produces three complementary evidence surfaces under ignored `artifacts/` paths:

- `results.jsonl` stores one compact outcome record per battle.
- `decisions.jsonl` stores the exact inbound Showdown protocol batches, each decision-time information state, legal actions, the chosen action, and the terminal observable state.
- `replays/` stores `poke-env` replay HTML.

Decision traces use schema `azelficoast.decision-trace` with an integer `schema_version`. Each process invocation receives a unique `run_id`, and `event_index` is monotonic within that run. Unknown opponent information remains unknown in the snapshots; the recorder does not fill hidden fields from later knowledge.

To place traces elsewhere, pass the global option before the command:

```bash
uv run azelficoast --decisions /tmp/jaxcalibur.jsonl challenge Jaxcalibur --battles 1
```

Future policy probabilities, beliefs, values, and search diagnostics can be added to the decision records without changing battle orchestration.

## Replayable decision corpus

Decision traces can be promoted into immutable offline fixtures:

```bash
uv run azelficoast corpus build artifacts/decisions.jsonl \
  --output artifacts/corpus.jsonl
```

Each fixture contains the normalized decision-time state, the decision-semantic Showdown protocol prefix observed before that decision, and the control action chosen in the original battle. Raw traces retain every inbound protocol batch; fixture construction removes parser-ignored transport noise such as server timestamps. The chosen action is evidence, not part of fixture identity. Later observations are never copied backward into earlier fixtures.

The corpus is deterministic and content-addressed. Rebuilding the same evidence produces the same bytes; attempting to overwrite an existing corpus path with different evidence fails closed.

Run a decision algorithm against every frozen fixture:

```bash
uv run azelficoast corpus evaluate artifacts/corpus.jsonl \
  --policy first-legal \
  --output artifacts/first-legal-evaluation.jsonl
```

Two deliberately simple policies currently exercise the common interface: `first-legal` and `recorded-mode`. Their agreement with the recorded heuristic action is an instrumentation sanity metric, not a claim about battle quality. A new algorithm can be loaded without changing the corpus code by passing `--policy package.module:policy_object`; the object must expose a string `name` and `choose(fixture)`. Determinized search and belief-state search should implement that same frozen-fixture interface so they can be compared on identical information histories.


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

## Adaptive simulator dispatch

The dispatcher models the two execution paths according to the work they actually perform rather
than using one population-size threshold:

```text
direct =
    fixed launch cost
  + logical worlds × per-world work
  + optional logical worlds² × bounded saturation term

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
by observed timing noise. Leave-one-out model selection compares four structural shapes: linear
versus bounded-quadratic direct work, each with or without an independent canonical-class term.
Selection first maximizes leave-one-out crossover-choice accuracy. If several models tie on the
decision the dispatcher actually makes, the simplest shape wins; latency MAE only breaks ties
between equally simple shapes. This prevents a saturation term from earning complexity merely by
fitting large, decision-irrelevant latency magnitudes far from the crossover.

An uncertainty guard derived from leave-one-out error is reported with each decision, but it is
diagnostic only. Both execution paths are semantically exact, so uncertainty about performance
does not override the path predicted to be faster.

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
Tinkaton versus Choice Band/Choice Scarf Galarian Zapdos, the conflict was large enough to change
the bounded root decision: world-aware determinization chose Protect while public-belief search
chose Gigaton Hammer.

The causal trace is unusually clean. After Protect, all 18 reconstructed worlds produce one
identical public observation and the immediate transition has no empirically required hidden
read. Every Choice Band world prefers Gigaton Hammer at the next decision; every Choice Scarf
world prefers Protect. Determinization can splice those incompatible continuations together.
Public-belief search must choose one action for the shared information set and chooses Play Rough,
lowering Protect enough for immediate Gigaton Hammer to become the root action.

This positive result remains bounded evidence rather than a competitive-play claim. The opponent
response is fixed to the observed locked move, the continuation horizon is one further decision,
and utility is material-only. The final public root margin is also narrow, so higher-sample,
multi-seed replication is required before treating the root flip as robust.

## Development

```bash
uv run pytest
uv run ruff check .
```

Current source boundary:

```text
Pokemon Showdown
      |
      +----> protocol observations ----+
      |                                |
      v                                v
poke-env harness                decision trace
      |                                ^
      v                                |
AzelficoastPlayer ---- chosen action --+
      |
      v
simple heuristic baseline (replace this)
```

The first research target is not a larger neural policy. It is an evaluation framework capable of comparing determinization against information-set-preserving belief search on the same battle states.
