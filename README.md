# Azelficoast

Competitive Pokemon battle AI research, beginning with a reproducible Gen 9 Random Battle evaluation harness.

The repository deliberately separates the battle orchestration surface from the decision system. `AzelficoastPlayer` is currently a `poke-env` simple-heuristics baseline; future belief tracking, opponent modeling, and search can replace that implementation without changing how experiments are run.

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
