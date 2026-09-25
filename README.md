# Azelficoast

Azelficoast is an evidence-first research system for competitive Pokémon battle AI under partial observability.

The project is investigating a specific failure mode of conventional search: **determinization can plan future actions using hidden information the player will not actually have**. Azelficoast compares that approach with public-belief search, where future choices must respect the information set visible at that point in the battle.

It is also building the machinery needed to make that comparison on real Pokémon states: instrumented battle traces, hidden-world reconstruction, a Showdown-backed mechanics oracle, exact bounded simulation, and compressed belief-state execution.

> **Status:** active research code, not a finished competitive bot. The live player uses bounded public-belief search only on admitted states and falls back to poke-env simple heuristics elsewhere.

## Matched experiment contract

The central experiment is executable, not a naming convention. `matched_comparison.py`
freezes one public decision state, one posterior, one Showdown revision, one depth,
and one compute ceiling into paired determinization and information-set work packets.
Settlement fails closed if either method changes the input, legal actions, budget, or
chosen-action semantics.

`matched_population.py` then requires the complete natural-state × posterior × depth
matrix and emits battle-clustered bootstrap rows for value optimism, regret, and policy
disagreement. The posterior treatment and opponent model remain explicit in every
packet and result.

The current natural-state oracle work still has a narrower boundary: it fixes the
observed opponent response. A result is marked as preserving two-player information
sets only when the executable opponent-model contract says so.

~~~bash
python -m azelficoast.matched_comparison freeze PLAN STATE POSTERIOR \
  --posterior-treatment generator_faithful --depth 2 --output packet.json
python -m azelficoast.matched_comparison settle packet.json DET.json INFO.json \
  --output result.json
python -m azelficoast.matched_population PLAN COHORT RESULTS \
  --output aggregate.json
~~~

## What exists today

~~~text
Pokémon Showdown
      |
      v
public observations + legal actions
      |
      +--> immutable decision traces
      |        |
      |        v
      |    hidden-world reconstruction
      |        |
      |        v
      |    public-belief search
      |
      +--> pinned Showdown mechanics
               |
               v
        exact bounded transitions
               |
               v
      dependency-aware execution
~~~

The repository currently contains:

- a Gen 9 Random Battle harness for local, challenge, accept, and ladder play;
- replayable, content-addressed decision corpora;
- live decision tracing that preserves what was actually public at decision time;
- hidden-world reconstruction from Showdown generator support and public evidence;
- determinization and information-set-preserving public-belief search on the same frozen states;
- bounded Gen 9 mechanics checked against pinned Pokémon Showdown execution;
- dependency signatures that collapse hidden worlds when a mechanic cannot distinguish them;
- class-native belief execution that retains weighted semantic support instead of expanding every logical world;
- interpreted, generated-native, and JAX execution paths with exact-result cross-checks;
- exact-head CI experiments that separate ordinary merge checks from expensive research evidence.

The bounded mechanics surface now includes ordinary attacks, ordering, two damaging actions, repeated Protect, voluntary switching, entry hazards, and Intimidate switch-in ordering. Each expansion is admitted only after comparison with pinned Showdown behavior.

## Research result so far

Azelficoast has reproduced the classic strategy-fusion problem in Pokémon-shaped models and in naturally mined Gen 9 Random Battle states.

The durable result is **not** that public-belief search always changes the root action. In robustness treatments, root-policy flips can move with chance samples. The stronger signal is continuous: determinization can assign optimistic value by stitching together incompatible future continuations from different hidden worlds.

That is why current work emphasizes measurements such as:

- determinization value bias;
- corrected public regret;
- information-set continuation conflict;
- which state structures predict those quantities;
- whether deeper search amplifies or reduces them;
- whether decision-quality differences eventually produce measurable battle outcomes.

Frozen experiment inputs and results live under [experiments/](experiments/). The implementation keeps negative results as evidence rather than tuning examples until a desired policy disagreement appears.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

~~~bash
uv sync
uv run pytest
uv run ruff check .
~~~

Simulator and JAX experiments use the optional simulator environment:

~~~bash
uv sync --extra simulator
~~~

### Run a local battle

poke-env expects a local Pokémon Showdown server on port 8000 by default.

~~~bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
~~~

Then, from this repository:

~~~bash
uv run azelficoast local --battles 10
~~~

Without a configured pinned Showdown checkout, the live player records evidence and uses its simple-heuristics fallback.

### Enable the bounded public-belief path

Pass a built pinned Showdown checkout with:

~~~bash
uv run azelficoast \
  --showdown-root /path/to/pokemon-showdown \
  local --battles 10
~~~

Only states admitted by the current live-belief implementation use public-belief search. Unsupported states fail closed to the heuristic policy rather than pretending broader mechanics coverage.

### Build a frozen decision corpus

Every instrumented run can emit battle outcomes, replay HTML, and decision traces under artifacts/.

~~~bash
uv run azelficoast corpus build artifacts/decisions.jsonl \
  --output artifacts/corpus.jsonl
~~~

Run a policy against the exact frozen fixtures:

~~~bash
uv run azelficoast corpus evaluate artifacts/corpus.jsonl \
  --policy first-legal
~~~

Policies can also be supplied as package.module:policy_object, allowing new search methods to be compared on identical information histories.

## Evidence model

The repository tries to keep claims narrower than the code around them.

A research claim is normally tied to:

1. a frozen state or preregistered population;
2. an exact Pokémon Showdown revision where mechanics matter;
3. deterministic or explicitly sampled hidden-world support;
4. an independent oracle or cross-implementation check;
5. exact-head CI evidence;
6. a stated boundary describing what the result does **not** establish.

Generated output is not treated as verified merely because it was produced. Negative controls, hostile dependency projections, and direct-vs-compressed execution checks are used throughout the simulator work.

## Where to look

| Area | Entry point |
| --- | --- |
| Live player and fallback boundary | [src/azelficoast/player.py](src/azelficoast/player.py) |
| Battle harness and trace capture | [src/azelficoast/harness.py](src/azelficoast/harness.py) |
| Frozen decision corpora | [src/azelficoast/corpus.py](src/azelficoast/corpus.py) |
| Live hidden-world reconstruction | [src/azelficoast/live_belief.py](src/azelficoast/live_belief.py) |
| Public-belief / fusion analysis | [src/azelficoast/fusion_search.py](src/azelficoast/fusion_search.py) |
| Natural-state mining | [src/azelficoast/real_belief_miner.py](src/azelficoast/real_belief_miner.py) |
| Exact Gen 9 damage semantics | [src/azelficoast/gen9_damage.py](src/azelficoast/gen9_damage.py) |
| Dependency-aware belief execution | [src/azelficoast/class_native_belief.py](src/azelficoast/class_native_belief.py) |
| Population study machinery | [src/azelficoast/population_study.py](src/azelficoast/population_study.py) |
| Frozen research evidence | [experiments/](experiments/) |
| Historical experiment narrative | [docs/research-notebook.md](docs/research-notebook.md) |

## Scope

Azelficoast currently targets Gen 9 Random Battles. Its simulator is deliberately incomplete: only mechanics with explicit Showdown-backed evidence belong to the admitted surface.

The project is not trying to replace Pokémon Showdown as the semantic authority. Showdown is the oracle; Azelficoast is trying to execute the battle-state distinctions relevant to search much more cheaply while preserving those semantics.

The longer-term question is simple to state:

> Can an AI reason over the uncertainty of a real competitive Pokémon battle without either exploding the state space or quietly granting itself information it does not have?

Azelficoast is the experiment for answering that question.
