# Azelficoast

Azelficoast is an evidence-first research system for competitive Pokémon battle AI under partial observability.

The project is investigating a specific failure mode of conventional search: **determinization can plan future actions using hidden information the player will not actually have**. Azelficoast compares that approach with public-belief search, where future choices must respect the information set visible at that point in the battle.

It is also building the machinery needed to make that comparison on real Pokémon states: instrumented battle traces, hidden-world reconstruction, pinned-Showdown transition evidence, verified whole-turn `TransitionProgram`s, learned policy/value evaluation, and compressed belief-state execution.

> **Status:** active research code, not a finished competitive bot. When the learned evaluator and pinned Showdown path are configured, confident admitted positions may use the learned policy directly and uncertain positions search a verified `TransitionProgram`. Unsupported or unverifiable states fail closed to the existing fallback policy.

## Matched experiment contract

The central experiment is executable, not a naming convention. `research/matched_comparison.py`
freezes one public decision state, one posterior, one Showdown revision, one depth,
and one compute ceiling into paired determinization and information-set work packets.
Settlement fails closed if either method changes the input, legal actions, budget,
evaluator identity, transition-program identity, or chosen-action semantics. Transition
compute is accounted in verified whole-turn execution classes consumed, rather than
silently charging the exhaustive hidden-world × action matrix after compression.

`research/matched_population.py` then requires the complete natural-state × posterior × depth
matrix and emits battle-clustered bootstrap rows for value optimism, regret, and policy
disagreement. The posterior treatment and opponent model remain explicit in every
packet and result.

The current natural-state oracle work still has a narrower boundary: it fixes the
observed opponent response. A result is marked as preserving two-player information
sets only when the executable opponent-model contract says so.

~~~bash
python -m azelficoast.research.matched_comparison freeze PLAN STATE POSTERIOR \
  --posterior-treatment generator_faithful --depth 2 --output packet.json
python -m azelficoast.research.matched_comparison settle packet.json DET.json INFO.json \
  --output result.json
python -m azelficoast.research.matched_population PLAN COHORT RESULTS \
  --output aggregate.json
~~~

## What exists today

~~~text
real decision state
      |
      +--> public trace --> hidden-world posterior
      |
      v
pinned Pokémon Showdown
      |
      +--> direct finite-support oracle -----------+
      |                                           |
      v                                           | verify
whole-turn TransitionProgram <--------------------+
      |
      +--> public successor state
      +--> public observation
      +--> successor legal actions
      +--> weighted execution classes
      |
      v
determinization / information-set search
      |
      v
same learned evaluator
~~~

The repository currently contains:

- a Gen 9 Random Battle harness for local, challenge, accept, and ladder play;
- replayable, content-addressed decision corpora;
- live decision tracing that preserves what was actually public at decision time;
- hidden-world reconstruction from Showdown generator support and public evidence;
- determinization and information-set-preserving public-belief search on the same frozen states;
- a verified whole-turn `TransitionProgram` boundary consumed directly by matched search and uncertain live learned search;
- pinned-Showdown direct-oracle verification of each finite-support program;
- dependency signatures and execution classes that collapse hidden worlds when the complete root turn cannot distinguish them;
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

Only states admitted by the current live-belief implementation enter this path. A confident learned prediction may choose directly; an uncertain learned prediction is evaluated through the whole-turn `TransitionProgram` and successor public beliefs. The older exhaustive-oracle path remains a compatibility/recovery fallback, not the primary learned-search interface. Unsupported states fail closed rather than pretending broader mechanics coverage.

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

### Build policy/value training records

Training records are joined from real completed battle traces and independently
settled matched-search evidence. The builder re-runs matched settlement from the
frozen packet plus both method receipts, resolves the exact posterior by digest, and
then attaches two value signals: the eventual battle outcome and the information-set
searched return.

~~~bash
uv run azelficoast training build artifacts/decisions.jsonl \
  --search-packet artifacts/search/packet.json \
  --search-receipt artifacts/search/determinization.json \
  --search-receipt artifacts/search/information-set.json \
  --posterior artifacts/search/posterior.json \
  --output artifacts/training.jsonl
~~~

Policy labels are never copied from the heuristic behavior player, and loose search
annotations are not accepted as training authority. The output preserves the settled
information-set root values, the exact posterior, teacher checkpoint digest, Showdown
revision, depth, authorized/consumed compute, evaluator-call count, and the
`TransitionProgram` digest shared by both matched methods.

Splitting is battle-grouped. If the same content-addressed fixture occurs in multiple
battles, those battles are joined into one split group so an identical information
state cannot appear on opposite sides of the train/validation/test boundary.

## Evidence model

The repository tries to keep claims narrower than the code around them.

A research claim is normally tied to:

1. a frozen state or preregistered population;
2. an exact Pokémon Showdown revision where mechanics matter;
3. deterministic or explicitly sampled hidden-world support;
4. an independent oracle or cross-implementation check;
5. exact-head CI evidence;
6. a stated boundary describing what the result does **not** establish.

Generated output is not treated as verified merely because it was produced. In particular, lazy whole-turn programs are checked against an independently generated direct Showdown oracle before their compression claim is accepted. Negative controls, hostile dependency projections, and direct-vs-compressed execution checks are used throughout the simulator work.

## Where to look

| Area | Entry point |
| --- | --- |
| Live player and fallback boundary | [src/azelficoast/live/player.py](src/azelficoast/live/player.py) |
| Battle harness and trace capture | [src/azelficoast/live/harness.py](src/azelficoast/live/harness.py) |
| Frozen decision corpora | [src/azelficoast/corpus.py](src/azelficoast/corpus.py) |
| Policy/value training records | [src/azelficoast/research/training_records.py](src/azelficoast/research/training_records.py) |
| Live hidden-world reconstruction and routing | [src/azelficoast/live/belief.py](src/azelficoast/live/belief.py) |
| Whole-turn program compiler/verifier | [src/azelficoast/whole_turn_program.py](src/azelficoast/whole_turn_program.py) |
| Search over verified programs | [src/azelficoast/search/transition_program.py](src/azelficoast/search/transition_program.py) |
| Matched search receipt executor | [src/azelficoast/research/matched_search.py](src/azelficoast/research/matched_search.py) |
| Public-belief / fusion analysis | [src/azelficoast/search/fusion.py](src/azelficoast/search/fusion.py) |
| Natural-state mining | [src/azelficoast/real_belief_miner.py](src/azelficoast/real_belief_miner.py) |
| Exact Gen 9 damage semantics | [src/azelficoast/gen9_damage.py](src/azelficoast/gen9_damage.py) |
| Dependency-aware belief execution | [src/azelficoast/class_native_belief.py](src/azelficoast/class_native_belief.py) |
| Population study machinery | [src/azelficoast/research/population.py](src/azelficoast/research/population.py) |
| Frozen research evidence | [experiments/](experiments/) |
| Current search/mechanics architecture | [docs/search-architecture.md](docs/search-architecture.md) |
| Historical experiment narrative | [docs/research-notebook.md](docs/research-notebook.md) |
| Data and evidence provenance | [docs/data-provenance.md](docs/data-provenance.md) |

## Scope

Azelficoast currently targets Gen 9 Random Battles. The owned compiled mechanics surface remains deliberately incomplete, but the search boundary is now whole-turn rather than mechanic-by-mechanic: a `TransitionProgram` is valid only for its supplied hidden-world support, root legal actions, public state, and exact Showdown revision.

The project is not trying to replace Pokémon Showdown as the semantic authority. Showdown remains the oracle. Azelficoast is trying to discover and execute only the battle-state distinctions that matter to a particular search transition, then independently prove that the compressed program agrees with direct Showdown execution on that finite support.

## License and third-party material

Azelficoast's original source code is licensed under the [MIT License](LICENSE).
Third-party software, data, trademarks, battle records, and other materials retain
their own terms and are not relicensed merely by appearing in or being used by
this repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[docs/data-provenance.md](docs/data-provenance.md).

Azelficoast is an unofficial research project and is not affiliated with or
endorsed by Nintendo, Game Freak, Creatures, The Pokémon Company, Pokémon
Showdown, or Smogon.

The longer-term question is simple to state:

> Can an AI reason over the uncertainty of a real competitive Pokémon battle without either exploding the state space or quietly granting itself information it does not have?

Azelficoast is the experiment for answering that question.
