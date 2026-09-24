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
