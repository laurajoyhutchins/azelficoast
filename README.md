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

Decision traces use schema `azelficoast.decision-trace` with an integer `schema_version` and a monotonic `event_index`. Unknown opponent information remains unknown in the snapshots; the recorder does not fill hidden fields from later knowledge.

To place traces elsewhere, pass the global option before the command:

```bash
uv run azelficoast --decisions /tmp/jaxcalibur.jsonl challenge Jaxcalibur --battles 1
```

Future policy probabilities, beliefs, values, and search diagnostics can be added to the decision records without changing battle orchestration.

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
