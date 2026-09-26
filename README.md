# Azelficoast

> A competitive Pokémon AI that tries very hard not to become psychic by accident.

<code>SELECT best_move FROM battle;</code>

Azelficoast is an experimental **Gen 9 Random Battle bot** built around a simple idea:

**A good Pokémon AI should reason about what it does not know, not quietly pretend it knows the opponent's hidden set.**

It watches a battle, builds a belief about the hidden possibilities, searches across those possibilities, and uses Pokémon Showdown itself to check the mechanics it relies on.

The fun part is what happened next: making that practical pulled in ideas from game-tree search, probabilistic inference, compilers, databases, JAX, and reproducible scientific experiments.

> **Status:** active research code, not a finished ladder monster. The sophisticated path is deliberately bounded. If Azelficoast cannot prove that a state is inside the mechanics it understands, it fails closed to a simpler fallback instead of inventing an answer.

## The 30-second version

A normal battle bot can easily do something subtly unfair:

1. Sample one possible hidden opponent set.
2. Search as if that sampled set were the truth.
3. Make later decisions using information the real player still would not have.
4. Average the results and call it uncertainty-aware.

That can make the search look smarter than it really is.

Azelficoast instead asks:

> What move is good when all of my future decisions are restricted to the information I would actually have at that moment?

That is the core experiment.

## The problem: future-you is not allowed to become psychic

Imagine the opponent has several plausible hidden items, moves, abilities, or Tera types.

A search algorithm can explore each possible world separately. The trap is that, several turns later, each branch may choose a different "perfect" continuation because that branch knows which hidden world it lives in.

The real player does not get that knowledge.

So this:

~~~text
possible world A -> future me chooses move X
possible world B -> future me chooses move Y
possible world C -> future me switches
~~~

can accidentally be scored as though one real player gets all three superpowers.

This is the classic **strategy-fusion** problem.

Azelficoast keeps future choices tied to the public information available in the battle. If two hidden worlds still look identical to the player, future-you has to make the same decision in both.

## What happens on a turn?

At a high level:

~~~text
Pokémon Showdown battle
        |
        v
what is publicly known?
        |
        v
plausible hidden worlds + probabilities
        |
        v
can the learned evaluator answer confidently?
      /   \
    yes    no
     |      |
     |      v
     |   compile the whole turn
     |      |
     |   verify it against pinned Showdown
     |      |
     |   collapse hidden worlds that behave identically
     |      |
     |   search the remaining uncertainty
     |      |
     +------+
        |
        v
choose a legal action
~~~

If the state is outside the verified mechanics surface, Azelficoast does not stretch the proof until it fits. It records the gap and uses the fallback policy.

## Why this is cool

### 1. It carries beliefs, not just guesses

Azelficoast reconstructs plausible hidden worlds from the information actually revealed in the battle.

That means the search can reason over questions such as:

- Which opponent sets are still possible?
- How much probability mass belongs to each one?
- Which differences matter for this turn?
- Which hidden details are irrelevant to the decision?

The posterior is part of the input, not secret knowledge smuggled in from the simulator.

### 2. It compiles whole Pokémon turns

Running Pokémon Showdown once is easy. Running it for every action, hidden set, chance outcome, and search node is expensive.

Azelficoast builds a **whole-turn TransitionProgram** for the current finite support.

The program captures things like:

- legal root actions;
- successor public states;
- observations revealed by the turn;
- next legal actions;
- weighted execution classes.

Then it checks that program against direct execution in a pinned Pokémon Showdown revision.

Hidden worlds that produce the same relevant behavior can be executed together instead of one by one.

So the basic trick is:

~~~text
thousands of plausible worlds
        |
        v
which distinctions can affect this turn?
        |
        v
much smaller verified execution classes
        |
        v
search those
~~~

The compression is only accepted when direct Showdown execution agrees on the supplied support.

### 3. The search engine has started looking suspiciously like a database

This is intentional.

Azelficoast now has a **read-only SQL decision layer** and a compiled JAX data plane. Search planning can use database-style ideas such as active-support pushdown, memoized frontiers, measured selectivity, cardinality-sensitive replanning, and correlation-aware statistics.

In less database-shaped English:

> Do less work early, keep reusable intermediate results, and spend compute only on distinctions that can still change the answer.

The SQL layer is not allowed to mutate battle authority. It is a query language for asking questions of the decision state. The canonical query is a real SQL file at `src/azelficoast/queries/decision.sql`; `docs/sql-writing.md` describes the writer surface and equivalence rules.

Yes, the joke at the top of the README is becoming architecture.

### 4. It can learn without letting training grade its own homework

Azelficoast can bootstrap from public Gen 9 Random Battle replays, build training records, train candidate evaluators, and run repeated improvement cycles.

But a candidate does not become the live evaluator merely because training produced it.

Promotion is evidence-gated. The repository checks held-out performance and hostile invariants, keeps immutable receipts, and changes the live pointer only when the candidate passes the admission rules.

The intended loop is:

~~~text
generate or import battles
        |
        v
find informative public states
        |
        v
reconstruct beliefs
        |
        v
run verified search teachers
        |
        v
train a candidate
        |
        v
hostile + scientific evaluation
        |
        v
promote only if the evidence says yes
~~~

## Try it

Azelficoast requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

~~~bash
uv sync
uv run pytest
uv run ruff check .
~~~

That is enough to explore the repository and run the ordinary test suite.

### Run a local battle

For live simulator work, install the optional simulator dependencies:

~~~bash
uv sync --extra simulator
~~~

Run a local Pokémon Showdown server:

~~~bash
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
~~~

Then, from Azelficoast:

~~~bash
uv run azelficoast local --battles 10
~~~

Without a configured pinned Showdown checkout, the live player can still record evidence and use the simple fallback policy.

To enable the bounded verified path:

~~~bash
uv run azelficoast \
  --showdown-root /path/to/pokemon-showdown \
  local --battles 10
~~~

Only admitted states enter the sophisticated path. Unsupported states fail closed instead of pretending the bot understands mechanics it has not verified.

## Learn from public battles

Azelficoast can use already-public Gen 9 Random Battle replays as a cold start rather than making unsuspecting ladder players train a badly tuned bot.

Import public replay evidence:

~~~bash
uv run azelficoast \
  --showdown-root /path/to/pinned/pokemon-showdown \
  corpus import-public \
  --max-battles 1000 \
  --min-rating 1500 \
  --output-root artifacts/public-replays
~~~

For an initial evaluator, public human decisions can be used explicitly as **imitation pretraining**:

~~~bash
uv sync --extra simulator

uv run azelficoast \
  --showdown-root /path/to/pinned/pokemon-showdown \
  training bootstrap-public artifacts/public-replays/decisions.jsonl
~~~

After the first checkpoint exists, the normal improvement cycle returns to settled information-set search for policy targets:

~~~bash
uv run azelficoast \
  --showdown-root /path/to/pinned/pokemon-showdown \
  training cycle artifacts/public-replays/decisions.jsonl
~~~

There is also an unattended local generation loop:

~~~bash
uv run azelficoast \
  --showdown-root /path/to/pokemon-showdown \
  training auto \
  --incumbent artifacts/evaluators/current.json \
  --generations 5 \
  --battles-per-generation 24 \
  --max-teacher-fixtures 64
~~~

The opponent league includes several styles rather than training only against RandomPlayer.

## What has the research found so far?

The strongest result is not "public-belief search always picks a different move." Real states are messier than that.

The more durable finding is that ordinary determinization can assign **optimistic value** to plans by stitching together future continuations that cannot all be available to one player under the same information history.

Azelficoast therefore measures things such as:

- determinization value bias;
- public regret;
- continuation conflicts between hidden worlds;
- policy disagreement;
- how those effects change with search depth and posterior assumptions;
- whether decision-quality improvements eventually show up in battle outcomes.

Frozen experiments live under [experiments/](experiments/). Negative results stay in the record too.

## Why all the verification machinery?

Because this project is very easy to fool.

A fast search is useless if it silently changes battle semantics. A beautiful experiment is useless if the two methods saw different inputs. A compressed transition is useless if the compression erased a distinction that matters.

So Azelficoast treats claims as things that need receipts.

When mechanics matter, evidence is tied to an exact Pokémon Showdown revision. Matched experiments freeze the decision state, posterior, legal actions, evaluator identity, transition-program identity, search depth, and compute budget. Generated transition programs are checked against direct Showdown execution before their compressed result is trusted.

The rule is simple:

> Generated output is not automatically verified output.

If you want the detailed scientific contract, start with [docs/search-architecture.md](docs/search-architecture.md), [docs/data-provenance.md](docs/data-provenance.md), and the code under [src/azelficoast/research/](src/azelficoast/research/).

## Repository map

| Area | What it is |
| --- | --- |
| [src/azelficoast/live/](src/azelficoast/live/) | Battle harness, live player, trace capture, belief routing |
| [src/azelficoast/core/](src/azelficoast/core/) | Core transition and mechanics-facing contracts |
| [src/azelficoast/search/](src/azelficoast/search/) | Search over public beliefs and verified transitions |
| [src/azelficoast/research/](src/azelficoast/research/) | Shared research contracts/orchestration; studies, experiments, mechanics, and verification live in explicit subpackages |
| [experiments/](experiments/) | Frozen research evidence |
| [docs/search-architecture.md](docs/search-architecture.md) | Deeper explanation of the search architecture |
| [docs/data-provenance.md](docs/data-provenance.md) | Where evidence comes from and how it is fenced |
| [docs/research-notebook.md](docs/research-notebook.md) | Historical research narrative |

Useful entry points:

- [src/azelficoast/live/player.py](src/azelficoast/live/player.py) — live decision boundary;
- [src/azelficoast/live/belief.py](src/azelficoast/live/belief.py) — hidden-world reconstruction and routing;
- [src/azelficoast/core/whole_turn_program.py](src/azelficoast/core/whole_turn_program.py) — whole-turn compiler and verifier;
- [src/azelficoast/search/transition_program.py](src/azelficoast/search/transition_program.py) — search over compiled transitions;
- [src/azelficoast/research/studies/matched_search.py](src/azelficoast/research/studies/matched_search.py) — matched search execution;
- [src/azelficoast/research/mechanics/](src/azelficoast/research/mechanics/) — mechanics experiments and independent checks.

## Current boundaries

Azelficoast currently targets **Gen 9 Random Battles**.

The owned compiled mechanics surface is intentionally incomplete. Pokémon Showdown remains the semantic authority. Azelficoast is not trying to rewrite the simulator.

Instead, it asks a narrower question:

> For this decision, which hidden distinctions actually matter, and can we prove that a faster representation behaves exactly like Showdown on the worlds we are considering?

That boundary is important. The project would rather say "I do not know how to verify this state yet" than make the README sound more complete than the code.

## The long-term goal

Build a Pokémon bot that can:

- learn from real public play;
- maintain honest uncertainty about hidden information;
- search deeply without exploding the state space;
- reuse compiled and cached work aggressively;
- improve itself behind deterministic evidence gates;
- and still trace every important mechanics claim back to Pokémon Showdown.

Or, in one sentence:

> **Can we make a battle AI that is fast, strong, uncertainty-aware, and provably not sneaking future information into its own search?**

That is Azelficoast.

## License and third-party material

Azelficoast's original source code is licensed under the [MIT License](LICENSE).

Third-party software, data, trademarks, battle records, and other materials retain their own terms and are not relicensed merely by appearing in or being used by this repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [docs/data-provenance.md](docs/data-provenance.md).

Azelficoast is an unofficial research project and is not affiliated with or endorsed by Nintendo, Game Freak, Creatures, The Pokémon Company, Pokémon Showdown, or Smogon.
