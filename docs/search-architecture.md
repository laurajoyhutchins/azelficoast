# Search and transition architecture

Azelficoast separates semantic authority from search execution.

Pokémon Showdown is the mechanics authority. Azelficoast may compress, specialize,
compile, cache, or vectorize transition work, but generated machinery is not accepted
as correct merely because it was produced. The search-facing mechanics artifact is a
verified whole-turn `TransitionProgram`.

## Core boundary

```text
public decision state + hidden-world support + root action
                         |
                         v
                pinned Pokémon Showdown
                  /               \
                 /                 \
        direct oracle          lazy producer
              |                     |
              |                     v
              |              TransitionProgram
              |                     |
              +------ verify --------+
                                    |
                                    v
                           search / evaluation
```

The direct oracle and the program producer are separate evidence paths. Verification
compares the program's representative execution classes with direct Showdown outcomes
for every member of the supplied finite support.

The important distinction is:

- **Showdown answers what the turn means.**
- **The posterior records which hidden worlds remain scientifically plausible.**
- **The program records which distinctions search actually has to execute.**
- **Search consumes the verified program, not the exhaustive oracle matrix.**

Posterior support and execution compression are deliberately different objects. A
hidden field may be irrelevant to the current root transition while remaining
meaningful to the posterior, learned evaluator, or later decisions. In particular,
moves and Tera type are retained in semantic Random Battle worlds even when the
current fixed-response mechanics program proves that it can execute several such
worlds through one class. The compression belongs in `TransitionProgram.classes`;
it must not delete or merge posterior worlds.

## TransitionProgram contract

A program is compiled per legal root action. Its execution classes bind hidden worlds
that have identical immediate whole-turn semantics on the supplied support.

Each chance outcome exposes only search-facing transition data:

- probability;
- public observation;
- public successor state;
- legal successor actions.

Historical hand-written continuation utility values are not part of the mechanics
contract. Leaf value comes from the learned evaluator used by the search experiment or
live policy.

A program also records evidence such as:

- exact Showdown revision;
- source fixture identity;
- hidden-world support;
- dependency fields;
- partition method;
- representative world for each class;
- class membership;
- semantic hashes;
- partition/effect signatures.

The resulting digest is part of matched-search evidence.

## Successor action semantics

"No normal move choice is available" is not synonymous with "the battle ended."

The program therefore distinguishes:

- normal legal actions;
- `<wait>`, when Showdown's public request requires the player to wait;
- `<terminal>`, when the battle is actually over.

This distinction is part of the verified transition semantics. It was exposed by the
real Gliscor/Urshifu candidate proof: after some root outcomes the public request is a
forced wait even though the battle continues.

## Information-set search

Both matched methods consume the same program and learned evaluator.

```text
TransitionProgram
      |
      +--> determinization
      |      condition on one hidden world at a time
      |
      +--> information_set
             regroup by public observation
             preserve posterior mass
             choose once per public information set
      |
      v
learned successor value
```

Determinization and information-set search differ in what information future choices
may use. They do not get different mechanics artifacts.

For information-set search, outcomes sharing the same public observation are combined
into a successor belief. Hidden-world mass is renormalized inside that information set,
and the learned evaluator receives the public successor state, posterior, and common
legal action surface.

## Matched experiment evidence

`research/matched_comparison.py` freezes the public input, posterior, evaluator, Showdown
revision, depth, and compute ceiling.

The receipt executor then binds each method to:

- the frozen packet;
- the evaluator checkpoint;
- the verified `TransitionProgram` digest;
- consumed transition-class count;
- learned evaluator-call count;
- root values and the derived chosen action.

Settlement requires both methods to name the same program digest. It fails closed if
either receipt changes the frozen input, evaluator, authorized budget, program,
root-value coverage, or chosen-action semantics.

The compute unit is a verified whole-turn execution class consumed by search. This is
deliberate: once several hidden worlds are proved equivalent for the whole root turn,
charging search as though each world had been independently executed would erase the
compression the program exists to provide.

Legacy frozen experiments may still present an exhaustive oracle at the receipt
boundary. That oracle is first compiled into a `TransitionProgram`; the search engine
itself does not walk the old world-by-action matrix.

## Live policy path

Live execution separates three different capabilities that used to be hidden behind one
narrow admission gate:

```text
DecisionFixture
      |
      v
Showdown state reconstruction
      |
      v
generator-faithful hidden-world posterior
      |
      v
learned policy/value model
   |             |
confident      uncertain
   |             |
   v             v
 action     opponent model available?
                    |             |
                   no            yes
                    |             |
                    v             v
              fail closed   TransitionProgram
                                  |
                                  v
                        information-set search
                                  |
                                  v
                                action
```

Posterior reconstruction does not require an opponent-response policy. A high-margin
learned decision may therefore use a reconstructable posterior even before the exact
search path has enough evidence to model the opponent's next choice.

Exact search is stricter. The current bounded opponent model repeats the latest move
observed from the current opposing active Pokémon. When that evidence is absent,
Azelficoast reports an explicit opponent-model gap rather than treating posterior
reconstruction, mechanics execution, and opponent behavior as one failure.

The posterior is generator-faithful: known public item evidence constrains the generated
set support, while unknown items remain uncertain instead of being restricted to Choice
items. Once an opponent response is available, pinned Showdown executes the complete
turn, including mechanics that do not have a hand-written Azelficoast kernel.

If program production or program search cannot be validated, the existing exhaustive
analysis path remains available as a compatibility/recovery boundary. Unsupported
states ultimately fail closed to the outer player fallback instead of silently
pretending broader coverage.

## Verification discipline

Three different statements must not be conflated:

1. **Produced:** a program was generated.
2. **Verified:** direct Showdown execution matched every program class member on the
   supplied finite support.
3. **General:** the same dependency projection is valid for unseen worlds, different
   actions, later turns, or another Showdown revision.

Azelficoast currently claims the second when its verifier passes. It does not infer the
third from one successful program.

Candidate CI exercises this distinction explicitly. The real-belief decision trace
generates both a direct oracle and a lazy whole-turn program, then independently checks
their equivalence before accepting the compression evidence.

## What this replaces

The project still contains bounded owned mechanics kernels and their Showdown-backed
tests. They remain useful for fast execution and research into compiled semantics.

What changed is the architectural unit presented to search.

Search no longer needs a roadmap that says "implement mechanic A, then mechanic B,
then mechanic C" before it can use compressed execution. Its contract is the complete
root transition. Mechanic-specific machinery can improve how that program is executed,
but it does not define the search API.

That keeps the optimization target where the scientific question lives:

> same public decision state, same posterior, same compute budget, same mechanics
> authority, different information-set discipline.
