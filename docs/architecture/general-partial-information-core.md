# General partial-information core

Azelficoast now separates reusable finite partial-information machinery from Pokémon
semantics.

## Boundary

The package under `azelficoast.core` may depend on Python data structures and generic
numerical interfaces. It must not depend on Pokémon species, moves, items, abilities,
Tera types, Showdown revisions, poke-env battle objects, or Random Battle generation.

The current reusable surface is:

- `core.contracts`: protocols for domain adapters, transition-oracle producers, and
  belief evaluators.
- `core.transition`: finite hidden-world/action transition validation and canonical
  hashing.
- `core.decision_relevance`: exact bounded quotienting of hidden worlds by decision
  semantics.
- `core.program`: structural transition-program lookup.
- `core.search`: determinization and information-set search over arbitrary finite
  transition programs.
- `core.projection`: integer-weight projection of finite support.

The core contracts are now canonical rather than compatibility-backed. Callers import
them directly from `azelficoast.core`; the former transition/search façade modules have
been removed. Persisted generic artifacts use core-owned schema identities:

- `azelficoast.core.transition-oracle`
- `azelficoast.core.transition-program-set`
- `azelficoast.core.weighted-transition-outcomes`
- `azelficoast.core.transition-program-verification`
- `azelficoast.core.decision-relevance-certificate`
- `azelficoast.core.partial-information-search`

A schema mismatch is a hard failure. There is no reader alias for the pre-cutover names.

## Domain adapter

Pokémon remains responsible for evidence acquisition and authoritative mechanics:

```text
Showdown / poke-env / Random Battle
              |
              v
      Pokémon domain adapter
              |
              v
finite worlds + transition programs
              |
              v
        azelficoast.core
   quotient / search / projection
              |
              v
       evidence + decisions
```

The generic layer does not attempt to simulate a domain. A second application can supply
its own hidden worlds, legal actions, public observations, successor public states, and
value evaluator while reusing the same information-flow rules.

## What is intentionally not generalized yet

`class_native_belief`, live Random Battle posterior reconstruction, Showdown build and
oracle integration, poke-env instrumentation, and Pokémon corpus normalization remain
domain code.

In particular, class-native support factors such as damage rolls or bench signatures
should not be renamed into generic abstractions until a second domain demonstrates which
structure is actually shared. Code generation is cheaper than committing the wrong
semantic boundary.

## Correctness rule

Generalization must preserve the existing research rule:

> Incidental representation choices may not change scientific results, and meaningful
> semantic mismatches must fail closed.

A reusable component therefore earns its place in `core` only when a non-Pokémon test
can exercise it without weakening Azelficoast's existing exact-result and hostile
correctness tests.
