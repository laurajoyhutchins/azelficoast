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
- `core.costing`: calibrated, identity-bound structural cost models for equivalent exact
  execution paths.
- `core.planning`: a small logical-plan / physical-plan boundary with deterministic
  EXPLAIN evidence for execution-path selection.
- `core.memo`: bounded memo groups for exact materialized semantic results; callers
  own equivalence identity and physical-alternative identity.
- `core.program`: structural transition-program lookup.
- `core.search`: determinization and information-set search over arbitrary finite
  transition programs.
- `core.projection`: integer-weight projection of finite support, including active-support
  compaction that pushes zero-mass filtering ahead of aggregation while preserving the
  verified global partition identity.

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


## Cost-based execution planning

The generic core now treats execution selection as a query-planning problem. Search
defines the logical computation; planning chooses between semantically equivalent exact
physical paths using a calibrated cost profile.

The initial logical algebra is intentionally small:

```text
scan -> filter -> project -> partition -> transition
     -> observe -> update_belief -> evaluate -> aggregate
```

This does not change battle or research semantics. The first physical choice is the
existing measured direct-versus-projected execution decision, promoted out of the
research namespace. `explain_physical_plan()` records the logical operators, structural
cardinalities, cost predictions, calibration identities, uncertainty guard, and selected
path so planner behavior is auditable and can later become optimization evidence.

Hardware/JAX target discovery remains research/runtime-specific; the core only consumes
an opaque target signature.


### Active-support pushdown

A compiled projection remains the semantic authority for class membership. Runtime
posterior updates frequently leave most canonical support classes at zero mass, so the
execution path now compacts that verified map before aggregation:

```text
canonical support
      |
      | filter weight > 0
      v
active canonical rows
      |
      | map through verified global class_ids
      v
active execution classes
      |
      | aggregate only touched classes
      v
projected batch
```

The optimization deliberately preserves global class IDs and the projection's canonical
representatives. It changes allocation and work, not equivalence semantics or evidence
identity. The adaptive-execution benchmark and bounded two-attack-turn runtime both use
this active slice rather than allocating a dense weight vector for every projection
class on each execution.


### Materialized evaluation frontiers

The first memoized semantic view is the evaluation frontier between verified mechanics
and learned numerical evaluation:

```text
verified program + posterior + search method
                  |
                  v
          memo group identity
             /          \
            v            v
   Python frontier   future physical alternatives
            |
            v
      learned evaluator
```

The memo key is conservative. It binds the admitted transition-program identity,
mechanics evidence, posterior semantics, transport binding, and search method. A hit may
skip rebuilding the successor information-set frontier, but it does not skip evaluator
execution and it does not reduce the reported verified transition-class count.

Accordingly search reports both semantic work and physical reuse:
`transition_evaluations` remains the scientific work unit, while
`frontier_memo_hit` and `frontier_builds` expose whether the physical frontier was
materialized during this call. Memo groups are bounded LRU state, not durable authority.
