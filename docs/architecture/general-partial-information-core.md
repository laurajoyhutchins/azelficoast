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
- `core.sql`: a read-only SQL front end over admitted decision relations, parsed by
  SQLite rather than by an Azelficoast-specific SQL grammar.
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


### SQL as a decision-query DSL

The relational planner now has an intentionally small executable SQL front end. The
writer-facing surface is deliberately smaller than the engine-facing schema. Ordinary
decision SQL starts from the read-only `action_value_terms` relation:

```sql
SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM action_value_terms
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC;
```

The canonical query is a first-class SQL source at
`src/azelficoast/queries/decision.sql`, and the admission/view schema lives beside it in
`decision_schema.sql`. Python loads those packaged resources rather than carrying a
second embedded copy of the SQL.

The exposed writer relations are:

```text
active_worlds(world_id, weight)

action_value_terms(action_id, weight, value)

action_statistics(
    action_id,
    posterior_mass,
    expected_value,
    worst_value,
    best_value
)
```

They are real SQLite views over the authoritative base relations:

```text
hidden_worlds
     |
     | active = 1 AND weight > 0
     v
active_worlds
     |
     +-- transitions
     +-- legal_actions
     +-- evaluations
     |
     v
action_value_terms
```

This keeps join plumbing out of ordinary query authoring without hiding the decision
calculation itself. The writer still chooses projection, aggregation, grouping, and
ordering in SQL. The exact view surface has its own content identity and can be exposed
to editors or future completion tooling with `describe_decision_sql_surface()`.

This is deliberately not a second mechanics engine. Trusted machinery still owns the
contents and identities of hidden worlds, legal actions, transitions, and evaluations.
The views only project already-admitted facts into a more useful relational vocabulary.

Admission uses Python's standard-library SQLite parser and authorizer. Only read-only
`SELECT` access and an explicit aggregate-function surface are accepted; mutation,
recursive SQL, unapproved functions, missing authority relations, and an unexpected
result shape fail closed. The exact SQL source is hashed, while SQLite's
`EXPLAIN QUERY PLAN` output is recorded together with the SQLite version because that
physical outline is environment-bound.

Admission and semantic recognition are separate. Portable recognition starts with a
finite rewrite family generated from reviewed relational rules:

```text
inner-join commutativity / associativity
predicate-conjunction commutativity
non-recursive projection-CTE inlining
```

Thus these may be different SQL source hashes but one logical query:

```text
canonical writer SQL --------+
inline SQL ------------------+--> decision semantic identity
reordered inner joins -------+
reordered filter predicates -+
```

There is also a stricter writer-ergonomics fallback for syntax SQLite itself erases. If
an admitted query compiles, under the exact same SQL surface and SQLite version, to the
same complete VDBE program as the canonical query, it receives the same semantic
identity with `sqlite-version-bound` equivalence evidence. This admits harmless forms
such as table aliases, redundant parentheses, and ordinal `ORDER BY` references without
teaching Azelficoast a second SQL parser.

```text
different SQL spelling
        |
        v
SQLite parse + authorization
        |
        v
EXPLAIN bytecode
        |
        +-- identical canonical program --> same decision semantics
        |
        +-- different program -----------> no authority
```

The VDBE path is deliberately environment-bound rather than treated as a portable proof.
EXPLAIN evidence records both the SQLite version and executable-program hash. The
portable reviewed relational rules remain separate evidence.

The writer surface now separates fixed query classes from source-derived policy SQL.
`decision.expected_value` retains the reviewed packed/JAX lowering, while
`analysis.action_summary` exposes reusable aggregate inputs through
`action_statistics`.

Policies use one generic `decision.policy` contract instead of registering one Python
query class per strategy. A writer supplies ordinary SQL over `action_statistics` that
returns `(action_id, score)`. Azelficoast derives the policy semantic identity from the
normalized SQL source plus the writer-surface and result contracts, then owns the final
deterministic ranking by `score DESC, action_id ASC`.

```text
action_value_terms
       |
       v
action_statistics
   /       |        \
  v        v         v
expected  maximin   risk-adjusted
value     SQL       SQL
  |        |         |
  v        x         x
packed/JAX   no physical lowering yet
```

Maximin and risk-adjusted policies are therefore examples of the same source-derived
policy language, not separate Python-side semantic classes. Editing a coefficient or
expression creates a new policy identity automatically; comments, whitespace, case, and
a trailing semicolon do not.

The semantic identity, rather than exact SQL spelling, keys any reviewed physical
lowering and planner statistics. A policy may be parsed, authorized, identified, and
explained without silently inheriting live execution authority.


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


### Advisory selectivity statistics

The SQL physical layer now has an `ANALYZE`-style in-memory statistics catalog for
logical row-reduction operators. Each histogram is keyed by logical operator plus a
semantic signature and records only observed input/output cardinalities.

The first consumers are `FILTER` and `PARTITION` in the reviewed decision SQL path.
Before execution, the planner emits smoothed forecasts for active posterior rows and
verified transition classes. After execution, exact packed/topology cardinalities are
fed back into the catalog and forecast error is recorded.

These statistics are deliberately **not authority**. A bad histogram may make a forecast
bad, but it cannot change the admitted SQL, transition program, information-set
partition, evaluator, or result. Exact validation and the cardinality envelope remain
the hard fences. The catalog is currently process-local; durable statistics should only
be added once invalidation identity and replay semantics are explicit.


### Extended statistics for correlated posterior fields

Single-column selectivity is not enough for Random Battle posteriors. Species, item,
ability, role, and move set are generated jointly, so multiplying independent marginal
estimates can be badly wrong.

The planner now measures weighted two-column dependency evidence for the admitted finite
posterior. The initial Pokémon profile records:

- species × item;
- species × ability;
- species × role;
- species × move-set.

For each pair it records distinct counts, total variation distance from the independent
product distribution, and weighted functional-prediction accuracy in both directions.
Those exact diagnostics are then reduced to a coarse correlation-regime signature.

Partition-cardinality history is keyed by that regime in addition to the semantic SQL
identity, search method, and transition-program schema. Equivalent SQL still shares one
statistics history, but materially different joint posterior structure no longer trains
the same partition estimator bucket.

This remains advisory. Extended statistics do not factorize the posterior, merge worlds,
change information sets, or authorize a physical path. Exact posterior validation,
verified transition classes, and the post-materialization cardinality fence remain the
authority.
