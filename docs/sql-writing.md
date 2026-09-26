# Writing Azelficoast decision SQL

Azelficoast's decision DSL is ordinary read-only SQLite.

The canonical query is not embedded in Python. Edit
`src/azelficoast/queries/decision.sql` as SQL. The companion
`decision_schema.sql` defines the empty admission schema and the writer-facing views
used by SQL tooling and semantic recognition.

## Writer-facing relations

Ordinary decision queries should prefer these relations:

```text
active_worlds(world_id, weight)
action_value_terms(action_id, weight, value)
action_statistics(action_id, posterior_mass, expected_value, worst_value, best_value)
```

The canonical policy is intentionally small:

```sql
SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM action_value_terms
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC;
```

`action_value_terms` hides mechanical join plumbing, not policy semantics. It expands
over the admitted hidden-world, transition, legal-action, and evaluation relations.
Writers still express aggregation, grouping, and ranking in SQL.

## What counts as the same query

SQLite parses and authorizes every query first. Azelficoast then has two equivalence
routes.

Reviewed relational rewrites are portable evidence. They currently cover inner-join
commutativity/associativity, active-support predicate reordering, and the reviewed CTE
inlining forms.

Writer spellings that are not in that finite family can still be accepted when SQLite
compiles them to the exact same VDBE program as the canonical query under the same SQL
surface and SQLite version. This lets aliases, redundant parentheses, comments, and
equivalent ordinal ordering stay ordinary SQL rather than becoming Azelficoast syntax.

Program equivalence is explicitly SQLite-version-bound. The explain receipt records the
SQLite version and executable-program hash.

## EXPLAIN AZELFICOAST

The packed decision path exposes a read-only optimizer receipt through
`explain_sql_packed_transition_program(...)`. It is the programmatic equivalent of
`EXPLAIN AZELFICOAST`: the SQL is parsed and semantically recognized, the verified
transition topology is compiled, and the planner reports the complete logical-to-physical
mapping without invoking the learned evaluator or choosing an action.

The receipt includes:

- the normal SQL admission and SQLite planner evidence;
- the nine semantic operators from scan through aggregate;
- fused physical groups for posterior packing, authorized topology construction, belief
  transport, packed evaluation, and root reduction;
- the costed outcome-aggregation/world-join order selected by compiled search;
- cheap lower-bound and realized topology cardinalities;
- advisory FILTER/PARTITION forecasts and posterior extended statistics when a planner
  statistics catalog is supplied;
- transition-program and compiled-topology identities; and
- an explicit authority record showing that mechanics and information-set grouping stay
  outside SQL.

Calling explain does not add observations to `PlannerStatistics`, run JAX, call the
learned evaluator, or select a battle action. This keeps introspection from changing the
plan it is trying to inspect.

```text
SQL source
   |
   v
semantic operators
   |
   +--> posterior-pack
   +--> authorized-topology
   |       `--> costed outcome/world join order
   +--> belief-transport
   +--> packed-evaluator
   +--> root-reduction
   |
   v
read-only optimizer receipt
```

## Relational information-set transport

The first multi-stage search calculation now exists as executable SQL rather than only
as a logical-operator label. `information_set_transport.sql` consumes an already
authorized leaf/world/chance incidence relation and performs:

```text
normalize prior world weights
        |
        v
worlds JOIN chance edges
        |
        v
SUM mass BY leaf, world
        |
        v
SUM mass BY leaf
        |
        v
conditional posterior = leaf/world mass / leaf mass
```

`transport_information_set_mass_sql(...)` executes that packaged query in SQLite and
returns the same dense `leaf_mass`, `leaf_world_mass`, and conditional
`leaf_world_weights` surfaces consumed by compiled search. Candidate coverage compares
this relational reference directly against the JAX transport on the exact same
Python-authorized topology.

This deliberately does **not** let SQL decide which worlds share an observation or
which edge belongs to which leaf. Those incidences still come from the independently
validated `CompiledSearchTopology`. SQL owns the numerical relational transform after
that authority boundary.

`EXPLAIN AZELFICOAST` now reports this relational transport contract alongside the
selected packed/JAX physical path. That makes the SQL implementation an executable
reference alternative without silently promoting it into the live hot path.

## Fail-closed rule

Syntactically valid SQL does not automatically become executable decision semantics.

A fixed query class may parse successfully and still receive no Azelficoast semantic
identity when it falls outside its reviewed equivalence rules. Source-derived policies
receive an identity for their exact normalized policy source, but that identity alone
does not grant packed/JAX execution. Physical lowering remains a separate reviewed
authority.

That boundary is intentional: SQL is the source language, but mechanics, posterior
meaning, evaluator semantics, and evidence admission remain outside the language.


## Named semantic query classes

SQL is no longer limited to alternate spellings of one expected-value query. The writer
surface has named semantic query classes, each with its own result contract and semantic
identity.

### `decision.expected_value`

This is the existing live decision policy and the only query class currently lowered to
the packed/JAX executor:

```sql
SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM action_value_terms
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC;
```

### `analysis.action_summary`

The `action_statistics` relation exposes reusable policy inputs:

```text
action_statistics(
    action_id,
    posterior_mass,
    expected_value,
    worst_value,
    best_value
)
```

The packaged `action_summary.sql` query returns those metrics directly. It is admitted
as its own semantic class but is not a live decision policy.

### `decision.policy`

Policies are now source-derived rather than registered one at a time in Python. A policy
is ordinary read-only SQL that directly reads `action_statistics` and returns exactly:

```text
action_id
score
```

Azelficoast owns the final ranking wrapper:

```sql
ORDER BY score DESC, action_id ASC
```

so every policy has deterministic tie breaking without repeating ranking boilerplate.

For example, `maximin.sql` is just a policy body:

```sql
WITH policy AS (
    SELECT
        action_id,
        worst_value AS score
    FROM action_statistics
)
SELECT
    action_id,
    score
FROM policy;
```

and `risk_adjusted.sql` composes the same relation differently:

```sql
WITH scored AS (
    SELECT
        action_id,
        expected_value - 0.25 * (expected_value - worst_value) AS score
    FROM action_statistics
)
SELECT
    action_id,
    score
FROM scored;
```

Neither policy needs a new Python query class. Its semantic identity is derived from the
normalized policy source, writer-surface identity, result contract, and system-owned
ranking contract. Comments, whitespace, case, and a trailing semicolon do not create a
new identity; a changed coefficient or expression does.

Policies deliberately have no packed/JAX execution authority yet. They can be parsed,
authorized, identified, explained, compared, and reviewed as decision semantics without
silently becoming the live battle policy.

That separation is the point of the named-query layer:

```text
SQL source
   |
   v
query class / policy source
   |
   +--> semantic identity
   |
   +--> result contract
   |
   +--> deterministic ranking contract
   |
   +--> optional physical lowering
```

Adding or editing a useful policy does not require Python registration and does not
silently grant it live battle authority.
