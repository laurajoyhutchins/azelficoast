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

This query asks for the exact expected value of every legal root action:

```sql
SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM action_value_terms
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC;
```

Because every row is part of the result contract, its packed/JAX lowering evaluates the
complete successor frontier.

### `decision.best_action`

This packaged query deliberately asks for less:

```sql
SELECT
    action_id,
    SUM(weight * value) AS expected_value
FROM action_value_terms
GROUP BY action_id
ORDER BY expected_value DESC, action_id ASC
LIMIT 1;
```

That `LIMIT 1` is semantic, not presentation sugar. It gives the physical planner
permission to use exact bound-based pruning. The packed value head is certified to stay
inside `[-1, 1]`; after an exact incumbent is established, a losing action may stop
evaluating leaves once its conservative upper bound is strictly below the incumbent's
conservative lower bound.

The result still contains the exact winning `action_id` and `expected_value`.
Unevaluated losers are retained only as certificate intervals in execution evidence.
They are never fabricated as SQL result rows.

Changing or removing `LIMIT 1` changes the semantic query class and therefore removes
this bounded-lowering authority.

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
        expected_value
            - :risk_aversion * (expected_value - worst_value) AS score
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
new identity; changing the SQL expression does.

### Policy parameters

Policies may use named SQLite parameters such as `:risk_aversion`. Parameter values are
bound separately from policy semantics:

```python
prepare_policy_query(
    RISK_ADJUSTED_SQL,
    parameters={"risk_aversion": 0.25},
)
```

The policy SQL keeps one semantic identity across a parameter sweep. Exact bindings get
their own deterministic evidence identity, and the prepared query derives a
`bound_semantic_identity` from the policy identity plus those bindings.

```text
risk_adjusted.sql
      |
      +--> policy semantic identity
      |
      +-- risk_aversion = 0.25 --> binding identity A --> bound identity A
      |
      +-- risk_aversion = 0.75 --> binding identity B --> bound identity B
```

Only named parameters are supported. Bindings must match the parameters in the parsed
SQLite program exactly. Values are restricted to finite SQLite integers and reals;
booleans, NaN, infinities, missing bindings, extra bindings, and positional parameters
fail closed. Integer and real values remain distinct in evidence, so `1` and `1.0`
cannot collapse accidentally.

Policies deliberately have no packed/JAX execution authority yet. They can be parsed,
authorized, identified, explained, compared, and reviewed as decision semantics without
silently becoming the live battle policy.

### Parameter sweep experiments

Policy experiments consume the SQL source and parameter grid as data. The generic
`azelficoast.research.policy_sweep` runner does not encode coefficient values.

The committed risk-adjusted experiment plan is:

```json
{
  "policy_resource": "risk_adjusted.sql",
  "parameter_grid": {
    "risk_aversion": [0.0, 0.25, 0.5, 0.75, 1.0]
  },
  "require_all_actions": true
}
```

For every grid point the runner prepares the same SQL policy, records its exact parameter
binding and bound semantic identity, and evaluates the same frozen
`action_statistics` fixtures. Evidence is settled only when the full Cartesian matrix
is present:

```text
one policy source
      |
      +-- binding 0 ----+
      +-- binding 1 ----+
      +-- binding 2 ----+--> same frozen fixtures --> matched evidence
      +-- binding 3 ----+
      +-- binding 4 ----+
```

The result records the plan identity, fixture-corpus identity, policy semantic identity,
every parameter-binding identity, every bound semantic identity, complete action
rankings, and chosen-action changes across the grid. Duplicate typed bindings,
incomplete matrices, non-finite fixture values, unknown actions, or nondeterministic
ranking fail closed.

This keeps scientific parameters in the experiment contract instead of hiding a sweep in
Python control flow.

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
