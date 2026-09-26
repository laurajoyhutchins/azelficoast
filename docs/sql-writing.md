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

## Fail-closed rule

Syntactically valid SQL does not automatically become executable decision semantics.

A query may parse successfully and still receive no Azelficoast semantic identity. A
changed filter, aggregate, ordering rule, function, relation, or executable SQLite
program must earn a reviewed semantic lowering before the packed/JAX path will execute
it.

That boundary is intentional: SQL is the source language, but mechanics, posterior
meaning, evaluator semantics, and evidence admission remain outside the language.
