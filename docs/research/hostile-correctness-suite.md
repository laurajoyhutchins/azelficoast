# Hostile correctness suite

The hostile suite tries to falsify an Azelficoast result by changing representation,
information availability, evidence, or authorized work. Each case carries a machine-
checked `hostile_contract` marker that records its mutation, expected invariant,
scientific threat, and enforcement layer.

The suite treats exact identities as exact comparisons. The centralized tolerances in
`tests/hostile/fixtures.py` apply only to normalized posterior arithmetic, learned
evaluator outputs, search values, and probability conservation. They do not apply to
selected actions, action/value association, semantic hashes, class membership, or
matched contract fields. The evaluator tolerance is `1e-6`; the other numeric
tolerances are `1e-12` or tighter. Tests use absolute tolerance with zero relative
tolerance so scale does not hide a discrepancy.

Posterior semantic identity canonicalizes each normalized mass to 15 significant
decimal digits. This removes binary-float noise from a positive global rescaling while
preserving mass changes at the precision used by the research inputs.

Cross-language semantic identity currently has a frozen fixture over the shared safe
integer JSON domain. Floating-point canonicalization across Python and ECMAScript is
not claimed by this suite yet; callers must use the serialization contract of the
specific artifact rather than assuming all JSON number spellings are interchangeable.


## CI tiers

| Tier | CI location | Main checks |
| --- | --- | --- |
| Fast hostile | Every pull request, in the ordinary test job | Posterior renaming, permutation, split/merge and scaling; action-order invariance; synthetic search canaries and budget receipts; matched-field drift and malformed evidence; population row-order invariance; frozen cross-language identity; artifact round trips. |
| Simulator hostile | Marked `hostile_simulator`; runs where JAX is installed and otherwise skips | Real JAX preprocessing and pooling invariance, transport-field poison, action-to-logit association, and checkpoint semantic identity. |
| Pinned Showdown hostile | `public-belief-exact-corpus` candidate-promotion workflow after the exact-SHA Showdown oracle is generated | Direct per-world transition comparisons, lazy-class certification, single-field marginalized-variant perturbations, relevant-field separation, representative swaps, stale read-trace rejection, and posterior mass conservation. |

The fast gate is:

```sh
pytest tests/hostile -m hostile_fast
```

The simulator gate is:

```sh
pytest tests/hostile -m hostile_simulator
```

The exact gate is run for each supported frozen candidate by
`.github/workflows/public-belief-exact-corpus.yml`. It uses Pokémon Showdown commit
`a5df8274e85b0889bf2a9b3422a08b39732374fc`. The probe writes both the exhaustive
world/action oracle and a separate lazy transition-program artifact from the same
candidate run. Keeping the certification artifact separate preserves the direct
oracle's semantic identity. The Python verifier checks every class member against the exhaustive direct outcomes;
the probe also reruns direct Showdown transitions after changing marginalized move
and Tera fields. The exact checks are not moved into ordinary CI because they require
the pinned simulator and candidate reconstruction.


## Invariants under attack

The fast tests exercise semantic posterior identity and downstream outputs after
world-ID renaming, support permutation, equivalent support splitting/merging, and
positive weight scaling. They include tiny analytically solvable search games so
expected posterior aggregation and information-set coupling do not come from the
production search implementation.

Matched-experiment tests mutate the frozen contract one field at a time, attack
receipts and compute ceilings, and permute complete population rows with fixed
bootstrap seeds. Invalid evidence must fail before statistical aggregation. A schema
field without a drift-matrix owner fails the hostile contract test.

Evaluator tests poison transport and realized-world routes through preprocessing and
run the actual JAX model. Checkpoint identity follows model parameters and relevant
architecture while ignoring path and container metadata.

The exact mechanics tests keep unread-field and relevant-field checks paired. A
field absent from dynamic read evidence is directly perturbed in Showdown and its
complete immediate transition distribution must remain equal. A field whose change
alters direct transitions must be present in read evidence and remain separated by
the certified classes. Lazy programs are checked against every world/action entry,
not only a scalar value or one representative. Probability mass is checked before
and after posterior reweighting.

## Scientific non-properties

These tests establish experimental validity and representation invariance. They do
not assume that deeper search improves a value, more compute improves an action,
determinization has a fixed bias direction, or a learned evaluator beats a heuristic.
Those are empirical questions for the research protocol.
